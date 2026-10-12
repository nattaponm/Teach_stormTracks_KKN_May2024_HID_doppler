"""
polar_tools.py — เครื่องมือข้อมูลเรดาร์แบบ polar (Py-ART) ของเรดาร์ KKN
(คัดลอกจาก repo Teach_HID_Doppler_KKN_summerStorm โดยไม่แก้อัลกอริทึม เพื่อให้ผลเทียบกับงานเดิมได้ตรงตัว)
==========================================================================================

ใช้กับไฟล์ ``data/polar_all/KKN_20240503_HHMM_polar.nc`` (CfRadial ขนาดเล็ก, ผ่าน gentle QC แล้ว)

หมวดฟังก์ชัน
-------------
1. อ่านไฟล์ / เวลา / meta สำหรับแผนที่        : polar_files, read_volume, volume_time, polar_meta
2. อุณหภูมิจาก sounding และเรขาคณิตลำคลื่น    : load_profile, add_temperature, beam_height_km
3. Doppler: dealiasing 3 วิธี                   : dealias_region, dealias_unwrap, dealias_sounding
4. อนุพันธ์ของลม Doppler                        : radial_divergence, azimuthal_shear
5. Dual-pol: ZDR bias, PhiDP, KDP, การลดทอน     : zdr_bias_light_rain, process_phidp, add_kdp, correct_attenuation
6. HID แบบ fuzzy (Besic et al. 2016)            : classify_hid, HID_NAMES, hid_cmap_norm
7. ความปั่นป่วน (PyTDA / NTDA)                 : add_edr
8. กริด / ภาคตัดขวาง / สถิติรอบจุด             : grid_lookup, to_grid, cross_section, site_gates
9. ประมวลผลครบ 1 volume (ใช้ทั้งวันใน NB06)    : process_volume, volume_metrics, swath_update
10. แผนที่ PPI                                    : ppi_map, field_cmaps

ชื่อ field ที่สร้างเพิ่ม (ใช้ซ้ำในทุก notebook)
-----------------------------------------------
temperature (°C), VEL_RB / VEL_UP / VEL_SND (m/s), DIV, AZSHEAR (s⁻¹),
PHIDP (°, ประมวลผลแล้ว), KDP (° km⁻¹), DBZ_C, ZDR_C (แก้ bias + ลดทอนแล้ว), HID, EDR (m^2/3 s⁻¹)
"""

import datetime as _dt
import glob as _glob
import os as _os
import re as _re
import warnings as _warnings

import numpy as np

# ---------------------------------------------------------------------------
# ค่าคงที่ของเคส
# ---------------------------------------------------------------------------
RADAR_FREQ_HZ = 5.6e9           # C-band
WAVELENGTH_M = 2.998e8 / RADAR_FREQ_HZ
BEAMWIDTH_DEG = 1.0
GATE_SPACING_KM = 0.5
F = dict(Z="reflectivity_qc", ZRAW="reflectivity_raw", V="velocity_qc", W="spectrum_width_qc",
         ZDR="differential_reflectivity_qc", PHI="differential_phase_qc",
         RHO="cross_correlation_ratio_qc", QC="qc_flag")

# ค่าที่ใช้ในงานวิจัยต้นฉบับของ KKN (ปรับได้)
ZDR_CORR_BY_SWEEP = {0: 0.20, 1: 0.39, 2: 0.39, 3: 0.39}   # dB (ค่าที่ "บวกเพิ่ม" ให้ ZDR; = −bias)
ZPHI_A, ZPHI_BETA = 0.073, 0.64884                          # Z-PHI สำหรับ C-band
QC_ZH_MIN, QC_RHOHV_MIN = 10.0, 0.85

# HID: ลำดับคลาสตาม Py-ART hydroclass_semisupervised (0 = ไม่จำแนก)
HID_NAMES = ["NC", "AG", "CR", "LR", "RP", "RN", "VI", "WS", "MH", "IH/HDG"]
HID_LONG = {"NC": "Not classified", "AG": "Aggregates", "CR": "Ice crystals",
            "LR": "Light rain", "RP": "Rimed particles (graupel)", "RN": "Rain",
            "VI": "Vertically aligned ice", "WS": "Wet snow", "MH": "Melting hail",
            "IH/HDG": "Ice hail / high-density graupel"}
HID_COLORS = ["#ffffff", "#add8e6", "#00ffff", "#0000ff", "#800080", "#008000",
              "#ffff00", "#ffa500", "#ff0000", "#8b0000"]
HID_GROUPS = {"liquid": [3, 5], "mixed": [4, 7, 8], "solid": [1, 2, 6, 9], "hail": [8, 9],
              "graupel": [4]}


# ---------------------------------------------------------------------------
# 1) อ่านไฟล์และ meta
# ---------------------------------------------------------------------------
def _quiet():
    import contextlib, io
    return contextlib.redirect_stdout(io.StringIO())


def polar_files(data_dir, start=None, end=None, folder="polar_all"):
    """รายชื่อไฟล์ polar เรียงตามเวลา — start/end = 'HH:MM' (UTC)."""
    files = sorted(_glob.glob(_os.path.join(data_dir, folder, "KKN_*_polar.nc")))
    out = []
    for f in files:
        hhmm = _re.search(r"_(\d{4})_polar", f).group(1)
        t = f"{hhmm[:2]}:{hhmm[2:]}"
        if (start is None or t >= start) and (end is None or t <= end):
            out.append(f)
    return out


def file_time(path):
    m = _re.search(r"KKN_(\d{8})_(\d{4})_polar", _os.path.basename(path))
    return _dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M")


def read_volume(path):
    """อ่านไฟล์ polar 1 ไฟล์ → pyart.Radar (เก็บเวลาไว้ใน radar.metadata['time_utc'])."""
    with _quiet():
        import pyart
    radar = pyart.io.read_cfradial(path)
    radar.metadata["time_utc"] = file_time(path)
    return radar


def volume_time(radar):
    return radar.metadata.get("time_utc")


def polar_meta(radar, half_km=240.0, res_km=1.0):
    """meta (แบบเดียวกับ repo pysteps) เพื่อใช้ฟังก์ชันแผนที่ใน mapplot และ kkn_tools."""
    lon = float(radar.longitude["data"][0])
    lat = float(radar.latitude["data"][0])
    alt = float(radar.altitude["data"][0])
    n = int(round(2 * half_km / res_km)) + 1
    x = np.linspace(-half_km, half_km, n) * 1000.0
    y = x[::-1].copy()                                      # แถว 0 = ทิศเหนือ (yorigin upper)
    X, Y = np.meshgrid(x, y)
    proj = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    d = res_km * 1000.0
    return {"projection": proj, "x1": x[0] - d / 2, "x2": x[-1] + d / 2,
            "y1": y[-1] - d / 2, "y2": y[0] + d / 2, "xpixelsize": d, "ypixelsize": d,
            "x": x, "y": y, "coverage": np.hypot(X, Y) <= half_km * 1000.0,
            "radar_lon": lon, "radar_lat": lat, "radar_alt_m": alt, "radar": "KKN",
            "unit": "", "yorigin": "upper"}


def nyquist(radar):
    return float(radar.instrument_parameters["nyquist_velocity"]["data"][0])


def field_data(radar, field, sweep=None, fill=np.nan):
    """ข้อมูล field เป็น numpy float (ค่าที่ mask → NaN) ทั้ง volume หรือเฉพาะ sweep."""
    d = np.ma.filled(radar.fields[field]["data"].astype(float), fill)
    return d if sweep is None else d[radar.get_slice(sweep)]


def add_field(radar, name, data, units="", long_name=None, mask_invalid=True):
    data = np.ma.masked_invalid(data) if mask_invalid else data
    radar.add_field(name, {"data": data, "units": units, "long_name": long_name or name,
                           "_FillValue": -9999.0}, replace_existing=True)
    return radar


# ---------------------------------------------------------------------------
# 2) sounding, อุณหภูมิ, เรขาคณิตลำคลื่น
# ---------------------------------------------------------------------------
def load_profile(path):
    """อ่าน sounding (CSV) → DataFrame ที่เรียงตามความสูง (m MSL)."""
    import pandas as pd
    df = pd.read_csv(path)
    return df.dropna(subset=["height", "temperature"]).sort_values("height").reset_index(drop=True)


def isotherm_height(df, temp_c):
    """ความสูง (m MSL) ที่อุณหภูมิเท่ากับ temp_c (ครั้งแรกจากพื้น)."""
    h, t = df["height"].values, df["temperature"].values
    for k in range(len(t) - 1):
        if (t[k] - temp_c) * (t[k + 1] - temp_c) <= 0 and t[k] != t[k + 1]:
            return float(h[k] + (temp_c - t[k]) * (h[k + 1] - h[k]) / (t[k + 1] - t[k]))
    return np.nan


def add_temperature(radar, df, name="temperature"):
    """อุณหภูมิ (°C) ที่ทุก gate จาก sounding ตามความสูง gate (m MSL)."""
    alt = radar.gate_altitude["data"]
    t = np.interp(alt, df["height"].values, df["temperature"].values)
    add_field(radar, name, t, "degC", "Temperature from sounding", mask_invalid=False)
    return radar


def beam_height_km(range_km, elev_deg, radar_alt_m=187.0, ke=4 / 3):
    """ความสูงกึ่งกลางลำคลื่น (km MSL) ตามแบบจำลองโลก 4/3 (Doviak & Zrnić 1993)."""
    a = 6371.0 * ke
    r = np.asarray(range_km, float)
    th = np.radians(elev_deg)
    return np.sqrt(r ** 2 + a ** 2 + 2 * r * a * np.sin(th)) - a + radar_alt_m / 1000.0


def beam_width_km(range_km, beamwidth_deg=BEAMWIDTH_DEG):
    return np.asarray(range_km, float) * np.radians(beamwidth_deg)


# ---------------------------------------------------------------------------
# 3) Doppler dealiasing
# ---------------------------------------------------------------------------
def _vel_filter(radar, vel=F["V"]):
    import pyart
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_invalid(vel)
    gf.exclude_masked(vel)
    return gf


REGION_TUNED = dict(centered=False, skip_between_rays=0, skip_along_ray=0)


def dealias_region(radar, vel=F["V"], name="VEL_RB", tuned=True, **kw):
    """วิธีที่ 1: region-based (Py-ART) — แบ่งพื้นที่ที่ความเร็วต่อเนื่อง แล้วเลื่อนทีละ 2Vny

    tuned=True  : ไม่เชื่อมพื้นที่ข้ามช่องว่าง (skip_* = 0) และไม่จัดค่าเฉลี่ยให้ใกล้ 0 (centered=False)
                  → เหมาะกับเซลล์พายุที่กระจัดกระจาย (เคสนี้: ตรงกับ unwrap/sounding > 93% ของ gate)
    tuned=False : ค่าตั้งต้นของ Py-ART (skip_* = 100, centered=True) — ใช้เปรียบเทียบใน NB2
    """
    import pyart
    opts = dict(REGION_TUNED) if tuned else {}
    opts.update(kw)
    out = pyart.correct.dealias_region_based(radar, vel_field=vel, gatefilter=_vel_filter(radar, vel),
                                             **opts)
    radar.add_field(name, out, replace_existing=True)
    return radar


def dealias_unwrap(radar, vel=F["V"], name="VEL_UP", unwrap_unit="sweep"):
    """วิธีที่ 2: unwrap phase (Py-ART) — มองความเร็วเป็นเฟส แล้วคลายเฟสแบบ 2 มิติ/3 มิติ."""
    import pyart
    out = pyart.correct.dealias_unwrap_phase(radar, vel_field=vel, unwrap_unit=unwrap_unit,
                                             gatefilter=_vel_filter(radar, vel))
    radar.add_field(name, out, replace_existing=True)
    return radar


def sounding_wind_profile(df, radar_alt_m=0.0):
    """ลมจาก sounding → (height m เหนือเรดาร์, speed m/s, direction °) — ตัดค่าที่ไม่ครบ."""
    d = df.dropna(subset=["height", "speed", "direction"])
    return d["height"].values - radar_alt_m, d["speed"].values, d["direction"].values


def simulated_velocity(radar, df, name="VEL_SIM"):
    """ความเร็วตามแนวรัศมีที่คาดจากลม sounding:  V_ref = U sinφ cosθ + V cosφ cosθ."""
    h, spd, wdir = sounding_wind_profile(df)
    u = -spd * np.sin(np.radians(wdir))
    v = -spd * np.cos(np.radians(wdir))
    alt = radar.gate_altitude["data"]
    U = np.interp(alt, h, u)
    V = np.interp(alt, h, v)
    az = np.radians(radar.azimuth["data"])[:, None]
    el = np.radians(radar.elevation["data"])[:, None]
    vr = (U * np.sin(az) + V * np.cos(az)) * np.cos(el)
    add_field(radar, name, vr, "m/s", "Radial velocity simulated from sounding", mask_invalid=False)
    return radar


def _nan_smooth(a, size):
    """ค่าเฉลี่ยเคลื่อนที่ที่ข้าม NaN (ใช้ทำ reference field เฉพาะที่)."""
    from scipy.ndimage import uniform_filter
    m = np.isfinite(a)
    num = uniform_filter(np.where(m, a, 0.0), size=size, mode=("wrap", "nearest"))
    den = uniform_filter(m.astype(float), size=size, mode=("wrap", "nearest"))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.2, num / den, np.nan)


def unfold_to_reference(v_obs, v_ref, vny):
    """V = V_obs + 2nVny  โดยเลือก n ที่ทำให้ |V − V_ref| น้อยที่สุด."""
    n = np.round((v_ref - v_obs) / (2.0 * vny))
    return v_obs + 2.0 * vny * n


def dealias_sounding(radar, df, vel=F["V"], name="VEL_SND", n_pass=2, window=(5, 21)):
    """วิธีที่ 3: ใช้ลมจาก sounding เป็น first guess (แนวคิดขั้นแรกของ 4DD; James & Houze 2001)

    pass 0 : คลายแต่ละ gate ให้ใกล้ V_ref จาก sounding
    pass 1+: คลายซ้ำให้ใกล้ค่าเฉลี่ยเฉพาะที่ (window rays × gates) ของผล pass ก่อน
             → แก้จุดที่ลมพายุต่างจาก sounding มาก (เช่น outflow)
    หมายเหตุ: Py-ART 2.x ยกเลิก dealias_fourdd แล้ว จึงเขียนหลักการเองให้เห็นทีละขั้น
    """
    vny = nyquist(radar)
    simulated_velocity(radar, df)
    vobs = field_data(radar, vel)
    vref = field_data(radar, "VEL_SIM")
    out = np.full_like(vobs, np.nan)
    for s in range(radar.nsweeps):
        sl = radar.get_slice(s)
        vs = unfold_to_reference(vobs[sl], vref[sl], vny)
        for _ in range(n_pass):
            vs = unfold_to_reference(vobs[sl], _nan_smooth(vs, window), vny)
            vs = np.where(np.isfinite(vs), vs, np.nan)
        out[sl] = vs
    add_field(radar, name, out, "m/s", "Dealiased velocity (sounding first guess)")
    return radar


def dealias_report(radar, fields=("VEL_RB", "VEL_UP", "VEL_SND"), vel=F["V"]):
    """ตารางสรุป: จำนวน gate ที่ถูกเลื่อน (|ΔV| > Vny) และช่วงความเร็วของแต่ละวิธี."""
    import pandas as pd
    v0 = field_data(radar, vel)
    vny = nyquist(radar)
    rows = []
    for f in fields:
        if f not in radar.fields:
            continue
        v = field_data(radar, f)
        ok = np.isfinite(v) & np.isfinite(v0)
        rows.append({"method": f, "valid_gates": int(ok.sum()),
                     "unfolded_gates": int((np.abs(v - v0)[ok] > vny).sum()),
                     "unfolded_pct": 100 * float((np.abs(v - v0)[ok] > vny).mean()) if ok.any() else np.nan,
                     "v_min": float(np.nanmin(v)), "v_max": float(np.nanmax(v)),
                     "v_p01": float(np.nanpercentile(v, 1)), "v_p99": float(np.nanpercentile(v, 99))})
    return pd.DataFrame(rows).set_index("method").round(2)


# ---------------------------------------------------------------------------
# 4) อนุพันธ์ของลม Doppler
# ---------------------------------------------------------------------------
def radial_divergence(radar, vel="VEL_RB", name="DIV", smooth=(3, 5), dgates=2):
    """∂Vr/∂r (s⁻¹) — บวก = ลมแยกออกตามแนวรัศมี (divergence, เช่น outflow ของ downburst)."""
    dr = float(np.median(np.diff(radar.range["data"])))
    v = field_data(radar, vel)
    out = np.full_like(v, np.nan)
    for s in range(radar.nsweeps):
        sl = radar.get_slice(s)
        vs = _nan_smooth(v[sl], smooth)
        vs = np.where(np.isfinite(v[sl]), vs, np.nan)
        g = np.full_like(vs, np.nan)
        g[:, dgates:-dgates] = (vs[:, 2 * dgates:] - vs[:, :-2 * dgates]) / (2 * dgates * dr)
        out[sl] = g
    add_field(radar, name, out, "s-1", "Radial divergence dVr/dr")
    return radar


def azimuthal_shear(radar, vel="VEL_RB", name="AZSHEAR", smooth=(3, 5), drays=2):
    """(1/r) ∂Vr/∂φ (s⁻¹) — ค่าบวก = หมุนทวนเข็ม (cyclonic ในซีกโลกเหนือ)."""
    r = radar.range["data"][None, :]
    v = field_data(radar, vel)
    out = np.full_like(v, np.nan)
    for s in range(radar.nsweeps):
        sl = radar.get_slice(s)
        az = radar.azimuth["data"][sl]
        vs = _nan_smooth(v[sl], smooth)
        vs = np.where(np.isfinite(v[sl]), vs, np.nan)
        dphi = np.radians((np.roll(az, -drays) - np.roll(az, drays)) % 360.0)[:, None]
        dv = np.roll(vs, -drays, axis=0) - np.roll(vs, drays, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[sl] = dv / (r * dphi)
    add_field(radar, name, out, "s-1", "Azimuthal shear (1/r) dVr/dphi")
    return radar


# ---------------------------------------------------------------------------
# 5) Dual-pol
# ---------------------------------------------------------------------------
def zdr_bias_light_rain(radar, df=None, zmin=20.0, zmax=28.0, rho_min=0.97, zdr_expected=0.2,
                        below_0c_m=1000.0, by_sweep=True):
    """ประมาณ ZDR bias จากฝนเบา (Z 20–28 dBZ, ρHV > 0.97, ต่ำกว่าระดับ 0 °C)

    bias = median(ZDR_obs) − ZDR ที่ควรเป็นในฝนเบา (~0.2 dB ที่ C-band)
    คืนค่า dict {sweep: (bias, n_gates)}
    """
    z, zdr, rho = (field_data(radar, F[k]) for k in ("Z", "ZDR", "RHO"))
    h0 = isotherm_height(df, 0.0) if df is not None else float(radar.metadata.get("height_0C_m_msl", 5000))
    ok = (z >= zmin) & (z <= zmax) & (rho > rho_min) & np.isfinite(zdr) & \
         (radar.gate_altitude["data"] < h0 - below_0c_m)
    out = {}
    sweeps = range(radar.nsweeps) if by_sweep else [None]
    for s in sweeps:
        m = ok if s is None else ok & _sweep_mask(radar, s)
        n = int(m.sum())
        out["all" if s is None else s] = (float(np.median(zdr[m]) - zdr_expected) if n > 30 else np.nan, n)
    return out


def _sweep_mask(radar, s):
    m = np.zeros((radar.nrays, radar.ngates), bool)
    m[radar.get_slice(s)] = True
    return m


def phidp_slope_by_sweep(radar, z_min=30.0, rho_min=0.95):
    """ความชันเฉลี่ยของ PhiDP ตามระยะ (°/km) ในแต่ละ sweep — ค่าลบ = ต้องกลับเครื่องหมาย."""
    phi = field_data(radar, F["PHI"])
    z, rho = field_data(radar, F["Z"]), field_data(radar, F["RHO"])
    r_km = radar.range["data"] / 1000.0
    out = {}
    for s in range(radar.nsweeps):
        sl = radar.get_slice(s)
        slopes = []
        for p, zz, rr in zip(phi[sl], z[sl], rho[sl]):
            m = np.isfinite(p) & (zz > z_min) & (rr > rho_min)
            if m.sum() < 20:
                continue
            u = np.degrees(np.unwrap(np.radians(p[m])))
            if np.ptp(r_km[m]) < 10:
                continue
            slopes.append(np.polyfit(r_km[m], u, 1)[0])
        out[s] = (float(np.median(slopes)) if slopes else np.nan, len(slopes))
    return out


def phidp_texture(phi, n=5):
    """ส่วนเบี่ยงเบนมาตรฐานเฉพาะที่ของ PhiDP ตามระยะ (° ) — ใช้แยก echo ฝนออกจาก noise."""
    from scipy.ndimage import uniform_filter1d
    m = np.isfinite(phi)
    p = np.where(m, phi, 0.0)
    mean = uniform_filter1d(p, n, axis=1)
    mean2 = uniform_filter1d(p * p, n, axis=1)
    cnt = uniform_filter1d(m.astype(float), n, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu, mu2 = mean / cnt, mean2 / cnt
        return np.where(cnt > 0.6, np.sqrt(np.maximum(mu2 - mu * mu, 0)), np.nan)


def process_phidp(radar, sign=None, rho_min=QC_RHOHV_MIN, z_min=QC_ZH_MIN, tex_max=20.0,
                  sys_gates=10, smooth_gates=7, median_gates=11, name="PHIDP", return_steps=False):
    """ประมวลผล PhiDP ทีละขั้น (ตามแนวรัศมีของแต่ละ ray)

    1) กลับเครื่องหมาย (KKN: PhiDP ลดลงตามระยะ → ×(−1)) — sign=None ตรวจอัตโนมัติรายมุมยก
    2) คัดเฉพาะ gate ที่เป็นฝน: ρHV > 0.85, Z > 10 dBZ, texture < 20°
    3) คลายรอยพับเฟส (unwrap ±180°) ตามระยะ
    4) ลบ system phase (median ของ 10 gate แรกที่ใช้ได้)
    5) ตัด spike (median 11 gate) แล้ว fit แบบ isotonic regression (ไม่ลดลงตามระยะ แต่ไม่ "ค้าง" ที่ยอด spike
       เหมือน cumulative max) จากนั้นทำให้เรียบ (7 gate)
    หมายเหตุ: ใช้ได้กับ gate ที่มี echo เท่านั้น — ช่วงไม่มี echo ระหว่างเซลล์ถูกเติมเพื่อการคำนวณแต่ mask ออกตอนท้าย
    """
    phi = field_data(radar, F["PHI"])
    z, rho = field_data(radar, F["Z"]), field_data(radar, F["RHO"])
    if sign is None:
        sl_ = phidp_slope_by_sweep(radar)
        sign = {s: (-1.0 if v[0] < 0 else 1.0) for s, v in sl_.items()}
    elif np.isscalar(sign):
        sign = {s: float(sign) for s in range(radar.nsweeps)}
    tex = phidp_texture(phi)
    steps = {"raw": phi.copy()}
    signed = phi.copy()
    for s in range(radar.nsweeps):
        signed[radar.get_slice(s)] *= sign[s]
    good = np.isfinite(signed) & (rho > rho_min) & (z > z_min) & (tex < tex_max)
    unw = np.full_like(phi, np.nan)
    sysp = np.full(radar.nrays, np.nan)
    for i in range(radar.nrays):
        m = good[i]
        if m.sum() < sys_gates:
            continue
        u = np.degrees(np.unwrap(np.radians(signed[i, m])))
        sysp[i] = np.median(u[:sys_gates])
        unw[i, m] = u
    # system phase: ใช้ median ราย sweep (เสถียรกว่ารายเรย์)
    sys_sweep = {}
    for s in range(radar.nsweeps):
        sl = radar.get_slice(s)
        sys_sweep[s] = float(np.nanmedian(sysp[sl])) if np.isfinite(sysp[sl]).any() else 0.0
        unw[sl] -= sys_sweep[s]
    # ลบรอบ 360° ที่เหลือ: ให้ค่าแรกของแต่ละ ray อยู่ใกล้ 0
    first = np.array([row[np.isfinite(row)][0] if np.isfinite(row).any() else 0.0 for row in unw])
    unw -= 360.0 * np.round(first / 360.0)[:, None]
    steps["signed"], steps["unwrapped"] = signed, unw.copy()
    # median → isotonic regression (ไม่ลดลงตามระยะ) → เติมช่องว่าง → ทำให้เรียบ
    proc = np.full_like(phi, np.nan)
    from scipy.ndimage import uniform_filter1d, median_filter
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    for i in range(radar.nrays):
        row = unw[i]
        m = np.isfinite(row)
        if m.sum() < sys_gates:
            continue
        idx = np.where(m)[0]
        vals = median_filter(row[m], size=min(median_gates, m.sum()), mode="nearest")  # ตัด spike
        fit = iso.fit_transform(idx.astype(float), vals)                              # monotonic fit
        filled = np.interp(np.arange(radar.ngates), idx, fit)
        filled[: idx[0]] = max(fit[0], 0.0)
        filled = np.maximum(uniform_filter1d(filled, smooth_gates), 0.0)
        filled[idx[-1] + 1:] = np.nan                       # ไม่ต่อเกินข้อมูลจริง
        proc[i] = filled
    steps["processed_filled"] = proc.copy()            # ใช้ต่อเนื่องสำหรับ Z-PHI
    proc = np.where(np.isfinite(z), proc, np.nan)        # แสดง/ใช้เฉพาะ gate ที่มี echo
    steps["processed"] = proc
    add_field(radar, name, proc, "deg", "Processed differential phase (unwrapped, sys-phase removed)")
    radar.metadata["phidp_sign"] = str(sign)
    radar.metadata["phidp_system_phase"] = str({k: round(v, 1) for k, v in sys_sweep.items()})
    return (radar, steps, sign, sys_sweep) if return_steps else radar


def add_kdp(radar, phidp="PHIDP", name="KDP", windsize=10, method="vulpiani"):
    """KDP = ½ dΦDP/dr (° km⁻¹) — vulpiani (Py-ART) หรือ 'slope' (ความชันเฉพาะที่แบบง่าย)."""
    if method == "vulpiani":
        import pyart
        kdp, _ = pyart.retrieve.kdp_vulpiani(radar, psidp_field=phidp, windsize=windsize, band="C")
        k = np.ma.filled(kdp["data"].astype(float), np.nan)
    else:
        phi = field_data(radar, phidp)
        dr_km = float(np.median(np.diff(radar.range["data"]))) / 1000.0
        h = windsize // 2
        k = np.full_like(phi, np.nan)
        k[:, h:-h] = (phi[:, 2 * h:] - phi[:, :-2 * h]) / (2 * h * dr_km) / 2.0
    k = np.where(np.isfinite(field_data(radar, phidp)) & np.isfinite(field_data(radar, F["Z"])), k, np.nan)
    k = np.clip(k, -2.0, 12.0)
    add_field(radar, name, k, "deg/km", "Specific differential phase")
    return radar


def correct_zdr_bias(radar, correction=None, name="ZDR_C0"):
    """ZDR_corr = ZDR + correction (ราย sweep; correction = −bias)

    correction=None ใช้ค่าในงานวิจัย KKN (ZDR_CORR_BY_SWEEP: +0.20, +0.39, +0.39, +0.39 dB)
    หรือใส่ dict ที่ได้จาก zdr_bias_light_rain() แล้วกลับเครื่องหมาย
    """
    correction = ZDR_CORR_BY_SWEEP if correction is None else correction
    zdr = field_data(radar, F["ZDR"])
    for s in range(radar.nsweeps):
        c = correction.get(s, 0.0) if isinstance(correction, dict) else float(correction)
        zdr[radar.get_slice(s)] += c
    add_field(radar, name, zdr, "dB", "ZDR (bias corrected)")
    return radar


def correct_attenuation(radar, phidp="PHIDP", refl=F["Z"], zdr="ZDR_C0", temp="temperature",
                        a_coef=ZPHI_A, beta=ZPHI_BETA):
    """แก้การลดทอนแบบ Z-PHI (Testud et al. 2000) → DBZ_C, ZDR_C, PIA, SPEC_AT."""
    import pyart
    with _quiet(), _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        spec_at, pia, cor_z, spec_diff_at, pida, cor_zdr = pyart.correct.calculate_attenuation_zphi(
            radar, a_coef=a_coef, beta=beta, refl_field=refl, phidp_field=phidp, zdr_field=zdr,
            temp_field=temp, temp_ref="temperature")
    for nm, d in (("SPEC_AT", spec_at), ("PIA", pia), ("DBZ_C", cor_z), ("PIDA", pida),
                  ("ZDR_C", cor_zdr)):
        radar.add_field(nm, d, replace_existing=True)
    # ไม่แก้ในบริเวณที่ไม่มี PhiDP → ใช้ค่าเดิม
    for nm, base in (("DBZ_C", refl), ("ZDR_C", zdr)):
        a = field_data(radar, nm)
        b = field_data(radar, base)
        add_field(radar, nm, np.where(np.isfinite(a), a, b), radar.fields[base].get("units", ""))
    return radar


# ---------------------------------------------------------------------------
# 6) HID
# ---------------------------------------------------------------------------
def classify_hid(radar, refl="DBZ_C", zdr="ZDR_C", kdp="KDP", rho=F["RHO"], temp="temperature",
                 name="HID", weights=(1.0, 1.0, 1.0, 0.75, 0.5), entropy=False):
    """จำแนกชนิดหยาดน้ำฟ้าแบบ fuzzy/semi-supervised (Besic et al. 2016) — คืน 0–9 (0 = NC)."""
    import pyart
    for f in (refl, zdr, kdp):
        a = field_data(radar, f)
        add_field(radar, f, a)
    with _quiet(), _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        out = pyart.retrieve.hydroclass_semisupervised(
            radar, refl_field=refl, zdr_field=zdr, kdp_field=kdp, rhv_field=rho, temp_field=temp,
            radar_freq=RADAR_FREQ_HZ, weights=np.asarray(weights, float), compute_entropy=entropy)
    hid = out["hydro"] if isinstance(out, dict) and "hydro" in out else out
    h = np.ma.filled(hid["data"], 0).astype(float)
    z = field_data(radar, refl)
    h[~np.isfinite(z) | (z < QC_ZH_MIN)] = 0
    add_field(radar, name, np.ma.masked_equal(h, 0), "", "Hydrometeor class (0=NC)", mask_invalid=False)
    if entropy and isinstance(out, dict) and "entropy" in out:
        radar.add_field("HID_ENTROPY", out["entropy"], replace_existing=True)
    return radar


def hid_cmap_norm():
    import matplotlib as mpl
    cmap = mpl.colors.ListedColormap(HID_COLORS[1:])
    norm = mpl.colors.BoundaryNorm(np.arange(0.5, 10.5, 1), cmap.N)
    return cmap, norm


def hid_counts(radar, field="HID", sweeps=None, mask=None):
    """จำนวน gate ต่อคลาส (dict ชื่อคลาส → จำนวน)."""
    h = np.ma.filled(radar.fields[field]["data"], 0).astype(int)
    m = np.zeros_like(h, bool)
    for s in (range(radar.nsweeps) if sweeps is None else sweeps):
        m[radar.get_slice(s)] = True
    if mask is not None:
        m &= mask
    return {HID_NAMES[c]: int(((h == c) & m).sum()) for c in range(1, 10)}


# ---------------------------------------------------------------------------
# 7) PyTDA — EDR จาก spectrum width
# ---------------------------------------------------------------------------
def add_edr(radar, sw=F["W"], refl=F["Z"], radius=2.0, half_km=200.0, name="EDR"):
    """EDR^{1/3} (m^{2/3} s⁻¹) ด้วย NTDA ใน PyTDA (Williams & Meymaris 2016)

    split_cut=False เพราะ KKN วัด Z และ Doppler ในรอบเดียวกันทุกมุมยก (ไม่ใช่ split cut แบบ WSR-88D)
    """
    import pytda
    kw = dict(radius=radius, split_cut=False, xran=[-half_km, half_km], yran=[-half_km, half_km],
              name_dz=refl, name_sw=sw, turb_name=name, beamwidth=BEAMWIDTH_DEG,
              gate_spacing=GATE_SPACING_KM, verbose=False)
    with _quiet(), _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        try:
            pytda.calc_turb_vol(radar, **kw)
        except ValueError:
            # บาง sweep ไม่มี gate ที่ใช้ได้ (เช่น พายุอยู่นอก ±half_km) → คำนวณทีละ sweep
            out = np.full((radar.nrays, radar.ngates), np.nan)
            for s in range(radar.nsweeps):
                sub = radar.extract_sweeps([s])
                try:
                    pytda.calc_turb_vol(sub, **kw)
                    out[radar.get_slice(s)] = field_data(sub, name)
                except ValueError:
                    pass
            add_field(radar, name, out, "m^(2/3) s^-1")
    e = field_data(radar, name)
    e[(e <= 0) | ~np.isfinite(field_data(radar, sw))] = np.nan
    add_field(radar, name, e, "m^(2/3) s^-1", "Eddy dissipation rate EDR^(1/3)")
    return radar


# ---------------------------------------------------------------------------
# 8) กริด / ภาคตัดขวาง / สถิติรอบจุด
# ---------------------------------------------------------------------------
def grid_lookup(radar, sweep, meta):
    """index (ray, gate) ที่ใกล้ที่สุดสำหรับทุก pixel ของกริด meta (nearest polar)."""
    X, Y = np.meshgrid(meta["x"], meta["y"])
    el = np.radians(float(radar.fixed_angle["data"][sweep]))
    sl = radar.get_slice(sweep)
    az = radar.azimuth["data"][sl]
    order = np.argsort(az)
    az_s = az[order]
    pa = np.degrees(np.arctan2(X, Y)) % 360.0
    k = np.searchsorted(az_s, pa) % len(az_s)
    km1 = (k - 1) % len(az_s)
    d1 = np.abs((az_s[k] - pa + 180) % 360 - 180)
    d0 = np.abs((az_s[km1] - pa + 180) % 360 - 180)
    ray = np.where(d1 < d0, k, km1)
    ray = order[ray] + sl.start
    rng = radar.range["data"]
    r = np.hypot(X, Y) / np.cos(el)
    gate = np.round((r - rng[0]) / (rng[1] - rng[0])).astype(int)
    ok = (gate >= 0) & (gate < radar.ngates) & meta["coverage"]
    return ray, np.clip(gate, 0, radar.ngates - 1), ok


def to_grid(radar, field, lookup):
    ray, gate, ok = lookup
    a = field_data(radar, field) if isinstance(field, str) else field
    g = a[ray, gate]
    g[~ok] = np.nan
    return g


def column_grid(radar, field, meta, how="max", lookups=None):
    """รวมทุก sweep ลงกริด 2 มิติ (max ของคอลัมน์) — คืน (grid, lookups)."""
    lookups = lookups or [grid_lookup(radar, s, meta) for s in range(radar.nsweeps)]
    stack = np.stack([to_grid(radar, field, lk) for lk in lookups])
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        g = np.nanmax(stack, 0) if how == "max" else np.nanmin(stack, 0)
    return g, lookups


def cross_section(radar, azimuth, field_list=None):
    """pseudo-RHI ตามแนว azimuth (Py-ART cross_section_ppi) — 4 มุมยก = 4 แถว."""
    import pyart
    xs = pyart.util.cross_section_ppi(radar, [float(azimuth)])
    return xs


def azimuth_to(meta, lon, lat):
    """มุม azimuth (°) และระยะ (km) จากเรดาร์ไปยังจุด lon/lat."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", meta["projection"], always_xy=True)
    x, y = tr.transform(lon, lat)
    return float(np.degrees(np.arctan2(x, y)) % 360.0), float(np.hypot(x, y) / 1000.0)


def site_gates(radar, meta, lon, lat, radius_km=15.0, sweeps=None):
    """mask ของ gate ที่อยู่ในรัศมี radius_km รอบจุด (ระยะบนพื้นราบ)."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", meta["projection"], always_xy=True)
    x0, y0 = tr.transform(lon, lat)
    d = np.hypot(radar.gate_x["data"] - x0, radar.gate_y["data"] - y0) / 1000.0
    m = d <= radius_km
    if sweeps is not None:
        sm = np.zeros_like(m)
        for s in sweeps:
            sm[radar.get_slice(s)] = True
        m &= sm
    return m


# ---------------------------------------------------------------------------
# 9) ประมวลผลครบ 1 volume (whole-day)
# ---------------------------------------------------------------------------
def process_volume(radar, df, edr=True, hid=True, dealias=("RB",), verbose=False):
    """ขั้นตอนมาตรฐานของ repo นี้ — ใส่ field ใหม่ลงใน radar แล้วคืน radar."""
    import time
    t0 = time.time()
    add_temperature(radar, df)
    if "RB" in dealias:
        dealias_region(radar)
    if "UP" in dealias:
        dealias_unwrap(radar)
    if "SND" in dealias:
        dealias_sounding(radar, df)
    vel = "VEL_RB" if "RB" in dealias else ("VEL_UP" if "UP" in dealias else "VEL_SND")
    radial_divergence(radar, vel)
    azimuthal_shear(radar, vel)
    process_phidp(radar)
    add_kdp(radar)
    correct_zdr_bias(radar)
    correct_attenuation(radar)
    if hid:
        classify_hid(radar)
    if edr:
        add_edr(radar)
    if verbose:
        print(f"  ประมวลผล {volume_time(radar):%H:%M} UTC เสร็จใน {time.time() - t0:.1f} s")
    return radar


HAIL_MIN_DBZ = 40.0
SWATH_FIELDS = ("Z_max", "hail_count", "graupel_count", "vel_abs_max", "div_max", "azshear_max",
                "edr_max", "kdp_max", "zdr_max")


def volume_metrics(radar, meta, sites=None, radius_km=15.0, rng_max_km=200.0):
    """ตัวชี้วัดของ 1 volume (dict) + กริดของ volume นี้ (dict) สำหรับ swath."""
    t = volume_time(radar)
    lk = [grid_lookup(radar, s, meta) for s in range(radar.nsweeps)]
    g = {}
    g["Z"] = to_grid(radar, "DBZ_C" if "DBZ_C" in radar.fields else F["Z"], lk[0])
    hidg = np.stack([np.nan_to_num(to_grid(radar, "HID", k)) for k in lk]) if "HID" in radar.fields else None
    if hidg is not None:
        zf = "DBZ_C" if "DBZ_C" in radar.fields else F["Z"]
        zg = np.stack([np.nan_to_num(to_grid(radar, zf, k), nan=-99) for k in lk])
        # ลูกเห็บแบบอนุรักษ์: คลาส MH หรือ IH/HDG และ Z ≥ HAIL_MIN_DBZ ที่ gate เดียวกัน
        g["hail"] = (np.isin(hidg, HID_GROUPS["hail"]) & (zg >= HAIL_MIN_DBZ)).any(0)
        g["graupel"] = np.isin(hidg, HID_GROUPS["graupel"]).any(0)
    if "VEL_RB" in radar.fields:
        g["vel_abs"] = np.abs(to_grid(radar, "VEL_RB", lk[0]))
        g["div"] = to_grid(radar, "DIV", lk[0])
        g["azshear"] = np.abs(to_grid(radar, "AZSHEAR", lk[0]))
    if "EDR" in radar.fields:
        g["edr"], _ = column_grid(radar, "EDR", meta, lookups=lk)
    g["kdp"], _ = column_grid(radar, "KDP", meta, lookups=lk)
    g["zdr"] = to_grid(radar, "ZDR_C", lk[0])

    X, Y = np.meshgrid(meta["x"], meta["y"])
    inr = np.hypot(X, Y) <= rng_max_km * 1000
    z = np.where(inr, g["Z"], np.nan)
    echo = z >= 20
    m = {"time_utc": t, "nyquist": nyquist(radar),
         "Z_max": float(np.nanmax(z)) if np.isfinite(z).any() else np.nan,
         "area_Z40_km2": float(np.nansum(z >= 40)), "area_Z50_km2": float(np.nansum(z >= 50))}
    if hidg is not None:
        m["area_hail_km2"] = float((g["hail"] & inr).sum())
        m["area_graupel_km2"] = float((g["graupel"] & inr).sum())
        cnt = hid_counts(radar)
        tot = max(sum(cnt.values()), 1)
        for k, v in cnt.items():
            m[f"frac_{k}"] = v / tot
        m["n_classified"] = tot
    if "VEL_RB" in radar.fields:
        va = np.where(inr & echo, g["vel_abs"], np.nan)
        dv = np.where(inr & echo, g["div"], np.nan)
        az = np.where(inr & echo, g["azshear"], np.nan)
        for k, a in (("vel_abs", va), ("div", dv), ("azshear", az)):
            m[f"{k}_p99"] = float(np.nanpercentile(a, 99)) if np.isfinite(a).sum() > 50 else np.nan
    if "EDR" in radar.fields:
        e = field_data(radar, "EDR")
        e = e[np.isfinite(e)]
        m["edr_p50"] = float(np.percentile(e, 50)) if e.size > 50 else np.nan
        m["edr_p95"] = float(np.percentile(e, 95)) if e.size > 50 else np.nan
        m["edr_max"] = float(e.max()) if e.size else np.nan
    m["area_kdp1_km2"] = float(np.nansum(np.where(inr, g["kdp"], np.nan) >= 1.0))
    if sites:
        R = None
        for name, (lon, lat) in sites.items():
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:4326", meta["projection"], always_xy=True)
            x0, y0 = tr.transform(lon, lat)
            R = np.hypot(X - x0, Y - y0) <= radius_km * 1000
            key = name.replace(" ", "_")
            zz = np.where(R, g["Z"], np.nan)
            m[f"{key}_Zmax"] = float(np.nanmax(zz)) if np.isfinite(zz).any() else np.nan
            if hidg is not None:
                m[f"{key}_hail_km2"] = float((g["hail"] & R).sum())
            if "VEL_RB" in radar.fields:
                m[f"{key}_velmax"] = float(np.nanmax(np.where(R, g["vel_abs"], np.nan))) if np.isfinite(np.where(R, g["vel_abs"], np.nan)).any() else np.nan
                m[f"{key}_divmax"] = float(np.nanmax(np.where(R, g["div"], np.nan))) if np.isfinite(np.where(R, g["div"], np.nan)).any() else np.nan
            if "EDR" in radar.fields:
                ee = np.where(R, g["edr"], np.nan)
                m[f"{key}_edrmax"] = float(np.nanmax(ee)) if np.isfinite(ee).any() else np.nan
    return m, g


def swath_init(meta):
    shp = (len(meta["y"]), len(meta["x"]))
    S = {k: np.full(shp, np.nan, "float32") for k in SWATH_FIELDS}
    S["hail_count"][:] = 0
    S["graupel_count"][:] = 0
    S["Z_max_time"] = np.full(shp, np.nan, "float32")       # ชั่วโมง UTC ที่ Z สูงสุด
    return S


def swath_update(S, g, t):
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        newer = np.nan_to_num(g["Z"], nan=-99) > np.nan_to_num(S["Z_max"], nan=-99)
        S["Z_max_time"][newer] = t.hour + t.minute / 60.0
        S["Z_max"] = np.fmax(S["Z_max"], g["Z"])
        if "hail" in g:
            S["hail_count"] += g["hail"]
            S["graupel_count"] += g["graupel"]
        if "vel_abs" in g:
            S["vel_abs_max"] = np.fmax(S["vel_abs_max"], g["vel_abs"])
            S["div_max"] = np.fmax(S["div_max"], g["div"])
            S["azshear_max"] = np.fmax(S["azshear_max"], g["azshear"])
        if "edr" in g:
            S["edr_max"] = np.fmax(S["edr_max"], g["edr"])
        S["kdp_max"] = np.fmax(S["kdp_max"], g["kdp"])
        S["zdr_max"] = np.fmax(S["zdr_max"], np.where(g["Z"] >= 20, g["zdr"], np.nan))
    return S


def save_swaths(path, S, meta, attrs=None):
    import xarray as xr
    ds = xr.Dataset({k: (("y", "x"), np.asarray(v, "float32")) for k, v in S.items()},
                    coords={"x": meta["x"], "y": meta["y"]})
    ds.attrs.update({"projection": meta["projection"], "radar_lon": meta["radar_lon"],
                     "radar_lat": meta["radar_lat"], **(attrs or {})})
    enc = {k: {"zlib": True, "complevel": 5, "dtype": "int16", "scale_factor": 0.01 if k not in
               ("hail_count", "graupel_count") else 1.0, "_FillValue": -32768} for k in S}
    enc["Z_max"]["scale_factor"] = 0.1
    ds.to_netcdf(path, encoding=enc)
    return path


def load_swaths(path):
    import xarray as xr
    ds = xr.open_dataset(path)
    meta = polar_meta_from_attrs(ds)
    return {k: ds[k].values for k in ds.data_vars}, meta


def polar_meta_from_attrs(ds):
    class _R:
        pass
    r = _R()
    r.longitude = {"data": [ds.attrs["radar_lon"]]}
    r.latitude = {"data": [ds.attrs["radar_lat"]]}
    r.altitude = {"data": [ds.attrs.get("radar_alt_m", 187.0)]}
    half = float(ds.x.values.max()) / 1000.0
    res = float(np.diff(ds.x.values)[0]) / 1000.0
    return polar_meta(r, half, res)


# ---------------------------------------------------------------------------
# 10) แผนที่ PPI
# ---------------------------------------------------------------------------
def field_cmaps():
    """สี/ช่วงค่ามาตรฐานของแต่ละตัวแปร: dict ชื่อ → (cmap, norm, label, ticks)."""
    import matplotlib as mpl
    try:
        import cmweather  # noqa: F401
    except Exception:
        pass

    def disc(name, levels, extend="both"):
        cm = mpl.colormaps[name] if isinstance(name, str) else name
        n_ext = {"both": 2, "min": 1, "max": 1, "neither": 0}[extend]
        cols = cm(np.linspace(0, 1, len(levels) - 1 + n_ext))
        if extend in ("both", "min"):
            under, cols = cols[0], cols[1:]
        if extend in ("both", "max"):
            over, cols = cols[-1], cols[:-1]
        c = mpl.colors.ListedColormap(cols)
        if extend in ("both", "min"):
            c.set_under(under)
        if extend in ("both", "max"):
            c.set_over(over)
        return c, mpl.colors.BoundaryNorm(levels, c.N), levels

    def cm_or(name, fallback):
        return name if name in mpl.colormaps else fallback

    out = {}
    from . import mapplot as _mp
    c, n = _mp.refl_cmap_norm()
    out["Z"] = (c, n, "Reflectivity $Z_H$ (dBZ)", _mp.REFL_LEVELS[::2])
    lv = np.arange(-24, 25, 3)
    c, n, t = disc(cm_or("balance", "RdBu_r"), lv)
    out["V"] = (c, n, "Radial velocity $V_r$ (m s$^{-1}$)", lv[::2])
    lv = np.arange(-8, 9, 2)
    c, n, t = disc(cm_or("balance", "RdBu_r"), lv)
    out["VA"] = (c, n, "Aliased $V_r$ (m s$^{-1}$)", lv)
    lv = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8])
    c, n, t = disc(cm_or("ChaseSpectral", "viridis"), lv, "max")
    out["W"] = (c, n, "Spectrum width $\\sigma_v$ (m s$^{-1}$)", lv)
    lv = np.array([-1, -0.5, 0, 0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5])
    c, n, t = disc(cm_or("HomeyerRainbow", "turbo"), lv)
    out["ZDR"] = (c, n, "$Z_{DR}$ (dB)", lv)
    lv = np.array([0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 1.0])
    c, n, t = disc(cm_or("plasmidis", "plasma"), lv, "min")
    out["RHO"] = (c, n, "$\\rho_{HV}$", lv[::2])
    lv = np.arange(-180, 181, 30)
    c, n, t = disc(cm_or("HomeyerRainbow", "turbo"), lv, "neither")
    out["PHI"] = (c, n, "$\\Phi_{DP}$ (°)", lv[::2])
    lv = np.array([0, 5, 10, 20, 30, 40, 60, 80, 100, 150])
    c, n, t = disc(cm_or("Theodore16", "magma_r"), lv, "max")
    out["PHIP"] = (c, n, "Processed $\\Phi_{DP}$ (°)", lv)
    lv = np.array([-0.5, 0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6])
    c, n, t = disc(cm_or("HomeyerRainbow", "turbo"), lv)
    out["KDP"] = (c, n, "$K_{DP}$ (° km$^{-1}$)", lv)
    lv = np.array([-4, -3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3, 4])    # หน่วย 10⁻³ s⁻¹
    c, n, t = disc("PuOr_r", lv)
    out["DIV"] = (c, n, "Radial divergence (10$^{-3}$ s$^{-1}$)", lv)
    out["AZSHEAR"] = (c, n, "Azimuthal shear (10$^{-3}$ s$^{-1}$)", lv)
    lv = np.array([0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4])
    c, n, t = disc("YlOrRd", lv, "max")
    out["EDR"] = (c, n, "EDR$^{1/3}$ (m$^{2/3}$ s$^{-1}$)", lv)
    c, n = hid_cmap_norm()
    out["HID"] = (c, n, "Hydrometeor class", np.arange(1, 10))
    lv = np.array([0, 0.5, 1, 2, 3, 4, 6, 8, 10])
    c, n, t = disc("Reds", lv, "max")
    out["PIA"] = (c, n, "PIA (dB)", lv)
    return out


def ppi_map(ax, radar, field, sweep, meta, kind=None, title=None, zoom=None, basemap=True,
            scalebar=True, north=True, gridline_labels=("left", "bottom"), points=True,
            cmap=None, norm=None, scale=1.0):
    """แผนที่ PPI คุณภาพตีพิมพ์ (projection AEQD + ขอบเขตจังหวัด/อำเภอ + วงระยะ)

    kind = กุญแจใน field_cmaps() เช่น 'Z', 'V', 'W', 'ZDR', 'RHO', 'KDP', 'HID', 'EDR', 'DIV'
    scale = ตัวคูณข้อมูลก่อนพลอต (เช่น 1e3 สำหรับ DIV → 10⁻³ s⁻¹ ให้ระบุ kind='DIV' ได้เลย)
    """
    import matplotlib as mpl
    from . import mapplot as mp
    cm = field_cmaps()
    lab, ticks = field, None
    if kind is not None:
        c0, n0, lab, ticks = cm[kind]
        cmap = cmap or c0
        norm = norm or n0
    crs = mp.get_crs(meta)
    ax.set_extent(zoom or mp.extent_of(meta), crs=crs)
    ax.set_facecolor("white")
    # พื้นเทาอ่อน = อยู่ในรัศมีเรดาร์
    from matplotlib.patches import Circle
    ax.add_patch(Circle((0, 0), meta["x2"] - meta["xpixelsize"] / 2, transform=crs, facecolor="#f4f4f4",
                        edgecolor="none", zorder=1))
    x, y, _ = radar.get_gate_x_y_z(sweep, edges=True)
    if kind in ("DIV", "AZSHEAR") and scale == 1.0:
        scale = 1e3                                         # s⁻¹ → 10⁻³ s⁻¹
    d = field_data(radar, field, sweep) * scale
    if kind == "HID":
        d = np.where(d >= 1, d, np.nan)
    im = ax.pcolormesh(x, y, np.ma.masked_invalid(d), cmap=cmap, norm=norm, transform=crs,
                       zorder=2, rasterized=True)
    im._label, im._ticks = lab, ticks
    if basemap:
        mp.add_basemap(ax, meta, labels=gridline_labels)
    if points:
        mp.add_points(ax, meta)
    if scalebar:
        mp.add_scalebar(ax)
    if north:
        mp.add_north_arrow(ax)
    if title:
        ax.set_title(title, loc="left", fontsize=mpl.rcParams["axes.titlesize"] - 1)
    return im


def hid_colorbar(fig, im, axs, orientation="horizontal", shrink=0.8, **kw):
    """แถบสี HID พร้อมชื่อย่อคลาส."""
    cb = fig.colorbar(im, ax=list(np.ravel(axs)), orientation=orientation, shrink=shrink,
                      pad=0.02, aspect=40, ticks=np.arange(1, 10), **kw)
    cb.set_ticklabels(HID_NAMES[1:])
    cb.set_label("Hydrometeor class (Py-ART semi-supervised)")
    cb.outline.set_linewidth(0.5)
    return cb


def plot_rhi(ax, xs, field, kind=None, cmap=None, norm=None, scale=1.0, xlim=None, ylim=(0, 14),
             isotherms=None):
    """พลอต pseudo-RHI จาก cross_section_ppi — แกน x = ระยะ (km), y = ความสูง (km MSL)."""
    cm = field_cmaps()
    lab = field
    if kind is not None:
        c0, n0, lab, _ = cm[kind]
        cmap, norm = cmap or c0, norm or n0
    rng = xs.range["data"] / 1000.0
    dr = rng[1] - rng[0]
    re = np.append(rng - dr / 2, rng[-1] + dr / 2)
    el = xs.elevation["data"]
    alt0 = float(xs.altitude["data"][0]) / 1000.0
    if kind in ("DIV", "AZSHEAR") and scale == 1.0:
        scale = 1e3
    d = field_data(xs, field) * scale
    if kind == "HID":
        d = np.where(d >= 1, d, np.nan)
    im = None
    for i, e in enumerate(el):
        lo, hi = e - BEAMWIDTH_DEG / 2, e + BEAMWIDTH_DEG / 2
        R = np.vstack([re, re])
        H = np.vstack([beam_height_km(re, lo, alt0 * 1000), beam_height_km(re, hi, alt0 * 1000)])
        Rg = R * np.cos(np.radians(np.vstack([np.full_like(re, lo), np.full_like(re, hi)])))
        im = ax.pcolormesh(Rg, H, np.ma.masked_invalid(d[i:i + 1, :]), cmap=cmap, norm=norm,
                           shading="flat", rasterized=True)
    if isotherms:
        for lbl, h in isotherms.items():
            ax.axhline(h / 1000.0, color="0.2", lw=0.8, ls="--")
            ax.text(0.99, h / 1000.0, f" {lbl}", transform=ax.get_yaxis_transform(), ha="right",
                    va="bottom", fontsize=7, color="0.2")
    ax.set_xlabel("Distance from radar (km)")
    ax.set_ylabel("Height (km MSL)")
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, lw=0.3, alpha=0.5)
    im._label = lab
    return im


def run_whole_day(data_dir, out_dir, sites=None, files=None, verbose=True):
    """ประมวลผลทุก volume ของวัน → ตารางตัวชี้วัด (CSV) + แผนที่ swath (NetCDF)

    ใช้เวลา ~10–15 s ต่อ volume (≈ 15–25 นาทีทั้งวันใน Colab)
    """
    import time
    import pandas as pd
    df = load_profile(_os.path.join(data_dir, "sounding", "sounding_48431_2024-05-03_00Z.csv"))
    files = files or polar_files(data_dir)
    _os.makedirs(out_dir, exist_ok=True)
    rows, S, meta, t0 = [], None, None, time.time()
    for k, f in enumerate(files):
        try:
            r = read_volume(f)
            meta = meta or polar_meta(r)
            S = S or swath_init(meta)
            process_volume(r, df)
            m, g = volume_metrics(r, meta, sites=sites)
            swath_update(S, g, m["time_utc"])
            rows.append(m)
            del r, g
            import gc
            gc.collect()
            if verbose:
                print(f"[{k + 1:2d}/{len(files)}] {m['time_utc']:%H:%M} UTC  Zmax={m['Z_max']:.1f}  "
                      f"hail={m.get('area_hail_km2', np.nan):.0f} km²  ({time.time() - t0:.0f} s)")
        except Exception as e:                                  # noqa: BLE001
            print(f"  ⚠️ {_os.path.basename(f)}: {e}")
    tab = pd.DataFrame(rows).set_index("time_utc")
    tab.to_csv(_os.path.join(out_dir, "KKN_20240503_volume_metrics.csv"), float_format="%.5g")
    save_swaths(_os.path.join(out_dir, "KKN_20240503_swaths.nc"), S, meta,
                {"n_volumes": len(rows), "hail_min_dbz": HAIL_MIN_DBZ,
                 "source": "polar_tools.run_whole_day"})
    return tab, S, meta


# ---------------------------------------------------------------------------
# 11) เครื่องมือตรวจผล Doppler
# ---------------------------------------------------------------------------
def jump_count(radar, field, sweeps=None, thr=None):
    """จำนวนคู่ gate ติดกันตามแนวรัศมีที่ความเร็วกระโดด > thr (ค่าตั้งต้น = Vny)

    หลังแก้การพับที่ถูกต้อง ค่าควร 'น้อยมาก' — ใช้เป็นตัวชี้วัดคุณภาพแบบไม่ต้องรู้คำตอบจริง
    """
    thr = nyquist(radar) if thr is None else thr
    v = field_data(radar, field)
    n = 0
    for s in (range(radar.nsweeps) if sweeps is None else sweeps):
        d = np.abs(np.diff(v[radar.get_slice(s)], axis=1))
        n += int(np.nansum(d > thr))
    return n


def site_velocity_stats(radar, meta, lon, lat, radius_km=10.0, vel="VEL_RB", sweep=0):
    """สถิติ Doppler รอบจุด: Vmax, Vmin, ΔV = Vmax − Vmin, DIVmax (10⁻³ s⁻¹), Zmax."""
    m = site_gates(radar, meta, lon, lat, radius_km, sweeps=[sweep])
    v = field_data(radar, vel)[m]
    z = field_data(radar, F["Z"])[m]
    out = {"n_gates": int(np.isfinite(v).sum()),
           "Zmax": float(np.nanmax(z)) if np.isfinite(z).any() else np.nan}
    if np.isfinite(v).sum() >= 5:
        out.update(Vmax=float(np.nanmax(v)), Vmin=float(np.nanmin(v)),
                   dV=float(np.nanmax(v) - np.nanmin(v)))
    else:
        out.update(Vmax=np.nan, Vmin=np.nan, dV=np.nan)
    if "DIV" in radar.fields:
        d = field_data(radar, "DIV")[m]
        out["DIVmax"] = float(np.nanmax(d) * 1e3) if np.isfinite(d).any() else np.nan
    return out


def keep_fields(radar, names):
    """ลบ field อื่นออกเพื่อประหยัดหน่วยความจำ (เช่น เมื่อเก็บหลาย volume ไว้พร้อมกัน)."""
    for f in list(radar.fields):
        if f not in names:
            radar.fields.pop(f)
    return radar
