"""ขั้นเตรียมข้อมูลเรดาร์ (prep/P1, P2) — ใช้โค้ดฐานเดียวกับ repo ก่อนหน้า

ที่มาของอัลกอริทึม (ไม่ได้แก้ค่าใด ๆ เพื่อให้ผลวันที่ 3 พ.ค. 2567 เทียบกับงานเดิมได้)
- gentle base-QC + pseudo-CAPPI : 00_prepare_pcappi_for_pysteps.ipynb (Teach_Pysteps_KKN_summerStorm)
- dealias / KDP / ZDR / Z-PHI / HID / EDR / ตัวชี้วัด : utils/polar_tools.py (Teach_HID_Doppler_KKN_summerStorm)

ส่วนที่เพิ่มใหม่: อ่านไฟล์ผ่าน utils/radar_io (ทน .bz2 ที่เป็น gzip), ทำงานหลายวันแบบขนาน,
เขียนผลเป็นกริด Cartesian 1 km (AEQD ศูนย์กลางเรดาร์, y เรียงจากใต้ขึ้นเหนือ แบบ pysteps)
"""
from __future__ import annotations

import datetime as _dt
import warnings

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter1d
from scipy.ndimage import label as cc_label

PREP_VERSION = "kkn-may2024-prep-1.0 (base: kkn-prep-1.0.0)"

# ---------------- gentle base-QC (ค่าเดียวกับ repo ก่อน) ----------------
ZH_ECHO_FLOOR_DBZ = 0.0
NONMET_RHOHV_MAX = 0.70
NONMET_ZH_MAX_DBZ = 25.0
SPECKLE_MIN_GATES = 12
FLARE_MIN_LEN_KM = 120.0
FLARE_ZH_MIN_DBZ = 5.0
FLARE_RHOHV_MEAN_MAX = 0.75
FLARE_VELVALID_MAX = 0.30
FLARE_PROTECT_DBZ = 45.0
FLARE_PROTECT_FRAC = 0.03
SW_MAX = 12.0
QC_SD_PHIDP_W = 5
QC_SD_PHIDP_MAX = 20.0

# ---------------- pseudo-CAPPI ----------------
PCAPPI_HEIGHT_M = 2000.0          # เหนือระดับเรดาร์
GRID_RES_M = 1000.0
RANGE_MAX_KM = 240.0
ZEROVALUE_DBZ = -15.0             # อยู่ในรัศมีแต่ไม่มี echo
PCAPPI_MIN_DBZ = 0.0
EFFECTIVE_EARTH_K = 4.0 / 3.0
AREA_THRESHOLDS_DBZ = [20, 30, 35, 40, 45, 50]

QC_BITS = {
    "no_echo_or_below_floor": 1,
    "flare_ray": 2,
    "nonmet_low_rhohv_weak_z": 4,
    "speckle": 8,
    "spectrum_width_invalid": 16,
    "velocity_invalid": 32,
    "phidp_texture_high": 64,
}
# ชื่อ field หลัง QC — ตรงกับไฟล์ lean polar ของ repo2 จึงใช้ polar_tools ได้ทันที
LEAN = {"REFL": "reflectivity_qc", "VEL": "velocity_qc", "SW": "spectrum_width_qc",
        "ZDR": "differential_reflectivity_qc", "PHIDP": "differential_phase_qc",
        "RHO": "cross_correlation_ratio_qc"}


# ======================================================================
# อ่านไฟล์และ QC (COMMON FUNCTIONS A ของ repo ก่อน)
# ======================================================================
def pick_field(radar, candidates):
    for c in candidates:
        if c in radar.fields:
            return c
    return None


def farr(radar, field):
    if field is None or field not in radar.fields:
        return None
    return np.ma.filled(radar.fields[field]["data"].astype(np.float32), np.nan)


def detect_fields(radar):
    f = {
        "REFL": pick_field(radar, ["corrected_reflectivity", "reflectivity"]),
        "VEL": pick_field(radar, ["velocity"]),
        "SW": pick_field(radar, ["spectrum_width"]),
        "ZDR": pick_field(radar, ["corrected_differential_reflectivity", "differential_reflectivity"]),
        "PHIDP": pick_field(radar, ["differential_phase"]),
        "RHO": pick_field(radar, ["cross_correlation_ratio"]),
    }
    if f["REFL"] is None:
        raise ValueError(f"No reflectivity field. Available: {list(radar.fields)}")
    return f


def radar_start_time(radar):
    import pyart
    t = pyart.util.datetime_from_radar(radar)
    return _dt.datetime(t.year, t.month, t.day, t.hour, t.minute, int(t.second))


def get_nyquist(radar):
    try:
        v = np.asarray(radar.instrument_parameters["nyquist_velocity"]["data"], float)
        v = v[np.isfinite(v) & (v > 0)]
        return float(np.nanmedian(v)) if v.size else np.nan
    except Exception:
        return np.nan


def sweep_index_1d(radar):
    swp = np.zeros(radar.nrays, dtype=np.int16)
    for k in range(radar.nsweeps):
        a = int(radar.sweep_start_ray_index["data"][k])
        b = int(radar.sweep_end_ray_index["data"][k])
        swp[a:b + 1] = k
    return swp


def find_flare_rays(radar, zh, rho, vel_finite):
    dr_km = float(np.nanmedian(np.diff(radar.range["data"])) / 1000.0)
    min_gate_count = int(FLARE_MIN_LEN_KM / max(dr_km, 0.1))
    valid = np.isfinite(zh) & (zh > FLARE_ZH_MIN_DBZ)
    spike = np.zeros(radar.nrays, dtype=bool)
    for i in range(radar.nrays):
        v = valid[i]
        if v.sum() < min_gate_count:
            continue
        if np.mean(zh[i][v] >= FLARE_PROTECT_DBZ) >= FLARE_PROTECT_FRAC:
            continue
        rho_ok = True if rho is None else np.nanmean(rho[i][v]) <= FLARE_RHOHV_MEAN_MAX
        vel_ok = True if vel_finite is None else np.mean(vel_finite[i][v]) <= FLARE_VELVALID_MAX
        spike[i] = rho_ok and vel_ok
    return spike, binary_dilation(spike, iterations=1)


def remove_speckles_polar(mask_keep, radar, min_gates=12, pad=8):
    keep = mask_keep.copy()
    removed = np.zeros_like(keep, dtype=bool)
    swp = sweep_index_1d(radar)
    structure = np.ones((3, 3), dtype=np.int8)
    for k in range(radar.nsweeps):
        ridx = np.where(swp == k)[0]
        if len(ridx) == 0:
            continue
        sub = keep[ridx, :].copy()
        if len(ridx) > 2 * pad:
            padded = np.vstack([sub[-pad:], sub, sub[:pad]])
            lab_p, _ = cc_label(padded, structure=structure)
            sizes = np.bincount(lab_p.ravel())
            lab = lab_p[pad:pad + len(ridx), :]
        else:
            lab, _ = cc_label(sub, structure=structure)
            sizes = np.bincount(lab.ravel())
        small = np.isin(lab, np.where(sizes < min_gates)[0]) & (lab > 0)
        sub[small] = False
        keep[ridx, :] = sub
        removed[ridx, :] = small
    return keep, removed


def circular_sd_deg_range(data_deg, window=5):
    x = np.asarray(data_deg, dtype=float)
    valid = np.isfinite(x)
    a = np.deg2rad(np.where(valid, x, 0.0))
    cnt = uniform_filter1d(valid.astype(float), window, axis=1, mode="nearest") * window
    s = uniform_filter1d(np.where(valid, np.sin(a), 0), window, axis=1, mode="nearest") * window
    c = uniform_filter1d(np.where(valid, np.cos(a), 0), window, axis=1, mode="nearest") * window
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.clip(np.hypot(s / cnt, c / cnt), 1e-6, 1.0)
        sd = np.rad2deg(np.sqrt(-2.0 * np.log(R)))
    sd[cnt < 3] = np.nan
    return sd.astype(np.float32)


def gentle_base_qc(radar, fields):
    """keep = ZH ≥ floor และไม่ใช่ flare ray และไม่ใช่ (ρHV ต่ำ + ZH อ่อน) และไม่ใช่ speckle"""
    zh = farr(radar, fields["REFL"])
    rho = farr(radar, fields["RHO"])
    vel = farr(radar, fields["VEL"])
    sw = farr(radar, fields["SW"])
    phidp = farr(radar, fields["PHIDP"])
    vel_finite = None if vel is None else np.isfinite(vel)

    echo = np.isfinite(zh) & (zh >= ZH_ECHO_FLOOR_DBZ)
    _, flare_dil = find_flare_rays(radar, zh, rho, vel_finite)
    flare = np.zeros_like(echo)
    flare[flare_dil, :] = True
    if rho is not None:
        nonmet = echo & ~flare & np.isfinite(rho) & (rho < NONMET_RHOHV_MAX) & (zh < NONMET_ZH_MAX_DBZ)
    else:
        nonmet = np.zeros_like(echo)
    keep, speckle = remove_speckles_polar(echo & ~flare & ~nonmet, radar, min_gates=SPECKLE_MIN_GATES)

    out = {"zh_raw": zh, "keep": keep, "reflectivity_qc": np.where(keep, zh, np.nan).astype(np.float32)}
    flag = np.zeros(zh.shape, dtype=np.uint8)
    flag[~echo] |= QC_BITS["no_echo_or_below_floor"]
    flag[flare] |= QC_BITS["flare_ray"]
    flag[nonmet] |= QC_BITS["nonmet_low_rhohv_weak_z"]
    flag[speckle] |= QC_BITS["speckle"]
    if vel is not None:
        vel_valid = keep & np.isfinite(vel)
        out["velocity_qc"] = np.where(vel_valid, vel, np.nan).astype(np.float32)
        flag[keep & ~vel_valid] |= QC_BITS["velocity_invalid"]
    if sw is not None:
        sw_valid = keep & np.isfinite(sw) & (sw >= 0.0) & (sw <= SW_MAX)
        out["spectrum_width_qc"] = np.where(sw_valid, sw, np.nan).astype(np.float32)
        flag[keep & ~sw_valid] |= QC_BITS["spectrum_width_invalid"]
    if phidp is not None:
        tex = circular_sd_deg_range(phidp, window=QC_SD_PHIDP_W)
        phase_ok = np.isfinite(phidp) & (tex < QC_SD_PHIDP_MAX)
        flag[keep & ~phase_ok] |= QC_BITS["phidp_texture_high"]
    out["qc_flag"] = flag
    n_raw = int(np.isfinite(zh).sum())
    out["counts"] = {"raw_finite": n_raw, "echo_floor": int(echo.sum()), "flare_rays": int(flare_dil.sum()),
                     "nonmet_removed": int(nonmet.sum()), "speckle_removed": int(speckle.sum()),
                     "kept": int(keep.sum()), "kept_pct_of_raw": round(100.0 * keep.sum() / max(n_raw, 1), 2)}
    return out


def to_lean(radar, fields, qc):
    """แทนที่ field ดิบด้วย field หลัง QC ที่ใช้ชื่อเดียวกับไฟล์ lean ของ repo2 (ทำในหน่วยความจำ)"""
    keep = qc["keep"]
    src = {
        "reflectivity_raw": qc["zh_raw"],
        LEAN["REFL"]: qc["reflectivity_qc"],
        LEAN["VEL"]: qc.get("velocity_qc"),
        LEAN["SW"]: qc.get("spectrum_width_qc"),
        LEAN["ZDR"]: None if fields["ZDR"] is None else np.where(keep, farr(radar, fields["ZDR"]), np.nan),
        LEAN["PHIDP"]: None if fields["PHIDP"] is None else np.where(keep, farr(radar, fields["PHIDP"]), np.nan),
        LEAN["RHO"]: None if fields["RHO"] is None else np.where(keep, farr(radar, fields["RHO"]), np.nan),
    }
    new = {}
    for name, data in src.items():
        if data is not None:
            new[name] = {"data": np.ma.masked_invalid(np.asarray(data, np.float32)), "units": "", "_FillValue": -9999.0}
    new["qc_flag"] = {"data": qc["qc_flag"], "units": "1"}
    radar.fields = new
    return radar


def read_qc_volume(path):
    """อ่าน UF → base-QC → radar ที่มี field แบบ lean พร้อมใช้กับ polar_tools"""
    from .radar_io import read_uf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        radar = read_uf(path)
    fields = detect_fields(radar)
    qc = gentle_base_qc(radar, fields)
    to_lean(radar, fields, qc)
    radar.metadata["qc_counts"] = qc["counts"]
    return radar


# ======================================================================
# กริดและ pseudo-CAPPI
# ======================================================================
def grid_axes(half_km=RANGE_MAX_KM, res_m=GRID_RES_M):
    h = half_km * 1e3
    x = np.arange(-h, h + res_m / 2, res_m)
    return x, x.copy()                     # y เรียงจากใต้ขึ้นเหนือ (yorigin = "lower")


def grid_meta(radar, half_km=RANGE_MAX_KM, res_m=GRID_RES_M):
    """meta แบบ polar_tools แต่ y เรียงจากใต้ขึ้นเหนือ ให้ตรงกับกริด pseudo-CAPPI"""
    from . import polar_tools as pt
    meta = pt.polar_meta(radar, half_km, res_m / 1000.0)
    x, y = grid_axes(half_km, res_m)
    X, Y = np.meshgrid(x, y)
    meta.update(x=x, y=y, coverage=np.hypot(X, Y) <= half_km * 1e3, yorigin="lower")
    return meta


def beam_height_and_ground_range(rng_m, elev_deg):
    ae = EFFECTIVE_EARTH_K * 6371000.0
    er = np.deg2rad(elev_deg)
    h = np.sqrt(rng_m ** 2 + ae ** 2 + 2 * rng_m * ae * np.sin(er)) - ae
    s = ae * np.arcsin(rng_m * np.cos(er) / (ae + h))
    return h, s


def make_pseudo_cappi(radar, zh_qc, height_m=PCAPPI_HEIGHT_M):
    """pseudo-CAPPI: ประมาณค่าในแนวตั้งด้วย Z เชิงเส้นระหว่าง 2 มุมยกที่คร่อมความสูงเป้าหมาย
    ใกล้เรดาร์ใช้มุมสูงสุด ไกลเรดาร์ใช้มุมต่ำสุด แนวราบใช้ gate ใกล้สุด (โค้ดเดียวกับ repo pysteps)"""
    x, y = grid_axes()
    X, Y = np.meshgrid(x, y)
    S = np.hypot(X, Y)
    AZ = np.rad2deg(np.arctan2(X, Y)) % 360.0
    rng = radar.range["data"].astype(float)
    ng = radar.ngates
    dr = float(np.median(np.diff(rng)))
    nsw = radar.nsweeps
    H = np.full((nsw,) + X.shape, np.nan, np.float32)
    ZL = np.full((nsw,) + X.shape, np.nan, np.float32)
    z_zero = 10.0 ** (ZEROVALUE_DBZ / 10.0)
    for k in range(nsw):
        sl = radar.get_slice(k)
        el = float(np.nanmean(radar.elevation["data"][sl]))
        h_g, s_g = beam_height_and_ground_range(rng, el)
        inside = (S >= s_g[0] - dr / 2) & (S <= s_g[-1] + dr / 2) & (S <= RANGE_MAX_KM * 1e3)
        gidx = np.clip(np.rint(np.interp(S, s_g, np.arange(ng))), 0, ng - 1).astype(int)
        hk = np.interp(S, s_g, h_g)
        az = radar.azimuth["data"][sl].astype(float) % 360.0
        order = np.argsort(az)
        az_s = az[order]
        ext = np.concatenate([az_s - 360.0, az_s, az_s + 360.0])
        idx_ext = np.tile(order, 3)
        j = np.clip(np.searchsorted(ext, AZ), 1, len(ext) - 1)
        jn = np.where(np.abs(ext[j - 1] - AZ) <= np.abs(ext[j] - AZ), j - 1, j)
        daz = np.abs(ext[jn] - AZ)
        ok = daz <= max(1.5 * np.median(np.diff(az_s)), 0.5)
        ray = sl.start + idx_ext[jn]
        cov = inside & ok
        val = zh_qc[ray, gidx]
        lin = np.where(np.isfinite(val), 10.0 ** (val / 10.0), z_zero)
        H[k] = np.where(cov, hk, np.nan)
        ZL[k] = np.where(cov, lin, np.nan)
    below = np.isfinite(H) & (H <= height_m)
    above = np.isfinite(H) & (H > height_m)
    has_lo, has_hi = below.any(0), above.any(0)
    k_lo = np.where(has_lo, nsw - 1 - np.argmax(below[::-1], axis=0), 0)
    k_hi = np.where(has_hi, np.argmax(above, axis=0), 0)
    take = lambda A, k: np.take_along_axis(A, k[None], axis=0)[0]  # noqa: E731
    h_lo, h_hi = take(H, k_lo), take(H, k_hi)
    z_lo, z_hi = take(ZL, k_lo), take(ZL, k_hi)
    w = np.clip((height_m - h_lo) / np.where(h_hi > h_lo, h_hi - h_lo, 1.0), 0, 1)
    zlin = np.full(X.shape, np.nan, np.float32)
    src_h = np.full(X.shape, np.nan, np.float32)
    both = has_lo & has_hi
    zlin[both] = (1 - w[both]) * z_lo[both] + w[both] * z_hi[both]
    src_h[both] = height_m
    only_lo = has_lo & ~has_hi
    zlin[only_lo], src_h[only_lo] = z_lo[only_lo], h_lo[only_lo]
    only_hi = has_hi & ~has_lo
    zlin[only_hi], src_h[only_hi] = z_hi[only_hi], h_hi[only_hi]
    coverage = np.isfinite(zlin)
    with np.errstate(divide="ignore", invalid="ignore"):
        dbz = 10.0 * np.log10(zlin)
    dbz[coverage & (dbz < PCAPPI_MIN_DBZ)] = ZEROVALUE_DBZ
    return {"dbz": dbz.astype(np.float32), "source_height_m": src_h, "coverage": coverage, "x": x, "y": y}


def echo_stats(dbz, prefix=""):
    px_km2 = (GRID_RES_M / 1000.0) ** 2
    out = {f"{prefix}area_ge{t}_km2": float(np.nansum(dbz >= t) * px_km2) for t in AREA_THRESHOLDS_DBZ}
    out[f"{prefix}max_dbz"] = float(np.nanmax(dbz)) if np.isfinite(dbz).any() else np.nan
    return out


# ======================================================================
# งานต่อ 1 volume (เรียกแบบขนานจาก P1/P2)
# ======================================================================
def p1_volume(path, slot_utc):
    """P1: UF → QC → pseudo-CAPPI 2 km + Z สูงสุดในคอลัมน์ + สถิติ echo"""
    from . import polar_tools as pt
    try:
        r = read_qc_volume(path)
        zqc = np.ma.filled(r.fields[LEAN["REFL"]]["data"].astype(np.float32), np.nan)
        pc = make_pseudo_cappi(r, zqc)
        meta = grid_meta(r)
        zcol, _ = pt.column_grid(r, LEAN["REFL"], meta, how="max")
        zcol = np.where(meta["coverage"] & np.isfinite(zcol), zcol, np.where(meta["coverage"], ZEROVALUE_DBZ, np.nan))
        row = {"slot_utc": slot_utc, "status": "ok", "scan_start": radar_start_time(r), "nyquist": get_nyquist(r),
               "nsweeps": r.nsweeps, **r.metadata["qc_counts"], **echo_stats(pc["dbz"]),
               **echo_stats(zcol, prefix="col_")}
        site = {"lat": float(r.latitude["data"][0]), "lon": float(r.longitude["data"][0]),
                "alt": float(r.altitude["data"][0])}
        return row, pc["dbz"], zcol.astype(np.float32), pc["source_height_m"], site
    except Exception as e:  # noqa: BLE001
        return {"slot_utc": slot_utc, "status": "failed", "error": f"{type(e).__name__}: {e}"[:300]}, None, None, None, None


PHIDP_SIGN = -1.0   # KKN: ΦDP ลดลงตามระยะทุก sweep (repo2) → กลับเครื่องหมายเสมอ


def process_volume_kkn(radar, df, edr=True, phidp_sign=PHIDP_SIGN):  # noqa: D401
    """เหมือน polar_tools.process_volume ทุกขั้น ยกเว้น ΦDP ใช้เครื่องหมายคงที่ (−1) แทนการตรวจอัตโนมัติ

    เหตุผล: การตรวจอัตโนมัติ (phidp_slope_by_sweep) คืน NaN เมื่อ sweep มี ray ที่ใช้ได้น้อย
    และ NaN < 0 เป็น False → ได้ +1 (ผิด) ทั้งที่ ray ที่มี gate ดี ≥ 10 ยังถูกประมวลผล
    ใน 92 volume ของวันที่ 3 พ.ค. พบ sweep ที่ได้เครื่องหมาย +1 จากกรณีนี้หลายครั้ง
    ค่าความชันอัตโนมัติยังถูกบันทึกในตัวชี้วัด (phidp_slope_s0..s3) เพื่อตรวจย้อนหลัง
    """
    from . import polar_tools as pt
    slopes = pt.phidp_slope_by_sweep(radar)
    pt.add_temperature(radar, df)
    pt.dealias_region(radar)
    pt.radial_divergence(radar, "VEL_RB")
    pt.azimuthal_shear(radar, "VEL_RB")
    pt.process_phidp(radar, sign=phidp_sign)
    pt.add_kdp(radar)
    pt.correct_zdr_bias(radar)
    pt.correct_attenuation(radar)
    pt.classify_hid(radar)
    if edr:
        pt.add_edr(radar)
    return {f"phidp_slope_s{s}": v[0] for s, v in slopes.items()}


# ผลิตภัณฑ์ polar บนกริด — ชื่อและหน่วยเดียวกับ track_tools.POLAR_VARS ของ repo3
# (div/azshear หน่วย s⁻¹; cell_table ของ repo3 คูณ 1e3 เอง)  ชื่อ: (หน่วย, scale int16, คำอธิบาย)
POLAR_ENCODING = {
    "Zc": ("dBZ", 0.1, "reflectivity, Z-PHI attenuation-corrected, lowest sweep"),
    "vel": ("m s-1", 0.1, "dealiased radial velocity (region-based, tuned), lowest sweep"),
    "div": ("s-1", 1e-5, "radial divergence dVr/dr, lowest sweep"),
    "azshear": ("s-1", 1e-5, "azimuthal shear (1/r) dVr/dphi, lowest sweep (signed)"),
    "edr": ("m2/3 s-1", 1e-3, "column-maximum EDR^(1/3) (PyTDA NTDA)"),
    "kdp": ("deg km-1", 0.01, "column-maximum KDP (Vulpiani)"),
    "zdr": ("dB", 0.01, "ZDR bias- and attenuation-corrected, lowest sweep; values outside -4..8 dB set missing"),
}
ZDR_VALID = (-4.0, 8.0)


def p2_volume(path, slot_utc, profile_df, do_edr=True):
    """P2: UF → QC → process_volume_kkn → กริด 1 km แบบ repo3 (track_tools.polar_volume_to_grid) + ตัวชี้วัด

    profile_df : DataFrame คอลัมน์ height (m MSL), temperature (°C) ครบ 0–20 km (จาก utils.sounding)
    คืน (dict ตัวชี้วัด, dict กริด {ชื่อ: array[y, x]} แถวแรก = ทิศเหนือ)
    """
    from . import polar_tools as pt
    from . import track_tools as tt
    try:
        r = read_qc_volume(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            diag = process_volume_kkn(r, profile_df, edr=do_edr)
        r.metadata["time_utc"] = slot_utc.to_pydatetime() if hasattr(slot_utc, "to_pydatetime") else slot_utc
        meta = pt.polar_meta(r, RANGE_MAX_KM, GRID_RES_M / 1000.0)          # yorigin upper เหมือน repo3
        if "EDR" not in r.fields:                                            # ไม่มี PyTDA → เติม NaN ให้ฟังก์ชันของ repo3 ทำงานได้
            pt.add_field(r, "EDR", np.full((r.nrays, r.ngates), np.nan))
        g = tt.polar_volume_to_grid(r, meta)
        g["zdr"] = np.where((g["zdr"] >= ZDR_VALID[0]) & (g["zdr"] <= ZDR_VALID[1]), g["zdr"], np.nan)
        m, _ = pt.volume_metrics(r, meta, rng_max_km=RANGE_MAX_KM)
        m.pop("time_utc", None)
        m.update({"slot_utc": slot_utc, "status": "ok", **diag, "edr_computed": bool(do_edr)})
        return m, g
    except Exception as e:  # noqa: BLE001
        return {"slot_utc": slot_utc, "status": "failed", "error": f"{type(e).__name__}: {e}"[:300]}, None


def hid_sensitivity(path, profiles, ref="own"):
    """จำแนก HID volume เดียวด้วยโปรไฟล์อุณหภูมิหลายแบบ → สัดส่วน gate ที่คลาสเปลี่ยนเทียบกับ ref

    profiles : {ชื่อ: DataFrame(height, temperature)}
    """
    import pandas as pd
    from . import polar_tools as pt
    out, H = [], {}
    for name, prof in profiles.items():
        r = read_qc_volume(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            process_volume_kkn(r, prof, edr=False)
        H[name] = np.ma.filled(r.fields["HID"]["data"], 0).astype(int)
    base = H[ref]
    valid = base > 0
    for name, h in H.items():
        row = {"profile": name, "n_classified": int(valid.sum()),
               "changed_pct": 100.0 * float(np.mean(h[valid] != base[valid])) if valid.any() else np.nan}
        for c in range(1, 10):
            row[pt.HID_NAMES[c]] = 100.0 * float(np.mean(h[h > 0] == c)) if (h > 0).any() else np.nan
        out.append(row)
    return pd.DataFrame(out).set_index("profile")
