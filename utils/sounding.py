"""Sounding (radiosonde) สำหรับ HID: ดาวน์โหลด → QC เชิงฟิสิกส์ → โปรไฟล์อุณหภูมิรายวันบนกริดความสูงเดียวกัน

ทำไมต้องมีโมดูลนี้
-----------------
การจำแนกชนิดหยาดน้ำฟ้า (HID) และการแก้การลดทอน Z-PHI ใช้อุณหภูมิทุก gate ซึ่ง
``polar_tools.add_temperature`` หาโดย ``np.interp`` — เหนือยอด sounding จะได้อุณหภูมิของระดับบนสุด
"ค้าง" ไปทั้งคอลัมน์ ถ้า sounding สั้น (เช่น 12 พ.ค. 2567 สูงแค่ 1.1 km) ทุก gate จะอุ่นกว่า 20 °C
และไม่มีน้ำแข็งเลย จึงต้องมีโปรไฟล์ที่ครบ 0–20 km ทุกวัน และบอกได้ว่ามาจากไหน

ลำดับการเลือกโปรไฟล์ของแต่ละวัน (ตกลงกับผู้สอนแล้ว)
-----------------------------------------------------
1. ``own``            โคราช (48431) ของวันนั้น ผ่าน QC และสูงถึง ≥ FULL_TOP_M
2. ``spliced``        โคราชของวันนั้นผ่านระดับ 0 °C แล้ว แต่สั้นกว่า FULL_TOP_M → ใช้ข้อมูลจริงถึงยอด
                       แล้วต่อด้วยโปรไฟล์เฉลี่ยทั้งเดือน (composite) ที่เลื่อนให้ต่อเนื่อง
3. ``ubon_own`` / ``ubon_spliced``   อุบลราชธานี (48407) วันเดียวกัน ถ้าโคราชใช้ไม่ได้
4. ``neighbor_mean``  ค่าเฉลี่ยของวันก่อนและหลังที่ใกล้ที่สุด (จากข้อ 1–2) ภายใน ±MAX_NEIGHBOR_DAYS
5. ``composite``      มัธยฐานทั้งเดือน (ใช้เมื่อไม่มีทางเลือกอื่น)
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

UWYO_ENDPOINT = "https://weather.uwyo.edu/wsgi/sounding"
BASE_DATASET_URL = ("https://raw.githubusercontent.com/nattaponm/Teaching_Sounding_SevereStorm_Korat/main/"
                    "korat_48431_sounding_20240301_20240531.csv.gz")
STATIONS = {"48431": "Nakhon Ratchasima", "48407": "Ubon Ratchathani"}

COLUMN_NAMES = ["pressure_hPa", "height_m", "temperature_C", "dewpoint_C", "relative_humidity_pct",
                "mixing_ratio_gkg", "wind_direction_deg", "wind_speed_source", "theta_K", "theta_e_K", "theta_v_K"]

Z_GRID = np.arange(0.0, 20001.0, 100.0)    # m MSL (ลำคลื่นสูงสุดของ KKN ที่ 240 km ≈ 17.8 km)
FULL_TOP_M = 12000.0                        # สูงกว่าระดับ −40 °C (~11.5 km) → ครอบคลุมทุกคลาสของ HID
MIN_ABOVE_0C_M = 1000.0                     # spliced ได้ต้องมีข้อมูลจริงเหนือระดับ 0 °C อย่างน้อยเท่านี้
SPLICE_BLEND_M = 2000.0                     # ระยะที่ค่าเบี่ยงเบนจาก composite ค่อย ๆ ลดเป็นศูนย์
MAX_NEIGHBOR_DAYS = 3
JUMP_MAX_K = 15.0                           # อุณหภูมิกระโดดระหว่างระดับติดกันเกินนี้ = ข้อมูลเสีย
WARM_ALOFT_M = 10000.0                      # เหนือความสูงนี้ อุณหภูมิต้องต่ำกว่า 0 °C


# ======================================================================
# 1) ดาวน์โหลดและอ่านตาราง Wyoming (ปรับจาก 00_download_Korat_sounding_MarMay2024)
# ======================================================================
def to_plain_text(raw):
    if "<html" in raw.lower() or "<pre" in raw.lower():
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw, "html.parser")
        pre = soup.find("pre")
        return (pre or soup).get_text("\n", strip=False)
    return raw


def detect_wind_unit(text):
    u = text.upper()
    if "SKNT" in u or "KNOT" in u:
        return "knot"
    if "SPED" in u or "M/S" in u:
        return "m/s"
    return "unknown"


def parse_wyoming_text(plain):
    """ตาราง TEXT:LIST 11 คอลัมน์ (กว้างคอลัมน์ละ 7 ตัวอักษร) → DataFrame หน่วยมาตรฐาน (ลม m/s)"""
    lines = plain.splitlines()
    hdr = next((i for i, ln in enumerate(lines)
                if all(k in ln.upper() for k in ("PRES", "HGHT", "TEMP", "DWPT"))), None)
    if hdr is None:
        raise ValueError("ไม่พบหัวตาราง sounding")
    unit = detect_wind_unit("\n".join(lines[hdr:hdr + 4]))
    stop = ("STATION INFORMATION", "STATION IDENTIFIER", "SHOWALTER", "LIFTED INDEX", "PRECIPITABLE WATER")
    rows = []
    for ln in lines[hdr + 1:]:
        s = ln.strip()
        if not s:
            continue
        if any(t in s.upper() for t in stop):
            break
        if "HPA" in s.upper() or set(s) <= set("- "):
            continue
        f = [ln.ljust(77)[i * 7:(i + 1) * 7].strip() for i in range(11)]
        try:
            p = float(f[0])
        except ValueError:
            continue
        if not 0 < p < 1200:
            continue
        rows.append({k: pd.to_numeric(v, errors="coerce") for k, v in zip(COLUMN_NAMES, f)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("ไม่มีระดับที่อ่านได้")
    df["wind_speed_ms"] = df["wind_speed_source"] * (0.514444 if unit == "knot" else 1.0 if unit == "m/s" else np.nan)
    df["wind_speed_source_unit"] = unit
    return df


def fetch_sounding(station, when, src="FM35", timeout=60, retries=3, session=None):
    """ดึงหนึ่ง launch จาก UWYO WSGI → (status, DataFrame|None, url, message)

    status: downloaded | no_data | error  (ไม่ throw — ให้ notebook ทำงานต่อได้แม้เครือข่ายขัดข้อง)
    """
    import requests
    s = session or requests.Session()
    params = {"datetime": pd.Timestamp(when).strftime("%Y-%m-%d %H:00:00"), "id": str(station),
              "src": src, "type": "TEXT:LIST"}
    msg, url = "", None
    for k in range(1, retries + 1):
        try:
            r = s.get(UWYO_ENDPOINT, params=params, timeout=timeout,
                      headers={"User-Agent": "Teach_stormTracks_KKN_May2024 educational notebook"})
            url = r.url
            if r.status_code in (400, 404):
                return "no_data", None, url, f"HTTP {r.status_code}"
            if r.status_code == 200:
                txt = to_plain_text(r.text)
                try:
                    return "downloaded", parse_wyoming_text(txt), url, "ok"
                except ValueError as e:
                    low = txt.lower()
                    if any(w in low for w in ("no data", "can't get", "not available", "no sounding")):
                        return "no_data", None, url, str(e)
                    return "error", None, url, f"HTTP 200 but parse failed: {e}"
            msg = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"[:200]
        time.sleep(2 * k)
    return "error", None, url, msg


def load_base_dataset(path_or_url=BASE_DATASET_URL, month=5, year=2024):
    """ไฟล์รวมของ repo Teaching_Sounding_SevereStorm_Korat (00 UTC, src=FM35) → {วันที่: DataFrame}"""
    d = pd.read_csv(path_or_url, parse_dates=["launch_datetime_utc"])
    d = d[(d["launch_datetime_utc"].dt.month == month) & (d["launch_datetime_utc"].dt.year == year)]
    return {pd.Timestamp(t).normalize(): g.drop(columns=["launch_datetime_utc"]).reset_index(drop=True)
            for t, g in d.groupby("launch_datetime_utc")}


# ======================================================================
# 2) QC เชิงฟิสิกส์
# ======================================================================
def qc_profile(df):
    """คืน (โปรไฟล์ที่ผ่าน QC, dict ข้อมูล QC)

    1. ตัดแถวที่ไม่มีความสูงหรืออุณหภูมิ เรียงตามความดันจากมากไปน้อย
    2. ความสูงต้องเพิ่มขึ้นเสมอ (ตัดแถวที่ซ้ำหรือย้อนกลับ)
    3. อุณหภูมิต้องอยู่ในช่วง −95 ถึง 50 °C
    4. **ตัดโปรไฟล์ตั้งแต่ระดับแรกที่** อุณหภูมิกระโดด > 15 K จากระดับก่อนหน้า หรือ > 0 °C เหนือ 10 km
       (เช่น 28 พ.ค. 2567 เหนือ 15.5 km อุณหภูมิเป็น +41 ถึง +49 °C จากเครื่องวัดขัดข้อง)
    """
    info = {"n_raw": len(df), "flags": []}
    d = df.dropna(subset=["pressure_hPa", "height_m", "temperature_C"]).sort_values("pressure_hPa", ascending=False)
    d = d[(d["temperature_C"] > -95) & (d["temperature_C"] < 50)]
    keep, hmax = [], -np.inf
    for i, h in zip(d.index, d["height_m"].values):
        if h > hmax:
            keep.append(i); hmax = h
    if len(keep) < len(d):
        info["flags"].append(f"non_monotonic:{len(d) - len(keep)}")
    d = d.loc[keep].reset_index(drop=True)
    cut = None
    T, H = d["temperature_C"].values, d["height_m"].values
    for k in range(1, len(d)):
        if abs(T[k] - T[k - 1]) > JUMP_MAX_K or (H[k] > WARM_ALOFT_M and T[k] > 0):
            cut = k
            info["flags"].append(f"cut_at_{H[k]:.0f}m(T={T[k]:.1f})")
            break
    if cut is not None:
        d = d.iloc[:cut]
    info.update(n_qc=len(d), sfc_m=float(d["height_m"].iloc[0]) if len(d) else np.nan,
                top_m=float(d["height_m"].iloc[-1]) if len(d) else np.nan,
                top_T=float(d["temperature_C"].iloc[-1]) if len(d) else np.nan)
    return d, info


def isotherm_height(z, t, temp_c):
    """ความสูงแรกที่อุณหภูมิลดลงผ่าน temp_c (ประมาณค่าเชิงเส้น)"""
    z, t = np.asarray(z, float), np.asarray(t, float)
    ok = np.isfinite(z) & np.isfinite(t)
    z, t = z[ok], t[ok]
    idx = np.where((t[:-1] >= temp_c) & (t[1:] < temp_c))[0]
    if not idx.size:
        return np.nan
    i = idx[0]
    return float(z[i] + (temp_c - t[i]) * (z[i + 1] - z[i]) / (t[i + 1] - t[i]))


def to_grid(d, z_grid=Z_GRID):
    """โปรไฟล์ → อุณหภูมิ จุดน้ำค้าง และลม (u, v) บนกริดความสูง (NaN นอกช่วงที่มีข้อมูล)"""
    out = {}
    h = d["height_m"].values.astype(float)
    inside = (z_grid >= h.min()) & (z_grid <= h.max())
    for col, name in (("temperature_C", "T"), ("dewpoint_C", "Td")):
        v = d[col].values.astype(float)
        ok = np.isfinite(v)
        out[name] = np.where(inside, np.interp(z_grid, h[ok], v[ok]) if ok.sum() > 1 else np.nan, np.nan)
    ws, wd = d["wind_speed_ms"].values.astype(float), d["wind_direction_deg"].values.astype(float)
    ok = np.isfinite(ws) & np.isfinite(wd)
    u, v = -ws * np.sin(np.radians(wd)), -ws * np.cos(np.radians(wd))
    for name, a in (("u", u), ("v", v)):
        out[name] = (np.where((z_grid >= h[ok].min()) & (z_grid <= h[ok].max()), np.interp(z_grid, h[ok], a[ok]), np.nan)
                     if ok.sum() > 1 else np.full_like(z_grid, np.nan))
    return out


def classify_quality(info, t_grid, z_grid=Z_GRID):
    """full | spliceable | unusable"""
    if not np.isfinite(info.get("top_m", np.nan)):
        return "unusable"
    if info["top_m"] >= FULL_TOP_M:
        return "full"
    z0 = isotherm_height(z_grid, t_grid, 0.0)
    if np.isfinite(z0) and info["top_m"] >= z0 + MIN_ABOVE_0C_M:
        return "spliceable"
    return "unusable"


# ======================================================================
# 3) โปรไฟล์รายวันสำหรับ HID
# ======================================================================
def splice(t_own, t_comp, z_grid=Z_GRID, blend_m=SPLICE_BLEND_M):
    """ข้อมูลจริงถึงยอด แล้วต่อด้วย composite + ค่าเบี่ยงเบนที่ยอด ซึ่งลดลงเป็นศูนย์ในช่วง blend_m

    T(z) = T_comp(z) + δ · max(0, 1 − (z − z_top)/blend_m),   δ = T_own(z_top) − T_comp(z_top)
    """
    ok = np.isfinite(t_own)
    if not ok.any():
        return t_comp.copy(), np.nan
    k = np.where(ok)[0].max()
    zt, delta = z_grid[k], t_own[k] - t_comp[k]
    out = t_own.copy()
    above = z_grid > zt
    out[above] = t_comp[above] + delta * np.clip(1 - (z_grid[above] - zt) / blend_m, 0, 1)
    below = ~ok & (z_grid < zt)                       # ใต้ระดับพื้นของสถานี (เช่น 0–188 m) → ใช้ค่าล่างสุด
    first = np.where(ok)[0].min()
    out[below] = t_own[first]
    return out, float(zt)


def fill_below_surface(a):
    ok = np.isfinite(a)
    if ok.any():
        a = a.copy(); a[: np.where(ok)[0].min()] = a[ok][0]
    return a


def build_daily_profiles(days, korat, ubon=None, z_grid=Z_GRID):
    """สร้างโปรไฟล์อุณหภูมิสำหรับ HID ทุกวัน

    korat, ubon : {วันที่: DataFrame ดิบ}
    คืน (xarray.Dataset[day, height], ตารางสรุปรายวัน, ตาราง QC ของทุกโปรไฟล์)
    """
    import xarray as xr
    days = [pd.Timestamp(d).normalize() for d in days]
    qc_rows, grids = [], {}
    for stn, src in (("48431", korat), ("48407", ubon or {})):
        for d, raw in src.items():
            q, info = qc_profile(raw)
            g = to_grid(q, z_grid) if len(q) > 1 else {k: np.full_like(z_grid, np.nan) for k in ("T", "Td", "u", "v")}
            info["quality"] = classify_quality(info, g["T"], z_grid)
            info.update(station=stn, day=pd.Timestamp(d).normalize(),
                        z0_m=isotherm_height(z_grid, g["T"], 0.0), source_label=raw.attrs.get("source", "base"))
            qc_rows.append(info)
            grids[(stn, info["day"])] = (g, info)
    qc = pd.DataFrame(qc_rows)
    full = [grids[k][0]["T"] for k, (g, i) in grids.items() if i["quality"] == "full" and k[0] == "48431"]
    if not full:
        full = [g["T"] for g, i in grids.values() if i["quality"] == "full"]
    comp_T = np.nanmedian(np.stack(full), axis=0)
    comp_T = fill_below_surface(comp_T)
    comp_Td = fill_below_surface(np.nanmedian(np.stack([g["Td"] for g, i in grids.values() if i["quality"] == "full"]), axis=0))

    T = np.full((len(days), len(z_grid)), np.nan)
    Td = np.full_like(T, np.nan)
    src_lab, splice_z = [""] * len(days), [np.nan] * len(days)
    # รอบแรก: โคราช / อุบล ของวันนั้น
    for i, d in enumerate(days):
        for stn, tag in (("48431", ""), ("48407", "ubon_")):
            if (stn, d) not in grids:
                continue
            g, info = grids[(stn, d)]
            if info["quality"] == "full":
                T[i] = fill_below_surface(np.where(np.isfinite(g["T"]), g["T"], comp_T))
                Td[i] = fill_below_surface(np.where(np.isfinite(g["Td"]), g["Td"], comp_Td))
                src_lab[i] = f"{tag}own"
                if info["top_m"] < z_grid[-1]:          # ต่อเหนือยอดเล็กน้อยให้ครบ 20 km
                    T[i], splice_z[i] = splice(g["T"], comp_T, z_grid)
                break
            if info["quality"] == "spliceable":
                T[i], splice_z[i] = splice(g["T"], comp_T, z_grid)
                Td[i] = np.where(np.isfinite(g["Td"]), fill_below_surface(g["Td"]), comp_Td)
                src_lab[i] = f"{tag}spliced"
                break
    # รอบสอง: วันที่ยังไม่มี → ค่าเฉลี่ยวันข้างเคียง (จากโปรไฟล์โคราชของรอบแรกเท่านั้น)
    base = {d: i for i, d in enumerate(days) if src_lab[i] in ("own", "spliced")}
    for i, d in enumerate(days):
        if src_lab[i]:
            continue
        prev = next((d - pd.Timedelta(days=k) for k in range(1, MAX_NEIGHBOR_DAYS + 1)
                     if d - pd.Timedelta(days=k) in base), None)
        nxt = next((d + pd.Timedelta(days=k) for k in range(1, MAX_NEIGHBOR_DAYS + 1)
                    if d + pd.Timedelta(days=k) in base), None)
        use = [x for x in (prev, nxt) if x is not None]
        if use:
            T[i] = np.mean([T[base[x]] for x in use], axis=0)
            Td[i] = np.mean([Td[base[x]] for x in use], axis=0)
            src_lab[i] = "neighbor_mean(" + ",".join(f"{x:%d}" for x in use) + ")"
        else:
            T[i], Td[i], src_lab[i] = comp_T, comp_Td, "composite"
    summ = pd.DataFrame({
        "day": days, "hid_profile_source": src_lab, "splice_height_m": splice_z,
        "z0C_m": [isotherm_height(z_grid, t, 0.0) for t in T],
        "zm10C_m": [isotherm_height(z_grid, t, -10.0) for t in T],
        "zm20C_m": [isotherm_height(z_grid, t, -20.0) for t in T],
        "zm40C_m": [isotherm_height(z_grid, t, -40.0) for t in T],
    })
    k_q = qc[qc["station"] == "48431"].set_index("day")
    summ["korat_quality"] = summ["day"].map(k_q["quality"]).fillna("missing")
    summ["korat_top_m"] = summ["day"].map(k_q["top_m"])
    summ["korat_qc_flags"] = summ["day"].map(k_q["flags"].apply(lambda f: ";".join(f)))
    ds = xr.Dataset(
        {"temperature": (("day", "height"), T.astype("float32"), {"units": "degC"}),
         "dewpoint": (("day", "height"), Td.astype("float32"), {"units": "degC"}),
         "composite_temperature": (("height",), comp_T.astype("float32"), {"units": "degC"}),
         "source": (("day",), np.array(src_lab, dtype=object))},
        coords={"day": pd.to_datetime(days), "height": ("height", z_grid.astype("float32"), {"units": "m MSL"})},
        attrs={"title": "Daily temperature profiles for HID / Z-PHI (KKN, May 2024)",
               "stations": "48431 Nakhon Ratchasima (primary), 48407 Ubon Ratchathani (fallback)",
               "source": "University of Wyoming upper-air archive", "full_top_m": FULL_TOP_M,
               "splice_blend_m": SPLICE_BLEND_M, "max_neighbor_days": MAX_NEIGHBOR_DAYS,
               "rules": "own > spliced > ubon_own > ubon_spliced > neighbor_mean > composite"})
    return ds, summ, qc


def profile_dataframe(ds, day):
    """โปรไฟล์หนึ่งวันในรูปแบบที่ polar_tools ใช้ (คอลัมน์ height [m MSL], temperature [°C])"""
    p = ds.sel(day=pd.Timestamp(day).normalize())
    return pd.DataFrame({"height": p["height"].values.astype(float), "temperature": p["temperature"].values.astype(float),
                         "dewpoint": p["dewpoint"].values.astype(float)})


# ======================================================================
# 4) ดัชนีสภาพแวดล้อม (เฉพาะ sounding จริงที่ครบ)
# ======================================================================
def environment_indices(df):
    """CAPE/CIN (SB, ML 100 hPa, MU), DCAPE, LCL, PW, lapse rate 700–500, shear 0–1 และ 0–6 km

    ใช้เฉพาะโปรไฟล์คุณภาพ full (ยอดสูงกว่า 12 km) — CAPE ของโปรไฟล์สั้นจะต่ำกว่าความจริง
    """
    import metpy.calc as mc
    from metpy.units import units
    d = df.dropna(subset=["pressure_hPa", "height_m", "temperature_C", "dewpoint_C"]).sort_values("pressure_hPa", ascending=False)
    d = d.drop_duplicates("pressure_hPa")
    p = d["pressure_hPa"].values * units.hPa
    T = d["temperature_C"].values * units.degC
    Td = d["dewpoint_C"].values * units.degC
    out = {}
    try:
        prof = mc.parcel_profile(p, T[0], Td[0]).to("degC")
        cape, cin = mc.cape_cin(p, T, Td, prof)
        out.update(SBCAPE_Jkg=float(cape.m), SBCIN_Jkg=float(cin.m))
        mlc, mlcin = mc.mixed_layer_cape_cin(p, T, Td, depth=100 * units.hPa)
        muc, mucin = mc.most_unstable_cape_cin(p, T, Td)
        out.update(MLCAPE_Jkg=float(mlc.m), MLCIN_Jkg=float(mlcin.m), MUCAPE_Jkg=float(muc.m))
        lcl_p, _ = mc.lcl(p[0], T[0], Td[0])
        out["LCL_hPa"] = float(lcl_p.m)
        out["PW_mm"] = float(mc.precipitable_water(p, Td).to("mm").m)
    except Exception as e:  # noqa: BLE001
        out["thermo_error"] = str(e)[:80]
    try:
        out["DCAPE_Jkg"] = float(mc.downdraft_cape(p, T, Td)[0].m)
    except Exception:
        out["DCAPE_Jkg"] = np.nan
    z, t = d["height_m"].values, d["temperature_C"].values
    pr = d["pressure_hPa"].values
    if pr.min() <= 500:
        z7, z5 = np.interp([700, 500], pr[::-1], z[::-1])
        t7, t5 = np.interp([700, 500], pr[::-1], t[::-1])
        out["lapse_700_500_Ckm"] = float((t7 - t5) / ((z5 - z7) / 1000))
    w = df.dropna(subset=["height_m", "wind_speed_ms", "wind_direction_deg"]).sort_values("height_m")
    zz = w["height_m"].values - w["height_m"].values[0]
    u = -w["wind_speed_ms"].values * np.sin(np.radians(w["wind_direction_deg"].values))
    v = -w["wind_speed_ms"].values * np.cos(np.radians(w["wind_direction_deg"].values))
    for depth in (1000, 6000):
        if len(zz) > 2 and zz.max() >= depth:
            out[f"shear_0_{depth // 1000}km_ms"] = float(np.hypot(np.interp(depth, zz, u) - u[0], np.interp(depth, zz, v) - v[0]))
    m = (zz >= 0) & (zz <= 6000)
    if m.sum() >= 3:
        out["mean_wind_0_6km_ms"] = float(np.hypot(u[m].mean(), v[m].mean()))
        out["mean_wind_0_6km_dir_to_deg"] = float(np.degrees(np.arctan2(u[m].mean(), v[m].mean())) % 360)
    return out
