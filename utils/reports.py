"""อ่าน ทำความสะอาด และจัดระดับรายงานภัยพิบัติจากไฟล์ Excel"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import repo_path

# ชื่อคอลัมน์ภาษาอังกฤษสำหรับใช้ในโค้ด (คอลัมน์ต้นฉบับภาษาไทยยังอยู่ครบ)
COLS = {
    "ลำดับ": "rec_id", "วัน": "day", "เดือน": "month", "ปี": "year_be",
    "เวลาเริ่มต้น": "t_start_raw", "เวลาสิ้นสุด": "t_end_raw",
    "จังหวัด": "province", "อำเภอ": "district",
    "ประเภทภัยพิบัติ": "hazard_type", "ลม": "wind", "ฝน": "rain", "ลูกเห็บ": "hail",
    "ขนาดลูกเห็บ": "hail_size", "ฟ้าผ่า": "lightning", "น้ำท่วม": "flood",
    "จำนวนตำบล": "n_tambon", "จำนวนหมู่บ้าน": "n_village",
    "บ้านเสียหาย \n(หลังคาเรือน)": "houses", "คนบาดเจ็บ": "injured", "คนเสียชีวิต": "dead",
    "สัตว์เลี้ยงตาย": "livestock_dead", "ทรัพย์สินเสียหายอื่นๆ": "other_property",
    "หมายเหตุ": "note",
}
NUMERIC = ["n_tambon", "n_village", "houses", "injured", "dead", "livestock_dead", "other_property"]
YN = ["wind", "rain", "hail", "lightning", "flood"]


def read_reports(s) -> pd.DataFrame:
    df = pd.read_excel(repo_path(s["reports"]["excel"]))
    return df.rename(columns=COLS)


def parse_clock(v):
    """แปลงเวลาแบบ ชั่วโมง.นาที เช่น 17.3 → (17, 30), 13.15 → (13, 15), 16.4 → (16, 40)
    คืน None ถ้าไม่ใช่ตัวเลข"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (int, np.integer, float, np.floating)):
        txt = repr(float(v))
    else:
        txt = str(v).strip()
        if not re.fullmatch(r"\d{1,2}([.:]\d{1,2})?", txt):
            return None
        txt = txt.replace(":", ".")
    hh, _, mm = txt.partition(".")
    mm = (mm + "00")[:2] if mm else "00"
    h, m = int(hh), int(mm)
    if not (0 <= h <= 24 and 0 <= m < 60):
        return None
    return h, m


def time_window_local(t_start, t_end, s):
    """หน้าต่างเวลาท้องถิ่นของรายงาน (ชั่วโมงทศนิยม, อาจเกิน 24 = ข้ามวัน) และชนิดของเวลา

    exact   : มีเวลาระบุ → [t − pad0, max(t, t_end) + pad1]
    period  : ระบุเป็นคำ (ช่วงเย็น ฯลฯ) → ตามตาราง period_windows_local
    unknown : ไม่ระบุ → ทั้งวัน 0–24 น.
    """
    pad0, pad1 = (p / 60 for p in s["reports"]["exact_time_pad_min"])
    c0 = parse_clock(t_start)
    if c0 is not None:
        h0 = c0[0] + c0[1] / 60
        c1 = parse_clock(t_end)
        h1 = c1[0] + c1[1] / 60 if c1 is not None else h0
        if h1 < h0:  # สิ้นสุดหลังเที่ยงคืน
            h1 += 24
        return h0 - pad0, h1 + pad1, "exact", h0
    key = str(t_start).strip()
    win = s["reports"]["period_windows_local"].get(key)
    if win:
        return float(win[0]), float(win[1]), "period", np.nan
    return 0.0, 24.0, "unknown", np.nan


def clean_reports(raw: pd.DataFrame, s, month=None, year_ce=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ทำความสะอาดรายงาน คืน (ตารางสะอาด, บันทึกการแก้ไข)"""
    month = month or s["project"]["month"]
    year_ce = year_ce or s["project"]["year_ce"]
    log = []
    df = raw.copy()
    df["year_ce"] = df["year_be"] - 543
    df = df[(df["year_ce"] == year_ce) & (df["month"] == month)].copy()
    log.append(("เลือกเดือน", f"{year_ce}-{month:02d}", len(df)))

    # 1) ช่องว่างหัวท้ายและชื่อที่สะกดต่าง
    for c in ["province", "district"]:
        df[c + "_raw"] = df[c]
        df[c] = df[c].astype(str).str.strip()
    df["province"] = df["province"].replace(s["reports"]["province_alias"])
    alias = s["reports"]["district_alias"]
    key = df["province"] + "|" + df["district"]
    hit = key.isin(alias)
    df.loc[hit, "district"] = key[hit].map(alias)
    changed = (df["province"] != df["province_raw"]) | (df["district"] != df["district_raw"])
    for _, r in df[changed].iterrows():
        log.append(("แก้ชื่อ", f"{r.province_raw}/{r.district_raw} → {r.province}/{r.district}", 1))

    # 2) รหัสข้อมูลขาด 9999 → NaN และแปลงเป็นตัวเลข
    miss = s["reports"]["missing_code"]
    for c in NUMERIC:
        v = pd.to_numeric(df[c], errors="coerce")
        n = int((v == miss).sum())
        if n:
            log.append(("9999 → NaN", c, n))
        df[c] = v.where(v != miss)
    for c in YN:
        df[c] = df[c].astype(str).str.strip().str.lower().eq("y")

    # 3) แถวซ้ำ (วัน อำเภอ เวลา และตัวเลขความเสียหายเหมือนกัน)
    sig = ["day", "province", "district", "t_start_raw", "n_tambon", "n_village", "houses", "injured", "dead"]
    dup = df.duplicated(subset=sig, keep="first")
    for _, r in df[dup].iterrows():
        log.append(("ตัดแถวซ้ำ", f"ลำดับ {r.rec_id}: {r.day}/{r.month} {r.province}/{r.district}", 1))
    df = df[~dup].copy()

    # 4) เวลา: ท้องถิ่น → UTC
    tw = df.apply(lambda r: time_window_local(r.t_start_raw, r.t_end_raw, s), axis=1, result_type="expand")
    df[["win_start_h", "win_end_h", "time_kind", "t_report_h"]] = tw
    df["date"] = pd.to_datetime(dict(year=df.year_ce, month=df.month, day=df.day))
    tz = s["project"]["tz_local"]
    loc0 = (df["date"] + pd.to_timedelta(df["win_start_h"], unit="h")).dt.tz_localize(tz)
    loc1 = (df["date"] + pd.to_timedelta(df["win_end_h"], unit="h")).dt.tz_localize(tz)
    df["win_start_utc"] = loc0.dt.tz_convert("UTC").dt.tz_localize(None)
    df["win_end_utc"] = loc1.dt.tz_convert("UTC").dt.tz_localize(None)
    df["t_report_local"] = df["date"] + pd.to_timedelta(df["t_report_h"], unit="h")
    df["t_report_utc"] = df["t_report_local"] - pd.Timedelta(hours=7)
    df["casualties"] = df[["injured", "dead"]].sum(axis=1, min_count=1)
    df = df.sort_values(["date", "province", "district"]).reset_index(drop=True)
    seq = df.groupby("date").cumcount() + 1
    df.insert(0, "report_id", [f"R{d:%m%d}-{i:02d}" for d, i in zip(df["date"], seq)])
    return df, pd.DataFrame(log, columns=["ขั้นตอน", "รายละเอียด", "จำนวน"])


def severity_index(df, weights=None):
    """ดัชนีความรุนแรงแบบ percentile-rank composite

    S_i = Σ_k w_k · P_k(x_ik) / Σ_k w_k   (เฉพาะตัวชี้วัด k ที่มีค่า)
    P_k = ลำดับเปอร์เซ็นไทล์ (0–1) ของตัวชี้วัด k เทียบกับทุกรายงาน (ค่าเท่ากันได้อันดับเฉลี่ย)
    ข้อดี: ไม่ขึ้นกับหน่วย และทนต่อค่าผิดปกติ (บ้านเสียหาย 1,400 หลังไม่กลบตัวชี้วัดอื่น)
    """
    weights = weights or {"houses": 1.0, "n_tambon": 1.0, "n_village": 1.0, "casualties": 1.0, "hail": 1.0}
    P = pd.DataFrame(index=df.index)
    for k in weights:
        x = df[k].astype(float)
        P[k] = x.rank(pct=True, method="average")
    W = pd.Series(weights)
    num = (P * W).sum(axis=1, min_count=1)
    den = P.notna().mul(W).sum(axis=1)
    return num / den, P


def tier_labels(score, labels=("น้อย", "กลาง", "มาก")):
    """แบ่ง 3 ระดับด้วย tertile"""
    return pd.qcut(score.rank(method="first"), q=len(labels), labels=list(labels))


def assign_phase(dates, s):
    out = pd.Series(index=dates.index, dtype=object)
    for p in s["phases"]:
        m = (dates >= p["start"]) & (dates <= p["end"])
        out[m] = p["key"]
    return out
