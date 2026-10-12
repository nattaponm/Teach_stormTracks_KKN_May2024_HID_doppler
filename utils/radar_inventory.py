"""สำรวจไฟล์เรดาร์ UF (ใช้ใน prep/P0 เท่านั้น — นิสิตไม่ต้องใช้)

ขั้นที่ 1 (เร็ว): อ่านแค่ชื่อและขนาดไฟล์ → เวลา UTC, ช่อง 15 นาที, ไฟล์ซ้ำ, ไฟล์เล็กผิดปกติ
ขั้นที่ 2 (ช้ากว่า): เปิดไฟล์ด้วย Py-ART อ่าน metadata → พิกัด มุมยก Nyquist ไฟล์เสีย
ขั้นที่ 3: สรุปรายวัน ช่องว่าง ช่วงต่อเนื่อง (segment) และเกณฑ์วันที่ใช้ได้
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- ขั้นที่ 1

def list_files(raw_root, s, month_prefix="2405"):
    """อ่านรายชื่อไฟล์ในโฟลเดอร์รายวัน raw_root/YYMMDD/ (ไม่เปิดไฟล์)"""
    rx = re.compile(s["radar_data"]["file_regex"])
    rows = []
    raw_root = Path(raw_root)
    for day_dir in sorted(p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith(month_prefix)):
        with os.scandir(day_dir) as it:
            for e in it:
                if not e.is_file():
                    continue
                m = rx.match(e.name)
                rows.append({
                    "folder": day_dir.name, "file": e.name, "path": str(Path(e.path)),
                    "size_mb": e.stat().st_size / 2**20,
                    "site": m["site"] if m else None, "prod": m["prod"] if m else None,
                    "time_utc": pd.to_datetime(m["ts"], format="%Y%m%d%H%M") if m else pd.NaT,
                    "name_ok": bool(m),
                })
    return pd.DataFrame(rows)


def assign_slots(df, s):
    """จับไฟล์เข้าช่องเวลามาตรฐาน 15 นาที แล้วติดสถานะ

    status: ok | offset (เหลื่อมเกินเกณฑ์) | small (ไฟล์เล็กผิดปกติ) | duplicate | bad_name | other_product
    """
    rd = s["radar_data"]
    step = f"{rd['nominal_step_min']}min"
    df = df.copy()
    df["status"] = "ok"
    df.loc[~df["name_ok"], "status"] = "bad_name"
    df.loc[df["name_ok"] & (df["prod"] != rd["product_code"]), "status"] = "other_product"
    df["slot_utc"] = df["time_utc"].dt.round(step)
    df["offset_min"] = (df["time_utc"] - df["slot_utc"]).dt.total_seconds() / 60
    good = df["status"] == "ok"
    df.loc[good & (df["offset_min"].abs() > rd["slot_tolerance_min"]), "status"] = "offset"
    # ไฟล์เล็กผิดปกติ เทียบกับมัธยฐานของวันเดียวกัน
    med = df[good].groupby("folder")["size_mb"].transform("median")
    small = good & (df["size_mb"] < rd["small_file_ratio"] * med.reindex(df.index))
    df.loc[small, "status"] = "small"
    # หลายไฟล์ในช่องเดียว: เก็บไฟล์ที่ใกล้เวลามาตรฐานที่สุด
    cand = df["status"].isin(["ok", "offset"])
    df["_abs"] = df["offset_min"].abs()
    best = df[cand].sort_values(["slot_utc", "_abs", "size_mb"], ascending=[True, True, False]) \
                   .drop_duplicates("slot_utc").index
    df.loc[cand & ~df.index.isin(best), "status"] = "duplicate"
    return df.drop(columns="_abs").sort_values("time_utc").reset_index(drop=True)


def slot_table(df, s, year=2024, month=5):
    """ตารางช่องเวลาครบทั้งเดือน (รวมช่องที่ไม่มีไฟล์ = missing)"""
    step = f"{s['radar_data']['nominal_step_min']}min"
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1) - pd.Timedelta(step)
    slots = pd.DataFrame({"slot_utc": pd.date_range(start, end, freq=step)})
    use = df[df["status"].isin(["ok", "offset", "small", "corrupt"])]
    pri = {"ok": 0, "offset": 1, "small": 2, "corrupt": 3}
    use = use.assign(_p=use["status"].map(pri)).sort_values(["slot_utc", "_p"]).drop_duplicates("slot_utc")
    out = slots.merge(use.drop(columns="_p"), on="slot_utc", how="left")
    out["status"] = out["status"].fillna("missing")
    out["date"] = out["slot_utc"].dt.normalize()
    return out


# ----------------------------------------------------------------- ขั้นที่ 2

def read_metadata(path):
    """เปิดไฟล์ UF (.bz2 ได้โดยตรง) แล้วคืน metadata ที่จำเป็นต่อการประมวลผล"""
    import warnings
    warnings.filterwarnings("ignore")
    import pyart

    try:
        radar = pyart.io.read_uf(path)
    except Exception as e:
        return {"path": path, "read_ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
    try:
        ny = radar.instrument_parameters.get("nyquist_velocity", {}).get("data") if radar.instrument_parameters else None
        ny_sweep = [float(np.nanmedian(ny[radar.get_slice(i)])) for i in range(radar.nsweeps)] if ny is not None else []
        rng = radar.range["data"]
        return {
            "path": path, "read_ok": True, "error": "",
            "lat": float(radar.latitude["data"][0]), "lon": float(radar.longitude["data"][0]),
            "alt_m": float(radar.altitude["data"][0]),
            "nsweeps": int(radar.nsweeps), "nrays": int(radar.nrays), "ngates": int(radar.ngates),
            "gate_m": float(np.median(np.diff(rng))) if rng.size > 1 else np.nan,
            "max_range_km": float(rng[-1] / 1000),
            "elevs": ";".join(f"{a:.1f}" for a in radar.fixed_angle["data"]),
            "nyquist": ";".join(f"{v:.2f}" for v in ny_sweep),
            "fields": ";".join(sorted(radar.fields)),
            "scan_start": str(pyart.util.datetime_from_radar(radar)),
        }
    except Exception as e:
        return {"path": path, "read_ok": False, "error": f"meta {type(e).__name__}: {e}"[:200]}


def read_metadata_many(paths, cache_csv=None, workers=None, progress_every=100):
    """อ่าน metadata แบบขนาน พร้อม cache (รันซ้ำได้ อ่านเฉพาะไฟล์ใหม่)"""
    done = pd.DataFrame()
    if cache_csv and Path(cache_csv).exists():
        done = pd.read_csv(cache_csv)
        paths = [p for p in paths if p not in set(done["path"])]
    print(f"ต้องอ่าน {len(paths)} ไฟล์ (มีใน cache แล้ว {len(done)})")
    rows = []
    workers = workers or max(1, (os.cpu_count() or 2))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(read_metadata, p) for p in paths]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % progress_every == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}")
                if cache_csv:
                    pd.concat([done, pd.DataFrame(rows)]).to_csv(cache_csv, index=False)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    if cache_csv:
        out.to_csv(cache_csv, index=False)
    return out


def scan_strategies(meta_files):
    """รวมกลุ่มรูปแบบการสแกน (มุมยก + Nyquist + range) ที่ใช้ และช่วงเวลาที่ใช้"""
    m = meta_files[meta_files["read_ok"] == True].copy()  # noqa: E712
    m["strategy"] = m["elevs"].astype(str) + " | Vny " + m["nyquist"].astype(str) + " | " + m["max_range_km"].round(0).astype(str) + " km"
    g = m.groupby("strategy").agg(n_files=("path", "size"), first=("slot_utc", "min"), last=("slot_utc", "max"),
                                  nsweeps=("nsweeps", "first"), gate_m=("gate_m", "first"))
    return g.sort_values("n_files", ascending=False)


# ----------------------------------------------------------------- ขั้นที่ 3

def segments(slots_one_day, s):
    """ช่วงต่อเนื่องของภาพที่ใช้ติดตามพายุได้: ช่องว่างไม่เกิน max_bridge_gap_min"""
    step = s["radar_data"]["nominal_step_min"]
    max_gap = s["radar_data"]["max_bridge_gap_min"]
    t = slots_one_day.loc[slots_one_day["status"].isin(["ok", "offset"]), "slot_utc"].sort_values()
    if t.empty:
        return pd.DataFrame(columns=["seg_id", "start_utc", "end_utc", "n_scans", "duration_min"])
    gap = t.diff().dt.total_seconds().div(60).fillna(step)
    seg_id = (gap > max(step, max_gap)).cumsum()
    g = t.groupby(seg_id.values)
    out = pd.DataFrame({"start_utc": g.min(), "end_utc": g.max(), "n_scans": g.size()})
    out["duration_min"] = (out["end_utc"] - out["start_utc"]).dt.total_seconds() / 60 + step
    out.index.name = "seg_id"
    return out.reset_index()


def daily_summary(slots, s):
    rd = s["radar_data"]
    h0, h1 = rd["core_window_utc"]
    rows = []
    for d, g in slots.groupby("date"):
        good = g["status"].isin(["ok", "offset"])
        core = g["slot_utc"].dt.hour.between(h0, h1 - 1)
        t_good = g.loc[good, "slot_utc"].sort_values()
        gaps = t_good.diff().dt.total_seconds().div(60)
        seg = segments(g, s)
        rows.append({
            "date": d,
            "n_slots": len(g),
            "n_ok": int((g["status"] == "ok").sum()),
            "n_offset": int((g["status"] == "offset").sum()),
            "n_small": int((g["status"] == "small").sum()),
            "n_corrupt": int((g["status"] == "corrupt").sum()),
            "n_missing": int((g["status"] == "missing").sum()),
            "coverage_day": good.mean(),
            "coverage_core": good[core].mean(),
            "longest_gap_min": float(gaps.max()) if len(gaps.dropna()) else np.nan,
            "n_segments": len(seg),
            "longest_segment_h": float(seg["duration_min"].max() / 60) if len(seg) else 0.0,
        })
    out = pd.DataFrame(rows)
    out["usable"] = out["coverage_core"] >= rd["usable_min_coverage"]
    return out
