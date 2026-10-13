"""P3: ติดตามพายุทั้งเดือนแบบ "เหตุการณ์พายุ" (episode) — ห่อฟังก์ชันของ repo3 (track_tools)

ทำไมไม่ติดตามทีละวัน
--------------------
ไฟล์ CAPPI แยกรายวัน (00:00–23:45 UTC) ถ้าติดตามทีละไฟล์ พายุที่ยังอยู่ตอนเที่ยงคืน UTC (07:00 น.
เวลาไทย) จะถูกตัดเป็นสองเส้น → อายุ/ระยะทางสั้นกว่าจริง จำนวนเส้นทางเกินจริง และเกิด "จุดกำเนิดปลอม"
ตอน 00 UTC ปัญหานี้รุนแรงในช่วงต้นฤดูฝนที่ฝนตกต่อเนื่องข้ามคืน

แนวคิด
------
1. ต่อภาพทั้งเดือนเป็นอนุกรมเวลาเดียว (ทุก 15 นาที)
2. ตัดเป็น **เหตุการณ์ (episode)** เฉพาะที่ **ช่วงสงบ** (พื้นที่ ≥ 35 dBZ < 20 km² ติดกัน ≥ 1 ชม.)
   หรือ **ช่วงข้อมูลขาด** — ตรงช่วงสงบไม่มีพายุที่ติดตามได้ จึงไม่มีเส้นทางถูกตัด
3. ติดตามทีละ episode (T-DaTing พารามิเตอร์เดิมของ repo3) ภาพ 5/10 นาทีสร้างต่อเนื่องข้ามเที่ยงคืนได้
4. เขียน label cube กลับเป็น **ไฟล์รายวัน** เหมือนเดิม (บทเรียนนิสิตใช้ ``day_file`` ได้ตามปกติ)

รหัสเส้นทาง
-----------
``track_id`` = episode × 10000 + ลำดับในเหตุการณ์ (ไม่ซ้ำทั้งเดือน, ค่าใน label = track_id + 1)
``uid``      = ``E###-####`` เช่น ``E012-0007``
``day``      = วันที่ UTC ของจุดเริ่ม (ตารางเส้นทาง) / ของเวลานั้น (ตารางเซลล์), ``day_local`` = วันที่เวลาไทย

censored
--------
``censored_start`` / ``censored_end`` = เส้นทางเริ่ม/จบที่ขอบ episode **ที่ติดกับข้อมูลขาดหรือขอบเดือน**
(อายุจริงอาจยาวกว่าที่วัดได้) ส่วน ``at_episode_edge`` = แตะขอบ episode ไม่ว่าด้วยเหตุใด (นิยามแบบเข้มงวด)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import track_tools as tt

MAIN_TAGS = {"15min": (15, None), "10min_linZ": (10, "linZ"), "05min_linZ": (5, "linZ")}
ID_BLOCK = 10000                       # track_id = episode * ID_BLOCK + k
LOCAL_OFFSET = pd.Timedelta(hours=7)   # เวลาประเทศไทย = UTC + 7


# ======================================================================
# 1) อนุกรมเวลาทั้งเดือน → episode
# ======================================================================
def segments_from_available(available, min_len=1):
    """ช่วงต่อเนื่องของช่องเวลาที่มีข้อมูล → list of (i0, i1) แบบรวมปลาย"""
    a = np.asarray(available, bool)
    out, i = [], 0
    while i < len(a):
        if a[i]:
            j = i
            while j + 1 < len(a) and a[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def scan_day(cappi_path, polar_path=None, thr_dbz=35.0):
    """อ่าน CAPPI 1 วัน → เวลา, ความพร้อม, พื้นที่ ≥ thr ต่อภาพ, ความถี่ราย pixel และผลรวมลูกเห็บ (ถ้ามี P2)"""
    from . import data_io
    Z, mt = data_io.load_pcappi(str(cappi_path))
    av = np.asarray(mt["available"], bool)
    ge35 = np.nan_to_num(Z, nan=-99) >= thr_dbz
    area = ge35.sum(axis=(1, 2)).astype(float)
    area[~av] = np.nan
    out = {"times": pd.to_datetime(list(mt["timestamps"])), "available": av, "area": area,
           "n_obs": int(av.sum()), "freq35": ge35[av].sum(0).astype(np.int32),
           "freq45": (np.nan_to_num(Z[av], nan=-99) >= 45).sum(0).astype(np.int32)}
    hail = np.zeros(Z.shape[1:], np.int32)
    if polar_path is not None and Path(polar_path).exists():
        P, _ = tt.load_polar_grid(str(polar_path))
        hail = (P["hail_col"] > 0).sum(0).astype(np.int32)
    out["hail"] = hail
    return out


def find_episodes(times, available, area, thr_area=20.0, quiet_frames=4, pad=2, min_len=3):
    """แบ่งอนุกรมเวลาทั้งเดือนเป็นเหตุการณ์พายุ

    times, available, area : อนุกรม 15 นาทีต่อเนื่อง (area = จำนวน pixel ≥ 35 dBZ, NaN เมื่อไม่มีข้อมูล)
    quiet_frames : จำนวนภาพ "สงบ" (area < thr_area) ติดกันอย่างน้อยเท่านี้จึงตัดเป็นคนละเหตุการณ์
    pad          : เผื่อภาพก่อน/หลังช่วงที่มีพายุ (ไม่เกินขอบข้อมูลต่อเนื่อง)

    คืน DataFrame: episode, i0, i1, start, end, n_frames, start_edge, end_edge (quiet / gap / month), n_days
    """
    times = pd.to_datetime(list(times))
    av = np.asarray(available, bool)
    act = av & (np.nan_to_num(np.asarray(area, float), nan=0) >= thr_area)
    rows = []
    for s, e in segments_from_available(av, 1):
        idx = np.flatnonzero(act[s:e + 1]) + s
        if idx.size == 0:
            continue
        cut = np.flatnonzero(np.diff(idx) - 1 >= quiet_frames)          # ช่องว่างสงบยาวพอ → ตัด
        for c0, c1 in zip(np.r_[idx[0], idx[cut + 1]], np.r_[idx[cut], idx[-1]]):
            i0, i1 = max(s, c0 - pad), min(e, c1 + pad)
            if i1 - i0 + 1 < min_len:
                continue
            se = "month" if i0 == 0 else ("gap" if i0 == s else "quiet")
            ee = "month" if i1 == len(av) - 1 else ("gap" if i1 == e else "quiet")
            rows.append({"i0": i0, "i1": i1, "start": times[i0], "end": times[i1], "n_frames": i1 - i0 + 1,
                         "start_edge": se, "end_edge": ee})
    E = pd.DataFrame(rows, columns=["i0", "i1", "start", "end", "n_frames", "start_edge", "end_edge"])
    E.insert(0, "episode", np.arange(len(E)))
    E["hours"] = (E["end"] - E["start"]).dt.total_seconds() / 3600 + 0.25
    E["n_days"] = (E["end"].dt.normalize() - E["start"].dt.normalize()).dt.days + 1
    E["crosses_00utc"] = E["n_days"] > 1
    return E


def month_series(scans):
    """รวมผล scan_day หลายวัน → times, available, area (เรียงเวลา)"""
    keys = sorted(scans)
    times = pd.DatetimeIndex(np.concatenate([scans[k]["times"].values for k in keys]))
    av = np.concatenate([scans[k]["available"] for k in keys])
    area = np.concatenate([scans[k]["area"] for k in keys])
    return times, av, area


# ======================================================================
# 2) โหลดข้อมูลช่วงเวลาที่คร่อมหลายไฟล์
# ======================================================================
def _days_between(t0, t1):
    return [d.strftime("%Y%m%d") for d in pd.date_range(pd.Timestamp(t0).normalize(), pd.Timestamp(t1).normalize())]


def load_cappi_range(cappi_dir, t0, t1):
    """CAPPI ช่วง [t0, t1] (อาจคร่อมหลายวัน) → Z[t, y, x], meta (timestamps/available ของช่วงนั้น)"""
    from . import data_io
    Zs, ts, av, meta = [], [], [], None
    for d in _days_between(t0, t1):
        Z, m = data_io.load_pcappi(str(Path(cappi_dir) / f"KKN_{d}_cappi_15min.nc"))
        t = pd.to_datetime(list(m["timestamps"]))
        k = (t >= t0) & (t <= t1)
        Zs.append(Z[k]); ts += list(t[k]); av.append(np.asarray(m["available"], bool)[k])
        meta = meta or m
    meta = dict(meta)
    meta["timestamps"] = np.array([t.to_pydatetime() for t in ts])
    meta["available"] = np.concatenate(av)
    return np.concatenate(Zs), meta


def load_polar_range(polar_dir, t0, t1):
    """ผลิตภัณฑ์ P2 ช่วง [t0, t1] → (dict[var] = array[t, y, x], times) หรือ (None, None) ถ้าไม่มีไฟล์"""
    parts, times = [], []
    for d in _days_between(t0, t1):
        p = Path(polar_dir) / f"KKN_{d}_polar_15min.nc"
        if not p.exists():
            continue
        P, T = tt.load_polar_grid(str(p))
        k = (T >= t0) & (T <= t1)
        if k.any():
            parts.append({v: a[k] for v, a in P.items()}); times += list(T[k])
    if not parts:
        return None, None
    return {v: np.concatenate([p[v] for p in parts]) for v in parts[0]}, pd.DatetimeIndex(times)


# ======================================================================
# 3) ติดตามหนึ่ง episode
# ======================================================================
def track_episode(ep, Z, meta, polar=None, polar_times=None, terrain=None, beam=None, tags=MAIN_TAGS,
                  start_edge="quiet", end_edge="quiet", params=None, track_kw=None):
    """T-DaTing ทุกความละเอียดของหนึ่งเหตุการณ์ → {tag: {labels, times, observed, cells, tracks, footprints, fp_masks}}"""
    ts_obs = list(pd.to_datetime(list(meta["timestamps"])).to_pydatetime())
    res = {}
    for tag, (dt, method) in tags.items():
        if method is None:
            cube, tl, obs = Z, ts_obs, np.ones(len(ts_obs), bool)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cube, tl, obs = tt.advect_interpolate(Z, ts_obs, n_sub=3, method=method)
            if dt == 10:
                cube, tl, obs = cube[::2], tl[::2], obs[::2]
        tracks = tt.run_tdating(cube, tl, params, track_kw)
        if not tracks:
            res[tag] = None
            continue
        lab = tt.tracks_to_labels(tracks, tl, cube.shape[1:]).astype(np.int32)
        lab[lab > 0] += ep * ID_BLOCK
        t_first, t_last = pd.Timestamp(tl[0]), pd.Timestamp(tl[-1])
        flags = []
        for k, tr in enumerate(tracks):
            t0, t1 = pd.Timestamp(tr["time"].iloc[0]), pd.Timestamp(tr["time"].iloc[-1])
            flags.append({"track_id": ep * ID_BLOCK + k, "episode": ep,
                          "at_episode_edge": (t0 == t_first) or (t1 == t_last),
                          "censored_start": (t0 == t_first) and start_edge != "quiet",
                          "censored_end": (t1 == t_last) and end_edge != "quiet"})
        C = tt.cell_table(lab, cube, tl, meta, terrain=terrain, polar=polar, polar_times=polar_times, beam=beam)
        C = tt.add_motion(C, terrain=terrain, meta=meta)
        T = tt.track_table(C).join(pd.DataFrame(flags).set_index("track_id"))
        uid = {k: f"E{ep:03d}-{k - ep * ID_BLOCK:04d}" for k in T.index}
        T.insert(0, "uid", [uid[k] for k in T.index])
        T.insert(1, "day", T["start"].dt.normalize())
        T.insert(2, "day_local", (T["start"] + LOCAL_OFFSET).dt.normalize())
        T["crosses_00utc"] = T["start"].dt.normalize() != T["end"].dt.normalize()
        T["crosses_local_midnight"] = (T["start"] + LOCAL_OFFSET).dt.normalize() != (T["end"] + LOCAL_OFFSET).dt.normalize()
        C.insert(0, "uid", C["track_id"].map(uid))
        C.insert(1, "day", C["time"].dt.normalize())
        C.insert(2, "day_local", (C["time"] + LOCAL_OFFSET).dt.normalize())
        C.insert(3, "episode", ep)
        fp = tt.footprint_masks(lab)
        gdf = tt.footprint_polygons(fp, meta, lab.shape[1:],
                                    extra=T[["uid", "episode", "day", "lifetime_min", "Zmax_obs", "path_km",
                                             "censored_start", "censored_end", "crosses_00utc"]])
        res[tag] = {"labels": lab, "times": list(pd.to_datetime(tl)), "observed": np.asarray(obs, bool),
                    "cells": C, "tracks": T, "footprints": gdf, "fp_masks": fp}
    return res


def terrain_from_dem(dem_path, meta, relief_km=5):
    """DEM 1 km (GeoTIFF AEQD จาก NB01) → กริดภูมิประเทศแบบ repo3: elevation, slope, relief, dhdx, dhdy, tpi"""
    import rasterio
    from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
    with rasterio.open(dem_path) as f:
        z = f.read(1).astype(float)
        left, top, res = f.bounds.left, f.bounds.top, f.res[0]
    c0 = int(round((meta["x"][0] - (left + res / 2)) / res))
    r0 = int(round(((top - res / 2) - meta["y"][0]) / res))
    h = z[r0:r0 + len(meta["y"]), c0:c0 + len(meta["x"])]
    h = np.where(np.isfinite(h), h, np.nanmin(h))
    d = meta["xpixelsize"]
    hs = uniform_filter(h, 3)
    dhdy_rows, dhdx = np.gradient(hs, d)
    dhdy = -dhdy_rows
    slope = np.degrees(np.arctan(np.hypot(dhdx, dhdy)))
    k = 2 * relief_km + 1
    return {"elevation": h, "slope": slope, "relief": maximum_filter(h, k) - minimum_filter(h, k),
            "dhdx": dhdx, "dhdy": dhdy, "tpi": h - uniform_filter(h, 11)}


def track_density(fp_masks, shape, track_ids=None):
    """จำนวนเส้นทางที่กวาดผ่านแต่ละ pixel"""
    n = np.zeros(int(np.prod(shape)), np.int32)
    for k, idx in fp_masks.items():
        if track_ids is None or k in track_ids:
            n[idx] += 1
    return n.reshape(shape)


def origin_grid(cells, keep, meta, shape):
    """จำนวนจุดเริ่มของเส้นทาง (ใน keep) ต่อ pixel"""
    origin = np.zeros(shape, np.int32)
    first = cells[cells["track_id"].isin(keep)].sort_values("time").groupby("track_id").first()
    ci = np.clip(np.round((first["cx"].values - meta["x"][0]) / meta["xpixelsize"]).astype(int), 0, shape[1] - 1)
    ri = np.clip(np.round((meta["y"][0] - first["cy"].values) / meta["ypixelsize"]).astype(int), 0, shape[0] - 1)
    np.add.at(origin, (ri, ci), 1)
    return origin


def run_episode(row, cappi_dir, polar_dir, parts_dir, terrain=None, beam=None, tags=MAIN_TAGS, min_life_min=30,
                params=None, track_kw=None):
    """งานของหนึ่ง worker: โหลดช่วงเวลา → ติดตาม → เขียน label แยกตามวัน (ไฟล์ส่วนย่อย) → คืนตาราง/กริดสรุป"""
    from .derived_io import write_labels
    import time as _time
    t_start = _time.time()
    ep = int(row["episode"])
    Z, meta = load_cappi_range(cappi_dir, row["start"], row["end"])
    P, PT = load_polar_range(polar_dir, row["start"], row["end"]) if polar_dir else (None, None)
    res = track_episode(ep, Z, meta, P, PT, terrain, beam, tags, row["start_edge"], row["end_edge"], params, track_kw)
    shape = Z.shape[1:]
    out = {"episode": ep, "tags": {}}
    for tag, r in res.items():
        if r is None:
            continue
        times = pd.DatetimeIndex(r["times"])
        for d in sorted(set(times.normalize())):
            k = times.normalize() == d
            write_labels(Path(parts_dir) / f"KKN_{d:%Y%m%d}_E{ep:03d}_labels_{tag}.nc", r["labels"][k],
                         list(times[k]), r["observed"][k], meta, {"episode": ep, "tag": tag})
        keep = set(r["tracks"].index[r["tracks"]["lifetime_min"] >= min_life_min])
        out["tags"][tag] = {"cells": r["cells"], "tracks": r["tracks"].reset_index(), "footprints": r["footprints"],
                            "density": track_density(r["fp_masks"], shape, keep),
                            "origin": origin_grid(r["cells"], keep, meta, shape)}
    out["sec"] = _time.time() - t_start
    return out


def merge_label_parts(parts_dir, out_dir, day, tag, meta, attrs=None, remove=True):
    """รวมไฟล์ส่วนย่อยของทุก episode ในวันเดียวกัน → KKN_YYYYMMDD_labels_<tag>.nc"""
    import xarray as xr
    from .derived_io import write_labels
    parts = sorted(Path(parts_dir).glob(f"KKN_{day}_E*_labels_{tag}.nc"))
    if not parts:
        return None
    L, T, O = [], [], []
    for p in parts:
        with xr.open_dataset(p) as ds:
            L.append(ds["label"].values.astype(np.int32)); T += list(pd.to_datetime(ds["time"].values))
            O.append(ds["observed"].values.astype(bool))
    L, O = np.concatenate(L), np.concatenate(O)
    o = np.argsort(np.array(T, dtype="datetime64[ns]"))
    out = Path(out_dir) / f"KKN_{day}_labels_{tag}.nc"
    write_labels(out, L[o], [T[i] for i in o], O[o], meta, attrs)
    if remove:
        for p in parts:
            p.unlink()
    return out
