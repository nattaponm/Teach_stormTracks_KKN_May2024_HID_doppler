"""P3: ติดตามพายุทั้งเดือน — ห่อฟังก์ชันของ repo3 (track_tools) ให้ทำงานกับข้อมูลที่มีช่องว่าง

ข้อแตกต่างจาก repo3 (ซึ่งใช้ 05–17 UTC ของวันเดียวที่ข้อมูลครบ)
-------------------------------------------------------------
* ติดตามเป็น **ช่วงต่อเนื่อง (segment)**: ภาพจริงที่ห่างกัน 15 นาทีพอดี ช่องที่ขาดตัดเส้นทาง
* ภาพ 5/10 นาทีสร้างเฉพาะภายใน segment (ไม่แทรกข้ามช่องว่าง)
* track_id ไม่ซ้ำกันภายในวัน และมี ``uid`` = YYYYMMDD-ttt ไม่ซ้ำกันทั้งเดือน
* เส้นทางที่เริ่ม/จบที่ขอบ segment ติดป้าย censored (อายุจริงอาจยาวกว่าที่วัดได้)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import track_tools as tt

MAIN_TAGS = {"15min": (15, None), "10min_linZ": (10, "linZ"), "05min_linZ": (5, "linZ")}


def segments_from_available(available, min_len=3):
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


def active_segments(Z, available, thr_dbz=35.0, min_area_km2=20.0, pad=2, min_len=3):
    """ตัด segment ให้เหลือเฉพาะช่วงที่มีพายุ (พื้นที่ ≥ thr_dbz อย่างน้อย min_area_km2) ± pad ภาพ"""
    area = np.array([np.nansum(z >= thr_dbz) for z in Z], float)
    out = []
    for i0, i1 in segments_from_available(available, min_len):
        act = np.where(area[i0:i1 + 1] >= min_area_km2)[0]
        if act.size == 0:
            continue
        a0, a1 = max(i0, i0 + act[0] - pad), min(i1, i0 + act[-1] + pad)
        if a1 - a0 + 1 >= min_len:
            out.append((a0, a1))
    return out


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


def track_day(day, Z, meta, polar=None, polar_times=None, terrain=None, beam=None, tags=MAIN_TAGS,
              thr_dbz=35.0, min_area_km2=20.0, params=None, track_kw=None):
    """T-DaTing ทุกความละเอียดของหนึ่งวัน

    คืน dict {tag: {"labels", "times", "observed", "cells", "tracks", "footprints"}} และตาราง segment
    """
    times_all = pd.to_datetime(list(meta["timestamps"]))
    segs = active_segments(Z, meta["available"], thr_dbz, min_area_km2)
    seg_tab = pd.DataFrame([{"day": day, "segment": s, "start": times_all[a], "end": times_all[b], "n_frames": b - a + 1}
                            for s, (a, b) in enumerate(segs)])
    res = {}
    for tag, (dt, method) in tags.items():
        lab_parts, t_parts, o_parts, z_parts, rows_tr = [], [], [], [], []
        offset = 0
        for s, (a, b) in enumerate(segs):
            Zs, ts = Z[a:b + 1], list(times_all[a:b + 1].to_pydatetime())
            if method is None:
                cube, tl, obs = Zs, ts, np.ones(len(ts), bool)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cube, tl, obs = tt.advect_interpolate(Zs, ts, n_sub=3, method=method)
                if dt == 10:
                    cube, tl, obs = cube[::2], tl[::2], obs[::2]
            tracks = tt.run_tdating(cube, tl, params, track_kw)
            lab = tt.tracks_to_labels(tracks, tl, cube.shape[1:]).astype(np.int32)
            lab[lab > 0] += offset
            for k in range(len(tracks)):
                tr = tracks[k]
                t0, t1 = pd.Timestamp(tr["time"].iloc[0]), pd.Timestamp(tr["time"].iloc[-1])
                rows_tr.append({"track_id": offset + k, "segment": s,
                                "censored_start": t0 == pd.Timestamp(tl[0]), "censored_end": t1 == pd.Timestamp(tl[-1])})
            offset += len(tracks)
            lab_parts.append(lab); t_parts += list(tl); o_parts.append(obs); z_parts.append(cube)
        if not lab_parts:
            res[tag] = None
            continue
        labels, cube = np.concatenate(lab_parts), np.concatenate(z_parts)
        observed = np.concatenate(o_parts)
        C = tt.cell_table(labels, cube, t_parts, meta, terrain=terrain, polar=polar, polar_times=polar_times, beam=beam)
        C = tt.add_motion(C, terrain=terrain, meta=meta)
        C.insert(0, "day", pd.Timestamp(day))
        T = tt.track_table(C.drop(columns="day"))
        T = T.join(pd.DataFrame(rows_tr).set_index("track_id"))
        T.insert(0, "day", pd.Timestamp(day))
        T.insert(1, "uid", [f"{pd.Timestamp(day):%Y%m%d}-{k:03d}" for k in T.index])
        C.insert(1, "uid", [f"{pd.Timestamp(day):%Y%m%d}-{k:03d}" for k in C["track_id"]])
        fp = tt.footprint_masks(labels)
        gdf = tt.footprint_polygons(fp, meta, labels.shape[1:], extra=T[["uid", "day", "lifetime_min", "Zmax_obs",
                                                                         "path_km", "censored_start", "censored_end"]])
        res[tag] = {"labels": labels, "times": t_parts, "observed": observed, "cells": C, "tracks": T,
                    "footprints": gdf, "fp_masks": fp}
    return res, seg_tab


def track_density(fp_masks, shape, track_ids=None):
    """จำนวนเส้นทางที่กวาดผ่านแต่ละ pixel"""
    n = np.zeros(int(np.prod(shape)), np.int32)
    for k, idx in fp_masks.items():
        if track_ids is None or k in track_ids:
            n[idx] += 1
    return n.reshape(shape)
