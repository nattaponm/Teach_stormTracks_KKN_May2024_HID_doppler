"""
track_tools.py — เส้นทางพายุ (T-DaTing) + กายภาพเมฆ + HID/Doppler + ภูมิประเทศ (เรดาร์ KKN, 3 พ.ค. 2567)
======================================================================================================

หมวดฟังก์ชัน
-------------
1. ข้อมูลกริด        : load_cube, write_cube_like, load_polar_grid, load_terrain, make_terrain_grid
2. แทรกภาพ           : advect_interpolate (linZ / nearest), loo_interpolation_test
3. T-DaTing          : run_tdating, tracks_to_labels, load_labels
4. Footprint         : footprint_masks, footprint_polygons, iou_matrix, match_tracks
5. คุณสมบัติเซลล์     : cell_table, add_motion, track_table, classify_tracks
6. สถิติ             : kruskal_dunn, cliffs_delta, partial_spearman, permutation_test, boxplot_groups
7. แผนที่            : plot_tracks, plot_footprints

ข้อตกลงของ repo นี้
-------------------
* กริด AEQD 1 km 481 × 481 (±240 km) แถวแรก = ทิศเหนือ (yorigin = upper) เหมือน repo pysteps
* ช่วงวิเคราะห์ 05:00–17:00 UTC (12:00–24:00 น.)
* ภาพ 5/10 นาที = ภาพ **สังเคราะห์** จาก advection interpolation (ข้อมูลจริงทุก 15 นาที)
* ค่า HID / Doppler / dual-pol ใช้ **เฉพาะเวลาที่วัดจริง** (นาทีที่หารด้วย 15 ลงตัว)
"""

import contextlib as _ctx
import datetime as _dt
import io as _io
import os as _os
import warnings as _warnings

import numpy as np
import pandas as pd

PARAMS = dict(minref=33, maxref=45, mindiff=5, minsize=15, minmax=40, mindis=8)   # ปรับสำหรับ KKN (repo pysteps NB6)
TRACK_KW = dict(mintrack=2, match_frac=0.25)
TAGS = ("15min", "10min_linZ", "05min_linZ", "10min_nearest", "05min_nearest")
MAIN = {15: "15min", 10: "10min_linZ", 5: "05min_linZ"}          # ชุดหลักที่ใช้ใน NB02–NB05 (linZ ต่อเนื่องกว่าสำหรับการติดตาม)
HID_NAMES = ["NC", "AG", "CR", "LR", "RP", "RN", "VI", "WS", "MH", "IH/HDG"]
ELEV0 = 0.48                                                        # มุมยกต่ำสุด (°)


def _quiet():
    return _ctx.redirect_stdout(_io.StringIO())


# ---------------------------------------------------------------------------
# 1) ข้อมูลกริด
# ---------------------------------------------------------------------------
def load_cube(path):
    """อ่านไฟล์ reflectivity (รูปแบบเดียวกับ repo pysteps) → (Z[t, y, x], meta) แถวแรก = ทิศเหนือ."""
    from .data_io import load_pcappi
    return load_pcappi(path)


def write_cube_like(template, Z, times, out_path, accutime, title=None, observed=None):
    """บันทึก cube (yorigin upper) ในรูปแบบไฟล์เดียวกับ template (pseudo-CAPPI ของ repo pysteps)."""
    import xarray as xr
    with xr.open_dataset(template) as ds0:
        attrs = dict(ds0.attrs)
        y_inc = float(ds0.y[0]) < float(ds0.y[-1])
        base = ds0[["source_height_m", "coverage_mask", "crs"]].load()
        lon, lat = ds0["lon"].load(), ds0["lat"].load()
        enc_r = {k: v for k, v in ds0["reflectivity"].encoding.items()
                 if k in ("dtype", "zlib", "complevel", "shuffle", "_FillValue", "scale_factor", "add_offset")}
    Zf = Z[:, ::-1] if y_inc else Z
    t = pd.to_datetime(list(times))
    ds = base.assign_coords(time=t)
    ds["reflectivity"] = (("time", "y", "x"), Zf.astype("float32"))
    ds["reflectivity"].attrs = {"units": "dBZ", "long_name": "Equivalent reflectivity factor"}
    ds["time_available"] = ("time", np.ones(len(t), "uint8"))
    ds["observed"] = ("time", (np.ones(len(t), bool) if observed is None else np.asarray(observed)).astype("uint8"))
    ds = ds.assign_coords(lon=lon, lat=lat)
    attrs.update(accutime=float(accutime), timestep=float(accutime),
                 title=title or attrs.get("title", ""),
                 comment="5/10-min frames are synthetic (advection interpolation); see variable 'observed'")
    ds.attrs = attrs
    enc = {"reflectivity": dict(enc_r, chunksizes=(1, Z.shape[1], Z.shape[2]))}
    ds.to_netcdf(out_path, encoding=enc)
    return out_path


def load_polar_grid(path):
    """ผลิตภัณฑ์จาก polar ราย 15 นาทีบนกริด 1 km → dict {ชื่อ: array[t, y, x]}, times (DatetimeIndex)."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        out = {k: ds[k].values for k in ds.data_vars}
        times = pd.to_datetime(ds["time"].values)
    return out, times


def make_terrain_grid(meta, data_dir, relief_km=5):
    """DEM (SRTM 600 m ใน data/basemap) → กริด 1 km: elevation, slope, relief, dh/dx, dh/dy, TPI."""
    import xarray as xr
    from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter
    with xr.open_dataset(_os.path.join(data_dir, "basemap", "terrain_KKN_aeqd_600m.nc")) as ds:
        h = ds["elevation"].astype(float).interp(x=meta["x"], y=meta["y"]).values
    h = np.where(np.isfinite(h), h, np.nanmin(h))
    d = meta["xpixelsize"]
    hs = uniform_filter(h, 3)                                   # ทำให้เรียบเล็กน้อยก่อนหาความชัน
    dhdy_rows, dhdx = np.gradient(hs, d)
    dhdy = -dhdy_rows                                           # แถวเพิ่มไปทางใต้ → กลับเครื่องหมาย
    slope = np.degrees(np.arctan(np.hypot(dhdx, dhdy)))
    k = 2 * relief_km + 1
    relief = maximum_filter(h, k) - minimum_filter(h, k)
    tpi = h - uniform_filter(h, 11)
    return {"elevation": h, "slope": slope, "relief": relief, "dhdx": dhdx, "dhdy": dhdy, "tpi": tpi}


def save_grids(path, grids, meta, times=None, attrs=None):
    """บันทึก dict ของ array (2D หรือ [t, y, x]) เป็น NetCDF แบบบีบอัด (แถวแรก = ทิศเหนือ)."""
    import xarray as xr
    coords = {"y": meta["y"], "x": meta["x"]}
    if times is not None:
        coords["time"] = pd.to_datetime(list(times))
    dv, enc = {}, {}
    for k, v in grids.items():
        v = np.asarray(v)
        dims = ("time", "y", "x") if v.ndim == 3 else ("y", "x")
        if v.dtype == bool:
            v = v.astype("uint8")
        dv[k] = (dims, v)
        if v.dtype.kind == "f":
            enc[k] = {"zlib": True, "complevel": 5, "dtype": "float32"}
        else:
            enc[k] = {"zlib": True, "complevel": 5}
    ds = xr.Dataset(dv, coords=coords, attrs={"projection": meta["projection"], "yorigin": "upper", **(attrs or {})})
    ds.to_netcdf(path, encoding=enc)
    return path


def load_terrain(path):
    import xarray as xr
    with xr.open_dataset(path) as ds:
        return {k: ds[k].values for k in ds.data_vars}


def beam_height_grid(meta, elev_deg=ELEV0):
    """ความสูงกึ่งกลางลำคลื่น (km MSL) ที่มุมยก elev_deg สำหรับทุก pixel."""
    X, Y = np.meshgrid(meta["x"], meta["y"])
    r = np.hypot(X, Y) / 1000.0
    a = 6371.0 * 4 / 3
    th = np.radians(elev_deg)
    return np.sqrt(r ** 2 + a ** 2 + 2 * r * a * np.sin(th)) - a + meta.get("radar_alt_m", 200.0) / 1000.0


# ---------------------------------------------------------------------------
# 2) แทรกภาพ (advection interpolation)
# ---------------------------------------------------------------------------
def advect_interpolate(cube, times, n_sub=3, method="nearest", fill_value=-15.0, motion_method="LK"):
    """เพิ่มความละเอียดเวลา Δt → Δt/n_sub

    method = "linZ"    : เลื่อนภาพก่อนและหลังตาม motion แล้ว **เฉลี่ยในหน่วย Z เชิงเส้น** (Germann & Zawadzki-style)
                         → เรียบ แต่แกนพายุถูกเฉลี่ยจนอ่อนลง
    method = "nearest" : เลื่อน **ภาพจริงที่ใกล้ที่สุด** ตาม motion เท่านั้น (ไม่เฉลี่ย)
                         → รักษาความเข้มของแกนพายุ แต่รูปร่างกระโดดที่กึ่งกลางช่วง
    คืนค่า (cube ใหม่, times ใหม่, observed[bool])
    """
    from pysteps import motion
    from scipy.ndimage import map_coordinates
    oflow = motion.get_method(motion_method)
    mask = np.all(np.isfinite(cube), axis=0)
    data = np.where(np.isfinite(cube), cube, fill_value)
    ny, nx = data.shape[1:]
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    times = list(times)
    dt_min = (times[1] - times[0]).total_seconds() / 60.0
    out, tt, obs = [data[0]], [times[0]], [True]
    for i in range(len(data) - 1):
        with _quiet():
            V = oflow(data[max(0, i - 1):i + 2], verbose=False)
        for k in range(1, n_sub):
            w = k / n_sub
            a = map_coordinates(data[i], (yy - w * V[1], xx - w * V[0]), order=1, mode="constant", cval=fill_value)
            b = map_coordinates(data[i + 1], (yy + (1 - w) * V[1], xx + (1 - w) * V[0]), order=1,
                                mode="constant", cval=fill_value)
            if method == "linZ":
                z = 10 * np.log10((1 - w) * 10 ** (a / 10) + w * 10 ** (b / 10))
            elif method == "nearest":
                z = a if w <= 0.5 else b
            else:
                raise ValueError("method = 'linZ' หรือ 'nearest'")
            out.append(z)
            tt.append(times[i] + _dt.timedelta(minutes=dt_min * w))
            obs.append(False)
        out.append(data[i + 1]); tt.append(times[i + 1]); obs.append(True)
    out = np.stack(out)
    out[:, ~mask] = np.nan
    return out, tt, np.array(obs)


def loo_interpolation_test(cube, times, methods=("linZ", "nearest"), thresholds=(35, 45)):
    """ทดสอบแบบตัดภาพออก (leave-one-out): ใช้ภาพทุก 30 นาทีแทรกกลับเป็น 15 นาที แล้วเทียบกับภาพจริงที่ตัดออก

    คืนตาราง CSI, RMSE และอัตราส่วนพื้นที่ ≥45 dBZ (ภาพแทรก/ภาพจริง) รวม baseline 'persistence'
    """
    from .kkn_tools import contingency
    sub = cube[::2]
    tsub = list(times)[::2]
    rows = []
    res = {m: advect_interpolate(sub, tsub, n_sub=2, method=m)[0] for m in methods}
    for j in range(len(sub) - 1):
        k = 2 * j + 1
        if k >= len(cube):
            break
        truth = cube[k]
        cand = {m: res[m][2 * j + 1] for m in methods}
        cand["persistence"] = cube[k - 1]
        for name, est in cand.items():
            r = {"time": times[k], "method": name}
            for thr in thresholds:
                r[f"CSI{thr}"] = contingency(est, truth, thr)["CSI"]
            e, t = np.nan_to_num(est, nan=-15), np.nan_to_num(truth, nan=-15)
            act = (e > 10) | (t > 10)
            r["RMSE_dB"] = float(np.sqrt(np.mean((e[act] - t[act]) ** 2))) if act.any() else np.nan
            r["A45_ratio"] = np.sum(e >= 45) / max(np.sum(t >= 45), 1)
            rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3) T-DaTing
# ---------------------------------------------------------------------------
def run_tdating(cube, times, params=None, track_kw=None):
    """รัน pysteps T-DaTing แบบเงียบ → list ของ DataFrame (หนึ่ง DataFrame ต่อหนึ่งเส้นทาง)."""
    from pysteps.tracking import tdating
    p = dict(PARAMS, **(params or {}))
    k = dict(TRACK_KW, **(track_kw or {}))
    with _quiet(), _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        tracks, _, _ = tdating.dating(np.nan_to_num(cube, nan=-15.0), list(times), **k, **p)
    return tracks


def tracks_to_labels(tracks, times, shape):
    """แปลงผล T-DaTing เป็น label cube [t, y, x] (ค่า = track_id + 1, 0 = ไม่มีเซลล์)."""
    times = list(pd.to_datetime(list(times)))
    lab = np.zeros((len(times),) + tuple(shape), dtype="int16")
    for k, tr in enumerate(tracks):
        for _, c in tr.iterrows():
            i = times.index(pd.Timestamp(c["time"]))
            lab[i, np.asarray(c["y"]), np.asarray(c["x"])] = k + 1
    return lab


def load_labels(path):
    """อ่าน label cube ที่ 00 บันทึกไว้ → (labels[t, y, x], times, observed)."""
    import xarray as xr
    with xr.open_dataset(path) as ds:
        return ds["label"].values.astype(int), pd.to_datetime(ds["time"].values), ds["observed"].values.astype(bool)


# ---------------------------------------------------------------------------
# 4) Footprint
# ---------------------------------------------------------------------------
def footprint_masks(labels, track_ids=None):
    """dict {track_id: flat index ของ pixel ที่เส้นทางกวาดผ่าน (union ทุกเวลา)}."""
    lab2 = labels.reshape(labels.shape[0], -1)
    t_idx, p_idx = np.nonzero(lab2)
    ids = lab2[t_idx, p_idx] - 1
    out = {}
    order = np.argsort(ids, kind="stable")
    ids_s, p_s = ids[order], p_idx[order]
    bounds = np.flatnonzero(np.diff(ids_s)) + 1
    for seg_ids, seg_p in zip(np.split(ids_s, bounds), np.split(p_s, bounds)):
        if len(seg_ids) == 0:
            continue
        k = int(seg_ids[0])
        if track_ids is None or k in track_ids:
            out[k] = np.unique(seg_p)
    return out


def footprint_polygons(fp, meta, shape, extra=None):
    """footprint (flat index) → GeoDataFrame ของ polygon (ระบบพิกัด AEQD ของเรดาร์)."""
    import geopandas as gpd
    import rasterio.features
    from rasterio.transform import from_origin
    from shapely.geometry import shape as _shape
    from shapely.ops import unary_union
    tr = from_origin(meta["x1"], meta["y2"], meta["xpixelsize"], meta["ypixelsize"])
    rows = []
    for k, idx in fp.items():
        m = np.zeros(np.prod(shape), "uint8"); m[idx] = 1
        m = m.reshape(shape)
        geoms = [_shape(g) for g, v in rasterio.features.shapes(m, mask=m > 0, transform=tr)]
        g = unary_union(geoms)
        r = {"track_id": k, "geometry": g, "area_km2": g.area / 1e6, "hull_km2": g.convex_hull.area / 1e6}
        if extra is not None and k in extra.index:
            r.update(extra.loc[k].to_dict())
        rows.append(r)
    gdf = gpd.GeoDataFrame(rows, crs=meta["projection"])
    gdf["compactness"] = gdf["area_km2"] / gdf["hull_km2"]
    return gdf


def iou_matrix(fpA, fpB):
    """IoU ระหว่าง footprint ทุกคู่ (A × B) — คืน DataFrame."""
    A, B = list(fpA), list(fpB)
    M = np.zeros((len(A), len(B)))
    setsB = {b: set(fpB[b].tolist()) for b in B}
    for i, a in enumerate(A):
        sa = set(fpA[a].tolist())
        for j, b in enumerate(B):
            inter = len(sa & setsB[b])
            if inter:
                M[i, j] = inter / (len(sa) + len(setsB[b]) - inter)
    return pd.DataFrame(M, index=A, columns=B)


def match_tracks(fpA, fpB, min_iou=0.1):
    """จับคู่เส้นทางระหว่างสองความละเอียด: เส้นทางใน A ↔ เส้นทางใน B ที่ IoU สูงสุด."""
    M = iou_matrix(fpA, fpB)
    best = M.idxmax(axis=1)
    val = M.max(axis=1)
    out = pd.DataFrame({"match": best.where(val >= min_iou), "iou": val})
    out.index.name = "track_id"
    return out


# ---------------------------------------------------------------------------
# 5) คุณสมบัติเซลล์และเส้นทาง
# ---------------------------------------------------------------------------
def _pct(a, q, nmin=3):
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size >= nmin else np.nan


def cell_table(labels, cube, times, meta, terrain=None, polar=None, polar_times=None, beam=None):
    """ตารางคุณสมบัติของทุกเซลล์ (หนึ่งแถว = หนึ่งเส้นทาง ณ หนึ่งเวลา)

    สมการหลัก
    ---------
    A = N_pixel × 1 km²,  D_eq = 2√(A/π)
    centroid ถ่วงน้ำหนักด้วย Z เชิงเส้น:  x_c = Σ x_i Z_i / Σ Z_i
    Z_mean = 10 log10( mean(10^(dBZ/10)) )
    ค่า HID/Doppler ดึงเฉพาะเวลาที่มีในชุด polar (ทุก 15 นาที) — เวลาอื่นเป็น NaN
    """
    X, Y = np.meshgrid(meta["x"], meta["y"])
    Xf, Yf = X.ravel(), Y.ravel()
    R = np.hypot(Xf, Yf) / 1000.0
    times = pd.to_datetime(list(times))
    pidx = {t: i for i, t in enumerate(pd.to_datetime(list(polar_times)))} if polar is not None else {}
    beamf = beam.ravel() if beam is not None else None
    T = {k: v.ravel() for k, v in terrain.items()} if terrain is not None else None
    rows = []
    for i, t in enumerate(times):
        lab = labels[i].ravel()
        nz = np.flatnonzero(lab)
        if nz.size == 0:
            continue
        z = np.nan_to_num(cube[i].ravel(), nan=-15.0)
        ids = lab[nz]
        P = None
        if t in pidx:
            j = pidx[t]
            P = {k: v[j].ravel() for k, v in polar.items()}
        for k in np.unique(ids):
            px = nz[ids == k]
            zz = z[px]
            w = 10 ** (zz / 10)
            r = {"track_id": int(k) - 1, "time": t, "observed": t.minute % 15 == 0,
                 "area_km2": float(px.size), "deq_km": 2 * np.sqrt(px.size / np.pi),
                 "cx": float(np.sum(Xf[px] * w) / w.sum()), "cy": float(np.sum(Yf[px] * w) / w.sum()),
                 "Zmax": float(zz.max()), "Zmean": float(10 * np.log10(w.mean())),
                 "A45_km2": float(np.sum(zz >= 45)), "range_km": float(R[px].mean())}
            if beamf is not None:
                r["beam_h_km"] = float(beamf[px].mean())
            if T is not None:
                r.update(dem_mean=float(T["elevation"][px].mean()), dem_max=float(T["elevation"][px].max()),
                         slope_mean=float(T["slope"][px].mean()), relief_mean=float(T["relief"][px].mean()))
            if P is not None:
                h0 = P["hid_s0"][px]
                valid0 = h0 > 0
                up = np.concatenate([P["hid_s2"][px], P["hid_s3"][px]])
                vup = up > 0
                v = P["vel"][px]
                r.update(
                    hail_low_frac=float(np.mean(np.isin(h0[valid0], [8, 9]))) if valid0.sum() >= 3 else np.nan,
                    rain_low_frac=float(np.mean(np.isin(h0[valid0], [3, 5]))) if valid0.sum() >= 3 else np.nan,
                    graupel_up_frac=float(np.mean(up[vup] == 4)) if vup.sum() >= 3 else np.nan,
                    ice_up_frac=float(np.mean(np.isin(up[vup], [1, 2, 6]))) if vup.sum() >= 3 else np.nan,
                    hail_col_km2=float(np.sum(P["hail_col"][px] > 0)),
                    V95=_pct(np.abs(v), 95), dV=float(np.nanmax(v) - np.nanmin(v)) if np.isfinite(v).sum() >= 5 else np.nan,
                    DIVmax=_pct(P["div"][px] * 1e3, 100, 1), AZSmax=_pct(np.abs(P["azshear"][px]) * 1e3, 100, 1),
                    EDR95=_pct(P["edr"][px], 95), KDP95=_pct(P["kdp"][px], 95), ZDR90=_pct(P["zdr"][px], 90))
            rows.append(r)
    C = pd.DataFrame(rows).sort_values(["track_id", "time"]).reset_index(drop=True)
    lon, lat = _to_lonlat(meta, C["cx"].values, C["cy"].values)
    C["lon"], C["lat"] = lon, lat
    return C


def _to_lonlat(meta, x, y):
    from pyproj import Transformer
    return Transformer.from_crs(meta["projection"], "EPSG:4326", always_xy=True).transform(x, y)


def add_motion(C, terrain=None, meta=None, smooth=3):
    """ความเร็ว/ทิศจาก centroid (ทำให้เรียบด้วย rolling mean) และแรงยกจากภูมิประเทศ

    u = Δx_c/Δt,  v = Δy_c/Δt,  speed = √(u² + v²),  ทิศที่ไป = atan2(u, v)
    w_oro = u ∂h/∂x + v ∂h/∂y   (m/s; บวก = เคลื่อนขึ้นลาด)
    """
    C = C.copy()
    out = []
    for k, g in C.groupby("track_id"):
        g = g.sort_values("time").copy()
        ts = (g["time"] - g["time"].iloc[0]).dt.total_seconds().values
        xs = g["cx"].rolling(smooth, center=True, min_periods=1).mean().values
        ys = g["cy"].rolling(smooth, center=True, min_periods=1).mean().values
        if len(g) >= 2:
            u = np.gradient(xs, ts); v = np.gradient(ys, ts)
        else:
            u = v = np.full(len(g), np.nan)
        g["u_ms"], g["v_ms"] = u, v
        out.append(g)
    C = pd.concat(out)
    C["speed_kmh"] = np.hypot(C["u_ms"], C["v_ms"]) * 3.6
    C["dir_to_deg"] = np.degrees(np.arctan2(C["u_ms"], C["v_ms"])) % 360
    if terrain is not None and meta is not None:
        ci = np.clip(np.round((C["cx"].values - meta["x"][0]) / meta["xpixelsize"]).astype(int), 0, len(meta["x"]) - 1)
        ri = np.clip(np.round((meta["y"][0] - C["cy"].values) / meta["ypixelsize"]).astype(int), 0, len(meta["y"]) - 1)
        C["w_oro_ms"] = C["u_ms"] * terrain["dhdx"][ri, ci] + C["v_ms"] * terrain["dhdy"][ri, ci]
        C["dem_centroid"] = terrain["elevation"][ri, ci]
    return C.sort_values(["track_id", "time"]).reset_index(drop=True)


def track_table(C):
    """สรุประดับเส้นทาง (หน่วยหลักของการวิเคราะห์สถิติ)

    lifetime = t_end − t_start,  L_path = Σ|Δx_c|,  L_straight = |x_end − x_start|,  sinuosity = L_path / L_straight
    """
    def agg(g):
        g = g.sort_values("time")
        dx, dy = np.diff(g["cx"].values), np.diff(g["cy"].values)
        lpath = float(np.sum(np.hypot(dx, dy)) / 1000)
        lstr = float(np.hypot(g["cx"].iloc[-1] - g["cx"].iloc[0], g["cy"].iloc[-1] - g["cy"].iloc[0]) / 1000)
        o = g[g["observed"]]
        r = {"start": g["time"].iloc[0], "end": g["time"].iloc[-1], "n_frames": len(g), "n_observed": len(o),
             "lifetime_min": (g["time"].iloc[-1] - g["time"].iloc[0]).total_seconds() / 60,
             "path_km": lpath, "straight_km": lstr, "sinuosity": lpath / lstr if lstr > 1 else np.nan,
             "start_lon": g["lon"].iloc[0], "start_lat": g["lat"].iloc[0],
             "area_max": g["area_km2"].max(), "deq_max": g["deq_km"].max(),
             "Zmax_all": g["Zmax"].max(), "Zmax_obs": o["Zmax"].max() if len(o) else np.nan,
             "A45_max_obs": o["A45_km2"].max() if len(o) else np.nan,
             "range_km": g["range_km"].mean()}
        for c in ("speed_kmh", "beam_h_km", "dem_mean", "dem_max", "slope_mean", "relief_mean", "w_oro_ms"):
            if c in g:
                r[c] = g[c].median() if c in ("speed_kmh", "w_oro_ms") else g[c].mean()
        if "dem_mean" in g:
            r["dem_start"] = g["dem_mean"].iloc[0]
        if "dir_to_deg" in g:
            a = np.radians(g["dir_to_deg"].dropna())
            r["dir_to_deg"] = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360) if len(a) else np.nan
        if "w_oro_ms" in g:
            r["upslope_frac"] = float(np.mean(g["w_oro_ms"].dropna() > 0)) if g["w_oro_ms"].notna().any() else np.nan
        for c in ("hail_low_frac", "hail_col_km2", "graupel_up_frac", "rain_low_frac", "ice_up_frac",
                  "V95", "dV", "DIVmax", "AZSmax", "EDR95", "KDP95", "ZDR90"):
            if c in o:
                r[c] = o[c].max() if o[c].notna().any() else np.nan
        return pd.Series(r)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        T = C.groupby("track_id").apply(agg, include_groups=False)
    return T


def classify_tracks(T, life_edges=(0, 45, 1e4), life_labels=("short (<45 min)", "long (≥45 min)"),
                    terrain_edges=(-1, 200, 400, 5000), terrain_labels=("<200 m", "200–400 m", ">400 m"),
                    range_edges=(0, 100, 160, 250), range_labels=("<100 km", "100–160 km", ">160 km")):
    """เพิ่มคอลัมน์คลาส: อายุ, ขนาด (tercile ของ D_eq), ความสูงภูมิประเทศ, ช่วงระยะจากเรดาร์."""
    T = T.copy()
    T["life_class"] = pd.cut(T["lifetime_min"], life_edges, labels=life_labels, right=False)
    q = T["deq_max"].quantile([1 / 3, 2 / 3]).values
    T["size_class"] = pd.cut(T["deq_max"], [-1, q[0], q[1], 1e4], labels=["small", "medium", "large"])
    if "dem_mean" in T:
        T["terrain_class"] = pd.cut(T["dem_mean"], terrain_edges, labels=terrain_labels)
    T["range_class"] = pd.cut(T["range_km"], range_edges, labels=range_labels)
    return T


# ---------------------------------------------------------------------------
# 6) สถิติ
# ---------------------------------------------------------------------------
def cliffs_delta(a, b):
    """Cliff's δ = P(A > B) − P(A < B)  (|δ| < 0.147 เล็กมาก, < 0.33 เล็ก, < 0.474 ปานกลาง, ≥ 0.474 ใหญ่)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = np.sum(a[:, None] > b[None, :]); lt = np.sum(a[:, None] < b[None, :])
    return (gt - lt) / (len(a) * len(b))


def kruskal_dunn(df, value, group):
    """Kruskal–Wallis + Dunn post-hoc (Bonferroni) — คืน (H, p, ตารางคู่)."""
    from scipy.stats import kruskal, norm, rankdata
    d = df[[value, group]].dropna()
    groups = [g for g in d[group].cat.categories if (d[group] == g).sum() > 0] if hasattr(d[group], "cat") \
        else sorted(d[group].unique())
    samples = [d.loc[d[group] == g, value].values for g in groups]
    if len([s for s in samples if len(s) > 0]) < 2:
        return np.nan, np.nan, pd.DataFrame()
    H, p = kruskal(*samples)
    rk = rankdata(d[value].values)
    N = len(d)
    _, counts = np.unique(d[value].values, return_counts=True)
    tie = 1 - np.sum(counts ** 3 - counts) / (N ** 3 - N)
    mr = {g: rk[(d[group] == g).values].mean() for g in groups}
    n = {g: (d[group] == g).sum() for g in groups}
    rows, m = [], len(groups) * (len(groups) - 1) / 2
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            se = np.sqrt(tie * N * (N + 1) / 12 * (1 / n[a] + 1 / n[b]))
            zz = (mr[a] - mr[b]) / se
            rows.append({"pair": f"{a} vs {b}", "z": zz, "p_bonf": min(1.0, 2 * norm.sf(abs(zz)) * m),
                         "cliffs_delta": cliffs_delta(samples[i], samples[j])})
    return H, p, pd.DataFrame(rows)


def partial_spearman(df, x, y, z):
    """Spearman ระหว่าง x กับ y โดยควบคุม z: correlation ของ residual หลังถดถอยอันดับของ x, y บนอันดับของ z."""
    from scipy.stats import pearsonr, rankdata
    d = df[[x, y, z]].dropna()
    rx, ry, rz = (rankdata(d[c]) for c in (x, y, z))
    A = np.vstack([rz, np.ones_like(rz)]).T
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    r, p = pearsonr(ex, ey)
    return r, p, len(d)


def permutation_test(sample, population, stat=np.median, n=5000, seed=0):
    """ทดสอบว่า stat(sample) สูงกว่าการสุ่มจาก population ขนาดเท่ากันหรือไม่ → (ค่าจริง, ค่าที่คาด, p ด้านเดียว)."""
    rng = np.random.default_rng(seed)
    sample, population = np.asarray(sample), np.asarray(population)
    obs = stat(sample)
    sims = np.array([stat(rng.choice(population, len(sample), replace=False)) for _ in range(n)])
    return float(obs), float(np.median(sims)), float((np.sum(sims >= obs) + 1) / (n + 1))


def boxplot_groups(ax, df, value, group, order=None, colors=None, ylabel=None, show_n=True, points=True):
    """boxplot ตามกลุ่ม + จุดข้อมูลจริง (jitter) + จำนวนตัวอย่างใต้แกน."""
    order = order or (list(df[group].cat.categories) if hasattr(df[group], "cat") else sorted(df[group].dropna().unique()))
    data = [df.loc[df[group] == g, value].dropna().values for g in order]
    bp = ax.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                    medianprops={"color": "k", "lw": 1.2})
    cols = colors or ["#d9d9d9"] * len(order)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    if points:
        rng = np.random.default_rng(1)
        for i, d in enumerate(data, start=1):
            ax.plot(i + rng.uniform(-0.15, 0.15, len(d)), d, ".", ms=3, color="0.25", alpha=0.6, zorder=3)
    ax.set_xticks(range(1, len(order) + 1), [f"{g}\n(n={len(d)})" if show_n else str(g) for g, d in zip(order, data)])
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    return bp


# ---------------------------------------------------------------------------
# 7) แผนที่
# ---------------------------------------------------------------------------
def plot_tracks(ax, C, meta, color_by="time", cmap="plasma", vmin=5, vmax=17, min_life=0, lw=0.6, ms=3,
                start_marker=True, T=None):
    """เส้นทาง (centroid) — จุดระบายสีตามเวลา UTC (ชั่วโมง) หรือคอลัมน์อื่น; ○ = จุดเริ่มต้น."""
    import matplotlib.pyplot as plt
    from . import mapplot as mp
    crs = mp.get_crs(meta)
    sc = None
    for k, g in C.groupby("track_id"):
        g = g.sort_values("time")
        life = (g["time"].iloc[-1] - g["time"].iloc[0]).total_seconds() / 60
        if life < min_life:
            continue
        ax.plot(g["cx"], g["cy"], color="k", lw=lw, transform=crs, zorder=6)
        cval = g["time"].dt.hour + g["time"].dt.minute / 60 if color_by == "time" else g[color_by]
        sc = ax.scatter(g["cx"], g["cy"], c=cval, cmap=cmap, vmin=vmin, vmax=vmax, s=ms, transform=crs, zorder=7,
                        linewidths=0)
        if start_marker:
            ax.plot(g["cx"].iloc[0], g["cy"].iloc[0], "o", ms=3, mfc="w", mec="k", mew=0.6, transform=crs, zorder=8)
    return sc


def plot_footprints(ax, gdf, meta, column="lifetime_min", cmap="viridis", vmin=None, vmax=None, alpha=0.75):
    """footprint polygon ระบายสีตามคอลัมน์ (เช่น อายุ)."""
    import matplotlib as mpl
    from . import mapplot as mp
    crs = mp.get_crs(meta)
    cm = mpl.colormaps[cmap]
    v = gdf[column].astype(float)
    norm = mpl.colors.Normalize(vmin if vmin is not None else v.min(), vmax if vmax is not None else v.max())
    for _, r in gdf.sort_values(column).iterrows():
        ax.add_geometries([r.geometry], crs=crs, facecolor=cm(norm(r[column])), edgecolor="k", linewidth=0.3,
                          alpha=alpha, zorder=5)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    return sm


# ---------------------------------------------------------------------------
# 8) จาก polar volume → กริด (ใช้ใน 00)
# ---------------------------------------------------------------------------
POLAR_VARS = ("Zc", "vel", "div", "azshear", "edr", "kdp", "zdr", "hid_s0", "hid_s1", "hid_s2", "hid_s3",
              "hail_col", "graupel_col")


def polar_volume_to_grid(radar, meta):
    """ประมวลผลแล้ว (pt.process_volume) → dict ของกริด 2 มิติบนกริด meta (nearest polar lookup)

    Zc, vel (VEL_RB), div, azshear, zdr : มุมยก 0.48°
    edr, kdp                             : ค่าสูงสุดในคอลัมน์ (4 มุมยก)
    hid_s0..s3                           : คลาส HID ของแต่ละมุมยก (0 = NC)
    hail_col                             : คอลัมน์มี MH/IH-HDG และ Z ≥ 40 dBZ ที่ gate เดียวกัน (เหมือน repo HID NB6)
    """
    from . import polar_tools as pt
    lk = [pt.grid_lookup(radar, s, meta) for s in range(radar.nsweeps)]
    g = {"Zc": pt.to_grid(radar, "DBZ_C", lk[0]), "vel": pt.to_grid(radar, "VEL_RB", lk[0]),
         "div": pt.to_grid(radar, "DIV", lk[0]), "azshear": pt.to_grid(radar, "AZSHEAR", lk[0]),
         "zdr": pt.to_grid(radar, "ZDR_C", lk[0])}
    g["edr"], _ = pt.column_grid(radar, "EDR", meta, lookups=lk)
    g["kdp"], _ = pt.column_grid(radar, "KDP", meta, lookups=lk)
    zs = []
    for s, k in enumerate(lk):
        h = np.nan_to_num(pt.to_grid(radar, "HID", k)).astype("int8")
        g[f"hid_s{s}"] = h
        zs.append(np.nan_to_num(pt.to_grid(radar, "DBZ_C", k), nan=-99))
    H = np.stack([g[f"hid_s{s}"] for s in range(len(lk))])
    g["hail_col"] = (np.isin(H, [8, 9]) & (np.stack(zs) >= 40)).any(0)
    g["graupel_col"] = (H == 4).any(0)
    return g


def load_set(der, tag):
    """โหลดชุดข้อมูลตาม tag ('15min', '10min_nearest', '05min_linZ', ...) → dict(labels, Z, times, observed, meta)."""
    lab, times, obs = load_labels(_os.path.join(der, f"KKN_20240503_labels_{tag}.nc"))
    if tag == "15min":
        Z, meta = load_cube(_os.path.join(der, "KKN_20240503_cappi_15min.nc"))
    else:
        Z, meta = load_cube(_os.path.join(der, f"KKN_20240503_cappi_05min_{tag.split('_')[1]}.nc"))
        if tag.startswith("10min"):
            Z = Z[::2]
    assert len(Z) == len(times)
    return {"labels": lab, "Z": Z, "times": times, "observed": obs, "meta": meta, "tag": tag,
            "dt": int(tag[:2])}


def tracks_near(T, C, lon, lat, t_ref=None, radius_km=15.0, window_min=45.0, meta=None):
    """เส้นทางที่ centroid ผ่านใกล้จุด (lon, lat) ภายใน radius_km (และภายใน ±window_min นาทีของ t_ref ถ้าระบุ)."""
    from pyproj import Transformer
    x0, y0 = Transformer.from_crs("EPSG:4326", meta["projection"], always_xy=True).transform(lon, lat)
    d = np.hypot(C["cx"] - x0, C["cy"] - y0) / 1000.0
    m = d <= radius_km
    if t_ref is not None:
        m &= (C["time"] - pd.Timestamp(t_ref)).abs() <= pd.Timedelta(minutes=window_min)
    near = C[m].assign(dist_km=d[m])
    if near.empty:
        return pd.DataFrame()
    best = near.groupby("track_id")["dist_km"].min().sort_values()
    out = T.loc[best.index].copy()
    out.insert(0, "min_dist_km", best.values)
    return out
