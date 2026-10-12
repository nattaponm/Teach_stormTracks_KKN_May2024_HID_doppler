"""
kkn_tools.py — ฟังก์ชันเสริมสำหรับเคสพายุฤดูร้อน 3 พ.ค. 2567 (เรดาร์ขอนแก่น KKN)
==============================================================================

* จุดรายงานความเสียหาย (data/reports) และจุดอ้างอิง
* แปลงพิกัด lon/lat ↔ x/y (AEQD) ↔ pixel
* Z–R + hail cap, ฝนสะสม, พื้นที่ echo, แยก convective/stratiform (Steiner et al. 1995)
* ความสูงลำคลื่นเรดาร์ (4/3 Earth) สำหรับ 4 มุมยกของ KKN
* ความเร็ว/ทิศของ motion field, Hovmöller diagram, advection interpolation
* แปลงผล T-DaTing (พิกัด pixel) → ตาราง/GeoDataFrame พิกัดจริง
"""

import datetime as _dt
import os as _os

import numpy as np

LT_OFFSET_H = 7                                   # เวลาไทย = UTC + 7
ELEVATIONS_DEG = (0.48, 1.48, 2.39, 3.39)          # มุมยกของ KKN ในเคสนี้
BEAMWIDTH_DEG = 1.0
KM_PER_PX = 1.0

# จุดอ้างอิงหลัก (ใช้ id จาก data/reports/KKN_20240503_reports.csv)
KEY_SITES = {
    "Phu Kradueng": 1,      # เลย 14:00 น.
    "Na Klang": 4,          # หนองบัวลำภู 16:30 น.
    "Suwannakhuha": 6,      # หนองบัวลำภู 16:30 น. (ต.นาด่าน)
    "Si Chiang Mai": 7,     # หนองคาย ~17:00 น.
    "Bueng Kan": 16,        # บึงกาฬ (กลางคืน)
    "Khon Kaen city": 17,   # ขอนแก่น (กลางคืน)
}


# ---------------------------------------------------------------------------
# เวลา
# ---------------------------------------------------------------------------
def to_local(ts):
    return ts + _dt.timedelta(hours=LT_OFFSET_H)


def hhmm(ts, local=False):
    return (to_local(ts) if local else ts).strftime("%H:%M")


# ---------------------------------------------------------------------------
# พิกัด
# ---------------------------------------------------------------------------
def lonlat_to_xy(meta, lon, lat):
    """lon/lat (WGS84) → x, y (เมตร) ในระบบ AEQD ของเรดาร์."""
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", meta["projection"], always_xy=True)
    return tr.transform(lon, lat)


def xy_to_lonlat(meta, x, y):
    from pyproj import Transformer
    tr = Transformer.from_crs(meta["projection"], "EPSG:4326", always_xy=True)
    return tr.transform(x, y)


def xy_to_pixel(meta, x, y):
    """x, y (เมตร) → (col, row) แบบทศนิยม (แถว 0 อยู่ทิศเหนือ)."""
    col = (np.asarray(x) - meta["x1"]) / meta["xpixelsize"] - 0.5
    row = (meta["y2"] - np.asarray(y)) / meta["ypixelsize"] - 0.5
    return col, row


def pixel_to_xy(meta, col, row):
    """(col, row) → x, y (เมตร) ที่กึ่งกลาง pixel — ใช้แปลงผล T-DaTing ก่อนทำ GIS."""
    x = meta["x1"] + (np.asarray(col, dtype=float) + 0.5) * meta["xpixelsize"]
    y = meta["y2"] - (np.asarray(row, dtype=float) + 0.5) * meta["ypixelsize"]
    return x, y


def distance_to_radar_km(meta):
    X, Y = np.meshgrid(meta["x"], meta["y"])
    return np.hypot(X, Y) / 1000.0


def range_mask(meta, radius_km=150.0):
    """True = อยู่ในรัศมี radius_km และอยู่ในพื้นที่ครอบคลุมของเรดาร์."""
    return (distance_to_radar_km(meta) <= radius_km) & meta["coverage"]


def box_around(meta, lon, lat, half_km):
    """กรอบ (x1, x2, y1, y2) เมตร รอบจุด lon/lat — ใช้ซูมแผนที่หรือ crop ข้อมูล."""
    x, y = lonlat_to_xy(meta, lon, lat)
    h = half_km * 1000.0
    return (x - h, x + h, y - h, y + h)


def crop(cube, meta, box):
    """ตัดข้อมูลเฉพาะกรอบ box = (x1, x2, y1, y2) เมตร คืนค่า (cube_crop, meta_crop)."""
    x, y = meta["x"], meta["y"]
    ci = np.where((x >= box[0]) & (x <= box[1]))[0]
    ri = np.where((y >= box[2]) & (y <= box[3]))[0]
    sl = (slice(ri[0], ri[-1] + 1), slice(ci[0], ci[-1] + 1))
    m = dict(meta)
    dx = meta["xpixelsize"] / 2
    m.update(x=x[sl[1]], y=y[sl[0]], x1=x[ci[0]] - dx, x2=x[ci[-1]] + dx,
             y1=y[ri[-1]] - dx, y2=y[ri[0]] + dx)
    for k in ("coverage", "source_height_m", "lon", "lat"):
        m[k] = meta[k][sl]
    return cube[(Ellipsis,) + sl], m


# ---------------------------------------------------------------------------
# จุดรายงานความเสียหาย
# ---------------------------------------------------------------------------
def load_reports(data_dir, meta=None):
    """อ่าน data/reports/KKN_20240503_reports.csv (+ คำนวณ x, y, ระยะ/มุมจากเรดาร์ ถ้าให้ meta)."""
    import pandas as pd
    df = pd.read_csv(_os.path.join(data_dir, "reports", "KKN_20240503_reports.csv"))
    if meta is not None:
        x, y = lonlat_to_xy(meta, df["lon"].values, df["lat"].values)
        df["x"], df["y"] = x, y
        df["range_km"] = np.hypot(x, y) / 1000.0
        df["azimuth_deg"] = (np.degrees(np.arctan2(x, y)) + 360) % 360
    return df


def site(df, name):
    """คืนค่าแถวของจุดอ้างอิงตามชื่อใน KEY_SITES."""
    return df.loc[df["id"] == KEY_SITES[name]].iloc[0]


def report_time_utc(row, date):
    """เวลา UTC ของรายงาน (datetime) หรือ None ถ้าระบุแค่ 'กลางคืน'."""
    if not isinstance(row["time_UTC"], str) or not row["time_UTC"]:
        return None
    hh, mm = map(int, row["time_UTC"].split(":"))
    return date.replace(hour=hh, minute=mm)


# ---------------------------------------------------------------------------
# Z–R และฝน
# ---------------------------------------------------------------------------
ZR_SETS = {
    "Marshall-Palmer": (200.0, 1.6),
    "WSR-88D convective": (300.0, 1.4),
    "Rosenfeld tropical": (250.0, 1.2),
}


def dbz_to_rainrate(dbz, a=200.0, b=1.6, hail_cap=55.0, min_dbz=15.0):
    """dBZ → อัตราฝน (mm/h) จาก Z = a R^b

    hail_cap : ตัดค่า dBZ ที่สูงกว่านี้ (ลูกเห็บทำให้ Z สูงเกินจริง) ใส่ None = ไม่ตัด
    min_dbz  : ต่ำกว่านี้ถือว่าไม่มีฝน (R = 0);  NaN (นอกรัศมี) คงเป็น NaN
    """
    dbz = np.asarray(dbz, dtype=float)
    z = dbz if hail_cap is None else np.minimum(dbz, hail_cap)
    with np.errstate(invalid="ignore", over="ignore"):
        r = (10.0 ** (z / 10.0) / a) ** (1.0 / b)
        r = np.where(dbz >= min_dbz, r, 0.0)
    r[~np.isfinite(dbz)] = np.nan
    return r


def rainrate_to_dbz(r, a=200.0, b=1.6):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(a * np.asarray(r, float) ** b)


def accumulate(rain, dt_min):
    """ฝนสะสม (mm) จากอัตราฝน (mm/h) ทุก dt_min นาที (ไม่นับ NaN)."""
    return np.nansum(rain, axis=0) * dt_min / 60.0


def echo_area(cube, thresholds=(20, 30, 40, 50), px_km2=1.0, mask=None):
    """พื้นที่ (km²) ที่ค่า ≥ threshold ในแต่ละเวลา → dict {thr: array(t)}."""
    out = {}
    m = np.ones(cube.shape[-2:], bool) if mask is None else mask
    for thr in thresholds:
        out[thr] = np.array([np.nansum((c >= thr) & m) * px_km2 for c in cube])
    return out


def point_series(cube, meta, lon, lat, radius_km=5.0, stat="max"):
    """ค่าสถิติ (max/mean) ภายในรัศมีรอบจุด lon/lat ทุกเวลา (cube: t, y, x)."""
    x0, y0 = lonlat_to_xy(meta, lon, lat)
    X, Y = np.meshgrid(meta["x"], meta["y"])
    m = np.hypot(X - x0, Y - y0) <= radius_km * 1000.0
    f = {"max": np.nanmax, "mean": np.nanmean}[stat]
    sub = cube[..., m]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return f(sub, axis=-1)


def convective_stratiform(dbz, radius_km=11.0, conv_dbz=40.0, px_km=1.0):
    """แยกฝน convective / stratiform แบบย่อของ Steiner et al. (1995)

    pixel เป็น convective ถ้า (1) Z ≥ conv_dbz หรือ (2) Z สูงกว่าค่าพื้นหลังเฉลี่ย
    (รัศมี radius_km) มากพอ:  ΔZ = 10 − Z_bg² / 180  (0 ≤ Z_bg < 42.43 dBZ)
    คืนค่า array: 0 = ไม่มีฝน, 1 = stratiform, 2 = convective (NaN = นอกรัศมี)
    """
    from scipy.ndimage import convolve
    r = int(round(radius_km / px_km))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (np.hypot(yy, xx) <= r).astype(float)
    z = np.where(np.isfinite(dbz) & (dbz >= 15), 10 ** (dbz / 10), 0.0)
    has = np.isfinite(dbz) & (dbz >= 15)
    num = convolve(z, k, mode="constant")
    den = convolve(has.astype(float), k, mode="constant")
    with np.errstate(divide="ignore", invalid="ignore"):
        zbg = 10 * np.log10(num / den)
    dz = np.where(zbg < 0, 10.0, np.where(zbg < 42.43, 10 - zbg ** 2 / 180, 0.0))
    conv = has & ((dbz >= conv_dbz) | (dbz - zbg >= dz))
    out = np.where(has, 1.0, 0.0)
    out[conv] = 2.0
    out[~np.isfinite(dbz)] = np.nan
    return out


# ---------------------------------------------------------------------------
# เรขาคณิตลำคลื่นเรดาร์
# ---------------------------------------------------------------------------
def beam_height_km(ground_range_km, elev_deg, k=4.0 / 3.0):
    """ความสูงกึ่งกลางลำคลื่นเหนือเรดาร์ (km) ที่ระยะบนพื้น (km) แบบ 4/3 Earth (ประมาณ)."""
    ae = k * 6371.0
    r = np.asarray(ground_range_km, float)
    e = np.radians(elev_deg)
    return np.sqrt(r ** 2 + ae ** 2 + 2 * r * ae * np.sin(e)) - ae


def beam_table(range_km, elevations=ELEVATIONS_DEG, beamwidth=BEAMWIDTH_DEG):
    """ตาราง: ความสูงกึ่งกลาง/ล่าง/บนของลำคลื่นแต่ละมุมยกที่ระยะ range_km."""
    import pandas as pd
    rows = []
    for e in elevations:
        rows.append({"elev_deg": e,
                     "centre_km": float(beam_height_km(range_km, e)),
                     "bottom_km": float(beam_height_km(range_km, e - beamwidth / 2)),
                     "top_km": float(beam_height_km(range_km, e + beamwidth / 2)),
                     "beam_width_km": float(np.radians(beamwidth) * range_km)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Motion field
# ---------------------------------------------------------------------------
def to_dbz_input(cube, zerovalue=-15.0):
    """เตรียม dBZ สำหรับ optical flow: NaN → zerovalue (pysteps ไม่รับ NaN)."""
    return np.where(np.isfinite(cube), cube, zerovalue)


def speed_dir(V, mask, km_per_px=KM_PER_PX, dt_min=15.0):
    """ความเร็วมัธยฐาน (km/h) และทิศที่ไป (องศาจากเหนือ) ของ motion field ภายใน mask

    pysteps: V[0] = u (คอลัมน์, ไปทางตะวันออก), V[1] = v (แถว, **ไปทางใต้** เมื่อ yorigin = upper)
    ทิศที่ไป = atan2(u, −v)
    """
    u, v = V[0][mask], V[1][mask]
    if u.size == 0:
        return np.nan, np.nan
    spd = np.hypot(u, v) * km_per_px * 60.0 / dt_min
    ang = (np.degrees(np.arctan2(np.mean(u), -np.mean(v))) + 360) % 360
    return float(np.median(spd)), float(ang)


def compass(deg):
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
             "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((deg % 360) / 22.5 + 0.5) % 16]


def hovmoller(cube, meta, azimuth_deg, half_width_km=20.0, max_range_km=240.0, stat="max"):
    """Hovmöller diagram ตามแนวรัศมีจากเรดาร์ (ระยะ–เวลา)

    สุ่มค่าในแถบกว้าง ±half_width_km ตั้งฉากกับแนว azimuth_deg แล้วเอาค่าสูงสุด (หรือเฉลี่ย)
    คืนค่า (distance_km [n], H [t, n])
    """
    d = np.arange(0, max_range_km + 1, 1.0)
    w = np.arange(-half_width_km, half_width_km + 1, 1.0)
    a = np.radians(azimuth_deg)
    ux, uy = np.sin(a), np.cos(a)              # ทิศตามแนว
    px, py = np.cos(a), -np.sin(a)             # ทิศตั้งฉาก
    X = (d[:, None] * ux + w[None, :] * px) * 1000.0
    Y = (d[:, None] * uy + w[None, :] * py) * 1000.0
    col, row = xy_to_pixel(meta, X, Y)
    col, row = np.rint(col).astype(int), np.rint(row).astype(int)
    ok = (col >= 0) & (col < cube.shape[-1]) & (row >= 0) & (row < cube.shape[-2])
    col, row = np.clip(col, 0, cube.shape[-1] - 1), np.clip(row, 0, cube.shape[-2] - 1)
    samp = cube[:, row, col]                    # (t, n, w)
    samp = np.where(ok[None], samp, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        H = np.nanmax(samp, axis=2) if stat == "max" else np.nanmean(samp, axis=2)
    return d, H


def advection_interpolate(cube, meta, n_sub=3, fill_value=-15.0, motion_method="LK"):
    """เพิ่มความละเอียดเวลาด้วย advection interpolation (เช่น 15 → 5 นาที เมื่อ n_sub = 3)

    cube : (t, y, x) dBZ ช่วงเวลาเท่ากัน;  คืนค่า (cube_ใหม่, timestamps_ใหม่, motion_fields)
    """
    from pysteps import motion
    from scipy.ndimage import map_coordinates

    oflow = motion.get_method(motion_method)
    mask = np.all(np.isfinite(cube), axis=0)
    data = np.where(np.isfinite(cube), cube, fill_value)
    ny, nx = data.shape[1:]
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    dt_min = float(meta["accutime"])
    ts = list(meta["timestamps"])
    out, times, vels = [data[0]], [ts[0]], []
    for i in range(len(data) - 1):
        i0 = max(0, i - 1)
        V = oflow(data[i0:i + 2], verbose=False)
        vels.append(V)
        for k in range(1, n_sub):
            w = k / n_sub
            a = map_coordinates(data[i], (yy - w * V[1], xx - w * V[0]), order=1,
                                mode="constant", cval=fill_value)
            b = map_coordinates(data[i + 1], (yy + (1 - w) * V[1], xx + (1 - w) * V[0]),
                                order=1, mode="constant", cval=fill_value)
            # เฉลี่ยในหน่วย Z เชิงเส้น ไม่ใช่ dBZ
            zl = (1 - w) * 10 ** (a / 10) + w * 10 ** (b / 10)
            out.append(10 * np.log10(zl))
            times.append(ts[i] + _dt.timedelta(minutes=dt_min * w))
        out.append(data[i + 1])
        times.append(ts[i + 1])
    out = np.stack(out)
    out[:, ~mask] = np.nan
    return out, np.array(times), np.stack(vels)


# ---------------------------------------------------------------------------
# Verification ง่าย ๆ
# ---------------------------------------------------------------------------
def contingency(fct, obs, thr, mask=None):
    """คืนค่า dict ของ hits, misses, false alarms, POD, FAR, CSI ที่ threshold thr."""
    m = np.isfinite(fct) & np.isfinite(obs)
    if mask is not None:
        m &= mask
    f, o = fct[m] >= thr, obs[m] >= thr
    H, M, F = np.sum(f & o), np.sum(~f & o), np.sum(f & ~o)
    pod = H / (H + M) if H + M else np.nan
    far = F / (H + F) if H + F else np.nan
    csi = H / (H + M + F) if H + M + F else np.nan
    return {"hits": int(H), "misses": int(M), "false_alarms": int(F),
            "POD": pod, "FAR": far, "CSI": csi}


def cost_loss_expense(p_forecast, event, threshold, C=1.0, L=10.0):
    """ค่าใช้จ่ายเฉลี่ยของการตัดสินใจเตือนภัยเมื่อใช้ความน่าจะเป็น ≥ threshold

    เตือน = เสียค่าป้องกัน C ทุกครั้ง;  ไม่เตือนแต่เกิดเหตุ = เสียหาย L
    """
    warn = np.asarray(p_forecast) >= threshold
    ev = np.asarray(event).astype(bool)
    return float(np.mean(np.where(warn, C, np.where(ev, L, 0.0))))


# ---------------------------------------------------------------------------
# T-DaTing → ตาราง / GIS
# ---------------------------------------------------------------------------
def track_table(track_list, meta):
    """ตารางสรุปแต่ละ track: เวลาเริ่ม/จบ, อายุ, ระยะทาง, ความเร็ว, ทิศ, dBZ สูงสุด, พื้นที่สูงสุด."""
    import pandas as pd
    rows = []
    px_km = meta["xpixelsize"] / 1000.0
    for k, tr in enumerate(track_list):
        x, y = pixel_to_xy(meta, tr.cen_x.values, tr.cen_y.values)
        lon, lat = xy_to_lonlat(meta, x, y)
        t = pd.to_datetime(tr.time.values)
        life = (t[-1] - t[0]).total_seconds() / 60.0
        dist = np.sum(np.hypot(np.diff(x), np.diff(y))) / 1000.0
        dx, dy = x[-1] - x[0], y[-1] - y[0]
        rows.append({
            "track_id": k + 1,
            "start_utc": t[0], "end_utc": t[-1],
            "start_lon": float(np.atleast_1d(lon)[0]), "start_lat": float(np.atleast_1d(lat)[0]),
            "n_frames": len(tr), "lifetime_min": life, "path_km": dist,
            "speed_kmh": dist / (life / 60.0) if life > 0 else np.nan,
            "direction_to_deg": (np.degrees(np.arctan2(dx, dy)) + 360) % 360 if life > 0 else np.nan,
            "max_dbz": float(np.nanmax(tr.max_ref)),
            "max_area_km2": float(np.nanmax(tr.area)) * px_km ** 2,
        })
    return pd.DataFrame(rows)


def tracks_to_gdf(track_list, meta, to_wgs84=True):
    """คืนค่า (points_gdf, lines_gdf) — ถ้า to_wgs84 จะแปลงเป็น lon/lat (EPSG:4326)."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString, Point

    pts, lines = [], []
    px2 = (meta["xpixelsize"] / 1000) ** 2
    for k, tr in enumerate(track_list):
        x, y = pixel_to_xy(meta, tr.cen_x.values, tr.cen_y.values)
        for xi, yi, (_, row) in zip(x, y, tr.iterrows()):
            pts.append({"track_id": k + 1, "time_utc": pd.Timestamp(row.time).isoformat(),
                        "max_dbz": float(row.max_ref), "area_km2": float(row.area) * px2,
                        "geometry": Point(float(xi), float(yi))})
        if len(x) >= 2:
            lines.append({"track_id": k + 1,
                          "start_utc": pd.Timestamp(tr.time.iloc[0]).isoformat(),
                          "end_utc": pd.Timestamp(tr.time.iloc[-1]).isoformat(),
                          "max_dbz": float(np.nanmax(tr.max_ref)), "n_frames": len(tr),
                          "geometry": LineString(list(zip(x, y)))})
    p = gpd.GeoDataFrame(pts, geometry="geometry", crs=meta["projection"])
    l = gpd.GeoDataFrame(lines, geometry="geometry", crs=meta["projection"])
    if to_wgs84:
        p, l = p.to_crs("EPSG:4326"), l.to_crs("EPSG:4326")
    return p, l


def contour_xy(cont, meta):
    """contour ของ T-DaTing (list ของ array [row, col]) → list ของ (x, y) เมตร."""
    out = []
    for c in cont:
        c = np.asarray(c)
        out.append(pixel_to_xy(meta, c[:, 1], c[:, 0]))
    return out


def tracks_near_point(track_list, meta, lon, lat, radius_km=15.0):
    """track ที่ centroid ผ่านใกล้จุด (lon, lat) ภายใน radius_km → ตาราง (track_id, เวลา, ระยะ)."""
    import pandas as pd
    x0, y0 = lonlat_to_xy(meta, lon, lat)
    rows = []
    for k, tr in enumerate(track_list):
        x, y = pixel_to_xy(meta, tr.cen_x.values, tr.cen_y.values)
        d = np.hypot(x - x0, y - y0) / 1000.0
        i = int(np.argmin(d))
        if d[i] <= radius_km:
            rows.append({"track_id": k + 1, "closest_utc": pd.Timestamp(tr.time.iloc[i]),
                         "distance_km": float(d[i]), "max_dbz_track": float(np.nanmax(tr.max_ref)),
                         "max_dbz_at_closest": float(tr.max_ref.iloc[i])})
    return pd.DataFrame(rows, columns=["track_id", "closest_utc", "distance_km",
                                       "max_dbz_track", "max_dbz_at_closest"])
