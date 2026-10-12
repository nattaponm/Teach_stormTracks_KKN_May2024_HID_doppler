"""เรขาคณิตเรดาร์ ขอบเขตการปกครอง และ DEM

ระบบพิกัดหลักของโครงการคือ Azimuthal Equidistant (AEQD) ที่มีศูนย์กลางที่เรดาร์
หน่วยเป็นเมตร: ระยะจากจุดศูนย์กลางถึงทุกจุดบนแผนที่ถูกต้อง จึงเหมาะกับวงระยะเรดาร์
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import Point

from .config import repo_path

EARTH_R_M = 6_371_000.0
ADMIN_GPKG = "data/gis/admin_kkn.gpkg"
ADMIN_SOURCE = "https://raw.githubusercontent.com/chingchai/OpenGISData-Thailand/master/{name}.geojson"


# ----------------------------------------------------------------- พิกัด

def radar_crs(s: dict) -> CRS:
    r = s["radar"]
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={r['lat']} +lon_0={r['lon']} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )


def lonlat_to_xy(lon, lat, s):
    t = Transformer.from_crs("EPSG:4326", radar_crs(s), always_xy=True)
    return t.transform(lon, lat)


def xy_to_lonlat(x, y, s):
    t = Transformer.from_crs(radar_crs(s), "EPSG:4326", always_xy=True)
    return t.transform(x, y)


def range_azimuth(x_m, y_m):
    """ระยะ (km) และมุมทิศ (องศาจากทิศเหนือ ตามเข็มนาฬิกา) จากพิกัด AEQD"""
    x_m, y_m = np.asarray(x_m, float), np.asarray(y_m, float)
    r_km = np.hypot(x_m, y_m) / 1000.0
    az = (np.degrees(np.arctan2(x_m, y_m)) + 360.0) % 360.0
    return r_km, az


def beam_height_km(r_km, elev_deg, antenna_alt_m=0.0, ke=4.0 / 3.0):
    """ความสูงศูนย์กลางลำคลื่นเหนือระดับทะเล (km) ตามแบบจำลองโลก 4/3 (Doviak & Zrnić 1993)

    h = sqrt(r² + (ke·R)² + 2·r·ke·R·sin θ) − ke·R + h0
    """
    r = np.asarray(r_km, float) * 1000.0
    kr = ke * EARTH_R_M
    th = np.radians(elev_deg)
    h = np.sqrt(r**2 + kr**2 + 2 * r * kr * np.sin(th)) - kr
    return (h + antenna_alt_m) / 1000.0


def beam_width_km(r_km, beamwidth_deg=1.0):
    """ความกว้างลำคลื่น (km) ≈ r·Δθ"""
    return np.asarray(r_km, float) * np.radians(beamwidth_deg)


def range_rings(s, rings_km=None):
    """วงระยะเป็น GeoDataFrame ใน AEQD"""
    rings_km = rings_km or s["geometry"]["range_rings_km"]
    geoms = [Point(0, 0).buffer(r * 1000.0, resolution=128).exterior for r in rings_km]
    return gpd.GeoDataFrame({"range_km": rings_km}, geometry=geoms, crs=radar_crs(s))


def coverage_disk(s, radius_km=None):
    radius_km = radius_km or s["radar"]["max_range_km"]
    return gpd.GeoDataFrame(
        {"range_km": [radius_km]}, geometry=[Point(0, 0).buffer(radius_km * 1000.0, resolution=256)],
        crs=radar_crs(s),
    )


# ----------------------------------------------------------------- ขอบเขตการปกครอง

def build_admin(s, out=None, buffer_km=None):
    """ดาวน์โหลดขอบเขตจังหวัด/อำเภอ/ตำบล (OpenGISData-Thailand, CC BY 4.0)
    แล้วตัดเฉพาะที่อยู่ในรัศมี buffer_km จากเรดาร์ เก็บเป็น GeoPackage 3 ชั้น
    ใช้ครั้งเดียวตอนสร้าง repo — ผลถูกเก็บไว้ใน data/gis แล้ว"""
    out = Path(out or repo_path(ADMIN_GPKG))
    buffer_km = buffer_km or s["geometry"]["admin_buffer_km"]
    disk = coverage_disk(s, buffer_km).to_crs(4326).geometry.iloc[0]
    for name, layer in [("provinces", "province"), ("districts", "district"), ("subdistricts", "subdistrict")]:
        g = gpd.read_file(ADMIN_SOURCE.format(name=name))
        if layer == "province":
            # จังหวัดเก็บทั้งประเทศ (ทำให้เรียบ) สำหรับแผนที่ย่อตำแหน่ง
            g_all = g.copy()
            g_all["geometry"] = g_all.geometry.simplify(0.01, preserve_topology=True)
            g_all.to_file(out, layer="province_th", driver="GPKG")
        g = g[g.intersects(disk)].copy()
        g["geometry"] = g.geometry.simplify(0.0005, preserve_topology=True)  # ~50 m
        g.to_file(out, layer=layer, driver="GPKG")
    return out


def load_admin(layer="district", s=None, to_radar=True):
    """layer: province | district | subdistrict | province_th"""
    g = gpd.read_file(repo_path(ADMIN_GPKG), layer=layer)
    if to_radar and s is not None:
        g = g.to_crs(radar_crs(s))
    return g


# ----------------------------------------------------------------- DEM

COP_URL = (
    "/vsicurl/https://copernicus-dem-90m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)


def dem_grid_profile(s, res_km=None, half_km=None):
    """โปรไฟล์ rasterio ของกริด AEQD ศูนย์กลางเรดาร์ (ขอบเซลล์ตรงกับกริดเรดาร์)"""
    from rasterio.transform import from_origin

    res = (res_km or s["grid"]["res_km"]) * 1000.0
    half = (half_km or s["grid"]["half_width_km"]) * 1000.0
    n = int(round(2 * half / res)) + 1
    transform = from_origin(-half - res / 2, half + res / 2, res, res)
    return dict(driver="GTiff", width=n, height=n, count=1, dtype="float32", crs=radar_crs(s).to_wkt(),
                transform=transform, nodata=np.nan, compress="deflate", predictor=3)


def build_dem(s, out=None, url_template=COP_URL, half_km=260):
    """สร้าง DEM 1 km ในพิกัด AEQD จาก Copernicus GLO-90 (AWS Open Data, ไม่ต้องลงทะเบียน)
    ค่าในเซลล์ 1 km = ค่าเฉลี่ยของจุด 90 m ทั้งหมดในเซลล์ (resampling แบบ average)"""
    import rasterio
    from rasterio.warp import reproject, Resampling

    out = Path(out or repo_path(s["dem"]["out"]))
    prof = dem_grid_profile(s, half_km=half_km)
    dst = np.full((prof["height"], prof["width"]), np.nan, np.float32)
    lon, lat = xy_to_lonlat(np.array([-1, 1, 1, -1]) * half_km * 1e3, np.array([-1, -1, 1, 1]) * half_km * 1e3, s)
    lats = range(int(np.floor(min(lat))), int(np.floor(max(lat))) + 1)
    lons = range(int(np.floor(min(lon))), int(np.floor(max(lon))) + 1)
    n_ok = 0
    for la in lats:
        for lo in lons:
            url = url_template.format(ns="N" if la >= 0 else "S", lat=abs(la), ew="E" if lo >= 0 else "W", lon=abs(lo))
            try:
                with rasterio.open(url) as src:
                    tmp = np.full_like(dst, np.nan)
                    reproject(rasterio.band(src, 1), tmp, dst_transform=prof["transform"], dst_crs=prof["crs"],
                              dst_nodata=np.nan, resampling=Resampling.average)
            except Exception as e:  # ไทล์ที่ไม่มี (เช่น ทะเล) หรือเครือข่ายขัดข้อง
                print(f"  ข้ามไทล์ {la},{lo}: {type(e).__name__}")
                continue
            fill = np.isnan(dst) & ~np.isnan(tmp)
            dst[fill] = tmp[fill]
            n_ok += 1
    if n_ok == 0:
        raise RuntimeError("ดาวน์โหลด DEM ไม่ได้เลย — ตรวจการเชื่อมต่ออินเทอร์เน็ต")
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **prof) as f:
        f.write(dst, 1)
    print(f"บันทึก DEM {prof['width']}×{prof['height']} จาก {n_ok} ไทล์ → {out}")
    return out


def load_dem(s, build_if_missing=True):
    """คืน (array, extent_m) ของ DEM ใน AEQD หรือ (None, None) ถ้าไม่มีและสร้างไม่ได้"""
    import rasterio

    path = s["dem"].get("path_override") or repo_path(s["dem"]["out"])
    path = Path(path)
    if not path.exists() and build_if_missing:
        try:
            print("ยังไม่มี DEM ในเครื่อง — กำลังสร้างจาก Copernicus GLO-90 (ครั้งแรกใช้เวลา 2–5 นาที)")
            build_dem(s, out=path)
        except Exception as e:
            print(f"สร้าง DEM ไม่สำเร็จ ({e}) — แผนที่จะแสดงโดยไม่มีภูมิประเทศ")
            return None, None
    if not path.exists():
        return None, None
    with rasterio.open(path) as f:
        z = f.read(1).astype(float)
        b = f.bounds
    return z, (b.left, b.right, b.bottom, b.top)


def zonal_stats(z, extent, geoms, funcs=("mean", "max")):
    """สถิติแบบโซน (zonal statistics) ของ raster ภายในแต่ละ polygon (พิกัด AEQD เดียวกัน)
    เซลล์ที่แตะขอบ polygon ถูกนับด้วย (all_touched) เพื่อให้อำเภอเล็กมีค่าเสมอ"""
    import pandas as pd
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds

    if z is None:
        return pd.DataFrame({f: [np.nan] * len(geoms) for f in funcs})
    left, right, bottom, top = extent
    tr = from_bounds(left, bottom, right, top, z.shape[1], z.shape[0])
    rows = []
    for g in geoms:
        if g is None or g.is_empty:
            rows.append({f: np.nan for f in funcs}); continue
        m = ~geometry_mask([g], out_shape=z.shape, transform=tr, all_touched=True)
        v = z[m]
        v = v[np.isfinite(v)]
        rows.append({f: (getattr(np, f)(v) if v.size else np.nan) for f in funcs})
    return pd.DataFrame(rows)


def sample_dem(z, extent, x_m, y_m):
    """ค่า DEM (เซลล์ใกล้สุด) ที่ตำแหน่ง x, y (เมตร, AEQD)"""
    if z is None:
        return np.full(np.shape(x_m), np.nan)
    left, right, bottom, top = extent
    ny, nx = z.shape
    col = np.clip(((np.asarray(x_m) - left) / (right - left) * nx).astype(int), 0, nx - 1)
    row = np.clip(((top - np.asarray(y_m)) / (top - bottom) * ny).astype(int), 0, ny - 1)
    return z[row, col]
