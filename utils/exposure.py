"""P5: ข้อมูลการสัมผัสภัย (GHSL) และการบังลำคลื่นเรดาร์ (beam blockage) บนกริด 1 km ของ KKN

1) GHSL (Global Human Settlement Layer, JRC, CC BY 4.0) ความละเอียด 1 km (Mollweide, EPSG:54009)
   * GHS_BUILT_S  = พื้นที่สิ่งปลูกสร้าง (m² ต่อเซลล์) ปี 2020
   * GHS_POP      = ประชากร (คนต่อเซลล์) ปี 2020
   แปลงเข้ากริด AEQD 1 km ของเรดาร์ด้วย resampling แบบเฉลี่ย (เซลล์ 1 km² ทั้งสองระบบ → ค่าต่อเซลล์คงเดิมโดยประมาณ)

2) Beam blockage fraction (BBF) ของมุมยกต่ำสุด (Bech et al. 2003, J. Atmos. Oceanic Technol.)
   ลำคลื่นหน้าตัดวงกลมรัศมี a = r·tan(θ_bw/2) ศูนย์กลางสูง h(r) (สมการ 4/3 รัศมีโลก)
   y = z_terrain − h(r)
   BBF = [ y√(a²−y²) + a² asin(y/a) + πa²/2 ] / (πa²)   (−a < y < a), 0 ถ้า y ≤ −a, 1 ถ้า y ≥ a
   BBF สะสม = ค่าสูงสุดตามระยะจากเรดาร์ (สิ่งที่บังใกล้กว่าบังไกลทั้งหมด)
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

GHSL = {
    "built_s": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/"
               "GHS_BUILT_S_E2020_GLOBE_R2023A_54009_1000/V1-0/GHS_BUILT_S_E2020_GLOBE_R2023A_54009_1000_V1_0.zip",
    "pop": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/"
           "GHS_POP_E2020_GLOBE_R2023A_54009_1000/V1-0/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.zip",
}
GHSL_CITATION = ("Pesaresi M., Politis P. (2023): GHS-BUILT-S R2023A; Schiavina M. et al. (2023): GHS-POP R2023A. "
                 "European Commission, Joint Research Centre (JRC). CC BY 4.0")


# ======================================================================
# 1) GHSL
# ======================================================================
def fetch_ghsl(kind, cache_dir):
    """ดาวน์โหลดไฟล์ zip ทั้งโลก 1 km (ครั้งเดียว เก็บใน cache_dir เช่น Drive) → path ของ zip"""
    import urllib.request
    url = GHSL[kind]
    out = Path(cache_dir) / url.rsplit("/", 1)[-1]
    if out.exists() and out.stat().st_size > 1e6:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    print(f"ดาวน์โหลด {out.name} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(out)
    return out


def grid_profile(meta):
    """กริดวิเคราะห์ (แถวแรก = เหนือ) จาก metadata ของ load_pcappi → (transform, crs, shape)"""
    from rasterio.transform import from_origin
    return (from_origin(meta["x1"], meta["y2"], meta["xpixelsize"], meta["ypixelsize"]), meta["projection"],
            (len(meta["y"]), len(meta["x"])))


def ghsl_to_grid(zip_path, meta):
    """อ่าน GeoTIFF ใน zip แล้ว reproject เข้ากริด AEQD ของเรดาร์ (average) → array (แถวแรก = เหนือ)"""
    import rasterio
    from rasterio.warp import Resampling, reproject
    tif = [n for n in zipfile.ZipFile(zip_path).namelist() if n.lower().endswith(".tif")][0]
    transform, crs, shape = grid_profile(meta)
    dst = np.full(shape, np.nan, np.float32)
    with rasterio.open(f"zip://{zip_path}!/{tif}") as src:
        reproject(rasterio.band(src, 1), dst, dst_transform=transform, dst_crs=crs,
                  src_nodata=src.nodata, dst_nodata=np.nan, resampling=Resampling.average)
    return np.where(dst < 0, np.nan, dst)


def zonal_sum_polygons(grid, meta, gdf, id_col):
    """ผลรวมของกริดภายในแต่ละ polygon (ทับซ้อนกันได้ เช่น footprint ของเส้นทาง) → Series ตาม id_col"""
    import rasterio.features
    transform, crs, shape = grid_profile(meta)
    g = gdf.to_crs(crs)
    x0, y0, dx = meta["x1"], meta["y2"], meta["xpixelsize"]
    v = np.nan_to_num(np.asarray(grid, float))
    out = {}
    for k, geom in zip(g[id_col], g.geometry):
        if geom is None or geom.is_empty:
            out[k] = np.nan
            continue
        minx, miny, maxx, maxy = geom.bounds
        c0, c1 = max(0, int((minx - x0) // dx)), min(shape[1], int((maxx - x0) // dx) + 1)
        r0, r1 = max(0, int((y0 - maxy) // dx)), min(shape[0], int((y0 - miny) // dx) + 1)
        if c1 <= c0 or r1 <= r0:
            out[k] = 0.0
            continue
        from rasterio.transform import from_origin
        m = rasterio.features.rasterize([(geom, 1)], out_shape=(r1 - r0, c1 - c0),
                                        transform=from_origin(x0 + c0 * dx, y0 - r0 * dx, dx, dx), fill=0,
                                        all_touched=False, dtype="uint8")
        out[k] = float((v[r0:r1, c0:c1] * m).sum())
    return pd.Series(out, name="sum")


# ======================================================================
# 2) Beam blockage
# ======================================================================
def bbf_partial(y, a):
    """สัดส่วนพื้นที่วงกลมรัศมี a ที่อยู่ใต้ระดับ y (วัดจากศูนย์กลาง) — สมการ Bech et al. (2003)"""
    y = np.clip(y, -a, a)
    return (y * np.sqrt(np.maximum(a ** 2 - y ** 2, 0)) + a ** 2 * np.arcsin(y / a) + np.pi * a ** 2 / 2) / (np.pi * a ** 2)


def beam_blockage(dem, dem_transform, antenna_alt_m, elev_deg=0.5, beamwidth_deg=1.0, rmax_km=240.0,
                  dr_km=0.25, daz_deg=0.25, ke=4.0 / 3.0):
    """BBF สะสมตามระยะของมุมยกหนึ่ง → (az [deg], r [km], bbf[az, r])

    dem: array (แถวแรก = เหนือ) ในระบบ AEQD ศูนย์กลางที่เรดาร์ (เมตร), dem_transform: affine ของ DEM
    """
    from scipy.ndimage import map_coordinates
    az = np.arange(0, 360, daz_deg)
    r = np.arange(dr_km, rmax_km + dr_km / 2, dr_km)
    A, Rr = np.meshgrid(np.radians(az), r * 1000, indexing="ij")
    Re = 6371000.0 * ke
    el = np.radians(elev_deg)
    s = Rr                                                     # ระยะตามพื้นผิว ≈ slant range (มุมต่ำ)
    h = np.sqrt(s ** 2 + Re ** 2 + 2 * s * Re * np.sin(el)) - Re + antenna_alt_m
    x, y = s * np.sin(A), s * np.cos(A)
    col = (x - dem_transform.c) / dem_transform.a - 0.5
    row = (y - dem_transform.f) / dem_transform.e - 0.5
    z = map_coordinates(np.nan_to_num(dem, nan=0.0), [row.ravel(), col.ravel()], order=1, mode="nearest").reshape(A.shape)
    a = s * np.tan(np.radians(beamwidth_deg) / 2)
    bbf = bbf_partial(z - h, a)
    return az, r, np.maximum.accumulate(bbf, axis=1)


def polar_to_grid(az, r, val, meta):
    """ค่าบนพิกัด polar → กริด (แถวแรก = เหนือ) แบบ nearest"""
    X, Y = np.meshgrid(meta["x"] / 1000, meta["y"] / 1000)
    R = np.hypot(X, Y)
    AZ = np.degrees(np.arctan2(X, Y)) % 360
    ia = np.clip(np.round(AZ / (az[1] - az[0])).astype(int) % len(az), 0, len(az) - 1)
    ir = np.clip(np.round((R - r[0]) / (r[1] - r[0])).astype(int), 0, len(r) - 1)
    out = val[ia, ir]
    return np.where(R <= r[-1], out, np.nan)


def azimuth_profile(grid, meta, r_min=80, r_max=200, bin_deg=5):
    """ค่าเฉลี่ยของกริดตามมุม (ในวงระยะ) หารด้วยค่ามัธยฐานทุกมุม → ใช้ตรวจการบังลำคลื่นจากข้อมูลจริง"""
    X, Y = np.meshgrid(meta["x"] / 1000, meta["y"] / 1000)
    R, AZ = np.hypot(X, Y), np.degrees(np.arctan2(X, Y)) % 360
    m = (R >= r_min) & (R <= r_max) & np.isfinite(grid)
    s = pd.Series(np.asarray(grid)[m]).groupby((AZ[m] // bin_deg * bin_deg).astype(int)).mean()
    return (s / s.median()).rename("relative")
