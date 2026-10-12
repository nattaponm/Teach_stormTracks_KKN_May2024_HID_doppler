"""
data_io.py — อ่าน/เขียนข้อมูลเรดาร์ KKN ที่ใช้ในรายวิชา
=========================================================

ข้อมูลหลัก ``data/KKN_20240503/KKN_20240503_pcappi2km_dbz.nc`` ถูกเตรียมโดย
``notebooks/00_prepare_pcappi_for_pysteps.ipynb`` (อ่านไฟล์ UF ดิบ → QC → pseudo-CAPPI 2 km)

* ตัวแปร ``reflectivity(time, y, x)`` หน่วย dBZ
    - ค่า echo = dBZ
    - ในรัศมีเรดาร์แต่ไม่มี echo = ``zerovalue`` (−15 dBZ)
    - นอกรัศมี หรือเวลาที่ไม่มีไฟล์ = NaN
* กริด Cartesian 1 km (481 × 481) ระบบพิกัด Azimuthal Equidistant ที่มีเรดาร์เป็นศูนย์กลาง
* แกนเวลาทุก 15 นาที (00:00–23:45 UTC) ตัวแปร ``time_available`` บอกว่าเวลาใดมีข้อมูล

ฟังก์ชัน ``load_pcappi`` คืนค่า ``(Z, metadata)`` ในรูปแบบเดียวกับ ``pysteps.io.read_timeseries``
โดย **พลิกแกน y ให้ทิศเหนืออยู่ด้านบน (yorigin = "upper")** ซึ่งเป็นค่าตั้งต้นของ pysteps
"""

import datetime as _dt
import os as _os

import numpy as np

DEFAULT_FILE = _os.path.join("KKN_20240503", "KKN_20240503_pcappi2km_dbz.nc")

_PYSTEPS_KEYS = ("projection", "x1", "x2", "y1", "y2", "xpixelsize", "ypixelsize",
                 "cartesian_unit", "unit", "transform", "accutime", "threshold",
                 "zerovalue", "institution")


def load_pcappi(path, start=None, end=None, available_only=False):
    """อ่าน pseudo-CAPPI → (Z [t, y, x] dBZ, metadata)

    Parameters
    ----------
    path : str
        ไฟล์ NetCDF หรือโฟลเดอร์ ``data/`` (จะใช้ไฟล์มาตรฐานของเคสนี้)
    start, end : str หรือ datetime, optional
        ช่วงเวลา UTC เช่น ``"2024-05-03 06:00"`` (รวมปลายทั้งสองด้าน)
    available_only : bool
        ``True`` = ตัดเวลาที่ไม่มีไฟล์ออก (ระวัง: ช่วงเวลาจะไม่เท่ากันอีกต่อไป)

    metadata ที่เพิ่มจาก pysteps
    ---------------------------
    ``timestamps`` (array ของ datetime), ``available`` (bool ต่อเวลา),
    ``coverage`` (2D bool: อยู่ในรัศมีเรดาร์), ``source_height_m`` (ความสูงของข้อมูลที่ใช้),
    ``lon``, ``lat`` (2D), ``x``, ``y`` (จุดกึ่งกลาง pixel, เมตร), ``radar_lon``, ``radar_lat``
    """
    import xarray as xr

    if _os.path.isdir(path):
        path = _os.path.join(path, DEFAULT_FILE)
    with xr.open_dataset(path) as ds:
        if start is not None or end is not None:
            ds = ds.sel(time=slice(start, end))
        Z = ds["reflectivity"].values.astype("float64")
        attrs = dict(ds.attrs)
        times = [t.to_pydatetime() for t in _pd_to_datetime(ds["time"].values)]
        available = ds["time_available"].values.astype(bool)
        cov = ds["coverage_mask"].values.astype(bool)
        src_h = ds["source_height_m"].values.astype(float)
        lon, lat = ds["lon"].values, ds["lat"].values
        x, y = ds["x"].values.astype(float), ds["y"].values.astype(float)

    # ไฟล์เก็บแบบ y เพิ่มขึ้นไปทางเหนือ → พลิกให้แถวแรกอยู่ทิศเหนือ (yorigin = "upper")
    if y[0] < y[-1]:
        Z, cov, src_h, lon, lat, y = (Z[:, ::-1], cov[::-1], src_h[::-1],
                                      lon[::-1], lat[::-1], y[::-1])

    if available_only:
        Z, times = Z[available], [t for t, a in zip(times, available) if a]
        available = available[available]

    meta = {k: attrs[k] for k in _PYSTEPS_KEYS if k in attrs}
    for k in ("x1", "x2", "y1", "y2", "xpixelsize", "ypixelsize", "accutime",
              "threshold", "zerovalue"):
        meta[k] = float(meta[k])
    meta.update({
        "yorigin": "upper",
        "timestamps": np.array(times),
        "available": np.asarray(available),
        "coverage": np.ascontiguousarray(cov),
        "source_height_m": np.ascontiguousarray(src_h),
        "lon": np.ascontiguousarray(lon), "lat": np.ascontiguousarray(lat),
        "x": x, "y": np.ascontiguousarray(y),
        "radar_lon": float(attrs["radar_longitude"]),
        "radar_lat": float(attrs["radar_latitude"]),
        "radar_alt_m": float(attrs.get("radar_altitude_m", np.nan)),
        "radar": attrs.get("radar", "KKN"),
    })
    return np.ascontiguousarray(Z), meta


def _pd_to_datetime(values):
    import pandas as pd
    return pd.to_datetime(values)


def subset_meta(meta, idx):
    """ตัด metadata ให้ตรงกับ index เวลา (เช่น ``Z[idx]``)."""
    m = dict(meta)
    m["timestamps"] = np.asarray(meta["timestamps"])[idx]
    m["available"] = np.asarray(meta["available"])[idx]
    return m


def time_index(meta, when):
    """หา index ของเวลา ``when`` (str 'HH:MM' UTC หรือ datetime) ใน metadata."""
    ts = list(meta["timestamps"])
    if isinstance(when, str):
        hh, mm = map(int, when.split(":"))
        when = ts[0].replace(hour=hh, minute=mm)
    return ts.index(when)


def save_geotiff(path, bands, meta, band_names=None, nodata=-9999.0):
    """บันทึกอาร์เรย์ (band, y, x) เป็น GeoTIFF ที่มีระบบพิกัด AEQD ของเรดาร์ (เปิดใน QGIS ได้)."""
    import rasterio
    from rasterio.transform import from_origin

    bands = np.asarray(bands, dtype="float32")
    if bands.ndim == 2:
        bands = bands[None]
    tr = from_origin(meta["x1"], meta["y2"], meta["xpixelsize"], meta["ypixelsize"])
    with rasterio.open(path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2],
                       count=bands.shape[0], dtype="float32", crs=meta["projection"],
                       transform=tr, nodata=nodata, compress="deflate") as dst:
        dst.write(np.where(np.isfinite(bands), bands, nodata))
        for i, n in enumerate(band_names or [], start=1):
            dst.set_band_description(i, str(n))
    return path


def load_polar(path):
    """อ่านไฟล์ polar ขนาดเล็ก (CfRadial) ด้วย Py-ART — ใช้ในส่วนเสริม Doppler ของ NB03."""
    import pyart
    return pyart.io.read_cfradial(path)
