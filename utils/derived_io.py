"""เขียนไฟล์ข้อมูลที่ประมวลผลแล้ว (derived) — รูปแบบเข้ากันได้กับ repo pysteps และ repo3

ไฟล์รายวัน (กริด AEQD 1 km 481 × 481 ศูนย์กลางเรดาร์)
- ``cappi/KKN_YYYYMMDD_cappi_15min.nc``  pseudo-CAPPI 2 km + Z สูงสุดในคอลัมน์ 96 ช่อง (รูปแบบ repo pysteps:
  y เพิ่มไปทางเหนือ; ``utils.data_io.load_pcappi`` พลิกให้แถวแรกเป็นทิศเหนือเอง)
- ``polar/KKN_YYYYMMDD_polar_15min.nc``  ผลิตภัณฑ์ polar บนกริด (รูปแบบ repo3: แถวแรก = ทิศเหนือ,
  ``track_tools.load_polar_grid`` อ่านได้ทันที) เฉพาะช่องเวลาที่มีพายุ
- ``tracks/KKN_YYYYMMDD_labels_<tag>.nc``  label cube ของ T-DaTing (ค่า = track_id + 1)
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import radar_prep as RP


def aeqd_proj4(lon0, lat0):
    return f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


def _crs_attrs(site):
    from pyproj import CRS
    return {"grid_mapping_name": "azimuthal_equidistant",
            "longitude_of_projection_origin": site["lon"], "latitude_of_projection_origin": site["lat"],
            "false_easting": 0.0, "false_northing": 0.0,
            "crs_wkt": CRS.from_proj4(aeqd_proj4(site["lon"], site["lat"])).to_wkt()}


def _write(ds, path, enc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    ds.to_netcdf(tmp, encoding=enc, engine="netcdf4")
    os.replace(tmp, path)
    return path


def i16(scale, chunks=None):
    e = {"dtype": "int16", "scale_factor": scale, "add_offset": 0.0, "_FillValue": np.int16(-32768),
         "zlib": True, "complevel": 5, "shuffle": True}
    if chunks:
        e["chunksizes"] = chunks
    return e


def lonlat_grid(site, ascending=True):
    from pyproj import Transformer
    x, y = RP.grid_axes()
    if not ascending:
        y = y[::-1]
    X, Y = np.meshgrid(x, y)
    tr = Transformer.from_crs(aeqd_proj4(site["lon"], site["lat"]), "EPSG:4326", always_xy=True)
    LON, LAT = tr.transform(X, Y)
    return x, y, LON.astype(np.float32), LAT.astype(np.float32)


# ----------------------------------------------------------------------------- P1
def write_cappi_day(path, slots, available, dbz, zcol, src_h, site, scan_times, nyquist):
    """ไฟล์ pseudo-CAPPI รายวัน ตัวแปรและ attribute เดียวกับ repo pysteps (+ zmax_column, nyquist)"""
    x, y, LON, LAT = lonlat_grid(site, ascending=True)
    ny, nx = dbz.shape[1:]
    res = RP.GRID_RES_M
    cov = np.isfinite(src_h)
    ds = xr.Dataset(
        {
            "reflectivity": (("time", "y", "x"), dbz, {
                "long_name": f"pseudo-CAPPI {RP.PCAPPI_HEIGHT_M / 1000:.1f} km equivalent reflectivity factor (gentle base-QC)",
                "standard_name": "equivalent_reflectivity_factor", "units": "dBZ", "grid_mapping": "crs",
                "coordinates": "lon lat",
                "comment": f"{RP.ZEROVALUE_DBZ} dBZ = inside radar coverage but no echo; NaN = outside coverage or missing volume"}),
            "zmax_column": (("time", "y", "x"), zcol, {
                "long_name": "maximum reflectivity in the column (all sweeps, gentle base-QC, nearest-gate)",
                "units": "dBZ", "grid_mapping": "crs"}),
            "source_height_m": (("y", "x"), src_h, {"long_name": "height above radar of the data used at each pixel", "units": "m"}),
            "coverage_mask": (("y", "x"), cov.astype(np.uint8), {"long_name": "radar coverage (1 = covered)", "units": "1"}),
            "time_available": (("time",), np.asarray(available, np.uint8), {"long_name": "1 = radar volume available for this slot"}),
            "volume_start_time": (("time",), pd.to_datetime(scan_times)),
            "nyquist_velocity": (("time",), np.asarray(nyquist, np.float32), {"units": "m s-1"}),
            "lon": (("y", "x"), LON, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": (("y", "x"), LAT, {"standard_name": "latitude", "units": "degrees_north"}),
            "crs": ((), np.int32(0), _crs_attrs(site)),
        },
        coords={"time": pd.to_datetime(slots),
                "x": ("x", x.astype(np.float32), {"standard_name": "projection_x_coordinate", "units": "m"}),
                "y": ("y", y.astype(np.float32), {"standard_name": "projection_y_coordinate", "units": "m"})},
        attrs={
            "title": "KKN pseudo-CAPPI 2.0 km reflectivity (pysteps format) + column-max reflectivity",
            "Conventions": "CF-1.8", "radar": "KKN", "radar_latitude": site["lat"], "radar_longitude": site["lon"],
            "radar_altitude_m": site["alt"], "pcappi_height_m_above_radar": RP.PCAPPI_HEIGHT_M,
            "method": ("pseudo-CAPPI: vertical interpolation in linear Z between the sweeps bracketing the target height; "
                       "lowest sweep beyond, highest sweep inside; nearest gate horizontally; 4/3 effective Earth radius"),
            "qc": "gentle base-QC (same thresholds as Teach_Pysteps_KKN_summerStorm)",
            "time_convention": "UTC; slot time = nominal 15-min time of the raw file",
            "source": "TMD KKN UF volumes", "processing_code": RP.PREP_VERSION,
            "history": f"created {_dt.datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}",
            "projection": aeqd_proj4(site["lon"], site["lat"]),
            "x1": float(x[0] - res / 2), "x2": float(x[-1] + res / 2), "y1": float(y[0] - res / 2), "y2": float(y[-1] + res / 2),
            "xpixelsize": res, "ypixelsize": res, "cartesian_unit": "m", "yorigin": "lower", "unit": "dBZ",
            "transform": "dB", "accutime": 15.0, "timestep": 15.0, "threshold": RP.PCAPPI_MIN_DBZ,
            "zerovalue": RP.ZEROVALUE_DBZ, "institution": "Thai Meteorological Department (radar data)",
        })
    enc = {"reflectivity": i16(0.1, (1, ny, nx)), "zmax_column": i16(0.1, (1, ny, nx)),
           "source_height_m": i16(1.0), "coverage_mask": {"zlib": True, "complevel": 5},
           "lon": {"zlib": True, "complevel": 5, "least_significant_digit": 4},
           "lat": {"zlib": True, "complevel": 5, "least_significant_digit": 4},
           "time": {"units": "minutes since 1970-01-01 00:00:00", "dtype": "int64"},
           "volume_start_time": {"units": "seconds since 1970-01-01 00:00:00", "dtype": "float64"}}
    return _write(ds, path, enc)


# ----------------------------------------------------------------------------- P2
def write_polar_day(path, times, grids, site, attrs=None):
    """ผลิตภัณฑ์ polar บนกริด (repo3: แถวแรก = ทิศเหนือ) — float ถูกบีบอัดเป็น int16, HID เป็น int8"""
    x, y, _, _ = lonlat_grid(site, ascending=False)
    ny, nx = len(y), len(x)
    dv, enc = {}, {"time": {"units": "minutes since 1970-01-01 00:00:00", "dtype": "int64"}}
    for k, v in grids.items():
        v = np.asarray(v)
        if k in RP.POLAR_ENCODING:
            units, scale, desc = RP.POLAR_ENCODING[k]
            dv[k] = (("time", "y", "x"), v.astype(np.float32), {"units": units, "long_name": desc})
            enc[k] = i16(scale, (1, ny, nx))
        elif k.startswith("hid_s"):
            dv[k] = (("time", "y", "x"), v.astype(np.int8), {
                "long_name": f"hydrometeor class, sweep {k[-1]} (Besic et al. 2016, Py-ART)",
                "flag_values": np.arange(10, dtype=np.int8), "flag_meanings": "NC AG CR LR RP RN VI WS MH IH/HDG"})
            enc[k] = {"zlib": True, "complevel": 5, "chunksizes": (1, ny, nx)}
        else:
            dv[k] = (("time", "y", "x"), v.astype(np.uint8), {"long_name": {
                "hail_col": "1 = MH or IH/HDG with Z >= 40 dBZ at any sweep of the column",
                "graupel_col": "1 = RP (graupel) at any sweep of the column"}.get(k, k)})
            enc[k] = {"zlib": True, "complevel": 5, "chunksizes": (1, ny, nx)}
    ds = xr.Dataset(dv, coords={"time": pd.to_datetime(list(times)), "y": y.astype(np.float64), "x": x.astype(np.float64)},
                    attrs={"title": "KKN polar products on the 1-km grid (repo3 layout)",
                           "projection": aeqd_proj4(site["lon"], site["lat"]), "yorigin": "upper",
                           "hid_classes": "NC,AG,CR,LR,RP,RN,VI,WS,MH,IH/HDG", "processing_code": RP.PREP_VERSION,
                           "processing": "radar_prep.process_volume_kkn = polar_tools.process_volume with fixed PhiDP sign (-1)",
                           "history": f"created {_dt.datetime.utcnow():%Y-%m-%dT%H:%M:%SZ}", **(attrs or {})})
    return _write(ds, path, enc)


# ----------------------------------------------------------------------------- P3
def write_labels(path, labels, times, observed, meta, attrs=None):
    ds = xr.Dataset({"label": (("time", "y", "x"), labels.astype(np.int32)),
                     "observed": ("time", np.asarray(observed, np.uint8))},
                    coords={"time": pd.to_datetime(list(times)), "y": meta["y"], "x": meta["x"]},
                    attrs={"projection": meta["projection"], "yorigin": "upper",
                           "label_meaning": "track_id + 1 (0 = no cell); track_id = episode*10000 + k, unique within the month", **(attrs or {})})
    return _write(ds, path, {"label": {"zlib": True, "complevel": 5, "chunksizes": (1,) + labels.shape[1:]},
                             "time": {"units": "minutes since 1970-01-01 00:00:00", "dtype": "int64"}})


def write_grids_2d(path, grids, meta, attrs=None):
    """กริด 2 มิติ (แถวแรก = ทิศเหนือ) เช่น ภูมิประเทศ หรือสรุปทั้งเดือน"""
    ds = xr.Dataset({k: (("y", "x"), np.asarray(v, np.float32)) for k, v in grids.items()},
                    coords={"y": meta["y"], "x": meta["x"]},
                    attrs={"projection": meta["projection"], "yorigin": "upper", **(attrs or {})})
    return _write(ds, path, {k: {"zlib": True, "complevel": 5, "least_significant_digit": 3} for k in grids})
