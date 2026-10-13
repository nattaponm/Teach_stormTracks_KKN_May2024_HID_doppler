"""ตัวโหลดข้อมูลสำหรับนิสิต — ไม่ต้องรู้ว่าไฟล์อยู่ใน repo หรือบน Zenodo

ตาราง/กริดสรุปทั้งเดือนอยู่ใน ``data/derived/`` ของ repo เสมอ
กริดรายวัน (CAPPI, polar, labels) ของ "วันหลัก" อยู่ใน ``data/days/YYYYMMDD/``
วันอื่นดาวน์โหลดจาก Zenodo ตาม ``config/data_sources.yaml`` แล้วเก็บไว้ใน ``data/cache/``
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd
import yaml

from .config import repo_path

KINDS = ("cappi_15min", "polar_15min", "labels_15min", "labels_10min_linZ", "labels_05min_linZ")


def sources():
    p = repo_path("config", "data_sources.yaml")
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def table(name):
    """ตารางทั้งเดือนใน data/derived เช่น 'tracks_15min.csv', 'cells_05min_linZ.csv.gz', 'footprints_15min.gpkg'"""
    p = repo_path("data", "derived", name)
    if name.endswith(".gpkg"):
        import geopandas as gpd
        return gpd.read_file(p)
    parse = [c for c in ("day", "day_local", "time", "start", "end", "slot_utc", "date") if c in pd.read_csv(p, nrows=0).columns]
    return pd.read_csv(p, parse_dates=parse)


def day_index():
    p = repo_path("data", "derived", "day_index.csv")
    return pd.read_csv(p, parse_dates=["day"]) if p.exists() else pd.DataFrame(columns=["day", "in_repo", "in_zenodo"])


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def day_file(day, kind="cappi_15min"):
    """path ของไฟล์รายวัน (ดาวน์โหลดจาก Zenodo อัตโนมัติถ้าไม่มีใน repo)"""
    assert kind in KINDS, f"kind ต้องเป็นหนึ่งใน {KINDS}"
    d = pd.Timestamp(day)
    name = f"KKN_{d:%Y%m%d}_{kind}.nc"
    local = repo_path("data", "days", f"{d:%Y%m%d}", name)
    if local.exists():
        return local
    cache = repo_path("data", "cache", name)
    if cache.exists():
        return cache
    src = sources()
    files = (src.get("zenodo") or {}).get("files", {})
    rec = (src.get("zenodo") or {}).get("record_id")
    if not rec or name not in files:
        raise FileNotFoundError(f"{name} ไม่มีใน repo และยังไม่ได้ระบุบน Zenodo (config/data_sources.yaml)")
    base = "https://sandbox.zenodo.org" if src["zenodo"].get("sandbox") else "https://zenodo.org"
    url = f"{base}/records/{rec}/files/{name}?download=1"
    cache.parent.mkdir(parents=True, exist_ok=True)
    print(f"ดาวน์โหลด {name} จาก Zenodo ({files[name].get('mb', '?')} MB) ...")
    urllib.request.urlretrieve(url, cache)
    want = files[name].get("sha256")
    if want and _sha256(cache) != want:
        cache.unlink()
        raise IOError(f"checksum ของ {name} ไม่ตรง — ลองใหม่อีกครั้ง")
    return cache


def load_cappi(day):
    from .data_io import load_pcappi
    return load_pcappi(str(day_file(day, "cappi_15min")))


def load_polar(day):
    from .track_tools import load_polar_grid
    return load_polar_grid(str(day_file(day, "polar_15min")))


def load_labels(day, tag="15min"):
    from .track_tools import load_labels
    return load_labels(str(day_file(day, f"labels_{tag}")))


def terrain():
    from .track_tools import load_terrain
    return load_terrain(str(repo_path("data", "derived", "terrain_1km.nc")))
