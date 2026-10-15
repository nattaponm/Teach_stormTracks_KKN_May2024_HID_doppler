"""ตัวโหลดข้อมูลสำหรับนิสิต — ไม่ต้องรู้ว่าไฟล์อยู่ใน repo หรือบน Zenodo

ตาราง/กริดสรุปทั้งเดือนอยู่ใน ``data/derived/`` ของ repo เสมอ
กริดรายวัน (CAPPI, polar, labels) ของ "วันหลัก" อยู่ใน ``data/days/YYYYMMDD/``
วันอื่นดาวน์โหลดจาก Zenodo ตาม ``config/data_sources.yaml`` แล้วเก็บไว้ใน ``data/cache/``

Zenodo รับได้ไม่เกิน 100 ไฟล์ต่อ record จึงเก็บ CAPPI/polar เป็นไฟล์รายวัน (61 ไฟล์)
และรวม labels ของแต่ละ Δt ไว้ใน zip เดียว (3 ไฟล์) — ``day_file`` แตก zip ให้อัตโนมัติ
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
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
    zen = sources().get("zenodo") or {}
    rec, files, archives = zen.get("record_id"), zen.get("files") or {}, zen.get("archives") or {}
    arc = next((a for a, m in archives.items() if name in (m.get("members") or [])), None)
    if not rec or (name not in files and arc is None):
        raise FileNotFoundError(f"{name} ไม่มีใน repo และยังไม่ได้ระบุบน Zenodo (config/data_sources.yaml)")
    base = "https://sandbox.zenodo.org" if zen.get("sandbox") else "https://zenodo.org"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if name in files:
        _download(f"{base}/records/{rec}/files/{name}?download=1", cache, files[name])
        return cache
    # ไฟล์อยู่ใน zip: ดาวน์โหลดทั้ง zip ครั้งเดียว แล้วแตกทุกวันลง data/cache/
    zpath = repo_path("data", "cache", arc)
    if not zpath.exists():
        _download(f"{base}/records/{rec}/files/{arc}?download=1", zpath, archives[arc])
    with zipfile.ZipFile(zpath) as z:
        z.extractall(cache.parent)
    want = (archives[arc].get("member_sha256") or {}).get(name)
    if want and _sha256(cache) != want:
        cache.unlink()
        raise IOError(f"checksum ของ {name} ใน {arc} ไม่ตรง — ลบ data/cache/{arc} แล้วลองใหม่")
    return cache


def _download(url, dst, meta):
    print(f"ดาวน์โหลด {dst.name} จาก Zenodo ({meta.get('mb', '?')} MB) ...")
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    want = meta.get("sha256")
    if want and _sha256(tmp) != want:
        tmp.unlink()
        raise IOError(f"checksum ของ {dst.name} ไม่ตรง — ลองใหม่อีกครั้ง")
    tmp.rename(dst)


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
