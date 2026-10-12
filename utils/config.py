"""อ่านค่าตั้งต้นของโครงการจาก config/settings.yaml (และ radar_site.json ถ้ามี)."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts) -> Path:
    """คืน path ภายใน repo ไม่ว่าจะรัน notebook จากโฟลเดอร์ใด"""
    return REPO_ROOT.joinpath(*parts)


def load_settings(path: str | Path | None = None) -> dict:
    path = Path(path) if path else repo_path("config", "settings.yaml")
    with open(path, encoding="utf-8") as f:
        s = yaml.safe_load(f)
    # พิกัดเรดาร์จริงที่ P0 อ่านจาก UF จะเขียนทับค่าโดยประมาณ
    site = repo_path("config", "radar_site.json")
    if site.exists():
        with open(site, encoding="utf-8") as f:
            real = json.load(f)
        for k in ("lat", "lon", "alt_m"):
            if real.get(k) is not None:
                s["radar"][k] = float(real[k])
        s["radar"]["site_source"] = "uf_header"
    else:
        s["radar"]["site_source"] = "approximate"
    return s
