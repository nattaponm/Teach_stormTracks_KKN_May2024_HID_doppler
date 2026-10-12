"""เปิดไฟล์เรดาร์ UF ของกรมอุตุนิยมวิทยาแบบทนทาน (ใช้ร่วมกันใน P0, P1, P2)

ทำไมไม่เรียก pyart.io.read_uf(path) ตรง ๆ:
Py-ART เปิดไฟล์ผ่าน fsspec ซึ่ง "เดา" ชนิดการบีบอัดจากนามสกุล (.bz2) ถ้าเนื้อไฟล์จริง
ไม่ใช่ bzip2 (เช่น gzip, zip หรือ UF ที่ไม่ได้บีบอัดแต่ตั้งชื่อ .bz2) จะได้ OSError: Invalid data stream
ฟังก์ชันนี้จึงอ่านไบต์แรก (magic bytes) แล้วเลือกวิธีแตกไฟล์ตามเนื้อจริง
"""
from __future__ import annotations

import bz2
import gzip
import io
import zipfile

MAGIC = {
    b"BZh": "bzip2",
    b"\x1f\x8b": "gzip",
    b"PK\x03\x04": "zip",
    b"Rar!": "rar",
    b"7z\xbc\xaf": "7z",
}


def detect(raw: bytes) -> str:
    for m, name in MAGIC.items():
        if raw.startswith(m):
            return name
    # UF: ระเบียนเริ่มด้วย "UF" หรือมีความยาวระเบียน 4 ไบต์นำหน้า (Fortran) แล้วตามด้วย "UF"
    if raw[:2] == b"UF" or raw[4:6] == b"UF":
        return "uf"
    return "unknown"


def _bz2_tolerant(raw: bytes) -> bytes:
    """แตก bzip2 หลายสตรีม และไม่สนข้อมูลขยะท้ายไฟล์ (เหมือนคำสั่ง bunzip2)"""
    out, data = [], raw
    while data:
        d = bz2.BZ2Decompressor()
        try:
            out.append(d.decompress(data))
        except OSError:
            if out:
                break
            raise
        if not d.eof:
            break
        data = d.unused_data
        if not data.startswith(b"BZh"):
            break
    return b"".join(out)


def uf_bytes(path) -> tuple[bytes, str]:
    """คืน (ไบต์ UF ที่แตกแล้ว, ชนิดการบีบอัดที่พบ)"""
    with open(path, "rb") as f:
        raw = f.read()
    kind = detect(raw)
    if kind == "bzip2":
        return _bz2_tolerant(raw), kind
    if kind == "gzip":
        return gzip.decompress(raw), kind
    if kind == "zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = max(z.infolist(), key=lambda i: i.file_size).filename
            return z.read(name), kind
    if kind == "rar":
        import rarfile  # pip install rarfile (และต้องมี unrar ในระบบ: apt-get install unrar)
        with rarfile.RarFile(io.BytesIO(raw)) as r:
            name = max(r.infolist(), key=lambda i: i.file_size).filename
            return r.read(name), kind
    if kind == "7z":
        import py7zr  # pip install py7zr
        with py7zr.SevenZipFile(io.BytesIO(raw)) as z:
            data = z.readall()
            return max(data.values(), key=lambda b: b.getbuffer().nbytes).getvalue(), kind
    return raw, kind  # uf หรือ unknown: ลองให้ Py-ART อ่านตรง ๆ


def read_uf(path, **kwargs):
    """อ่านไฟล์ UF (บีบอัดแบบใดก็ได้) เป็นวัตถุ Radar ของ Py-ART"""
    import pyart

    data, _ = uf_bytes(path)
    return pyart.io.read_uf(io.BytesIO(data), **kwargs)


def magic_hex(path, n=8) -> str:
    with open(path, "rb") as f:
        return f.read(n).hex(" ")
