"""ส่งต่อผลลัพธ์ระหว่าง notebook ผ่าน Google Drive และ commit ขึ้น GitHub อย่างปลอดภัย

ทำไมต้องมี
----------
Colab ให้ **เครื่องเสมือนแยกกันต่อหนึ่ง notebook** ไฟล์ที่ P0 เขียนลงโฟลเดอร์ repo (``/content/...``) จึงไม่มีใน
เครื่องของบท 01 หรือ P4 และหายไปเมื่อปิด runtime โมดูลนี้เก็บสำเนาของ "ไฟล์ที่จะเข้า repo" ไว้ใน Drive

    <drive_derived_root>/repo_mirror/   ← สำเนาของ data/inventory, data/reports, data/derived, ...

* ``pull(OUT_ROOT)``  ต้น notebook: Drive → โฟลเดอร์ repo (ไฟล์ใน Drive ชนะ เพราะเป็นผลล่าสุดของขั้นก่อนหน้า)
* ``push(OUT_ROOT)``  ท้าย notebook: โฟลเดอร์ repo → Drive (ไฟล์ที่เพิ่งสร้างชนะ)
* ``git_push(...)``   commit + push ขึ้น GitHub ด้วย token จาก Colab Secrets — **ไม่พิมพ์ token ออกมาทุกกรณี**

คัดลอกเฉพาะไฟล์ที่เนื้อหาเปลี่ยน (เทียบ sha256) จึงเร็วและรันซ้ำได้ ไม่มีไฟล์ UF อยู่ในรายการ
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from .config import REPO_ROOT

# ไฟล์/โฟลเดอร์ที่ prep สร้างแล้วต้องเข้า repo (ห้ามใส่ข้อมูลเรดาร์ดิบ)
ITEMS = ("config/radar_site.json", "config/data_sources.yaml",
         "data/inventory", "data/reports", "data/sounding", "data/derived", "data/days",
         "data/gis/dem_kkn_1km.tif", "data/gis/dem_kkn_1km_max.tif")
MIRROR = "repo_mirror"
SKIP_SUFFIX = (".uf", ".bz2", ".gz.part", ".pyc")

GITIGNORE = """# ข้อมูลเรดาร์ดิบ — ห้าม commit (ลิขสิทธิ์ กรมอุตุนิยมวิทยา)
*.uf
*.uf.bz2
*.uf.gz

# ผลลัพธ์ชั่วคราว
/figures/
__pycache__/
.ipynb_checkpoints/
*.pyc
ref_*.nc
data/cache/
"""


def _sha(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def _files(base: Path, item: str):
    p = base / item
    if p.is_file():
        yield item
    elif p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.name != ".gitkeep" and not f.name.endswith(SKIP_SUFFIX):
                yield str(f.relative_to(base))


def _copy(src_base: Path, dst_base: Path, items=ITEMS, verbose=True):
    changed = []
    for item in items:
        for rel in _files(src_base, item):
            s, d = src_base / rel, dst_base / rel
            if d.exists() and d.stat().st_size == s.stat().st_size and _sha(d) == _sha(s):
                continue
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
            changed.append(rel)
    if verbose:
        mb = sum((dst_base / r).stat().st_size for r in changed) / 1e6
        print(f"  คัดลอก {len(changed)} ไฟล์ ({mb:.1f} MB)" + (": " + ", ".join(changed[:8]) + (" …" if len(changed) > 8 else "")
                                                            if changed else ""))
    return changed


def mirror_dir(out_root) -> Path:
    return Path(out_root) / MIRROR


def pull(out_root, items=ITEMS, verbose=True):
    """ต้น notebook: ดึงผลของขั้นก่อนหน้าจาก Drive (repo_mirror) มาใส่โฟลเดอร์ repo ของเครื่องนี้"""
    m = mirror_dir(out_root)
    if not m.exists():
        if verbose:
            print(f"sync.pull: ยังไม่มี {m} (ครั้งแรก) — ข้าม")
        return []
    if verbose:
        print(f"sync.pull: {m} → {REPO_ROOT}")
    return _copy(m, REPO_ROOT, items, verbose)


def push(out_root, items=ITEMS, verbose=True):
    """ท้าย notebook: เก็บผลที่จะเข้า repo ไว้ใน Drive (repo_mirror) ให้ notebook อื่น/runtime ใหม่ใช้ต่อ"""
    m = mirror_dir(out_root)
    m.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"sync.push: {REPO_ROOT} → {m}")
    return _copy(REPO_ROOT, m, items, verbose)


def status(out_root, items=ITEMS):
    """ตารางเปรียบเทียบไฟล์ระหว่าง repo ในเครื่องนี้กับ repo_mirror ใน Drive"""
    import pandas as pd
    m = mirror_dir(out_root)
    rows = {}
    for base, col in ((REPO_ROOT, "repo"), (m, "drive")):
        for item in items:
            for rel in _files(base, item):
                rows.setdefault(rel, {})[col] = _sha(base / rel)[:10]
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=["repo", "drive"])
    df["same"] = df["repo"] == df["drive"]
    return df.sort_index()


# ----------------------------------------------------------------------------------------------------------
# GitHub
# ----------------------------------------------------------------------------------------------------------
def ensure_gitignore():
    p = REPO_ROOT / ".gitignore"
    if not p.exists() or "*.uf.bz2" not in p.read_text(encoding="utf-8"):
        p.write_text(GITIGNORE, encoding="utf-8")
        print("สร้าง/แก้ .gitignore")


def git_push(message, paths=ITEMS, secret_name="KKN_commit", owner="nattaponm",
             repo="Teach_stormTracks_KKN_May2024_HID_doppler", branch="main",
             user_name="nattaponm", user_email=None, dry_run=False):
    """commit ``paths`` แล้ว push ขึ้น GitHub ด้วย fine-grained token ใน Colab Secrets (สิทธิ์ Contents: Read and write)

    ทุกข้อความที่พิมพ์ผ่านตัวกรองที่แทน token ด้วย *** และไม่พิมพ์ URL ที่มี token
    dry_run=True: แสดงเฉพาะรายการไฟล์ที่จะ commit
    """
    from google.colab import userdata  # ใช้ได้เฉพาะบน Colab
    token = userdata.get(secret_name)
    url = f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"

    def clean(text):
        return (text or "").replace(url, "<remote>").replace(token, "***") if token else (text or "")

    def git(*a, check=True, show=True):
        """รัน git แล้วคืนข้อความที่ **กรอง token แล้ว** เท่านั้น (ไม่คืน object ดิบ)"""
        r = subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True)
        out = clean(r.stdout + r.stderr).strip()
        if show:
            print(f"$ git {a[0]}  (rc={r.returncode})" + (f"\n{out}" if out else ""))
        if check and r.returncode != 0:
            raise RuntimeError(f"git {a[0]} ล้มเหลว — ดูข้อความด้านบน")
        return out

    ensure_gitignore()
    git("config", "user.name", user_name, show=False)
    # เครื่อง Colab ไม่มีอีเมลผู้ commit → ใช้อีเมล noreply ของ GitHub (ไม่เปิดเผยอีเมลจริงใน commit สาธารณะ)
    git("config", "user.email", user_email or f"{user_name}@users.noreply.github.com", show=False)
    existing = [p for p in (".gitignore",) + tuple(paths) if (REPO_ROOT / p).exists()]
    git("add", "--", *existing)
    st = git("diff", "--cached", "--name-status", show=False)      # เฉพาะไฟล์ที่ staged จริง
    print("ไฟล์ที่จะ commit:\n" + (st if st else "(ไม่มีการเปลี่ยนแปลง)"))
    if dry_run or not st:
        return
    git("commit", "-q", "-m", message)
    git("pull", "--rebase", "--autostash", "-q", url, branch)     # autostash: มีไฟล์อื่นแก้ค้างก็ไม่ล้ม
    git("push", "-q", url, f"HEAD:{branch}")
    print(f"✅ push เสร็จ → https://github.com/{owner}/{repo}/tree/{branch}")
