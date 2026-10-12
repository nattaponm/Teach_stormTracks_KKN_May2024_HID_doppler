"""มาตรฐานแผนที่และกราฟสำหรับตีพิมพ์ (ใช้ร่วมกันทุก notebook)

องค์ประกอบ: ฟอนต์ไทย (Sarabun), ภูมิประเทศแบบ hillshade, วงระยะเรดาร์, เส้นกริดละติจูด–ลองจิจูด,
มาตราส่วน, ลูกศรทิศเหนือ, แผนที่ย่อตำแหน่ง และการบันทึกรูป PNG 300 dpi + PDF เวกเตอร์
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LightSource, LinearSegmentedColormap
import matplotlib.patheffects as pe

from .config import repo_path
from . import geo

HALO = [pe.withStroke(linewidth=1.8, foreground="white", alpha=0.85)]
TIER_COLORS = {"น้อย": "#fdcc8a", "กลาง": "#fc8d59", "มาก": "#b30000"}
PHASE_COLORS = {"outbreak": "#c2410c", "late_pre": "#ca8a04", "rainy": "#0369a1"}
TERRAIN = LinearSegmentedColormap.from_list(
    "kkn_terrain", ["#e8eedf", "#d9e2c3", "#c9c39a", "#b39b74", "#8c7a63", "#6b6159"])


def setup_style(base_size=9):
    """ตั้งฟอนต์ไทยและรูปแบบรูปมาตรฐานวารสาร"""
    for ttf in repo_path("assets", "fonts").glob("*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    mpl.rcParams.update({
        "font.family": ["Sarabun", "DejaVu Sans"],      # DejaVu เป็น fallback สำหรับอักขระพิเศษ (⁻¹, ², …)
        "font.size": base_size,
        "axes.titlesize": base_size + 1,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,          # ฝังฟอนต์ใน PDF ให้แก้ไขได้ใน Illustrator/Inkscape
        "axes.unicode_minus": False,
    })


def savefig(fig, name, outdir=None):
    outdir = Path(outdir or repo_path("figures"))
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}")
    print(f"บันทึกรูป → {outdir / name}.png / .pdf")


# ----------------------------------------------------------------- องค์ประกอบแผนที่

def add_hillshade(ax, z, extent, alpha=1.0, vmax=1200):
    """ภูมิประเทศแบบสีตามความสูง + เงาภูเขา (แสงจากทิศตะวันตกเฉียงเหนือ)"""
    if z is None:
        return None
    ls = LightSource(azdeg=315, altdeg=45)
    zf = np.where(np.isfinite(z), z, 0.0)
    rgb = ls.shade(zf, cmap=TERRAIN, blend_mode="soft", vert_exag=8, dx=1000, dy=1000, vmin=0, vmax=vmax)
    rgb[~np.isfinite(z)] = 1.0
    l, r, b, t = extent
    return ax.imshow(rgb, extent=(l / 1e3, r / 1e3, b / 1e3, t / 1e3), origin="upper", alpha=alpha, zorder=0)


def terrain_colorbar(fig, ax, vmax=1200, label="ความสูง (ม. รทก.)", **kw):
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, vmax), cmap=TERRAIN)
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02, **kw)
    cb.set_label(label)
    cb.outline.set_linewidth(0.5)
    return cb


def plot_km(gdf, ax, **kw):
    """วาด GeoDataFrame (AEQD, เมตร) บนแกนหน่วย km"""
    g = gdf.copy()
    g["geometry"] = g.geometry.scale(1e-3, 1e-3, origin=(0, 0))
    out = g.plot(ax=ax, **kw)
    ax.set_xlabel(""); ax.set_ylabel("")   # geopandas ใส่ชื่อแกนตาม CRS ให้อัตโนมัติ — ไม่ต้องการบนแผนที่
    return out


def add_range_rings(ax, s, rings_km=None, color="0.25", label=True, name=True):
    rings_km = rings_km or s["geometry"]["range_rings_km"]
    th = np.linspace(0, 2 * np.pi, 361)
    for r in rings_km:
        lw = 1.0 if r == s["radar"]["max_range_km"] else 0.5
        ax.plot(r * np.sin(th), r * np.cos(th), color=color, lw=lw, ls="-" if lw > 0.6 else "--", zorder=12)
        if label:
            a = np.radians(325)   # มุมป้ายวงระยะ: ซ้ายบน ไม่ทับแผนที่ย่อและคำอธิบายสัญลักษณ์
            ax.text(r * np.sin(a), r * np.cos(a), f"{r} km", fontsize=7, color=color, ha="center", va="center",
                    zorder=13, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
    ax.plot(0, 0, marker="^", ms=7, mfc="k", mec="white", mew=0.8, zorder=14)
    if name:
        ax.annotate(s["radar"]["name"], (0, 0), xytext=(5, -10), textcoords="offset points", fontsize=8,
                    fontweight="bold", zorder=14, path_effects=HALO)


def add_graticule(ax, s, step_deg=1.0, color="0.5", labels=True):
    """เส้นละติจูด–ลองจิจูดบนแผนที่ AEQD พร้อมป้ายกำกับที่ขอบแกน"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xs = np.array([x0, x1, x1, x0]) * 1e3
    ys = np.array([y0, y0, y1, y1]) * 1e3
    lon_c, lat_c = geo.xy_to_lonlat(xs, ys, s)
    lons = np.arange(np.floor(min(lon_c)), np.ceil(max(lon_c)) + step_deg, step_deg)
    lats = np.arange(np.floor(min(lat_c)), np.ceil(max(lat_c)) + step_deg, step_deg)
    t = np.linspace(0, 1, 200)
    xt, xl, yt, yl = [], [], [], []
    for lo in lons:
        la = lats.min() - 1 + t * (lats.max() - lats.min() + 2)
        x, y = geo.lonlat_to_xy(np.full_like(la, lo), la, s)
        x, y = np.asarray(x) / 1e3, np.asarray(y) / 1e3
        ax.plot(x, y, color=color, lw=0.3, ls=":", zorder=4)
        xb = np.interp(y0, y, x)  # จุดตัดขอบล่าง
        if x0 < xb < x1:
            xt.append(xb); xl.append(f"{lo:.0f}°E")
    for la in lats:
        lo = lons.min() - 1 + t * (lons.max() - lons.min() + 2)
        x, y = geo.lonlat_to_xy(lo, np.full_like(lo, la), s)
        x, y = np.asarray(x) / 1e3, np.asarray(y) / 1e3
        ax.plot(x, y, color=color, lw=0.3, ls=":", zorder=4)
        yb = np.interp(x0, x, y)  # จุดตัดขอบซ้าย
        if y0 < yb < y1:
            yt.append(yb); yl.append(f"{la:.0f}°N")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    if labels:
        ax.set_xticks(xt, xl)
        ax.set_yticks(yt, yl, rotation=90, va="center")
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(length=2, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(True)


def add_scalebar(ax, length_km=100, loc=(0.05, 0.05), height_frac=0.008):
    """มาตราส่วนแบบแถบขาว–ดำ (หน่วย km) — แผนที่ AEQD ระยะถูกต้องจากศูนย์กลาง"""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bx = x0 + loc[0] * (x1 - x0)
    by = y0 + loc[1] * (y1 - y0)
    h = height_frac * (y1 - y0)
    half = length_km / 2
    ax.add_patch(plt.Rectangle((bx, by), half, h, fc="k", ec="k", lw=0.5, zorder=20))
    ax.add_patch(plt.Rectangle((bx + half, by), half, h, fc="white", ec="k", lw=0.5, zorder=20))
    for xv, txt in [(bx, "0"), (bx + half, f"{half:g}"), (bx + length_km, f"{length_km:g} km")]:
        ax.text(xv, by + 2.2 * h, txt, ha="center", va="bottom", fontsize=7, zorder=20, path_effects=HALO)


def add_north_arrow(ax, loc=(0.05, 0.90), size=0.06):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + loc[0] * (x1 - x0)
    y = y0 + loc[1] * (y1 - y0)
    L = size * (y1 - y0)
    ax.annotate("", xy=(x, y + L / 2), xytext=(x, y - L / 2),
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7", lw=1.2, color="k"), zorder=20)
    ax.text(x, y + L / 2 + 0.01 * (y1 - y0), "N", ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=20)


def add_locator_inset(ax, s, bounds=(0.75, 0.635, 0.25, 0.365)):
    """แผนที่ย่อประเทศไทยแสดงตำแหน่งเรดาร์และรัศมี 240 km"""
    ins = ax.inset_axes(bounds)
    ins.set_zorder(40)
    th = geo.load_admin("province_th", to_radar=False)
    th.plot(ax=ins, color="0.88", edgecolor="0.6", lw=0.15)
    disk = geo.coverage_disk(s).to_crs(4326)
    disk.boundary.plot(ax=ins, color="#b30000", lw=0.8)
    ins.plot(s["radar"]["lon"], s["radar"]["lat"], "^", ms=3, color="k")
    ins.set_xlim(97.2, 106.0)
    ins.set_ylim(5.5, 20.6)
    ins.set_xticks([]); ins.set_yticks([]); ins.set_xlabel(""); ins.set_ylabel("")
    ins.set_facecolor("white")
    for sp in ins.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)
    return ins


def base_map(ax, s, dem=None, extent_km=255, provinces=True, rings=True, province_labels=True,
             graticule=True, scalebar=True, north=True, dem_alpha=1.0):
    """แผนที่ฐาน: DEM + ขอบจังหวัด + วงระยะ + กริดพิกัด + มาตราส่วน + ทิศเหนือ"""
    z, ext = dem if dem is not None else (None, None)
    add_hillshade(ax, z, ext, alpha=dem_alpha)
    if provinces:
        prov = geo.load_admin("province", s)
        plot_km(prov, ax, facecolor="none", edgecolor="0.3", lw=0.5, zorder=9)  # เหนือชั้นข้อมูลที่ระบายสี
        if province_labels:
            for _, r in prov.iterrows():
                p = r.geometry.representative_point()
                if np.hypot(p.x, p.y) / 1e3 < extent_km - 10:
                    ax.text(p.x / 1e3, p.y / 1e3, r.pro_th, fontsize=6.5, color="0.2", ha="center",
                            va="center", zorder=11, path_effects=HALO)
    ax.set_xlim(-extent_km, extent_km)
    ax.set_ylim(-extent_km, extent_km)
    ax.set_aspect("equal")
    if rings:
        add_range_rings(ax, s)
    if graticule:
        add_graticule(ax, s)
    if scalebar:
        add_scalebar(ax)
    if north:
        add_north_arrow(ax)
    return ax
