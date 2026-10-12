"""ฟังก์ชันวิเคราะห์ที่ใช้ร่วมกันในบทเรียน 02–06

หมวด
----
1. โหลดข้อมูลทั้งเดือน + จัดกลุ่ม     : month_tracks, month_cells, add_phase, life_classes
2. สถิติที่คำนึงว่า "วัน" เป็นหน่วยอิสระ  : kruskal_by_day_permutation, day_block_summary, effect_table
3. จับคู่เส้นทางกับรายงานความเสียหาย   : match_tracks_reports
4. สถิติเชิงพื้นที่บนขอบเขตการปกครอง    : zonal_by_polygon, queen_weights, morans_i, getis_ord_gi
5. แผนที่กริดบนแกน km                    : show_grid
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import load_settings

PHASE_ORDER = ["outbreak", "late_pre", "rainy"]


# ======================================================================
# 1) ข้อมูลทั้งเดือน
# ======================================================================
def add_phase(df, col="day", s=None):
    s = s or load_settings()
    d = pd.to_datetime(df[col])
    out = pd.Series(index=df.index, dtype=object)
    for p in s["phases"]:
        out[(d >= p["start"]) & (d <= p["end"])] = p["key"]
    df = df.copy()
    df["phase"] = pd.Categorical(out, categories=PHASE_ORDER, ordered=True)
    return df


def month_tracks(tag="15min", min_life=30):
    from .data import table
    T = table(f"tracks_{tag}.csv")
    T = add_phase(T)
    if min_life:
        T = T[T["lifetime_min"] >= min_life].copy()
    return T.reset_index(drop=True)


def month_cells(tag="15min"):
    from .data import table
    return add_phase(table(f"cells_{tag}.csv.gz"))


def life_classes(T, edges=(0, 45, 75, 1e4), labels=("สั้น", "กลาง", "ยาว")):
    """จัดกลุ่มอายุเส้นทาง (นาที): ค่าตั้งต้น สั้น < 45, กลาง 45–<75, ยาว ≥ 75

    อายุจาก T-DaTing เป็นจำนวนเท่าของ Δt (15 นาทีมีค่า 30, 45, 60, …) การแบ่งด้วย tercile จึงมักได้ขอบซ้ำกัน
    จึงใช้ขอบคงที่ที่ทำให้แต่ละกลุ่มมีขนาดพอทดสอบได้ในชุดข้อมูลนี้ (ปรับ edges ได้)
    """
    T = T.copy()
    T["life_class"] = pd.cut(T["lifetime_min"], edges, labels=list(labels), right=False)
    return T


# ======================================================================
# 2) สถิติ
# ======================================================================
def kruskal_by_day_permutation(df, value, group, strata="day", n=2000, seed=0):
    """Kruskal–Wallis H + p-value จากการสลับป้ายกลุ่ม **ภายในวันเดียวกัน**

    เส้นทางในวันเดียวกันเกิดในบรรยากาศเดียวกัน จึงไม่เป็นอิสระต่อกัน การสลับป้ายภายในวัน (stratified
    permutation) รักษาโครงสร้าง "ผลของวัน" ไว้ p-value จึงไม่ต่ำเกินจริงแบบ Kruskal–Wallis ปกติ
    """
    from scipy.stats import kruskal
    d = df[[value, group, strata]].dropna()
    groups = [g for g in d[group].unique()]
    if len(groups) < 2:
        return {"H": np.nan, "p_classic": np.nan, "p_perm_within_day": np.nan, "n": len(d)}

    def H(lbl):
        parts = [d[value].values[lbl == g] for g in groups]
        parts = [p for p in parts if len(p) > 0]
        return kruskal(*parts).statistic if len(parts) >= 2 else np.nan

    lab = d[group].values
    h0, p0 = kruskal(*[d.loc[d[group] == g, value] for g in groups])
    rng = np.random.default_rng(seed)
    idx = [np.where(d[strata].values == k)[0] for k in d[strata].unique()]
    sims = np.empty(n)
    for i in range(n):
        lb = lab.copy()
        for ix in idx:
            lb[ix] = rng.permutation(lb[ix])
        sims[i] = H(lb)
    return {"H": float(h0), "p_classic": float(p0), "p_perm_within_day": float((np.sum(sims >= h0) + 1) / (n + 1)),
            "n": len(d), "n_days": d[strata].nunique()}


def day_block_summary(df, value, group, strata="day", stat="median"):
    """สรุปค่าของแต่ละกลุ่มเป็นรายวัน (หนึ่งค่า/วัน/กลุ่ม) — ใช้เปรียบเทียบเมื่อมีหลายวัน"""
    return df.groupby([strata, group], observed=True)[value].agg(stat).unstack(group)


def effect_table(df, values, group, order=None):
    """ตารางค่ามัธยฐานรายกลุ่ม + Kruskal–Wallis (classic และ permutation ภายในวัน) + Cliff's δ กลุ่มแรก vs สุดท้าย"""
    from .track_tools import cliffs_delta
    if order is None:
        order = list(df[group].cat.categories) if hasattr(df[group], "cat") else sorted(df[group].dropna().unique())
    rows = []
    for v in values:
        r = {"ตัวแปร": v}
        for g in order:
            r[f"มัธยฐาน {g}"] = df.loc[df[group] == g, v].median()
        k = kruskal_by_day_permutation(df, v, group)
        r.update({"H": k["H"], "p (classic)": k["p_classic"], "p (สลับในวัน)": k["p_perm_within_day"],
                  f"Cliff δ ({order[-1]} vs {order[0]})": cliffs_delta(df.loc[df[group] == order[-1], v],
                                                                         df.loc[df[group] == order[0], v])})
        rows.append(r)
    return pd.DataFrame(rows).set_index("ตัวแปร")


# ======================================================================
# 3) เส้นทาง ↔ รายงานความเสียหาย
# ======================================================================
def match_tracks_reports(footprints, tracks, reports, buffer_km=0.0):
    """จับคู่: footprint ของเส้นทางซ้อนทับ polygon อำเภอ **และ** ช่วงเวลาเส้นทางซ้อนทับหน้าต่างเวลารายงาน

    footprints : GeoDataFrame (AEQD) คอลัมน์ uid
    tracks     : DataFrame คอลัมน์ uid, start, end
    reports    : GeoDataFrame อำเภอ (CRS ใดก็ได้) คอลัมน์ report_id, win_start_utc, win_end_utc
    คืน DataFrame คู่ (report_id, uid, overlap_km2)
    """
    import geopandas as gpd
    rep = reports.to_crs(footprints.crs).copy()
    if buffer_km:
        rep["geometry"] = rep.buffer(buffer_km * 1000)
    fp = footprints.merge(tracks[["uid", "start", "end"]], on="uid", how="inner")
    pairs = gpd.sjoin(fp[["uid", "start", "end", "geometry"]], rep[["report_id", "win_start_utc", "win_end_utc", "geometry"]],
                      how="inner", predicate="intersects")
    t_ok = (pairs["start"] <= pairs["win_end_utc"]) & (pairs["end"] >= pairs["win_start_utc"])
    pairs = pairs[t_ok].copy()
    if len(pairs):
        geo_r = rep.set_index("report_id").geometry
        pairs["overlap_km2"] = [g.intersection(geo_r.loc[r]).area / 1e6 for g, r in zip(pairs.geometry, pairs["report_id"])]
    return pd.DataFrame(pairs.drop(columns=["geometry", "index_right"], errors="ignore"))


# ======================================================================
# 4) สถิติเชิงพื้นที่
# ======================================================================
def zonal_by_polygon(grid, meta, polys, id_col, stats=("mean", "max", "sum", "count")):
    """zonal statistics ของกริด (แถวแรก = ทิศเหนือ) ภายในแต่ละ polygon (CRS ใดก็ได้) → DataFrame"""
    import rasterio.features
    from rasterio.transform import from_origin
    g = polys.to_crs(meta["projection"]).reset_index(drop=True)
    tr = from_origin(meta["x1"], meta["y2"], meta["xpixelsize"], meta["ypixelsize"])
    lab = rasterio.features.rasterize(((geom, i + 1) for i, geom in enumerate(g.geometry)), out_shape=grid.shape,
                                      transform=tr, fill=0, all_touched=False, dtype="int32")
    v = np.asarray(grid, float).ravel()
    l = lab.ravel()
    ok = np.isfinite(v) & (l > 0)
    # รหัสเดียวกันอาจมีหลาย polygon (multi-part) → รวมตามค่ารหัส ไม่ใช่ตามแถว
    ids = g[id_col].values[l[ok] - 1]
    df = pd.DataFrame({id_col: ids, "v": v[ok]}).groupby(id_col)["v"].agg(list(stats))
    return df


def queen_weights(polys):
    """เมทริกซ์น้ำหนัก queen contiguity (ติดกันแม้แค่จุดเดียว) แบบ row-standardized"""
    from scipy import sparse
    g = polys.reset_index(drop=True)
    j = g.sjoin(g[["geometry"]], predicate="touches")
    W = sparse.coo_matrix((np.ones(len(j)), (j.index.values, j["index_right"].values)), shape=(len(g), len(g))).tocsr()
    W.data[:] = 1.0
    rs = np.asarray(W.sum(1)).ravel()
    rs[rs == 0] = 1
    return sparse.diags(1 / rs) @ W


def morans_i(x, W, n_perm=999, seed=0):
    """Moran's I = (n/S0) · zᵀWz / zᵀz และ p-value จากการสุ่มสลับค่า"""
    x = np.asarray(x, float)
    z = x - x.mean()
    s0 = W.sum()
    I = len(x) / s0 * (z @ (W @ z)) / (z @ z)
    rng = np.random.default_rng(seed)
    sims = np.array([len(x) / s0 * (zp @ (W @ zp)) / (zp @ zp) for zp in (rng.permutation(z) for _ in range(n_perm))])
    return float(I), float((np.sum(sims >= I) + 1) / (n_perm + 1)), float(-1 / (len(x) - 1))


def getis_ord_gi(x, W_binary_incl_self):
    """Getis–Ord Gi* (z-score): ค่าบวกมาก = hot spot, ลบมาก = cold spot

    Gi* = (Σ_j w_ij x_j − X̄ Σ_j w_ij) / ( S √[(n Σ_j w_ij² − (Σ_j w_ij)²)/(n−1)] )
    """
    x = np.asarray(x, float)
    n = len(x)
    W = W_binary_incl_self
    xb, S = x.mean(), np.sqrt((x ** 2).mean() - x.mean() ** 2)
    wsum = np.asarray(W.sum(1)).ravel()
    w2 = np.asarray(W.multiply(W).sum(1)).ravel()
    num = W @ x - xb * wsum
    den = S * np.sqrt((n * w2 - wsum ** 2) / (n - 1))
    return num / den


def binary_weights_self(polys):
    from scipy import sparse
    g = polys.reset_index(drop=True)
    j = g.sjoin(g[["geometry"]], predicate="touches")
    W = sparse.coo_matrix((np.ones(len(j)), (j.index.values, j["index_right"].values)), shape=(len(g), len(g))).tocsr()
    W.data[:] = 1.0
    return W + sparse.identity(len(g), format="csr")


# ======================================================================
# 5) แผนที่
# ======================================================================
def show_grid(ax, grid, meta, **kw):
    """แสดงกริด (แถวแรก = ทิศเหนือ) บนแกนหน่วย km แบบเดียวกับ utils.maps.base_map"""
    ext = (meta["x1"] / 1e3, meta["x2"] / 1e3, meta["y1"] / 1e3, meta["y2"] / 1e3)
    kw.setdefault("interpolation", "nearest")
    kw.setdefault("zorder", 2)
    return ax.imshow(grid, extent=ext, origin="upper", **kw)
