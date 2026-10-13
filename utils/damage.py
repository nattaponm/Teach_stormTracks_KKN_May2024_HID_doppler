"""ความเสียหายจากพายุฤดูร้อน: ทำความสะอาด → รวม 3 ระดับ (อำเภอ-วัน → เหตุการณ์ → วัน) → จัดระดับ

ตัวชี้วัดหลัก = จำนวนบ้านเรือนเสียหาย (หลังคาเรือน) กติกาเสริมทั้งหมดอยู่ใน ``config/damage_rules.yaml``

หลักการรวม
---------
* **ความรุนแรง** (ระดับ) รวมด้วยค่าสูงสุด, **ผลกระทบ** (จำนวนบ้าน คน) รวมด้วยผลรวม
* อำเภอ-วันเดียวกันที่มีหลายแถว → รวมเป็นแถวเดียว (บ้าน = ผลรวมของค่าที่ทราบ)
* ยอดรวมระดับจังหวัด (แยกรายอำเภอไม่ได้) → นับที่ระดับเหตุการณ์ เป็นส่วนที่ "ยังไม่ได้แจกแจง" = total − Σ ค่าที่ทราบในกลุ่ม
* เหตุการณ์ = กลุ่มของอำเภอ-วันที่เชื่อมกันด้วย (ก) เส้นทางพายุเดียวกัน หรือ (ข) ยอดรวมจังหวัดเดียวกัน (union–find)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from .config import repo_path

ORDER5 = ["C", "B", "Weak", "Moderate", "Severe"]          # ลำดับกลุ่มเส้นทางสำหรับการทดสอบแนวโน้ม


def load_rules(path=None):
    with open(path or repo_path("config", "damage_rules.yaml"), encoding="utf-8") as f:
        r = yaml.safe_load(f)
    for k in ("exclude", "casualty_not_wind", "groups", "upgrade_severe", "houses_from_note"):
        r[k] = r.get(k) or {}
    return r


def classify(houses, scheme="primary", rules=None, upgrade=None):
    """จำนวนบ้าน → ระดับ (Categorical มีลำดับ) ; upgrade=True → Severe ; บ้านไม่ทราบ → NaN"""
    rules = rules or load_rules()
    c = rules["classes"][scheme]
    edges = [float(e) for e in c["edges"]]
    out = pd.cut(pd.Series(houses, dtype=float), edges, labels=c["labels"], right=False)
    if upgrade is not None:
        out = out.astype(object)
        out[np.asarray(upgrade, bool)] = c["labels"][-1]
    return pd.Categorical(out, categories=c["labels"], ordered=True)


# ======================================================================
# 1) ระดับอำเภอ-วัน
# ======================================================================
def district_days(reports, rules=None, geometry=None):
    """รายงานรายแถว (reports_may2024_clean.csv) → ตารางอำเภอ-วัน พร้อมป้ายกติกา

    คอลัมน์สำคัญ: dd_id, date, houses (NaN = ไม่ทราบ), dead_wind, hail, upgrade, group, excluded, excl_reason,
    report_ids, win_start_utc, win_end_utc, class3 / class4 (ระดับอำเภอ)
    geometry: GeoDataFrame ของรายงาน (คอลัมน์ report_id, geometry) ถ้าต้องการ polygon อำเภอ
    """
    rules = rules or load_rules()
    R = reports.copy()
    R["date"] = pd.to_datetime(R["date"])
    for c in ("win_start_utc", "win_end_utc"):
        R[c] = pd.to_datetime(R[c])
    rid = R["report_id"]
    for k, v in rules["houses_from_note"].items():
        R.loc[rid == k, "houses"] = float(v["houses"])
    R["excluded"] = rid.isin(rules["exclude"])
    R["excl_reason"] = rid.map({k: v.get("reason", "") for k, v in rules["exclude"].items()})
    R["dead_wind"] = np.where(rid.isin(rules["casualty_not_wind"]), 0, R["dead"].fillna(0))
    R["upgrade"] = rid.isin(rules["upgrade_severe"])
    gmap = {r: g for g, v in rules["groups"].items() for r in v["reports"]}
    R["group"] = rid.map(gmap)
    R["hail"] = R["hail"].astype(bool)

    def agg(g):
        h = g["houses"]
        return pd.Series({
            "province": g["province"].iloc[0], "district": g["district"].iloc[0], "pro_code": g["pro_code"].iloc[0],
            "report_ids": ";".join(g["report_id"]), "n_rows": len(g),
            "houses": h.sum(min_count=1), "dead_wind": g["dead_wind"].sum(), "injured": g["injured"].fillna(0).sum(),
            "hail": g["hail"].any(), "upgrade": g["upgrade"].any(),
            "group": g["group"].dropna().iloc[0] if g["group"].notna().any() else None,
            "excluded": g["excluded"].all(), "excl_reason": "; ".join(g["excl_reason"].dropna()),
            "win_start_utc": g["win_start_utc"].min(), "win_end_utc": g["win_end_utc"].max(),
            "coverage": g["coverage"].iloc[0], "r_km": g["r_km"].iloc[0], "az_deg": g["az_deg"].iloc[0],
            "phase": g["phase"].iloc[0]})
    D = R.groupby(["date", "amp_code"], sort=True).apply(agg, include_groups=False).reset_index()
    D.insert(0, "dd_id", [f"D{d:%m%d}-{int(a)}" for d, a in zip(D["date"], D["amp_code"])])
    D["houses_known"] = D["houses"].notna()
    D["class3"] = classify(D["houses"], "primary", rules, D["upgrade"])
    D["class4"] = classify(D["houses"], "robust4", rules, D["upgrade"])
    D["in_main"] = (~D["excluded"]) & D["date"].dt.day.between(*rules["analysis"]["main_days"]) & \
                   (D["coverage"] != "นอกรัศมี")
    if geometry is not None:
        import geopandas as gpd
        first = R.drop_duplicates(["date", "amp_code"])[["date", "amp_code", "report_id"]]
        geo = geometry.set_index("report_id").geometry
        D["geometry"] = [geo.get(r) for r in first.set_index(["date", "amp_code"]).loc[
            list(zip(D["date"], D["amp_code"])), "report_id"]]
        D = gpd.GeoDataFrame(D, geometry="geometry", crs=geometry.crs)
    return D


# ======================================================================
# 2) ระดับเหตุการณ์
# ======================================================================
def _union_find(nodes, edges):
    parent = {n: n for n in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return {n: find(n) for n in nodes}


def build_events(D, pairs, rules=None):
    """อำเภอ-วัน + คู่ (dd_id, uid) จากการจับคู่เส้นทาง → (D พร้อม event_id, ตารางเหตุการณ์)

    pairs : DataFrame คอลัมน์ report_id (= dd_id), uid — ใช้ผลของ Δt = 15 นาทีเป็นหลักในการนิยามเหตุการณ์
    """
    rules = rules or load_rules()
    D = D.copy()
    use = D[~D["excluded"]]
    nodes = list(use["dd_id"])
    edges = []
    p = pairs[pairs["report_id"].isin(nodes)]
    for _, g in p.groupby("uid"):                                   # เส้นทางเดียวกัน → เหตุการณ์เดียวกัน
        ids = list(g["report_id"].unique())
        edges += [(ids[0], b) for b in ids[1:]]
    for _, g in use.dropna(subset=["group"]).groupby("group"):        # ยอดรวมจังหวัดเดียวกัน
        ids = list(g["dd_id"])
        edges += [(ids[0], b) for b in ids[1:]]
    root = _union_find(nodes, edges)
    roots = sorted(set(root.values()), key=lambda r: (D.set_index("dd_id").loc[r, "date"], r))
    eid = {r: f"EV{i + 1:02d}" for i, r in enumerate(roots)}
    D["event_id"] = D["dd_id"].map(lambda d: eid.get(root.get(d)))
    matched = set(p["report_id"])
    D["radar_matched"] = D["dd_id"].isin(matched)

    rows = []
    for ev, g in D.dropna(subset=["event_id"]).groupby("event_id"):
        known = g["houses"].sum(min_count=1)
        extra = 0.0
        for gr in g["group"].dropna().unique():                      # ส่วนที่ยังไม่แจกแจงของยอดรวมจังหวัด
            tot = float(rules["groups"][gr]["total"])
            in_grp = D[D["group"] == gr]["houses"].sum()
            extra += max(0.0, tot - in_grp)
        unknown_dd = int((~g["houses_known"] & g["group"].isna()).sum())
        houses = (0.0 if pd.isna(known) else known) + extra
        has_number = g["houses_known"].any() or g["group"].notna().any()
        rows.append({"event_id": ev, "date": g["date"].min(), "n_districts": len(g),
                     "districts": ";".join(g["province"] + "/" + g["district"]),
                     "houses": houses if has_number else np.nan, "houses_unallocated": extra,
                     "n_district_unknown": unknown_dd, "dead_wind": g["dead_wind"].sum(), "hail": g["hail"].any(),
                     "upgrade": g["upgrade"].any(), "radar_matched": g["radar_matched"].any(),
                     "in_main": g["in_main"].any(), "phase": g["phase"].iloc[0],
                     "win_start_utc": g["win_start_utc"].min(), "win_end_utc": g["win_end_utc"].max()})
    E = pd.DataFrame(rows)
    E["class3"] = classify(E["houses"], "primary", rules, E["upgrade"])
    E["class4"] = classify(E["houses"], "robust4", rules, E["upgrade"])
    # ตรวจความทนทาน: ไม่ทราบจำนวนบ้าน → Weak
    E["class3_unknown_weak"] = pd.Categorical(E["class3"].astype(object).where(E["class3"].notna(), "Weak"),
                                              categories=E["class3"].cat.categories, ordered=True)
    D = D.merge(E[["event_id", "class3", "class4"]].rename(columns={"class3": "event_class3",
                                                                      "class4": "event_class4"}),
                on="event_id", how="left")
    return D, E


# ======================================================================
# 3) ระดับวัน
# ======================================================================
def day_table(E, days, rules=None, col="class3"):
    """ระดับวัน: ระดับสูงสุดของเหตุการณ์ในวันนั้น ('None' = ไม่มีรายงาน) + ผลรวมบ้าน/เหตุการณ์"""
    days = pd.to_datetime(pd.Series(days)).dt.normalize()
    e = E[E[col].notna()].copy()
    e["day"] = pd.to_datetime(e["date"]).dt.normalize()
    cats = list(E[col].cat.categories)
    g = e.groupby("day").agg(n_events=("event_id", "size"), houses=("houses", "sum"),
                             max_code=(col, lambda s: s.cat.codes.max()), hail=("hail", "any"))
    out = pd.DataFrame({"day": days}).merge(g, left_on="day", right_index=True, how="left")
    out["n_events"] = out["n_events"].fillna(0).astype(int)
    out["day_class"] = ["None" if pd.isna(c) else cats[int(c)] for c in out["max_code"]]
    out["day_class"] = pd.Categorical(out["day_class"], ["None"] + cats, ordered=True)
    return out.drop(columns="max_code")


# ======================================================================
# 4) ป้ายของเส้นทาง: C / B / A-ระดับ
# ======================================================================
def label_tracks(T, pairs, D, report_days_local, col="event_class3"):
    """เส้นทาง (ตาราง tracks_<tag>) → คอลัมน์ group5 (C, B, Weak, Moderate, Severe), is_A, is_rep

    A = จับคู่กับอำเภอ-วันที่ไม่ถูกตัดออก (ระดับ = ระดับสูงสุดของเหตุการณ์ที่จับคู่)
    B = วันท้องถิ่นเดียวกับวันที่มีรายงาน แต่ไม่จับคู่  ·  C = วันท้องถิ่นที่ไม่มีรายงาน
    is_rep = เส้นทางตัวแทนของอำเภอ-วัน (ซ้อนทับอำเภอมากที่สุดในหน้าต่างเวลา)
    """
    T = T.copy()
    ok = D[~D["excluded"]].set_index("dd_id")
    p = pairs[pairs["report_id"].isin(ok.index)].copy()
    p["cls"] = ok.loc[p["report_id"], col].values
    cats = list(D[col].cat.categories)
    p["code"] = pd.Categorical(p["cls"], cats, ordered=True).codes
    best = p[p["code"] >= 0].groupby("uid")["code"].max()
    rep = set(p.sort_values("overlap_km2", ascending=False).drop_duplicates("report_id")["uid"]) \
        if "overlap_km2" in p else set(p["uid"])
    rdays = set(pd.to_datetime(pd.Series(list(report_days_local))).dt.normalize())
    dl = pd.to_datetime(T["day_local"]).dt.normalize()
    T["is_A"] = T["uid"].isin(set(p["uid"]))
    T["is_rep"] = T["uid"].isin(rep)
    g = np.where(dl.isin(rdays), "B", "C").astype(object)
    a_cls = T["uid"].map(best)
    g = np.where(T["is_A"] & a_cls.notna(), [cats[int(c)] if pd.notna(c) else None for c in a_cls], g)
    g = np.where(T["is_A"] & a_cls.isna(), "A-unknown", g)
    T["group5"] = g
    order = ["C", "B"] + cats
    T["group_ord"] = pd.Categorical(T["group5"].where(T["group5"].isin(order)), order, ordered=True)
    T["group3"] = pd.Categorical(np.where(T["is_A"], "A", T["group5"].where(T["group5"].isin(["B", "C"]))),
                                 ["C", "B", "A"], ordered=True)
    return T


# ======================================================================
# 5) สถิติ
# ======================================================================
def jonckheere(values, groups, order, n_perm=2000, strata=None, seed=0):
    """Jonckheere–Terpstra trend test (ค่ามีแนวโน้มเพิ่มตามลำดับกลุ่ม) — p จากการสลับป้าย (ภายใน strata ถ้าระบุ)

    J = Σ_{i<j} #{(x ∈ g_i, y ∈ g_j): y > x} + ½ #{y = x}
    คืน (J มาตรฐาน z, p ด้านเดียว)
    """
    v = np.asarray(values, float)
    g = pd.Categorical(groups, order, ordered=True).codes
    ok = np.isfinite(v) & (g >= 0)
    v, g = v[ok], g[ok]
    st = None if strata is None else np.asarray(strata)[ok]

    def J(gg):
        s = 0.0
        for i in range(len(order)):
            xi = np.sort(v[gg == i])
            if xi.size == 0:
                continue
            for j in range(i + 1, len(order)):
                yj = v[gg == j]
                if yj.size == 0:
                    continue
                lo = np.searchsorted(xi, yj, side="left")
                hi = np.searchsorted(xi, yj, side="right")
                s += lo.sum() + 0.5 * (hi - lo).sum()
        return s
    j0 = J(g)
    rng = np.random.default_rng(seed)
    sims = np.empty(n_perm)
    idx = [np.flatnonzero(st == k) for k in np.unique(st)] if st is not None else [np.arange(len(g))]
    for k in range(n_perm):
        gg = g.copy()
        for ix in idx:
            gg[ix] = rng.permutation(gg[ix])
        sims[k] = J(gg)
    z = (j0 - sims.mean()) / (sims.std() + 1e-12)
    return float(z), float((np.sum(sims >= j0) + 1) / (n_perm + 1))


def cluster_bootstrap_delta(a, b, cluster_a, cluster_b, n_boot=1000, seed=0):
    """Cliff's δ (a vs b) + CI 95% จาก bootstrap แบบสุ่ม "วัน" (cluster) แทนการสุ่มเส้นทาง"""
    from .track_tools import cliffs_delta
    a, b = np.asarray(a, float), np.asarray(b, float)
    ca, cb = np.asarray(cluster_a), np.asarray(cluster_b)
    ma, mb = np.isfinite(a), np.isfinite(b)
    a, ca, b, cb = a[ma], ca[ma], b[mb], cb[mb]
    d0 = cliffs_delta(a, b)
    rng = np.random.default_rng(seed)
    ua, ub = np.unique(ca), np.unique(cb)
    ia = {k: np.flatnonzero(ca == k) for k in ua}
    ib = {k: np.flatnonzero(cb == k) for k in ub}
    bs = []
    for _ in range(n_boot):
        sa = np.concatenate([ia[k] for k in rng.choice(ua, len(ua))])
        sb = np.concatenate([ib[k] for k in rng.choice(ub, len(ub))])
        bs.append(cliffs_delta(a[sa], b[sb]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(d0), float(lo), float(hi)


def fdr_bh(p):
    """Benjamini–Hochberg: q-value"""
    p = np.asarray(p, float)
    n = np.sum(np.isfinite(p))
    o = np.argsort(np.where(np.isfinite(p), p, np.inf))
    q = np.full_like(p, np.nan)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = o[rank - 1]
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return q


def strat_perm_delta(x, is_case, strata=None, n_perm=2000, seed=0):
    """Cliff's δ (case vs control) + p สองด้านจากการสลับป้ายภายใน strata

    ใช้อันดับ (rank) ของข้อมูลรวม: U = Σ rank(case) − n₁(n₁+1)/2 ,  δ = 2U/(n₁n₂) − 1
    การสลับป้ายไม่เปลี่ยนอันดับ จึงคำนวณได้เร็ว
    """
    from scipy.stats import rankdata
    x = np.asarray(x, float)
    y = np.asarray(is_case, bool)
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    st = np.zeros(len(x), int) if strata is None else pd.factorize(np.asarray(strata)[ok])[0]
    n1, n2 = y.sum(), (~y).sum()
    if n1 < 3 or n2 < 3:
        return np.nan, np.nan, int(n1), int(n2)
    rk = rankdata(x)

    def delta(lbl):
        u = rk[lbl].sum() - n1 * (n1 + 1) / 2
        return 2 * u / (n1 * n2) - 1
    d0 = delta(y)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(st == k) for k in np.unique(st)]
    sims = np.empty(n_perm)
    for i in range(n_perm):
        lb = y.copy()
        for ix in groups:
            lb[ix] = lb[rng.permutation(ix)]
        sims[i] = delta(lb)
    p = (np.sum(np.abs(sims) >= abs(d0)) + 1) / (n_perm + 1)
    return float(d0), float(p), int(n1), int(n2)


def growth_metrics(C):
    """จากตารางเซลล์ (เฉพาะภาพจริง) → ต่อเส้นทาง: อัตราโตของพื้นที่สูงสุด (km²/ชม.), การกระโดดของลูกเห็บ/KDP สูงสุด (ต่อ 15 นาที),
    นาทีจากจุดเริ่มถึงพื้นที่สูงสุด"""
    c = C[C["observed"]].sort_values(["uid", "time"]).copy()
    dt = c.groupby("uid")["time"].diff().dt.total_seconds() / 3600
    c["dA_dt"] = c.groupby("uid")["area_km2"].diff() / dt
    for v in ("hail_col_km2", "KDP95"):
        if v in c:
            c[f"d_{v}"] = c.groupby("uid")[v].diff()
    g = c.groupby("uid")
    out = pd.DataFrame({"growth_max_km2h": g["dA_dt"].max()})
    for v in ("hail_col_km2", "KDP95"):
        if f"d_{v}" in c:
            out[f"jump_{v}"] = g[f"d_{v}"].max()
    t0 = g["time"].min()
    tmax = c.loc[c.groupby("uid")["area_km2"].idxmax()].set_index("uid")["time"]
    out["t_to_peak_min"] = (tmax - t0).dt.total_seconds() / 60
    return out.reset_index()
