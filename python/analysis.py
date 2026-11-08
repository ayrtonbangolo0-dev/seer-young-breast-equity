"""Provisional analysis: early-onset breast cancer, SEER-17 Nov 2025 (dx 2010-2023, ages 20-49).
Usage: python3 analysis.py <data_dir> <out_dir>
data_dir must hold Breast11.txt (race x year x subtype rates), Breast12.txt (age x year x subtype rates)
and _slim_caselisting.csv (case listing, label format)."""
import sys, os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
YEARS = list(range(2010, 2024))
SUBS = ["HR+/HER2+", "HR-/HER2+", "HR+/HER2-", "HR-/HER2-"]          # rate-file codes 0-3
SUBLAB = {"HR+/HER2-": "HR+/HER2−", "HR+/HER2+": "HR+/HER2+", "HR-/HER2+": "HR−/HER2+", "HR-/HER2-": "Triple-negative"}
RACES = ["Non-Hispanic White", "Non-Hispanic Black", "Non-Hispanic American Indian/Alaska Native",
         "Non-Hispanic Asian or Pacific Islander", "Hispanic (All Races)"]
RACELAB = dict(zip(RACES, ["NH White", "NH Black", "NH AIAN", "NH API", "Hispanic"]))
AGEGRP = ["20-24 years", "25-29 years", "30-34 years", "35-39 years", "40-44 years", "45-49 years"]
STD = np.array([66537, 64571, 71044, 80762, 81851, 72118], float)   # 2000 US std pop per million, ages 20-49
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 2})

# ---------------------------------------------------------------- generic GLM (IRLS)
def glm(y, X, family, offset=None, n=None, maxit=50):
    y = np.asarray(y, float); X = np.asarray(X, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    b = np.zeros(X.shape[1])
    if family == "poisson": b[0] = np.log(max(y.sum(), 1) / np.exp(off).sum())
    for _ in range(maxit):
        eta = X @ b + off
        if family == "poisson":
            mu = np.exp(eta); w = mu; z = eta - off + (y - mu) / mu
        else:
            p = 1 / (1 + np.exp(-eta)); mu = n * p; w = n * p * (1 - p); z = eta - off + (y - mu) / np.maximum(w, 1e-12)
        XtW = X.T * w
        bn = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-10: b = bn; break
        b = bn
    eta = X @ b + off
    if family == "poisson":
        mu = np.exp(eta); w = mu; pr = (y - mu) / np.sqrt(mu)
    else:
        p = 1 / (1 + np.exp(-eta)); mu = n * p; w = n * p * (1 - p); pr = (y - mu) / np.sqrt(np.maximum(w, 1e-12))
    df = len(y) - X.shape[1]
    phi = max(1.0, (pr ** 2).sum() / df) if df > 0 else 1.0     # quasi-likelihood overdispersion
    cov = np.linalg.inv((X.T * w) @ X) * phi
    return b, cov, phi, df

# ---------------------------------------------------------------- 1. INCIDENCE (rate files)
def read_rate(f, names):
    t = pd.read_csv(os.path.join(D, f), sep="\t", header=None, names=names, dtype=str)
    for c in ["rate", "se", "lci", "uci", "count", "pop"]:
        t[c] = pd.to_numeric(t[c].str.replace(",", "").str.strip().replace("~", np.nan), errors="coerce")
    for c in names[:3]: t[c] = t[c].astype(int)
    t = t[t.year > 0].copy(); t["year"] = t.year + 1999
    return t

r11 = read_rate("Breast11.txt", ["race", "year", "sub", "rate", "se", "lci", "uci", "count", "pop"])
r12 = read_rate("Breast12.txt", ["age", "year", "sub", "rate", "se", "lci", "uci", "count", "pop"])
r12 = r12[r12.age.between(5, 10) & r12.year.between(2010, 2023)]
r11 = r11[r11.year.between(2010, 2023)]

def adj(df_age, ages):
    """df_age: rows age(5-10), year, count, pop -> direct age-adjusted rate per 100k + SE."""
    w = STD[[a - 5 for a in ages]]; w = w / w.sum()
    out = []
    for y, g in df_age.groupby("year"):
        g = g.set_index("age").reindex(ages)
        r = g["count"] / g["pop"]
        out.append(dict(year=y, count=g["count"].sum(), rate=1e5 * (w * r).sum(),
                        se=1e5 * np.sqrt((w ** 2 * g["count"] / g["pop"] ** 2).sum())))
    return pd.DataFrame(out)

series = {}   # (group, subtype) -> df year/rate/se/count
bands = {"Ages 20-49": list(range(5, 11)), "Ages 20-39": [5, 6, 7, 8], "Ages 40-49": [9, 10]}
for bname, ages in bands.items():
    sub_all = r12[r12["sub"] <= 5].groupby(["age", "year"]).agg(count=("count", "sum"), pop=("pop", "first")).reset_index()
    series[(bname, "All subtypes")] = adj(sub_all[sub_all.age.isin(ages)], ages)
    for k, s in enumerate(SUBS):
        g = r12[(r12["sub"] == k) & r12.age.isin(ages)][["age", "year", "count", "pop"]]
        series[(bname, s)] = adj(g, ages)
for ri, rname in enumerate(RACES):
    g = r11[r11.race == ri]
    tot = g[g["sub"] <= 5].groupby("year").agg(rate=("rate", "sum"), count=("count", "sum"),
                                               se=("se", lambda v: np.sqrt((v ** 2).sum()))).reset_index()
    series[(RACELAB[rname], "All subtypes")] = tot
    for k, s in enumerate(SUBS):
        series[(RACELAB[rname], s)] = g[g["sub"] == k][["year", "rate", "se", "count"]].reset_index(drop=True)

def trend(df):
    """Weighted log-linear regression on 2010-2019 (pre-pandemic); APC, plus O/E for 2021-2023."""
    d = df.dropna(subset=["rate"]); d = d[d.rate > 0]
    pre = d[d.year.between(2010, 2019)]
    x = pre.year.values - 2010; ylog = np.log(pre.rate.values)
    w = (pre.rate.values / np.maximum(pre.se.values, 1e-9)) ** 2
    X = np.c_[np.ones_like(x), x]
    XtW = X.T * w; cov0 = np.linalg.inv(XtW @ X); b = cov0 @ XtW @ ylog
    resid = ylog - X @ b; dfree = len(x) - 2
    s2 = max(1.0, (w * resid ** 2).sum() / dfree); cov = cov0 * s2
    tq = stats.t.ppf(0.975, dfree)
    apc = 100 * (np.exp(b[1]) - 1); lo = 100 * (np.exp(b[1] - tq * np.sqrt(cov[1, 1])) - 1); hi = 100 * (np.exp(b[1] + tq * np.sqrt(cov[1, 1])) - 1)
    p = 2 * stats.t.sf(abs(b[1] / np.sqrt(cov[1, 1])), dfree)
    post = d[d.year.between(2021, 2023)]
    Xp = np.c_[np.ones(len(post)), post.year.values - 2010]
    pred = Xp @ b
    # O/E pooled over 2021-23 on the log scale: mean(log obs) - mean(pred); var = model + sampling
    lo_obs = np.log(post.rate.values); v_obs = (post.se.values / post.rate.values) ** 2
    a = np.ones(len(post)) / len(post)
    diff = a @ (lo_obs - pred); vdiff = a @ Xp @ cov @ Xp.T @ a + (a ** 2 * v_obs).sum()
    oe = np.exp(diff); oe_lo = np.exp(diff - 1.96 * np.sqrt(vdiff)); oe_hi = np.exp(diff + 1.96 * np.sqrt(vdiff))
    r2010 = d[d.year == 2010].rate.iloc[0]; r2023 = d[d.year == 2023].rate.iloc[0]
    return dict(rate_2010=r2010, rate_2019=d[d.year == 2019].rate.iloc[0], rate_2020=d[d.year == 2020].rate.iloc[0],
                rate_2023=r2023, cases_total=int(d["count"].sum()),
                APC_2010_19=apc, APC_lo=lo, APC_hi=hi, APC_p=p,
                OE_2021_23=oe, OE_lo=oe_lo, OE_hi=oe_hi,
                b0=b[0], b1=b[1])

rows = []
for (grp, s), df in series.items():
    t = trend(df); rows.append(dict(group=grp, subtype=SUBLAB.get(s, s), **t))
inc = pd.DataFrame(rows)

# ---------------------------------------------------------------- 2. CASE LISTING
cl = pd.read_csv(os.path.join(D, "_slim_caselisting.csv"), dtype=str, keep_default_na=False)
n_raw = len(cl)
flow = [("Case listing rows (exported)", n_raw)]
cl = cl[cl["Site recode ICD-O-3/WHO 2008"] == "Breast"]; flow.append(("Site recode = Breast (drops lymphoma etc.)", len(cl)))
cl = cl[~cl["Type of Reporting Source"].isin(["Autopsy only", "Death certificate only"])]; flow.append(("Excluding autopsy/DCO", len(cl)))
cl["year"] = cl["Year of diagnosis"].astype(int)
cl["race"] = cl["Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)"].map(RACELAB).fillna("Unknown")
cl["agegrp"] = cl["Age recode with <1 year olds and 90+"].str.replace(" years", "")
cl["age"] = pd.to_numeric(cl["Age recode with single ages and 90+"].str.extract(r"(\d+)")[0])
cl["ageband"] = np.where(cl.age < 40, "20-39", "40-49")
cl["subtype"] = cl["Breast Subtype (2010+)"].map(SUBLAB).fillna("Unknown")
stg = cl["Combined Summary Stage with Expanded Regional Codes (2004+)"]
cl["stage"] = np.select([stg.eq("Localized only"), stg.str.startswith("Regional"), stg.str.startswith("Distant")],
                        ["Localized", "Regional", "Distant"], "Unknown")
g17, g18 = cl["Grade Recode (thru 2017)"], cl["Derived Summary Grade 2018 (2018+)"]
cl["grade"] = np.select(
    [g17.str.contains("Grade I$|Grade I;|Well", regex=True) & ~g17.str.contains("II"),
     g17.str.contains("Grade II$", regex=True), g17.str.contains("Grade III|Grade IV"),
     g18.str.contains(r"category \(1\)|^Well|^Low", regex=True), g18.str.contains(r"category \(2\)|^Moderately", regex=True),
     g18.str.contains(r"category \(3\)|^Poorly|^High|^Undiff", regex=True)],
    ["G1", "G2", "G3", "G1", "G2", "G3"], "Unknown")
inc_lab = cl["Median household income inflation adj to 2024"]
def inc_num(s):
    if s.startswith("<"): return 35000
    if s.startswith("$120"): return 125000
    try: return float(s.split(" - ")[0].replace("$", "").replace(",", ""))
    except: return np.nan
cl["income_k"] = inc_lab.map(inc_num) / 1000
cl["income_cat"] = pd.cut(cl.income_k, [0, 74.9, 99.9, 1e9], labels=["<$75k", "$75-99k", "≥$100k"]).astype(object).where(cl.income_k.notna(), "Unknown")
ru = cl["Rural-Urban Continuum Code"]
cl["metro"] = np.select([ru.str.contains("ge 1 million"), ru.str.startswith("Counties in metropolitan"), ru.str.startswith("Nonmetro")],
                        ["Large metro", "Other metro", "Nonmetro"], "Unknown")
ms = cl["Marital status at diagnosis"]
cl["married"] = np.select([ms.str.startswith("Married"), ms.eq("Unknown")], ["Married", "Unknown"], "Not married")
cl["period"] = np.select([cl.year <= 2019, cl.year == 2020], ["2010-2019", "2020"], "2021-2023")

# ---- Table 1
def tab1(df, col_by):
    out = []
    levels = sorted(df[col_by].unique())
    N = df.groupby(col_by).size()
    out.append(["N", ""] + [f"{N[l]:,}" for l in levels] + [f"{len(df):,}"])
    for var, lab, order in [("ageband", "Age", ["20-39", "40-49"]), ("race", "Race/ethnicity", list(RACELAB.values()) + ["Unknown"]),
                            ("subtype", "Subtype", list(SUBLAB.values()) + ["Unknown"]),
                            ("stage", "Stage (SEER summary)", ["Localized", "Regional", "Distant", "Unknown"]),
                            ("grade", "Grade", ["G1", "G2", "G3", "Unknown"]),
                            ("income_cat", "County median income", ["<$75k", "$75-99k", "≥$100k", "Unknown"]),
                            ("metro", "Rurality", ["Large metro", "Other metro", "Nonmetro", "Unknown"]),
                            ("married", "Marital status", ["Married", "Not married", "Unknown"])]:
        ct = pd.crosstab(df[var], df[col_by]); tot = df[var].value_counts()
        for o in order:
            if o not in ct.index: continue
            out.append([lab, o] + [f"{ct.loc[o, l]:,} ({100 * ct.loc[o, l] / N[l]:.1f})" for l in levels] + [f"{tot[o]:,} ({100 * tot[o] / len(df):.1f})"])
            lab = ""
    return pd.DataFrame(out, columns=["Characteristic", "Level"] + levels + ["Total"])
t1 = tab1(cl, "period")

# ---------------------------------------------------------------- 3. STAGE SHIFT (ITS)
ks = cl[cl.stage != "Unknown"].copy(); ks["dist"] = (ks.stage == "Distant").astype(int)
prop_year = ks.groupby("year").agg(n=("dist", "size"), d=("dist", "sum")).reset_index(); prop_year["pct"] = 100 * prop_year.d / prop_year.n

def its(df, adjust=True):
    """Binomial ITS on aggregated cells. Pre-trend 2010-19, 2020 excluded, level change for 2021-23 (post indicator).
    Returns OR for post vs. expected-from-trend + per-year observed/expected %."""
    df = df[df.year != 2020]
    keys = ["year"] + (["agegrp", "race", "subtype"] if adjust else [])
    a = df.groupby(keys).agg(n=("dist", "size"), d=("dist", "sum")).reset_index()
    X = [np.ones(len(a)), a.year - 2010, (a.year >= 2021).astype(float)]
    names = ["const", "trend", "post"]
    if adjust:
        for v in ["agegrp", "race", "subtype"]:
            for lev in sorted(a[v].unique())[1:]:
                X.append((a[v] == lev).astype(float)); names.append(f"{v}={lev}")
    X = np.column_stack(X)
    b, cov, phi, dfree = glm(a.d.values, X, "binomial", n=a.n.values)
    i = names.index("post"); se = np.sqrt(cov[i, i])
    j = names.index("trend")
    return dict(OR_post=np.exp(b[i]), lo=np.exp(b[i] - 1.96 * se), hi=np.exp(b[i] + 1.96 * se),
                p=2 * stats.norm.sf(abs(b[i] / se)), OR_trend_per_yr=np.exp(b[j]),
                trend_lo=np.exp(b[j] - 1.96 * np.sqrt(cov[j, j])), trend_hi=np.exp(b[j] + 1.96 * np.sqrt(cov[j, j])),
                dispersion=phi, n=int(a.n.sum()), distant=int(a.d.sum()))

its_rows = [dict(group="All", **its(ks))]
for r in list(RACELAB.values()):
    its_rows.append(dict(group=f"Race: {r}", **its(ks[ks.race == r])))
for s in SUBLAB.values():
    its_rows.append(dict(group=f"Subtype: {s}", **its(ks[ks.subtype == s])))
for bnd in ["20-39", "40-49"]:
    its_rows.append(dict(group=f"Age {bnd}", **its(ks[ks.ageband == bnd])))
its_df = pd.DataFrame(its_rows)

# ---------------------------------------------------------------- 4. SURVIVAL (dx 2010-2018)
sv = cl[cl.year <= 2018].copy(); flow_s = [("Diagnosed 2010-2018", len(sv))]
sv = sv[sv["Survival months flag"].isin(["Complete dates are available and there are more than 0 days of survival",
                                         "Incomplete dates are available and there cannot be zero days of follow-up"])]
flow_s.append(("Valid survival time > 0", len(sv)))
sv = sv[sv["SEER cause-specific death classification"] != "Dead (missing/unknown COD)"]
flow_s.append(("Known cause of death", len(sv)))
sv["t"] = pd.to_numeric(sv["Survival months"]).astype(float)
sv["ev"] = np.select([sv["SEER cause-specific death classification"].eq("Dead (attributable to this cancer dx)"),
                      sv["SEER other cause of death classification"].eq("Dead (attributable to causes other than this cancer dx)")], [1, 2], 0)

def aj(t, ev, tmax=60):
    """Aalen-Johansen CIF for cause 1 and 2; returns time grid & cifs."""
    o = np.argsort(t); t, ev = t[o], ev[o]
    ut = np.unique(t[ev > 0])
    S = 1.0; c1 = c2 = 0.0; T = [0]; C1 = [0]; C2 = [0]
    n_at = len(t); idx = 0
    tt = np.unique(t)
    cnt = pd.DataFrame({"t": t, "e1": ev == 1, "e2": ev == 2}).groupby("t").agg(n=("e1", "size"), e1=("e1", "sum"), e2=("e2", "sum"))
    at_risk = len(t)
    for time, r in cnt.iterrows():
        if r.e1 + r.e2 > 0:
            c1 += S * r.e1 / at_risk; c2 += S * r.e2 / at_risk
            S *= 1 - (r.e1 + r.e2) / at_risk
        T.append(time); C1.append(c1); C2.append(c2)
        at_risk -= r.n
    return np.array(T), np.array(C1), np.array(C2)

def cif_at(t, ev, h=60):
    T, C1, C2 = aj(t, ev); k = np.searchsorted(T, h, side="right") - 1
    return C1[k], C2[k]

rng = np.random.default_rng(2025)
cif_rows = []
for s in ["All"] + list(SUBLAB.values()):
    for r in ["All"] + list(RACELAB.values()):
        g = sv[((sv.subtype == s) | (s == "All")) & ((sv.race == r) | (r == "All"))]
        if len(g) < 50: continue
        t, e = g.t.values, g.ev.values
        c1, c2 = cif_at(t, e)
        bs = [cif_at(t[ix], e[ix])[0] for ix in (rng.integers(0, len(t), len(t)) for _ in range(100))]
        cif_rows.append(dict(subtype=s, race=r, n=len(g), bc_deaths=int((e == 1).sum()),
                             CIF5_bc=100 * c1, lo=100 * np.percentile(bs, 2.5), hi=100 * np.percentile(bs, 97.5), CIF5_other=100 * c2))
cif = pd.DataFrame(cif_rows)

# ---- cause-specific Cox (Breslow ties)
def cox(t, e, X, maxit=60):
    o = np.argsort(-t, kind="stable"); t, e, X = t[o], e[o], X[o]
    b = np.zeros(X.shape[1])
    # group ties: risk set for time t includes all with T >= t; with descending sort, cumulative sums give that
    ut, first = np.unique(-t, return_index=True)    # -t ascending == t descending
    last = np.r_[first[1:], len(t)] - 1
    for _ in range(maxit):
        eta = X @ b; eta -= eta.max(); r = np.exp(eta)
        S0 = np.cumsum(r); S1 = np.cumsum(X * r[:, None], 0)
        S2 = np.cumsum(X[:, :, None] * X[:, None, :] * r[:, None, None], 0)
        d = np.add.reduceat(e.astype(float), first)
        sx = np.add.reduceat(X * e[:, None], first, axis=0)
        s0, s1, s2 = S0[last], S1[last], S2[last]
        m = d > 0
        g = sx[m].sum(0) - (d[m, None] * s1[m] / s0[m, None]).sum(0)
        H = -(d[m, None, None] * (s2[m] / s0[m, None, None] - (s1[m] / s0[m, None])[:, :, None] * (s1[m] / s0[m, None])[:, None, :])).sum(0)
        step = np.linalg.solve(-H, g)
        mx = np.max(np.abs(step))
        if mx > 1: step = step / mx
        b = b + step
        if np.max(np.abs(step)) < 1e-8: break
    return b, np.linalg.inv(-H)

cox_df = sv.copy()
covs = {"race": ("NH White", ["NH Black", "NH AIAN", "NH API", "Hispanic", "Unknown"]),
        "subtype": ("HR+/HER2−", ["HR+/HER2+", "HR−/HER2+", "Triple-negative", "Unknown"]),
        "stage": ("Localized", ["Regional", "Distant", "Unknown"]),
        "grade": ("G1", ["G2", "G3", "Unknown"]),
        "agegrp": ("45-49", ["20-24", "25-29", "30-34", "35-39", "40-44"]),
        "income_cat": ("≥$100k", ["$75-99k", "<$75k", "Unknown"]),
        "metro": ("Large metro", ["Other metro", "Nonmetro", "Unknown"]),
        "married": ("Married", ["Not married", "Unknown"])}
cols, names = [], []
for v, (ref, levs) in covs.items():
    for l in levs:
        x = (cox_df[v] == l).astype(float).values
        if x.sum() < 20: continue
        cols.append(x); names.append(f"{v}: {l} (ref {ref})")
cols.append((cox_df.year - 2010).values.astype(float)); names.append("Year of diagnosis (per year)")
X = np.column_stack(cols)
def run_cox(evcode, mask=None):
    m = np.ones(len(cox_df), bool) if mask is None else mask
    keep = X[m].std(0) > 0
    keep &= (X[m] != 0).sum(0) >= 100
    # drop linearly dependent columns
    idx = np.where(keep)[0]; sel = []
    for j in idx:
        if np.linalg.matrix_rank(np.c_[np.ones(m.sum()), X[m][:, sel + [j]]]) == len(sel) + 2: sel.append(j)
    keep = np.zeros_like(keep); keep[sel] = True
    b, V = cox(cox_df.t.values[m], (cox_df.ev.values[m] == evcode).astype(int), X[m][:, keep])
    se = np.sqrt(np.diag(V)); nm = [n for n, k in zip(names, keep) if k]
    return pd.DataFrame(dict(term=nm, HR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=2 * stats.norm.sf(np.abs(b / se))))
cox_all = run_cox(1)
# race HRs within subtype (cause-specific, BC death), adjusted for everything else
race_by_sub = []
for s in list(SUBLAB.values()):
    m = (cox_df.subtype == s).values
    r = run_cox(1, m); r = r[r.term.str.startswith("race")]; r.insert(0, "subtype", s); race_by_sub.append(r)
race_by_sub = pd.concat(race_by_sub)

# ---------------------------------------------------------------- 5. FIGURES
def fig_incidence():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, band in zip(axes, ["Ages 20-39", "Ages 40-49"]):
        for i, s in enumerate(["HR+/HER2-", "HR+/HER2+", "HR-/HER2-", "HR-/HER2+"]):
            d = series[(band, s)]; row = inc[(inc.group == band) & (inc.subtype == SUBLAB[s])].iloc[0]
            ax.plot(d.year, d.rate, color=COL[i], marker="o", ms=4, label=SUBLAB[s])
            yy = np.arange(2010, 2024); ax.plot(yy, np.exp(row.b0 + row.b1 * (yy - 2010)), color=COL[i], lw=1, ls="--", alpha=.7)
        ax.axvspan(2019.5, 2020.5, color="#e6e5e0", alpha=.7, lw=0)
        ax.set_title(band, loc="left", fontsize=10, color="#0b0b0b"); ax.set_xlabel("Year of diagnosis")
        ax.set_ylabel("Age-adjusted rate per 100,000")
        ax.set_yscale("log")
        tk = [1, 2, 3, 5, 10, 15] if band == "Ages 20-39" else [5, 10, 20, 50, 100]
        ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(tk)); ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(v) for v in tk]))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_ylim(tk[0] * 1.2 if band == "Ages 20-39" else tk[0], tk[-1] * 1.5)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Breast cancer incidence by subtype (dashed = 2010–2019 trend projected; grey = 2020)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(os.path.join(OUT, "fig1_incidence_by_subtype.png"), dpi=300); plt.close(fig)

def fig_stage():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, r in enumerate(RACELAB.values()):
        if r == "NH AIAN": continue
        g = ks[ks.race == r].groupby("year").dist.mean() * 100
        ax.plot(g.index, g.values, color=COL[i], marker="o", ms=4, label=r)
    g = ks.groupby("year").dist.mean() * 100
    ax.plot(g.index, g.values, color="#0b0b0b", lw=2.5, label="All")
    ax.axvspan(2019.5, 2020.5, color="#e6e5e0", alpha=.7, lw=0)
    ax.set_ylabel("% diagnosed at distant stage"); ax.set_xlabel("Year of diagnosis")
    ax.set_title("Distant-stage share among known-stage cases, ages 20–49", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_distant_stage_by_race.png"), dpi=300); plt.close(fig)

def fig_cif():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, s in zip(axes, ["Triple-negative", "HR+/HER2−"]):
        for i, r in enumerate(RACELAB.values()):
            if r == "NH AIAN": continue
            g = sv[(sv.subtype == s) & (sv.race == r)]
            T, C1, _ = aj(g.t.values, g.ev.values)
            k = T <= 96; ax.step(T[k], 100 * C1[k], where="post", color=COL[i], label=r)
        ax.set_title(s, loc="left", fontsize=10); ax.set_xlabel("Months since diagnosis"); ax.set_xticks(range(0, 97, 12))
    axes[0].set_ylabel("Cumulative incidence of breast cancer death (%)"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Breast cancer death accounting for competing causes (Aalen–Johansen), dx 2010–2018", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_cif_by_race.png"), dpi=300); plt.close(fig)

fig_incidence(); fig_stage(); fig_cif()

# ---------------------------------------------------------------- 6. WRITE
rate_long = pd.concat([df.assign(group=g, subtype=SUBLAB.get(s, s)) for (g, s), df in series.items()])[["group", "subtype", "year", "count", "rate", "se"]]
with pd.ExcelWriter(os.path.join(OUT, "provisional_results.xlsx")) as xw:
    pd.DataFrame(flow + [("— Survival cohort —", "")] + flow_s, columns=["Step", "N"]).to_excel(xw, "Flow", index=False)
    t1.to_excel(xw, "Table1_by_period", index=False)
    inc.drop(columns=["b0", "b1"]).round(3).to_excel(xw, "Incidence_trends", index=False)
    rate_long.round(3).to_excel(xw, "Rates_by_year", index=False)
    prop_year.round(2).to_excel(xw, "Distant_pct_by_year", index=False)
    its_df.round(4).to_excel(xw, "Stage_shift_ITS", index=False)
    cif.round(2).to_excel(xw, "CIF_5yr", index=False)
    cox_all.round(4).to_excel(xw, "CSCox_BCdeath", index=False)
    race_by_sub.round(4).to_excel(xw, "CSCox_race_by_subtype", index=False)
print("FLOW", flow, flow_s)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(inc.drop(columns=["b0", "b1"]).round(2).to_string())
print(prop_year.round(2).to_string()); print(its_df.round(3).to_string())
print(cif.round(1).to_string()); print(cox_all.round(3).to_string()); print(race_by_sub.round(3).to_string())
