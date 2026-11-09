"""Revision analyses (internal review, Nov 2026).
1. Stage shift: race x post interaction (each race vs NH White) in one pooled ITS model with race-specific
   pre-trends; Holm and Benjamini-Hochberg adjustment across the 11 subgroup level-change tests;
   2018 staging-system level-shift check; race x metro strata; all non-large-metro counties combined.
2. De novo metastatic HR+/HER2-: single Cox model (NH White + NH Black) with race x period terms, so
   period HRs within race, Black-White HRs by period and interaction ratios come from one model;
   3-year overall survival and 36-month restricted mean survival time (RMST) instead of medians.
3. Neoadjuvant response: Black-White CR ORs split at the 2018 source-item change; share of missing /
   not-stated response by race among women with systemic therapy before surgery.
"""
import numpy as np, pandas as pd
from scipy import stats
from lib2 import *

cl = pd.read_pickle("output/cl.pkl")
R = ["NH White", "NH Black", "Hispanic", "NH API"]
res = {}

def wald(b, V, i):
    se = np.sqrt(V[i, i]); return np.exp(b[i]), np.exp(b[i] - 1.96 * se), np.exp(b[i] + 1.96 * se), 2 * stats.norm.sf(abs(b[i] / se))

# =============================================== 1. STAGE SHIFT
ks = cl[(cl.stage != "Unknown") & (cl.year != 2020)].copy(); ks["dist"] = (ks.stage == "Distant").astype(int)

def agg(df, keys):
    return df.groupby(keys).agg(n=("dist", "size"), d=("dist", "sum")).reset_index()

def fit(a, X):
    y = np.r_[np.ones(len(a)), np.zeros(len(a))]; w = np.r_[a.d.values, (a.n - a.d).values]
    return logit_w(y, np.r_[X, X], w)

def its_level(df, extra2018=False):
    a = agg(df, ["year", "agegrp", "race", "subtype"])
    Xc, _ = design(a, {"agegrp": "45-49", "race": "NH White", "subtype": "HR+/HER2−"}, numeric=[])
    cols = [a.year.values - 2010, (a.year >= 2021).astype(float).values]
    if extra2018: cols.append((a.year >= 2018).astype(float).values)
    b, V = fit(a, np.c_[np.column_stack(cols), Xc])
    out = dict(zip(["OR", "lo", "hi", "p"], wald(b, V, 2)))
    if extra2018: out.update(dict(zip(["OR_2018", "lo_2018", "hi_2018", "p_2018"], wald(b, V, 3))))
    return out, int(a.n.sum())

# 1a. the 11 subgroup tests + multiplicity
groups = [("All", ks)] + [(f"Race: {r}", ks[ks.race == r]) for r in R + ["NH AIAN"]] + \
         [(f"Subtype: {s}", ks[ks.subtype == s]) for s in ["HR+/HER2−", "HR+/HER2+", "HR−/HER2+", "Triple-negative"]] + \
         [(f"Age {b_}", ks[ks.ageband == b_]) for b_ in ["20-39", "40-49"]]
rows = []
for lab, df in groups:
    o, n = its_level(df); o2, _ = its_level(df, extra2018=True)
    rows.append(dict(group=lab, n=n, **o, OR_adj2018=o2["OR"], lo_adj2018=o2["lo"], hi_adj2018=o2["hi"],
                     OR_2018_step=o2["OR_2018"], lo_2018_step=o2["lo_2018"], hi_2018_step=o2["hi_2018"], p_2018_step=o2["p_2018"]))
t = pd.DataFrame(rows)
sub = t.group != "All"
p = t.loc[sub, "p"].values; m_ = len(p); o_ = np.argsort(p)
holm = np.empty(m_); run = 0
for k, i in enumerate(o_): run = max(run, min(1, (m_ - k) * p[i])); holm[i] = run
bh = np.empty(m_); prev = 1
for k, i in reversed(list(enumerate(o_))): prev = min(prev, p[i] * m_ / (k + 1)); bh[i] = prev
t.loc[sub, "p_holm"] = holm; t.loc[sub, "q_BH"] = bh
res["S5a_ITS_subgroups_multiplicity"] = t

# 1b. pooled interaction model: race-specific intercepts and pre-trends, common covariates, race x post
d = ks[ks.race.isin(R + ["NH AIAN"])]
a = agg(d, ["year", "agegrp", "race", "subtype"])
Xc, _ = design(a, {"agegrp": "45-49", "subtype": "HR+/HER2−"}, numeric=[])
yr = a.year.values - 2010; post = (a.year >= 2021).astype(float).values
races = ["NH Black", "Hispanic", "NH API", "NH AIAN"]
Rd = np.column_stack([(a.race == r).astype(float).values for r in races])
X = np.c_[yr, post, Rd, Rd * yr[:, None], Rd * post[:, None], Xc]
b, V = fit(a, X)
rows = [dict(term="Post-pandemic level change, NH White (reference)", **dict(zip(["OR", "lo", "hi", "p"], wald(b, V, 2))))]  # b[0] = intercept
base = 3 + 2 * len(races)  # intercept, yr, post, Rd, Rd*yr, then Rd*post
for k, r in enumerate(races):
    rows.append(dict(term=f"Ratio of post-pandemic ORs, {r} vs NH White (race x post)", **dict(zip(["OR", "lo", "hi", "p"], wald(b, V, base + k)))))
# joint test of heterogeneity across races
idx = list(range(base, base + len(races))); bb = b[idx]; VV = V[np.ix_(idx, idx)]
chi = float(bb @ np.linalg.solve(VV, bb)); rows.append(dict(term=f"Joint Wald test, race x post (df={len(races)})", OR=np.nan, lo=np.nan, hi=np.nan, p=stats.chi2.sf(chi, len(races))))
# Black+Hispanic combined vs White
a2 = a.copy(); a2["grp"] = np.where(a2.race.isin(["NH Black", "Hispanic"]), "BH", np.where(a2.race == "NH White", "W", "O"))
a2 = a2[a2.grp != "O"]
Xc2, _ = design(a2, {"agegrp": "45-49", "subtype": "HR+/HER2−", "race": "NH White"}, numeric=[])
yr2 = a2.year.values - 2010; post2 = (a2.year >= 2021).astype(float).values; g2 = (a2.grp == "BH").astype(float).values
b2, V2 = fit(a2, np.c_[yr2, post2, g2 * yr2, g2 * post2, Xc2])
rows.append(dict(term="Ratio of post-pandemic ORs, Black and Hispanic combined vs NH White", **dict(zip(["OR", "lo", "hi", "p"], wald(b2, V2, 4)))))
res["S5b_ITS_race_interaction"] = pd.DataFrame(rows)

# 1c. metro: race x metro strata; all non-large-metro counties combined
rows = []
for lab, df in [("All races", ks)] + [(r, ks[ks.race == r]) for r in ["NH White", "NH Black", "Hispanic", "NH API"]]:
    for mlab, mm in [("Large metro", df.metro == "Large metro"), ("Other metro", df.metro == "Other metro"),
                     ("Nonmetro", df.metro == "Nonmetro"), ("All non-large-metro", df.metro.isin(["Other metro", "Nonmetro"]))]:
        dd = df[mm]
        if dd.dist.sum() < 40: continue
        o, n = its_level(dd); rows.append(dict(race=lab, counties=mlab, n=n, distant=int(dd.dist.sum()), **o))
res["S5c_ITS_race_by_metro"] = pd.DataFrame(rows)
# distribution of cases by metro
res["S5d_metro_distribution"] = (ks.groupby(["race", "metro"]).size().rename("n").reset_index()
                                 .assign(pct=lambda x: 100 * x.n / x.groupby("race").n.transform("sum")))

# =============================================== 2. DE NOVO METASTATIC HR+/HER2-
VALID = ["Complete dates are available and there are more than 0 days of survival",
         "Incomplete dates are available and there cannot be zero days of follow-up"]
m = cl[(cl.stage == "Distant") & (cl.year <= 2020) & cl.race.isin(R) & cl["Survival months flag"].isin(VALID)].copy()
m["t"] = pd.to_numeric(m["Survival months"]).astype(float)
m["death"] = (m["Vital status recode (study cutoff used)"] == "Dead").astype(float)
m["per"] = np.select([m.year <= 2013, m.year <= 2017], ["2010-2013", "2014-2017"], "2018-2020")
m["sub3"] = np.select([m.subtype.eq("HR+/HER2−"), m.subtype.isin(["HR+/HER2+", "HR−/HER2+"]), m.subtype.eq("Triple-negative")], ["HR+/HER2−", "HER2+", "Triple-negative"], "Unknown")
mets = lambda s: m[f"SEER Combined Mets at DX-{s} (2010+)"].eq("Yes")
m["metsite"] = np.select([mets("brain"), mets("liver") | mets("lung"), mets("bone")], ["Brain", "Visceral", "Bone only/other"], "Other/unknown")

def km(t, e):
    o = np.argsort(t); t, e = t[o], e[o]
    g = pd.DataFrame({"t": t, "e": e}).groupby("t").agg(n=("e", "size"), d=("e", "sum"))
    n = len(t); S = 1.0; T = [0.0]; Ss = [1.0]; V = 0.0; Vs = [0.0]
    for tm, r in g.iterrows():
        if n > r.d: V += r.d / (n * (n - r.d))
        S *= 1 - r.d / n; n -= r.n; T.append(tm); Ss.append(S); Vs.append(V)
    return np.array(T), np.array(Ss), np.array(Vs)
def at(T, S, h): return S[np.searchsorted(T, h, side="right") - 1]
def rmst(T, S, h=36):
    T2 = np.r_[T[T < h], h]; S2 = S[:len(T2)]; return float(np.sum(np.diff(T2) * S2[:-1]))

rows = []
for (s, r, p_), g in m[m.sub3 != "Unknown"].groupby(["sub3", "race", "per"]):
    if len(g) < 40: continue
    T, S, Vg = km(g.t.values, g.death.values)
    s3 = at(T, S, 36); v3 = Vg[np.searchsorted(T, 36, side="right") - 1]; se3 = s3 * np.sqrt(v3)
    reps = []
    rng = np.random.default_rng(11)
    for _ in range(200):
        ii = rng.integers(0, len(g), len(g)); Tb, Sb, _ = km(g.t.values[ii], g.death.values[ii]); reps.append(rmst(Tb, Sb))
    rows.append(dict(subtype=s, race=r, period=p_, n=len(g), deaths=int(g.death.sum()),
                     OS_3yr=100 * s3, OS_3yr_lo=100 * max(0, s3 - 1.96 * se3), OS_3yr_hi=100 * min(1, s3 + 1.96 * se3),
                     RMST36=rmst(T, S), RMST36_lo=np.percentile(reps, 2.5), RMST36_hi=np.percentile(reps, 97.5),
                     at_risk_36mo=int((g.t >= 36).sum())))
res["S7a_MBC_3yOS_RMST"] = pd.DataFrame(rows)

rows = []
for s in ["HR+/HER2−", "HER2+"]:
    dd = m[(m.sub3 == s) & m.race.isin(["NH White", "NH Black"])].copy()
    blk = (dd.race == "NH Black").astype(float).values
    p2 = (dd.per == "2014-2017").astype(float).values; p3 = (dd.per == "2018-2020").astype(float).values
    Xc, _ = design(dd, {"agegrp": "45-49", "metsite": "Bone only/other", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro", "married": "Married"}, numeric=[])
    X = np.c_[blk, p2, p3, blk * p2, blk * p3, Xc]
    b, V = cox_w(dd.t.values, dd.death.values, X)
    def lin(c, lab):
        c = np.r_[c, np.zeros(len(b) - len(c))]; est = c @ b; se = np.sqrt(c @ V @ c)
        rows.append(dict(subtype=s, quantity=lab, HR=np.exp(est), lo=np.exp(est - 1.96 * se), hi=np.exp(est + 1.96 * se), p=2 * stats.norm.sf(abs(est / se)), n=len(dd), deaths=int(dd.death.sum())))
    lin([1, 0, 0, 0, 0], "Black vs White, 2010-2013")
    lin([1, 0, 0, 1, 0], "Black vs White, 2014-2017")
    lin([1, 0, 0, 0, 1], "Black vs White, 2018-2020")
    lin([0, 1, 0, 0, 0], "White: 2014-2017 vs 2010-2013")
    lin([0, 0, 1, 0, 0], "White: 2018-2020 vs 2010-2013")
    lin([0, 1, 0, 1, 0], "Black: 2014-2017 vs 2010-2013")
    lin([0, 0, 1, 0, 1], "Black: 2018-2020 vs 2010-2013")
    lin([0, 0, 0, 1, 0], "Interaction (ratio of HRs): Black x 2014-2017")
    lin([0, 0, 0, 0, 1], "Interaction (ratio of HRs): Black x 2018-2020")
    # pooled 2014-2020 interaction
    late = (p2 + p3); X2 = np.c_[blk, late, blk * late, Xc]; b2, V2 = cox_w(dd.t.values, dd.death.values, X2)
    se = np.sqrt(V2[2, 2]); rows.append(dict(subtype=s, quantity="Interaction (ratio of HRs): Black x 2014-2020 pooled", HR=np.exp(b2[2]), lo=np.exp(b2[2] - 1.96 * se), hi=np.exp(b2[2] + 1.96 * se), p=2 * stats.norm.sf(abs(b2[2] / se)), n=len(dd), deaths=int(dd.death.sum())))
res["S7b_MBC_single_model"] = pd.DataFrame(rows)

# =============================================== 3. NEOADJUVANT RESPONSE: 2018 split, missingness
neo = cl["Response to Neoadjuvant Therapy Recode (2010+)"]
cl["neo_resp"] = np.select([neo.eq("Stated as complete response (CR)"), neo.eq("Stated as partial response (PR)"),
                            neo.eq("Stated as no response (NR)"), neo.str.startswith("Stated as response to treatment")],
                           ["CR", "PR", "NR", "Response NOS"], "Not given/unknown")
seq = cl["RX Summ--Systemic/Sur Seq (2007+)"]
pre = seq.isin(["Systemic therapy before surgery", "Systemic therapy both before and after surgery"])
tr = cl[pre & cl.race.isin(R) & cl.stage.isin(["Localized", "Regional"]) & cl.subtype.isin(["Triple-negative", "HR−/HER2+", "HR+/HER2+"])].copy()
tr["era"] = np.where(tr.year <= 2017, "2010-2017", "2018-2023")
res["S4d_response_missing_by_race"] = (tr.assign(missing=(tr.neo_resp == "Not given/unknown"), nos=(tr.neo_resp == "Response NOS"))
                                       .groupby(["subtype", "era", "race"]).agg(n=("missing", "size"), pct_not_stated=("missing", "mean"), pct_response_NOS=("nos", "mean"))
                                       .reset_index().assign(pct_not_stated=lambda x: 100 * x.pct_not_stated, pct_response_NOS=lambda x: 100 * x.pct_response_NOS))
na = cl[cl.race.isin(R) & (cl.neo_resp != "Not given/unknown") & cl.subtype.isin(["Triple-negative", "HR−/HER2+", "HR+/HER2+"])].copy()
na["cr"] = (na.neo_resp == "CR").astype(float); na["yr"] = na.year - 2010
rows = []
for s in ["Triple-negative", "HR−/HER2+", "HR+/HER2+"]:
    for era, dd in [("2010-2017", na[(na.subtype == s) & (na.year <= 2017)]), ("2018-2023", na[(na.subtype == s) & (na.year >= 2018)])]:
        X, nm = design(dd, {"race": "NH White", "agegrp": "45-49", "stage": "Localized", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro", "registry": dd.registry.mode()[0]}, numeric=["yr"])
        b, V = logit_w(dd.cr.values, X); se = np.sqrt(np.diag(V))[1:]
        j = nm.index("race: NH Black (ref NH White)") if "race: NH Black (ref NH White)" in nm else [k for k, n_ in enumerate(nm) if "NH Black" in n_][0]
        rows.append(dict(subtype=s, era=era, n=len(dd), cr_pct_white=100 * dd[dd.race == "NH White"].cr.mean(), cr_pct_black=100 * dd[dd.race == "NH Black"].cr.mean(),
                         OR_black=np.exp(b[1 + j]), lo=np.exp(b[1 + j] - 1.96 * se[j]), hi=np.exp(b[1 + j] + 1.96 * se[j])))
res["S4e_CR_by_era"] = pd.DataFrame(rows)

with pd.ExcelWriter("output/results_v7_revision.xlsx") as xw:
    for k, v in res.items(): v.round(4).to_excel(xw, sheet_name=k[:31], index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
for k, v in res.items(): print("=====", k); print(v.round(3).to_string())
