"""Second internal review (Nov 2026).
1. Stage shift, all from ONE specification (quasi-binomial ITS on aggregated cells, as in analysis.py / Figure 1):
   subgroup level changes + Holm/BH; 2018 staging step; post-2020 slope; pooled race x post interaction overall
   and WITHIN county type; race x county type strata; leave-one-registry-out.
2. De novo metastatic HR+/HER2-: single Cox model truncated at 36 months; proportional-hazards test (Grambsch-Therneau
   type correlation of scaled Schoenfeld residuals with rank time); covariate-standardized (adjusted) RMST to 36 months.
3. Time to treatment >60 days: year categorical; restricted to 2010-2020.
4. Complete response excluding 'response NOS' (all subtypes, all years and by era).
"""
import numpy as np, pandas as pd, os, sys
from scipy import stats
from lib2 import *

cl = pd.read_pickle("output/cl.pkl")
R4 = ["NH White", "NH Black", "Hispanic", "NH API"]
res = {}

# ------------------------------------------------------------- quasi-binomial GLM (identical to analysis.py)
def glm_qb(d, n, X, maxit=60):
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = X @ b; p = 1 / (1 + np.exp(-eta)); mu = n * p; w = n * p * (1 - p)
        z = eta + (d - mu) / np.maximum(w, 1e-12); XtW = X.T * w
        bn = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-10: b = bn; break
        b = bn
    p = 1 / (1 + np.exp(-(X @ b))); mu = n * p; w = n * p * (1 - p)
    pr = (d - mu) / np.sqrt(np.maximum(w, 1e-12)); df = len(d) - X.shape[1]
    phi = max(1.0, (pr ** 2).sum() / df) if df > 0 else 1.0
    return b, np.linalg.inv((X.T * w) @ X) * phi

def dummies(a, cols):
    out, names = [], []
    for v in cols:
        for lev in sorted(a[v].unique())[1:]:
            out.append((a[v] == lev).astype(float).values); names.append(f"{v}={lev}")
    return out, names

def wald(b, V, i):
    se = np.sqrt(V[i, i]); return dict(OR=np.exp(b[i]), lo=np.exp(b[i] - 1.96 * se), hi=np.exp(b[i] + 1.96 * se), p=2 * stats.norm.sf(abs(b[i] / se)))

ks = cl[(cl.stage != "Unknown") & (cl.year != 2020)].copy(); ks["dist"] = (ks.stage == "Distant").astype(int)

def its(df, step2018=False, slope=False):
    a = df.groupby(["year", "agegrp", "race", "subtype"]).agg(n=("dist", "size"), d=("dist", "sum")).reset_index()
    yr = a.year.values - 2010; post = (a.year >= 2021).astype(float).values
    cols = [np.ones(len(a)), yr, post]; names = ["const", "trend", "post"]
    if step2018: cols.append((a.year >= 2018).astype(float).values); names.append("step2018")
    if slope: cols.append(post * (a.year.values - 2021)); names.append("postslope")
    dm, dn = dummies(a, ["agegrp", "race", "subtype"]); X = np.column_stack(cols + dm)
    b, V = glm_qb(a.d.values.astype(float), a.n.values.astype(float), X)
    out = {"n": int(a.n.sum()), "distant": int(a.d.sum()), **{f"post_{k}": v for k, v in wald(b, V, 2).items()}}
    if step2018: out.update({f"step2018_{k}": v for k, v in wald(b, V, names.index("step2018")).items()})
    if slope:
        out.update({f"slope_{k}": v for k, v in wald(b, V, names.index("postslope")).items()})
        g = np.zeros(len(b)); g[2] = 1; g[names.index("postslope")] = 1; est = g @ b; se = np.sqrt(g @ V @ g)
        out.update(avg_excess_OR=np.exp(est), avg_lo=np.exp(est - 1.96 * se), avg_hi=np.exp(est + 1.96 * se))
    return out

# 1a. subgroups (Figure 1 rows) + multiplicity + 2018 step + slope
groups = [("All women 20-49", ks)] + [(r, ks[ks.race == r]) for r in R4 + ["NH AIAN"]] + \
         [(s, ks[ks.subtype == s]) for s in ["HR+/HER2−", "HR+/HER2+", "HR−/HER2+", "Triple-negative"]] + \
         [(f"Age {b_}", ks[ks.ageband == b_]) for b_ in ["20-39", "40-49"]]
t = pd.DataFrame([dict(group=lab, **its(df)) for lab, df in groups])
t2 = pd.DataFrame([dict(group=lab, **{("s18_" + k): v for k, v in its(df, step2018=True).items() if k not in ("n", "distant")}) for lab, df in groups])
t3 = pd.DataFrame([dict(group=lab, **{("sl_" + k): v for k, v in its(df, slope=True).items() if k not in ("n", "distant")}) for lab, df in groups])
t = t.merge(t2, on="group").merge(t3, on="group")
sub = t.group != "All women 20-49"; p = t.loc[sub, "post_p"].values; m_ = len(p); o_ = np.argsort(p)
holm = np.empty(m_); run = 0
for k, i in enumerate(o_): run = max(run, min(1, (m_ - k) * p[i])); holm[i] = run
bh = np.empty(m_); prev = 1
for k, i in reversed(list(enumerate(o_))): prev = min(prev, p[i] * m_ / (k + 1)); bh[i] = prev
t.loc[sub, "p_holm"] = holm; t.loc[sub, "q_BH"] = bh
res["S5a_subgroups"] = t

# 1b. pooled interaction models: overall and within county type
def interaction(df, label):
    d = df[df.race.isin(R4 + ["NH AIAN"])]
    a = d.groupby(["year", "agegrp", "race", "subtype"]).agg(n=("dist", "size"), d=("dist", "sum")).reset_index()
    yr = a.year.values - 2010; post = (a.year >= 2021).astype(float).values
    races = [r for r in ["NH Black", "Hispanic", "NH API", "NH AIAN"] if (a.race == r).any() and a[a.race == r].d.sum() >= 40]
    a = a[a.race.isin(["NH White"] + races)].reset_index(drop=True)
    yr = a.year.values - 2010; post = (a.year >= 2021).astype(float).values
    Rd = np.column_stack([(a.race == r).astype(float).values for r in races])
    dm, _ = dummies(a, ["agegrp", "subtype"])
    X = np.column_stack([np.ones(len(a)), yr, post, Rd, Rd * yr[:, None], Rd * post[:, None]] + dm)
    b, V = glm_qb(a.d.values.astype(float), a.n.values.astype(float), X)
    out = [dict(setting=label, term="Post-pandemic level change, NH White", **wald(b, V, 2))]
    base = 3 + 2 * len(races)
    for k, r in enumerate(races): out.append(dict(setting=label, term=f"Ratio of ORs, {r} vs NH White", **wald(b, V, base + k)))
    idx = list(range(base, base + len(races))); chi = float(b[idx] @ np.linalg.solve(V[np.ix_(idx, idx)], b[idx]))
    out.append(dict(setting=label, term=f"Joint test race x post (df={len(races)})", OR=np.nan, lo=np.nan, hi=np.nan, p=stats.chi2.sf(chi, len(races))))
    return out
rows = interaction(ks, "All counties") + interaction(ks[ks.metro == "Large metro"], "Large metro") + \
       interaction(ks[ks.metro.isin(["Other metro", "Nonmetro"])], "Outside large metro")
res["S5b_interaction_by_county"] = pd.DataFrame(rows)

# 1c. race x county type strata
rows = []
for r in ["All"] + R4:
    df = ks if r == "All" else ks[ks.race == r]
    for lab, mm in [("Large metro", df.metro == "Large metro"), ("Outside large metro", df.metro.isin(["Other metro", "Nonmetro"])),
                    ("Other metro", df.metro == "Other metro"), ("Nonmetro", df.metro == "Nonmetro")]:
        if df[mm].dist.sum() < 40: continue
        rows.append(dict(race=r, counties=lab, **its(df[mm])))
res["S5c_race_by_county"] = pd.DataFrame(rows)

# 1d. leave-one-registry-out: White outside large metro; Black and Hispanic interaction (all counties)
rows = []
for reg in sorted(ks.registry.unique()):
    k2 = ks[ks.registry != reg]
    w = its(k2[(k2.race == "NH White") & k2.metro.isin(["Other metro", "Nonmetro"])])
    ia = {x["term"]: x for x in interaction(k2, "")}
    rows.append(dict(registry_dropped=reg, White_outside_OR=w["post_OR"], White_outside_lo=w["post_lo"], White_outside_hi=w["post_hi"],
                     Black_vs_White=ia["Ratio of ORs, NH Black vs NH White"]["OR"], Black_lo=ia["Ratio of ORs, NH Black vs NH White"]["lo"], Black_hi=ia["Ratio of ORs, NH Black vs NH White"]["hi"],
                     Hispanic_vs_White=ia["Ratio of ORs, Hispanic vs NH White"]["OR"], Hisp_lo=ia["Ratio of ORs, Hispanic vs NH White"]["lo"], Hisp_hi=ia["Ratio of ORs, Hispanic vs NH White"]["hi"]))
res["S5d_leave_one_registry_out"] = pd.DataFrame(rows)
# White outside large metro by registry (where are the cases?)
wo = ks[(ks.race == "NH White") & ks.metro.isin(["Other metro", "Nonmetro"])]
rows = []
for reg, g in wo.groupby("registry"):
    pre = g[g.year <= 2019]; post_ = g[g.year >= 2021]
    rows.append(dict(registry=reg, n=len(g), pct_distant_2010_19=100 * pre.dist.mean(), pct_distant_2021_23=100 * post_.dist.mean() if len(post_) else np.nan,
                     **({k: v for k, v in its(g).items() if k.startswith("post_")} if g.dist.sum() >= 40 else {})))
res["S5e_White_outside_by_registry"] = pd.DataFrame(rows)
res["S5f_county_type_distribution"] = (ks.groupby(["race", "metro"]).size().rename("n").reset_index()
                                      .assign(pct=lambda x: 100 * x.n / x.groupby("race").n.transform("sum")))

# ------------------------------------------------------------- 2. METASTATIC HR+/HER2-
VALID = ["Complete dates are available and there are more than 0 days of survival",
         "Incomplete dates are available and there cannot be zero days of follow-up"]
m = cl[(cl.stage == "Distant") & (cl.year <= 2020) & cl.race.isin(["NH White", "NH Black"]) & cl["Survival months flag"].isin(VALID) & cl.subtype.eq("HR+/HER2−")].copy()
m["t"] = pd.to_numeric(m["Survival months"]).astype(float)
m["death"] = (m["Vital status recode (study cutoff used)"] == "Dead").astype(float)
m["per"] = np.select([m.year <= 2013, m.year <= 2017], ["2010-2013", "2014-2017"], "2018-2020")
mets = lambda s: m[f"SEER Combined Mets at DX-{s} (2010+)"].eq("Yes")
m["metsite"] = np.select([mets("brain"), mets("liver") | mets("lung"), mets("bone")], ["Brain", "Visceral", "Bone only/other"], "Other/unknown")
m = m.reset_index(drop=True)
blk = (m.race == "NH Black").astype(float).values
p2 = (m.per == "2014-2017").astype(float).values; p3 = (m.per == "2018-2020").astype(float).values
Xc, cn = design(m, {"agegrp": "45-49", "metsite": "Bone only/other", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro", "married": "Married"}, numeric=[])
X = np.c_[blk, p2, p3, blk * p2, blk * p3, Xc]; xn = ["blk", "p2", "p3", "blk:p2", "blk:p3"] + cn

def lin_rows(b, V, label):
    out = []
    for c, q in [([1, 0, 0, 0, 0], "Black vs White, 2010-2013"), ([1, 0, 0, 1, 0], "Black vs White, 2014-2017"), ([1, 0, 0, 0, 1], "Black vs White, 2018-2020"),
                 ([0, 1, 0, 0, 0], "White: 2014-2017 vs 2010-2013"), ([0, 0, 1, 0, 0], "White: 2018-2020 vs 2010-2013"),
                 ([0, 1, 0, 1, 0], "Black: 2014-2017 vs 2010-2013"), ([0, 0, 1, 0, 1], "Black: 2018-2020 vs 2010-2013"),
                 ([0, 0, 0, 1, 0], "Interaction: Black x 2014-2017"), ([0, 0, 0, 0, 1], "Interaction: Black x 2018-2020")]:
        c = np.r_[c, np.zeros(len(b) - 5)]; est = c @ b; se = np.sqrt(c @ V @ c)
        out.append(dict(model=label, quantity=q, HR=np.exp(est), lo=np.exp(est - 1.96 * se), hi=np.exp(est + 1.96 * se), p=2 * stats.norm.sf(abs(est / se))))
    return out
t_all, e_all = m.t.values, m.death.values
t36 = np.minimum(t_all, 36.0); e36 = np.where(t_all <= 36, e_all, 0.0)
b_all, V_all = cox_w(t_all, e_all, X); b36, V36 = cox_w(t36, e36, X)
rows = lin_rows(b_all, V_all, "All follow-up") + lin_rows(b36, V36, "Follow-up truncated at 36 months")
# pooled 2014-2020 interaction, truncated
late = p2 + p3; X2 = np.c_[blk, late, blk * late, Xc]; b2, V2 = cox_w(t36, e36, X2); se = np.sqrt(V2[2, 2])
rows.append(dict(model="Follow-up truncated at 36 months", quantity="Interaction: Black x 2014-2020 pooled", HR=np.exp(b2[2]), lo=np.exp(b2[2] - 1.96 * se), hi=np.exp(b2[2] + 1.96 * se), p=2 * stats.norm.sf(abs(b2[2] / se))))
res["S7b_MBC_cox_all_vs_36mo"] = pd.DataFrame(rows)

# proportional hazards: correlation of Schoenfeld residuals with rank of event time (Grambsch-Therneau style, per covariate)
def schoenfeld_test(t, e, X, b):
    o = np.argsort(t, kind="stable"); t, e, X = t[o], e[o], X[o]
    w = np.exp(X @ b); n = len(t)
    # risk-set sums from the end
    Sw = np.cumsum(w[::-1])[::-1]; Swx = np.cumsum((w[:, None] * X)[::-1], axis=0)[::-1]
    ev = np.where(e == 1)[0]
    # ties: use first index of each tied time for risk set
    first = {}
    for i in range(n): first.setdefault(t[i], i)
    r = np.array([X[i] - Swx[first[t[i]]] / Sw[first[t[i]]] for i in ev])
    rk = stats.rankdata(t[ev])
    out = {}
    for j in range(X.shape[1]):
        rho, pv = stats.pearsonr(rk, r[:, j]) if r[:, j].std() > 0 else (np.nan, np.nan)
        out[j] = (rho, pv)
    # global: chi2 ~ sum over tested terms (approximate, independent)
    return out
ph = schoenfeld_test(t_all, e_all, X, b_all)
res["S7c_PH_test"] = pd.DataFrame([dict(model="All follow-up", term=xn[j], rho=ph[j][0], p=ph[j][1]) for j in range(5)])
ph36 = schoenfeld_test(t36, e36, X, b36)
res["S7c_PH_test"] = pd.concat([res["S7c_PH_test"], pd.DataFrame([dict(model="Truncated at 36 months", term=xn[j], rho=ph36[j][0], p=ph36[j][1]) for j in range(5)])])

# adjusted (standardized) RMST to 36 months from the truncated Cox model
def breslow(t, e, X, b):
    o = np.argsort(t); t, e, X = t[o], e[o], X[o]; w = np.exp(X @ b)
    Sw = np.cumsum(w[::-1])[::-1]; times = np.unique(t[e == 1]); H = []; acc = 0.0
    idx0 = {}
    for i, tt in enumerate(t): idx0.setdefault(tt, i)
    for tt in times:
        d = ((t == tt) & (e == 1)).sum(); acc += d / Sw[idx0[tt]]; H.append(acc)
    return times, np.array(H)
def std_rmst(t, e, X, b, combos):
    times, H0 = breslow(t, e, X, b); grid = np.r_[0, times[times < 36], 36]
    Hg = np.r_[0, H0[times < 36], H0[times < 36][-1] if (times < 36).any() else 0]
    out = {}
    for lab, vals in combos.items():
        Xs = X.copy(); Xs[:, :5] = vals; lp = Xs @ b
        S = np.exp(-np.outer(np.exp(lp), Hg)).mean(axis=0)          # standardized survival curve
        out[lab] = float(np.sum(np.diff(grid) * S[:-1])), float(np.interp(36, grid, S))
    return out
combos = {("White", "2010-2013"): [0, 0, 0, 0, 0], ("White", "2014-2017"): [0, 1, 0, 0, 0], ("White", "2018-2020"): [0, 0, 1, 0, 0],
          ("Black", "2010-2013"): [1, 0, 0, 0, 0], ("Black", "2014-2017"): [1, 1, 0, 1, 0], ("Black", "2018-2020"): [1, 0, 1, 0, 1]}
est = std_rmst(t36, e36, X, b36, combos)
rng = np.random.default_rng(31); reps = []
for _ in range(int(os.environ.get("BOOTR", "300"))):
    ii = rng.integers(0, len(m), len(m)); bb, _ = cox_w(t36[ii], e36[ii], X[ii]); reps.append(std_rmst(t36[ii], e36[ii], X[ii], bb, combos))
rows = []
for k in combos:
    r_ = np.array([x[k][0] for x in reps]); s_ = np.array([x[k][1] for x in reps])
    rows.append(dict(race=k[0], period=k[1], adj_RMST36=est[k][0], lo=np.percentile(r_, 2.5), hi=np.percentile(r_, 97.5),
                     adj_S36_pct=100 * est[k][1], S36_lo=100 * np.percentile(s_, 2.5), S36_hi=100 * np.percentile(s_, 97.5)))
for rc in ["White", "Black"]:
    d_ = np.array([x[(rc, "2018-2020")][0] - x[(rc, "2010-2013")][0] for x in reps])
    rows.append(dict(race=rc, period="change 2010-13 to 2018-20", adj_RMST36=est[(rc, "2018-2020")][0] - est[(rc, "2010-2013")][0], lo=np.percentile(d_, 2.5), hi=np.percentile(d_, 97.5)))
dd = np.array([(x[("Black", "2018-2020")][0] - x[("Black", "2010-2013")][0]) - (x[("White", "2018-2020")][0] - x[("White", "2010-2013")][0]) for x in reps])
rows.append(dict(race="Black minus White", period="difference in change", adj_RMST36=(est[("Black", "2018-2020")][0] - est[("Black", "2010-2013")][0]) - (est[("White", "2018-2020")][0] - est[("White", "2010-2013")][0]), lo=np.percentile(dd, 2.5), hi=np.percentile(dd, 97.5)))
res["S7d_MBC_adjusted_RMST36"] = pd.DataFrame(rows)
# covariate mix over time
res["S7e_MBC_covariate_mix"] = pd.concat([
    m.groupby(["race", "per"]).metsite.value_counts(normalize=True).mul(100).rename("pct").reset_index().rename(columns={"metsite": "level"}).assign(variable="metsite"),
    m.groupby(["race", "per"]).grade.value_counts(normalize=True).mul(100).rename("pct").reset_index().rename(columns={"grade": "level"}).assign(variable="grade"),
    m.groupby(["race", "per"]).agegrp.value_counts(normalize=True).mul(100).rename("pct").reset_index().rename(columns={"agegrp": "level"}).assign(variable="agegrp")])

# ------------------------------------------------------------- 3. TIME TO TREATMENT
tt = cl[cl.ttt_days.notna() & cl.race.isin(R4 + ["NH AIAN"])].copy(); tt["gt60"] = (tt.ttt_days > 60).astype(float)
rows = []
refs = {"race": "NH White", "agegrp": "45-49", "stage": "Localized", "subtype": "HR+/HER2−", "income_cat": "≥$100k", "metro": "Large metro", "married": "Married"}
for lab, d in [("2010-2023, year categorical", tt), ("2010-2020, year categorical", tt[tt.year <= 2020])]:
    d = d.copy(); d["yearcat"] = d.year.astype(str)
    X, nm = design(d, {**refs, "yearcat": "2010", "registry": d.registry.mode()[0]}, numeric=[])
    b, V = logit_w(d.gt60.values, X); se = np.sqrt(np.diag(V))[1:]
    for j, n_ in enumerate(nm):
        if n_.startswith("race"): rows.append(dict(model=lab, term=n_, OR=np.exp(b[1 + j]), lo=np.exp(b[1 + j] - 1.96 * se[j]), hi=np.exp(b[1 + j] + 1.96 * se[j]), n=len(d)))
res["S3c_TTT_year_categorical"] = pd.DataFrame(rows)
res["S3d_TTT_by_race_period"] = (tt.assign(per=np.where(tt.year <= 2019, "2010-2019", np.where(tt.year == 2020, "2020", "2021-2023")))
                                 .groupby(["race", "per"]).agg(n=("gt60", "size"), pct_gt60=("gt60", "mean")).reset_index().assign(pct_gt60=lambda x: 100 * x.pct_gt60))

# ------------------------------------------------------------- 4. CR excluding response NOS
neo = cl["Response to Neoadjuvant Therapy Recode (2010+)"]
cl["neo_resp"] = np.select([neo.eq("Stated as complete response (CR)"), neo.eq("Stated as partial response (PR)"),
                            neo.eq("Stated as no response (NR)"), neo.str.startswith("Stated as response to treatment")],
                           ["CR", "PR", "NR", "Response NOS"], "Not given/unknown")
na = cl[cl.race.isin(R4) & cl.neo_resp.isin(["CR", "PR", "NR", "Response NOS"]) & cl.subtype.isin(["Triple-negative", "HR−/HER2+", "HR+/HER2+"])].copy()
na["cr"] = (na.neo_resp == "CR").astype(float); na["yr"] = na.year - 2010
rows = []
for s in ["Triple-negative", "HR−/HER2+", "HR+/HER2+"]:
    for era, sel in [("2010-2023", na.year > 0), ("2010-2017", na.year <= 2017), ("2018-2023", na.year >= 2018)]:
        for den, dd in [("NOS counted as not complete", na[sel & (na.subtype == s)]), ("NOS excluded", na[sel & (na.subtype == s) & (na.neo_resp != "Response NOS")])]:
            X, nm = design(dd, {"race": "NH White", "agegrp": "45-49", "stage": "Localized", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro", "registry": dd.registry.mode()[0]}, numeric=["yr"])
            b, V = logit_w(dd.cr.values, X); se = np.sqrt(np.diag(V))[1:]
            j = [k for k, n_ in enumerate(nm) if "NH Black" in n_][0]
            rows.append(dict(subtype=s, era=era, denominator=den, n=len(dd), OR_black=np.exp(b[1 + j]), lo=np.exp(b[1 + j] - 1.96 * se[j]), hi=np.exp(b[1 + j] + 1.96 * se[j])))
res["S4d_CR_by_era_and_NOS"] = pd.DataFrame(rows)

with pd.ExcelWriter("output/results_v9_review2.xlsx") as xw:
    for k, v in res.items(): v.round(4).to_excel(xw, sheet_name=k[:31], index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
for k, v in res.items(): print("=====", k); print(v.round(3).to_string())
