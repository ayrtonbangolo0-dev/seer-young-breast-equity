"""Fine-Gray sensitivity, subgroup analyses and IORW mediation. Run after fg_run.py (uses sv.pkl / cl.pkl)."""
import numpy as np, pandas as pd, time
from lib2 import *
rng = np.random.default_rng(20251108)
import os
B = int(os.environ.get("BOOT", "100")); BM = int(os.environ.get("BOOTM", "60"))
sv = pd.read_pickle("output/sv.pkl"); cl = pd.read_pickle("output/cl.pkl")
res = {}

# =============================================================== 1. FINE-GRAY with bootstrap CIs
refs = {"race": "NH White", "subtype": "HR+/HER2−", "stage": "Localized", "grade": "G1", "agegrp": "45-49",
        "income_cat": "≥$100k", "metro": "Large metro", "married": "Married"}
def fg_fit(d, refs):
    X, nm = design(d, refs, numeric=["yr"])
    G = km_censor(d.t.values, d.ev.values)
    b, V = cox_w(d.t.values, (d.ev.values == 1).astype(float), X, comp=(d.ev.values == 2).astype(float), G=G)
    return pd.Series(b, nm)
def boot(fun, d, B=B):
    est = fun(d); reps = []
    for _ in range(B):
        ix = rng.integers(0, len(d), len(d)); r = fun(d.iloc[ix]); reps.append(r.reindex(est.index))
    reps = pd.DataFrame(reps)
    return pd.DataFrame({"est": np.exp(est), "lo": np.exp(reps.quantile(.025)), "hi": np.exp(reps.quantile(.975))})
t0 = time.time()
fg_all = boot(lambda d: fg_fit(d, refs), sv); print("FG overall", time.time() - t0)
res["FineGray_overall"] = fg_all.reset_index().rename(columns={"index": "term", "est": "sHR"})
rows = []
for s in ["HR+/HER2−", "HR+/HER2+", "HR−/HER2+", "Triple-negative"]:
    d = sv[sv.subtype == s]; r2 = {k: v for k, v in refs.items() if k != "subtype"}
    f = boot(lambda x: fg_fit(x, r2), d, B=BM)
    f = f[f.index.str.startswith("race")]; f.insert(0, "subtype", s); rows.append(f.reset_index().rename(columns={"index": "term", "est": "sHR"}))
res["FineGray_race_by_subtype"] = pd.concat(rows)
print(res["FineGray_race_by_subtype"].round(3).to_string())

# =============================================================== 2. SUBGROUPS
main_races = ["NH White", "NH Black", "Hispanic", "NH API", "NH AIAN"]
# 2a. time to treatment by race and period
tt = cl[cl.ttt_days.notna() & cl.race.isin(main_races)].copy()
tt["gt60"] = (tt.ttt_days > 60).astype(int); tt["gt90"] = (tt.ttt_days > 90).astype(int)
tt["per"] = np.where(tt.year <= 2019, "2010-2019", np.where(tt.year == 2020, "2020", "2021-2023"))
a = tt.groupby(["race", "per"]).agg(n=("ttt_days", "size"), median_days=("ttt_days", "median"), pct_gt60=("gt60", "mean"), pct_gt90=("gt90", "mean")).reset_index()
a[["pct_gt60", "pct_gt90"]] *= 100
res["TTT_by_race_period"] = a
# adjusted OR of >60 days, by race (adj age group, year, stage, subtype, income, rurality, marital, registry)
Xr, nr = design(tt, {"race": "NH White", "agegrp": "45-49", "stage": "Localized", "subtype": "HR+/HER2−", "income_cat": "≥$100k", "metro": "Large metro", "married": "Married", "registry": tt.registry.mode()[0]}, numeric=[])
Xr = np.c_[Xr, tt.year.values - 2010]; nr = nr + ["year"]
b, V = logit_w(tt.gt60.values.astype(float), Xr); se = np.sqrt(np.diag(V))[1:]
o = pd.DataFrame({"term": nr, "OR": np.exp(b[1:]), "lo": np.exp(b[1:] - 1.96 * se), "hi": np.exp(b[1:] + 1.96 * se)})
res["TTT_gt60_adjOR"] = o[o.term.str.startswith("race") | o.term.str.startswith("year")]
# does >60/>90 delay predict BC death? (survival cohort, adjusted)
svt = sv.copy(); svt["ttt_cat"] = cl.set_index("Patient ID").loc[svt["Patient ID"], "ttt_cat"].values
X, nm = design(svt, dict(refs, ttt_cat="0-30"), numeric=["yr"])
b, V = cox_w(svt.t.values, (svt.ev.values == 1).astype(float), X); se = np.sqrt(np.diag(V))
o = pd.DataFrame({"term": nm, "HR": np.exp(b), "lo": np.exp(b - 1.96 * se), "hi": np.exp(b + 1.96 * se)})
res["TTT_and_BCdeath"] = o[o.term.str.startswith("ttt")]

# 2b. CoC accreditation (dx 2018-2023)
cc = cl[(cl.year >= 2018) & cl.coc.isin(["CoC (analytic)", "Non-CoC"]) & cl.race.isin(main_races)].copy()
cc["noncoc"] = (cc.coc == "Non-CoC").astype(int)
res["CoC_by_race"] = (cc.groupby("race").agg(n=("noncoc", "size"), pct_nonCoC=("noncoc", "mean")).reset_index().assign(pct_nonCoC=lambda d: 100 * d.pct_nonCoC))
c2 = cc[cc["Survival months flag"].isin(["Complete dates are available and there are more than 0 days of survival", "Incomplete dates are available and there cannot be zero days of follow-up"]) & (cc["SEER cause-specific death classification"] != "Dead (missing/unknown COD)")].copy()
c2["t"] = pd.to_numeric(c2["Survival months"]).astype(float)
c2["ev"] = np.select([c2["SEER cause-specific death classification"].eq("Dead (attributable to this cancer dx)")], [1], 0)
c2["yr"] = c2.year - 2018
X, nm = design(c2, dict(refs, coc="CoC (analytic)"), numeric=["yr"])
b, V = cox_w(c2.t.values, c2.ev.values.astype(float), X); se = np.sqrt(np.diag(V))
o = pd.DataFrame({"term": nm, "HR": np.exp(b), "lo": np.exp(b - 1.96 * se), "hi": np.exp(b + 1.96 * se)})
res["CoC_BCdeath_2018_23"] = o[o.term.str.startswith("coc") | o.term.str.startswith("race")].assign(n=len(c2), deaths=int(c2.ev.sum()))

# 2c. stage-shift ITS by income and rurality
ks = cl[cl.stage != "Unknown"].copy(); ks["dist"] = (ks.stage == "Distant").astype(int)
def its(df):
    df = df[df.year != 2020]
    a = df.groupby(["year", "agegrp", "race", "subtype"]).agg(n=("dist", "size"), d=("dist", "sum")).reset_index()
    X, nm = design(a, {"agegrp": "45-49", "race": "NH White", "subtype": "HR+/HER2−"}, numeric=[])
    X = np.c_[a.year - 2010, (a.year >= 2021).astype(float), X]
    # aggregated binomial via expanded weights
    y = np.r_[np.ones(len(a)), np.zeros(len(a))]; w = np.r_[a.d.values, (a.n - a.d).values]; XX = np.r_[X, X]
    b, V = logit_w(y, XX, w)
    se = np.sqrt(V[2, 2]); return np.exp(b[2]), np.exp(b[2] - 1.96 * se), np.exp(b[2] + 1.96 * se), int(a.n.sum())
rows = []
for col in ["income_cat", "metro"]:
    for lev in sorted(ks[col].unique()):
        if lev == "Unknown": continue
        o_, l_, h_, n_ = its(ks[ks[col] == lev]); rows.append(dict(subgroup=f"{col}={lev}", OR_post=o_, lo=l_, hi=h_, n=n_))
for r_ in ["NH Black", "Hispanic"]:
    for lev in ["<$75k", "$75-99k", "≥$100k"]:
        o_, l_, h_, n_ = its(ks[(ks.race == r_) & (ks.income_cat == lev)]); rows.append(dict(subgroup=f"{r_} & income {lev}", OR_post=o_, lo=l_, hi=h_, n=n_))
res["StageShift_by_SES"] = pd.DataFrame(rows)

# 2d. detailed Asian / Pacific Islander subgroups
dr = cl["Race recode (with detailed Asian and Native Hawaiian other PI)"]
api = cl[(cl.race == "NH API")].copy(); api["detail"] = dr[api.index]
sva = sv[sv.race == "NH API"].copy(); sva["detail"] = cl.set_index("Patient ID").loc[sva["Patient ID"], "Race recode (with detailed Asian and Native Hawaiian other PI)"].values
def aj5(t, e):
    o = np.argsort(t); t, e = t[o], e[o]; S = 1; c1 = 0; n = len(t)
    cnt = pd.DataFrame({"t": t, "e1": e == 1, "e2": e == 2}).groupby("t").agg(n=("e1", "size"), e1=("e1", "sum"), e2=("e2", "sum"))
    for tm, r in cnt.iterrows():
        if tm > 60: break
        c1 += S * r.e1 / n; S *= 1 - (r.e1 + r.e2) / n; n -= r.n
    return 100 * c1
rows = []
for k, gdf in api.groupby("detail"):
    if len(gdf) < 300: continue
    s_ = sva[sva.detail == k]
    rows.append(dict(group=k, n=len(gdf), pct_age20_39=100 * (gdf.ageband == "20-39").mean(), pct_TNBC=100 * (gdf.subtype == "Triple-negative").mean(),
                     pct_HER2pos=100 * gdf.subtype.isin(["HR+/HER2+", "HR−/HER2+"]).mean(),
                     pct_distant=100 * (gdf.stage == "Distant").sum() / (gdf.stage != "Unknown").sum(),
                     CIF5_bc=aj5(s_.t.values, s_.ev.values) if len(s_) > 200 else np.nan))
res["API_subgroups"] = pd.DataFrame(rows).sort_values("n", ascending=False)

# =============================================================== 3. MEDIATION (IORW)
conf = {"agegrp": "45-49", "registry": cl.registry.mode()[0]}
blocks = {
    "Tumor biology (subtype, grade)": {"subtype": "HR+/HER2−", "grade": "G1"},
    "Stage at diagnosis": {"stage": "Localized"},
    "Area SES & marital (county income, rurality, marital)": {"income_cat": "≥$100k", "metro": "Large metro", "married": "Married"},
    "Treatment (surgery, radiation, chemo, time to treatment)": {"surgery": "BCS", "radiation": "Yes", "chemo": "Yes", "ttt_cat": "0-30"},
}
cum_sets = {"Biology": ["Tumor biology (subtype, grade)"],
            "+ Stage": ["Tumor biology (subtype, grade)", "Stage at diagnosis"],
            "+ SES & marital": ["Tumor biology (subtype, grade)", "Stage at diagnosis", "Area SES & marital (county income, rurality, marital)"],
            "+ Treatment (all)": list(blocks)}
single_sets = {k: [k] for k in blocks}

def iorw_cox(d, med_refs):
    A = (d.race == EXPO).astype(float).values
    Xc, _ = design(d, conf, numeric=["yr"])
    Xm, _ = design(d, med_refs, numeric=[])
    nm_ = Xm.shape[1]
    g_, _ = logit_w(A, np.c_[Xm, Xc])
    lp_m = Xm @ g_[1:1 + nm_]
    w = np.where(A == 1, np.exp(-lp_m), 1.0)
    e1 = (d.ev.values == 1).astype(float)
    bt, _ = cox_w(d.t.values, e1, np.c_[A, Xc])
    bd, _ = cox_w(d.t.values, e1, np.c_[A, Xc], w=w)
    return np.array([bt[0], bd[0]])

def iorw_logit(d, med_refs):
    A = (d.race == EXPO).astype(float).values
    Xc, _ = design(d, conf, numeric=["yr"])
    Xm, _ = design(d, med_refs, numeric=[])
    nm_ = Xm.shape[1]
    g_, _ = logit_w(A, np.c_[Xm, Xc]); w = np.where(A == 1, np.exp(-(Xm @ g_[1:1 + nm_])), 1.0)
    y = d.dist.values.astype(float)
    bt, _ = logit_w(y, np.c_[A, Xc]); bd, _ = logit_w(y, np.c_[A, Xc], w)
    return np.array([bt[1], bd[1]])

def summarize(fun, d, refs_list, B):
    med_refs = {}
    for blk in refs_list: med_refs.update(blocks[blk])
    est = fun(d, med_refs); reps = []
    for _ in range(B):
        ix = rng.integers(0, len(d), len(d)); reps.append(fun(d.iloc[ix], med_refs))
    reps = np.array(reps)
    pm = lambda x: 100 * (x[0] - x[1]) / x[0]
    pms = np.array([pm(r) for r in reps])
    return dict(total=np.exp(est[0]), total_lo=np.exp(np.percentile(reps[:, 0], 2.5)), total_hi=np.exp(np.percentile(reps[:, 0], 97.5)),
                direct=np.exp(est[1]), direct_lo=np.exp(np.percentile(reps[:, 1], 2.5)), direct_hi=np.exp(np.percentile(reps[:, 1], 97.5)),
                indirect=np.exp(est[0] - est[1]), pct_mediated=pm(est), pm_lo=np.percentile(pms, 2.5), pm_hi=np.percentile(pms, 97.5))

med_rows = []
for EXPO in ["NH Black", "Hispanic"]:
    d = sv[sv.race.isin([EXPO, "NH White"])].copy()
    d["ttt_cat"] = cl.set_index("Patient ID").loc[d["Patient ID"], "ttt_cat"].values
    for k in ["surgery", "radiation", "chemo"]:
        d[k] = cl.set_index("Patient ID").loc[d["Patient ID"], k].values
    for label, sets in list(cum_sets.items()) + [(f"Only: {k}", v) for k, v in single_sets.items()]:
        t0 = time.time()
        r = summarize(iorw_cox, d, sets, B=BM)
        med_rows.append(dict(outcome="BC death (HR)", exposure=f"{EXPO} vs NH White", mediators=label, **r)); print(EXPO, label, round(time.time() - t0), "s", round(r["pct_mediated"], 1))
    # distant stage at diagnosis: biology + SES (stage is the outcome; treatment is post-diagnosis)
    dk = cl[cl.race.isin([EXPO, "NH White"]) & (cl.stage != "Unknown")].copy(); dk["dist"] = (dk.stage == "Distant").astype(int); dk["yr"] = dk.year - 2010
    for label, sets in [("Biology", ["Tumor biology (subtype, grade)"]), ("SES & marital", ["Area SES & marital (county income, rurality, marital)"]),
                        ("Biology + SES & marital", ["Tumor biology (subtype, grade)", "Area SES & marital (county income, rurality, marital)"])]:
        r = summarize(iorw_logit, dk, sets, B=BM)
        med_rows.append(dict(outcome="Distant stage at dx (OR)", exposure=f"{EXPO} vs NH White", mediators=label, **r)); print(EXPO, "stage", label, round(r["pct_mediated"], 1))
res["Mediation_IORW"] = pd.DataFrame(med_rows)

with pd.ExcelWriter("output/results_v3_additions.xlsx") as xw:
    for k, v in res.items(): v.round(4).to_excel(xw, sheet_name=k[:31], index=False)
pd.set_option("display.width", 250)
for k, v in res.items(): print("=====", k); print(v.round(3).to_string())
