"""Neoadjuvant complete response (CR) by race, and de novo metastatic survival trends by race and period."""
import numpy as np, pandas as pd, os
from lib2 import *
rng = np.random.default_rng(7)
BM = int(os.environ.get("BOOTM", "100"))
cl = pd.read_pickle("output/cl.pkl")
R = ["NH White", "NH Black", "Hispanic", "NH API"]
res = {}
VALID = ["Complete dates are available and there are more than 0 days of survival",
         "Incomplete dates are available and there cannot be zero days of follow-up"]

# ======================================================== A. NEOADJUVANT RESPONSE
neo = cl["Response to Neoadjuvant Therapy Recode (2010+)"]
cl["neo_resp"] = np.select([neo.eq("Stated as complete response (CR)"), neo.eq("Stated as partial response (PR)"),
                            neo.eq("Stated as no response (NR)"), neo.str.startswith("Stated as response to treatment")],
                           ["CR", "PR", "NR", "Response NOS"], "Not given/unknown")
cl["neo_any"] = cl.neo_resp != "Not given/unknown"
na = cl[cl.race.isin(R) & cl.neo_any & cl.subtype.isin(["Triple-negative", "HR−/HER2+", "HR+/HER2+", "HR+/HER2−"])].copy()
na["cr"] = (na.neo_resp == "CR").astype(float)
na["yr"] = na.year - 2010
# crude
crude = na.groupby(["subtype", "race"]).agg(n=("cr", "size"), cr_pct=("cr", "mean")).reset_index(); crude["cr_pct"] *= 100
# sensitivity denominator: exclude 'Response NOS'
na2 = na[na.neo_resp != "Response NOS"]
crude2 = na2.groupby(["subtype", "race"]).cr.mean().mul(100).rename("cr_pct_excl_NOS").reset_index()
res["CR_crude"] = crude.merge(crude2, on=["subtype", "race"])
# use of neoadjuvant therapy by race (denominator: stage I-III with known response field)
use = cl[cl.race.isin(R) & cl.stage.isin(["Localized", "Regional"]) & cl.subtype.isin(["Triple-negative", "HR−/HER2+", "HR+/HER2+", "HR+/HER2−"])]
res["Neoadjuvant_use"] = use.groupby(["subtype", "race"]).neo_any.mean().mul(100).rename("pct_neoadj").reset_index()
# adjusted OR of CR, by subtype
rows = []
for s in ["Triple-negative", "HR−/HER2+", "HR+/HER2+", "HR+/HER2−"]:
    for lab, d in [("all stated responses", na[na.subtype == s]), ("excluding response NOS", na2[na2.subtype == s])]:
        X, nm = design(d, {"race": "NH White", "agegrp": "45-49", "stage": "Localized", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro", "registry": d.registry.mode()[0]}, numeric=["yr"])
        b, V = logit_w(d.cr.values, X); se = np.sqrt(np.diag(V))[1:]
        for j, n_ in enumerate(nm):
            if n_.startswith("race"):
                rows.append(dict(subtype=s, denominator=lab, term=n_, OR=np.exp(b[1 + j]), lo=np.exp(b[1 + j] - 1.96 * se[j]), hi=np.exp(b[1 + j] + 1.96 * se[j]), n=len(d)))
res["CR_adjOR"] = pd.DataFrame(rows)

# CR -> survival, and CR as mediator of Black-White gap (neoadjuvant-treated TNBC + HER2+, dx 2010-2018)
sv = na[(na.year <= 2018) & na["Survival months flag"].isin(VALID) & (na["SEER cause-specific death classification"] != "Dead (missing/unknown COD)")].copy()
sv["t"] = pd.to_numeric(sv["Survival months"]).astype(float)
sv["ev"] = np.select([sv["SEER cause-specific death classification"].eq("Dead (attributable to this cancer dx)"),
                      sv["SEER other cause of death classification"].eq("Dead (attributable to causes other than this cancer dx)")], [1, 2], 0)
sv["resp"] = sv.neo_resp
rows = []
for s in ["Triple-negative", "HR−/HER2+", "HR+/HER2+"]:
    d = sv[sv.subtype == s]
    X, nm = design(d, {"resp": "CR", "race": "NH White", "agegrp": "45-49", "stage": "Localized", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro", "married": "Married"}, numeric=["yr"])
    b, V = cox_w(d.t.values, (d.ev.values == 1).astype(float), X); se = np.sqrt(np.diag(V))
    for j, n_ in enumerate(nm):
        if n_.startswith("resp") or n_.startswith("race: NH Black"):
            rows.append(dict(subtype=s, term=n_, HR=np.exp(b[j]), lo=np.exp(b[j] - 1.96 * se[j]), hi=np.exp(b[j] + 1.96 * se[j]), n=len(d), deaths=int((d.ev == 1).sum())))
res["CR_and_BCdeath"] = pd.DataFrame(rows)

# IORW: Black vs White, mediator = response category (+ stage, grade as confounder-mediators? keep stage/grade as confounders)
def iorw(d):
    A = (d.race == "NH Black").astype(float).values
    Xc, _ = design(d, {"agegrp": "45-49", "stage": "Localized", "grade": "G1", "subtype": "Triple-negative"}, numeric=["yr"])
    Xm, _ = design(d, {"resp": "CR"}, numeric=[])
    g_, _ = logit_w(A, np.c_[Xm, Xc]); w = np.where(A == 1, np.exp(-(Xm @ g_[1:1 + Xm.shape[1]])), 1.0)
    e1 = (d.ev.values == 1).astype(float)
    bt, _ = cox_w(d.t.values, e1, np.c_[A, Xc]); bd, _ = cox_w(d.t.values, e1, np.c_[A, Xc], w=w)
    return np.array([bt[0], bd[0]])
d = sv[sv.race.isin(["NH Black", "NH White"]) & sv.subtype.isin(["Triple-negative", "HR−/HER2+", "HR+/HER2+"])].reset_index(drop=True)
est = iorw(d); reps = np.array([iorw(d.iloc[rng.integers(0, len(d), len(d))]) for _ in range(BM)])
pm = lambda x: 100 * (x[0] - x[1]) / x[0]
res["CR_mediation_IORW"] = pd.DataFrame([dict(population="Neoadjuvant-treated TNBC/HER2+, Black vs White, dx 2010-2018", n=len(d),
    HR_total=np.exp(est[0]), total_lo=np.exp(np.percentile(reps[:, 0], 2.5)), total_hi=np.exp(np.percentile(reps[:, 0], 97.5)),
    HR_direct=np.exp(est[1]), direct_lo=np.exp(np.percentile(reps[:, 1], 2.5)), direct_hi=np.exp(np.percentile(reps[:, 1], 97.5)),
    pct_mediated=pm(est), pm_lo=np.percentile([pm(r) for r in reps], 2.5), pm_hi=np.percentile([pm(r) for r in reps], 97.5))])

# ======================================================== B. DE NOVO METASTATIC
m = cl[(cl.stage == "Distant") & (cl.year <= 2020) & cl.race.isin(R) & cl["Survival months flag"].isin(VALID)].copy()
m["t"] = pd.to_numeric(m["Survival months"]).astype(float)
m["death"] = (m["Vital status recode (study cutoff used)"] == "Dead").astype(float) if "Vital status recode (study cutoff used)" in m else np.nan
m["bc"] = m["SEER cause-specific death classification"].eq("Dead (attributable to this cancer dx)").astype(float)
m["per"] = np.select([m.year <= 2013, m.year <= 2017], ["2010-2013", "2014-2017"], "2018-2020")
m["sub3"] = np.select([m.subtype.eq("HR+/HER2−"), m.subtype.isin(["HR+/HER2+", "HR−/HER2+"]), m.subtype.eq("Triple-negative")], ["HR+/HER2−", "HER2+", "Triple-negative"], "Unknown")
mets = lambda s: m[f"SEER Combined Mets at DX-{s} (2010+)"].eq("Yes")
m["metsite"] = np.select([mets("brain"), mets("liver") | mets("lung"), mets("bone")], ["Brain", "Visceral", "Bone only/other"], "Other/unknown")
m["yr"] = m.year - 2010

def km(t, e):
    o = np.argsort(t); t, e = t[o], e[o]
    dfm = pd.DataFrame({"t": t, "e": e}).groupby("t").agg(n=("e", "size"), d=("e", "sum"))
    n = len(t); S = 1.0; T = [0]; Ss = [1.0]
    for tm, r in dfm.iterrows():
        S *= 1 - r.d / n; n -= r.n; T.append(tm); Ss.append(S)
    return np.array(T), np.array(Ss)
def med(T, S): k = np.where(S <= 0.5)[0]; return T[k[0]] if len(k) else np.nan
def at(T, S, h): return S[np.searchsorted(T, h, side="right") - 1]
rows, curves = [], []
for (s, r, p), g in m[m.sub3 != "Unknown"].groupby(["sub3", "race", "per"]):
    if len(g) < 40: continue
    T, S = km(g.t.values, g.death.values)
    rows.append(dict(subtype=s, race=r, period=p, n=len(g), deaths=int(g.death.sum()), median_OS=med(T, S), OS_3yr=100 * at(T, S, 36)))
    for tt_, ss_ in zip(T, S):
        if tt_ <= 72: curves.append(dict(subtype=s, race=r, period=p, month=tt_, surv=ss_))
res["MBC_KM"] = pd.DataFrame(rows)
# Cox (all-cause death): race x period within subtype; adjusted
rows = []
for s in ["HR+/HER2−", "HER2+", "Triple-negative"]:
    d = m[m.sub3 == s].copy()
    for r in R:
        for p in ["2014-2017", "2018-2020"]:
            pass
    # period effect within each race (improvement over time)
    for r in R:
        dr = d[d.race == r]
        if len(dr) < 150: continue
        X, nm = design(dr, {"per": "2010-2013", "agegrp": "45-49", "metsite": "Bone only/other", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro", "married": "Married"}, numeric=[])
        b, V = cox_w(dr.t.values, dr.death.values, X); se = np.sqrt(np.diag(V))
        for j, n_ in enumerate(nm):
            if n_.startswith("per"):
                rows.append(dict(subtype=s, model="period effect within race", race=r, term=n_, HR=np.exp(b[j]), lo=np.exp(b[j] - 1.96 * se[j]), hi=np.exp(b[j] + 1.96 * se[j]), n=len(dr)))
    # race gap within each period
    for p in ["2010-2013", "2014-2017", "2018-2020"]:
        dp = d[d.per == p]
        X, nm = design(dp, {"race": "NH White", "agegrp": "45-49", "metsite": "Bone only/other", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro", "married": "Married"}, numeric=["yr"])
        b, V = cox_w(dp.t.values, dp.death.values, X); se = np.sqrt(np.diag(V))
        for j, n_ in enumerate(nm):
            if n_.startswith("race"):
                rows.append(dict(subtype=s, model="race gap within period", race=p, term=n_, HR=np.exp(b[j]), lo=np.exp(b[j] - 1.96 * se[j]), hi=np.exp(b[j] + 1.96 * se[j]), n=len(dp)))
    # formal interaction test: Black x later period (2014-2020 vs 2010-13), White+Black only
    dd = d[d.race.isin(["NH White", "NH Black"])].copy(); dd["late"] = (dd.year >= 2014).astype(float); dd["blk"] = (dd.race == "NH Black").astype(float)
    Xc, _ = design(dd, {"agegrp": "45-49", "metsite": "Bone only/other", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro", "married": "Married"}, numeric=[])
    X = np.c_[dd.blk, dd.late, dd.blk * dd.late, Xc]
    b, V = cox_w(dd.t.values, dd.death.values, X); se = np.sqrt(V[2, 2])
    rows.append(dict(subtype=s, model="interaction Black x 2014-2020", race="", term="ratio of HRs", HR=np.exp(b[2]), lo=np.exp(b[2] - 1.96 * se), hi=np.exp(b[2] + 1.96 * se), n=len(dd)))
res["MBC_Cox"] = pd.DataFrame(rows)
res["MBC_metsite"] = m.groupby(["race"]).metsite.value_counts(normalize=True).mul(100).rename("pct").reset_index()
pd.DataFrame(curves).to_csv("output/mbc_km_curves.csv", index=False)

with pd.ExcelWriter("output/results_v4_neoadj_mbc.xlsx") as xw:
    for k, v in res.items(): v.round(4).to_excel(xw, sheet_name=k[:31], index=False)
pd.set_option("display.width", 250)
for k, v in res.items(): print("=====", k); print(v.round(3).to_string())
