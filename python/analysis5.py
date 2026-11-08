import numpy as np, pandas as pd
from lib2 import *
cl = pd.read_pickle("output/cl.pkl")
R = ["NH White", "NH Black", "Hispanic", "NH API"]
res = {}
# ---------- A. time-to-treatment diagnostics
d = cl[cl.ttt_days.notna() & cl.stage.isin(["Localized", "Regional"])].copy()
seq = d["RX Summ--Systemic/Sur Seq (2007+)"]
d["first"] = np.select([seq.isin(["Systemic therapy before surgery", "Systemic therapy both before and after surgery"]),
                        seq.eq("Surgery both before and after systemic therapy"), d.surgery.isin(["BCS", "Mastectomy", "Surgery NOS"])],
                       ["Systemic first", "Surgery-systemic-surgery", "Surgery first"], "Other/none")
g = d.groupby("year").agg(n=("ttt_days", "size"), median_days=("ttt_days", "median"), pct_day0=("ttt_days", lambda x: 100 * (x == 0).mean()),
                          pct_day1_7=("ttt_days", lambda x: 100 * x.between(1, 7).mean()),
                          median_excl_day0=("ttt_days", lambda x: x[x > 0].median()), pct_gt60=("ttt_days", lambda x: 100 * (x > 60).mean()),
                          pct_seq_surg_sys_surg=("first", lambda x: 100 * (x == "Surgery-systemic-surgery").mean())).reset_index()
med_first = d[d.ttt_days > 0].groupby(["year", "first"]).ttt_days.median().unstack().reset_index()
res["TTT_by_year"] = g.merge(med_first, on="year")
# race ORs for >60 days, separately before and after 2021
rows = []
tt = d[d.race.isin(R)].copy(); tt["gt60"] = (tt.ttt_days > 60).astype(float)
for lab, sub in [("2010-2020", tt[tt.year <= 2020]), ("2021-2023", tt[tt.year >= 2021])]:
    X, nm = design(sub, {"race": "NH White", "agegrp": "45-49", "stage": "Localized", "subtype": "HR+/HER2−", "income_cat": "≥$100k", "metro": "Large metro", "married": "Married", "registry": sub.registry.mode()[0]}, numeric=[])
    X = np.c_[X, sub.year.values - 2010]
    b, V = logit_w(sub.gt60.values, X); se = np.sqrt(np.diag(V))[1:]
    for j, n_ in enumerate(nm):
        if n_.startswith("race"):
            rows.append(dict(period=lab, term=n_, OR=np.exp(b[1 + j]), lo=np.exp(b[1 + j] - 1.96 * se[j]), hi=np.exp(b[1 + j] + 1.96 * se[j]), n=len(sub)))
res["TTT_gt60_raceOR_by_period"] = pd.DataFrame(rows)

# ---------- B. stage-shift ITS: level-only vs level + post-slope
ks = cl[cl.stage != "Unknown"].copy(); ks["dist"] = (ks.stage == "Distant").astype(int)
def its(df, slope):
    df = df[df.year != 2020]
    a = df.groupby(["year", "agegrp", "race", "subtype"]).agg(n=("dist", "size"), d=("dist", "sum")).reset_index()
    Xc, _ = design(a, {"agegrp": "45-49", "race": "NH White", "subtype": "HR+/HER2−"}, numeric=[])
    post = (a.year >= 2021).astype(float).values
    cols = [a.year.values - 2010, post] + ([post * (a.year.values - 2021)] if slope else [])
    X = np.c_[np.column_stack(cols), Xc]
    y = np.r_[np.ones(len(a)), np.zeros(len(a))]; w = np.r_[a.d.values, (a.n - a.d).values]; XX = np.r_[X, X]
    b, V = logit_w(y, XX, w)
    out = dict(level_OR=np.exp(b[2]), level_lo=np.exp(b[2] - 1.96 * np.sqrt(V[2, 2])), level_hi=np.exp(b[2] + 1.96 * np.sqrt(V[2, 2])))
    if slope:
        out.update(slope_OR_per_yr=np.exp(b[3]), slope_lo=np.exp(b[3] - 1.96 * np.sqrt(V[3, 3])), slope_hi=np.exp(b[3] + 1.96 * np.sqrt(V[3, 3])))
        # average post-period excess (2021-2023) = level + mean(slope*0,1,2) -> exp(b2 + b3*1)
        g_ = np.zeros(len(b)); g_[2] = 1; g_[3] = 1; v = g_ @ V @ g_
        out.update(avg_excess_OR=np.exp(b[2] + b[3]), avg_lo=np.exp(b[2] + b[3] - 1.96 * np.sqrt(v)), avg_hi=np.exp(b[2] + b[3] + 1.96 * np.sqrt(v)))
    return out
rows = []
groups = [("All", ks)] + [(f"Race: {r}", ks[ks.race == r]) for r in R] + [(f"Subtype: {s}", ks[ks.subtype == s]) for s in ["HR+/HER2−", "HR+/HER2+", "HR−/HER2+", "Triple-negative"]] + [(f"Age {b_}", ks[ks.ageband == b_]) for b_ in ["20-39", "40-49"]]
for lab, df in groups:
    a1 = its(df, False); a2 = its(df, True)
    rows.append(dict(group=lab, **{f"levelonly_{k}": v for k, v in a1.items()}, **a2))
res["StageShift_ITS_slope"] = pd.DataFrame(rows)
with pd.ExcelWriter("output/results_v6_ttt_its_slope.xlsx") as xw:
    for k, v in res.items(): v.round(4).to_excel(xw, sheet_name=k[:31], index=False)
pd.set_option("display.width", 250)
for k, v in res.items(): print("=====", k); print(v.round(3).to_string())
