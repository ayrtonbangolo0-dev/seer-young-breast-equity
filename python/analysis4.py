"""Neoadjuvant chemotherapy use in high-risk (T2+ or N+) TNBC / HER2+ by race over time, and chemo-resistant histologies by race."""
import numpy as np, pandas as pd
from lib2 import *
cl = pd.read_pickle("output/cl.pkl")
R = ["NH White", "NH Black", "Hispanic", "NH API"]
res = {}

# ------------------------------------------------ cohort
size = pd.to_numeric(cl["Tumor Size Over Time Recode (1988+)"], errors="coerce")
size = size.where(size <= 400)   # 990+ codes are not sizes
stg = cl["Combined Summary Stage with Expanded Regional Codes (2004+)"]
cl["nodepos"] = stg.isin(["Regional lymph nodes involved only", "Regional by both direct extension and lymph node involvement"])
cl["t4ext"] = stg.isin(["Regional by direct extension only", "Regional by both direct extension and lymph node involvement"])
cl["t2plus"] = (size > 20) | cl.t4ext
cl["highrisk"] = np.where(cl.t2plus | cl.nodepos, True, np.where(size.notna(), False, np.nan))
seq = cl["RX Summ--Systemic/Sur Seq (2007+)"]
cl["preop_sys"] = seq.isin(["Systemic therapy before surgery", "Systemic therapy both before and after surgery"])
cl["neoadj_chemo"] = cl.preop_sys & cl.chemo.eq("Yes")
cl["neoadj_resp"] = cl["Response to Neoadjuvant Therapy Recode (2010+)"].str.startswith("Stated")
cl["sub2"] = np.select([cl.subtype.eq("Triple-negative"), cl.subtype.isin(["HR+/HER2+", "HR−/HER2+"])], ["Triple-negative", "HER2+"], "Other")

hr = cl[(cl.sub2 != "Other") & cl.stage.isin(["Localized", "Regional"]) & (cl.highrisk == True) & cl.race.isin(R)].copy()
flow = [("TNBC or HER2+, non-metastatic, T2+ or N+, 4 main race groups", len(hr))]
res["NoSurgery_share"] = hr.groupby(["sub2", "race"]).surgery.apply(lambda s: 100 * s.isin(["None"]).mean()).rename("pct_no_surgery").reset_index()
hr = hr[hr.surgery.isin(["BCS", "Mastectomy", "Surgery NOS"])]
flow.append(("... with surgery (denominator)", len(hr)))
res["Flow"] = pd.DataFrame(flow, columns=["step", "n"])

# yearly proportions
yr = hr.groupby(["sub2", "race", "year"]).agg(n=("neoadj_chemo", "size"), neo=("neoadj_chemo", "mean"), neo_resp=("neoadj_resp", "mean")).reset_index(); yr["pct"] = 100 * yr.neo; yr["pct_resp_def"] = 100 * yr.neo_resp; yr = yr.drop(columns="neo_resp")
res["Neoadj_by_year"] = yr.drop(columns="neo")
# approval eras
def era(s, y):
    if s == "HER2+": return "2010-2013" if y <= 2013 else ("2014-2018 (pertuzumab)" if y <= 2018 else "2019-2023 (T-DM1/KATHERINE)")
    return "2010-2016" if y <= 2016 else ("2017-2021 (CREATE-X era)" if y <= 2021 else "2022-2023 (KEYNOTE-522)")
hr["era"] = [era(s, y) for s, y in zip(hr.sub2, hr.year)]
e = hr.groupby(["sub2", "race", "era"]).agg(n=("neoadj_chemo", "size"), pct=("neoadj_chemo", "mean"), pct_resp_def=("neoadj_resp", "mean")).reset_index(); e["pct"] *= 100; e["pct_resp_def"] *= 100
res["Neoadj_by_era"] = e
# adjusted models: within subtype, logistic neoadj ~ race * era + age + T2/N + grade + income + metro + registry
rows = []
for s in ["Triple-negative", "HER2+"]:
    d = hr[hr.sub2 == s].copy()
    d["tn"] = np.select([d.t2plus & d.nodepos, d.t2plus], ["T2+N+", "T2+N0"], "T1N+")
    eras = sorted(d.era.unique())
    # race effect within each era (adjusted)
    for er in eras:
        de = d[d.era == er]
        X, nm = design(de, {"race": "NH White", "agegrp": "45-49", "tn": "T2+N0", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro", "registry": de.registry.mode()[0]}, numeric=[])
        X = np.c_[X, de.year.values - 2010]
        for defn, yv in [("seq+chemo", de.neoadj_chemo), ("stated response", de.neoadj_resp)]:
            b, V = logit_w(yv.values.astype(float), X); se = np.sqrt(np.diag(V))[1:]
            for j, n_ in enumerate(nm):
                if n_.startswith("race"):
                    rows.append(dict(subtype=s, era=er, definition=defn, term=n_, OR=np.exp(b[1 + j]), lo=np.exp(b[1 + j] - 1.96 * se[j]), hi=np.exp(b[1 + j] + 1.96 * se[j]), n=len(de)))
    # change from previous era to latest era, Black vs White difference (interaction)
    last, prev = eras[-1], eras[-2]
    dd = d[d.era.isin([prev, last]) & d.race.isin(["NH White", "NH Black"])].copy()
    dd["late"] = (dd.era == last).astype(float); dd["blk"] = (dd.race == "NH Black").astype(float)
    Xc, _ = design(dd, {"agegrp": "45-49", "tn": "T2+N0", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro"}, numeric=[])
    b, V = logit_w(dd.neoadj_chemo.values.astype(float), np.c_[dd.blk, dd.late, dd.blk * dd.late, Xc]); se = np.sqrt(V[3, 3])
    rows.append(dict(subtype=s, era=f"{prev} -> {last}", term="interaction: Black x latest era (ratio of ORs)", OR=np.exp(b[3]), lo=np.exp(b[3] - 1.96 * se), hi=np.exp(b[3] + 1.96 * se), n=len(dd)))
    # same for Hispanic
    dd = d[d.era.isin([prev, last]) & d.race.isin(["NH White", "Hispanic"])].copy()
    dd["late"] = (dd.era == last).astype(float); dd["h"] = (dd.race == "Hispanic").astype(float)
    Xc, _ = design(dd, {"agegrp": "45-49", "tn": "T2+N0", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro"}, numeric=[])
    b, V = logit_w(dd.neoadj_chemo.values.astype(float), np.c_[dd.h, dd.late, dd.h * dd.late, Xc]); se = np.sqrt(V[3, 3])
    rows.append(dict(subtype=s, era=f"{prev} -> {last}", term="interaction: Hispanic x latest era (ratio of ORs)", OR=np.exp(b[3]), lo=np.exp(b[3] - 1.96 * se), hi=np.exp(b[3] + 1.96 * se), n=len(dd)))
res["Neoadj_adjOR"] = pd.DataFrame(rows)

# ------------------------------------------------ histology
h = pd.to_numeric(cl["Histologic Type ICD-O-3"], errors="coerce")
cl["histo"] = np.select(
    [h.eq(8530), h.between(8570, 8576) | h.isin([8032, 8033, 8052]), h.isin([8520, 8524]), h.eq(8522), h.isin([8480, 8481]),
     h.isin([8510, 8512, 8513]), h.eq(8401), h.eq(8200), h.eq(8507), h.isin([8503, 8504, 8050, 8260]), h.eq(8211), h.isin([8500, 8523, 8010, 8140])],
    ["Inflammatory", "Metaplastic", "Lobular", "Mixed ductal-lobular", "Mucinous", "Medullary", "Apocrine", "Adenoid cystic", "Micropapillary",
     "Papillary", "Tubular", "Ductal/NST"], "Other")
low_resp = ["Metaplastic", "Lobular", "Mixed ductal-lobular", "Mucinous", "Apocrine", "Adenoid cystic", "Tubular", "Papillary", "Inflammatory"]
c4 = cl[cl.race.isin(R) & cl.subtype.isin(["Triple-negative", "HR−/HER2+", "HR+/HER2+", "HR+/HER2−"])].copy()
c4["sub2"] = np.where(c4.subtype.isin(["HR+/HER2+", "HR−/HER2+"]), "HER2+", c4.subtype)
dist = c4.groupby(["sub2", "race"]).histo.value_counts(normalize=True).mul(100).rename("pct").reset_index()
res["Histology_by_race"] = dist.pivot_table(index=["sub2", "histo"], columns="race", values="pct").reset_index()
# adjusted OR for key histologies (Black vs White), within subtype
rows = []
for s in ["Triple-negative", "HER2+", "HR+/HER2−"]:
    d = c4[c4.sub2 == s]
    for hh in ["Metaplastic", "Lobular", "Inflammatory", "Mucinous", "Apocrine", "Medullary"]:
        y = (d.histo == hh).astype(float).values
        if y.sum() < 40: continue
        X, nm = design(d, {"race": "NH White", "agegrp": "45-49"}, numeric=[]); X = np.c_[X, d.year - 2010]
        b, V = logit_w(y, X); se = np.sqrt(np.diag(V))[1:]
        for j, n_ in enumerate(nm):
            if n_.startswith("race"):
                rows.append(dict(subtype=s, histology=hh, term=n_, OR=np.exp(b[1 + j]), lo=np.exp(b[1 + j] - 1.96 * se[j]), hi=np.exp(b[1 + j] + 1.96 * se[j]), events=int(y.sum())))
res["Histology_adjOR"] = pd.DataFrame(rows)
# CR by histology (shows which respond poorly) and CR race-OR with / without histology adjustment
neo = cl["Response to Neoadjuvant Therapy Recode (2010+)"]
cl["neo_any"] = neo.str.startswith("Stated"); cl["cr"] = neo.eq("Stated as complete response (CR)").astype(float)
na = cl[cl.race.isin(R) & cl.neo_any & cl.subtype.isin(["Triple-negative", "HR−/HER2+", "HR+/HER2+"])].copy()
na["sub2"] = np.where(na.subtype == "Triple-negative", "Triple-negative", "HER2+")
res["CR_by_histology"] = na.groupby(["sub2", "histo"]).agg(n=("cr", "size"), cr_pct=("cr", "mean")).reset_index().assign(cr_pct=lambda d: 100 * d.cr_pct).query("n >= 30")
na["histo2"] = np.where(na.histo.isin(["Ductal/NST", "Medullary", "Micropapillary", "Other"]), na.histo, "Low-response histology")
na["histo2"] = np.where(na.histo.isin(["Medullary", "Micropapillary"]), "Other", na.histo2)
na["yr"] = na.year - 2010
rows = []
for s in ["Triple-negative", "HR−/HER2+", "HR+/HER2+"]:
    d = na[na.subtype == s]
    base = {"race": "NH White", "agegrp": "45-49", "stage": "Localized", "grade": "G1", "income_cat": "≥$100k", "metro": "Large metro", "registry": d.registry.mode()[0]}
    for lab, spec in [("without histology", base), ("+ histology", dict(base, histo2="Ductal/NST"))]:
        X, nm = design(d, spec, numeric=["yr"]); b, V = logit_w(d.cr.values, X); se = np.sqrt(np.diag(V))[1:]
        j = nm.index("race: NH Black (ref NH White)")
        rows.append(dict(subtype=s, model=lab, OR_Black=np.exp(b[1 + j]), lo=np.exp(b[1 + j] - 1.96 * se[j]), hi=np.exp(b[1 + j] + 1.96 * se[j]), n=len(d),
                         pct_lowresp_hist_Black=100 * (d[d.race == "NH Black"].histo2 == "Low-response histology").mean(),
                         pct_lowresp_hist_White=100 * (d[d.race == "NH White"].histo2 == "Low-response histology").mean()))
res["CR_OR_histology_adjusted"] = pd.DataFrame(rows)

with pd.ExcelWriter("output/results_v5_neoadj_use_histology.xlsx") as xw:
    for k, v in res.items(): v.round(4).to_excel(xw, sheet_name=k[:31], index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
for k, v in res.items():
    if k == "Neoadj_by_year": continue
    print("=====", k); print(v.round(2).to_string())
print(yr.pivot_table(index=["sub2", "year"], columns="race", values=["pct","pct_resp_def"]).round(1).to_string())
