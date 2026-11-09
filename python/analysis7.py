"""Revision: bootstrap reruns with >=500 replicates (PART env: black | hispanic | fg | crmed)."""
import numpy as np, pandas as pd, time
from lib2 import *
rng = np.random.default_rng(20251108)
import os
B = int(os.environ.get("BOOT", "500")); BM = B; PART = os.environ["PART"]
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

res = {}
if PART == "fg":
    res["FineGray_overall"] = boot(lambda d: fg_fit(d, refs), sv).reset_index().rename(columns={"index": "term", "est": "sHR"})
    rows = []
    for s in ["HR+/HER2−", "HR+/HER2+", "HR−/HER2+", "Triple-negative"]:
        d = sv[sv.subtype == s]; r2 = {k: v for k, v in refs.items() if k != "subtype"}
        f = boot(lambda x: fg_fit(x, r2), d); f = f[f.index.str.startswith("race")]; f.insert(0, "subtype", s)
        rows.append(f.reset_index().rename(columns={"index": "term", "est": "sHR"})); print(s, flush=True)
    res["FineGray_race_by_subtype"] = pd.concat(rows)

# ---- mediation definitions (from analysis2.py)

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


if PART in ("black", "hispanic"):
    EXPO = {"black": "NH Black", "hispanic": "Hispanic"}[PART]; med_rows = []
    d = sv[sv.race.isin([EXPO, "NH White"])].copy()
    d["ttt_cat"] = cl.set_index("Patient ID").loc[d["Patient ID"], "ttt_cat"].values
    for k in ["surgery", "radiation", "chemo"]:
        d[k] = cl.set_index("Patient ID").loc[d["Patient ID"], k].values
    for label, sets in cum_sets.items():
        t0 = time.time(); r = summarize(iorw_cox, d, sets, B=B)
        med_rows.append(dict(outcome="BC death (HR)", exposure=f"{EXPO} vs NH White", mediators=label, B=B, **r)); print(EXPO, label, round(time.time() - t0), "s", flush=True)
    dk = cl[cl.race.isin([EXPO, "NH White"]) & (cl.stage != "Unknown")].copy(); dk["dist"] = (dk.stage == "Distant").astype(int); dk["yr"] = dk.year - 2010
    for label, sets in [("Biology", ["Tumor biology (subtype, grade)"]), ("SES & marital", ["Area SES & marital (county income, rurality, marital)"]),
                        ("Biology + SES & marital", ["Tumor biology (subtype, grade)", "Area SES & marital (county income, rurality, marital)"])]:
        r = summarize(iorw_logit, dk, sets, B=B)
        med_rows.append(dict(outcome="Distant stage at dx (OR)", exposure=f"{EXPO} vs NH White", mediators=label, B=B, **r)); print(EXPO, "stage", label, flush=True)
    res["Mediation_IORW"] = pd.DataFrame(med_rows)
if PART == "crmed":
    import subprocess
    env = dict(os.environ, BOOTM=str(B)); subprocess.run(["python", "python/analysis3.py"], env=env, check=True)
with pd.ExcelWriter(f"output/results_v8_boot500_{PART}.xlsx") as xw:
    for k, v in res.items(): v.round(4).to_excel(xw, sheet_name=k[:31], index=False)
for k, v in res.items(): print("=====", k); print(v.round(3).to_string())
