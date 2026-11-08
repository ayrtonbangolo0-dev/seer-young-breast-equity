"""Shared helpers: data prep (re-using analysis.py), weighted Cox, Fine-Gray (IPCW), weighted logistic."""
import sys, os, numpy as np, pandas as pd
from scipy import stats

def load(data_dir):
    sys.argv = ["x", data_dir, "/tmp/_unused"]
    src = open(os.path.join(os.path.dirname(__file__), "analysis.py")).read()
    prep = src.split("# ---------------------------------------------------------------- 2. CASE LISTING")[1] \
              .split("# ---- Table 1")[0]
    head = src.split("# ---------------------------------------------------------------- 1. INCIDENCE")[0]
    g = {}
    exec(head, g)
    g["D"] = data_dir
    exec(prep, g)
    cl = g["cl"]
    ex = pd.read_csv(os.path.join(data_dir, "extra.csv"), dtype=str, keep_default_na=False)
    ex = ex.drop(columns=[c for c in ["Year of diagnosis", "Site recode ICD-O-3/WHO 2008",
                                      "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)",
                                      "Age recode with single ages and 90+",
                                      "Race recode (with detailed Asian and Native Hawaiian other PI)"] if c in ex.columns])
    cl = cl.merge(ex, on="Patient ID", how="left", suffixes=("", "_x"))
    # ---- treatment / care variables
    s1 = pd.to_numeric(cl["RX Summ--Surg Prim Site (1998-2022)"], errors="coerce")
    s2 = cl["RX Summ--Surg Prim Site 2023 (2023+)"]
    # 2023+ codes: letter + 3 digits (e.g. A220 = partial mastectomy 22.0, A420 = total mastectomy 42.0, A000 none, A990 unknown)
    s2n = pd.to_numeric(s2.str.extract(r"^[A-Z](\d{3})$")[0], errors="coerce") / 10
    code = s1.fillna(s2n)
    cl["surgery"] = np.select([code.eq(0), code.between(19, 24.9), code.between(30, 80.9), code.eq(90)],
                              ["None", "BCS", "Mastectomy", "Surgery NOS"], "Unknown")
    rr = cl["Radiation recode"]
    cl["radiation"] = np.where(rr.str.contains("Beam|implant|isotope|Combination|Radiation, NOS", regex=True), "Yes", "No/Unknown")
    cl["chemo"] = np.where(cl["Chemotherapy recode (yes, no/unk)"].eq("Yes"), "Yes", "No/Unknown")
    ttt = pd.to_numeric(cl["Time from diagnosis to treatment in days recode"], errors="coerce")
    cl["ttt_days"] = ttt
    cl["ttt_cat"] = pd.cut(ttt, [-1, 30, 60, 90, 1e6], labels=["0-30", "31-60", "61-90", ">90"]).astype(object).where(ttt.notna(), "Unknown")
    coc = cl["CoC Accredited Flag (2018+)"]
    cl["coc"] = np.select([coc.str.startswith("ANALYTIC"), coc.str.startswith("NON-ANALYTIC"), coc.str.startswith("Abstract from facility WITHOUT")],
                          ["CoC (analytic)", "CoC (non-analytic)", "Non-CoC"], "N/A (pre-2018)")
    cl["registry"] = cl["SEER registry (with CA and GA as whole states)"]
    return g, cl

def dummies(df, specs):
    """specs: {col: (ref, [levels])}; returns X, names (drops levels with <min count)."""
    cols, names = [], []
    for v, (ref, levs) in specs.items():
        for l in levs:
            x = (df[v] == l).astype(float).values
            if x.sum() < 30: continue
            cols.append(x); names.append(f"{v}: {l} (ref {ref})")
    return cols, names

def auto_specs(df, cols_refs):
    out = {}
    for v, ref in cols_refs.items():
        levs = [l for l in sorted(map(str, df[v].dropna().unique())) if l != ref]
        out[v] = (ref, levs)
    return out

def design(df, cols_refs, numeric=()):
    cols, names = dummies(df, auto_specs(df, cols_refs))
    for c in numeric:
        cols.append(df[c].values.astype(float)); names.append(c)
    X = np.column_stack(cols)
    # drop collinear
    keep = []
    for j in range(X.shape[1]):
        if np.linalg.matrix_rank(np.c_[np.ones(len(X)), X[:, keep + [j]]]) == len(keep) + 2: keep.append(j)
    return X[:, keep], [names[j] for j in keep]

def _grouped_sums(X, r, grp, ng):
    p = X.shape[1]
    A0 = np.bincount(grp, weights=r, minlength=ng)
    A1 = np.zeros((ng, p))
    for j in range(p): A1[:, j] = np.bincount(grp, weights=X[:, j] * r, minlength=ng)
    A2 = np.zeros((ng, p, p))
    for j in range(p):
        for k in range(j, p):
            s = np.bincount(grp, weights=X[:, j] * X[:, k] * r, minlength=ng); A2[:, j, k] = s; A2[:, k, j] = s
    return A0, A1, A2

def cox_w(t, e, X, w=None, comp=None, G=None, maxit=60):
    """Weighted Breslow Cox. If comp (competing-event indicator) and G (censoring survival fn) are given,
    fits Fine-Gray: competing-event subjects stay in risk sets after their event with weight G(t)/G(T_i)."""
    n, p = X.shape
    w = np.ones(n) if w is None else w
    ut = np.unique(t); ng = len(ut); grp = np.searchsorted(ut, t)
    d = np.bincount(grp, weights=w * e, minlength=ng)
    sx = np.zeros((ng, p))
    for j in range(p): sx[:, j] = np.bincount(grp, weights=w * e * X[:, j], minlength=ng)
    if comp is not None:
        Gt = G(ut); GTi = np.maximum(G(t), 1e-6)
    b = np.zeros(p)
    for it in range(maxit):
        eta = X @ b; eta -= eta.max(); r = np.exp(eta)
        A0, A1, A2 = _grouped_sums(X, w * r, grp, ng)
        S0 = np.cumsum(A0[::-1])[::-1]; S1 = np.cumsum(A1[::-1], 0)[::-1]; S2 = np.cumsum(A2[::-1], 0)[::-1]
        if comp is not None:
            cw = w * r * comp / GTi
            C0, C1, C2 = _grouped_sums(X, cw, grp, ng)
            # strictly earlier competing events: cumulative up to previous group
            c0 = np.r_[0, np.cumsum(C0)[:-1]]; c1 = np.vstack([np.zeros(p), np.cumsum(C1, 0)[:-1]]); c2 = np.concatenate([np.zeros((1, p, p)), np.cumsum(C2, 0)[:-1]])
            S0 = S0 + Gt * c0; S1 = S1 + Gt[:, None] * c1; S2 = S2 + Gt[:, None, None] * c2
        m = d > 0
        mb = S1[m] / S0[m, None]
        g = sx[m].sum(0) - (d[m, None] * mb).sum(0)
        I = (d[m, None, None] * (S2[m] / S0[m, None, None] - mb[:, :, None] * mb[:, None, :])).sum(0)
        step = np.linalg.solve(I, g); mx = np.abs(step).max()
        if mx > 1: step /= mx
        b = b + step
        if mx < 1e-8: break
    return b, np.linalg.inv(I)

def km_censor(t, e_any):
    """KM estimate of censoring survival G(t) (censoring = no event of any type)."""
    ut = np.unique(t); grp = np.searchsorted(ut, t)
    n_at = np.cumsum(np.bincount(grp, minlength=len(ut))[::-1])[::-1]
    c = np.bincount(grp, weights=(e_any == 0).astype(float), minlength=len(ut))
    surv = np.cumprod(1 - c / n_at)
    surv_left = np.r_[1.0, surv[:-1]]   # G(t-)
    def G(x):
        k = np.searchsorted(ut, x, side="left")
        return surv_left[np.clip(k, 0, len(ut) - 1)]
    return G

def logit_w(y, X, w=None, maxit=50):
    n = len(y); w = np.ones(n) if w is None else w
    X1 = np.c_[np.ones(n), X]; b = np.zeros(X1.shape[1])
    for _ in range(maxit):
        p = 1 / (1 + np.exp(-(X1 @ b))); W = w * p * (1 - p)
        H = (X1.T * W) @ X1 + 1e-8 * np.eye(len(b)); gr = X1.T @ (w * (y - p))
        st = np.linalg.solve(H, gr); b += st
        if np.abs(st).max() < 1e-9: break
    return b, np.linalg.inv(H)

def fmt(est, lo, hi, d=2): return f"{est:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"
