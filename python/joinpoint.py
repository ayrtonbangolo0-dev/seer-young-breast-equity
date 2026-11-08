"""Joinpoint-equivalent segmented log-linear regression (Kim et al., Stat Med 2000).
- log(rate) ~ continuous piecewise-linear in year, joinpoints at observed years (grid search)
- heteroscedastic weights = 1 / var(log rate) = (rate / SE)^2
- 2020 excluded; max 2 joinpoints; >=2 observations from each end and between joinpoints
- model selection: sequential permutation tests (999 permutations), Bonferroni-adjusted overall alpha 0.05
- segment APC and AAPC with t-based 95% CIs (df = n - 2k - 2), conditional on the selected joinpoints
"""
import itertools, os, sys, numpy as np, pandas as pd
from scipy import stats
rng = np.random.default_rng(522)
NPERM = 999

def design(x, taus):
    return np.column_stack([np.ones_like(x), x] + [np.clip(x - t, 0, None) for t in taus])

def wls(X, y, w):
    W = np.sqrt(w)
    b, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    r = y - X @ b
    return b, (w * r ** 2).sum()

def candidates(x, k):
    n = len(x)
    if k == 0: return [()]
    idx = range(2, n - 2)          # >=2 obs from either end (joinpoint at an observed x, obs i counted on both sides)
    out = []
    for c in itertools.combinations(idx, k):
        if all(c[j + 1] - c[j] >= 2 for j in range(k - 1)): out.append(tuple(x[i] for i in c))
    return out

def best_fit(x, y, w, k):
    best = None
    for taus in candidates(x, k):
        b, sse = wls(design(x, taus), y, w)
        if best is None or sse < best[2]: best = (taus, b, sse)
    return best

def perm_test(x, y, w, k0, k1):
    t0, b0, s0 = best_fit(x, y, w, k0); t1, b1, s1 = best_fit(x, y, w, k1)
    T = (s0 - s1) / s1
    fitted0 = design(x, t0) @ b0; res = (y - fitted0) * np.sqrt(w)   # standardized residuals
    cnt = 0
    for _ in range(NPERM):
        ystar = fitted0 + rng.permutation(res) / np.sqrt(w)
        a0 = best_fit(x, ystar, w, k0)[2]; a1 = best_fit(x, ystar, w, k1)[2]
        if (a0 - a1) / a1 >= T: cnt += 1
    return (cnt + 1) / (NPERM + 1)

def select(x, y, w, kmax):
    ka, kb, alpha = 0, kmax, 0.05
    while ka < kb:
        p = perm_test(x, y, w, ka, kb)
        if p < alpha / (kmax - ka): ka += 1      # Bonferroni-style adjustment (Kim et al.)
        else: kb -= 1
    return ka

def summarize(x, y, w, k, years_for_aapc):
    taus, b, sse = best_fit(x, y, w, k)
    X = design(x, taus); n, p = X.shape
    df = n - 2 * k - 2
    sigma2 = sse / df
    cov = np.linalg.inv((X.T * w) @ X) * sigma2
    tq = stats.t.ppf(0.975, df)
    # segment slopes = cumulative sums of b[1], b[2], ...
    L = np.zeros((k + 1, p))
    for s in range(k + 1): L[s, 1:2 + s] = 1
    slopes = L @ b; se = np.sqrt(np.diag(L @ cov @ L.T))
    bounds = [x.min()] + list(taus) + [x.max()]
    segs = []
    for s in range(k + 1):
        segs.append(dict(segment=f"{int(bounds[s])}-{int(bounds[s+1])}", APC=100 * (np.exp(slopes[s]) - 1),
                         lo=100 * (np.exp(slopes[s] - tq * se[s]) - 1), hi=100 * (np.exp(slopes[s] + tq * se[s]) - 1),
                         p=2 * stats.t.sf(abs(slopes[s] / se[s]), df)))
    aapcs = {}
    for (a, z) in years_for_aapc:
        wts = np.zeros(k + 1)
        for s in range(k + 1):
            lo_, hi_ = max(bounds[s], a), min(bounds[s + 1], z)
            wts[s] = max(0, hi_ - lo_)
        wts /= wts.sum()
        l = wts @ L; est = l @ b; v = l @ cov @ l
        aapcs[f"AAPC {a}-{z}"] = (100 * (np.exp(est) - 1), 100 * (np.exp(est - tq * np.sqrt(v)) - 1), 100 * (np.exp(est + tq * np.sqrt(v)) - 1))
    return taus, segs, aapcs

if __name__ == "__main__":
    sys.argv = ["x", "data", "output"]
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis.py")).read().split("def trend(df):")[0]
    g = {}; exec(src, g)
    series = g["series"]
    keep = [("Ages 20-49", "All subtypes"), ("Ages 20-39", "All subtypes"), ("Ages 40-49", "All subtypes")] + \
           [("Ages 20-49", s) for s in ["HR+/HER2-", "HR+/HER2+", "HR-/HER2+", "HR-/HER2-"]] + \
           [(r, "All subtypes") for r in ["NH White", "NH Black", "Hispanic", "NH API", "NH AIAN"]] + \
           [("Ages 20-39", "HR-/HER2-"), ("Ages 40-49", "HR-/HER2-")]
    rows = []
    for grp, s in keep:
        d = series[(grp, s)].dropna(subset=["rate"]); d = d[(d.year != 2020) & (d.rate > 0)].sort_values("year")
        x = d.year.values.astype(float); y = np.log(d.rate.values); w = (d.rate.values / d.se.values) ** 2
        k = select(x, y, w, kmax=2)
        taus, segs, aapcs = summarize(x, y, w, k, [(2010, 2019), (2010, 2023)])
        rows.append(dict(group=grp, subtype=g["SUBLAB"].get(s, s), n_points=len(x), joinpoints=k,
                         joinpoint_years=", ".join(str(int(t)) for t in taus) or "none",
                         segments="; ".join(f"{q['segment']}: {q['APC']:.2f} ({q['lo']:.2f} to {q['hi']:.2f})" + ("*" if q["p"] < 0.05 else "") for q in segs),
                         **{f"{k_} (95% CI)": f"{v[0]:.2f} ({v[1]:.2f} to {v[2]:.2f})" for k_, v in aapcs.items()},
                         **{f"{k_}_est": v[0] for k_, v in aapcs.items()}))
        print(rows[-1]["group"], rows[-1]["subtype"], k, rows[-1]["joinpoint_years"], rows[-1]["segments"], rows[-1]["AAPC 2010-2019 (95% CI)"], rows[-1]["AAPC 2010-2023 (95% CI)"], flush=True)
    pd.DataFrame(rows).to_csv("output/joinpoint_results.csv", index=False)
