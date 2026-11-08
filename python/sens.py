import os, sys, numpy as np, pandas as pd
sys.argv=["x","data","output"]
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis.py")).read().split("# ---------------------------------------------------------------- 2. CASE LISTING")[0]
exec(src)
# sensitivity: redistribute unknown/not-available subtype proportionally within age x year
r = r12.copy()
tot = r.groupby(["age","year"])["count"].sum().rename("tot")
known = r[r["sub"]<=3].groupby(["age","year"])["count"].sum().rename("known")
r = r.join(tot, on=["age","year"]).join(known, on=["age","year"])
r["count"] = np.where(r["sub"]<=3, r["count"]*r.tot/r.known, 0)
rows=[]
for bname, ages in bands.items():
    for k,s in enumerate(SUBS):
        g=r[(r["sub"]==k)&r.age.isin(ages)][["age","year","count","pop"]]
        t=trend(adj(g,ages)); rows.append(dict(group=bname, subtype=SUBLAB[s], **{k2:t[k2] for k2 in ["rate_2010","rate_2019","rate_2023","APC_2010_19","APC_lo","APC_hi","OE_2021_23","OE_lo","OE_hi"]}))
    # 2014-2019 pre-trend alternative for main
sens=pd.DataFrame(rows); print(sens.round(3).to_string())
sens.to_csv("output/sens_unknown_redistributed.csv",index=False)
def trend_break(df):
    d=df[df.rate>0]; pre=d[d.year.between(2010,2019)]
    x=pre.year.values-2010; X=np.c_[np.ones_like(x),x,(pre.year.values>=2018).astype(float)]
    w=(pre.rate.values/pre.se.values)**2; XtW=X.T*w; cov0=np.linalg.inv(XtW@X); b=cov0@XtW@np.log(pre.rate.values)
    res=np.log(pre.rate.values)-X@b; s2=max(1,(w*res**2).sum()/(len(x)-3)); cov=cov0*s2
    post=d[d.year.between(2021,2023)]; Xp=np.c_[np.ones(len(post)),post.year.values-2010,np.ones(len(post))]
    a=np.ones(len(post))/len(post); diff=a@(np.log(post.rate.values)-Xp@b)
    v=a@Xp@cov@Xp.T@a+(a**2*(post.se.values/post.rate.values)**2).sum()
    return np.exp(diff),np.exp(diff-1.96*np.sqrt(v)),np.exp(diff+1.96*np.sqrt(v)), np.exp(b[2])
out=[]
for bname in ["Ages 20-49","Ages 20-39","Ages 40-49"]:
    for s in SUBS+["All subtypes"]:
        oe,lo,hi,step=trend_break(series[(bname,s)]); out.append(dict(group=bname,subtype=SUBLAB.get(s,s),OE_break=oe,lo=lo,hi=hi,step_2018=step))
print(pd.DataFrame(out).round(3).to_string())
pd.DataFrame(out).to_csv("output/sens_2018_break.csv",index=False)
