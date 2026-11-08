import time, numpy as np, pandas as pd
from lib2 import *
g, cl = load("data")
print(cl.shape, cl.surgery.value_counts().to_dict(), cl.coc.value_counts().to_dict(), cl.ttt_cat.value_counts().to_dict())
sv = cl[(cl.year<=2018) & cl["Survival months flag"].isin(["Complete dates are available and there are more than 0 days of survival","Incomplete dates are available and there cannot be zero days of follow-up"]) & (cl["SEER cause-specific death classification"]!="Dead (missing/unknown COD)")].copy()
sv["t"]=pd.to_numeric(sv["Survival months"]).astype(float)
sv["ev"]=np.select([sv["SEER cause-specific death classification"].eq("Dead (attributable to this cancer dx)"), sv["SEER other cause of death classification"].eq("Dead (attributable to causes other than this cancer dx)")],[1,2],0)
sv["yr"]=sv.year-2010
refs={"race":"NH White","subtype":"HR+/HER2−","stage":"Localized","grade":"G1","agegrp":"45-49","income_cat":"≥$100k","metro":"Large metro","married":"Married"}
X,names=design(sv,refs,numeric=["yr"])
t0=time.time(); b,V=cox_w(sv.t.values,(sv.ev.values==1).astype(float),X); print("cox",time.time()-t0)
G=km_censor(sv.t.values, sv.ev.values)
t0=time.time(); bf,Vf=cox_w(sv.t.values,(sv.ev.values==1).astype(float),X,comp=(sv.ev.values==2).astype(float),G=G); print("fg",time.time()-t0)
se=np.sqrt(np.diag(V)); sef=np.sqrt(np.diag(Vf))
out=pd.DataFrame({"term":names,"csHR":np.exp(b),"cs_lo":np.exp(b-1.96*se),"cs_hi":np.exp(b+1.96*se),"sHR":np.exp(bf),"fg_lo":np.exp(bf-1.96*sef),"fg_hi":np.exp(bf+1.96*sef)})
print(out.round(3).to_string())
sv.to_pickle("output/sv.pkl"); cl.to_pickle("output/cl.pkl")
out.to_csv("output/fine_gray_overall.csv",index=False)
