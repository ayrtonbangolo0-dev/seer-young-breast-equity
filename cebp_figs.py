import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.6,
                     "xtick.major.width":0.6,"ytick.major.width":0.6,"lines.linewidth":1.2,"savefig.dpi":600})
BLUE="#2a78d6"; GREY="#8a8984"; DARK="#3a3a38"
OUT="cebp/"
# ---------- Figure 1: forest
rows=[("All women 20–49",1.044,0.958,1.138,"Overall"),
("NH White",0.942,0.820,1.082,"Race/ethnicity"),("NH Black",1.250,1.026,1.523,"Race/ethnicity"),("Hispanic",1.227,1.018,1.481,"Race/ethnicity"),
("NH Asian/Pacific Islander",0.922,0.722,1.177,"Race/ethnicity"),("NH American Indian/Alaska Native",0.451,0.181,1.122,"Race/ethnicity"),
("HR+/HER2−",1.012,0.893,1.148,"Subtype"),("HR+/HER2+",1.140,0.941,1.381,"Subtype"),("HR−/HER2+",1.000,0.761,1.315,"Subtype"),("Triple-negative",1.189,0.959,1.474,"Subtype"),
("Age 20–39",1.036,0.903,1.188,"Age"),("Age 40–49",1.048,0.940,1.169,"Age")]
fig,ax=plt.subplots(figsize=(6.5,4.2))
y=0; ys=[]; prev=None
for lab,o,l,h,g in rows:
    if prev and g!=prev: y+=0.6
    ys.append(y); prev=g; y+=1
ys=-np.array(ys, float)
for (lab,o,l,h,g),yy in zip(rows,ys):
    sig=l>1 or h<1; c=BLUE if sig else GREY
    ax.plot([max(l,0.3),h],[yy,yy],color=c,lw=1.4); ax.plot(o,yy,"s",color=c,ms=5)
    ax.text(2.25,yy,f"{o:.2f} ({l:.2f}–{h:.2f})",va="center",fontsize=7.5,color="black" if sig else DARK)
ax.axvline(1,color=DARK,lw=0.8)
ax.set_xscale("log"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xlim(0.3,2.2); ax.set_xticks([0.5,0.75,1,1.5,2]); ax.set_xticklabels(["0.5","0.75","1","1.5","2"])
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows],fontsize=7.5); ax.tick_params(axis="y",length=0)
ax.spines["left"].set_visible(False)
ax.set_xlabel("Odds ratio for distant stage, 2021–2023 vs. 2010–2019 trend (log scale)")
ax.text(2.25,ys[0]+0.9,"OR (95% CI)",fontsize=7.5,fontweight="bold")
ax.annotate("AIAN lower limit 0.18",xy=(0.3,ys[5]),xytext=(0.31,ys[5]+0.45),fontsize=6.5,color=DARK)
fig.tight_layout(); fig.savefig(OUT+"Figure1.png"); fig.savefig(OUT+"Figure1.tiff"); plt.close(fig)
# ---------- Figure 2: mediation cumulative
med=[("Black vs. White",["Biology","+ Stage","+ Area SES & marital","+ Treatment"],[19.5,41.6,47.9,47.7],[17.3,38.0,41.1,40.9],[22.6,44.9,53.7,52.3],"HR 2.05 → 1.46"),
     ("Hispanic vs. White",["Biology","+ Stage","+ Area SES & marital","+ Treatment"],[28.6,47.0,51.2,56.9],[23.5,37.8,43.5,48.1],[35.4,57.0,62.1,68.9],"HR 1.34 → 1.13")]
fig,axes=plt.subplots(1,2,figsize=(6.5,3.0),sharex=True)
shades=["#c6daf3","#8fb7ea","#5a95df","#2a78d6"]
for ax,(t,labs,pm,lo,hi,note) in zip(axes,med):
    yy=np.arange(4)[::-1]
    ax.barh(yy,pm,color=shades,height=0.65)
    ax.errorbar(pm,yy,xerr=[np.array(pm)-np.array(lo),np.array(hi)-np.array(pm)],fmt="none",ecolor="black",elinewidth=0.7,capsize=2)
    for v,h_,y_ in zip(pm,hi,yy): ax.text(h_+1.5,y_,f"{v:.0f}%",va="center",fontsize=7.5)
    ax.set_yticks(yy); ax.set_yticklabels(labs); ax.tick_params(axis="y",length=0)
    ax.set_title(f"{t}\n{note} after all blocks",fontsize=8,loc="left",fontweight="bold")
    ax.set_xlim(0,80); ax.set_xlabel("% of excess hazard mediated (cumulative)")
fig.tight_layout(w_pad=2.5); fig.savefig(OUT+"Figure2.png", bbox_inches="tight", pad_inches=0.05); fig.savefig(OUT+"Figure2.tiff"); plt.close(fig)
# ---------- Figure 3: MBC Black-White HR by period
per=["2010–2013","2014–2017","2018–2020"]
hrp=[(1.319,1.094,1.589),(1.786,1.466,2.177),(1.717,1.354,2.176)]
h2=[(1.651,1.284,2.123),(1.576,1.201,2.069),(1.069,0.684,1.671)]
fig,ax=plt.subplots(figsize=(4.2,3.2))
x=np.arange(3)
for vals,off,c,lab,mk in [(hrp,-0.08,BLUE,"HR+/HER2−","o"),(h2,0.08,GREY,"HER2+","s")]:
    v=np.array(vals); ax.errorbar(x+off,v[:,0],yerr=[v[:,0]-v[:,1],v[:,2]-v[:,0]],fmt=mk+"-",color=c,ms=5,capsize=2,lw=1.2,label=lab)
    for xi,vi in zip(x+off,v[:,0]): ax.text(xi+(-0.1 if off<0 else 0.1),vi,f"{vi:.2f}",ha="right" if off<0 else "left",va="center",fontsize=7)
ax.axhline(1,color=DARK,lw=0.8,ls="--")
ax.set_yscale("log"); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_yticks([0.75,1,1.5,2]); ax.set_yticklabels(["0.75","1","1.5","2"]); ax.set_ylim(0.6,2.5)
ax.set_xticks(x); ax.set_xticklabels(per); ax.set_xlim(-0.5,2.5)
ax.set_ylabel("Hazard ratio for death, Black vs. White (log scale)")
ax.legend(frameon=False,fontsize=7.5,loc="lower left")
fig.tight_layout(); fig.savefig(OUT+"Figure3.png"); fig.savefig(OUT+"Figure3.tiff"); plt.close(fig)
