#!/usr/bin/env bash
# Run from the repository root after python/prepare_inputs.py has populated ./data
set -euo pipefail
mkdir -p output
python python/analysis.py data output          # incidence, stage-shift ITS, survival, Table/Figure drafts
python python/sens.py                          # HER2 / unknown-subtype sensitivity for incidence
python python/joinpoint.py                     # joinpoint regression (permutation test)
python python/fg_run.py                        # builds output/cl.pkl and output/sv.pkl; overall Fine-Gray
BOOT=100 BOOTM=60 python python/analysis2.py   # Fine-Gray bootstrap, care-delivery subgroups, IORW mediation (~30 min)
BOOTM=100 python python/analysis3.py           # neoadjuvant response; de novo metastatic survival
python python/analysis4.py                     # neoadjuvant use in high-risk disease; histology
python python/analysis5.py                     # time-to-treatment coding diagnostics; stage-shift slope sensitivity
python python/analysis6.py                     # revision: race x period interaction, multiplicity, 2018 step, race x county type;
                                               #   metastatic single-model Cox, 3-y OS and RMST; response by era and missingness
for P in black hispanic fg crmed; do PART=$P BOOT=500 python python/analysis7.py; done   # 500-replicate bootstraps (~2 h)
BOOTR=300 python python/analysis8.py           # second review: interactions within county type, leave-one-registry-out; metastatic Cox truncated at 36 mo, PH tests, adjusted RMST; TTT with categorical year; CR excluding response NOS
python python/cebp_figs.py                     # manuscript Figures 1-3 (TIFF/PDF)
