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
python python/cebp_figs.py                     # manuscript Figures 1-3
