# Post-pandemic stage shift and racial/ethnic gaps in treatment response and survival among women aged 20–49 years with breast cancer (SEER 17, 2010–2023)

Analysis code for the manuscript submitted to *Cancer Epidemiology, Biomarkers & Prevention*.

**No data are included.** The study uses SEER Research Plus data (November 2025 submission), which are available from the National Cancer Institute under a data use agreement (https://seer.cancer.gov/data/). Do not commit SEER extracts to this repository; `.gitignore` excludes `data/` and `output/`.

## 1. Export the data from SEER*Stat

Database: *Incidence – SEER Research Plus Data, 17 Registries, Nov 2025 Sub (2000–2023)* (linked to county attributes).
Selection for all sessions: female; site recode ICD-O-3/WHO 2008 = Breast; malignant behavior; year of diagnosis 2010–2023; age 20–49 years; excluding autopsy-only and death-certificate-only cases.

| Export | Session | Layout | Extra selection | Format |
|---|---|---|---|---|
| Case listing | Case Listing | variables listed in `python/prepare_inputs.py` | sequence number = "One primary only" or "1st of 2 or more primaries" | labels, comma-delimited, variable names included |
| Race rates | Rate (age-adjusted, 2000 US Std) | page = race and origin recode; rows = year; columns = Breast Subtype (2010+) | none (all malignant tumors) | numeric, tab-delimited, no names |
| Age rates | Rate | page = age recode (5-year groups); rows = year; columns = Breast Subtype (2010+) | none | numeric, tab-delimited, no names |

## 2. Build the inputs

```
python python/prepare_inputs.py --caselisting Breast15.csv --race-rates Breast14.txt --age-rates Breast13.txt
```

## 3. Run

```
pip install -r requirements.txt
./run_all.sh
```

Outputs (tables as .xlsx/.csv, figures as .png/.tiff) are written to `output/`. The full run takes about 40 minutes, mostly bootstrap resampling in `analysis2.py`.

| Script | Content |
|---|---|
| `analysis.py` | Age-adjusted incidence, observed/expected ratios, stage-shift interrupted time series, cumulative incidence and cause-specific Cox models |
| `sens.py` | Incidence sensitivity analyses (unknown subtype redistributed; 2018 HER2 coding break) |
| `joinpoint.py` | Joinpoint regression (log-linear, SE-weighted, permutation test; Kim et al. 2000) |
| `lib2.py` | Shared helpers: data preparation, weighted Cox, Fine–Gray (IPCW), weighted logistic |
| `fg_run.py` | Builds analysis cohorts; overall Fine–Gray model |
| `analysis2.py` | Fine–Gray with bootstrap CIs; time to treatment; Commission on Cancer accreditation; stage shift by area; Asian/Pacific Islander subgroups; inverse odds ratio weighting mediation |
| `analysis3.py` | Neoadjuvant complete response; de novo metastatic survival by period |
| `analysis4.py` | Preoperative chemotherapy in high-risk triple-negative and HER2-positive disease; histology |
| `analysis5.py` | Time-to-treatment coding diagnostics; stage-shift post-period slope sensitivity |
| `analysis6.py` | Revision analyses: race × post-period interaction and Holm/BH adjustment; 2018 staging step; race by county type; metastatic single-model Cox, 3-year OS and 36-month RMST; neoadjuvant response by era and missingness |
| `analysis8.py` | Second review: race × post-period interaction within county type; leave-one-registry-out; metastatic Cox with follow-up truncated at 36 months, proportional-hazards tests and adjusted RMST; time to treatment with categorical year; complete response excluding response NOS |
| `analysis7.py` | 500-replicate bootstraps for Fine–Gray, IORW mediation and response mediation (`PART=black|hispanic|fg|crmed`) |
| `cebp_figs.py` | Manuscript Figures 1–3 |
| `R/` | R versions of the main models (survival, cmprsk, data.table, ggplot2); set `SEER_DATA_DIR` to the folder with the exports |

Random seeds are fixed in each script. Results were produced with Python 3.13, NumPy, pandas, SciPy and matplotlib.

## Citation

See `CITATION.cff`. Data citation: Surveillance, Epidemiology, and End Results (SEER) Program (www.seer.cancer.gov) SEER*Stat Database: Incidence – SEER Research Plus Data, 17 Registries, Nov 2025 Sub (2000–2023), National Cancer Institute, DCCPS, Surveillance Research Program, released April 2026, based on the November 2025 submission.

## License

MIT (code only).
