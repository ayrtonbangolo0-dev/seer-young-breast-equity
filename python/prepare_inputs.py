"""Build the analysis inputs in ./data from SEER*Stat exports.

Usage:
  python python/prepare_inputs.py --caselisting Breast15.csv --race-rates Breast14.txt --age-rates Breast13.txt

Exports expected (see README):
  case listing: labels, comma-delimited, variable names included
  rate files:   numeric, tab-delimited, no names (race x year x subtype; age x year x subtype)
"""
import argparse, os, shutil, pandas as pd

SLIM = ["Age recode with <1 year olds and 90+", "Sex", "Year of diagnosis", "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)",
        "Race recode (with detailed Asian and Native Hawaiian other PI)", "Site recode ICD-O-3/WHO 2008", "Behavior code ICD-O-3", "Histologic Type ICD-O-3",
        "Grade Recode (thru 2017)", "Derived Summary Grade 2018 (2018+)", "Laterality", "Diagnostic Confirmation",
        "Combined Summary Stage with Expanded Regional Codes (2004+)", "Derived EOD 2018 Stage Group Recode (2018+)", "Derived AJCC Stage Group, 7th ed (2010-2015)",
        "7th Edition Stage Group Recode (2016-2017)", "RX Summ--Surg Prim Site (1998-2022)", "RX Summ--Surg Prim Site 2023 (2023+)", "Radiation recode",
        "Chemotherapy recode (yes, no/unk)", "Tumor Size Over Time Recode (1988+)", "Breast Subtype (2010+)", "ER Status Recode Breast Cancer (1990+)",
        "PR Status Recode Breast Cancer (1990+)", "Derived HER2 Recode (2010+)", "COD to site recode", "SEER cause-specific death classification",
        "SEER other cause of death classification", "Survival months", "Survival months flag", "Vital status recode (study cutoff used)", "Sequence number",
        "Total number of in situ/malignant tumors for patient", "Age recode with single ages and 90+", "SEER registry (with CA and GA as whole states)",
        "Patient ID", "Type of Reporting Source", "Marital status at diagnosis", "Median household income inflation adj to 2024", "Rural-Urban Continuum Code", "PRCDA 2020"]
EXTRA = ["CoC Accredited Flag (2018+)", "Time from diagnosis to treatment in days recode", "RX Summ--Surg Prim Site (1998-2022)", "RX Summ--Surg Prim Site 2023 (2023+)",
         "Reason no cancer-directed surgery", "Radiation recode", "Chemotherapy recode (yes, no/unk)", "RX Summ--Systemic/Sur Seq (2007+)", "RX Summ--Surg/Rad Seq",
         "Regional nodes examined (1988+)", "Regional nodes positive (1988+)", "SEER registry (with CA and GA as whole states)", "PRCDA 2020",
         "SEER Combined Mets at DX-bone (2010+)", "SEER Combined Mets at DX-brain (2010+)", "SEER Combined Mets at DX-liver (2010+)", "SEER Combined Mets at DX-lung (2010+)",
         "Race recode (with detailed Asian and Native Hawaiian other PI)", "Response to Neoadjuvant Therapy Recode (2010+)", "Age recode with single ages and 90+",
         "Patient ID", "Year of diagnosis", "Site recode ICD-O-3/WHO 2008", "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)"]

ap = argparse.ArgumentParser()
ap.add_argument("--caselisting", required=True); ap.add_argument("--race-rates", required=True); ap.add_argument("--age-rates", required=True)
ap.add_argument("--out", default="data")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); os.makedirs("output", exist_ok=True)
df = pd.read_csv(a.caselisting, usecols=sorted(set(SLIM) | set(EXTRA)), dtype=str, keep_default_na=False)
df[SLIM].to_csv(os.path.join(a.out, "_slim_caselisting.csv"), index=False)
df[EXTRA].to_csv(os.path.join(a.out, "extra.csv"), index=False)
shutil.copy(a.race_rates, os.path.join(a.out, "Breast11.txt"))   # race x year x subtype
shutil.copy(a.age_rates, os.path.join(a.out, "Breast12.txt"))    # age x year x subtype
print(f"Wrote {len(df):,} cases and rate files to {a.out}/")
