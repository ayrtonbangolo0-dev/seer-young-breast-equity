# =============================================================================
# Early-onset breast cancer, SEER-17 Research Plus (Nov 2025 sub, dx 2010-2023)
# Incidence trends, post-pandemic stage shift, competing-risk survival
#
# Inputs (same folder as this script, or edit DATA_DIR):
#   Breast15.csv  - case listing, exported as LABELS, comma-delimited, names included
#   Breast14.txt  - rate session: Race/ethnicity (page) x Year (row) x Subtype (col)
#   Breast13.txt  - rate session: Age group (page) x Year (row) x Subtype (col)
# Rate files: numeric format, tab-delimited, no names (SEER*Stat default) -- as you exported them.
#
# Packages: data.table, survival, cmprsk, ggplot2  (install.packages(c(...)) once)
# Joinpoint: import Breast13/Breast14 .dic/.txt directly into the Joinpoint program
# for the published APC/AAPC; the log-linear APC below is a cross-check.
# =============================================================================

suppressPackageStartupMessages({
  library(data.table); library(survival); library(cmprsk); library(ggplot2)
})
DATA_DIR <- Sys.getenv("SEER_DATA_DIR", "data")   # folder with Breast13.txt, Breast14.txt, Breast15.csv
OUT_DIR  <- file.path(DATA_DIR, "R_output"); dir.create(OUT_DIR, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1. CASE LISTING: read only the columns we need
# -----------------------------------------------------------------------------
keep <- c("Age recode with <1 year olds and 90+", "Age recode with single ages and 90+",
          "Year of diagnosis", "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)",
          "Race recode (with detailed Asian and Native Hawaiian other PI)",
          "Site recode ICD-O-3/WHO 2008", "Type of Reporting Source", "Sequence number",
          "Breast Subtype (2010+)", "Combined Summary Stage with Expanded Regional Codes (2004+)",
          "Grade Recode (thru 2017)", "Derived Summary Grade 2018 (2018+)",
          "Median household income inflation adj to 2024", "Rural-Urban Continuum Code",
          "Marital status at diagnosis", "Survival months", "Survival months flag",
          "SEER cause-specific death classification", "SEER other cause of death classification",
          "Patient ID",
          # care-delivery variables used by seer_breast_mediation.R
          "CoC Accredited Flag (2018+)", "Time from diagnosis to treatment in days recode",
          "RX Summ--Surg Prim Site (1998-2022)", "RX Summ--Surg Prim Site 2023 (2023+)",
          "Radiation recode", "Chemotherapy recode (yes, no/unk)",
          "SEER registry (with CA and GA as whole states)")
dt <- fread(file.path(DATA_DIR, "Breast15.csv"), select = keep, colClasses = "character")
setnames(dt, make.names(names(dt)))
flow <- data.table(step = "Exported", n = nrow(dt))

dt <- dt[Site.recode.ICD.O.3.WHO.2008 == "Breast"]                         # drops breast lymphoma etc.
flow <- rbind(flow, list("Site recode = Breast", nrow(dt)))
dt <- dt[!Type.of.Reporting.Source %in% c("Autopsy only", "Death certificate only")]
flow <- rbind(flow, list("Excluding autopsy/DCO", nrow(dt)))
# If you re-exported with 'First malignant primary indicator', also keep one row per patient:
dt <- dt[!duplicated(Patient.ID)]
flow <- rbind(flow, list("One tumor per patient", nrow(dt)))

# ---- recodes
dt[, year := as.integer(Year.of.diagnosis)]
dt[, race := fcase(
  Race.and.origin.recode..NHW..NHB..NHAIAN..NHAPI..Hispanic. == "Non-Hispanic White", "NH White",
  Race.and.origin.recode..NHW..NHB..NHAIAN..NHAPI..Hispanic. == "Non-Hispanic Black", "NH Black",
  Race.and.origin.recode..NHW..NHB..NHAIAN..NHAPI..Hispanic. == "Non-Hispanic American Indian/Alaska Native", "NH AIAN",
  Race.and.origin.recode..NHW..NHB..NHAIAN..NHAPI..Hispanic. == "Non-Hispanic Asian or Pacific Islander", "NH API",
  Race.and.origin.recode..NHW..NHB..NHAIAN..NHAPI..Hispanic. == "Hispanic (All Races)", "Hispanic",
  default = "Unknown")]
dt[, race := factor(race, c("NH White", "NH Black", "NH AIAN", "NH API", "Hispanic", "Unknown"))]
dt[, agegrp := factor(sub(" years", "", Age.recode.with..1.year.olds.and.90.))]
dt[, agegrp := relevel(agegrp, ref = "45-49")]
dt[, age := as.integer(sub("\\D.*$", "", Age.recode.with.single.ages.and.90.))]
dt[, ageband := ifelse(age < 40, "20-39", "40-49")]
dt[, subtype := fcase(Breast.Subtype..2010.. == "HR+/HER2-", "HR+/HER2-",
                      Breast.Subtype..2010.. == "HR+/HER2+", "HR+/HER2+",
                      Breast.Subtype..2010.. == "HR-/HER2+", "HR-/HER2+",
                      Breast.Subtype..2010.. == "HR-/HER2-", "Triple-negative", default = "Unknown")]
dt[, subtype := factor(subtype, c("HR+/HER2-", "HR+/HER2+", "HR-/HER2+", "Triple-negative", "Unknown"))]
stg <- dt$Combined.Summary.Stage.with.Expanded.Regional.Codes..2004..
dt[, stage := factor(fcase(stg == "Localized only", "Localized", startsWith(stg, "Regional"), "Regional",
                           startsWith(stg, "Distant"), "Distant", default = "Unknown"),
                     c("Localized", "Regional", "Distant", "Unknown"))]
g17 <- dt$Grade.Recode..thru.2017.; g18 <- dt$Derived.Summary.Grade.2018..2018..
dt[, grade := fcase(
  grepl("Grade I$", g17) | grepl("category \\(1\\)|^Well|^Low", g18), "G1",
  grepl("Grade II$", g17) | grepl("category \\(2\\)|^Moderately", g18), "G2",
  grepl("Grade III|Grade IV", g17) | grepl("category \\(3\\)|^Poorly|^High|^Undiff", g18), "G3",
  default = "Unknown")]
dt[, grade := factor(grade, c("G1", "G2", "G3", "Unknown"))]
inc <- dt$Median.household.income.inflation.adj.to.2024
inc_k <- suppressWarnings(as.numeric(gsub("[$,]", "", sub(" - .*", "", inc)))) / 1000
inc_k[startsWith(inc, "<")] <- 35; inc_k[startsWith(inc, "$120")] <- 125
dt[, income := factor(fcase(is.na(inc_k), "Unknown", inc_k < 75, "<$75k", inc_k < 100, "$75-99k", default = ">=$100k"),
                      c(">=$100k", "$75-99k", "<$75k", "Unknown"))]
ru <- dt$Rural.Urban.Continuum.Code
dt[, metro := factor(fcase(grepl("ge 1 million", ru), "Large metro", startsWith(ru, "Counties in metropolitan"), "Other metro",
                           startsWith(ru, "Nonmetro"), "Nonmetro", default = "Unknown"),
                     c("Large metro", "Other metro", "Nonmetro", "Unknown"))]
dt[, married := factor(fcase(startsWith(Marital.status.at.diagnosis, "Married"), "Married",
                             Marital.status.at.diagnosis == "Unknown", "Unknown", default = "Not married"),
                       c("Married", "Not married", "Unknown"))]
dt[, period := fcase(year <= 2019, "2010-2019", year == 2020, "2020", default = "2021-2023")]

# -----------------------------------------------------------------------------
# 2. TABLE 1 (by period)
# -----------------------------------------------------------------------------
tab1 <- rbindlist(lapply(c("ageband", "race", "subtype", "stage", "grade", "income", "metro", "married"), function(v) {
  x <- dt[, .N, by = c(v, "period")][, pct := 100 * N / sum(N), by = period]
  x <- dcast(x, as.formula(paste(v, "~ period")), value.var = c("N", "pct"))
  setnames(x, v, "level"); x[, variable := v]; x
}), fill = TRUE)
fwrite(tab1, file.path(OUT_DIR, "table1_by_period.csv"))
fwrite(flow, file.path(OUT_DIR, "flow.csv"))

# -----------------------------------------------------------------------------
# 3. INCIDENCE from rate files (age-adjusted, 2000 US std)
# -----------------------------------------------------------------------------
read_rate <- function(f, page) {
  x <- fread(file.path(DATA_DIR, f), header = FALSE, colClasses = "character",
             col.names = c(page, "yr", "sub", "rate", "se", "lci", "uci", "count", "pop"))
  num <- function(v) suppressWarnings(as.numeric(gsub(",", "", trimws(v))))
  x[, c("rate", "se", "count", "pop") := lapply(.SD, num), .SDcols = c("rate", "se", "count", "pop")]
  x[, (page) := as.integer(get(page))][, sub := as.integer(sub)][, year := as.integer(yr) + 1999L]
  x[year >= 2010 & year <= 2023]
}
sublab <- c("HR+/HER2+", "HR-/HER2+", "HR+/HER2-", "Triple-negative", "Unknown", "Not available")
r12 <- read_rate("Breast13.txt", "age")[age %in% 5:10]
std <- data.table(age = 5:10, w = c(66537, 64571, 71044, 80762, 81851, 72118))   # 2000 US std, 20-49
adj_rate <- function(d, ages) {
  w <- std[age %in% ages][, w := w / sum(w)]
  d <- merge(d[age %in% ages], w, by = "age")
  d[, .(count = sum(count), rate = 1e5 * sum(w * count / pop),
        se = 1e5 * sqrt(sum(w^2 * count / pop^2))), by = year]
}
bands <- list(`20-49` = 5:10, `20-39` = 5:8, `40-49` = 9:10)
inc_series <- rbindlist(lapply(names(bands), function(b) {
  all <- r12[, .(count = sum(count), pop = pop[1]), by = .(age, year)]
  out <- list(adj_rate(all, bands[[b]])[, `:=`(band = b, subtype = "All")])
  for (k in 0:3) out[[k + 2]] <- adj_rate(r12[sub == k], bands[[b]])[, `:=`(band = b, subtype = sublab[k + 1])]
  rbindlist(out)
}))
fwrite(inc_series, file.path(OUT_DIR, "incidence_by_year.csv"))

apc_fit <- function(d) {                     # weighted log-linear, 2010-2019; O/E 2021-23
  pre <- d[year <= 2019 & rate > 0]
  m <- lm(log(rate) ~ I(year - 2010), data = pre, weights = (rate / se)^2)
  b <- coef(m)[2]; ci <- confint(m)[2, ]
  post <- d[year >= 2021]
  p <- predict(m, newdata = post, se.fit = TRUE)
  diff <- mean(log(post$rate) - p$fit)
  vd <- mean(p$se.fit)^2 + sum((post$se / post$rate)^2) / nrow(post)^2   # approximate
  data.table(APC = 100 * (exp(b) - 1), APC_lo = 100 * (exp(ci[1]) - 1), APC_hi = 100 * (exp(ci[2]) - 1),
             OE_2021_23 = exp(diff), OE_lo = exp(diff - 1.96 * sqrt(vd)), OE_hi = exp(diff + 1.96 * sqrt(vd)))
}
apc <- inc_series[, apc_fit(.SD), by = .(band, subtype)]
fwrite(apc, file.path(OUT_DIR, "incidence_APC_and_OE.csv"))

ggplot(inc_series[band != "20-49" & subtype != "All"], aes(year, rate, colour = subtype)) +
  annotate("rect", xmin = 2019.5, xmax = 2020.5, ymin = -Inf, ymax = Inf, fill = "grey90") +
  geom_line(linewidth = .8) + geom_point(size = 1.5) + scale_y_log10() +
  facet_wrap(~ band, scales = "free_y") + labs(x = "Year of diagnosis", y = "Age-adjusted rate per 100,000", colour = NULL) +
  theme_minimal(base_size = 10) + theme(legend.position = "bottom")
ggsave(file.path(OUT_DIR, "fig1_incidence.png"), width = 9, height = 4, dpi = 300)

# -----------------------------------------------------------------------------
# 4. STAGE SHIFT: interrupted time series (2020 excluded)
# -----------------------------------------------------------------------------
ks <- dt[stage != "Unknown"][, distant := as.integer(stage == "Distant")]
its_fit <- function(d, adjust = TRUE) {
  d <- d[year != 2020][, `:=`(t = year - 2010, post = as.integer(year >= 2021))]
  f <- if (adjust) distant ~ t + post + agegrp + race + subtype else distant ~ t + post
  m <- glm(f, family = quasibinomial, data = d)
  s <- summary(m)$coefficients["post", ]
  data.table(OR_post = exp(s[1]), lo = exp(s[1] - 1.96 * s[2]), hi = exp(s[1] + 1.96 * s[2]), p = s[4], n = nrow(d))
}
its <- rbind(its_fit(ks)[, group := "All"],
             ks[, its_fit(.SD), by = race][, group := paste("Race:", race)][, race := NULL],
             ks[, its_fit(.SD), by = subtype][, group := paste("Subtype:", subtype)][, subtype := NULL],
             ks[, its_fit(.SD), by = ageband][, group := paste("Age:", ageband)][, ageband := NULL], fill = TRUE)
fwrite(its, file.path(OUT_DIR, "stage_shift_ITS.csv"))
# Sensitivity: add a post-period slope term (t_post = pmax(year - 2020, 0)) and/or use 2018+ cases only.

# -----------------------------------------------------------------------------
# 5. SURVIVAL: dx 2010-2018, competing risks
# -----------------------------------------------------------------------------
sv <- dt[year <= 2018 &
         Survival.months.flag %in% c("Complete dates are available and there are more than 0 days of survival",
                                     "Incomplete dates are available and there cannot be zero days of follow-up") &
         SEER.cause.specific.death.classification != "Dead (missing/unknown COD)"]
sv[, time := as.numeric(Survival.months)]
sv[, status := fcase(SEER.cause.specific.death.classification == "Dead (attributable to this cancer dx)", 1L,
                     SEER.other.cause.of.death.classification == "Dead (attributable to causes other than this cancer dx)", 2L,
                     default = 0L)]
flow <- rbind(flow, list("Survival cohort (dx 2010-2018, valid time, known COD)", nrow(sv)))
fwrite(flow, file.path(OUT_DIR, "flow.csv"))

# 5a. Cumulative incidence (Aalen-Johansen) + Gray's test, by race within subtype
cif_tab <- rbindlist(lapply(levels(droplevels(sv$subtype)), function(s) {
  d <- sv[subtype == s & race != "Unknown"]
  ci <- cuminc(d$time, d$status, group = droplevels(d$race), cencode = 0)
  tp <- timepoints(ci, 60)
  est <- tp$est[grep(" 1$", rownames(tp$est)), 1]; v <- tp$var[grep(" 1$", rownames(tp$var)), 1]
  data.table(subtype = s, race = sub(" 1$", "", names(est)), CIF5_bc = 100 * est,
             lo = 100 * pmax(est - 1.96 * sqrt(v), 0), hi = 100 * (est + 1.96 * sqrt(v)),
             gray_p = ci$Tests[1, "pv"])
}))
fwrite(cif_tab, file.path(OUT_DIR, "CIF_5yr_by_race_subtype.csv"))

# 5b. Cause-specific Cox (breast cancer death)
form <- ~ race + subtype + stage + grade + agegrp + income + metro + married + I(year - 2010)
cs <- coxph(update(form, Surv(time, status == 1) ~ .), data = sv)
cs_tab <- data.table(term = names(coef(cs)), HR = exp(coef(cs)), exp(confint(cs)), p = summary(cs)$coefficients[, 5])
fwrite(cs_tab, file.path(OUT_DIR, "cox_cause_specific.csv"))

# 5c. Fine-Gray subdistribution hazards (breast cancer death; other death = competing)
X <- model.matrix(form, data = sv)[, -1]
fg <- crr(sv$time, sv$status, cov1 = X, failcode = 1, cencode = 0)
fg_s <- summary(fg)$coef
fg_tab <- data.table(term = rownames(fg_s), sHR = fg_s[, "exp(coef)"],
                     lo = exp(fg_s[, "coef"] - 1.96 * fg_s[, "se(coef)"]),
                     hi = exp(fg_s[, "coef"] + 1.96 * fg_s[, "se(coef)"]), p = fg_s[, "p-value"])
fwrite(fg_tab, file.path(OUT_DIR, "fine_gray.csv"))
# crr() on ~85k rows takes a few minutes. If too slow: survival::finegray() + coxph(), which is equivalent:
#   fgd <- finegray(Surv(time, factor(status, 0:2, c("cens","bc","other"))) ~ ., data = sv, etype = "bc")
#   coxph(Surv(fgstart, fgstop, fgstatus) ~ race + subtype + ..., weights = fgwt, data = fgd)

# 5d. Race x subtype interaction (does the Black-White gap differ by subtype?)
cs_int <- coxph(Surv(time, status == 1) ~ race * subtype + stage + grade + agegrp + income + metro + married + I(year - 2010),
                data = sv[race != "Unknown" & subtype != "Unknown"])
print(anova(cs_int))

cat("\nDone. Results in:", OUT_DIR, "\n")
