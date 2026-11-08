# =============================================================================
# Add-on to seer_breast_analysis.R: care-delivery subgroups + IORW mediation
# Run seer_breast_analysis.R first (it creates `dt`, `sv`, `ks`, OUT_DIR).
# Needs the extra columns below in the fread() `keep` list of the main script:
#   "CoC Accredited Flag (2018+)", "Time from diagnosis to treatment in days recode",
#   "RX Summ--Surg Prim Site (1998-2022)", "RX Summ--Surg Prim Site 2023 (2023+)",
#   "Radiation recode", "Chemotherapy recode (yes, no/unk)",
#   "SEER registry (with CA and GA as whole states)",
#   "Race recode (with detailed Asian and Native Hawaiian other PI)"
# (and "Insurance Recode (2007+)" if you add it to the export)
# =============================================================================
library(data.table); library(survival)
set.seed(20251108)

add_care_vars <- function(d) {
  s1 <- suppressWarnings(as.numeric(d$RX.Summ..Surg.Prim.Site..1998.2022.))
  s2 <- d$RX.Summ..Surg.Prim.Site.2023..2023..
  d[, surgery := fcase(s1 == 0, "None", s1 >= 19 & s1 <= 24, "BCS", s1 >= 30 & s1 <= 80, "Mastectomy",
                       startsWith(s2, "A"), "BCS", startsWith(s2, "B"), "Mastectomy", s2 %in% c("00", "000"), "None",
                       default = "Unknown")]
  d[, radiation := ifelse(grepl("Beam|implant|isotope|Combination|Radiation, NOS", Radiation.recode), "Yes", "No/Unknown")]
  d[, chemo := ifelse(Chemotherapy.recode..yes..no.unk. == "Yes", "Yes", "No/Unknown")]
  d[, ttt_days := suppressWarnings(as.numeric(Time.from.diagnosis.to.treatment.in.days.recode))]
  d[, ttt_cat := fcase(is.na(ttt_days), "Unknown", ttt_days <= 30, "0-30", ttt_days <= 60, "31-60", ttt_days <= 90, "61-90", default = ">90")]
  d[, coc := fcase(startsWith(CoC.Accredited.Flag..2018.., "ANALYTIC"), "CoC", startsWith(CoC.Accredited.Flag..2018.., "Abstract from facility WITHOUT"), "Non-CoC",
                   default = NA_character_)]
  d[, registry := SEER.registry..with.CA.and.GA.as.whole.states.]
  for (v in c("surgery", "radiation", "chemo", "ttt_cat", "registry")) set(d, j = v, value = factor(d[[v]]))
  d[, surgery := relevel(surgery, "BCS")][, radiation := relevel(radiation, "Yes")][, chemo := relevel(chemo, "Yes")][, ttt_cat := relevel(ttt_cat, "0-30")]
  d
}
dt <- add_care_vars(dt); sv <- add_care_vars(sv)

# ---- (a) time to treatment
tt <- dt[!is.na(ttt_days) & race %in% c("NH White", "NH Black", "Hispanic", "NH API", "NH AIAN")]
fwrite(tt[, .(n = .N, median_days = median(ttt_days), pct_gt60 = 100 * mean(ttt_days > 60), pct_gt90 = 100 * mean(ttt_days > 90)), by = .(race, period)],
       file.path(OUT_DIR, "ttt_by_race_period.csv"))
m_ttt <- glm(I(ttt_days > 60) ~ race + agegrp + stage + subtype + income + metro + married + registry + I(year - 2010), family = binomial, data = droplevels(tt))
fwrite(data.table(term = names(coef(m_ttt)), OR = exp(coef(m_ttt)), exp(confint.default(m_ttt)))[grepl("^race", term)],
       file.path(OUT_DIR, "ttt_gt60_adjOR.csv"))

# ---- (b) CoC accreditation, dx 2018+
cc <- dt[year >= 2018 & !is.na(coc)]
fwrite(cc[, .(n = .N, pct_nonCoC = 100 * mean(coc == "Non-CoC")), by = race], file.path(OUT_DIR, "coc_by_race.csv"))

# ---- (c) IORW mediation (Tchetgen Tchetgen 2013), Black or Hispanic vs NH White
blocks <- list(biology = c("subtype", "grade"), stage = "stage",
               ses = c("income", "metro", "married"), treatment = c("surgery", "radiation", "chemo", "ttt_cat"))
iorw_once <- function(d, meds, expo) {
  d <- droplevels(d); d$A <- as.integer(d$race == expo)
  fA <- glm(reformulate(c(meds, "agegrp", "registry", "I(year - 2010)"), "A"), family = binomial, data = d)
  mm <- model.matrix(reformulate(meds), d)[, -1, drop = FALSE]
  g <- coef(fA)[colnames(mm)]; g[is.na(g)] <- 0
  d$w <- ifelse(d$A == 1, exp(-(mm %*% g)), 1)
  tot <- coef(coxph(Surv(time, status == 1) ~ A + agegrp + registry + I(year - 2010), data = d))["A"]
  dir <- coef(coxph(Surv(time, status == 1) ~ A + agegrp + registry + I(year - 2010), data = d, weights = w))["A"]
  c(total = unname(tot), direct = unname(dir))
}
med_run <- function(expo, meds, B = 200) {
  d <- sv[race %in% c(expo, "NH White")]
  est <- iorw_once(d, meds, expo)
  bs <- t(replicate(B, iorw_once(d[sample(.N, replace = TRUE)], meds, expo)))
  pm <- function(x) 100 * (x[1] - x[2]) / x[1]
  data.table(exposure = expo, mediators = paste(meds, collapse = "+"),
             HR_total = exp(est[1]), HR_direct = exp(est[2]), pct_mediated = pm(est),
             pm_lo = quantile(apply(bs, 1, pm), .025), pm_hi = quantile(apply(bs, 1, pm), .975))
}
sets <- list(blocks$biology, c(blocks$biology, blocks$stage), c(blocks$biology, blocks$stage, blocks$ses), unlist(blocks))
med <- rbindlist(lapply(c("NH Black", "Hispanic"), function(e) rbindlist(lapply(sets, function(m) med_run(e, m)))))
fwrite(med, file.path(OUT_DIR, "mediation_iorw_bcdeath.csv"))
print(med)
