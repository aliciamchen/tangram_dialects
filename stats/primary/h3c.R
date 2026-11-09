# H3c: are the final Phase 1 descriptions of groups that anticipate the
# social task less concrete, more lexically unique and made of rarer words?
# Three separate predictions, each its own two-tailed test, comparing
# social_first with social_mixed on each speaker's final Phase 1 description
# of each tangram (the model_inputs step keeps the last block per speaker and
# tangram).
#
# The three models are fit by fit_h3c() (R/properties.R, which states them):
# the two proportions at the level of the content word with a per-description
# intercept, the effect size an odds ratio, and word frequency as a linear
# model, each with game, group, participant and tangram intercepts. Where a
# fit is singular, the variance components at zero are named; the estimate is
# then that of the same model without those terms.
#
# Writes results/estimates/primary/h3c.csv: the condition
# coefficient of each of the three models (social_first minus social_mixed,
# with the odds ratio for the two logistic ones).

source(here::here("stats", "config.R"))
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
})
list2env(load_tables(), envir = globalenv())

if (!has_rows(final_props)) {
  stop(
    "final_phase_properties.csv is missing; run the model_inputs step first.",
    call. = FALSE
  )
}
p1_final_social <- final_phase1_properties(final_props, SOCIAL_CONDITIONS)
if (nrow(p1_final_social) == 0 || n_distinct(p1_final_social$condition) < 2) {
  stop(
    "Insufficient data for H3c (both social conditions with Phase 1 utterances).",
    call. = FALSE
  )
}

fits <- lapply(names(H3C_MEASURES), function(measure) {
  fit <- fit_h3c(p1_final_social, measure)
  if (is.null(fit$model)) {
    cat(sprintf("\n=== %s === nothing to fit\n", fit$label))
    return(fit)
  }
  cat(sprintf(
    "\n=== %s === (%d observations in %d descriptions)\n",
    fit$label,
    nrow(fit$data),
    fit$n_descriptions
  ))
  print(summary(fit$model))
  report_variance_components(fit$model)
  if (is_logistic(fit$model)) {
    report_odds_ratios(fit$model)
  } else {
    report_effect_sizes(fit$model)
  }
  fit
})

# The reference level is social_mixed, so the coefficient is social_first
# minus social_mixed.
rows <- bind_rows(lapply(fits, function(fit) {
  if (is.null(fit$model)) {
    return(tibble())
  }
  tidy_coefficients(
    fit$model,
    fit$measure,
    terms = "conditionsocial_first",
    n_descriptions = fit$n_descriptions,
    n_games = n_distinct(p1_final_social$gameId)
  )
}))
write_estimates(rows, "primary/h3c", source = "stats/primary/h3c.R")
