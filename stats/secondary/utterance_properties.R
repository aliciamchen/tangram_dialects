# Secondary analysis 7 (exploratory): the H3c description properties
# (concreteness, lexical uniqueness, word frequency) across all four
# conditions, for each speaker's final description of each tangram in each
# phase, to ask whether group-specific language maintained passively
# (refer_separated) differs from language maintained under a social goal
# (social_mixed). The models are those of H3c, fit by fit_h3c()
# (R/properties.R) on all four conditions. The named contrast is
# refer_separated minus social_mixed; all pairwise comparisons follow,
# Tukey-adjusted.
#
# Writes results/estimates/secondary/utterance_properties.csv (the
# named contrast and the pairwise comparisons, per phase and measure) and
# secondary/utterance_properties_by_condition.csv (the marginal means).

source(here::here("stats", "config.R"))
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})
list2env(load_tables(), envir = globalenv())

# The measure ids the estimates tables carry, by the H3C_MEASURES name
MEASURE_IDS <- c(
  h3c_concreteness = "concreteness",
  h3c_uniqueness = "uniqueness",
  h3c_word_frequency = "mean_zipf_freq"
)

if (!has_rows(final_props)) {
  stop(
    "final_phase_properties.csv is missing; run the model_inputs step first.",
    call. = FALSE
  )
}

rows <- tibble()
means <- tibble()
for (ph in c(1, 2)) {
  props <- final_phase_properties(final_props, phase = ph)
  if (nrow(props) == 0 || n_distinct(props$condition) < 2) {
    cat(sprintf("Phase %d: not enough conditions with data.\n", ph))
    next
  }
  for (h3c_measure in names(H3C_MEASURES)) {
    measure <- MEASURE_IDS[[h3c_measure]]
    fit <- fit_h3c(props, h3c_measure)
    if (is.null(fit$model)) {
      next
    }
    m <- fit$model
    linear <- !is_logistic(m)
    cat(sprintf("\n=== Phase %d: %s ===\n", ph, fit$label))
    em <- if (linear) {
      emmeans(m, "condition")
    } else {
      emmeans(m, "condition", type = "response")
    }
    em_table <- as.data.frame(summary(em))
    print(em_table)
    means <- bind_rows(
      means,
      tibble(
        phase = ph,
        measure = measure,
        condition = as.character(em_table$condition),
        estimate = if ("prob" %in% names(em_table)) {
          em_table$prob
        } else {
          em_table$emmean
        },
        std_error = em_table$SE,
        lower = if ("asymp.LCL" %in% names(em_table)) {
          em_table$asymp.LCL
        } else {
          em_table$lower.CL
        },
        upper = if ("asymp.UCL" %in% names(em_table)) {
          em_table$asymp.UCL
        } else {
          em_table$upper.CL
        },
        scale = if (linear) "mean" else "probability"
      )
    )
    lv <- levels(droplevels(fit$data$condition))
    if (all(c("refer_separated", "social_mixed") %in% lv)) {
      cat(
        "Passive vs active maintenance (refer_separated minus social_mixed):\n"
      )
      named <- contrast(
        em,
        list(
          passive_minus_active = pairwise_weights(
            lv,
            "refer_separated",
            "social_mixed"
          )
        )
      )
      print(summary(named, infer = c(TRUE, TRUE)))
      rows <- bind_rows(
        rows,
        tidy_contrasts(
          named,
          "passive_vs_active",
          phase = ph,
          measure = measure,
          adjustment = "none",
          n_obs = nobs(m),
          random_effects = random_effects_structure(m)
        )
      )
    }
    cat("All pairwise comparisons (Tukey):\n")
    pw <- pairs(em)
    print(summary(pw))
    rows <- bind_rows(
      rows,
      tidy_contrasts(
        pw,
        "pairwise",
        phase = ph,
        measure = measure,
        adjustment = "tukey",
        n_obs = nobs(m),
        random_effects = random_effects_structure(m)
      )
    )
  }
}
write_estimates(
  rows,
  "secondary/utterance_properties",
  source = "stats/secondary/utterance_properties.R"
)
write_estimates(
  means,
  "secondary/utterance_properties_by_condition",
  source = "stats/secondary/utterance_properties.R"
)
