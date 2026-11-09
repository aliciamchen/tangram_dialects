# H3c: the description-property models, one fitter for the primary test
# (primary/h3c.R), its rerun on the complete games (primary/robustness.R) and
# the four-condition comparison (secondary/utterance_properties.R), so that
# each measure's formula is written once.
#
# Concreteness and lexical uniqueness are proportions of a description's
# content words, so they are modeled at the level of the content word, one
# binary observation each (word_level_measure() in R/prepare.R), with game,
# group, participant, tangram and description intercepts; a binomial on the
# counts with a per-description intercept is the same likelihood, and the
# effect size is the odds ratio. Word frequency is the mean Zipf frequency of
# the content words and is a linear model with game, group, participant and
# tangram intercepts. The game intercept stays although Phase 1 groups never
# interact: condition was assigned at the game level, so the game is what a
# condition contrast is tested against. Every model goes through
# fit_progressively(), so a singular fit ends at the random-intercepts-only
# structure with the steps recorded.
#
# Usage:
#   fit <- fit_h3c(props, "h3c_concreteness")   # props from final_phase_properties()
#   fit$model      the fit, or NULL when there was nothing to fit
#   fit$data       the rows it was fit to (one per content word, or per description)
#   fit$n_descriptions, fit$label
#
# Tests: stats/tests/test_properties.R

# The three measures: the count column behind each proportion (NULL for the
# linear measure), the outcome column of the model, and a label for printing
H3C_MEASURES <- list(
  h3c_concreteness = list(
    numerator = "n_concrete",
    outcome = "is_concrete",
    label = "Concreteness"
  ),
  h3c_uniqueness = list(
    numerator = "n_unique_words",
    outcome = "is_unique",
    label = "Lexical uniqueness"
  ),
  h3c_word_frequency = list(
    numerator = NULL,
    outcome = "mean_zipf_freq",
    label = "Word frequency"
  )
)

h3c_formula <- function(measure) {
  spec <- H3C_MEASURES[[match.arg(measure, names(H3C_MEASURES))]]
  terms <- c(
    "condition",
    "(1 | gameId)",
    "(1 | group)",
    "(1 | playerId)",
    "(1 | target)",
    if (!is.null(spec$numerator)) "(1 | descriptionId)"
  )
  stats::reformulate(terms, response = spec$outcome)
}

# One H3c model on `props`, a table of final descriptions with their
# properties (final_phase_properties() in R/prepare.R) and a `condition`
# factor. Returns the model as NULL when the table is empty, has fewer than
# two conditions, or has no description with the measure.
fit_h3c <- function(props, measure, verbose = TRUE) {
  measure <- match.arg(measure, names(H3C_MEASURES))
  spec <- H3C_MEASURES[[measure]]
  out <- list(
    measure = measure,
    label = spec$label,
    model = NULL,
    data = tibble(),
    n_descriptions = 0L
  )
  if (!has_rows(props)) {
    return(out)
  }
  if (is.null(spec$numerator)) {
    d <- props |> filter(!is.na(.data[[spec$outcome]]))
    family <- NULL
    n_descriptions <- nrow(d)
  } else {
    d <- word_level_measure(props, spec$numerator, spec$outcome)
    family <- binomial
    n_descriptions <- n_distinct(d$descriptionId)
  }
  out$data <- d
  out$n_descriptions <- as.integer(n_descriptions)
  if (nrow(d) == 0 || n_distinct(d$condition) < 2) {
    return(out)
  }
  out$model <- fit_progressively(
    h3c_formula(measure),
    data = d,
    family = family,
    verbose = verbose
  )
  out
}
