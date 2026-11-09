# Tests for the H3c fitter in stats/R/properties.R.

if (!exists("fit_h3c")) {
  source(here::here("stats", "config.R"))
}
if (!exists("check")) {
  source(here::here("stats", "tests", "check.R"))
}

check(
  "the word-level measures carry a description intercept and the linear one does not",
  identical(
    deparse1(h3c_formula("h3c_concreteness")),
    "is_concrete ~ condition + (1 | gameId) + (1 | group) + (1 | playerId) + (1 | target) + (1 | descriptionId)"
  ) &&
    identical(
      deparse1(h3c_formula("h3c_uniqueness")),
      "is_unique ~ condition + (1 | gameId) + (1 | group) + (1 | playerId) + (1 | target) + (1 | descriptionId)"
    ) &&
    identical(
      deparse1(h3c_formula("h3c_word_frequency")),
      "mean_zipf_freq ~ condition + (1 | gameId) + (1 | group) + (1 | playerId) + (1 | target)"
    )
)
check(
  "an unknown measure is refused",
  inherits(try(h3c_formula("h3c_length"), silent = TRUE), "try-error")
)
check(
  "an empty table gives no model",
  is.null(fit_h3c(tibble(), "h3c_concreteness")$model)
)

# Six games of three speakers and four tangrams, three per social condition,
# with more concrete words in social_mixed
set.seed(3)
props <- expand_grid(
  gameId = sprintf("g%d", 1:6),
  player = 1:3,
  target = sprintf("t%d", 1:4)
) |>
  mutate(
    condition = factor(
      ifelse(gameId %in% c("g1", "g2", "g3"), "social_mixed", "social_first"),
      levels = SOCIAL_CONDITIONS
    ),
    playerId = paste(gameId, player),
    group = paste(gameId, "A"),
    phaseNum = 1,
    blockNum = 5,
    n_content_words = 8L,
    n_concrete = rbinom(n(), 8, ifelse(condition == "social_first", 0.3, 0.6)),
    n_unique_words = rbinom(n(), 8, 0.4),
    mean_zipf_freq = rnorm(n(), 4.5 - 0.3 * (condition == "social_first"), 0.2)
  )

fit <- fit_h3c(props, "h3c_concreteness", verbose = FALSE)
check(
  "the word-level model is fit to one row per content word and counts the descriptions",
  inherits(fit$model, "glmerMod") &&
    fit$n_descriptions == nrow(props) &&
    nrow(fit$data) == sum(props$n_content_words)
)
check(
  "the coefficient is social_first minus social_mixed",
  "conditionsocial_first" %in%
    names(fixef(fit$model)) &&
    fixef(fit$model)[["conditionsocial_first"]] < 0
)
freq <- fit_h3c(props, "h3c_word_frequency", verbose = FALSE)
check(
  "the linear model is fit to one row per description",
  inherits(freq$model, "lmerMod") &&
    freq$n_descriptions == nrow(props) &&
    nrow(freq$data) == nrow(props)
)
check("a table with one condition gives no model but keeps its rows", {
  one <- fit_h3c(
    props |> filter(condition == "social_mixed"),
    "h3c_uniqueness",
    verbose = FALSE
  )
  is.null(one$model) && nrow(one$data) > 0
})
