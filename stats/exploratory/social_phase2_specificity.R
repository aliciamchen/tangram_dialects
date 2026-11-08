# Exploratory, not preregistered: does Phase 2 group specificity differ
# between the two social conditions? The game-level regression
#
#   gs_phase2 ~ condition + gs_phase1
#
# has the same form as the H1/H2 model, fit by ordinary least squares on the
# two social conditions with equal weight for every game whose two endpoints
# are estimable, with one contrast, social_first minus social_mixed, tested
# two-tailed because no direction was predicted. It is the comparison the
# H1/H2 figure draws beside the planned contrasts (the adjusted panel of
# report/figures/confirmatory_h1_h2.py). In social_first the manipulation
# precedes Phase 1, so its Phase 1 score is an outcome rather than a baseline,
# and the adjustment compares the two conditions at equal Phase 1 specificity
# rather than estimating the whole effect of introducing the social task
# first. The per-game estimates come from primary/group_specificity.R.
#
# Writes results/estimates/exploratory/social_phase2_specificity.csv (the
# contrast, with the games in the model),
# exploratory/social_phase2_specificity_coefficients.csv, and
# exploratory/social_phase2_specificity_robustness.csv, the same contrast on
# the games that kept all nine players, the complete-games restriction of
# primary/robustness.R, which the SI's version of the adjusted panel draws
# (report/figures/intact_games.py).

source(here::here("stats", "config.R"))
suppressPackageStartupMessages(library(emmeans))

gs <- gs_wide(require_estimates(
  "primary/group_specificity",
  "stats/primary/group_specificity.R"
))

fit <- fit_game_level(
  gs,
  outcome = "gs_phase2",
  predictors = "gs_phase1",
  conditions = SOCIAL_CONDITIONS,
  contrasts = list(
    socialfirst_vs_socialmixed = c("social_first", "social_mixed")
  )
)
if (is.null(fit$contrasts) || nrow(fit$data) < 4) {
  stop(
    "The social Phase 2 comparison needs both social conditions with at least two games each",
    if (!is.null(fit$note)) paste0(": ", fit$note) else "",
    ".",
    call. = FALSE
  )
}
cat(sprintf("Games in the model: %d\n", nrow(fit$data)))
print(knitr::kable(fit$data |> count(condition, name = "games")))
print(summary(fit$model))
print(summary(fit$contrasts, infer = c(TRUE, TRUE)))

write_estimates(
  tidy_contrasts(
    fit$contrasts,
    "social_phase2_specificity",
    n_games = nrow(fit$data),
    measure = "sbert"
  ),
  "exploratory/social_phase2_specificity",
  source = "stats/exploratory/social_phase2_specificity.R"
)
write_estimates(
  tidy_coefficients(
    fit$model,
    "social_phase2_specificity",
    n_games = nrow(fit$data)
  ),
  "exploratory/social_phase2_specificity_coefficients",
  source = "stats/exploratory/social_phase2_specificity.R"
)

# ── Complete nine-player games ────────────────────────────────────────────
complete <- complete_games(read_table("players.csv"), read_table("games.csv"))
fit_complete <- fit_game_level(
  gs |> filter(gameId %in% complete),
  outcome = "gs_phase2",
  predictors = "gs_phase1",
  conditions = SOCIAL_CONDITIONS,
  contrasts = list(
    socialfirst_vs_socialmixed = c("social_first", "social_mixed")
  )
)
if (is.null(fit_complete$contrasts) || nrow(fit_complete$data) < 4) {
  stop(
    "The social Phase 2 comparison on complete games needs both social conditions with at least two games each",
    if (!is.null(fit_complete$note)) paste0(": ", fit_complete$note) else "",
    ".",
    call. = FALSE
  )
}
cat(sprintf(
  "\nComplete nine-player games in the model: %d of %d\n",
  nrow(fit_complete$data),
  length(complete)
))
print(knitr::kable(fit_complete$data |> count(condition, name = "games")))
print(summary(fit_complete$contrasts, infer = c(TRUE, TRUE)))

write_estimates(
  tidy_contrasts(
    fit_complete$contrasts,
    "social_phase2_specificity",
    restriction = "complete_games",
    n_games = nrow(fit_complete$data),
    measure = "sbert"
  ),
  "exploratory/social_phase2_specificity_robustness",
  source = "stats/exploratory/social_phase2_specificity.R"
)
