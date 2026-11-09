# Attrition: who was removed and which games ran to completion, as the
# pipeline codes them, and whether dropout differs across conditions.
#
# The facts are columns preprocessing/preprocessing.py writes. On players.csv:
# `removed` (isActive is FALSE), `removalReason` (idle, low accuracy, group
# disbanded, insufficient groups, game terminated or other; empty for a
# player who finished) and `removedNoFault` (the group fell apart, too few
# groups remained, or the batch was stopped). On games.csv: `ranToCompletion`
# (at least two original groups viable at the end; these games enter the
# primary analyses) and `completeNine` (ran to completion with all nine
# players). The helpers here restrict them to the real games and set the
# condition factor. The two tests of differential dropout and the rerun on
# complete games (see robustness_rerun()) are additions to the
# preregistration, which commits to reporting attrition by condition and to
# rerunning the Phase 2 analyses on the games that kept all three groups.
#
# Tests: stats/tests/test_attrition.R

# One row per player of a real game: condition, removed, reason ("completed"
# for a player who finished) and whether the removal was through no fault of
# their own
player_attrition <- function(players, games) {
  require_columns(
    players,
    c("removed", "removalReason", "removedNoFault"),
    "players"
  )
  real <- games |> filter(!is.na(condition)) |> select(gameId, condition)
  players |>
    inner_join(real, by = "gameId") |>
    transmute(
      gameId,
      playerId,
      condition = factor(as.character(condition), levels = CONDITION_ORDER),
      originalGroup,
      removed = as_flag(removed),
      reason = ifelse(removed, as.character(removalReason), "completed"),
      no_fault = as_flag(removedNoFault)
    )
}

# One row per real game: whether it ran to completion and whether all nine
# players stayed
game_status <- function(games) {
  require_columns(games, c("ranToCompletion", "completeNine"), "games")
  games |>
    filter(!is.na(condition)) |>
    transmute(
      gameId,
      condition = factor(as.character(condition), levels = CONDITION_ORDER),
      ran_to_completion = as_flag(ranToCompletion),
      complete_nine = as_flag(completeNine)
    )
}

complete_games <- function(games) {
  game_status(games) |> filter(complete_nine) |> pull(gameId)
}

# Is dropout differential across conditions? Two tests: a chi-square on
# players removed by condition (players are clustered in games, so its p-value
# is anticonservative and is reported as descriptive), and Fisher's exact test
# on games that stayed complete by condition (the game is the unit of
# replication).
differential_dropout_tests <- function(pa, status) {
  out <- list()
  if (
    n_distinct(pa$condition[!is.na(pa$condition)]) >= 2 &&
      any(pa$removed) &&
      !all(pa$removed)
  ) {
    tab <- table(droplevels(pa$condition), pa$removed)
    out$players <- suppressWarnings(chisq.test(tab))
  }
  if (
    n_distinct(status$condition) >= 2 &&
      any(status$complete_nine) &&
      !all(status$complete_nine)
  ) {
    tab <- table(droplevels(status$condition), status$complete_nine)
    out$games <- fisher.test(tab)
  }
  out
}
