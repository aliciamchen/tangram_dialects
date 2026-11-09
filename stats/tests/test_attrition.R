# Tests for the attrition helpers in stats/R/attrition.R.

if (!exists("player_attrition")) {
  source(here::here("stats", "config.R"))
}
if (!exists("check")) {
  source(here::here("stats", "tests", "check.R"))
}

# Three games as the pipeline codes them: g1 complete, g2 lost one idle player
# and one low-accuracy group member but ran on, g3 ended with one group left.
# g0 is a test game without a condition and must be ignored.
games <- tibble(
  gameId = c("g1", "g2", "g3", "g0"),
  condition = c("refer_separated", "refer_mixed", "social_mixed", NA),
  activeGroups = c(3, 2, 1, 3),
  ranToCompletion = c(TRUE, TRUE, FALSE, TRUE),
  completeNine = c(TRUE, FALSE, FALSE, TRUE)
)
mk_players <- function(game, reasons, no_fault) {
  n_removed <- length(reasons)
  tibble(
    gameId = game,
    playerId = sprintf("%s_p%d", game, 1:9),
    originalGroup = rep(c("A", "B", "C"), each = 3),
    removed = c(rep(TRUE, n_removed), rep(FALSE, 9 - n_removed)),
    removalReason = c(reasons, rep(NA_character_, 9 - n_removed)),
    removedNoFault = c(no_fault, rep(FALSE, 9 - n_removed))
  )
}
players <- bind_rows(
  mk_players("g1", character(0), logical(0)),
  mk_players("g2", c("idle", "low accuracy"), c(FALSE, FALSE)),
  mk_players(
    "g3",
    c("idle", "idle", "group disbanded", "insufficient groups"),
    c(FALSE, FALSE, TRUE, TRUE)
  ),
  mk_players("g0", character(0), logical(0))
)

pa <- player_attrition(players, games)
check(
  "players of test games are dropped",
  nrow(pa) == 27 && !"g0" %in% pa$gameId
)
check("removed players are read from the table", sum(pa$removed) == 6)
check(
  "the reason is the pipeline's category, 'completed' for a player who finished",
  identical(
    sort(pa$reason[pa$gameId == "g3"]),
    sort(c(
      "idle",
      "idle",
      "group disbanded",
      "insufficient groups",
      rep("completed", 5)
    ))
  )
)
check(
  "no-fault removals are read from the table",
  sum(pa$no_fault) == 2 &&
    all(pa$reason[pa$no_fault] %in% c("group disbanded", "insufficient groups"))
)
check(
  "the condition is a factor in CONDITION_ORDER",
  identical(levels(pa$condition), CONDITION_ORDER)
)

# Flags read as text, as a table written and read back may carry them
pa_text <- player_attrition(
  players |>
    mutate(
      removed = ifelse(removed, "True", "False"),
      removedNoFault = ifelse(removedNoFault, "True", "False")
    ),
  games
)
check("flags written as text are read as flags", sum(pa_text$removed) == 6)

status <- game_status(games)
check("game_status has one row per real game", nrow(status) == 3)
check(
  "a game with two groups left ran to completion but is not a complete nine-player game",
  status$ran_to_completion[status$gameId == "g2"] &&
    !status$complete_nine[status$gameId == "g2"]
)
check(
  "a game that ended early neither ran to completion nor is complete",
  !status$ran_to_completion[status$gameId == "g3"] &&
    !status$complete_nine[status$gameId == "g3"]
)
check(
  "complete_games returns the real games that kept all nine players",
  identical(complete_games(games), "g1")
)
check(
  "a table without the attrition columns is refused",
  inherits(
    try(player_attrition(players |> select(-removed), games), silent = TRUE),
    "try-error"
  ) &&
    inherits(
      try(game_status(games |> select(-completeNine)), silent = TRUE),
      "try-error"
    )
)

tests <- differential_dropout_tests(pa, status)
check(
  "both dropout tests run when there is variation",
  all(c("players", "games") %in% names(tests)) &&
    inherits(tests$players, "htest") &&
    inherits(tests$games, "htest")
)
none <- differential_dropout_tests(
  pa |> mutate(removed = FALSE),
  status |> mutate(complete_nine = TRUE)
)
check("no test is run when nobody dropped out", length(none) == 0)
