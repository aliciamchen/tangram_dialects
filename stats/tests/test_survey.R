# Tests for the exit-survey helpers in stats/R/survey.R.

if (!exists("exit_survey_responses")) {
  source(here::here("stats", "config.R"))
}
if (!exists("check")) {
  source(here::here("stats", "tests", "check.R"))
}

games <- tibble(
  gameId = c("g1", "g2", "g0"),
  condition = c("social_mixed", "refer_separated", NA)
)

# One completer with both pages, one removed player with page 1 only, one
# player who never answered, one player in a test game
players_new <- tibble(
  gameId = c("g1", "g1", "g1", "g0"),
  playerId = c("p1", "p2", "p3", "p0"),
  originalName = c("Repi", "Minu", "Laju", "Hera"),
  originalGroup = "A",
  removed = c(FALSE, TRUE, FALSE, FALSE),
  exitSurvey_understood = c("yes", "yes", NA, "yes"),
  exitSurvey_groupIdentification = c("6", "3", NA, "7"),
  exitSurvey_groupCloseness = c("5", "2", NA, "7"),
  exitSurvey_groupLanguage = c("yes", "no", NA, "yes"),
  exitSurvey_strategy = c("short labels", "guessing", NA, "x"),
  exitSurvey_feltHuman = c("yes", NA, NA, "no"),
  exitSurvey_age = c("31", NA, NA, "40"),
  exitSurvey_gender = c("Female", NA, NA, "male"),
  exitSurvey_education = c("bachelor", NA, NA, "other"),
  exitSurvey_fair = c("yes", NA, NA, "yes"),
  exitSurvey_feedback = c("fun", NA, NA, "")
)
s <- exit_survey_responses(players_new, games)
check(
  "respondents are players of real games with any survey field answered, not only those with an age",
  nrow(s) == 2 && setequal(s$playerId, c("p1", "p2"))
)
check(
  "the exitSurvey_ prefix is removed and the Likert items are numeric",
  all(
    c("groupIdentification", "groupCloseness", "feltHuman", "strategy") %in%
      names(s)
  ) &&
    is.numeric(s$groupIdentification) &&
    s$groupIdentification[s$playerId == "p2"] == 3
)
check(
  "removed players are labeled from the pipeline's flag",
  s$respondent[s$playerId == "p2"] == "removed" &&
    s$respondent[s$playerId == "p1"] == "completed"
)
check(
  "categorical answers are lower-cased",
  s$gender[s$playerId == "p1"] == "female"
)
check(
  "has_field is TRUE only for items with answers",
  has_field(s, "feltHuman") && !has_field(s, "nonexistent")
)

# A survey table without the group or felt-human items
players_old <- tibble(
  gameId = "g2",
  playerId = c("q1", "q2"),
  originalName = c("Bavi", "Lika"),
  originalGroup = "B",
  removed = FALSE,
  exitSurvey_age = c(25, 30),
  exitSurvey_gender = c("male", "female"),
  exitSurvey_education = "bachelor",
  exitSurvey_understood = "yes",
  exitSurvey_fair = "yes",
  exitSurvey_strategy = c("describe shapes", "copy others"),
  exitSurvey_feedback = c("", "great")
)
o <- exit_survey_responses(players_old, games)
check(
  "items without a column are absent",
  !has_field(o, "groupIdentification") && !has_field(o, "feltHuman")
)
check(
  "no survey columns gives an empty table",
  nrow(exit_survey_responses(
    players_old |> select(-starts_with("exitSurvey_")),
    games
  )) ==
    0
)
