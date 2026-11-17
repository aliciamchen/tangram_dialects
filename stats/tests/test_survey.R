# Tests for the exit-survey helper in stats/R/survey.R. The table itself (who
# is a respondent, the typed items) is tested where it is built:
# preprocessing/test_preprocessing.py, TestExitSurvey.

if (!exists("has_field")) {
  source(here::here("stats", "config.R"))
}
if (!exists("check")) {
  source(here::here("stats", "tests", "check.R"))
}

# The table as the pipeline writes it: one row per respondent, the Likert
# items numeric, the closed questions lower case
exit_survey <- tibble(
  gameId = c("g1", "g1", "g2"),
  playerId = c("p1", "p2", "p3"),
  condition = c("social_mixed", "social_mixed", "refer_separated"),
  originalGroup = c("A", "A", "B"),
  respondent = c("completed", "removed", "completed"),
  groupIdentification = c(6, 3, NA),
  groupLanguage = c("yes", "no", NA),
  feltHuman = c(NA_character_, NA_character_, NA_character_)
)

check(
  "has_field is TRUE only for items with answers",
  has_field(exit_survey, "groupIdentification") &&
    !has_field(exit_survey, "feltHuman") &&
    !has_field(exit_survey, "nonexistent")
)

check(
  "an empty table has no fields",
  !has_field(tibble(), "groupIdentification")
)
