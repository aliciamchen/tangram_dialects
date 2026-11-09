# Exit-survey responses.
#
# Page 1 of the survey asks whether the instructions were understood, two
# 7-point items on identification with and closeness to the Phase 1 group,
# whether the group developed its own way of describing the pictures, and the
# strategy; page 2 asks whether the participant felt they were playing with
# humans, plus age, gender, education, pay fairness, and feedback. Players
# removed at the Phase 1 accuracy check answer both pages too. `has_field()`
# says whether an item has any answer in a survey table, so a consumer can
# skip an item nobody answered. The counts and means the article quotes are
# written by report/descriptives.py; this is the table the survey tests of
# checks/dropout_and_survey.R are run on.
#
# Tests: stats/tests/test_survey.R

# One row per respondent (any survey field answered) in a real game, with the
# exitSurvey_ prefix removed, condition, and whether the player completed the
# game or was removed (`removed`, as the pipeline coded it).
exit_survey_responses <- function(players, games) {
  survey_cols <- grep("^exitSurvey_", names(players), value = TRUE)
  if (length(survey_cols) == 0) {
    return(tibble())
  }
  require_columns(players, "removed", "players")
  real <- games |> filter(!is.na(condition)) |> select(gameId, condition)
  df <- players |>
    inner_join(real, by = "gameId") |>
    filter(if_any(
      all_of(survey_cols),
      ~ !is.na(.x) & as.character(.x) != ""
    )) |>
    rename_with(~ str_remove(.x, "^exitSurvey_"), all_of(survey_cols)) |>
    mutate(
      condition = factor(as.character(condition), levels = CONDITION_ORDER),
      respondent = ifelse(as_flag(removed), "removed", "completed")
    )
  for (item in c("groupIdentification", "groupCloseness", "age")) {
    if (item %in% names(df)) {
      df[[item]] <- suppressWarnings(as.numeric(df[[item]]))
    }
  }
  for (item in c(
    "gender",
    "education",
    "understood",
    "groupLanguage",
    "feltHuman"
  )) {
    if (item %in% names(df)) {
      df[[item]] <- str_to_lower(as.character(df[[item]]))
    }
  }
  df
}

has_field <- function(survey, field) {
  field %in% names(survey) && any(!is.na(survey[[field]]))
}
