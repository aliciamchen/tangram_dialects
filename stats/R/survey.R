# Exit-survey responses.
#
# Page 1 of the survey asks whether the instructions were understood, two
# 7-point items on identification with and closeness to the Phase 1 group,
# whether the group developed its own way of describing the pictures, and the
# strategy; page 2 asks whether the participant felt they were playing with
# humans, plus age, gender, education, pay fairness, and feedback. Players
# removed at the Phase 1 accuracy check answer both pages too.
#
# The pipeline writes the respondents as exit_survey.csv
# (build_exit_survey() in preprocessing/preprocessing.py): one row per player
# of a real game who answered any item, with the condition, whether the player
# completed the game or was removed (`respondent`), the Likert items and age as
# numbers and the closed questions in lower case. The survey tests of
# checks/dropout_and_survey.R, the survey macros of report/descriptives.py and
# the survey figure all read that table, so none of them decides who a
# respondent is. `has_field()` says whether an item has any answer in it, so a
# consumer can skip an item nobody answered.
#
# Tests: stats/tests/test_survey.R

has_field <- function(survey, field) {
  field %in% names(survey) && any(!is.na(survey[[field]]))
}
