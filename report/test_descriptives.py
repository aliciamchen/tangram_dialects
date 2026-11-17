"""
Unit tests for the descriptive statistics (report/descriptives.py) on
small hand-built tables: the attrition definitions, the description
coverage, the endpoint macros and the exit-survey recodes.

Run with:
    uv run pytest report/test_descriptives.py -v
"""

import numpy as np
import pandas as pd

import pytest

from preprocessing.preprocessing import build_exit_survey
from report.descriptives import (
    classifier_macros,
    condition_count_macros,
    condition_counts,
    count_words,
    description_coverage,
    description_coverage_by_condition,
    endpoint_macros,
    felt_human_flags,
    final_words_by_condition,
    game_status,
    group_attrition,
    length_flag_count,
    listener_accuracy_at_phase2_ends,
    listener_accuracy_by_speaker_group,
    phase1_final_words_by_condition,
    phase1_specificity_by_condition,
    phase_specificity_change_by_condition,
    player_attrition,
    response_opportunities,
    reshuffle_macros,
    response_opportunity_macros,
    session_minutes_macros,
    specificity_measure_agreement,
    survey_macros,
)

# The attrition columns are the pipeline's (preprocessing.add_removal_columns and
# add_completion_columns); the tables here carry them as written.
GAMES = pd.DataFrame(
    {
        "gameId": ["g1", "g2", "g0"],
        "condition": ["social_mixed", "refer_separated", None],
        "activeGroups": [3, 1, 3],
        "ranToCompletion": [True, False, True],
        "completeNine": [False, False, True],
    }
)
# g1: group A keeps one of two, group B keeps one of two, group C keeps both;
# g2: both groups lose everyone or all but one; g0 is a test game
PLAYERS = pd.DataFrame(
    {
        "gameId": ["g1"] * 6 + ["g2"] * 3 + ["g0"],
        "playerId": [f"p{i}" for i in range(10)],
        "originalGroup": ["A", "A", "B", "B", "C", "C", "A", "A", "B", "A"],
        "isActive": [True, False, True, False, True, True, False, False, True, True],
        "exitReason": [None, "player timeout", None, "group disbanded", None, None,
                       "low accuracy", "low accuracy", None, None],
        "removed": [False, True, False, True, False, False, True, True, False, False],
        "removalReason": [None, "idle", None, "group disbanded", None, None,
                          "low accuracy", "low accuracy", None, None],
        "removedNoFault": [False, False, False, True, False, False, False, False, False, False],
        "exitSurvey_understood": ["yes", "yes", None, "yes", None, None, None, None, "Yes", "yes"],
        "exitSurvey_groupIdentification": ["6", "3", None, "7", None, None, None, None, "5", "7"],
        "exitSurvey_feltHuman": ["yes", "no", None, "yes", None, None, None, None, "no", "no"],
        "exitSurvey_age": ["31", "x", None, "40", None, None, None, None, "25", "50"],
    }
)


def test_player_attrition_reads_the_pipelines_columns_for_the_real_games():
    pa = player_attrition(PLAYERS, GAMES)
    assert len(pa) == 9, "players of the test game are left out"
    by = pa.set_index("playerId")
    assert not by.loc["p0", "removed"] and by.loc["p0", "reason"] == "completed"
    assert by.loc["p1", "reason"] == "idle" and not by.loc["p1", "no_fault"]
    assert by.loc["p3", "reason"] == "group disbanded" and by.loc["p3", "no_fault"]
    assert int(pa["removed"].sum()) == 4 and int(pa["no_fault"].sum()) == 1


def test_group_and_game_status_follow_the_servers_rules():
    ga = group_attrition(PLAYERS, GAMES)
    lost = ga.set_index(["gameId", "originalGroup"])["lost"]
    assert lost.loc[("g1", "A")] and lost.loc[("g1", "B")], "one active member left is below the minimum group size"
    assert not lost.loc[("g1", "C")]
    assert lost.loc[("g2", "A")] and lost.loc[("g2", "B")]
    assert int(ga["lost"].sum()) == 4
    status = game_status(GAMES).set_index("gameId")
    assert status.loc["g1", "ran_to_completion"] and not status.loc["g1", "complete_nine"]
    assert not status.loc["g2", "ran_to_completion"]
    assert "g0" not in status.index


def test_description_coverage_counts_speaker_trials_without_an_utterance():
    trials = pd.DataFrame(
        {
            "gameId": ["g1", "g1", "g1", "g0"],
            "playerId": ["p0", "p0", "p2", "p9"],
            "role": ["speaker", "speaker", "speaker", "speaker"],
            "phaseNum": [1.0, 1.0, 2.0, 1.0],
            "blockNum": [0.0, 1.0, 0.0, 0.0],
            "target": ["t1", "t2", "t1", "t1"],
            "condition": ["social_mixed", "social_mixed", "social_mixed", None],
        }
    )
    utts = pd.DataFrame(
        {"gameId": ["g1", "g1"], "playerId": ["p0", "p0"], "phaseNum": [1, 1], "blockNum": [0, 1], "target": ["t1", "t2"]}
    )
    assert description_coverage(trials, utts) == (1, 3)
    by = description_coverage_by_condition(trials, utts)
    assert list(by["condition"].astype(str)) == ["social_mixed"], "the test game's trials are left out"
    assert by.loc[0, "speaker_trials"] == 3 and by.loc[0, "without"] == 1
    assert abs(by.loc[0, "prop_without"] - 1 / 3) < 1e-12


def test_response_opportunities_count_what_became_of_each_opportunity():
    trials = pd.DataFrame(
        {
            "gameId": ["g1"] * 6,
            "role": ["listener"] * 5 + ["speaker"],
            "phaseNum": [1.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            "condition": ["social_mixed"] * 6,
            "responseOpportunity": [True, True, True, True, False, True],
            # correct on time, a timeout, a late correct answer (coded 0), an excluded row
            "correct": [1, 0, 0, np.nan, np.nan, np.nan],
            "onTime": [True, False, False, True, np.nan, np.nan],
            "lateClick": [False, False, True, False, False, False],
        }
    )
    guesses = pd.DataFrame(
        {
            "gameId": ["g1"] * 2,
            "phaseNum": [2, 2],
            "condition": ["social_mixed"] * 2,
            "responseOpportunity": [True, True],
            "correct": [1, 0],
            "onTime": [True, True],
            "lateSocialGuess": [False, False],
        }
    )
    out = response_opportunities(trials, guesses)
    ref = out[out["task"] == "referential"].iloc[0]
    assert len(out) == 2 and ref["phaseNum"] == 1, "a row that is no opportunity, or a speaker's, is not counted"
    assert ref["opportunities"] == 4 and ref["scored"] == 3 and ref["on_time"] == 1
    assert abs(ref["accuracy"] - 1 / 3) < 1e-12 and ref["accuracy_on_time"] == 1.0
    assert ref["late"] == 1 and abs(ref["late_rate"] - 1 / 3) < 1e-12
    m = response_opportunity_macros(out)
    assert m["overviewRefPhaseOneSocialMixedOpportunities"] == "4"
    assert m["overviewRefPhaseOneSocialMixedAccuracy"] == "0.333"
    assert m["overviewSocialPhaseTwoSocialMixedOnTimeRate"] == "1.000"


def test_condition_counts_follow_the_attrition_definitions():
    t = {"games": GAMES, "players": PLAYERS, "participant_exclusions": pd.DataFrame({"playerId": ["p1"], "reason": ["x"]})}
    counts = condition_counts(t).set_index("condition")
    assert list(counts.index) == ["refer_separated", "social_mixed"], "conditions with no games are left out"
    assert counts.loc["social_mixed", "games"] == 1 and counts.loc["social_mixed", "games_ran_to_completion"] == 1
    assert counts.loc["social_mixed", "removed"] == 2 and counts.loc["social_mixed", "removed_no_fault"] == 1
    assert counts.loc["social_mixed", "groups_lost"] == 2 and counts.loc["social_mixed", "excluded_participants"] == 1
    assert counts.loc["refer_separated", "games_ran_to_completion"] == 0
    m = condition_count_macros(condition_counts(t))
    assert m["overviewNGamesCompleteSocialMixed"] == 0 and m["overviewNExcludedParticipantsReferSeparated"] == 0


def test_endpoint_macros_report_the_first_and_last_observed_block():
    df = pd.DataFrame(
        {
            "condition": ["refer_mixed"] * 5 + [None],
            "gameBlock": [2, 2, 5, 12, 12, 1],
            "value": [0.2, 0.4, 0.5, 0.8, np.nan, 0.9],
        }
    )
    m = endpoint_macros("x", df, "value", digits=2)
    assert m == {"xN": "4", "xFirstBlock": 2, "xLastBlock": 12, "xFirst": "0.30", "xLast": "0.80"}
    assert endpoint_macros("x", df.iloc[0:0], "value") == {}


def test_semantic_drift_macros_describe_refer_separated_with_a_game_level_interval():
    from report.descriptives import semantic_drift_macros

    rows = []
    for g, m in [("a", 0.6), ("b", 0.8), ("c", 0.9)]:
        rows += [{"gameId": g, "condition": "refer_separated", "simPhase1Phase2": m + d} for d in (-0.05, 0.0, 0.05)]
    rows += [{"gameId": "z", "condition": "refer_mixed", "simPhase1Phase2": 0.1}] * 3
    out = semantic_drift_macros(pd.DataFrame(rows))
    assert set(out) == {"exploratoryDriftMean", "exploratoryDriftLo", "exploratoryDriftHi", "exploratoryDriftGames"}
    assert out["exploratoryDriftMean"] == "0.767"  # refer_separated rows only
    assert float(out["exploratoryDriftLo"]) <= 0.767 <= float(out["exploratoryDriftHi"])
    assert out["exploratoryDriftGames"] == "3"
    assert semantic_drift_macros(pd.DataFrame()) == {}


def test_phase1_specificity_and_words_by_condition_follow_the_condition_order():
    gs = pd.DataFrame(
        {
            "condition": ["social_first", "social_first", "refer_mixed", "refer_mixed", "refer_mixed"],
            "window": ["phase1_final", "phase1_final", "phase1_final", "phase2_final", "phase1_final"],
            "coefficient": [0.2, 0.4, 0.5, 9.0, np.nan],
            "estimable": [True, True, True, True, False],
        }
    )
    out = phase1_specificity_by_condition(gs)
    assert list(out["condition"].astype(str)) == ["refer_mixed", "social_first"]
    by = out.set_index(out["condition"].astype(str))
    assert by.loc["social_first", "games"] == 2 and abs(by.loc["social_first", "mean"] - 0.3) < 1e-12
    utts = pd.DataFrame(
        {
            "gameId": ["g1"] * 4 + ["g2"] * 2,
            "condition": ["social_first"] * 4 + ["refer_mixed"] * 2,
            "phaseNum": [1, 1, 1, 2, 1, 1],
            "blockNum": [0, 3, 5, 0, 4, 5],
            "finalWindow": [False, True, True, False, True, True],
            "uttLength": [10, 6, 4, 1, 3, 5],
        }
    )
    words = phase1_final_words_by_condition(utts)
    words = words.set_index(words["condition"].astype(str))
    assert words.loc["social_first", "mean_words"] == 5.0, "blocks 3 and 5 are in the pipeline's final window"
    assert words.loc["refer_mixed", "mean_words"] == 4.0


def test_felt_human_flags_count_games_and_groups_from_the_pipelines_survey_table():
    # The respondents and their typed answers are the pipeline's (tested in preprocessing/test_preprocessing.py)
    survey = build_exit_survey(PLAYERS, GAMES)
    assert set(survey["playerId"]) == {"p0", "p1", "p3", "p8"}, "respondents of real games with any answer"
    assert len(felt_human_flags(survey)) == 2 and len(felt_human_flags(survey, level="group")) == 2


def test_phase_specificity_change_pairs_games_within_condition():
    gs = pd.DataFrame(
        {
            "gameId": ["g1", "g1", "g2", "g2", "g3", "g4", "g4"],
            "condition": ["refer_mixed"] * 4 + ["refer_mixed"] + ["social_mixed"] * 2,
            "window": ["phase1_final", "phase2_final"] * 2 + ["phase1_final", "phase1_final", "phase2_final"],
            "coefficient": [0.4, 0.2, 0.6, 0.3, 0.5, 0.4, 0.4],
            "estimable": [True] * 6 + [False],
        }
    )
    out = phase_specificity_change_by_condition(gs)
    out = out.set_index(out["condition"].astype(str))
    rm = out.loc["refer_mixed"]
    assert rm["games_phase1"] == 3 and rm["games_phase2"] == 2 and rm["games_paired"] == 2
    assert abs(rm["mean_phase1"] - 0.5) < 1e-12, "the Phase 1 mean uses every estimable Phase 1 game"
    assert abs(rm["change"] - (-0.25)) < 1e-12, "the change averages the paired differences"
    assert abs(rm["ratio"] - 0.5) < 1e-12, "the ratio compares the paired means"
    assert rm["change_lower"] <= rm["change"] <= rm["change_upper"]
    sm = out.loc["social_mixed"]
    assert sm["games_phase2"] == 0 and np.isnan(sm["change"]), "an inestimable endpoint gives no change"
    assert list(out.index) == ["refer_mixed", "social_mixed"]


def test_listener_accuracy_by_speaker_group_averages_listeners_with_game_intervals():
    advantage = pd.DataFrame(
        {
            "gameId": ["g1", "g1", "g2", "g3"],
            "condition": ["social_mixed", "social_mixed", "social_mixed", "refer_mixed"],
            "playerId": ["p1", "p2", "p3", "p4"],
            "acc_in": [1.0, 0.8, 0.9, 1.0],
            "acc_out": [0.5, 0.7, np.nan, 0.8],
        }
    )
    out = listener_accuracy_by_speaker_group(advantage)
    by = out.set_index([out["condition"].astype(str), "listener_type"])
    assert abs(by.loc[("social_mixed", "in_group"), "accuracy"] - 0.9) < 1e-12
    assert by.loc[("social_mixed", "in_group"), "listeners"] == 3
    assert abs(by.loc[("social_mixed", "out_group"), "accuracy"] - 0.6) < 1e-12, "a missing accuracy is dropped"
    assert by.loc[("social_mixed", "out_group"), "listeners"] == 2
    assert ("refer_mixed", "in_group") not in by.index, "one listener cannot be bootstrapped"
    assert list(out["condition"].astype(str).unique()) == ["social_mixed"]


def test_listener_accuracy_at_phase2_ends_reads_the_first_and_last_block_of_the_mixed_conditions():
    # Two refer_mixed games over Phase 2 blocks 1 to 3 and a refer_separated game,
    # whose speakers are all in-group; then the rows the summary leaves out: a
    # middle block, a Phase 1 trial, a listener with no opportunity and a speaker
    rows = [
        ("g1", "refer_mixed", "listener", 2, 1, True, True, 1), ("g1", "refer_mixed", "listener", 2, 1, False, True, 0),
        ("g2", "refer_mixed", "listener", 2, 1, True, True, 0), ("g2", "refer_mixed", "listener", 2, 1, False, True, 0),
        ("g1", "refer_mixed", "listener", 2, 3, True, True, 1), ("g1", "refer_mixed", "listener", 2, 3, False, True, 1),
        ("g2", "refer_mixed", "listener", 2, 3, True, True, 1), ("g2", "refer_mixed", "listener", 2, 3, False, True, 0),
        ("g3", "refer_separated", "listener", 2, 1, True, True, 1), ("g3", "refer_separated", "listener", 2, 3, True, True, 1),
        ("g1", "refer_mixed", "listener", 2, 2, False, True, 1),
        ("g1", "refer_mixed", "listener", 1, 1, True, True, 0),
        ("g1", "refer_mixed", "listener", 2, 1, False, False, np.nan),
        ("g1", "refer_mixed", "speaker", 2, 1, True, False, np.nan),
    ]
    trials = pd.DataFrame(rows, columns=["gameId", "condition", "role", "phaseNum", "block", "inGroupSpeaker",
                                         "responseOpportunity", "correct"])
    out = listener_accuracy_at_phase2_ends(trials)
    by = out.set_index([out["condition"].astype(str), "block", "listener_type"])
    assert by.loc[("refer_mixed", "first", "in_group"), "accuracy"] == 0.5
    assert by.loc[("refer_mixed", "first", "out_group"), "accuracy"] == 0.0, "the listener without an opportunity is not a miss"
    assert by.loc[("refer_mixed", "last", "in_group"), "accuracy"] == 1.0
    assert by.loc[("refer_mixed", "last", "out_group"), "accuracy"] == 0.5
    assert by.loc[("refer_mixed", "first", "in_group"), "games"] == 2
    assert len(out) == 4, "the condition without out-group speakers and the middle block are left out"
    assert listener_accuracy_at_phase2_ends(trials[trials["phaseNum"] == 1]).empty


def test_specificity_measure_agreement_correlates_the_endpoints_estimable_on_both_measures():
    index = {"gameId": ["g1", "g1", "g2", "g2"], "condition": ["refer_mixed"] * 2 + ["social_mixed"] * 2,
             "window": ["phase1_final", "phase2_final"] * 2}
    sbert = pd.DataFrame({**index, "coefficient": [0.1, 0.2, 0.3, 0.9], "estimable": [True, True, True, False]})
    jaccard = pd.DataFrame({**index, "coefficient": [0.2, 0.4, 0.6, 0.1], "estimable": [True] * 4})
    out = specificity_measure_agreement(sbert, jaccard)
    assert out == {"exploratoryGsMeasuresR": "1.00", "exploratoryGsMeasuresN": "3"}, "the inestimable endpoint is left out"
    assert specificity_measure_agreement(sbert, pd.DataFrame()) == {}


def test_count_words_counts_whitespace_separated_runs():
    assert count_words(pd.Series(["a rooster head", "  witch ", None, ""])) == 4


def test_final_words_by_condition_takes_each_phase_separately():
    utts = pd.DataFrame(
        {
            "gameId": ["g1"] * 5,
            "condition": ["social_first"] * 5,
            "phaseNum": [1, 1, 2, 2, 2],
            "blockNum": [4, 5, 0, 1, 5],
            "finalWindow": [True, True, False, False, True],
            "uttLength": [6, 4, 10, 8, 2],
        }
    )
    p1 = final_words_by_condition(utts, phase=1)
    p1 = p1.set_index(p1["condition"].astype(str))
    p2 = final_words_by_condition(utts, phase=2)
    p2 = p2.set_index(p2["condition"].astype(str))
    assert p1.loc["social_first", "mean_words"] == 5.0
    assert p2.loc["social_first", "mean_words"] == 2.0, "Phase 2 blocks 0 and 1 are outside the final window"
    assert phase1_final_words_by_condition(utts).equals(final_words_by_condition(utts, phase=1))


def test_final_words_read_the_window_from_the_pipeline_not_from_the_last_block_played():
    """A game that stopped after its fourth Phase 2 block has that block in the window, not its last three blocks."""
    utts = pd.DataFrame(
        {
            "gameId": ["g1"] * 3,
            "condition": ["refer_separated"] * 3,
            "phaseNum": [2, 2, 2],
            "blockNum": [1, 2, 3],
            "finalWindow": ["False", "False", "True"],
            "uttLength": [9, 7, 2],
        }
    )
    out = final_words_by_condition(utts, phase=2)
    assert out["mean_words"].tolist() == [2.0] and out["games"].tolist() == [1]


def test_session_minutes_macros_take_real_games_only():
    players = PLAYERS.assign(minutesSpent=[40, 10, 50, 12, 48, 52, 30, 31, 55, 99])
    out = session_minutes_macros(players, GAMES)
    assert out["overviewMinutesN"] == "9", "the test game g0 is left out"
    assert out["overviewMinutesMedian"] == "40"
    assert float(out["overviewMinutesLowerDecile"]) < float(out["overviewMinutesMedian"]) < float(out["overviewMinutesUpperDecile"])
    assert session_minutes_macros(players.drop(columns=["minutesSpent"]), GAMES) == {}


def test_survey_macros_count_each_demographic_option():
    # The respondents of real games are p0, p1, p3 and p8; p9 is in the test game
    players = PLAYERS.assign(
        exitSurvey_gender=["Female", "male", None, "female", None, None, None, None, "non-binary", "male"],
    )
    out = survey_macros({"games": GAMES, "players": players, "exit_survey": build_exit_survey(players, GAMES)})
    assert out["surveyGenderN"] == 4 and out["surveyGenderFemale"] == 2 and out["surveyGenderFemalePct"] == "50.0"
    assert out["surveyGenderMale"] == 1 and out["surveyGenderNonBinary"] == 1
    assert out["surveyGenderOther"] == 0 and out["surveyGenderNotSaid"] == 0, "an option nobody chose is written as 0"
    assert out["surveyAgeMin"] == "25" and out["surveyAgeMax"] == "40"
    assert "surveyEducationN" not in out, "an item the dataset lacks contributes no macro"
    unknown = players.assign(exitSurvey_gender=["robot"] + list(players["exitSurvey_gender"][1:]))
    with pytest.raises(SystemExit):
        survey_macros({"games": GAMES, "players": unknown, "exit_survey": build_exit_survey(unknown, GAMES)})


def test_classifier_macros_count_the_speaker_messages_left_out():
    messages = pd.DataFrame({
        "gameId": ["g1"] * 3, "roundId": ["r1"] * 3, "senderId": ["a", "a", "b"], "timestamp": [1, 2, 3],
        "text": ["a bird", "thanks", "the bird?"], "senderRole": ["speaker", "speaker", "listener"],
        "condition": ["refer_mixed"] * 3,
    })
    classified = messages.iloc[:2].assign(is_referential=[True, False], llm_label=["R", "NR"])
    out = classifier_macros(messages, classified)
    assert out["overviewNSpeakerMessages"] == "2", "listener messages are never classified"
    assert out["overviewNNonReferentialMessages"] == "1" and out["overviewPctNonReferentialMessages"] == "50.0"
    assert classifier_macros(messages, pd.DataFrame()) == {}


def test_length_flag_count_takes_real_games_only():
    players = PLAYERS.assign(lengthIncreaseFlag=[True, False, False, True, False, False, False, False, False, True])
    assert length_flag_count(players, GAMES) == 2, "the flagged player of the test game is left out"
    assert length_flag_count(PLAYERS, GAMES) is None


def test_reshuffle_macros_count_each_reshuffled_round_once():
    # Two reshuffled rounds, three rows each (a full one and one with a pair),
    # a round that was not reshuffled and a test game's round
    trials = pd.DataFrame({
        "gameId": ["g1"] * 3 + ["g1"] * 3 + ["g2"] + ["g0"],
        "roundId": ["r1"] * 3 + ["r2"] * 3 + ["r3"] + ["r4"],
        "condition": ["social_mixed"] * 6 + ["refer_separated"] + [None],
        "reshuffleGroups": [3] * 3 + [3] * 3 + [np.nan] + [3],
        "reshuffleTrios": [3] * 3 + [2] * 3 + [np.nan] + [3],
        "reshuffleTriosOk": [3] * 3 + [2] * 3 + [np.nan] + [3],
        "reshufflePairs": [0] * 3 + [1] * 3 + [np.nan] + [0],
    })
    out = reshuffle_macros(trials)
    assert out["overviewReshuffleNTrials"] == "2", "the unshuffled round and the test game are left out"
    assert out["overviewReshuffleNGroups"] == "6" and out["overviewReshuffleNPairs"] == "1"
    assert out["overviewReshuffleNTrios"] == "5" and out["overviewReshuffleNTriosOk"] == "5"
    assert reshuffle_macros(trials.drop(columns=["reshufflePairs"])) == {}
