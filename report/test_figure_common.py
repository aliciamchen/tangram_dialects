"""
Tests for what the figure modules share (report/figures/_common.py): the
descriptive convention-word coding, which words count as a group's convention
and whose words a description is coded as using, and the restriction under
which the SI draws the main panels again, which games it keeps and whose
estimates a test then reads.

Run with:
    uv run pytest report/test_figure_common.py -v
"""

import types

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from report.figures._common import (  # noqa: E402
    USES,
    code_convention_use,
    convention_vocabulary,
    draw_stimulus,
    intact_games,
    keep_games,
    network_layout,
    restricted_games,
    significance_stars,
    contrast_estimates,
)


def _utterance(player, group, phase, block, text, game="g", target="t"):
    return {"gameId": game, "playerId": player, "originalGroup": group, "target": target, "phaseNum": phase,
            "blockNum": block, "utterance": text}


def _phase1():
    return pd.DataFrame([
        # Group A agrees on "red rooster"; one member's earlier description does not count
        _utterance("a1", "A", 1, 0, "a chicken"),
        _utterance("a1", "A", 1, 5, "the red rooster"),
        _utterance("a2", "A", 1, 5, "red rooster again"),
        _utterance("a3", "A", 1, 5, "a rooster"),
        # B and C both say "bird", which therefore codes neither of them
        _utterance("b1", "B", 1, 5, "ugly bird"),
        _utterance("b2", "B", 1, 5, "the ugly bird"),
        _utterance("c1", "C", 1, 5, "bird on skates"),
        _utterance("c2", "C", 1, 5, "a bird with skates"),
    ])


def test_convention_vocabulary_keeps_words_two_members_share_and_no_other_group_has():
    vocabulary = convention_vocabulary(_phase1())
    assert vocabulary[("g", "A", "t")] == {"red", "rooster"}
    assert vocabulary[("g", "B", "t")] == {"ugly"}
    assert vocabulary[("g", "C", "t")] == {"skates"}


def test_convention_vocabulary_needs_two_final_descriptions():
    table = _phase1()
    table = table[table["playerId"] != "b2"]
    assert ("g", "B", "t") not in convention_vocabulary(table)


def test_code_convention_use_names_whose_words_a_description_carries():
    vocabulary = convention_vocabulary(_phase1())
    phase2 = pd.DataFrame([
        _utterance("a1", "A", 2, 6, "red rooster"),
        _utterance("a2", "A", 2, 6, "the rooster, the ugly one"),
        _utterance("a3", "A", 2, 6, "ugly thing"),
        _utterance("b1", "B", 2, 6, "a thing"),
    ])
    coded = code_convention_use(phase2, vocabulary)
    assert list(coded["use"]) == ["own", "both", "other", "neither"]
    assert list(coded["variant"]) == ["A", "both", "B", "none"]
    assert set(coded["use"]) <= set(USES)


def test_code_convention_use_drops_a_description_whose_group_has_no_vocabulary():
    coded = code_convention_use(pd.DataFrame([_utterance("d1", "A", 2, 6, "rooster", game="other")]), convention_vocabulary(_phase1()))
    assert coded.empty


def test_code_convention_use_drops_a_description_whose_group_words_another_group_shares():
    # C's only shared word is "bird", which B shares too, so C has no convention words of its own
    table = _phase1()
    table.loc[table["playerId"] == "c1", "utterance"] = "a bird"
    table.loc[table["playerId"] == "c2", "utterance"] = "the bird"
    vocabulary = convention_vocabulary(table)
    assert vocabulary[("g", "C", "t")] == set()
    coded = code_convention_use(pd.DataFrame([_utterance("c1", "C", 2, 6, "ugly bird")]), vocabulary)
    assert coded.empty


def test_network_layout_places_nodes_evenly_on_the_unit_circle():
    positions = network_layout(9)
    assert positions.shape == (9, 2)
    assert np.allclose(np.linalg.norm(positions, axis=1), 1)
    steps = np.diff(np.arctan2(positions[:, 1], positions[:, 0]))
    assert np.allclose(np.abs((steps + np.pi) % (2 * np.pi) - np.pi), 2 * np.pi / 9)


def test_draw_stimulus_is_upright_whichever_way_the_y_axis_runs():
    # A pointer whose apex is at the top of the SVG (small y, since SVG y runs downward).
    polygons = [[[0, 10], [10, 10], [5, 0]]]
    for inverted in (False, True):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 100)
        ax.set_ylim((100, 0) if inverted else (0, 100))
        draw_stimulus(ax, polygons, (50, 50), 20)
        apex = ax.patches[0].get_xy()[2]
        assert np.isclose(apex[0], 50) and np.isclose(abs(apex[1] - 50), 10)
        # The apex is displayed above the center on both axes.
        assert (apex[1] < 50) if inverted else (apex[1] > 50)
        plt.close(fig)


# ── Restrictions ─────────────────────────────────────────────


def _dataset(tmp_path) -> types.SimpleNamespace:
    """Three games as the pipeline codes them: one whole, one that lost a player, one that ended with one group; the primary H4a test and the complete-games reruns of H1/H2 and H4a."""
    tables, estimates = tmp_path / "tables", tmp_path / "estimates"
    tables.mkdir()
    (estimates / "primary").mkdir(parents=True)
    pd.DataFrame({
        "gameId": ["whole", "short", "ended"], "condition": ["refer_mixed"] * 3, "activeGroups": [3, 3, 1],
        "ranToCompletion": [True, True, False], "completeNine": [True, False, False],
    }).to_csv(tables / "games.csv", index=False)
    pd.DataFrame({"analysis": ["h4a"], "term": ["H4a"], "p_value": [0.1], "n_games": [59]}).to_csv(
        estimates / "primary" / "h4.csv", index=False
    )
    pd.DataFrame({"analysis": ["h1_h2", "h4a"], "term": ["H1", "H4a"], "p_value": [0.01, 0.2],
                  "restriction": ["complete_games"] * 2, "n_games": [48, 33]}).to_csv(estimates / "primary" / "robustness.csv", index=False)
    (estimates / "exploratory").mkdir()
    pd.DataFrame({"analysis": ["social"], "term": ["first_vs_mixed"], "p_value": [0.5], "restriction": ["complete_games"]}).to_csv(
        estimates / "exploratory" / "social_robustness.csv", index=False
    )
    return types.SimpleNamespace(tables=tables, estimates=estimates)


def test_intact_games_are_the_games_that_kept_all_nine_players(tmp_path):
    assert intact_games(_dataset(tmp_path)) == {"whole"}


def test_restricted_games_names_a_rerun_and_keeps_every_game_without_one(tmp_path):
    dirs = _dataset(tmp_path)
    assert restricted_games(dirs, None) is None
    assert restricted_games(dirs, "complete_games") == {"whole"}
    with pytest.raises(ValueError, match="No restriction named"):
        restricted_games(dirs, "three_groups")


def test_keep_games_keeps_the_named_games_or_all_of_them():
    table = pd.DataFrame({"gameId": ["a", "b"], "value": [1, 2]})
    assert keep_games(table, {"b"})["value"].tolist() == [2]
    assert keep_games(table, None)["value"].tolist() == [1, 2]
    assert keep_games(table, None) is not table  # a copy, safe to add columns to


def test_contrast_estimates_read_the_primary_table_or_the_rerun(tmp_path):
    dirs = _dataset(tmp_path)
    assert contrast_estimates(dirs, "primary/h4", "h4a")["p_value"].item() == 0.1
    assert contrast_estimates(dirs, "primary/h4", "h4a", "complete_games")["p_value"].item() == 0.2
    assert contrast_estimates(dirs, "primary/h3a", "h3a", "complete_games").empty
    reruns = contrast_estimates(dirs, "exploratory/social", "social", "complete_games", reruns="exploratory/social_robustness")
    assert reruns["p_value"].item() == 0.5


def test_significance_stars_mark_the_thresholds_and_refuse_a_missing_p_value():
    assert [significance_stars(p) for p in (0.0004, 0.004, 0.04, 0.4)] == ["***", "**", "*", "n.s."]
    with pytest.raises(ValueError, match="needs a p-value"):
        significance_stars(float("nan"))
