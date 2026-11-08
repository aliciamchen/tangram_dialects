"""The preregistered tests of H3, that anticipating the social-identification
incentive shapes Phase 1 conventions, each restricted to the two social
conditions with the directional contrast social_first > social_mixed (for
concreteness and word frequency, the prediction is lower in social_first).
Each panel puts one game per point in a column per condition under the
condition's mean as a black line and its bootstrap interval over games as an
outlined box, with the preregistered test's significance as a bracket
(significance_stars()); the per-game values are the data each test models,
summarized per game.

h3a: each game's group specificity at the end of Phase 1, the per-game
estimates of stats/primary/group_specificity.R that stats/primary/h3a.R
regresses on condition.

h3b_rate: within-group similarity over Phase 1 blocks 2 to 6, the per-group,
tangram and block means stats/primary/h3b.R models
(model_inputs/within_group_block_means.csv), as a trajectory with every game
a thin line behind its condition's mean and interval, for the test of the
slope, its significance beside the line ends; h3b_block3: each game's mean of those at block 3, for the test of early
alignment.

h3c_concreteness, h3c_uniqueness, h3c_word_frequency: the final Phase 1
descriptions of stats/primary/h3c.R (model_inputs/final_phase_properties.csv),
per game the share of content words that are concrete, the share that no
other group used for the tangram (the per-word outcomes the logistic models
use), and the mean Zipf frequency of the descriptions.

Under a restriction (`restriction`, a rerun of stats/primary/robustness.R
named in _common.RESTRICTIONS) the games are those the rerun keeps and each
bracket is the rerun's where the script refit the test (it refits H3a, H3b
and H3c on the complete games) and absent where it did not. The SI's
intact_games.py reads this table for its summary panels and draws none of
these.

The panels are written to results/plots/confirmatory_tests/ (FOLDER), carry no
letter or title, and leave the game counts and the test statistics to the
caption."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.figures._common import (
    condition_summary,
    draw_bracket,
    draw_columns,
    draw_trajectory,
    keep_games,
    numeric,
    panel_figure,
    read_model_input,
    restricted_games,
    specificity_endpoints,
    significance_stars,
    contrast_estimates,
    trajectory_table,
)
from report.plot_style import GAME_POINT, INK, SMALL_SIZE, SQUARE_PANEL_AXES, columns_panel_axes

FOLDER = "confirmatory_tests"
CONDITIONS = ["social_mixed", "social_first"]
RATE_BLOCKS = range(2, 7)  # Phase 1 blocks 2 to 6, where within-group pairs exist
EARLY_BLOCK = 3
GAME_JITTER = (0.18, 0.0)
SEED = 20261101
MEAN_HALF = 0.26
MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.42)
# panel: (estimates table, analysis, measure or None, y label, y limits, y ticks)
PANELS = {
    "h3a": ("primary/h3a", "h3a", "sbert", "Group specificity,\nend of Phase 1", (0, 0.8), [0, 0.4, 0.8]),
    "h3b_block3": ("primary/h3b", "h3b_block3", "sbert", "Within-group\nsimilarity, block 3", (0.2, 1.0), [0.2, 0.6, 1.0]),
    "h3c_concreteness": ("primary/h3c", "h3c_concreteness", None, "Concreteness", (0, 0.6), [0, 0.3, 0.6]),
    "h3c_uniqueness": ("primary/h3c", "h3c_uniqueness", None, "Lexical uniqueness", (0, 0.4), [0, 0.2, 0.4]),
    "h3c_word_frequency": ("primary/h3c", "h3c_word_frequency", None, "Word frequency\n(mean Zipf)", (3.8, 5.2), [4.0, 4.5, 5.0]),
}


def _p_value(dirs, table: str, analysis: str, measure: str | None, restriction: str | None = None) -> float:
    """The test's two-tailed p-value, NaN under a restriction the test was not rerun under."""
    est = contrast_estimates(dirs, table, analysis, restriction)
    if measure is not None and "measure" in est:
        est = est[est["measure"] == measure]
    if est.empty and restriction is not None:
        return float("nan")
    return float(est["p_value"].item())


def data(dirs, restriction: str | None = None) -> pd.DataFrame:
    kept = restricted_games(dirs, restriction)
    parts = []
    ends = keep_games(specificity_endpoints(dirs).dropna(subset=["phase1"]), kept)
    parts.append(ends[ends["condition"].isin(CONDITIONS)].assign(value=lambda d: d["phase1"], panel="h3a"))

    within = read_model_input(dirs, "within_group_block_means.csv")
    within = keep_games(within[within["condition"].isin(CONDITIONS) & (numeric(within["phaseNum"]) == 1)], kept)
    within["gameBlock"] = numeric(within["gameBlock"]).astype(int)
    within["similarity"] = numeric(within["similarity"])
    trajectory = trajectory_table(within[within["gameBlock"].isin(RATE_BLOCKS)], "similarity", "h3b_rate", CONDITIONS)
    early = within[within["gameBlock"] == EARLY_BLOCK].groupby(["gameId", "condition"], as_index=False)["similarity"].mean()
    parts.append(early.assign(value=early["similarity"], panel="h3b_block3"))

    final = read_model_input(dirs, "final_phase_properties.csv")
    final = keep_games(final[final["condition"].isin(CONDITIONS) & (numeric(final["phaseNum"]) == 1)], kept)
    for col in ("n_content_words", "n_concrete", "n_unique_words", "mean_zipf_freq"):
        final[col] = numeric(final[col])
    per_game = final.groupby(["gameId", "condition"]).agg(
        words=("n_content_words", "sum"), concrete=("n_concrete", "sum"), unique=("n_unique_words", "sum"),
        zipf=("mean_zipf_freq", "mean"),
    ).reset_index()
    parts.append(per_game.assign(value=per_game["concrete"] / per_game["words"], panel="h3c_concreteness"))
    parts.append(per_game.assign(value=per_game["unique"] / per_game["words"], panel="h3c_uniqueness"))
    parts.append(per_game.assign(value=per_game["zipf"], panel="h3c_word_frequency"))

    rows = []
    for part in parts:
        panel = part["panel"].iloc[0]
        games = part[["gameId", "condition", "value"]].assign(level="game", panel=panel)
        table, analysis, measure = PANELS[panel][:3]
        summary = condition_summary(games, "value", CONDITIONS).assign(
            panel=panel, p_value=_p_value(dirs, table, analysis, measure, restriction)
        )
        rows += [games, summary]
    rate_p = _p_value(dirs, "primary/h3b", "h3b_rate", "sbert", restriction)
    rows.append(trajectory.assign(p_value=rate_p))
    return pd.concat(rows, ignore_index=True)


def plot(df: pd.DataFrame) -> dict:
    panels = {}
    for panel, (_, _, _, ylabel, ylim, yticks) in PANELS.items():
        table = df[df["panel"] == panel]
        fig, ax = panel_figure(axes=columns_panel_axes(len(CONDITIONS)), **MARGINS)
        draw_columns(ax, table, GAME_POINT, GAME_JITTER, SEED, MEAN_HALF, CONDITIONS, level="game")
        span = ylim[1] - ylim[0]
        ax.set_ylim(ylim[0], ylim[1] + 0.2 * span)
        ax.set_yticks(yticks)
        p = table.loc[table["level"] == "condition", "p_value"].iloc[0]
        if np.isfinite(p):  # no mark for a test not run under the restriction
            draw_bracket(ax, 0, 1, ylim[1] + 0.07 * span, significance_stars(p))
        ax.set_ylabel(ylabel)
        panels[panel] = fig
    rate = df[df["panel"] == "h3b_rate"]
    fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **MARGINS)
    draw_trajectory(ax, rate, CONDITIONS, blocks=RATE_BLOCKS)
    ax.set_ylim(0.2, 1.0)
    ax.set_yticks([0.2, 0.6, 1.0])
    ax.set_ylabel("Within-group similarity")
    # The slope test's significance beside the line ends, where a bracket between columns has no place
    last = max(RATE_BLOCKS)
    ends = rate[(rate["level"] == "condition") & (rate["gameBlock"] == last)]
    if np.isfinite(rate["p_value"].iloc[0]):
        ax.annotate(significance_stars(rate["p_value"].iloc[0]), xy=(last, ends["value"].mean()), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=SMALL_SIZE, color=INK, annotation_clip=False)
    panels["h3b_rate"] = fig
    return {key: panels[key] for key in ["h3a", "h3b_rate", "h3b_block3", "h3c_concreteness", "h3c_uniqueness", "h3c_word_frequency"]}
