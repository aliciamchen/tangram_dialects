"""How group-specific the descriptions of each target tangram were: one row per
tangram, ordered by its specificity at the end of Phase 1, with the end of
Phase 1 over all games as a filled point in ink and the end of Phase 2 in each
condition as a hollow point in the condition's color, each with its bootstrap
interval over games. The ordering shows which shapes left groups room to
differ, and the Phase 2 points whether the conditions' differences hold shape
by shape rather than resting on a few.

A tangram's specificity in a game is descriptive: the mean similarity of its
final descriptions between speakers of the same original group minus that
between speakers of different groups, on the sentence embeddings (the final
windows of measures/pairwise_similarities.csv). The preregistered per-game
estimate pools the tangrams in one model and has no per-tangram value. The
Phase 1 point pools all four conditions, social_first included, whose players
knew of the social goal from the start. Each game used one of two sets of six
tangrams, so a tangram's Phase 1 point rests on half the games and each of its
Phase 2 points on about fifteen.

One tall panel on TANGRAM_ROWS_PANEL_AXES, each row labeled by its tangram in
gray left of the axis, with no letter, title or key; which mark is which phase
and condition is left to the legend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.bootstrap import bootstrap_mean
from report.figures._common import (
    SPECIFICITY_LIMITS,
    in_conditions,
    panel_figure,
    read_measure,
    read_table,
    stimulus_row_labels,
)
from report.plot_style import (
    CONDITION_COLORS,
    CONDITION_ORDER,
    INK,
    INTERVAL_LINE,
    ROW_POINT,
    TANGRAM_ROWS_PANEL_AXES,
)

WINDOWS = {"phase1_final": 1, "phase2_final": 2}
POOLED = "all"  # the condition of the Phase 1 rows, pooled over every game
# The marks of a row top to bottom, and the space between them in rows
MARKS = [(1, POOLED)] + [(2, c) for c in CONDITION_ORDER]
STEP = 0.15
GLYPH = dict(size=0.18, gap=0.06)  # inches
MARGINS = dict(left=0.5, right=0.12, top=0.1, bottom=0.42)  # the y label and the glyphs, nothing, nothing, the x label


def data(dirs) -> pd.DataFrame:
    pairs = read_measure(dirs, "pairwise_similarities.csv")
    pairs = pairs[pairs["window"].isin(WINDOWS)].copy()
    pairs["pair_type"] = np.where(pairs["sameGroup"].astype(int) == 1, "within", "between")
    means = pairs.groupby(["gameId", "target", "window", "pair_type"])["similarity"].mean().unstack("pair_type")
    games = (means["within"] - means["between"]).rename("value").dropna().reset_index()
    games = in_conditions(games.merge(read_table(dirs, "games.csv")[["gameId", "condition"]], on="gameId"))
    games["phase"] = games["window"].map(WINDOWS)
    rows = []
    for (target, phase), part in games.groupby(["target", "phase"]):
        series = [(POOLED, part)] if phase == 1 else list(part.groupby("condition"))
        for condition, sub in series:
            est = bootstrap_mean(sub, "value")
            rows.append({"target": target, "phase": phase, "condition": condition, "value": est.mean,
                         "lower": est.lower, "upper": est.upper, "n_games": est.n_games, "n_obs": est.n_obs})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame):
    fig, ax = panel_figure(axes=TANGRAM_ROWS_PANEL_AXES, **MARGINS)
    order = df[df["phase"] == 1].sort_values("value", ascending=False)["target"].tolist()
    row = {t: len(order) - i for i, t in enumerate(order)}  # the highest Phase 1 value in the top row
    for k, (phase, condition) in enumerate(MARKS):
        part = df[(df["phase"] == phase) & (df["condition"] == condition)]
        y = part["target"].map(row) + ((len(MARKS) - 1) / 2 - k) * STEP
        color = INK if condition == POOLED else CONDITION_COLORS[condition]
        face = color if phase == 1 else "white"
        ax.hlines(y, part["lower"], part["upper"], color=color, zorder=2, **INTERVAL_LINE)
        ax.scatter(part["value"], y, facecolors=face, edgecolors=color, zorder=3, **ROW_POINT)
    ax.set_xlim(*SPECIFICITY_LIMITS)
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_ylim(0.5, len(order) + 0.5)
    stimulus_row_labels(ax, order, [row[t] for t in order], **GLYPH)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Group specificity")
    ax.set_ylabel("Tangrams (ordered by the end of Phase 1)", labelpad=(GLYPH["size"] + GLYPH["gap"]) * 72 + 3)
    return fig  # the game counts, 60 per tangram in Phase 1 and about 15 per condition in Phase 2, are left to the caption
