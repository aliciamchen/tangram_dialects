"""Group recognition in Phase 2 of the two social conditions, two panels.

By block: the share of speakers whom listeners judged to be from their own
original group, split by whether the speaker was (solid: the hit rate) or was
not (dashed: the false-alarm rate), over the judgments made in time, every
game a thin line behind its condition's mean and bootstrap interval, with
chance, a listener answering at random, at one half. The gap between the two bundles is
how well listeners told the groups apart.

Against specificity: one point per game, its recognition accuracy over the
phase (correct judgments over response opportunities, as the pipeline coded
them) against its group specificity at the end of Phase 2
(stats/primary/group_specificity.R), the association the preregistered
secondary analysis (stats/secondary/specificity_vs_social_accuracy.R) tests,
with the condition means and their intervals.

The social task is played only in Phase 2, so the block panel spans blocks 7
to 12. Both panels are placed by hand in the article in the square plotting
area of the Phase 2 panels, carry no letter or title, and leave the game
counts to the caption. Under a restriction (`restriction`, a rerun of
stats/primary/robustness.R named in _common.RESTRICTIONS) the table holds
only the games the rerun keeps, for the SI's version of the block panel
(intact_games.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.figures._common import (
    SPECIFICITY_LIMITS,
    condition_summary,
    draw_scatter,
    draw_trajectory,
    game_means,
    is_true,
    keep_games,
    panel_figure,
    reference_line,
    restricted_games,
    scatter_table,
    social_guesses,
    specificity_endpoints,
    trajectory_table,
)
from report.plot_style import BETWEEN_LINESTYLE, PHASE2_RANGE, PHASE2_TICKS, SQUARE_PANEL_AXES

SOCIAL_CONDITIONS = ["social_mixed", "social_first"]
SPEAKERS = ["in_group", "out_group"]
LINESTYLES = {"in_group": "-", "out_group": BETWEEN_LINESTYLE}
CHANCE = 0.5  # a listener answering at random says "same group" on half the trials
MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.42)  # the y label, nothing, nothing, the x axis


def data(dirs, restriction: str | None = None) -> pd.DataFrame:
    guesses = keep_games(social_guesses(dirs), restricted_games(dirs, restriction))
    guesses["speaker"] = np.where(is_true(guesses["speakerWasSameGroup"]), "in_group", "out_group")
    judged = guesses[is_true(guesses["onTime"]) & guesses["socialGuess"].notna()].copy()
    judged["same"] = (judged["socialGuess"] == "same_group").astype(float)
    by_block = [
        trajectory_table(part, "same", "by_block", SOCIAL_CONDITIONS).assign(speaker=speaker)
        for speaker, part in judged.groupby("speaker")
    ]
    ends = specificity_endpoints(dirs)
    games = (
        game_means(guesses, "correct").rename(columns={"value": "x"})
        .merge(ends[["gameId", "phase2"]].rename(columns={"phase2": "y"}), on="gameId")
        .dropna(subset=["y"])
    )
    trials = guesses[guesses["gameId"].isin(games["gameId"])]
    scatter = scatter_table(
        games, condition_summary(trials, "correct", SOCIAL_CONDITIONS), condition_summary(games, "y", SOCIAL_CONDITIONS)
    ).assign(panel="vs_specificity")
    return pd.concat([*by_block, scatter], ignore_index=True)


def plot(df: pd.DataFrame) -> dict:
    return {
        "by_block": _by_block(df[df["panel"] == "by_block"]),
        "vs_specificity": _vs_specificity(df[df["panel"] == "vs_specificity"]),
    }


def _by_block(table: pd.DataFrame):
    """Hits (solid) and false alarms (dashed), the condition in the color."""
    fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **MARGINS)
    for speaker in SPEAKERS:
        draw_trajectory(ax, table[table["speaker"] == speaker], SOCIAL_CONDITIONS, blocks=PHASE2_RANGE, linestyle=LINESTYLES[speaker])
    ax.set_xticks(PHASE2_TICKS)
    ax.set_ylim(0, 1)
    reference_line(ax, CHANCE, "chance")
    ax.set_ylabel("Share judged in-group")
    return fig


def _vs_specificity(table: pd.DataFrame):
    fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **MARGINS)
    draw_scatter(ax, table, SOCIAL_CONDITIONS)
    ax.set_xlim(CHANCE, 1.0)
    ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_ylim(*SPECIFICITY_LIMITS)
    ax.set_xlabel("Group recognition accuracy")
    ax.set_ylabel("Group specificity,\nend of Phase 2")
    return fig
