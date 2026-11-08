"""Referential accuracy, description length and the similarity of each description
to the one before it, block by block across the whole game with the four
conditions overlaid, one panel each: every game a thin line behind its
condition's mean and bootstrap interval. Accuracy is split by whether the
speaker was from the listener's original group, in-group solid and out-group
dashed; in Phase 1 every speaker is in-group, and in refer_separated
throughout. The similarity to the previous description is `simAdjacent`,
between successive descriptions of a tangram by the same original group, so
that in Phase 2 of the mixed conditions, where the group's members no longer
converse, it measures how much of the group's convention each description
keeps. Over Phase 1 the three quantities are the ones the pooled trend models
(stats/checks/convention_trends.R) test.

The panels are placed by hand beside the group-specificity panel in the
article, so they share its plotting area, carry no letter or title, and leave
the game count (30 per condition) to the caption. Under a restriction
(`restriction`, a rerun of stats/primary/robustness.R named in
_common.RESTRICTIONS) the table holds only the games the rerun keeps, for the
SI's version of the accuracy panel (intact_games.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.figures._common import (
    draw_trajectory,
    is_true,
    keep_games,
    listener_trials,
    numeric,
    panel_figure,
    read_measure,
    reference_line,
    restricted_games,
    speaker_utterances,
    trajectory_table,
)
from report.plot_style import (
    BETWEEN_LINESTYLE,
    BLOCK_PANEL_AXES,
    CONDITION_ORDER,
    SPARSE_BLOCK_TICKS,
)

PANELS = {
    "accuracy": "Referential accuracy",
    "length": "Description length\n(words)",
    "adjacent_similarity": "Similarity to the\nprevious description",
}
# Similarity on the same scale as the group-specificity panel beside it
LIMITS = {"accuracy": (0, 1), "length": (0, None), "adjacent_similarity": (0.1, 1.0)}
MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.42)  # the y label, nothing, nothing, the block axis
SPEAKERS = ["in_group", "out_group"]
LINESTYLES = {"in_group": "-", "out_group": BETWEEN_LINESTYLE}
CHANCE = 1 / 16  # one target among the sixteen tangrams displayed


def data(dirs, restriction: str | None = None) -> pd.DataFrame:
    kept = restricted_games(dirs, restriction)
    listeners = keep_games(listener_trials(dirs), kept)
    listeners["speaker"] = np.where(is_true(listeners["inGroupSpeaker"]), "in_group", "out_group")
    adjacent = read_measure(dirs, "adjacent_similarities.csv")
    adjacent["simAdjacent"] = numeric(adjacent["simAdjacent"])
    adjacent = keep_games(adjacent[adjacent["simAdjacent"].notna()], kept)
    adjacent["gameBlock"] = numeric(adjacent["gameBlock"]).astype(int)
    return pd.concat(
        [
            *(
                trajectory_table(part, "correct", "accuracy").assign(speaker=speaker)
                for speaker, part in listeners.groupby("speaker")
            ),
            trajectory_table(keep_games(speaker_utterances(dirs), kept), "uttLength", "length"),
            trajectory_table(adjacent, "simAdjacent", "adjacent_similarity"),
        ],
        ignore_index=True,
    )


def plot(df: pd.DataFrame) -> dict:
    panels = {}
    for panel, ylabel in PANELS.items():
        table = df[df["panel"] == panel]
        fig, ax = panel_figure(axes=BLOCK_PANEL_AXES, **MARGINS)
        if panel == "accuracy":
            _accuracy(ax, table)
        else:
            draw_trajectory(ax, table, CONDITION_ORDER)
            ax.set_ylim(*LIMITS[panel])
        ax.set_xticks(SPARSE_BLOCK_TICKS)
        ax.set_ylabel(ylabel)
        panels[panel] = fig
    return panels


def _accuracy(ax, table: pd.DataFrame) -> None:
    """Accuracy with in-group (solid) and out-group (dashed) speakers, the condition in the color."""
    for speaker in SPEAKERS:
        draw_trajectory(ax, table[table["speaker"] == speaker], CONDITION_ORDER, linestyle=LINESTYLES[speaker])
    ax.set_ylim(*LIMITS["accuracy"])
    reference_line(ax, CHANCE, "chance")
