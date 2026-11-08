"""How long listeners took to choose a shape, block by block across the whole
game with the four conditions overlaid: every game a thin line behind its
condition's mean and bootstrap interval, Phase 2 split by whether the speaker
was from the listener's original group, in-group solid and out-group dashed
(in Phase 1 every speaker is in-group, and in refer_separated throughout).
A description in another group's convention takes longer to resolve, and the
gap narrows over the phase as listeners learn the other groups' words. The
time is `selectionRt`, from the shapes appearing on the listener's screen to
the listener's choice, both ends on the listener's own clock, so it is
comparable across participants and conditions; a listener who made no choice
has none. The selection stage is 45 s in Phase 1 and 25 s in Phase 2, which
is part of the drop at the boundary. Not preregistered.

The panel shares the plotting area of the block panels, carries no letter or
title, and leaves the game counts to the caption."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.figures._common import (
    draw_trajectory,
    is_true,
    listener_trials,
    numeric,
    panel_figure,
    trajectory_table,
)
from report.plot_style import BETWEEN_LINESTYLE, BLOCK_PANEL_AXES, CONDITION_ORDER, SPARSE_BLOCK_TICKS

SPEAKERS = ["in_group", "out_group"]
LINESTYLES = {"in_group": "-", "out_group": BETWEEN_LINESTYLE}
MAX_SECONDS = 60  # a longer time is a clock artifact, not a choice
MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.42)  # the y label, nothing, nothing, the block axis


def data(dirs) -> pd.DataFrame:
    trials = listener_trials(dirs)
    trials["seconds"] = numeric(trials["selectionRt"]) / 1000
    trials = trials[trials["seconds"].between(0, MAX_SECONDS, inclusive="right")].copy()
    trials["speaker"] = np.where(is_true(trials["inGroupSpeaker"]), "in_group", "out_group")
    return pd.concat(
        [trajectory_table(part, "seconds", "response_time").assign(speaker=speaker) for speaker, part in trials.groupby("speaker")],
        ignore_index=True,
    )


def plot(df: pd.DataFrame):
    """In-group (solid) and out-group (dashed) speakers, the condition in the color."""
    fig, ax = panel_figure(axes=BLOCK_PANEL_AXES, **MARGINS)
    for speaker in SPEAKERS:
        draw_trajectory(ax, df[df["speaker"] == speaker], CONDITION_ORDER, linestyle=LINESTYLES[speaker])
    ax.set_xticks(SPARSE_BLOCK_TICKS)
    ax.set_ylim(0, None)
    ax.set_ylabel("Listener response time\nfor referent (s)")
    return fig
