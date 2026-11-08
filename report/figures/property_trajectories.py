"""The three description properties block by block across the whole game with
the four conditions overlaid, one panel each: concreteness (the share of a
description's content words that match the concrete stems), lexical
uniqueness (the share that appear in no other original group's descriptions
of the tangram, in the same phase, across games) and word frequency (the mean
Zipf frequency of the content words). Each panel draws every game as a thin
line behind its condition's mean and bootstrap interval over games; a game's
mean at a block rests on about eighteen descriptions, so the game lines are
noisy, and the y range holds all of them. Uniqueness is measured against the vocabulary of its own
phase, so its level steps at the phase boundary with the pool as well as with
the speakers. The exploratory models of the same properties at the end of
each phase are in stats/secondary/utterance_properties.R.

The panels are placed by hand beside other twelve-block panels in the
article, so they share the block-panel plotting area and carry no letter or
title; the game count, 30 per condition, is left to the caption."""

from __future__ import annotations

import pandas as pd

from report.figures._common import (
    draw_trajectory,
    in_conditions,
    numeric,
    panel_figure,
    read_measure,
    trajectory_table,
)
from report.plot_style import BLOCK_PANEL_AXES, CONDITION_ORDER, SPARSE_BLOCK_TICKS

PANELS = {
    "concreteness": "Concreteness",  # what each share counts is in the caption
    "uniqueness": "Lexical uniqueness",
    "word_frequency": "Word frequency\n(mean Zipf)",
}
MEASURES = {"concreteness": "concreteness", "uniqueness": "uniqueness", "word_frequency": "mean_zipf_freq"}
# The shares from zero; Zipf frequency has no natural zero; each range holds every game's line
LIMITS = {"concreteness": (0, 0.6), "uniqueness": (0, 0.4), "word_frequency": (3.2, 5.2)}
MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.42)  # the y label, nothing, nothing, the block axis
DESCRIPTION_KEYS = ["gameId", "playerId", "target", "gameBlock"]


def data(dirs) -> pd.DataFrame:
    properties = read_measure(dirs, "description_properties.csv")
    uniqueness = read_measure(dirs, "lexical_uniqueness.csv")
    for table in (properties, uniqueness):
        table["gameBlock"] = numeric(table["gameBlock"]).astype(int)
    descriptions = in_conditions(properties.merge(uniqueness[DESCRIPTION_KEYS + ["uniqueness"]], on=DESCRIPTION_KEYS))
    parts = []
    for panel, measure in MEASURES.items():
        descriptions[measure] = numeric(descriptions[measure])
        parts.append(trajectory_table(descriptions[descriptions[measure].notna()], measure, panel))
    return pd.concat(parts, ignore_index=True)


def plot(df: pd.DataFrame) -> dict:
    panels = {}
    for panel, ylabel in PANELS.items():
        table = df[df["panel"] == panel]
        fig, ax = panel_figure(axes=BLOCK_PANEL_AXES, **MARGINS)
        draw_trajectory(ax, table, CONDITION_ORDER)
        ax.set_xticks(SPARSE_BLOCK_TICKS)
        ax.set_ylim(*LIMITS[panel])
        ax.set_ylabel(ylabel)
        panels[panel] = fig
    return panels
