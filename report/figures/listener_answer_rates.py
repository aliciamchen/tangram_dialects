"""Each listener's two answer rates in Phase 2 of the two social conditions:
the share of in-group speakers the listener judged in-group (the hit rate)
and the share of out-group speakers the listener judged in-group (the
false-alarm rate), each a column of listeners, one pair of columns per
condition, under the mean as a black line and its bootstrap interval over
games as an outlined box, with chance, a listener answering at random, at one
half. The gap between a pair's two columns is how well listeners told the
groups apart; where the pair sits on the scale is which answer they favored.
The two rates sit about equally far from their corners, so listeners' accuracy
came from telling the groups apart, not from answering "same group" to
everyone, and listeners differ in how well they tell rather than in which
answer they lean to.

The rates are over the judgments made in time, as the by-block panel of
group_recognition.py counts them, for every listener who judged at least one
speaker of each kind. A listener is a point lighter than a game point
(RESPONDENT_POINT), jittered across only, the jitter seeded so that the PDF
reproduces.

Unlike the article's panels, this is a complete figure for the SI: each pair
of columns sits under its condition's name, and the columns are ticked by the
speaker's group, in-group or out-group."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.figures._common import (
    condition_summary,
    draw_column,
    is_true,
    panel_figure,
    reference_line,
    social_guesses,
)
from report.plot_style import COLUMNS_PANEL_AXES, CONDITION_COLORS, CONDITION_LABELS, INK, RESPONDENT_POINT, SMALL_SIZE

SOCIAL_CONDITIONS = ["social_mixed", "social_first"]
SPEAKERS = ["in_group", "out_group"]
SPEAKER_LABELS = {"in_group": "In-\ngroup", "out_group": "Out-\ngroup"}  # the tick labels, two lines so that a pair's columns can sit close
CHANCE = 0.5  # a listener answering at random says "same group" on half the trials
OFFSET = 0.22  # a pair's columns, either side of the condition's position
JITTER = (0.09, 0.0)  # of a column's width; the rates are not jittered up and down
SEED = 20261108
MEAN_HALF = 0.14  # half the width of the mean's line and box
NAME_HEIGHT = 1.04  # the condition names, above the columns, in axes fraction
LIMITS = (-0.03, 1.03)  # a little past the rates' range, so that the many listeners at 0 and 1 are drawn whole
MARGINS = dict(left=0.62, right=0.12, top=0.2, bottom=0.5)  # the y label, nothing, the condition names, the x axis


def data(dirs) -> pd.DataFrame:
    guesses = social_guesses(dirs)
    judged = guesses[is_true(guesses["onTime"]) & guesses["socialGuess"].notna() & guesses["condition"].isin(SOCIAL_CONDITIONS)].copy()
    judged["speaker"] = np.where(is_true(judged["speakerWasSameGroup"]), "in_group", "out_group")
    judged["same"] = (judged["socialGuess"] == "same_group").astype(float)
    listeners = (
        judged.groupby(["condition", "gameId", "playerId", "speaker"])["same"].agg(["mean", "size"]).reset_index()
        .rename(columns={"mean": "value", "size": "n_obs"})
    )
    both = listeners.groupby(["gameId", "playerId"])["speaker"].transform("nunique") == len(SPEAKERS)
    listeners = listeners[both].drop(columns="playerId").assign(level="listener")
    summary = condition_summary(listeners, "value", SOCIAL_CONDITIONS, keys=["speaker"])
    return pd.concat([listeners, summary], ignore_index=True)


def plot(df: pd.DataFrame):
    fig, ax = panel_figure(axes=COLUMNS_PANEL_AXES, **MARGINS)
    rng = np.random.default_rng(SEED)
    ticks = []
    for i, condition in enumerate(SOCIAL_CONDITIONS):
        for speaker, sign in zip(SPEAKERS, (-1, 1)):
            x = i + sign * OFFSET
            rows = df[(df["level"] == "listener") & (df["condition"] == condition) & (df["speaker"] == speaker)]
            mean = df[(df["level"] == "condition") & (df["condition"] == condition) & (df["speaker"] == speaker)]
            draw_column(ax, x, rows["value"], None if mean.empty else mean.iloc[0], CONDITION_COLORS[condition],
                        RESPONDENT_POINT, JITTER, rng, MEAN_HALF)
            ticks.append((x, SPEAKER_LABELS[speaker]))
        ax.text(i, NAME_HEIGHT, CONDITION_LABELS[condition], transform=ax.get_xaxis_transform(), fontsize=SMALL_SIZE, color=INK,
                ha="center", va="bottom", clip_on=False)
    reference_line(ax, CHANCE, "chance")
    ax.set_xticks([x for x, _ in ticks])
    ax.set_xticklabels([label for _, label in ticks])
    ax.set_xlim(-0.6, len(SOCIAL_CONDITIONS) - 0.4)
    ax.set_ylim(*LIMITS)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Speaker's group")
    ax.set_ylabel("Share judged in-group")
    return fig
