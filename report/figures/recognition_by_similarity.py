"""How listeners decided whether a speaker was from their own group: the
share answering "same group" against how similar the description was to the
listener's own group's Phase 1 convention, for speakers who were in the
listener's group (solid, circles) and speakers who were not (dashed,
triangles), in the two social conditions pooled. Both curves rise steeply with
similarity, from near zero to over nine in ten: listeners judged by whether
the description sounded like their own group's, and the gap between the
curves at middling similarity is whatever else they used.

A group's convention for a tangram is the mean embedding of its members'
final Phase 1 descriptions (phase1_conventions()), and a judged description
is scored by its cosine similarity to the listener's group's convention, so
the x variable is the listener's, not the speaker's; the outcomes-by-alignment
panel of the convention-alignment figure conditions on the speaker's. The
judgments are binned by equal widths of similarity (BINS), a bin is drawn
where it holds at least MIN_OBS judgments, and each point is the share in the
bin with its bootstrap interval over games, set at the bin's mean similarity.
The measure is descriptive and not preregistered. The panel is placed by hand
in the article in the square plotting area of the Phase 2 panels, carries no
letter or title, and leaves the game counts to the caption."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.bootstrap import bootstrap_mean
from report.figures._common import (
    is_true,
    numeric,
    panel_figure,
    phase1_conventions,
    read_table,
    reference_line,
    social_guesses,
    utterance_embeddings,
)
from report.plot_style import (
    BETWEEN_LINESTYLE,
    BETWEEN_MARKER,
    INK,
    INTERVAL_LINE,
    MARKER_AREA,
    MEAN_LINE,
    PHASE_POINT,
    SQUARE_PANEL_AXES,
)

SOCIAL_CONDITIONS = ["social_mixed", "social_first"]
SPEAKERS = ["in_group", "out_group"]
LINESTYLES = {"in_group": "-", "out_group": BETWEEN_LINESTYLE}
MARKERS = {"in_group": "o", "out_group": BETWEEN_MARKER}  # both filled: a hollow mark would read as Phase 2
BINS = np.linspace(0.1, 1.0, 10)  # nine equal widths of similarity
MIN_OBS = 30  # judgments a bin needs to be drawn
CHANCE = 0.5  # a listener answering at random says "same group" on half the trials
MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.55)  # the y label, nothing, nothing, the two-line x label


def data(dirs) -> pd.DataFrame:
    utterances, embeddings = utterance_embeddings(dirs)
    conventions = phase1_conventions(utterances, embeddings)
    guesses = social_guesses(dirs)
    guesses = guesses[
        is_true(guesses["onTime"]) & guesses["socialGuess"].notna() & guesses["condition"].isin(SOCIAL_CONDITIONS)
    ].copy()
    guesses["same"] = (guesses["socialGuess"] == "same_group").astype(float)
    guesses["speaker"] = np.where(is_true(guesses["speakerWasSameGroup"]), "in_group", "out_group")
    rounds = read_table(dirs, "trials.csv")[["gameId", "roundId", "trialNum"]].drop_duplicates()
    rounds["trialNum"] = numeric(rounds["trialNum"]).astype(int)
    described = utterances[["gameId", "trialNum", "playerId", "row"]].rename(columns={"playerId": "speakerId"})
    judged = guesses.merge(rounds, on=["gameId", "roundId"]).merge(described, on=["gameId", "trialNum", "speakerId"])
    similarity = []
    for j in judged.itertuples():  # the listener's own group's convention
        convention = conventions.get((j.gameId, j.originalGroup, j.target))
        similarity.append(np.nan if convention is None else float(embeddings[j.row] @ convention))
    judged["similarity"] = similarity
    judged["bin"] = pd.cut(judged["similarity"], BINS, labels=False)
    rows = []
    for (speaker, b), part in judged.dropna(subset=["bin"]).groupby(["speaker", "bin"]):
        if len(part) < MIN_OBS:
            continue
        est = bootstrap_mean(part, "same")
        rows.append({
            "speaker": speaker, "bin": int(b), "x": part["similarity"].mean(), "value": est.mean, "lower": est.lower,
            "upper": est.upper, "n_games": est.n_games, "n_obs": est.n_obs, "level": "condition",
        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame):
    fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **MARGINS)
    for speaker in SPEAKERS:
        s = df[df["speaker"] == speaker].sort_values("x")
        ax.plot(s["x"], s["value"], color=INK, linestyle=LINESTYLES[speaker], zorder=2, **MEAN_LINE)
        ax.vlines(s["x"], s["lower"], s["upper"], color=INK, zorder=3, **INTERVAL_LINE)
        marker = MARKERS[speaker]
        ax.scatter(s["x"], s["value"], color=INK, marker=marker, zorder=4, **{**PHASE_POINT, "s": PHASE_POINT["s"] * MARKER_AREA[marker]})
    reference_line(ax, CHANCE, "chance", side="left")  # the out-group curve crosses the line at the right
    ax.set_xlim(BINS[0], BINS[-1])
    ax.set_xticks([0.2, 0.6, 1.0])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Similarity to the listener's\nown group's convention")
    ax.set_ylabel("Share answering “same group”")
    return fig
