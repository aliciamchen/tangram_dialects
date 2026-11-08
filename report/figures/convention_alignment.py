"""Where Phase 2 descriptions stand relative to the conventions the groups
formed in Phase 1, three panels from one measure. A group's convention for a
tangram is the mean embedding of its members' final Phase 1 descriptions of
it, and every Phase 2 description is scored by its cosine similarity to its
own group's convention and to each other group's.

similarity: the similarity to the speaker's own group's convention (solid)
and to the other groups' conventions (the mean over them, dashed), block by
block through Phase 2 with the four conditions overlaid, each game a thin
line behind its condition's mean and bootstrap interval. A falling solid line is drift away from
the group's own convention; the dashed line falling is further divergence
from the other groups, rising is adoption of their descriptions.

borrowed: the share of descriptions closer to another group's convention
than to the speaker's own, by block, drawn the same way. In refer_separated, whose groups never
meet, the share is the floor the measure has when nothing is borrowed.

outcomes: in the two social conditions pooled, what happened to a
description as a function of its alignment with its own group's convention
(the similarity to the own convention minus the similarity to the closest
other group's), the descriptions cut into thirds by it, low, middle and
high: the share of in-group listeners who selected the target (solid), of
out-group listeners who did (dashed, the in-group/out-group convention), and
of listeners who judged the speaker's group correctly, each a point with its
bootstrap interval over games, the three series dodged by points within a
third. In-group comprehension and recognition rise with alignment while
out-group comprehension falls: the tradeoff a group-specific description
makes, as one scissors shape.

A game's per-block value rests on about fifty descriptions, so its line is
noisy; the y ranges hold every game's line. The measure is descriptive; the preregistration does not name it. It
is computed here from the embeddings the measures step wrote
(data/<dataset>/measures/embeddings.npy, one row per utterance of the table
the models read); a description whose own group left no final Phase 1
description of the tangram, or whose game has only one other group's
convention to compare with, is left out. The panels are placed by hand in
the article in the square plotting area of the Phase 2 panels, carry no
letter or title, and leave the game counts to the caption."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.bootstrap import bootstrap_mean
from report.figures._common import (
    dodge,
    draw_trajectory,
    is_true,
    label_lines,
    listener_trials,
    panel_figure,
    phase1_conventions,
    social_guesses,
    trajectory_table,
    utterance_embeddings,
)
from report.plot_style import (
    BETWEEN_LINESTYLE,
    CONDITION_ORDER,
    GROUP_ORDER,
    INK,
    INTERVAL_LINE,
    MEAN_LINE,
    PHASE2_RANGE,
    PHASE2_TICKS,
    PHASE_POINT,
    SQUARE_PANEL_AXES,
)

SOCIAL_CONDITIONS = ["social_mixed", "social_first"]
ANCHORS = ["own", "other"]
ANCHOR_LINESTYLES = {"own": "-", "other": BETWEEN_LINESTYLE}
OUTCOMES = ["in_group", "recognized", "out_group"]
OUTCOME_WORDS = {
    "in_group": "understood by\nin-group listener",
    "recognized": "group judged\ncorrectly",
    "out_group": "understood by\nout-group listener",
}
OUTCOME_COLORS = {o: INK for o in OUTCOMES}
OUTCOME_LINESTYLES = {"in_group": "-", "recognized": "-", "out_group": BETWEEN_LINESTYLE}
OUTCOME_DODGE = {"in_group": -1.6, "recognized": 0.0, "out_group": 1.6}  # points, the three series side by side within a third
THIRDS = ["low", "middle", "high"]
MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.42)  # the y label, nothing, nothing, the x axis
OUTCOME_MARGINS = dict(left=0.62, right=1.05, top=0.12, bottom=0.42)  # the two-line names need more room
KEYS = ["gameId", "speakerId", "target", "gameBlock", "condition"]


def _alignment(dirs) -> pd.DataFrame:
    """One row per Phase 2 description: its similarity to its own group's convention and to the other groups'."""
    utterances, embeddings = utterance_embeddings(dirs)
    conventions = phase1_conventions(utterances, embeddings)
    rows = []
    for (game, target), part in utterances[utterances["phaseNum"] == 2].groupby(["gameId", "target"]):
        groups = [g for g in GROUP_ORDER if (game, g, target) in conventions]
        if len(groups) < 2:
            continue
        sims = embeddings[part["row"].to_numpy()] @ np.stack([conventions[(game, g, target)] for g in groups]).T
        for sim, utterance in zip(sims, part.itertuples()):
            if utterance.originalGroup not in groups:
                continue
            own = groups.index(utterance.originalGroup)
            others = np.delete(sim, own)
            rows.append({
                "gameId": game, "speakerId": utterance.playerId, "target": target, "gameBlock": utterance.gameBlock,
                "condition": utterance.condition, "own": sim[own], "other": others.mean(),
                "borrowed": float(others.max() > sim[own]), "alignment": sim[own] - others.max(),
            })
    return pd.DataFrame(rows)


def _outcomes(dirs, alignment: pd.DataFrame) -> pd.DataFrame:
    """Each outcome's share by third of alignment, with its bootstrap interval over games;
    `x` is the third's position, 0 to 2."""
    trials = listener_trials(dirs, phase=2)
    trials["in_group"] = is_true(trials["inGroupSpeaker"])
    guesses = social_guesses(dirs)
    social = alignment[alignment["condition"].isin(SOCIAL_CONDITIONS)]
    edges = social["alignment"].quantile(np.linspace(0, 1, len(THIRDS) + 1)).to_numpy().copy()
    edges[0] -= 1e-9
    binned = social.assign(bin=pd.cut(social["alignment"], edges, labels=False))
    series = {
        "in_group": trials[trials["in_group"] & trials["condition"].isin(SOCIAL_CONDITIONS)],
        "recognized": guesses[guesses["condition"].isin(SOCIAL_CONDITIONS)],
        "out_group": trials[~trials["in_group"] & trials["condition"].isin(SOCIAL_CONDITIONS)],
    }
    rows = []
    for name, part in series.items():
        joined = part.merge(binned[KEYS + ["bin"]], on=KEYS)
        for b, cell in joined.groupby("bin"):
            est = bootstrap_mean(cell, "correct")
            rows.append({"series": name, "bin": int(b), "x": int(b), "value": est.mean, "lower": est.lower,
                         "upper": est.upper, "n_games": est.n_games, "n_obs": est.n_obs, "level": "condition"})
    return pd.DataFrame(rows).assign(panel="outcomes")


def data(dirs) -> pd.DataFrame:
    alignment = _alignment(dirs)
    return pd.concat(
        [
            *(trajectory_table(alignment, anchor, "similarity").assign(anchor=anchor) for anchor in ANCHORS),
            trajectory_table(alignment, "borrowed", "borrowed"),
            _outcomes(dirs, alignment),
        ],
        ignore_index=True,
    )


def plot(df: pd.DataFrame) -> dict:
    return {
        "similarity": _similarity(df[df["panel"] == "similarity"]),
        "borrowed": _borrowed(df[df["panel"] == "borrowed"]),
        "outcomes": _plot_outcomes(df[df["panel"] == "outcomes"]),
    }


def _similarity(table: pd.DataFrame):
    """Own convention solid and the other groups' dashed, the condition in the color."""
    fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **MARGINS)
    for anchor in ANCHORS:
        draw_trajectory(ax, table[table["anchor"] == anchor], CONDITION_ORDER, blocks=PHASE2_RANGE,
                        linestyle=ANCHOR_LINESTYLES[anchor])
    ax.set_xticks(PHASE2_TICKS)
    ax.set_ylim(0.2, 1.0)
    ax.set_ylabel("Similarity to the\nPhase 1 conventions")
    return fig


def _borrowed(table: pd.DataFrame):
    fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **MARGINS)
    draw_trajectory(ax, table, CONDITION_ORDER, blocks=PHASE2_RANGE)
    ax.set_xticks(PHASE2_TICKS)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Share of descriptions\ncloser to another\ngroup's convention")
    return fig


def _plot_outcomes(table: pd.DataFrame):
    """The outcome shares over the thirds of alignment, each series named beside its right end.

    Two of the lines are alike in color and style, so their names stay on the
    panel, the one panel that names its lines."""
    fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **OUTCOME_MARGINS)
    for name in OUTCOMES:
        s = table[table["series"] == name].sort_values("x")
        shift = dodge(ax, OUTCOME_DODGE[name])
        ax.plot(s["x"], s["value"], color=OUTCOME_COLORS[name], linestyle=OUTCOME_LINESTYLES[name], zorder=2,
                transform=shift, **MEAN_LINE)
        ax.vlines(s["x"], s["lower"], s["upper"], color=OUTCOME_COLORS[name], zorder=3, transform=shift, **INTERVAL_LINE)
        ax.scatter(s["x"], s["value"], color=OUTCOME_COLORS[name], zorder=4, transform=shift, **PHASE_POINT)
    ax.set_xlim(-0.4, len(THIRDS) - 0.6)
    ax.set_xticks(range(len(THIRDS)))
    ax.set_xticklabels(THIRDS)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    label_lines(ax, table, "series", OUTCOMES, OUTCOME_WORDS, OUTCOME_COLORS, x="x", y="value")
    ax.set_xlabel("Alignment with own group's convention")
    ax.set_ylabel("Share of listener trials")
    return fig
