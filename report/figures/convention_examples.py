"""Two pairs of example games, each a refer_mixed game beside a social_mixed
game, following one tangram's descriptions through the game: for each original
group, where its descriptions sit on the first principal component of the
pair's descriptions of that tangram, the horizontal axis of the pair's
convention maps, block by block, with the conventions written out at the end
of each phase. Beside each pair, the two games' within- and
between-group similarity over all six tangrams, with every game of the two
conditions as thin lines behind them.

The panels are saved one PDF each, keyed pair<i>_similarity, pair<i>_refer_mixed
and pair<i>_social_mixed, each with the PANEL_AXES plotting area, for assembly
by hand and with no title, since the legend says which game and condition a
panel shows; the tangram sits in headroom above the data, where nothing
reaches it.

The examples are fixed in EXAMPLES. The first pair is the refer_mixed and
social_mixed games, on the same tangram set with all nine speakers to the end
of Phase 2 and clear Phase 1 conventions, whose within- and between-group
similarity curves over blocks 1 to 6 are closest (root-mean-square distance
0.038), so that the two games differ only after the manipulation; its tangram
is the one where the three conventions merge after mixing. The second pair is
the refer_mixed game in which one group gives up its convention for another
group's word (ring pop) rather than listing both, found by searching every
refer_mixed game for late descriptions nearer another group's Phase 1
convention than their own and reading the candidates, with the social_mixed
game matched to it the same way (distance 0.052). The search and the matching
are in reviews/atlas-build/embedding-maps/.
"""

from __future__ import annotations

import itertools
import textwrap

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.isotonic import isotonic_regression

from report.figures._common import (
    _label_color,
    dodge,
    numeric,
    panel_figure,
    read_model_input,
    stimulus_glyph,
    utterance_embeddings,
)
from report.plot_style import (
    BETWEEN_LINESTYLE,
    CONDITION_COLORS,
    GAME_LINE,
    GROUP_COLORS,
    MEAN_LINE,
    PHASE_BOUNDARY,
    SPARSE_BLOCK_TICKS,
    add_phase_boundary,
)

EXAMPLES = [
    {"refer_mixed": "01M3848KY063ZJ41GK9A2ZE3WN", "social_mixed": "01M46HTS6K2YWM89XND61RF41V", "target": "page7-255"},
    {"refer_mixed": "01M3DA03BJCFQ05X7RXFCSEVVD", "social_mixed": "01M44CVQC5MM5PTN0PSSVPE1PZ", "target": "page3-193"},
]
CONDITIONS = ["refer_mixed", "social_mixed"]
LABEL_BLOCKS = 2  # a convention is read from the last two blocks of a phase
LABEL_CHARS = 28  # a convention label wraps to lines this wide
LABEL_MARGIN = 1.25  # inches right of a strands panel's axes, room for labels LABEL_CHARS wide
LABEL_SIZE = 6  # points
HEADROOM = 0.32  # of the y range, above the data of a strands panel, where the tangram sits
DODGE_POINTS = {"A": -2.4, "B": 0.0, "C": 2.4}  # the three groups' dots side by side within a block


# ── Data ─────────────────────────────────────────────────────


def pair_projection(utterances: pd.DataFrame, embeddings: np.ndarray, example: dict) -> pd.DataFrame:
    """Every description of the example pair's tangram in its two games on the first two
    principal components of their sentence embeddings, fitted on both games together so that
    every example panel of the pair shares one projection: the strands plot the first
    component over blocks, the convention maps both. One row per description, indexed by its
    row in the embeddings, with columns x and y. Descriptive, like every measure a figure
    computes from the embeddings."""
    games = [example[c] for c in CONDITIONS]
    d = utterances[utterances["gameId"].isin(games) & (utterances["target"] == example["target"])]
    if d["gameId"].nunique() != len(games):
        raise ValueError(f"The example panels need both games of the pair describing {example['target']}.")
    coordinates = PCA(2, random_state=0).fit_transform(embeddings[d["row"].to_numpy()])
    return pd.DataFrame(coordinates, index=d["row"].to_numpy(), columns=["x", "y"])


def _convention(d: pd.DataFrame, embeddings: np.ndarray, group: str, block: int) -> tuple[str, float]:
    """The group's convention at `block`: of its descriptions in the LABEL_BLOCKS blocks
    up to it, the one most similar to the others (the shorter on a tie), and its mean coordinate."""
    e = d[(d["originalGroup"] == group) & d["gameBlock"].between(block - LABEL_BLOCKS + 1, block)]
    if e.empty:
        return "", np.nan
    X = embeddings[e["row"].to_numpy()]
    score = ((X @ X.T).sum(1) - 1) / max(len(e) - 1, 1)
    i = sorted(range(len(e)), key=lambda i: (-round(float(score[i]), 6), len(str(e["utterance"].iloc[i]))))[0]
    return " ".join(str(e["utterance"].iloc[i]).split()), float(e["value"].mean())


def data(dirs) -> pd.DataFrame:
    similarity = read_model_input(dirs, "block_similarity_means.csv")
    similarity = similarity[similarity["condition"].isin(CONDITIONS)].copy()
    similarity["gameBlock"] = numeric(similarity["gameBlock"]).astype(int)
    pair_of = {ex[c]: i + 1 for i, ex in enumerate(EXAMPLES) for c in CONDITIONS}
    similarity["pair"] = similarity["gameId"].map(pair_of).fillna(0).astype(int)
    similarity = similarity.rename(columns={"similarity": "value"})[
        ["gameId", "condition", "gameBlock", "pair_type", "value", "pair"]
    ].assign(part="similarity")

    u, embeddings = utterance_embeddings(dirs)
    strands, labels = [], []
    for i, ex in enumerate(EXAMPLES, start=1):
        coordinates = pair_projection(u, embeddings, ex)
        for cond in CONDITIONS:
            d = u[(u["gameId"] == ex[cond]) & (u["target"] == ex["target"])].copy()
            d["value"] = coordinates.loc[d["row"].to_numpy(), "x"].to_numpy()
            strands.append(d.assign(pair=i, part="strands")[
                ["pair", "condition", "gameId", "target", "originalGroup", "playerId", "gameBlock", "phaseNum",
                 "utterance", "value", "part"]
            ])
            for g in GROUP_COLORS:
                for phase, block in ((1, 6), (2, 12)):
                    text, value = _convention(d, embeddings, g, block)
                    labels.append({"pair": i, "condition": cond, "gameId": ex[cond], "target": ex["target"],
                                   "originalGroup": g, "phase": phase, "utterance": text, "value": value, "part": "labels"})
    return pd.concat([similarity, *strands, pd.DataFrame(labels)], ignore_index=True)


# ── Drawing ──────────────────────────────────────────────────


def _segments(x, y):
    """(x, y) split at the phase boundary, so that no line crosses the gap in the axis."""
    x, y = np.asarray(x), np.asarray(y)
    for m in (x < PHASE_BOUNDARY, x > PHASE_BOUNDARY):
        if m.any():
            yield x[m], y[m]


def _block_axis(ax):
    add_phase_boundary(ax)
    ax.set_xticks(SPARSE_BLOCK_TICKS)
    ax.set_xlabel("Block")


def _label_ends(ax, items, size, pad=5.0):
    """Labels `pad` points right of the end of lines: (x, y, text, color), each as near
    its line's end as it can sit without overlapping another or leaving the axes."""
    if not items:
        return
    to_points = 72 / ax.figure.dpi
    anchors = np.array([ax.transData.transform((x, y))[1] * to_points for x, y, _, _ in items])
    heights = np.array([1.25 * size * (text.count("\n") + 1) for _, _, text, _ in items])
    box = ax.get_window_extent()
    placed = _placements(anchors, heights, box.y0 * to_points, box.y1 * to_points)
    for (x, y, text, color), y0, y1 in zip(items, anchors, placed):
        ax.annotate(text, xy=(x, y), xycoords="data", xytext=(pad, y1 - y0), textcoords="offset points",
                    ha="left", va="center", fontsize=size, color=_label_color(color), annotation_clip=False,
                    linespacing=1.2)


def _placements(anchors, heights, low, high, gap=1.0):
    """Label centers between `low` and `high` that clear each other by `gap` and move least
    from their anchors in least squares: for each order of the labels, the nearest positions
    in that order (an isotonic regression on the anchors less the spacing, clipped to the
    bounds), and of those the order that moves them least, the order of the anchors on a tie,
    so that two lines ending level may have their labels set either way round."""
    best = None
    for order in itertools.permutations(range(len(anchors))):
        o = list(order)
        h = heights[o]
        offsets = np.concatenate([[0.0], np.cumsum((h[:-1] + h[1:]) / 2 + gap)])
        z = np.clip(isotonic_regression(anchors[o] - offsets), low + h[0] / 2, high - h[-1] / 2 - offsets[-1])
        placed = np.empty(len(o))
        placed[o] = z + offsets
        cost = ((placed - anchors) ** 2).sum()
        if best is None or cost < best[0] - 1e-9:
            best = (cost, placed)
    return best[1]


def _wrap(text: str) -> list[str]:
    return textwrap.wrap(text, width=LABEL_CHARS, break_long_words=False) or [""]


def _similarity_panel(table: pd.DataFrame, pair: int):
    fig, ax = panel_figure(left=0.62, right=0.12)
    styles = {"within": "-", "between": BETWEEN_LINESTYLE}
    background = table[table["pair"] != pair]
    for (_, cond, pair_type), g in background.groupby(["gameId", "condition", "pair_type"]):
        g = g.sort_values("gameBlock")
        for xs, ys in _segments(g["gameBlock"], g["value"]):
            ax.plot(xs, ys, color=CONDITION_COLORS[cond], linestyle=styles[pair_type], zorder=1, **GAME_LINE)
    for cond in CONDITIONS:
        for pair_type, style in styles.items():
            g = table[(table["pair"] == pair) & (table["condition"] == cond) & (table["pair_type"] == pair_type)]
            g = g.sort_values("gameBlock")
            for xs, ys in _segments(g["gameBlock"], g["value"]):
                ax.plot(xs, ys, color=CONDITION_COLORS[cond], linestyle=style, zorder=3, **MEAN_LINE)
    _block_axis(ax)
    ax.set_ylim(0.2, 0.9)
    ax.set_ylabel("Similarity between\ndescriptions")
    return fig


def _strands_panel(strands: pd.DataFrame, labels: pd.DataFrame, condition: str):
    fig, ax = panel_figure(left=0.34, right=LABEL_MARGIN)
    d = strands.sort_values("gameBlock")
    ends = {}  # each group's line end, where its label is set
    for g, color in GROUP_COLORS.items():
        e = d[d["originalGroup"] == g]
        shift = dodge(ax, DODGE_POINTS[g])
        means = e.groupby("gameBlock")["value"].mean()
        ends[g] = means.iloc[-1]
        for xs, ys in _segments(means.index.to_numpy(), means.to_numpy()):
            ax.plot(xs, ys, color=color, linewidth=MEAN_LINE["linewidth"] * 0.85, zorder=2, solid_capstyle="round",
                    transform=shift)
        for phase, face in ((1, color), (2, "white")):
            p = e[e["phaseNum"] == phase]
            ax.scatter(p["gameBlock"], p["value"], s=9, facecolors=face, edgecolors=color, linewidths=0.7, zorder=3,
                       transform=shift)
    _block_axis(ax)
    lo, hi = d["value"].min(), d["value"].max()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + HEADROOM * (hi - lo))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("First principal\ncomponent")
    stimulus_glyph(ax, d["target"].iloc[0])
    items = []
    for g, color in GROUP_COLORS.items():
        first = labels[(labels["originalGroup"] == g) & (labels["phase"] == 1)]
        last = labels[(labels["originalGroup"] == g) & (labels["phase"] == 2)]
        if last.empty or not last["utterance"].iloc[0]:
            continue
        a, b = first["utterance"].iloc[0], last["utterance"].iloc[0]
        tail = _wrap(b)  # written as before -> after even when unchanged, so every group reads the same way
        lines = _wrap(a) + ["$\\rightarrow$ " + tail[0]] + ["    " + line for line in tail[1:]]
        items.append((12, ends[g], "\n".join(lines), color))
    _label_ends(ax, items, LABEL_SIZE, pad=5 + max(DODGE_POINTS.values()))  # clear of the dodged end points
    return fig


def plot(df: pd.DataFrame) -> dict:
    similarity = df[df["part"] == "similarity"]
    panels = {}
    for i in range(1, len(EXAMPLES) + 1):
        panels[f"pair{i}_similarity"] = _similarity_panel(similarity, i)
        for cond in CONDITIONS:
            strands = df[(df["part"] == "strands") & (df["pair"] == i) & (df["condition"] == cond)]
            labels = df[(df["part"] == "labels") & (df["pair"] == i) & (df["condition"] == cond)]
            panels[f"pair{i}_{cond}"] = _strands_panel(strands, labels, cond)
    return panels
