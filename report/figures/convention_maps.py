"""One tangram's descriptions in each pair of example games as a
two-dimensional map: the first two principal components of the sentence
embeddings of every description of the pair's tangram in its two games,
fitted per pair (`pair_projection` in convention_examples) so that the pair's two panels
share their axes, one point per description in
its original group's color, filled in Phase 1 and hollow in Phase 2, and an
arrow from each group's Phase 1 centroid to its Phase 2 centroid. The arrows
show where each group's descriptions moved once the groups were mixed: under
referential goals toward one another, under the recognition incentive barely
at all. The strands of the convention-examples panels plot the first of these
components over blocks; this map keeps two.

Two panels per pair, keyed pair<i>_<condition> as in the convention-examples
panels (the rooster tangram in the first pair, the ring-pop one in the second),
each the square plotting area with the
tangram in headroom above the data, no ticks (the components have no units a
reader needs), no letter and no title; the legend says which game a panel
shows. The measure is descriptive and computed from the embeddings the
measures step wrote."""

from __future__ import annotations

import pandas as pd
from matplotlib.patches import FancyArrowPatch

from report.figures._common import panel_figure, stimulus_glyph, utterance_embeddings
from report.figures.convention_examples import EXAMPLES, HEADROOM, pair_projection
from report.plot_style import GROUP_COLORS, PHASE_POINT, SQUARE_PANEL_AXES

CONDITIONS = ["refer_mixed", "social_mixed"]
PADDING = 0.06  # of the data range, around the points
ARROW = dict(arrowstyle="-|>", mutation_scale=8, linewidth=1.3, shrinkA=0, shrinkB=0)
MARGINS = dict(left=0.3, right=0.12, top=0.12, bottom=0.3)  # the axis names only


def data(dirs) -> pd.DataFrame:
    utterances, embeddings = utterance_embeddings(dirs)
    parts = []
    for i, pair in enumerate(EXAMPLES, start=1):
        coordinates = pair_projection(utterances, embeddings, pair)
        d = utterances.loc[utterances["row"].isin(coordinates.index)]
        points = pd.DataFrame({
            "part": "point", "pair": i, "condition": d["condition"].to_numpy(),
            "originalGroup": d["originalGroup"].to_numpy(), "phaseNum": d["phaseNum"].to_numpy(),
            "gameBlock": d["gameBlock"].to_numpy(), "x": coordinates.loc[d["row"], "x"].to_numpy(),
            "y": coordinates.loc[d["row"], "y"].to_numpy(), "target": pair["target"],
        })
        centroids = (
            points.groupby(["pair", "condition", "originalGroup", "phaseNum"], as_index=False)[["x", "y"]].mean()
            .assign(part="centroid", target=pair["target"])
        )
        parts += [points, centroids]
    return pd.concat(parts, ignore_index=True)


def plot(df: pd.DataFrame) -> dict:
    panels = {}
    for pair in sorted(df["pair"].unique()):
        panels |= _pair_panels(df[df["pair"] == pair], int(pair))
    return panels


def _pair_panels(df: pd.DataFrame, pair: int) -> dict:
    points = df[df["part"] == "point"]
    centroids = df[df["part"] == "centroid"]
    x0, x1 = points["x"].min(), points["x"].max()
    y0, y1 = points["y"].min(), points["y"].max()
    xlim = (x0 - PADDING * (x1 - x0), x1 + PADDING * (x1 - x0))
    ylim = (y0 - PADDING * (y1 - y0), y1 + HEADROOM * (y1 - y0))
    panels = {}
    for condition in CONDITIONS:
        fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **MARGINS)
        p = points[points["condition"] == condition]
        for group, color in GROUP_COLORS.items():
            e = p[p["originalGroup"] == group]
            for phase, face in ((1, color), (2, "white")):
                q = e[e["phaseNum"] == phase]
                ax.scatter(q["x"], q["y"], facecolors=face, edgecolors=color, zorder=2, **PHASE_POINT)
            c = centroids[(centroids["condition"] == condition) & (centroids["originalGroup"] == group)].set_index("phaseNum")
            if {1, 2} <= set(c.index):
                ax.add_patch(FancyArrowPatch((c.loc[1, "x"], c.loc[1, "y"]), (c.loc[2, "x"], c.loc[2, "y"]), color=color,
                                             zorder=3, **ARROW))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("First principal component")
        ax.set_ylabel("Second principal\ncomponent")
        stimulus_glyph(ax, points["target"].iloc[0])
        panels[f"pair{pair}_{condition}"] = fig
    return panels
