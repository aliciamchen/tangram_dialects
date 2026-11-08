"""Each game's group specificity against what shuffling the group labels would
give: one row per game, ordered by the specificity at the end of Phase 2 with
the condition in the color. A game's specificity is the preregistered per-game
estimate (stats/primary/group_specificity.R: the same-group coefficient of a
mixed model of pair similarity with tangram and speaker-pair intercepts), the
filled point at the end of Phase 1 and the hollow point at the end of Phase 2,
joined by a thin line. The gray band is the central 95% of the Phase 1
coefficient under the thousand random relabelings of the game's speakers into
groups that the Phase 1 check runs with the same model
(stats/checks/phase1_group_specificity.R, which writes the range). Nearly
every filled point is clear of its band, so the Phase 1 conventions are real,
and ordered this way the conditions sort themselves: separated games at the
top with lines pointing right, refer_mixed games at the bottom with lines
pointing left toward the band, the social conditions between with short
lines. Games without a Phase 2 estimate sit at the bottom.

One tall panel on the ROWS_PANEL_AXES plotting area, its type a step larger
than the other panels' so that it matches theirs once reduced to sit beside
them, the games unlabeled, with no letter and no title; the conditions and which mark is which phase are left
to the legend."""

from __future__ import annotations

import pandas as pd

from report.figures._common import estimates, panel_figure, specificity_endpoints
from report.plot_style import (
    CONDITION_COLORS,
    CONDITION_ORDER,
    NULL_BAND,
    REFERENCE_LINE,
    LABEL_SIZE,
    ROWS_PANEL_AXES,
    ROW_CONNECTOR,
    ROW_POINT,
    TITLE_SIZE,
)

LIMITS = (-0.25, 0.8)
MARGINS = dict(left=0.3, right=0.12, top=0.1, bottom=0.42)  # the y label, nothing, nothing, the x label


def data(dirs) -> pd.DataFrame:
    null = estimates(dirs, "checks/phase1_group_specificity")[["gameId", "perm_lower", "perm_upper"]]
    table = specificity_endpoints(dirs).merge(null.rename(columns={"perm_lower": "lower", "perm_upper": "upper"}),
                                              on="gameId", how="left")
    columns = ["gameId", "condition", "phase1", "lower", "upper", "phase2"]
    return table[columns].dropna(subset=["phase1"]).reset_index(drop=True)


def plot(df: pd.DataFrame):
    fig, ax = panel_figure(axes=ROWS_PANEL_AXES, **MARGINS)
    d = df.sort_values("phase2", ascending=False, na_position="last").reset_index(drop=True)
    d["y"] = len(d) - d.index  # the highest Phase 2 value in the top row
    ax.axvline(0, **REFERENCE_LINE)
    ax.hlines(d["y"], d["lower"], d["upper"], zorder=1, **NULL_BAND)
    for condition in CONDITION_ORDER:
        color = CONDITION_COLORS[condition]
        g = d[d["condition"] == condition]
        with_end = g.dropna(subset=["phase2"])
        ax.hlines(with_end["y"], with_end["phase1"], with_end["phase2"], color=color, zorder=2, **ROW_CONNECTOR)
        ax.scatter(with_end["phase2"], with_end["y"], facecolors="white", edgecolors=color, zorder=4, **ROW_POINT)
        ax.scatter(g["phase1"], g["y"], facecolors=color, edgecolors=color, zorder=5, **ROW_POINT)
    ax.set_xlim(*LIMITS)
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_ylim(0, len(d) + 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    # The type one step larger than the other panels': this tall panel is reduced
    # further than they are to sit beside them, and then matches their type
    ax.tick_params(labelsize=LABEL_SIZE)
    ax.set_xlabel("Group specificity", fontsize=TITLE_SIZE)
    ax.set_ylabel("Games (ordered by the end of Phase 2)", fontsize=TITLE_SIZE)
    return fig
