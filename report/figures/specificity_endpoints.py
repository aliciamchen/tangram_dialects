"""Group specificity at the end of Phase 2 against the same game's group
specificity at the end of Phase 1, one point per game with the condition
means and their bootstrap intervals, and the line of no change. This is the
comparison the preregistered H1 and H2 models make (stats/primary/h1_h2.R:
Phase 2 specificity adjusted for Phase 1): separated games lie above the
line, mixed games under referential goals far below it, and the social
conditions near it. The per-game estimates are those of
stats/primary/group_specificity.R; a game appears when both of its endpoints
are estimable.

Each condition has its own point shape as well as its color, since the four
overlap where the line crosses the cloud. Which side of the line is more
group-specific is left to the legend. The panel is placed by hand in the
article in the square plotting area of the Phase 2 panels, carries no letter or title, and leaves the game counts to the
caption."""

from __future__ import annotations

import pandas as pd

from report.figures._common import (
    SPECIFICITY_LIMITS,
    condition_summary,
    draw_scatter,
    panel_figure,
    scatter_table,
    specificity_endpoints,
)
from report.plot_style import NOTE_SIZE, REFERENCE_GRAY, REFERENCE_LINE, SQUARE_PANEL_AXES

MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.42)  # the y label, nothing, nothing, the x axis


def data(dirs) -> pd.DataFrame:
    games = specificity_endpoints(dirs).dropna(subset=["phase1", "phase2"]).rename(columns={"phase1": "x", "phase2": "y"})
    return scatter_table(games, condition_summary(games, "x"), condition_summary(games, "y"))


def plot(df: pd.DataFrame):
    fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **MARGINS)
    lo, hi = SPECIFICITY_LIMITS
    ax.plot([lo, hi], [lo, hi], zorder=0, **REFERENCE_LINE)
    ax.annotate("no change", xy=(hi, hi), xytext=(-1, 1), textcoords="offset points", ha="right", va="bottom",
                rotation=45, rotation_mode="anchor", fontsize=NOTE_SIZE, color=REFERENCE_GRAY)
    draw_scatter(ax, df, shapes=True)  # the four conditions overlap near the line, so each has a shape too
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks([0, 0.4, 0.8])
    ax.set_yticks([0, 0.4, 0.8])
    ax.set_aspect("equal")
    ax.set_xlabel("Group specificity,\nend of Phase 1")
    ax.set_ylabel("Group specificity,\nend of Phase 2")
    return fig
