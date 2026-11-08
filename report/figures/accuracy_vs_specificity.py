"""The price of keeping a group's language in Phase 2: each game's referential
accuracy over the phase (correct selections over listener response
opportunities, as the pipeline coded them) against its group specificity at
the end of the phase (stats/primary/group_specificity.R), one point per game
with the condition means and their bootstrap intervals. Separated groups
keep both; contact under referential goals trades specificity for accuracy
with a mixed audience; the identity incentive keeps specificity at a cost in
accuracy. Within a condition the two are barely related, so the picture is
one of condition means.

The panel is placed by hand in the article in the square plotting area of the
Phase 2 panels, carries no letter or title, and leaves the game counts to the
caption."""

from __future__ import annotations

import pandas as pd

from report.figures._common import (
    SPECIFICITY_LIMITS,
    condition_summary,
    draw_scatter,
    game_means,
    listener_trials,
    panel_figure,
    scatter_table,
    specificity_endpoints,
)
from report.plot_style import SQUARE_PANEL_AXES

MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.42)  # the y label, nothing, nothing, the x axis


def data(dirs) -> pd.DataFrame:
    trials = listener_trials(dirs, phase=2)
    ends = specificity_endpoints(dirs)
    games = (
        game_means(trials, "correct").rename(columns={"value": "x"})
        .merge(ends[["gameId", "phase2"]].rename(columns={"phase2": "y"}), on="gameId")
        .dropna(subset=["y"])
    )
    trials = trials[trials["gameId"].isin(games["gameId"])]
    return scatter_table(games, condition_summary(trials, "correct"), condition_summary(games, "y"))


def plot(df: pd.DataFrame):
    fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **MARGINS)
    draw_scatter(ax, df)
    ax.set_xlim(0.6, 1.0)
    ax.set_xticks([0.6, 0.8, 1.0])
    ax.set_ylim(*SPECIFICITY_LIMITS)
    ax.set_xlabel("Referential accuracy,\nPhase 2")
    ax.set_ylabel("Group specificity,\nend of Phase 2")
    return fig
