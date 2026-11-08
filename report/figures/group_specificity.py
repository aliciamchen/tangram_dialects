"""How alike descriptions of the same tangram are within an original group and
between groups, block by block across the whole game, with the four conditions
overlaid in one panel: each game a thin line behind its condition's mean and
bootstrap interval, within-group similarity solid and between-group similarity
dashed. The per-game endpoint estimates and the preregistered H1 and H2
contrasts (stats/primary/) summarize the gap between the solid and dashed
lines at the end of each phase.

The panel is placed by hand beside other panels in the article, so it has a
fixed plotting area with explicit margins and carries no letter or title; it
leads its figure, so the plotting area is LARGE_BLOCK_PANEL_AXES. Under a
restriction (`restriction`, a rerun of stats/primary/robustness.R named in
_common.RESTRICTIONS) the table holds only the games the rerun keeps, for the
SI's version of the panel (intact_games.py)."""

from __future__ import annotations

import pandas as pd

from report.bootstrap import bootstrap_mean
from report.figures._common import draw_trajectory, game_means, keep_games, numeric, panel_figure, read_model_input, restricted_games
from report.plot_style import (
    BETWEEN_LINESTYLE,
    CONDITION_ORDER,
    LARGE_BLOCK_PANEL_AXES,
)

PAIR_TYPES = ["within", "between"]
LINESTYLES = {"within": "-", "between": BETWEEN_LINESTYLE}
# A condition's mean at a block is drawn only where at least this many of its
# games contribute. Within-group pairs exist from the second block of a phase;
# the few at block 7 arise only in games short a member, where the reshuffle
# can no longer keep two members of a group apart, so they are omitted.
MIN_GAMES = 10
MARGINS = dict(left=0.42, right=0.12, top=0.12, bottom=0.42)  # the y label, nothing, nothing, the block axis


def data(dirs, restriction: str | None = None) -> pd.DataFrame:
    means = read_model_input(dirs, "block_similarity_means.csv")
    means = keep_games(means[means["condition"].isin(CONDITION_ORDER)], restricted_games(dirs, restriction))
    means["gameBlock"] = numeric(means["gameBlock"]).astype(int)
    cell = ["condition", "gameBlock", "pair_type"]
    n_games = means.groupby(cell)["gameId"].transform("nunique")
    means = means[n_games >= MIN_GAMES]
    games = game_means(means, "similarity", ["gameBlock", "pair_type"])
    rows = []
    for (c, b, p), part in means.groupby(cell):
        est = bootstrap_mean(part, "similarity")
        rows.append({"condition": c, "gameBlock": b, "pair_type": p, "value": est.mean, "lower": est.lower,
                     "upper": est.upper, "n_games": est.n_games, "n_obs": est.n_obs, "level": "condition"})
    return pd.concat([games, pd.DataFrame(rows)], ignore_index=True)


def plot(df: pd.DataFrame):
    fig, ax = panel_figure(axes=LARGE_BLOCK_PANEL_AXES, **MARGINS)
    for pair_type in PAIR_TYPES:
        draw_trajectory(ax, df[df["pair_type"] == pair_type], CONDITION_ORDER, linestyle=LINESTYLES[pair_type])
    ax.set_ylim(0.1, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Similarity between descriptions")
    return fig  # the game count, 30 per condition, is left to the caption
