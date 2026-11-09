"""Every player of every game: when each left and why, a panel per condition.

Thirty games per condition, each nine rows in the original groups' colors, a
row running from the first block to the trial after which the server removed
the player (its removal timestamp falls between that trial's end and the next
trial's start for every removed player of the full sample) or to the end of
the game, with the reason as a mark in ink at a removed player's end: a cross
for a player idle for three rounds, a filled square for a group that failed
the Phase 1 accuracy screen, which sits in the gap between the phases, a
hollow circle for the last member of a group that fell below two, and a
hollow triangle for the players removed because fewer than two groups were
left. Games are ordered by what happened to them: complete games first, then
games that kept three groups but lost a player, then games that kept two,
then games that ended early, and within these by how many players were lost
and when the first left.

Unlike the article's panels, this is a complete figure for the SI: the four
condition panels side by side, each on ATTRITION_PANEL_AXES under its
condition's name, with one key for the reason marks below them; the group
colors are left to the caption. The reason categories are those the pipeline
writes in `removalReason` on players.csv."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from report.figures._common import is_true, read_table
from report.plot_style import (
    ATTRITION_PANEL_AXES,
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_ORDER,
    GAME_BLOCKS,
    GROUP_COLORS,
    GROUP_ORDER,
    LABEL_SIZE,
    MARKER_SIZE,
    PHASE_BOUNDARY,
    PLAYER_BAR,
    REMOVAL_MARK,
    REMOVAL_MARKERS,
    add_phase_boundary,
)

TRIALS_PER_BLOCK = 6
X_START, X_END = 0.5, GAME_BLOCKS + 0.5
OUTCOMES = ["complete", "three groups", "two groups", "ended early"]
REASON_ORDER = ["idle", "low accuracy", "group disbanded", "insufficient groups"]  # the removal categories the figure has a mark for
REASON_LABELS = {
    "idle": "Idle for three rounds",
    "low accuracy": "Group failed the Phase 1 accuracy screen",
    "group disbanded": "Group fell below two members",
    "insufficient groups": "Fewer than two groups left",
}
PHASE_SCREEN = PHASE_BOUNDARY  # where the accuracy screen's removals sit: the gap between the phases
ROWS_PER_GAME = 10  # nine players and a gap
MARGINS = dict(left=0.1, right=0.1, top=0.3, bottom=0.75, gap=0.3)  # in inches: nothing, nothing, the names, the x axis and the key
KEY_HANDLES = dict(handlelength=1.6, columnspacing=1.4, handletextpad=0.5)


def trial_end(t):
    """The end of 0-indexed trial `t` in game-block units, where block b spans b - 0.5 to b + 0.5."""
    t = np.asarray(t, dtype=float)
    return t // TRIALS_PER_BLOCK + 0.5 + (t % TRIALS_PER_BLOCK + 1) / TRIALS_PER_BLOCK


def player_table(dirs) -> pd.DataFrame:
    """One row per player of a real game: the group, when the player's time in the game ended and why."""
    players = read_table(dirs, "players.csv")
    games = read_table(dirs, "games.csv")
    games = games[games["condition"].isin(CONDITION_ORDER)]
    trials = read_table(dirs, "trials.csv")[["gameId", "playerId", "trialNum"]]
    last = trials.groupby(["gameId", "playerId"])["trialNum"].max().rename("lastTrial").reset_index()
    p = players.merge(last, on=["gameId", "playerId"], how="left").merge(games[["gameId", "condition", "activeGroups"]], on="gameId")
    p["removed"] = is_true(p["removed"])
    p["reason"] = p["removalReason"].where(p["removed"], "stayed")
    unmarked = set(p.loc[p["removed"], "reason"]) - set(REASON_ORDER)
    if unmarked:
        raise ValueError(f"no mark for the removal reason(s) {sorted(unmarked)}")
    p["xLeft"] = np.where(p["removed"], trial_end(p["lastTrial"]), X_END)
    p = p.sort_values(["gameId", "originalGroup", "playerIndex"]).reset_index(drop=True)
    p["slot"] = p["originalGroup"].map({g: i for i, g in enumerate(GROUP_ORDER)}) * 3 + p.groupby(["gameId", "originalGroup"]).cumcount()
    return p


def game_order(p: pd.DataFrame) -> pd.DataFrame:
    """One row per game with its outcome and its row within its condition's panel."""
    g = p.groupby(["gameId", "condition"]).agg(
        activeGroups=("activeGroups", "first"), nRemoved=("removed", "sum"), start=("gameStartTime", "min"),
        firstRemoval=("xLeft", lambda x: x[x < X_END].min() if (x < X_END).any() else np.nan),
    ).reset_index()
    g["outcome"] = np.select(
        [g["activeGroups"] == 1, g["activeGroups"] == 2, g["nRemoved"] > 0], OUTCOMES[3:0:-1], OUTCOMES[0]
    )
    g["outcomeRank"] = g["outcome"].map({o: i for i, o in enumerate(OUTCOMES)})
    g = g.sort_values(["condition", "outcomeRank", "nRemoved", "firstRemoval", "start"], na_position="last")
    g["row"] = g.groupby("condition").cumcount()
    return g[["gameId", "condition", "outcome", "row"]]


def data(dirs) -> pd.DataFrame:
    p = player_table(dirs)
    g = game_order(p)
    table = p.merge(g[["gameId", "outcome", "row"]], on="gameId")
    columns = ["gameId", "condition", "outcome", "row", "playerId", "originalGroup", "slot", "removed", "reason", "xLeft"]
    return table[columns].sort_values(["condition", "row", "slot"]).reset_index(drop=True)


def plot(df: pd.DataFrame):
    conditions = [c for c in CONDITION_ORDER if (df["condition"] == c).any()]
    w, h = ATTRITION_PANEL_AXES
    m = MARGINS
    width = m["left"] + len(conditions) * w + (len(conditions) - 1) * m["gap"] + m["right"]
    height = m["bottom"] + h + m["top"]
    fig = plt.figure(figsize=(width, height))
    for i, c in enumerate(conditions):
        ax = fig.add_axes([(m["left"] + i * (w + m["gap"])) / width, m["bottom"] / height, w / width, h / height])
        _panel(ax, df[df["condition"] == c])
        ax.set_title(CONDITION_LABELS[c], loc="left", color=CONDITION_COLORS[c], fontsize=LABEL_SIZE, pad=4)
    draw_key(fig)
    return fig


def draw_key(fig) -> None:
    """The key below the panels, centered: the reasons in two columns of two, the screen and the disbanded
    group in the first, too few groups and idle in the second. The group colors are left to the caption."""
    marks = {
        r: Line2D([], [], linestyle="none", markersize=MARKER_SIZE, markeredgewidth=0.7, label=REASON_LABELS[r], **REMOVAL_MARKERS[r])
        for r in REASON_ORDER
    }
    reasons = [marks["low accuracy"], marks["group disbanded"], marks["insufficient groups"], marks["idle"]]
    fig.legend(handles=reasons, loc="lower center", bbox_to_anchor=(0.5, 0), ncol=2, frameon=False, **KEY_HANDLES)


def _panel(ax, table: pd.DataFrame):
    y = -(table["row"] * ROWS_PER_GAME + table["slot"])
    for (_, r), yi in zip(table.iterrows(), y):
        ax.plot([X_START, r["xLeft"]], [yi, yi], color=GROUP_COLORS[r["originalGroup"]], zorder=2, **PLAYER_BAR)
        if r["removed"]:
            ax.plot([r["xLeft"]], [yi], zorder=4, clip_on=False, **REMOVAL_MARK, **REMOVAL_MARKERS[r["reason"]])
    add_phase_boundary(ax)
    ax.set_ylim(-(table["row"].max() + 1) * ROWS_PER_GAME + 1.5, 0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Block")
