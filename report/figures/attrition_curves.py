"""Attrition over the game, two panels: how many games were still whole, and
when and why players were removed.

Games: by condition, the number of games of the thirty started that were
still fully intact, with all nine players (solid), and still running, with
two or more groups (dashed, BETWEEN_LINESTYLE), as step curves over the
blocks on the broken block axis. The legend names the lines "Fully intact
(9 players)" and "Still running (two or more groups)". A game stops being
intact at the trial after which its first player was removed and stops
running when the server ended it; no game ended during Phase 1, so the dashed
lines sit at thirty until the accuracy screen.

Removals: the players removed in each block, and at the Phase 1 accuracy
screen between blocks 6 and 7, stacked by reason in REMOVAL_SHADES, ink to
light gray: idle for three rounds, the group failed the screen, the group
fell below two members, fewer than two groups were left. The screen is most
of the attrition. The reason categories are those of stats/R/attrition.R.

Unlike the article's panels, this is a complete figure for the SI: the two
panels side by side on CURVES_PANEL_AXES, each with its key to its right, centered on the plot, the
games panel's naming the conditions, the removals panel's the four reasons;
the games panel names its two lines in a key inside the plot, at the lower
left, where no curve reaches."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from report.figures.attrition_timelines import PHASE_SCREEN, REASON_LABELS, REASON_ORDER, X_END, X_START, player_table
from report.plot_style import (
    BETWEEN_LINESTYLE,
    CURVES_PANEL_AXES,
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_ORDER,
    GAME_BLOCKS,
    INK,
    MEAN_LINE,
    NOTE_SIZE,
    PHASE1_BLOCKS,
    REMOVAL_SHADES,
    add_phase_boundary,
)

GAMES_PER_CONDITION = 30
EDGES = np.concatenate([[X_START], np.arange(1, GAME_BLOCKS + 1) + 0.5])  # the start and the end of every block
SLOTS = [*range(1, PHASE1_BLOCKS + 1), PHASE_SCREEN, *range(PHASE1_BLOCKS + 1, GAME_BLOCKS + 1)]  # the blocks and the screen
LINE_LABELS = {"intact": "Fully intact (9 players)", "running": "Still running (two or more groups)"}
# In inches: the y label, the key to the right of each panel (the condition names, the reasons), the top, the x axis, and the gap between the panels
MARGINS = dict(left=0.5, keys=(0.95, 1.75), top=0.1, bottom=0.42, gap=0.3)
KEY = dict(frameon=False, fontsize=NOTE_SIZE, handlelength=1.6, handletextpad=0.5, borderaxespad=0)


def data(dirs) -> pd.DataFrame:
    p = player_table(dirs)
    games = p.groupby(["gameId", "condition"]).agg(
        activeGroups=("activeGroups", "first"), xEnd=("xLeft", "max"),
        firstRemoval=("xLeft", lambda x: x[x < X_END].min() if (x < X_END).any() else np.nan),
    ).reset_index()
    rows = []
    for c in CONDITION_ORDER:
        gc = games[games["condition"] == c]
        for e in EDGES:
            intact = int((gc["firstRemoval"].isna() | (gc["firstRemoval"] > e)).sum())
            running = int(((gc["xEnd"] > e) | ((e >= X_END) & (gc["activeGroups"] >= 2))).sum())
            rows.append(dict(panel="games", condition=c, x=e, intact=intact, running=running))
    removed = p[p["removed"]].copy()
    at_screen = (removed["xLeft"] == PHASE_SCREEN) & (removed["reason"] != "idle")
    removed["slot"] = np.where(at_screen, PHASE_SCREEN, np.ceil(removed["xLeft"] - 0.5))
    for slot in SLOTS:
        for reason in REASON_ORDER:
            rows.append(dict(panel="removals", slot=slot, reason=reason, removed=int(((removed["slot"] == slot) & (removed["reason"] == reason)).sum())))
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame):
    w, h = CURVES_PANEL_AXES
    m = MARGINS
    width = 2 * (m["left"] + w) + sum(m["keys"]) + m["gap"]
    height = m["bottom"] + h + m["top"]
    fig = plt.figure(figsize=(width, height))
    x0 = m["left"]
    for i, draw in enumerate([_games, _removals]):
        ax = fig.add_axes([x0 / width, m["bottom"] / height, w / width, h / height])
        handles = draw(ax, df[df["panel"] == ("games" if i == 0 else "removals")])
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.04, 0.5), **KEY)
        x0 += w + m["keys"][i] + m["gap"] + m["left"]
    return fig


def _games(ax, table: pd.DataFrame) -> list:
    for c in CONDITION_ORDER:
        t = table[table["condition"] == c].sort_values("x")
        if t.empty:
            continue
        ax.step(t["x"], t["running"], where="post", color=CONDITION_COLORS[c], linestyle=BETWEEN_LINESTYLE, zorder=2, **MEAN_LINE)
        ax.step(t["x"], t["intact"], where="post", color=CONDITION_COLORS[c], zorder=3, **MEAN_LINE)
    add_phase_boundary(ax)
    ax.set_ylim(0, GAMES_PER_CONDITION + 1)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel("Block")
    ax.set_ylabel("Games")
    lines = [
        Line2D([], [], color=INK, label=LINE_LABELS["intact"], **MEAN_LINE),
        Line2D([], [], color=INK, linestyle=BETWEEN_LINESTYLE, label=LINE_LABELS["running"], **MEAN_LINE),
    ]
    ax.add_artist(ax.legend(handles=lines, loc="lower left", **KEY))  # kept when the condition key is added
    return [Line2D([], [], color=CONDITION_COLORS[c], label=CONDITION_LABELS[c], **MEAN_LINE) for c in CONDITION_ORDER]


def _removals(ax, table: pd.DataFrame) -> list:
    x = np.arange(len(SLOTS))
    bottom = np.zeros(len(SLOTS))
    for reason in REASON_ORDER:
        counts = np.array([table[(table["slot"] == s) & (table["reason"] == reason)]["removed"].sum() for s in SLOTS])
        ax.bar(x, counts, bottom=bottom, width=0.75, color=REMOVAL_SHADES[reason], edgecolor="white", linewidth=0.4)
        bottom += counts
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) if s != PHASE_SCREEN else "\nscreen" for s in SLOTS])
    ax.tick_params(axis="x", labelsize=NOTE_SIZE)
    ax.set_xlabel("Block")
    ax.set_ylabel("Players removed")
    return [Patch(facecolor=REMOVAL_SHADES[r], edgecolor="none", label=REASON_LABELS[r]) for r in REASON_ORDER]
