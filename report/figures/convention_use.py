"""Whose words Phase 2 descriptions use, block by block, one panel per
condition: the share of descriptions that use only their own group's
convention words, both their own and another group's, another group's only,
or neither, as stacked areas over blocks 7 to 12, each category a step of the
condition's color, divided by thin white lines. Under referential goals the
own-word share falls while the both-words and other-word shares grow; under
the recognition incentive the shares barely move; in refer_separated, whose
groups never meet, the other-word share is the floor the coding has when
nothing is borrowed. The coding separates hybridization (both words) from
replacement (another group's word), which a similarity measure cannot, and
shows why mixed-condition descriptions stayed long: carrying two words is a
visible strategy.

A group's convention words for a tangram are the content words in at least
two of its members' final Phase 1 descriptions of it, minus the words another
group of the game shares (convention_vocabulary()); a description whose own
group has none is left out. The coding is descriptive and not preregistered.
The shares are pooled over the condition's descriptions in the block, with no
interval.

Beside them, one panel for all four conditions (keyed borrowed_per_game): each
game's share of descriptions using another group's words, alone or with its
own group's (the both and other categories), over the final three blocks
(LATE_BLOCKS), a point per game in a column per condition under the
condition's mean as a black line and its bootstrap interval over games as an
outlined box. The panels are placed by hand in the article, carry no letter or
title, and leave the game counts and the block window to the caption."""

from __future__ import annotations

import pandas as pd

from report.figures._common import (
    USES,
    code_convention_use,
    condition_summary,
    convention_vocabulary,
    draw_columns,
    game_means,
    panel_figure,
    speaker_utterances,
)
from report.plot_style import (
    COLUMNS_PANEL_AXES,
    CONDITION_COLORS,
    CONDITION_ORDER,
    GAME_POINT,
    STACK_ALPHAS,
    STACK_PANEL_AXES,
)

DIVIDER = dict(color="white", linewidth=0.6)  # the line between two stacked categories
BORROWED = ["both", "other"]  # the categories that use another group's words
LATE_BLOCKS = (10, 12)  # the final three blocks of Phase 2, the per-game panel's window
GAME_JITTER = (0.18, 0.0)  # of a column's width; the shares are not jittered up and down
SEED = 20261101
MEAN_HALF = 0.26  # half the width of the mean's line and box, of a column's width
MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.42)  # the y label, nothing, nothing, the block axis


def data(dirs) -> pd.DataFrame:
    utterances = speaker_utterances(dirs)
    coded = code_convention_use(utterances[utterances["phaseNum"] == 2], convention_vocabulary(utterances))
    rows = []
    for (condition, block), part in coded.groupby(["condition", "gameBlock"]):
        shares = part["use"].value_counts(normalize=True)
        for use in USES:
            rows.append({
                "condition": condition, "gameBlock": int(block), "use": use, "value": float(shares.get(use, 0.0)),
                "n_obs": len(part), "n_games": part["gameId"].nunique(), "level": "condition", "panel": "shares",
            })
    late = coded[coded["gameBlock"].between(*LATE_BLOCKS)].assign(borrowed=lambda d: d["use"].isin(BORROWED).astype(float))
    per_game = [game_means(late, "borrowed"), condition_summary(late, "borrowed")]
    return pd.concat([pd.DataFrame(rows), *(t.assign(panel="borrowed") for t in per_game)], ignore_index=True)


def plot(df: pd.DataFrame) -> dict:
    shares = df[df["panel"] == "shares"]
    panels = {}
    for condition in CONDITION_ORDER:
        table = shares[shares["condition"] == condition]
        if table.empty:
            continue
        fig, ax = panel_figure(axes=STACK_PANEL_AXES, **MARGINS)
        blocks = sorted(table["gameBlock"].unique())
        bottom = pd.Series(0.0, index=blocks)
        for i, (use, alpha) in enumerate(zip(USES, STACK_ALPHAS)):
            share = table[table["use"] == use].set_index("gameBlock")["value"].reindex(blocks).fillna(0.0)
            ax.fill_between(blocks, bottom.to_numpy(), (bottom + share).to_numpy(), color=CONDITION_COLORS[condition],
                            alpha=alpha, linewidth=0, zorder=1)
            if i:
                ax.plot(blocks, bottom.to_numpy(), zorder=2, **DIVIDER)
            bottom = bottom + share
        ax.set_xticks(blocks)
        ax.set_xlim(blocks[0], blocks[-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("Block")
        ax.set_ylabel("Share of\ndescriptions")
        panels[condition] = fig
    fig, ax = panel_figure(axes=COLUMNS_PANEL_AXES, **MARGINS)
    draw_columns(ax, df[df["panel"] == "borrowed"], GAME_POINT, GAME_JITTER, SEED, MEAN_HALF, level="game")
    ax.set_ylim(0, 0.5)
    ax.set_yticks([0, 0.2, 0.4])
    ax.set_ylabel("Share using another\ngroup's words")
    panels["borrowed_per_game"] = fig
    return panels
