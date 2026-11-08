"""How one group's words spread over the contact network: the players of the
refer_mixed game of each example pair at three Phase 2 blocks, each a node ringed in
its original group's color and filled by whose convention words its latest
description of the example tangram used (a group's color for that group's
words alone, ink for its own and another group's words together, white for
neither), with a line between the members of each trio of that block. In the
first pair's game, by block 12 most members of two groups carry both words or
the third group's, and the third group is unmoved; in the second's, by block
12 the rocket group's members carry the ring-pop group's words, alongside
their own or in their place, and most other players carry two groups' words.
The second game lost a player of
group B in block 1, so it played Phase 2 with eight, and its panels draw the
eight. The strands of the convention-examples panel
follow the same game and tangram as trajectories; this shows who was talking
to whom as the words moved.

The coding is the descriptive convention-word coding of
convention_vocabulary() and code_convention_use(): a group's convention
words are the content words in at least two of its members' final Phase 1
descriptions of the tangram, minus the words another group of the game
shares. Three panels per game, keyed pair<i>_block7, pair<i>_block9 and
pair<i>_block12 as in the convention-examples panels, each the square plotting
area with no axes, letter or title; the legend says which block a panel shows. The players are laid out as in the contact-network panels of
the same game, and the table carries them as node indices, never as ids."""

from __future__ import annotations

import pandas as pd

from report.figures._common import (
    code_convention_use,
    convention_vocabulary,
    is_true,
    network_layout,
    numeric,
    panel_figure,
    read_table,
    speaker_utterances,
)
from report.figures.convention_examples import EXAMPLES
from report.plot_style import GROUP_COLORS, INK, NETWORK_EDGE, RINGED_NODE, SQUARE_PANEL_AXES

BLOCKS = [7, 9, 12]
MARGINS = dict(left=0.12, right=0.12, top=0.12, bottom=0.12)


def _face(variant: str) -> str:
    if variant in GROUP_COLORS:
        return GROUP_COLORS[variant]
    return INK if variant == "both" else "white"


def data(dirs) -> pd.DataFrame:
    utterances = speaker_utterances(dirs)
    all_players = read_table(dirs, "players.csv")
    all_trials = read_table(dirs, "trials.csv")
    rows = []
    for pair, example in enumerate(EXAMPLES, start=1):
        rows += _game(utterances, all_players, all_trials, example["refer_mixed"], example["target"], pair)
    return pd.DataFrame(rows)


def _game(utterances, players, trials, game: str, target: str, pair: int) -> list[dict]:
    """The rows of one game: the players still in it at the end, each block's variant and trio."""
    in_game = utterances[utterances["gameId"] == game]
    coded = code_convention_use(in_game[in_game["target"] == target], convention_vocabulary(in_game))
    players = players[(players["gameId"] == game) & is_true(players["isActive"])]
    players = players.sort_values(["originalGroup", "playerIndex"]).reset_index(drop=True)
    trials = trials[(trials["gameId"] == game) & (trials["target"] == target)].copy()
    trials["gameBlock"] = numeric(trials["gameBlock"]).astype(int)
    in_phase2 = set(trials.loc[trials["gameBlock"] >= min(BLOCKS), "playerId"])
    if in_phase2 != set(players["playerId"]):
        raise ValueError(f"The convention-spread panels of pair {pair} need the same players in Phase 2 as at the end.")
    rows = []
    for block in BLOCKS:
        latest = coded[coded["gameBlock"] <= block].sort_values(["gameBlock", "trialNum"])
        variant = latest.groupby("playerId").tail(1).set_index("playerId")["variant"]
        trio = trials[trials["gameBlock"] == block].set_index("playerId")["currentGroup"]
        for i, p in enumerate(players.itertuples()):
            rows.append({
                "pair": pair, "gameBlock": block, "node": i, "originalGroup": p.originalGroup,
                "variant": variant.get(p.playerId, "none"), "trio": trio.get(p.playerId),
            })
    return rows


def plot(df: pd.DataFrame) -> dict:
    panels = {}
    for pair in sorted(df["pair"].unique()):
        panels |= _game_panels(df[df["pair"] == pair], int(pair))
    return panels


def _game_panels(df: pd.DataFrame, pair: int) -> dict:
    positions = network_layout(int(df["node"].max()) + 1)
    panels = {}
    for block in BLOCKS:
        table = df[df["gameBlock"] == block].sort_values("node")
        fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **MARGINS)
        for _, members in table.dropna(subset=["trio"]).groupby("trio"):
            idx = members["node"].to_numpy()
            for i, a in enumerate(idx):
                for b in idx[i + 1:]:
                    ax.plot(positions[[a, b], 0], positions[[a, b], 1], zorder=1, **NETWORK_EDGE)
        ax.scatter(
            positions[:, 0], positions[:, 1], facecolors=[_face(v) for v in table["variant"]],
            edgecolors=[GROUP_COLORS[g] for g in table["originalGroup"]], zorder=2, **RINGED_NODE,
        )
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.axis("off")
        panels[f"pair{pair}_block{block}"] = fig
    return panels
