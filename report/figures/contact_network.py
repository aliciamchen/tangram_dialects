"""Who played with whom in one game: the nine players of the refer_mixed
example game as a network in each phase, a node per player in its original
group's color and a line between two players who spent trials in the same
trio. In Phase 1 the three triads are closed: every pair of a triad shares
all 36 trials and no other pair meets. In Phase 2 the reshuffle puts one
player of each rotation index in every trio, so two players with the same
index never meet and every other pair shares a part of the trials (8 to 15
in this game; the table carries the count). Every mixed game has this
structure; the
example is the refer_mixed game of the first convention-examples pair, which
kept all nine players to the end, so that the network panels show the same
players as the convention-spread panels.

The mixing itself is shown as a sequence: the trios of three consecutive
Phase 2 trials of the same game, a triangle per trio, so that read after the
Phase 1 panel they show the groups dissolving and re-forming from trial to
trial, each trio one player of each rotation index and two players of one
original group with one of another. The reshuffle draws each trial's trios at
random from the four arrangements its rule allows in a block, so about one
trial in four repeats the trios of the trial before, and in one of the four
every trio is three neighbors on the circle, a pair and the player beside it
across the boundary between two groups, which reads as the Phase 1 triads
shifted by one place although every trio is mixed. The sequence is the first
run of three trials in which the trios change at every trial and no trio is
three neighbors on the circle.

Five panels, keyed phase1 and phase2 for the networks and mixing_step1 to
mixing_step3 for the sequence, each the square plotting area with no axes,
letter or title, for assembly by hand; which phase or trial a panel shows goes
in the legend. The nodes carry no labels: the color names the group and the
rotation index is left to the legend. In Phase 2 of a mixed condition the
players' names and avatars were masked, so on the Phase 2 network and the
sequence a node is gray with a dotted ring in its group's color, the group
known to the analysis but not to the other players. The table carries the players as node
indices, never as ids."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from report.figures._common import is_true, network_layout, numeric, panel_figure, read_table
from report.figures.convention_examples import EXAMPLES
from report.plot_style import GROUP_COLORS, MASKED_NODE, NETWORK_EDGE, NETWORK_NODE, SQUARE_PANEL_AXES

EXAMPLE_GAME = EXAMPLES[0]["refer_mixed"]
PHASES = {"phase1": 1, "phase2": 2}
SEQUENCE_TRIALS = 3  # the consecutive Phase 2 trials drawn as a sequence of trios
MARGINS = dict(left=0.12, right=0.12, top=0.12, bottom=0.12)


def data(dirs) -> pd.DataFrame:
    players = read_table(dirs, "players.csv")
    players = players[(players["gameId"] == EXAMPLE_GAME) & is_true(players["isActive"])]
    players = players.sort_values(["originalGroup", "playerIndex"]).reset_index(drop=True)
    if len(players) != 9:
        raise ValueError("The contact network needs the example game's nine players.")
    node_of = {p: i for i, p in enumerate(players["playerId"])}
    trials = read_table(dirs, "trials.csv")
    trials = trials[trials["gameId"] == EXAMPLE_GAME].copy()
    trials["phaseNum"] = numeric(trials["phaseNum"]).astype(int)
    shared = Counter()
    for (_, _, phase), trio in trials.groupby(["roundId", "currentGroup", "phaseNum"]):
        members = sorted({node_of[p] for p in trio["playerId"] if p in node_of})
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                shared[(phase, a, b)] += 1
    nodes = pd.DataFrame({
        "part": "node",
        "node": range(len(players)),
        "originalGroup": players["originalGroup"].to_numpy(),
        "playerIndex": numeric(players["playerIndex"]).astype(int).to_numpy(),
    })
    edges = pd.DataFrame([
        {"part": "edge", "phaseNum": phase, "a": a, "b": b, "trials": n} for (phase, a, b), n in sorted(shared.items())
    ])
    return pd.concat([nodes, edges, _mixing_sequence(trials, node_of)], ignore_index=True)


def _mixing_sequence(trials: pd.DataFrame, node_of: dict) -> pd.DataFrame:
    """The trios of the first SEQUENCE_TRIALS consecutive Phase 2 trials that are all arranged
    differently, with no trio three neighbors on the circle of network_layout().

    One row per pair of players in a trio, with the step in the sequence, the
    trial's place in Phase 2 counted from 1, its block and the trio."""
    phase2 = trials[trials["phaseNum"] == 2].copy()
    phase2["trialNum"] = numeric(phase2["trialNum"]).astype(int)
    order = sorted(phase2["trialNum"].unique())
    arrangement = {
        trial: frozenset(frozenset(node_of[p] for p in trio["playerId"] if p in node_of)
                         for _, trio in part.groupby("currentGroup"))
        for trial, part in phase2.groupby("trialNum")
    }
    n = len(node_of)
    neighbors = {frozenset({i, (i + 1) % n, (i + 2) % n}) for i in range(n)}
    runs = [order[i:i + SEQUENCE_TRIALS] for i in range(len(order) - SEQUENCE_TRIALS + 1)]
    run = next((r for r in runs if len({arrangement[t] for t in r}) == SEQUENCE_TRIALS
                and not any(trio in neighbors for t in r for trio in arrangement[t])), None)
    if run is None:
        raise ValueError("The example game has no run of Phase 2 trials whose trios change at every trial "
                         "without three neighbors on the circle.")
    rows = []
    for step, trial in enumerate(run, 1):
        block = int(numeric(phase2.loc[phase2["trialNum"] == trial, "block"]).iloc[0])
        for k, trio in enumerate(sorted(arrangement[trial], key=min)):
            members = sorted(trio)
            if len(members) != 3:
                raise ValueError("Each trial of the mixing sequence needs three full trios.")
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    rows.append({"part": "trio", "step": step, "phase2Trial": order.index(trial) + 1,
                                 "block": block, "trio": k, "a": a, "b": b})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> dict:
    nodes = df[df["part"] == "node"].sort_values("node")
    edges = df[df["part"] == "edge"]
    positions = network_layout(len(nodes))
    colors = [GROUP_COLORS[g] for g in nodes["originalGroup"]]
    trios = df[df["part"] == "trio"]
    panels = {}
    selections = {key: edges[edges["phaseNum"] == phase] for key, phase in PHASES.items()}
    selections |= {f"mixing_step{step}": trios[trios["step"] == step] for step in range(1, SEQUENCE_TRIALS + 1)}
    for key, selection in selections.items():
        fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **MARGINS)
        for e in selection.itertuples():
            (x0, y0), (x1, y1) = positions[int(e.a)], positions[int(e.b)]
            ax.plot([x0, x1], [y0, y1], zorder=1, **NETWORK_EDGE)
        if key == "phase1":
            ax.scatter(positions[:, 0], positions[:, 1], c=colors, zorder=2, **NETWORK_NODE)
        else:
            ax.scatter(positions[:, 0], positions[:, 1], edgecolors=colors, zorder=2, **MASKED_NODE)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.axis("off")
        panels[key] = fig
    return panels
