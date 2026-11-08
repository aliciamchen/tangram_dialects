"""Task, condition contrasts, and verbatim descriptions from an illustrative game.

The example is the final Phase 1 description of one tangram in each original
group of a social-mixed game. It illustrates lexical variation, not an effect
size or a representative game. Transformed SVG polygons are stored in the
source-data table so plot() can draw the vector stimuli from that table alone.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np
import pandas as pd

from report.figures._common import speaker_utterances, stimulus_polygons
from report.plot_style import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_ORDER,
    DESIGN_FIGURE,
    GROUP_COLORS,
    GROUP_ORDER,
    INK,
    MUTED,
    PANEL_EDGE,
    PANEL_FACE,
)

EXAMPLE_GAME = "01M352E5Q7JZJRTCJC0YP6A7X1"
EXAMPLE_TARGET = "page3-35"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _stimulus(target):
    svg = PROJECT_ROOT / "experiment/client/public" / f"tangram_{target}.svg"
    return {"target": target, "polygons": json.dumps(stimulus_polygons(target)),
            "source": str(svg.relative_to(PROJECT_ROOT))}


def data(dirs) -> pd.DataFrame:
    u = speaker_utterances(dirs)
    example = u[
        (u["gameId"] == EXAMPLE_GAME)
        & (u["target"] == EXAMPLE_TARGET)
        & (u["phaseNum"] == 1)
        & (u["block"] == 6)
    ].copy()
    if len(example) != 3 or set(example["originalGroup"]) != set(GROUP_ORDER):
        raise ValueError("The design figure needs the three verified Phase 1 example descriptions.")
    stimulus = _stimulus(EXAMPLE_TARGET)
    example = example[
        ["gameId", "condition", "originalGroup", "phaseNum", "block", "target", "utterance"]
    ].assign(panel="example", polygons=stimulus["polygons"], source=stimulus["source"])
    constants = (PROJECT_ROOT / "experiment/shared/constants.js").read_text()
    match = re.search(r"export const all_tangrams = \[(.*?)\];", constants, re.S)
    if not match:
        raise ValueError("The experiment must define its displayed tangrams.")
    targets = re.findall(r'"([^"]+)"', match[1])
    if len(targets) != 16 or len(set(targets)) != 16:
        raise ValueError("The task schematic requires the experiment's 16 distinct shapes.")
    stimuli = pd.DataFrame([
        dict(_stimulus(target), panel="stimulus", display_order=i)
        for i, target in enumerate(targets)
    ])
    design = pd.DataFrame({
        "condition": CONDITION_ORDER,
        "phase1_goal": ["Reference", "Reference", "Reference", "Reference; later social goal announced"],
        "phase2_contact": ["Separated", "Mixed each trial", "Mixed each trial", "Mixed each trial"],
        "phase2_goal": ["Reference", "Reference", "Reference + group recognition", "Reference + group recognition"],
        "announcement": ["Between phases", "Between phases", "Between phases", "Before Phase 1"],
        "panel": "design",
        "source": "experiment/shared/constants.js; writing/preregistration/main.tex",
    })
    return pd.concat([design, example, stimuli], ignore_index=True)


def _heading(ax, letter, text):
    ax.text(0, 1, letter, weight="bold", fontsize=10, va="top")
    ax.text(0.035, 1, text, fontsize=9, va="top")


def _arrow(ax, start, end, **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9,
                               linewidth=0.9, color=MUTED, **kwargs))


def _shape(ax, polygons, center, width, height):
    points = np.concatenate([np.asarray(p) for p in polygons])
    lo, hi = points.min(axis=0), points.max(axis=0)
    # Equal scale in physical units, accounting for the axes' aspect ratio.
    box = ax.get_position()
    inches = ax.figure.get_size_inches()
    aspect = box.width * inches[0] / (box.height * inches[1])
    scale = min(width / (hi[0] - lo[0]), height / ((hi[1] - lo[1]) * aspect))
    for polygon in polygons:
        p = (np.asarray(polygon) - (hi + lo) / 2) * scale
        p[:, 1] *= -aspect
        p += np.asarray(center)
        ax.add_patch(Polygon(p, closed=True, facecolor=INK, edgecolor=INK, linewidth=0.2))


def _groups(ax, x, y, mixed=False, scale=1):
    """Three triads, with two original-group members in each mixed triad."""
    assignments = [list(g * 3) for g in GROUP_ORDER]
    if mixed:
        assignments = [["A", "A", "B"], ["B", "B", "C"], ["C", "C", "A"]]
    for j, groups in enumerate(assignments):
        cx = x + j * 0.059 * scale
        for dx, dy, group in zip([-0.012, 0.012, 0], [0.008, 0.008, -0.019], groups):
            ax.scatter(cx + dx * scale, y + dy * scale, s=60 * scale**2,
                       color=GROUP_COLORS[group], edgecolor="white", linewidth=0.4, zorder=3)
            ax.text(cx + dx * scale, y + dy * scale, group, ha="center", va="center",
                    fontsize=6.4 * scale, color=INK, zorder=4)


def _box(ax, xy, width, height):
    ax.add_patch(FancyBboxPatch(xy, width, height, boxstyle="round,pad=0.008",
                              facecolor=PANEL_FACE, edgecolor=PANEL_EDGE, linewidth=0.8))


def _bracket(ax, x, top, bottom, label):
    ax.plot([x - 0.007, x, x, x - 0.007], [top, top, bottom, bottom],
            color=MUTED, linewidth=0.8)
    ax.text(x + 0.012, (top + bottom) / 2, label, va="center", fontsize=7,
            linespacing=1.25)


def plot(df: pd.DataFrame):
    fig = plt.figure(figsize=DESIGN_FIGURE)
    task = fig.add_axes([0.025, 0.655, 0.95, 0.325])
    example_ax = fig.add_axes([0.025, 0.47, 0.95, 0.135])
    design_ax = fig.add_axes([0.025, 0.065, 0.95, 0.355])
    for ax in [task, example_ax, design_ax]:
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")

    example = df[df["panel"] == "example"].sort_values("originalGroup")
    _heading(task, "a", "One conversation: describe, select, and receive feedback")
    task.text(0, 0.83, "Speaker sees the target", fontsize=8, weight="bold")
    task.text(0.36, 0.83, "Shared chat", fontsize=8, weight="bold")
    task.text(0.69, 0.83, "Two listeners respond", fontsize=8, weight="bold")
    stimuli = df[df["panel"] == "stimulus"].sort_values("display_order")
    for i, row in enumerate(stimuli.itertuples()):
        x, y = 0.031 + (i % 4) * 0.061, 0.695 - (i // 4) * 0.166
        if row.target == EXAMPLE_TARGET:
            task.add_patch(Rectangle((x - 0.028, y - 0.076), 0.056, 0.152,
                                     fill=False, edgecolor=INK, linewidth=1.2))
        _shape(task, json.loads(row.polygons), (x, y), 0.045, 0.13)
    task.text(0, 0.055, "16 shapes; target highlighted only for speaker", fontsize=7, color=MUTED)
    _arrow(task, (0.263, 0.48), (0.338, 0.48))
    _box(task, (0.355, 0.30), 0.225, 0.40)
    quote = example.loc[example["originalGroup"] == "B", "utterance"].item()
    task.text(0.372, 0.625, "Speaker", fontsize=7, color=MUTED)
    task.text(0.468, 0.485, f'“{quote}”', ha="center", va="center", fontsize=10)
    task.text(0.36, 0.17, "Listeners can reply\nin the same chat.", fontsize=7, linespacing=1.4)
    for i, y in enumerate([0.64, 0.39], start=1):
        _arrow(task, (0.594, 0.49), (0.675, y))
        _box(task, (0.689, y - 0.092), 0.30, 0.185)
        task.text(0.705, y + 0.035, f"Listener {i}", fontsize=7.5, weight="bold")
        task.text(0.705, y - 0.048, "Select a shape from their own grid", fontsize=7)
    task.text(0.69, 0.17, "All players see the selections\nand the correct target.", fontsize=7, linespacing=1.4)
    task.text(0, -0.08, "6 targets per block  ·  Speaker rotates each block  ·  6 blocks per phase",
              fontsize=7.5, color=MUTED)

    _heading(example_ax, "b", "Phase 1: different groups develop different descriptions")
    for x, row in zip([0.016, 0.363, 0.681], example.itertuples()):
        example_ax.scatter([x + 0.012], [0.51], s=150,
                           color=GROUP_COLORS[row.originalGroup], edgecolor="white")
        example_ax.text(x + 0.012, 0.51, row.originalGroup, ha="center", va="center", fontsize=7, weight="bold")
        wrapped = textwrap.fill(f'“{row.utterance}”', width=24)
        example_ax.text(x + 0.039, 0.51, wrapped, fontsize=8, va="center", linespacing=1.15)
    example_ax.text(0, 0.06, "Same target as in a. Verbatim descriptions from three groups at the end of Phase 1 in one illustrative game.",
                    fontsize=7, color=MUTED)

    _heading(design_ax, "c", "Compare what happens when established groups meet")
    design_ax.text(0.21, 0.85, "Phase 1: form conventions", fontsize=8, ha="center", weight="bold")
    design_ax.text(0.595, 0.85, "Phase 2: use conventions", fontsize=8, ha="center", weight="bold")
    _box(design_ax, (0.01, 0.255), 0.34, 0.50)
    _groups(design_ax, 0.096, 0.60, scale=1.25)
    design_ax.text(0.18, 0.445, "Separate groups; reference only", fontsize=7.5, ha="center")
    design_ax.text(0.18, 0.325, "Later task revealed at transition", fontsize=7, ha="center", color=MUTED)
    _box(design_ax, (0.01, 0.005), 0.34, 0.19)
    design_ax.text(0.18, 0.125, "Same task + advance notice", fontsize=7.5, ha="center")
    design_ax.text(0.18, 0.047, "Later group-recognition reward announced", fontsize=7, ha="center", color=MUTED)
    y_rows = [0.68, 0.49, 0.30, 0.10]
    for y, condition in zip(y_rows, CONDITION_ORDER):
        color = CONDITION_COLORS[condition]
        _arrow(design_ax, (0.365, y), (0.415, y))
        design_ax.text(0.431, y + 0.050, CONDITION_LABELS[condition],
                       color=color, fontsize=8, va="center", weight="bold")
        goal = "Identify shape" if condition.startswith("refer") else "Identify shape + original group"
        design_ax.text(0.431, y - 0.032, goal, fontsize=7.2, va="center")
        _groups(design_ax, 0.676, y + 0.05, mixed=condition != "refer_separated", scale=0.96)
    _bracket(design_ax, 0.869, 0.68, 0.49, "Contact\nH1")
    _bracket(design_ax, 0.869, 0.48, 0.30, "Signaling\nH2")
    _bracket(design_ax, 0.869, 0.29, 0.10, "Anticipation\nH3, H4")
    fig.text(0.025, 0.029,
             "Mixed trials: identities masked; one in-group and one out-group listener per speaker when all nine players remain.",
             fontsize=7, color=MUTED)
    fig.text(0.025, 0.008,
             "A, B, C denote original groups. Social task: “Was the speaker in your original group?” (Yes / No).",
             fontsize=7, color=MUTED)
    return fig
