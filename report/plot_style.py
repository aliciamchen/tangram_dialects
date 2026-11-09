"""
The one description of how the article's figures look.

Every figure module in report/figures/ takes its palette, labels, type sizes,
line and marker styles, figure sizes and matplotlib settings from here and
defines none of its own, so a change here changes every figure. The drawing
idioms built on these values are in report/figures/_common.py. The R model
scripts draw nothing.

Usage:
    from report.figures._common import panel_figure
    from report.plot_style import CONDITION_COLORS, PANEL_AXES, apply_style
    apply_style()
    fig, ax = panel_figure(left=0.42, right=0.12, axes=PANEL_AXES)
    ...
    # the report.figures runner saves the figure as results/plots/<module>.pdf
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


# ── Conditions and groups ────────────────────────────────────

CONDITION_ORDER = ["refer_separated", "refer_mixed", "social_mixed", "social_first"]

CONDITION_LABELS = {
    "refer_separated": "Refer separated",
    "refer_mixed": "Refer mixed",
    "social_mixed": "Social mixed",
    "social_first": "Social-first",
}

# Blues for the referential conditions and reds for the social ones, the
# darker member for the condition with less contact between groups. The hues
# are those of the ends of ColorBrewer's RdBu, each light member leaning toward
# cyan or orange, with the light members darkened to 3:1 contrast on white.
# Every pair of the four passes the colorblind separation, lightness, chroma
# and contrast checks of the dataviz palette validator (worst pair 16.8 OKLab
# units apart under protan and deutan vision, 16.6 under tritan).
CONDITION_COLORS = {
    "refer_separated": "#125CA1",
    "refer_mixed": "#4D9AC6",
    "social_mixed": "#DC744F",
    "social_first": "#B01C2E",
}

GROUP_ORDER = ["A", "B", "C"]
# Teal, gold and plum: hues away from the condition blues and reds, and
# every pair passes the same validator checks (worst pair 12 OKLab units apart
# under protan and deutan vision, 6.6 under tritan)
GROUP_COLORS = {"A": "#11866D", "B": "#BA8E1C", "C": "#B068C1"}

# ── Blocks ───────────────────────────────────────────────────
# Blocks are 1-indexed in every figure, as in the paper and in the `block` and
# `gameBlock` columns of the processed tables: Phase 1 is blocks 1 to 6 and
# Phase 2 is 7 to 12 on the game-block axis. Nothing here re-indexes.

PHASE1_BLOCKS = 6
PHASE2_BLOCKS = 6
GAME_BLOCKS = PHASE1_BLOCKS + PHASE2_BLOCKS
PHASE_BOUNDARY = PHASE1_BLOCKS + 0.5
# The game-block axis is broken between the phases: the space between blocks 6
# and 7 is this many blocks wider than the space between other blocks
PHASE_GAP = 0.8
# The ticks of a game-block axis too narrow to tick every block: the first,
# third and last block of each phase
SPARSE_BLOCK_TICKS = [1, 3, 6, 7, 9, 12]
# The blocks of a panel that spans Phase 2 alone, and its ticks
PHASE2_RANGE = range(PHASE1_BLOCKS + 1, GAME_BLOCKS + 1)
PHASE2_TICKS = [b for b in SPARSE_BLOCK_TICKS if b > PHASE1_BLOCKS]

# ── Type, in points ──────────────────────────────────────────
# apply_style() sets matplotlib's defaults from these, multiplied by its
# `scale`; text a figure sizes itself takes them as they are

TITLE_SIZE = 9
LABEL_SIZE = 8  # axis labels and running text
SMALL_SIZE = 7  # tick labels, legends, direct labels and footnotes
NOTE_SIZE = 6.5  # muted notes such as game counts and reference-line labels

# ── Marks ────────────────────────────────────────────────────

# Open axes drawn in black, thin lines, translucent intervals in the line's
# color, and names set beside the lines they belong to rather than in a keyed
# legend. A line's name carries its identity, so lines have no per-condition
# markers; a scatter of games, whose points of four conditions overlap, can
# give each condition a shape as well (CONDITION_MARKERS).
INK = "#000000"
MUTED = "#595959"  # secondary notes such as game counts
GAME_LINE = dict(alpha=0.08, linewidth=0.5)  # one game behind the mean, in its condition's color; overlaps darken
MEAN_LINE = dict(linewidth=1.3)
MARKER_SIZE = 4.5  # legend keys, and matplotlib's default marker
CI_ALPHA = 0.3  # an interval in its line's color; two that overlap darken where they do
GAME_POINT = dict(s=7, alpha=0.35, linewidths=0)  # one game as a point behind the condition mean, in its condition's color
MEAN_POINT = dict(s=20, linewidths=0)  # the condition mean among the game points, with its intervals as a cross
INTERVAL_LINE = dict(linewidth=0.9)  # the arms of that cross
# A shape per condition for a scatter whose points overlap, with an area scale
# per shape so that the four look equally heavy at the same nominal size
CONDITION_MARKERS = {"refer_separated": "o", "refer_mixed": "s", "social_mixed": "^", "social_first": "D"}
MARKER_AREA = {"o": 1.0, "s": 0.75, "^": 1.2, "D": 0.7}
SHAPED_GAME_POINT = dict(GAME_POINT, s=12, alpha=0.35)  # a game point that carries a shape, large enough to read in print
REFERENCE_GRAY = "#BFBFBF"  # a reference line and its label (chance, no change), set in the same light gray
REFERENCE_LINE = dict(color=REFERENCE_GRAY, linewidth=0.75, linestyle=(0, (4, 2)))  # chance, zero, no effect: a thin dashed gray line
BOUNDARY_LINE = REFERENCE_LINE  # the phase boundary, drawn like every other reference line
DIRECT_LABEL_IN_COLOR = True  # a line's name in the line's color; False sets it in INK, as Nature's artwork guide asks
BETWEEN_LINESTYLE = (0, (3, 1.5))  # between-group similarity where it shares a panel with within-group (solid) and the color carries the condition
BETWEEN_MARKER = "^"  # the mark of the second kind of line where the lines carry points: a filled triangle, since a hollow circle means Phase 2
STIMULUS_GLYPH = "#B8BEBB"  # a stimulus drawn small and light in a panel, as a reminder of which tangram it shows
# A mark that is filled before the manipulation and hollow after it: the face
# the series color or white, the edge the series color
PHASE_POINT = dict(s=12, linewidths=0.8)
ROW_POINT = dict(s=5, linewidths=0.6)  # the same on a figure with one row per game, where the rows are a few points apart
ROW_CONNECTOR = dict(linewidth=0.6, alpha=0.6)  # the line joining a game's two endpoints on such a row, in its condition's color
RESPONDENT_POINT = dict(s=7, alpha=0.11, linewidths=0)  # one respondent among hundreds that overlap, lighter than a game point
# The condition mean over such respondents: a black line across the column, its
# bootstrap interval the outline of a box around the line
RESPONDENT_MEAN = dict(color=INK, linewidth=1.0, solid_capstyle="butt")
RESPONDENT_INTERVAL = dict(facecolor="none", edgecolor=INK, linewidth=0.6)
# The range of a permutation null behind an observed value, light so that the points stand out
NULL_BAND = dict(color="#A6A6A6", alpha=0.25, linewidth=1.5)
# A network of a game's players: a node per player, its rim white where the
# color alone carries the group (NETWORK_NODE) or wider where the rim carries
# the group and the face something else (RINGED_NODE); edges in ink, thin and
# translucent so that crossings darken, all of one width
NETWORK_NODE = dict(s=70, edgecolors="white", linewidths=0.8)
RINGED_NODE = dict(s=70, linewidths=1.8)
# A player whose identity the other players could not see (Phase 2 of the mixed
# conditions, where everyone appeared as "Player" with an avatar redrawn every
# trial): a gray node with a dotted ring in its original group's color, the
# group known to the analysis but not to the other players
MASKED_FACE = "#B8BEBB"
MASKED_NODE = dict(s=70, facecolors=MASKED_FACE, linewidths=1.8, linestyles=(0, (1, 1)))
NETWORK_EDGE = dict(color=INK, alpha=0.4, linewidth=0.9, solid_capstyle="round")
# Stacked shares of one series: the alpha of the series color for each category in order
STACK_ALPHAS = [1.0, 0.6, 0.3, 0.1]
# A player's time in the game on the attrition timelines: a bar in the original group's
# color from the first block to the trial after which the server removed the player, or
# to the end of the game, with a mark in ink at a removed player's end for the reason
# (REMOVAL_MARKERS: lines for the player's own inactivity, a filled square for the Phase 1
# accuracy screen, hollow marks for the two removals that follow from others' leaving)
PLAYER_BAR = dict(linewidth=0.9, solid_capstyle="butt")
REMOVAL_MARK = dict(markersize=1.5, markeredgewidth=0.45, linestyle="none")
REMOVAL_MARKERS = {
    "idle": dict(marker="x", markerfacecolor=INK, markeredgecolor=INK),
    "low accuracy": dict(marker="s", markerfacecolor=INK, markeredgecolor=INK),
    "group disbanded": dict(marker="o", markerfacecolor="white", markeredgecolor=INK),
    "insufficient groups": dict(marker="v", markerfacecolor="white", markeredgecolor=INK),
}
# The same reasons as stacked bars of removals per block: ink to light gray in the order above
REMOVAL_SHADES = {"idle": INK, "low accuracy": "#7A7A7A", "group disbanded": "#B5B5B5", "insufficient groups": "#DCDCDC"}
# A category that is an absence rather than a condition (a convention nobody uses any more), in light gray
ABSENT = "#DCDCDC"

# ── Sizes, in inches ─────────────────────────────────────────
# Every article panel is written as its own PDF for assembly by hand, so the
# sizes here are plotting areas, not figure sizes; the whole-figure sizes the
# exploration notebooks use are in explore/_shared/style.py.

# The plotting area of a panel written as its own PDF for assembly by hand, so
# that panels line up when set side by side; the margins hold what each carries
PANEL_AXES = (1.44, 0.96)
# The plotting area of such a panel that spans all twelve blocks, a little
# larger than PANEL_AXES for the broken axis and the names at the line ends,
# and ticked with SPARSE_BLOCK_TICKS; the type is the same size as everywhere,
# so it is large on the panel and survives the panel being scaled down in print
BLOCK_PANEL_AXES = (1.8, 1.05)
# The plotting area of a block panel that leads its figure (the similarity
# trajectories of group_specificity.py), wider and taller than BLOCK_PANEL_AXES
# and ticked at every block; it is enlarged a little when placed, so that its
# type reads a little larger than that of the panels beside it
LARGE_BLOCK_PANEL_AXES = (3.15, 1.7)
# The plotting area of a panel of Phase 2 outcomes, square: one quantity against
# another with a point per game, or a trajectory over the six Phase 2 blocks
# set beside such a panel
SQUARE_PANEL_AXES = (1.2, 1.2)
# The plotting area of a panel of stacked shares over the six Phase 2 blocks (the
# word-use panels), one per condition and so smaller than PANEL_AXES
STACK_PANEL_AXES = (1.1, 0.75)
# The plotting area of a panel with a jittered column per condition (the exit
# survey's respondents, the word-use shares of games), the narrowest that keeps
# the four two-line condition names apart, and as tall as SQUARE_PANEL_AXES
COLUMNS_PANEL_AXES = (1.6, 1.2)
COLUMN_WIDTH = COLUMNS_PANEL_AXES[0] / 4  # one condition's column, for a panel of fewer conditions


def columns_panel_axes(n: int) -> tuple[float, float]:
    """The plotting area of a panel with a column for each of `n` conditions (the confirmatory tests)."""
    return (COLUMN_WIDTH * n, COLUMNS_PANEL_AXES[1])


# The plotting area of a panel with one row per game (the permutation null),
# tall so that 120 rows are a few points apart
ROWS_PANEL_AXES = (1.76, 3.33)
# The plotting area of a panel with one row per target tangram and, within a row,
# a mark per phase and condition (group specificity by tangram): twelve rows of
# five marks, tall enough that the marks of a row stay clear of each other
TANGRAM_ROWS_PANEL_AXES = (1.8, 3.6)
# The plotting area of a panel with a row per player of every game of a condition (the
# attrition timelines): thirty games of nine rows and a gap, tall enough that the rows
# are over a point and a half apart and the removal marks on neighboring rows stay clear
# of each other, and narrow enough that four panels sit side by side across a page
ATTRITION_PANEL_AXES = (1.5, 7.0)
# The plotting area of the two attrition-curve panels: a block panel a little taller,
# since the figure stands alone in the SI with its keys beside the panels
CURVES_PANEL_AXES = (BLOCK_PANEL_AXES[0], 1.3)
FONT_FAMILY = ["Arial Nova", "Helvetica Neue", "Arial", "DejaVu Sans"]


def _register_local_fonts() -> None:
    """Make the fonts in the user's font directory known to matplotlib."""
    for pattern in ("~/Library/Fonts/ArialNova*.ttf", "~/Library/Fonts/Arial Nova*.ttf"):
        for path in glob.glob(str(Path(pattern).expanduser())):
            font_manager.fontManager.addfont(path)


def apply_style(scale: float = 1.0) -> None:
    """Set the matplotlib defaults every figure shares.

    `scale` multiplies the type sizes: 1 for the article's panels, about 2 for
    a figure drawn at slide size.
    """
    _register_local_fonts()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": LABEL_SIZE * scale,
            "axes.titlesize": TITLE_SIZE * scale,
            "axes.titleweight": "normal",
            "axes.labelsize": LABEL_SIZE * scale,
            "xtick.labelsize": SMALL_SIZE * scale,
            "ytick.labelsize": SMALL_SIZE * scale,
            "legend.fontsize": SMALL_SIZE * scale,
            "legend.frameon": False,
            "legend.title_fontsize": SMALL_SIZE * scale,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "text.color": INK,
            "axes.linewidth": 0.75,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "lines.linewidth": MEAN_LINE["linewidth"],
            "lines.markersize": MARKER_SIZE,
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,  # editable text in the PDF
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _open_phase_gap(x):
    """Game block to axis position: the space between blocks 6 and 7 is widened by PHASE_GAP."""
    return x + PHASE_GAP * np.clip(x - PHASE1_BLOCKS, 0, 1)


def _close_phase_gap(x):
    """Axis position to game block, the inverse of _open_phase_gap."""
    return x - PHASE_GAP * np.clip((x - PHASE1_BLOCKS) / (1 + PHASE_GAP), 0, 1)


def add_phase_boundary(ax, blocks: int = GAME_BLOCKS) -> None:
    """Break a game-block axis between the phases, mark the boundary and tick every block.

    The x scale widens the space between blocks 6 and 7, so the data stay in
    game blocks. The axis line stops short of the gap on both sides, and a
    dashed line marks the boundary in its middle. Draw each phase as its own
    segment so that no line crosses the gap, and leave the x limits as set
    here, since the two axis lines end at them.
    """
    ax.set_xscale("function", functions=(_open_phase_gap, _close_phase_gap))
    lo, hi = 0.6, blocks + 0.4
    ax.set_xlim(lo, hi)
    ax.set_xticks(range(1, blocks + 1))
    ax.axvline(PHASE_BOUNDARY, **BOUNDARY_LINE)
    # Each axis line runs past its phase's last tick as far as at the outer ends
    overhang = (hi - blocks) / (1 + PHASE_GAP)
    ax.spines["bottom"].set_visible(False)
    for x0, x1 in [(lo, PHASE1_BLOCKS + overhang), (PHASE1_BLOCKS + 1 - overhang, hi)]:
        ax.add_artist(Line2D(
            [x0, x1], [0, 0], transform=ax.get_xaxis_transform(), color=plt.rcParams["axes.edgecolor"],
            linewidth=plt.rcParams["axes.linewidth"], solid_capstyle="butt", clip_on=False, zorder=2.5,
        ))

