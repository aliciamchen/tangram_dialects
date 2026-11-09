"""
What the figure modules share: reading the tables, the two summaries every
figure rests on, and the drawing idioms.

Not a figure itself (the leading underscore keeps it out of the runner).

Two summaries, both over games because the game is the unit the conditions
were assigned to:

    game_means(df, value, keys)          one mean per game (and per block or
                                         other key): the thin lines and the
                                         points behind every mean
    condition_summary(df, value, ...)    per condition (and key), the pooled
                                         mean over rows with its percentile
                                         bootstrap interval over games from
                                         report.bootstrap, and the game count

The drawing idioms, each taking the long table the module's data() built:

    draw_trajectory(ax, table, ...)      a quantity over blocks: per-game lines
                                         behind the condition means with their
                                         intervals
    draw_scatter(ax, table)              one quantity against another: a point
                                         per game behind the condition means
                                         with their intervals as crosses;
                                         scatter_table() builds its table

And for a figure assembled by hand from separate panels:

    panel_figure(left, right, ...)       one PANEL_AXES-sized plotting area with
                                         the given margins, inches
    dodge(ax, points)                    a transform that shifts marks sideways
                                         by points, for side-by-side marks on
                                         the broken block axis (a dodge in data
                                         units would land in the gap)
    stimulus_glyph(ax, target, ...)      the tangram, small and light, in an
                                         inset where no data can reach it;
                                         stimulus_polygons(target) reads it
                                         from its SVG
    draw_stimulus(ax, polygons, ...)     the tangram in ink at a size and place
                                         in data units (a screen's grid)
    stimulus_row_labels(ax, targets, ys) each tangram as the label of its row,
                                         a gray glyph left of the axes
    network_layout(n)                    the players of a game on a circle, in
                                         group order, for a network panel

And the restriction under which the SI draws the main panels again
(intact_games.py):

    intact_games(dirs)                   the games that kept all nine players
    restricted_games(dirs, restriction)  the games a rerun of
                                         stats/primary/robustness.R keeps, by
                                         the rerun's name (RESTRICTIONS)
    keep_games(df, games)                the rows of those games
    contrast_estimates(dirs, table, ...) a preregistered test's estimates, the
                                         rerun's under a restriction

And the descriptive codings of the Phase 1 conventions, which no model uses:

    phase1_conventions(utterances, e)    each group's convention for a tangram
                                         as the unit mean embedding of its
                                         members' final Phase 1 descriptions
    convention_vocabulary(utterances)    each group's convention words for a
                                         tangram: the content words two of its
                                         members' final Phase 1 descriptions
                                         share and no other group has
    code_convention_use(utterances, v)   whose convention words a description
                                         uses: own, both, other or neither

The long table has a `level` column: "game" rows carry one game's value,
"condition" rows the pooled mean with `lower`, `upper` and `n_games`. A
scatter's table carries `x` and `y` in place of `value`, each with its own
interval.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.patches import Polygon, Rectangle

from report.bootstrap import bootstrap_mean
from report.descriptives import utterances_file
from report.inferential import read_estimates
from report.plot_style import (
    CI_ALPHA,
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_MARKERS,
    CONDITION_ORDER,
    DIRECT_LABEL_IN_COLOR,
    GAME_LINE,
    GAME_POINT,
    GROUP_ORDER,
    INK,
    INTERVAL_LINE,
    MARKER_AREA,
    MEAN_LINE,
    MEAN_POINT,
    MUTED,
    NOTE_SIZE,
    PHASE1_BLOCKS,
    PHASE_BOUNDARY,
    REFERENCE_GRAY,
    REFERENCE_LINE,
    RESPONDENT_INTERVAL,
    RESPONDENT_MEAN,
    SHAPED_GAME_POINT,
    SMALL_SIZE,
    add_phase_boundary,
    PANEL_AXES,
    STIMULUS_GLYPH,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Reading ──────────────────────────────────────────────────


def is_true(series: pd.Series) -> pd.Series:
    return series.map(lambda v: v is True or str(v).lower() == "true")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def in_conditions(df: pd.DataFrame, conditions=CONDITION_ORDER) -> pd.DataFrame:
    return df[df["condition"].isin(conditions)].copy()


def read_table(dirs, name: str) -> pd.DataFrame:
    return pd.read_csv(dirs.tables / name, low_memory=False)


def read_measure(dirs, name: str) -> pd.DataFrame:
    return pd.read_csv(dirs.measures / name)


def read_model_input(dirs, name: str) -> pd.DataFrame:
    return pd.read_csv(dirs.model_inputs / name)


def estimates(dirs, name: str) -> pd.DataFrame:
    return read_estimates(dirs.estimates, name)


def specificity_endpoints(dirs, table: str = "primary/group_specificity", measure: str = "sbert") -> pd.DataFrame:
    """One row per game with its group specificity at the end of each phase, `phase1` and `phase2`.

    From the per-game estimates of stats/primary/group_specificity.R, on the
    sentence embeddings unless another estimates `table` and its `measure`
    are named (primary/group_specificity_jaccard, "jaccard"); NaN where a
    window has no estimate."""
    columns = ["gameId", "condition", "phase1", "phase2"]
    gs = estimates(dirs, table)
    if gs.empty:
        return pd.DataFrame(columns=columns)
    gs = gs[(gs["measure"] == measure) & is_true(gs["estimable"])]
    wide = gs.pivot(index=["gameId", "condition"], columns="window", values="coefficient").reset_index()
    for window in ("phase1_final", "phase2_final"):
        if window not in wide:
            wide[window] = np.nan
    wide = wide.rename(columns={"phase1_final": "phase1", "phase2_final": "phase2"})
    return in_conditions(wide)[columns]


def utterance_embeddings(dirs) -> tuple[pd.DataFrame, np.ndarray]:
    """The utterance table the models read with its sentence embeddings, one unit-length row per utterance, the table's `row` column giving each utterance's row in them.

    The measures step wrote embeddings.npy from this table, so the two agree
    row for row; a table rebuilt since would not, and is refused."""
    u = read_table(dirs, utterances_file(dirs.tables))
    e = np.load(dirs.measures / "embeddings.npy").astype(float)
    if len(e) != len(u):
        raise ValueError(f"{len(e)} embeddings for {len(u)} utterances: rebuild the measures")
    for col in ("phaseNum", "gameBlock", "block", "blockNum", "trialNum"):
        u[col] = numeric(u[col]).astype(int)
    u["row"] = np.arange(len(u))
    return in_conditions(u), e / np.linalg.norm(e, axis=1, keepdims=True)


def listener_trials(dirs, phase: int | None = None) -> pd.DataFrame:
    """Listener response opportunities with an outcome, under the response-opportunity rule the pipeline coded."""
    t = read_table(dirs, "trials.csv")
    t = t[(t["role"] == "listener") & is_true(t["responseOpportunity"])].copy()
    t["correct"] = numeric(t["correct"])
    t = t[t["correct"].notna()]
    for col in ("phaseNum", "gameBlock", "block"):
        t[col] = numeric(t[col]).astype(int)
    if phase is not None:
        t = t[t["phaseNum"] == phase]
    return in_conditions(t)


def speaker_utterances(dirs) -> pd.DataFrame:
    """The utterance table the models read (the filtered one when it is current)."""
    u = read_table(dirs, utterances_file(dirs.tables))
    for col in ("phaseNum", "gameBlock", "block"):
        u[col] = numeric(u[col]).astype(int)
    u["uttLength"] = numeric(u["uttLength"])
    return in_conditions(u)


def social_guesses(dirs) -> pd.DataFrame:
    """Social-identification response opportunities with an outcome."""
    g = read_table(dirs, "social_guesses.csv")
    g = g[is_true(g["responseOpportunity"])].copy()
    g["correct"] = numeric(g["correct"])
    g = g[g["correct"].notna()]
    for col in ("gameBlock", "block"):
        g[col] = numeric(g[col]).astype(int)
    return in_conditions(g)


# ── Restrictions ─────────────────────────────────────────────
# A figure drawn again on a subset of games, for the SI: the subset is one a
# rerun of stats/primary/robustness.R keeps, named as primary/robustness.csv
# names the rerun, so that the figure's games and the rerun's are the same
# games and its brackets and contrasts can be the rerun's.


def intact_games(dirs) -> set[str]:
    """The games that kept all nine players to the end: `completeNine` on games.csv, as the pipeline coded it, the games the complete-games rerun keeps."""
    games = read_table(dirs, "games.csv")
    return set(games.loc[is_true(games["completeNine"]), "gameId"])


RESTRICTIONS = {"complete_games": intact_games}  # by the rerun's name in primary/robustness.csv


def restricted_games(dirs, restriction: str | None) -> set[str] | None:
    """The games a restriction keeps, or None for every game."""
    if restriction is None:
        return None
    if restriction not in RESTRICTIONS:
        raise ValueError(f"No restriction named {restriction!r}: one of {', '.join(RESTRICTIONS)}")
    return RESTRICTIONS[restriction](dirs)


def keep_games(df: pd.DataFrame, games: set[str] | None) -> pd.DataFrame:
    """A copy of the rows of `df` whose game is in `games`, or of all of them for None."""
    return (df if games is None else df[df["gameId"].isin(games)]).copy()


def contrast_estimates(
    dirs, table: str, analysis: str, restriction: str | None = None, reruns: str = "primary/robustness"
) -> pd.DataFrame:
    """The rows of a test: those of `analysis` in `table` (primary/h1_h2, primary/h4), or under a restriction the rerun's rows in `reruns`, the table that holds the test's reruns by restriction, empty where the test was not rerun."""
    if restriction is None:
        est = estimates(dirs, table)
    else:
        est = estimates(dirs, reruns)
        est = est[est["restriction"] == restriction]
    return est[est["analysis"] == analysis]


# ── Summaries ────────────────────────────────────────────────


def game_means(df: pd.DataFrame, value: str, keys: list[str] | None = None) -> pd.DataFrame:
    """One row per game (and per extra key): the mean of `value` over the game's rows and their count."""
    keys = ["gameId", "condition"] + list(keys or [])
    out = df.groupby(keys, dropna=False)[value].agg(["mean", "size"]).reset_index()
    return out.rename(columns={"mean": "value", "size": "n_obs"}).assign(level="game")


def condition_summary(
    df: pd.DataFrame, value: str, conditions=CONDITION_ORDER, keys: list[str] | None = None
) -> pd.DataFrame:
    """Per condition (and per extra key), the pooled mean with its bootstrap interval over games."""
    rows = []
    keys = list(keys or [])
    groups = df.groupby(["condition"] + keys, dropna=False) if keys else [((c,), df[df["condition"] == c]) for c in conditions]
    for key, part in groups:
        key = key if isinstance(key, tuple) else (key,)
        if key[0] not in conditions or part[value].notna().sum() == 0:
            continue
        est = bootstrap_mean(part, value)
        rows.append({
            "condition": key[0],
            **dict(zip(keys, key[1:])),
            "value": est.mean,
            "lower": est.lower,
            "upper": est.upper,
            "n_games": est.n_games,
            "n_obs": est.n_obs,
            "level": "condition",
        })
    return pd.DataFrame(rows)


def trajectory_table(df: pd.DataFrame, value: str, panel: str, conditions=CONDITION_ORDER) -> pd.DataFrame:
    """The long table of a block trajectory: per-game means and condition summaries by game block."""
    d = in_conditions(df, conditions)
    games = game_means(d, value, ["gameBlock"])
    summary = condition_summary(d, value, conditions, ["gameBlock"])
    return pd.concat([games, summary], ignore_index=True).assign(panel=panel)


def scatter_table(games: pd.DataFrame, x_summary: pd.DataFrame, y_summary: pd.DataFrame) -> pd.DataFrame:
    """The long table of one quantity against another: per-game rows with `x` and `y`, and per condition the two means with their intervals.

    `games` holds gameId, condition, x and y; the summaries are
    condition_summary() tables of the two quantities over the same games, each
    pooled over whatever rows its quantity is defined on."""
    xs = x_summary.rename(columns={"value": "x", "lower": "x_lower", "upper": "x_upper"})
    ys = y_summary.rename(columns={"value": "y", "lower": "y_lower", "upper": "y_upper"})
    summary = xs[["condition", "x", "x_lower", "x_upper"]].merge(ys[["condition", "y", "y_lower", "y_upper"]], on="condition")
    summary["n_games"] = summary["condition"].map(games.groupby("condition")["gameId"].nunique())
    summary["level"] = "condition"
    games = games[["gameId", "condition", "x", "y"]].assign(level="game")
    return pd.concat([games, summary], ignore_index=True)


# ── Drawing ──────────────────────────────────────────────────


def _label_color(color: str) -> str:
    """The color a direct label is set in: the line's own, or ink."""
    return color if DIRECT_LABEL_IN_COLOR else INK


def _spread(positions, heights) -> np.ndarray:
    """Positions whose neighbors are at least the mean of their two heights apart, in the order of `positions` and with their mean kept."""
    positions = np.asarray(positions, dtype=float)
    heights = np.broadcast_to(np.asarray(heights, dtype=float), positions.shape)
    out = positions.copy()
    order = np.argsort(positions, kind="stable")
    for lower, upper in zip(order[:-1], order[1:]):
        out[upper] = max(out[upper], out[lower] + (heights[lower] + heights[upper]) / 2)
    return out + (positions.mean() - out.mean())


def label_lines(
    ax, table: pd.DataFrame, key: str, values, labels: dict, colors: dict, side: str = "right",
    x: str = "gameBlock", y: str = "value",
) -> None:
    """Each line's name beside its end, in place of a legend.

    `table` holds the condition-level rows of a trajectory and `key` is the
    column that tells the lines apart; `x` and `y` name the position columns,
    and the end is the row with the largest `x` (the smallest for side "left"),
    so a table with one row per series names points. Labels that would collide
    are pushed apart vertically, keeping the order of the lines, a label of two
    lines taking the room of two, and the stack is kept inside the axes. Call
    it after the y limits are final.
    """
    summary = table[table["level"] == "condition"]
    ends = {}
    for v in values:
        s = summary[summary[key] == v].sort_values(x)
        if not s.empty:
            end = s.iloc[-1] if side == "right" else s.iloc[0]
            ends[v] = (end[x], end[y])
    size = mpl.rcParams["legend.fontsize"]
    to_points = 72 / ax.figure.dpi
    anchors = [ax.transData.transform(xy)[1] * to_points for xy in ends.values()]
    heights = np.asarray([1.3 * size * (labels[v].count("\n") + 1) for v in ends])
    placed = _spread(anchors, heights)
    bottom, top = ax.transAxes.transform([(0, 0), (0, 1)])[:, 1] * to_points
    placed = placed - max(0.0, (placed + heights / 2).max() - top)
    placed = placed + max(0.0, bottom - (placed - heights / 2).min())
    for (v, xy), y0, y1 in zip(ends.items(), anchors, placed):
        ax.annotate(
            labels[v], xy=xy, xycoords="data", xytext=(4 if side == "right" else -4, y1 - y0),
            textcoords="offset points", ha="left" if side == "right" else "right", va="center",
            fontsize=size, color=_label_color(colors[v]), annotation_clip=False,
        )


def phase_segments(df: pd.DataFrame) -> list[pd.DataFrame]:
    """The rows of a trajectory split at the phase boundary, so that no line crosses the break in the axis."""
    parts = [df[df["gameBlock"] < PHASE_BOUNDARY], df[df["gameBlock"] > PHASE_BOUNDARY]]
    return [part for part in parts if not part.empty]


def draw_trajectory(
    ax, table: pd.DataFrame, conditions=CONDITION_ORDER, blocks=range(1, 13), game_lines: bool = True, linestyle="-",
) -> None:
    """Per-game lines behind the condition means and their intervals, over game blocks, one segment per phase.

    `blocks` spanning both phases gives the broken block axis; blocks of one
    phase give a plain axis. `linestyle` applies to the game lines and the
    means alike, for a second quantity drawn into the same panel
    (between-group similarity as BETWEEN_LINESTYLE beside within-group
    similarity)."""
    games = table[table["level"] == "game"]
    summary = table[table["level"] == "condition"]
    if game_lines:
        for (_, c), g in games.groupby(["gameId", "condition"]):
            for part in phase_segments(g.sort_values("gameBlock")):
                ax.plot(part["gameBlock"], part["value"], color=CONDITION_COLORS[c], linestyle=linestyle, zorder=1, **GAME_LINE)
    for c in conditions:
        s = summary[summary["condition"] == c].sort_values("gameBlock")
        if s.empty:
            continue
        for part in phase_segments(s):
            ax.fill_between(
                part["gameBlock"], part["lower"], part["upper"], color=CONDITION_COLORS[c], alpha=CI_ALPHA,
                linewidth=0, zorder=2,
            )
            ax.plot(part["gameBlock"], part["value"], color=CONDITION_COLORS[c], linestyle=linestyle, zorder=3, **MEAN_LINE)
    blocks = list(blocks)
    if min(blocks) <= PHASE1_BLOCKS < max(blocks):
        add_phase_boundary(ax, blocks=max(blocks))
    else:
        ax.set_xticks(blocks)
        ax.set_xlim(min(blocks) - 0.4, max(blocks) + 0.4)
    ax.set_xlabel("Block")


def reference_line(ax, y: float, label: str | None = None, side: str = "right") -> None:
    """A dashed gray line across the panel at `y` (chance, no change), its label in the line's gray above its right end, or its left end where the data reach the right one."""
    ax.axhline(y, **REFERENCE_LINE)
    if label:
        right = side == "right"
        ax.annotate(label, xy=(1.0 if right else 0.0, y), xycoords=("axes fraction", "data"), xytext=(-2 if right else 2, 2),
                    textcoords="offset points", ha="right" if right else "left", va="bottom", fontsize=NOTE_SIZE, color=REFERENCE_GRAY)


SPECIFICITY_LIMITS = (0, 0.8)  # the group-specificity axis the Phase 2 scatter panels share


def draw_scatter(ax, table: pd.DataFrame, conditions=CONDITION_ORDER, shapes: bool = False) -> None:
    """Each game a point behind its condition's mean, the mean a larger point with its two intervals as a cross.

    With `shapes`, each condition's points and mean take its shape from
    CONDITION_MARKERS as well as its color, for a scatter whose conditions
    overlap, and the game points are drawn a little larger (SHAPED_GAME_POINT)
    so that the shapes read in print; otherwise every point is a circle."""
    games = table[table["level"] == "game"]
    summary = table[table["level"] == "condition"]
    for c in conditions:
        color = CONDITION_COLORS[c]
        marker = CONDITION_MARKERS[c] if shapes else "o"
        area = MARKER_AREA[marker]
        point = SHAPED_GAME_POINT if shapes else GAME_POINT
        g = games[games["condition"] == c]
        ax.scatter(g["x"], g["y"], color=color, marker=marker, zorder=1, **{**point, "s": point["s"] * area})
        s = summary[summary["condition"] == c]
        if s.empty:
            continue
        s = s.iloc[0]
        ax.plot([s["x_lower"], s["x_upper"]], [s["y"], s["y"]], color=color, zorder=3, **INTERVAL_LINE)
        ax.plot([s["x"], s["x"]], [s["y_lower"], s["y_upper"]], color=color, zorder=3, **INTERVAL_LINE)
        ax.scatter([s["x"]], [s["y"]], color=color, marker=marker, zorder=4, **{**MEAN_POINT, "s": MEAN_POINT["s"] * area})


# ── Panels assembled by hand ─────────────────────────────────


def draw_column(ax, x: float, values, summary, color: str, point: dict, jitter: tuple[float, float], rng, half: float) -> None:
    """One column at `x`: `values` as points in `color`, jittered by `jitter` (across, then up and down, 0 for none) with `rng`, under the mean of `summary`, a row with `value`, `lower` and `upper` (or None for no mean), as a black line `half` either side and its interval as an outlined box."""
    values = np.asarray(values, dtype=float)
    xs = x + rng.uniform(-jitter[0], jitter[0], len(values))
    ys = values + (rng.uniform(-jitter[1], jitter[1], len(values)) if jitter[1] else 0)
    ax.scatter(xs, ys, color=color, zorder=1, **point)
    if summary is None:
        return
    ax.add_patch(Rectangle((x - half, summary["lower"]), 2 * half, summary["upper"] - summary["lower"], zorder=2, **RESPONDENT_INTERVAL))
    ax.plot([x - half, x + half], [summary["value"]] * 2, zorder=3, **RESPONDENT_MEAN)


def draw_columns(
    ax, table: pd.DataFrame, point: dict, jitter: tuple[float, float], seed: int, half: float,
    conditions=CONDITION_ORDER, level: str = "respondent",
) -> None:
    """One column per condition: its points (`level` rows of `table`, respondents
    or games) jittered by `jitter`, a fraction of a column across and of the
    value's unit up and down (0 for none), under the condition's mean as a black
    line `half` a column either side and its interval as an outlined box.

    The condition names are the tick labels, each on two lines broken at its
    space or its hyphen so that the columns can sit close; the jitter is seeded
    so that the PDF reproduces."""
    rng = np.random.default_rng(seed)
    for i, condition in enumerate(conditions):
        r = table[(table["level"] == level) & (table["condition"] == condition)]
        s = table[(table["level"] == "condition") & (table["condition"] == condition)]
        draw_column(ax, i, r["value"], None if s.empty else s.iloc[0], CONDITION_COLORS[condition], point, jitter, rng, half)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([CONDITION_LABELS[c].replace(" ", "\n", 1).replace("-", "-\n", 1) for c in conditions])
    ax.set_xlim(-0.6, len(conditions) - 0.4)


def significance_stars(p: float) -> str:
    """A planned contrast's two-tailed p-value as stars: *** below .001, ** below .01, * below .05, n.s. otherwise.

    A missing p-value is refused rather than marked n.s.: a test that was not
    run (contrast_estimates() under a restriction) gets no mark."""
    if not np.isfinite(p):
        raise ValueError("A significance mark needs a p-value; a test that was not run gets no mark")
    for cut, stars in ((0.001, "***"), (0.01, "**"), (0.05, "*")):
        if p < cut:
            return stars
    return "n.s."


def draw_bracket(ax, x0: float, x1: float, y: float, text: str) -> None:
    """A bracket from column `x0` to column `x1` at height `y`, its ends dropping a little, with `text` above it."""
    drop = 0.025 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x0, x0, x1, x1], [y - drop, y, y, y - drop], color=INK, linewidth=0.7, solid_capstyle="butt",
            clip_on=False)
    ax.text((x0 + x1) / 2, y + (0 if text.startswith("*") else 0.4 * drop), text, ha="center", va="bottom",
            fontsize=SMALL_SIZE, color=INK, clip_on=False)


def draw_contrasts(ax, table: pd.DataFrame, null: float = 0.0, log: bool = False) -> None:
    """Planned contrasts as rows, top to bottom: each estimate a point with its 95% interval as a line, the null
    value as a reference line, and each contrast's `label` as its tick label; `table` holds label, estimate,
    lower and upper, and `log` sets a logarithmic axis for odds ratios."""
    ys = np.arange(len(table))[::-1]
    for y, row in zip(ys, table.itertuples()):
        ax.plot([row.lower, row.upper], [y, y], color=INK, solid_capstyle="butt", **INTERVAL_LINE)
        ax.scatter([row.estimate], [y], color=INK, zorder=3, **MEAN_POINT)
    ax.axvline(null, **REFERENCE_LINE)
    if log:
        ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels(table["label"])
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def panel_figure(left: float, right: float, top: float = 0.12, bottom: float = 0.42, axes=PANEL_AXES):
    """A figure holding one plotting area of size `axes` with the given margins, all in inches.

    Panels written one PDF each share the plotting area, so they line up when
    set side by side; a panel's margins hold only what it carries outside the
    axes (its y label, its direct labels), and no panel carries a title."""
    w, h = left + axes[0] + right, bottom + axes[1] + top
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([left / w, bottom / h, axes[0] / w, axes[1] / h])
    return fig, ax


def dodge(ax, points: float):
    """A data transform shifted sideways by `points`, for marks set side by side within a block.

    On the broken block axis (add_phase_boundary) the x scale is a function, so
    a dodge in data units would be stretched and could land in the gap; a dodge
    in points keeps the marks a fixed distance apart whatever the axis width."""
    return transforms.offset_copy(ax.transData, fig=ax.figure, x=points, y=0, units="points")


def stimulus_polygons(target: str) -> list[list[list[float]]]:
    """The polygons of a tangram, read from its SVG in experiment/client/public/ with any matrix transform applied."""
    svg = PROJECT_ROOT / "experiment/client/public" / f"tangram_{target}.svg"
    polygons = []
    for element in ET.parse(svg).getroot().iter("{http://www.w3.org/2000/svg}polygon"):
        points = np.asarray([float(v) for v in element.attrib["points"].replace(",", " ").split()]).reshape(-1, 2)
        transform = element.get("transform")
        if transform:
            match = re.fullmatch(r"matrix\(([^)]+)\)", transform)
            if not match:
                raise ValueError(f"Unsupported stimulus transform in {svg.name}: {transform}")
            a, b, c, d, e, f = [float(v) for v in match[1].replace(",", " ").split()]
            points = points @ np.asarray([[a, b], [c, d]]) + [e, f]
        polygons.append(points.tolist())
    if not polygons:
        raise ValueError(f"No stimulus polygons in {svg.name}")
    return polygons


def stimulus_glyph(ax, target: str, box=(0.82, 0.79, 0.17, 0.19)) -> None:
    """The tangram, small and in STIMULUS_GLYPH gray, in an inset at `box` (axes fractions).

    Reserve the room: a strands panel keeps headroom above its data so that no
    mark or label reaches the glyph."""
    inset = ax.inset_axes(list(box))
    polygons = [np.asarray(p) for p in stimulus_polygons(target)]
    points = np.concatenate(polygons)
    lo, hi = points.min(0), points.max(0)
    for p in polygons:
        q = (p - lo) / (hi - lo).max()
        inset.add_patch(Polygon(np.column_stack([q[:, 0], 1 - q[:, 1]]), closed=True, facecolor=STIMULUS_GLYPH, edgecolor="none"))
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.set_aspect("equal")
    inset.axis("off")


def draw_stimulus(ax, polygons, center, size: float, color=INK) -> None:
    """A tangram's polygons (stimulus_polygons()) drawn in data units, fitted to a square of `size` centered at `center`.

    For an axes with equal aspect. The polygons come in the SVG's coordinates,
    y downward, and are drawn upright whichever way the axes' y axis runs."""
    polygons = [np.asarray(p, dtype=float) for p in polygons]
    points = np.concatenate(polygons)
    lo, hi = points.min(0), points.max(0)
    scale = size / (hi - lo).max()
    sign = 1.0 if ax.yaxis_inverted() else -1.0
    for p in polygons:
        q = (p - (lo + hi) / 2) * scale
        xy = np.column_stack([center[0] + q[:, 0], center[1] + sign * q[:, 1]])
        ax.add_patch(Polygon(xy, closed=True, facecolor=color, edgecolor=color, linewidth=0.2))


def stimulus_row_labels(ax, targets, ys, size: float = 0.2, gap: float = 0.06, color=MUTED) -> None:
    """Each tangram in `targets` as the label of its row at data height `ys`, in place of tick labels.

    A glyph `size` inches square, centered on its row and set `gap` inches
    left of the axes, each in an inset of its own so that it keeps its shape
    whatever the axes' aspect. Set the y limits first."""
    fig = ax.figure
    position = ax.get_position()
    width, height = position.width * fig.get_figwidth(), position.height * fig.get_figheight()
    y0, y1 = ax.get_ylim()
    for target, y in zip(targets, ys):
        middle = (y - y0) / (y1 - y0)
        inset = ax.inset_axes([-(gap + size) / width, middle - size / height / 2, size / width, size / height])
        draw_stimulus(inset, stimulus_polygons(target), (0, 0), 1.0, color=color)
        inset.set_xlim(-0.52, 0.52)  # a hair of room so that the glyph's outline is not clipped
        inset.set_ylim(-0.52, 0.52)
        inset.set_aspect("equal")
        inset.axis("off")
    ax.set_yticks([])


def network_layout(n: int, start: float = 110.0) -> np.ndarray:
    """The positions of `n` nodes on the unit circle, clockwise from `start` degrees.

    For a game's players sorted by original group and rotation index, so that
    each group's three members sit together and the layout is the same in
    every network panel of the same game."""
    angles = np.deg2rad(start - 360.0 * np.arange(n) / n)
    return np.column_stack([np.cos(angles), np.sin(angles)])


# ── The Phase 1 conventions ──────────────────────────────────
# Descriptive codings of what each group settled on in Phase 1, in the
# embeddings and in words. The preregistration names neither and no model
# uses them; a measure a model also used would belong in preprocessing.


def phase1_conventions(utterances: pd.DataFrame, embeddings: np.ndarray) -> dict[tuple, np.ndarray]:
    """Each group's Phase 1 convention for each tangram as a unit vector, keyed (gameId, originalGroup, target).

    The mean embedding of the group's members' final Phase 1 descriptions of
    the tangram; `utterances` and `embeddings` are what utterance_embeddings()
    returns."""
    final = utterances[utterances["phaseNum"] == 1].sort_values("blockNum").groupby(["gameId", "playerId", "target"]).tail(1)
    conventions = {}
    for key, part in final.groupby(["gameId", "originalGroup", "target"]):
        v = embeddings[part["row"].to_numpy()].mean(axis=0)
        conventions[key] = v / np.linalg.norm(v)
    return conventions


@lru_cache(maxsize=1)
def _stopwords() -> frozenset:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


def content_words(text) -> set[str]:
    """The distinct content words of a description, as preprocessing/compute_measures.py defines them: lowercased alphabetic sequences, NLTK English stopwords removed, one-character words dropped."""
    if not isinstance(text, str):
        return set()
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _stopwords() and len(w) > 1}


def convention_vocabulary(utterances: pd.DataFrame) -> dict[tuple, set[str]]:
    """Each group's convention words for each tangram, keyed (gameId, originalGroup, target).

    The content words in at least two of the group's members' final Phase 1
    descriptions of the tangram, minus any word another group of the same game
    has for it by the same rule, so that a word codes one group. A group with
    fewer than two final descriptions has no entry; a group whose shared
    words another group all shares too has an empty set."""
    final = utterances[utterances["phaseNum"] == 1].sort_values("blockNum").groupby(["gameId", "playerId", "target"]).tail(1)
    shared = {}
    for (game, group, target), part in final.groupby(["gameId", "originalGroup", "target"]):
        if len(part) < 2:
            continue
        counts = Counter(w for text in part["utterance"] for w in content_words(text))
        shared[(game, group, target)] = {w for w, n in counts.items() if n >= 2}
    return {
        (game, group, target): words.difference(*(shared.get((game, other, target), set()) for other in GROUP_ORDER if other != group))
        for (game, group, target), words in shared.items()
    }


USES = ["own", "both", "other", "neither"]


def code_convention_use(utterances: pd.DataFrame, vocabulary: dict) -> pd.DataFrame:
    """The utterances whose own group has convention words for the tangram, with `use` and `variant`.

    A description whose own group has none (no entry, or an empty set because
    another group shares every word its members shared) is left out: it could
    not use its own group's words, only another's or none.

    `use` is whose convention words the description carries: "own" (its own
    group's only), "both" (its own and another group's), "other" (another
    group's only, the group with most words in it on a tie) or "neither";
    `variant` is the group whose words it carries, a group letter, "both" or
    "none". `vocabulary` is convention_vocabulary()'s."""
    rows = []
    for u in utterances.itertuples():
        own = vocabulary.get((u.gameId, u.originalGroup, u.target))
        if not own:
            continue
        words = content_words(u.utterance)
        uses_own = bool(words & own)
        best, best_n = None, 0
        for other in GROUP_ORDER:
            if other == u.originalGroup:
                continue
            n = len(words & vocabulary.get((u.gameId, other, u.target), set()))
            if n > best_n:
                best, best_n = other, n
        if uses_own and best:
            use, variant = "both", "both"
        elif uses_own:
            use, variant = "own", u.originalGroup
        elif best:
            use, variant = "other", best
        else:
            use, variant = "neither", "none"
        rows.append((u.Index, use, variant))
    coded = pd.DataFrame(rows, columns=["index", "use", "variant"]).set_index("index")
    return utterances.join(coded, how="inner")
