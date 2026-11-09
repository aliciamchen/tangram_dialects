"""The two example pairs of the convention-examples figure (EXAMPLES), each
with the refer_separated game matched to it on Phase 1, every description of
the pair's tangram in the three games written out with each word marked by
where it comes from (word_source()): the own group's final Phase 1 words in
bold, its earlier Phase 1 words underlined with dots, another group's Phase 1
words on a tint of that group's color, new words in italic, and function
words, one-letter words and punctuation in gray (WORD_SOURCE_TEXT). For each
game, first each member's last Phase 1 description of the tangram, grouped
by original group, then the Phase 2 descriptions block by block with the
block's number beside the first. Each description has a bar in its speaker's
group color beside its lines, and its lines after the first are indented,
so that where one description ends and the next begins shows also within a
group.

It shows what the word-sources panel counts: in the refer-separated games
each group shortens its own label, in the refer-mixed games speakers take up
another group's word beside or in place of their own (ring pop), and in the
social-mixed games they keep their group's wording and add to it. Digits and
one-letter words are not content words, so a convention written in them
('z and 3') is all gray, and a sentence in the caption says so.

A complete SI figure per pair, keyed pair1 and pair2, since the two together
run past the journal's 170 mm height limit: each is drawn at the SI's full
width with its key above the three games side by side (refer separated,
refer mixed, social mixed) and the pair's tangram in the margin beside the
condition names, set in color or in ink as DIRECT_LABEL_IN_COLOR says. The
rows line up across the three games (the condition names, the two headings,
each group's Phase 1 lines and each Phase 2 block), so that a block reads
across. A heading has more space above it than below, so that it sits with
the lines it introduces. The text is set word by word so that each word
carries its own mark and a description wraps over lines, never cut; the
widths are measured with the figure's own fonts. The coding is descriptive
and not preregistered.
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from report.figures._common import (
    _label_color,
    content_words,
    draw_stimulus,
    phase1_wording,
    speaker_utterances,
    stimulus_polygons,
    word_source,
    word_stem,
)
from report.figures.convention_examples import CONDITIONS, EXAMPLES
from report.plot_style import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    GROUP_COLORS,
    GROUP_ORDER,
    INK,
    LABEL_SIZE,
    MUTED,
    PHASE2_RANGE,
    SMALL_SIZE,
    TITLE_SIZE,
    WORD_SOURCE_TEXT,
    WORD_TINT_ALPHA,
    WORD_UNDERLINE,
)

GAMES = ["refer_separated", *CONDITIONS]  # the columns, left to right

# The layout, in points: a margin for the tangram, a column per game, and in each column a
# gutter for the block numbers and the group bars before the text
FIGURE_WIDTH = 6.9 * 72  # the SI's full width of 180 mm, less the padding save_figure() adds either side
GLYPH_MARGIN = 36.0
GLYPH_SIZE = 28.0
COLUMN_GAP = 14.0
BLOCK_RIGHT = 10.0  # where a block number ends
BAR_X, BAR_WIDTH = 13.0, 1.6  # a group bar's left edge and width
TEXT_X = 18.0  # where the text starts
TEXT_SIZE = SMALL_SIZE
HANG = TEXT_SIZE  # a description's indent after its first line, so that each description starts flush left
LINE = 1.38 * TEXT_SIZE  # from one line's baseline to the next, also between two descriptions, since space between them would take pair 2 past 170 mm
ASCENT, DESCENT = 0.8, 0.25  # of the type size, a line's extent above and below its baseline
# The white space before each row: a heading has more above it than below
NAME_GAP = 6.0  # under the condition names, above the first heading
HEADING_GAP = 5.0  # under a heading, above its first row
GROUP_GAP = 5.0  # between two groups' Phase 1 lines, or two Phase 2 blocks
SECTION_GAP = 14.0  # under the last Phase 1 lines, above the second heading
KEY_GAP = 16.0  # between the key and the condition names
KEY_ITEM_GAP = 16.0
KEY_TINT = "#8A8A8A"  # the key's tint, neutral since a borrowed word takes its lender's color
TINT_PAD = 0.6  # a tint's overhang either side of its word
SECTIONS = {"phase1": "End of Phase 1", "phase2": "Phase 2"}
KEY = [("own_final", "Own group's final Phase 1 words"), ("own_earlier", "own group's earlier Phase 1 words"),
       ("other", "another group's Phase 1 words"), ("new", "new words")]
# The rows, top to bottom, each with the white space above it
ROWS = (
    [(("name", None), 0.0), (("heading", "phase1"), NAME_GAP)]
    + [(("phase1", g), HEADING_GAP if i == 0 else GROUP_GAP) for i, g in enumerate(GROUP_ORDER)]
    + [(("heading", "phase2"), SECTION_GAP)]
    + [(("phase2", b), HEADING_GAP if i == 0 else GROUP_GAP) for i, b in enumerate(PHASE2_RANGE)]
)


def _pieces(text: str, game: str, group: str, target: str, wording: dict) -> list[tuple[int, str, str, str | None]]:
    """A description as (chunk, piece, source, lender) rows: each whitespace-separated chunk split into its alphabetic runs and the rest, each run that is a content word coded by word_source(), everything else "function"."""
    rows = []
    for chunk, part in enumerate(str(text).split()):
        for piece in re.findall(r"[A-Za-z]+|[^A-Za-z]+", part):
            words = content_words(piece) if piece[0].isalpha() else set()
            source, lender = word_source(word_stem(next(iter(words))), game, group, target, wording) if words else ("function", None)
            rows.append((chunk, piece, source, lender))
    return rows


def data(dirs) -> pd.DataFrame:
    utterances = speaker_utterances(dirs)
    wording = phase1_wording(utterances)
    rows = []
    for pair, example in enumerate(EXAMPLES, start=1):
        target = example["target"]
        for condition in GAMES:
            game = example[condition]
            d = utterances[(utterances["gameId"] == game) & (utterances["target"] == target)]
            if d.empty:
                raise ValueError(f"No descriptions of {target} in the {condition} game of example pair {pair}.")
            last = d[d["phaseNum"] == 1].sort_values("blockNum").groupby("playerId").tail(1)
            sections = {
                "phase1": last.sort_values(["originalGroup", "blockNum", "trialNum"]),
                "phase2": d[d["phaseNum"] == 2].sort_values(["blockNum", "trialNum", "originalGroup"]),
            }
            for section, part in sections.items():
                for line, u in enumerate(part.itertuples()):
                    for chunk, piece, source, lender in _pieces(u.utterance, game, u.originalGroup, target, wording):
                        rows.append({
                            "pair": pair, "target": target, "condition": condition, "section": section, "line": line,
                            "group": u.originalGroup, "block": int(u.gameBlock), "chunk": chunk, "text": piece,
                            "source": source, "lender": lender,
                        })
    return pd.DataFrame(rows)


# ── Layout ───────────────────────────────────────────────────
# Everything is placed in points from the top left, as drawing operations
# (dicts with a `kind` and a `y`), each row laid out from its own top and
# moved into place once the row's height across the three games is known.


class _Measure:
    """Text widths in points, measured with the fonts the figure is drawn in."""

    def __init__(self):
        self.renderer = FigureCanvasAgg(Figure(dpi=72)).get_renderer()
        self.cache = {}

    def __call__(self, text: str, size: float, weight: str = "normal", style: str = "normal") -> float:
        key = (text, size, weight, style)
        if key not in self.cache:
            prop = FontProperties(family="sans-serif", size=size, weight=weight, style=style)
            self.cache[key] = self.renderer.get_text_width_height_descent(text, prop, ismath=False)[0]
        return self.cache[key]


def _font(source: str) -> dict:
    style = WORD_SOURCE_TEXT[source]
    return dict(weight=style.get("weight", "normal"), style=style.get("style", "normal"))


def _text(x: float, y: float, s: str, size: float, **style) -> dict:
    return dict(kind="text", x=x, y=y, s=s, size=size, style=style)


def _set(measure: _Measure, pieces: pd.DataFrame, x0: float, baseline: float, width: float) -> tuple[list, int]:
    """One description set from (x0, baseline), wrapping between chunks within `width` with its lines after the first indented by HANG: its operations and its number of lines."""
    space = measure("a a", TEXT_SIZE) - measure("aa", TEXT_SIZE)
    ops, tints = [], []
    x, indent, lines = 0.0, 0.0, 1
    for _, chunk in pieces.groupby("chunk", sort=True):
        w = sum(measure(p.text, TEXT_SIZE, **_font(p.source)) for p in chunk.itertuples())
        if x > indent and x + space + w > width:
            x, indent, lines = HANG, HANG, lines + 1
        elif x > indent:
            x += space
        y = baseline + (lines - 1) * LINE
        for p in chunk.itertuples():
            pw = measure(p.text, TEXT_SIZE, **_font(p.source))
            ops.append(_text(x0 + x, y, p.text, TEXT_SIZE, **WORD_SOURCE_TEXT[p.source]))
            if p.source == "own_earlier":
                ops.append(dict(kind="underline", x=x0 + x, x1=x0 + x + pw, y=y + 0.18 * TEXT_SIZE))
            if p.source == "other":
                last = tints[-1] if tints else None
                if last and last["y"] == y and last["lender"] == p.lender and x0 + x - last["x1"] <= space + 0.1:
                    last["x1"] = x0 + x + pw  # one tint across a borrowed phrase
                else:
                    tints.append(dict(x0=x0 + x, x1=x0 + x + pw, y=y, lender=p.lender))
            x += pw
    for t in tints:
        ops.append(dict(kind="tint", x=t["x0"] - TINT_PAD, y=t["y"], w=t["x1"] - t["x0"] + 2 * TINT_PAD, color=GROUP_COLORS[t["lender"]]))
    return ops, lines


def _bar(x: float, first: float, last: float, group: str) -> dict:
    """A group bar beside the lines from baseline `first` to baseline `last`."""
    top = first - ASCENT * TEXT_SIZE
    return dict(kind="bar", x=x, y=top, h=last + DESCENT * TEXT_SIZE - top, color=GROUP_COLORS[group])


def _row(measure: _Measure, row: tuple, game: pd.DataFrame, condition: str, x0: float, width: float) -> tuple[list, float]:
    """One row of one game's column laid out from its own top: its operations and its height (0 when empty)."""
    kind, which = row
    if kind == "name":
        name = CONDITION_LABELS[condition]
        return [_text(x0 + TEXT_X, ASCENT * TITLE_SIZE, name, TITLE_SIZE, color=_label_color(CONDITION_COLORS[condition]))], (ASCENT + DESCENT) * TITLE_SIZE
    if kind == "heading":
        return [_text(x0 + TEXT_X, ASCENT * LABEL_SIZE, SECTIONS[which], LABEL_SIZE, color=INK)], (ASCENT + DESCENT) * LABEL_SIZE
    column = "group" if kind == "phase1" else "block"
    rows = game[(game["section"] == kind) & (game[column] == which)]
    ops, y = [], ASCENT * TEXT_SIZE
    for _, description in rows.groupby("line", sort=True):
        set_ops, lines = _set(measure, description, x0 + TEXT_X, y, width - TEXT_X)
        ops += set_ops
        ops.append(_bar(x0 + BAR_X, y, y + (lines - 1) * LINE, description["group"].iloc[0]))
        y += lines * LINE
    if not ops:
        return [], 0.0
    last = y - LINE
    if kind == "phase2":
        ops.append(_text(x0 + BLOCK_RIGHT, ASCENT * TEXT_SIZE, str(which), TEXT_SIZE, color=MUTED, ha="right"))
    return ops, last + DESCENT * TEXT_SIZE


def _key(measure: _Measure) -> tuple[list, float]:
    ops, x, y = [], GLYPH_MARGIN + TEXT_X, ASCENT * TEXT_SIZE
    for source, label in KEY:
        w = measure(label, TEXT_SIZE, **_font(source))
        if source == "other":
            ops.append(dict(kind="tint", x=x - TINT_PAD, y=y, w=w + 2 * TINT_PAD, color=KEY_TINT))
        if source == "own_earlier":
            ops.append(dict(kind="underline", x=x, x1=x + w, y=y + 0.18 * TEXT_SIZE))
        ops.append(_text(x, y, label, TEXT_SIZE, **WORD_SOURCE_TEXT[source]))
        x += w + KEY_ITEM_GAP
    return ops, (ASCENT + DESCENT) * TEXT_SIZE


def plot(df: pd.DataFrame) -> dict:
    return {f"pair{pair}": _pair_figure(rows) for pair, rows in df.groupby("pair", sort=True)}


def _pair_figure(rows: pd.DataFrame):
    """One pair's figure: the key, then the three games side by side, row by row, with the tangram in the margin."""
    measure = _Measure()
    width = (FIGURE_WIDTH - GLYPH_MARGIN - (len(GAMES) - 1) * COLUMN_GAP) / len(GAMES)
    ops, top = _key(measure)
    top += KEY_GAP
    ops.append(dict(kind="glyph", x=GLYPH_SIZE / 2, y=top + GLYPH_SIZE / 2, target=rows["target"].iloc[0]))
    y = top
    for row, gap in ROWS:
        laid = [
            _row(measure, row, rows[rows["condition"] == condition], condition, GLYPH_MARGIN + i * (width + COLUMN_GAP), width)
            for i, condition in enumerate(GAMES)
        ]
        height = max(h for _, h in laid)
        if not height:
            continue
        y += gap if y > top else 0.0
        for row_ops, _ in laid:
            for op in row_ops:
                op["y"] += y
            ops += row_ops
        y += height
    return _draw(ops, y)


def _draw(ops: list, height: float):
    """The figure the operations describe, FIGURE_WIDTH by `height` points."""
    fig = plt.figure(figsize=(FIGURE_WIDTH / 72, height / 72))
    fx = lambda x: x / FIGURE_WIDTH
    fy = lambda y: 1 - y / height
    for op in ops:
        kind = op["kind"]
        if kind == "text":
            style = dict(op["style"])
            fig.text(fx(op["x"]), fy(op["y"]), op["s"], fontsize=op["size"], fontweight=style.pop("weight", "normal"),
                     fontstyle=style.pop("style", "normal"), ha=style.pop("ha", "left"), va="baseline", zorder=3, **style)
        elif kind == "underline":
            fig.add_artist(Line2D([fx(op["x"]), fx(op["x1"])], [fy(op["y"])] * 2, transform=fig.transFigure, zorder=2, **WORD_UNDERLINE))
        elif kind == "tint":
            low, high = op["y"] + 0.26 * TEXT_SIZE, op["y"] - 0.84 * TEXT_SIZE
            fig.add_artist(Rectangle((fx(op["x"]), fy(low)), op["w"] / FIGURE_WIDTH, (low - high) / height, transform=fig.transFigure,
                                     facecolor=op["color"], alpha=WORD_TINT_ALPHA, edgecolor="none", zorder=1))
        elif kind == "bar":
            fig.add_artist(Rectangle((fx(op["x"]), fy(op["y"] + op["h"])), BAR_WIDTH / FIGURE_WIDTH, op["h"] / height,
                                     transform=fig.transFigure, facecolor=op["color"], edgecolor="none", zorder=1))
        elif kind == "glyph":
            ax = fig.add_axes([fx(op["x"] - GLYPH_SIZE / 2), fy(op["y"] + GLYPH_SIZE / 2), GLYPH_SIZE / FIGURE_WIDTH, GLYPH_SIZE / height])
            draw_stimulus(ax, stimulus_polygons(op["target"]), (0, 0), 1.0, color=MUTED)
            ax.set_xlim(-0.52, 0.52)
            ax.set_ylim(-0.52, 0.52)
            ax.set_aspect("equal")
            ax.axis("off")
    return fig
