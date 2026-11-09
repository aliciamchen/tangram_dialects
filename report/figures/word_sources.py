"""Where the words of Phase 2 descriptions come from, against the end of
Phase 1: per condition, the change in content words from each speaker's last
Phase 1 description of a tangram to their Phase 2 descriptions of it, split
by the source of the words (word_source()). Below zero, in ABSENT gray, the
shortfall in the own group's final Phase 1 words: how many fewer of them a
Phase 2 description carries than the speaker's last Phase 1 description had
words. Above zero, stacked in steps of the condition's color, the words it
adds: the own group's earlier Phase 1 words, another group's final Phase 1
words and new words. The black point is the net change, the sum of the four,
with its bootstrap interval over games.

Separated speakers only shed words. Speakers in the three mixed conditions
shed some of their group's wording and replace it: the revived earlier words
and the new words are about the same in the three, so the incentive does not
change how much speakers elaborate for a mixed audience, while refer-mixed
speakers carry about twice as many of another group's words as speakers in
the two social conditions. The panel answers where the length of mixed-
condition descriptions comes from, which the length trajectory cannot.

The baseline is the speaker's own last Phase 1 description of the same
tangram, so that groups removed at the Phase 1 accuracy screen, which have no
Phase 2, do not enter it; a Phase 2 description whose speaker has none is
left out. Words are the distinct content-word stems of a description
(content_stems()), so the counts run below the word counts of the length
trajectory. The coding is descriptive and not preregistered. The four parts
are pooled means over Phase 2 descriptions and stack to the net change; only
the net change carries an interval, as stacked shares carry none. The panel
has the COLUMNS_PANEL_AXES plotting area, a column per condition like the
word-use panel it may sit beside, and leaves the game counts to the caption.
Unlike the other main panels it names its parts itself, as the convention
fates figure does but in smaller, regular type: beside the last bar, each with its
gloss below it and a line to its segment (stack_labels()). The net change
is left to the legend.
"""

from __future__ import annotations

import pandas as pd

from report.figures._common import (
    WORD_SOURCES,
    code_word_sources,
    condition_summary,
    condition_ticks,
    content_stems,
    panel_figure,
    phase1_wording,
    reference_line,
    speaker_utterances,
    stack_labels,
)
from report.plot_style import (
    ABSENT,
    COLUMNS_PANEL_AXES,
    CONDITION_COLORS,
    CONDITION_ORDER,
    INK,
    INTERVAL_LINE,
    MEAN_POINT,
    STACK_ALPHAS,
    STACK_DIVIDER,
    STACK_OUTLINE,
)

ADDED = ["own_earlier", "other", "new"]  # the sources stacked above zero, bottom to top
PARTS = ["dropped"] + ADDED  # with the shortfall below zero, the parts of the net change
BAR_HALF = 0.32  # half a bar's width, of a column
# The names beside the last bar, in the order of their heights in `plot()`: the shortfall, then the added
# sources from the bottom of the stack up; the net change is left to the legend
LABELS = [
    ("Own final words dropped", "fewer than at the end of Phase 1"),
    ("Own group's earlier words", "used in Phase 1 but not at its end"),
    ("Another group's words", "from its final Phase 1 descriptions"),
    ("New words", "in no group's Phase 1 descriptions"),
]
NAME_SIZE = 5.5  # the names and their glosses, under the tick labels and in regular weight, as the author asked
LABEL_GAP = 0.22  # the least distance between two names, as a share of the axis, roomier than the fates figure's
MARGINS = dict(left=0.62, right=2.0, top=0.12, bottom=0.42)  # the y label, the names, nothing, the condition names


def data(dirs) -> pd.DataFrame:
    utterances = speaker_utterances(dirs)
    wording = phase1_wording(utterances)
    keys = ["gameId", "playerId", "target"]
    last = utterances[utterances["phaseNum"] == 1].sort_values("blockNum").groupby(keys).tail(1)
    start = last[keys].assign(start=[len(content_stems(t)) for t in last["utterance"]])
    coded = code_word_sources(utterances[utterances["phaseNum"] == 2], wording).merge(start, on=keys)
    coded["dropped"] = coded["own_final"] - coded["start"]
    coded["net"] = coded["content"] - coded["start"]
    rows = [
        {"condition": c, "part": p, "value": float(part[p].mean()), "n_obs": len(part), "n_games": part["gameId"].nunique(), "level": "condition"}
        for c, part in coded.groupby("condition") for p in PARTS
    ]
    return pd.concat([pd.DataFrame(rows), condition_summary(coded, "net").assign(part="net")], ignore_index=True)


def plot(df: pd.DataFrame):
    fig, ax = panel_figure(axes=COLUMNS_PANEL_AXES, **MARGINS)
    reference_line(ax, 0)
    for i, condition in enumerate(CONDITION_ORDER):
        parts = df[df["condition"] == condition].set_index("part")
        color = CONDITION_COLORS[condition]
        ax.bar(i, parts.loc["dropped", "value"], width=2 * BAR_HALF, color=ABSENT, linewidth=0, zorder=1)
        bottom = 0.0
        for j, source in enumerate(ADDED):
            height = parts.loc[source, "value"]
            ax.bar(i, height, bottom=bottom, width=2 * BAR_HALF, color=color, alpha=STACK_ALPHAS[WORD_SOURCES.index(source)], linewidth=0, zorder=1)
            if j:
                ax.plot([i - BAR_HALF, i + BAR_HALF], [bottom] * 2, zorder=2, **STACK_DIVIDER)
            bottom += height
        ax.plot([i - BAR_HALF, i - BAR_HALF, i + BAR_HALF, i + BAR_HALF], [0, bottom, bottom, 0], color=color, zorder=3, **STACK_OUTLINE)
        net = parts.loc["net"]
        ax.plot([i, i], [net["lower"], net["upper"]], color=INK, zorder=4, **INTERVAL_LINE)
        ax.scatter([i], [net["value"]], color=INK, zorder=5, **MEAN_POINT)
    ax.set_ylim(-2.5, 1.5)
    last = df[df["condition"] == CONDITION_ORDER[-1]].set_index("part")["value"]
    added = last[ADDED]
    heights = [last["dropped"] / 2, *(added.cumsum() - added / 2)]
    stack_labels(ax, len(CONDITION_ORDER) - 1 + BAR_HALF, [(*label, y) for label, y in zip(LABELS, heights)],
                 gap=LABEL_GAP, size=NAME_SIZE, weight="normal")
    condition_ticks(ax)
    ax.set_yticks([-2, -1, 0, 1])
    ax.set_ylabel("Change in content words\nfrom the end of Phase 1")
    return fig
