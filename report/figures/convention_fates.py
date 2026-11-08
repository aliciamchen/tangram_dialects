"""What became of each group's Phase 1 conventions by the end of Phase 2, one
stacked bar per condition: the share of conventions that the group's own
members and another group's members both used in the final three blocks
(shared), that only its own members used (kept), that only another group's
members used (taken over) and that nobody used (gone). Under referential
goals most conventions end shared; under the recognition incentive most are
kept; taken over is rare everywhere, so mixing adds a group's words to the
others' rather than replacing them.

A group's convention for a tangram is its convention words
(convention_vocabulary(): the content words in at least two of its members'
final Phase 1 descriptions of the tangram, minus any word another group of
the game shares by the same rule), and a description uses the convention when
it carries any of those words, as code_convention_use() codes it. The fate is
read over LATE_BLOCKS, the final three blocks of Phase 2, in which each
tangram is described about nine times, about three by the group's own members
and six by the others': "used" means at least one description there by a
member of the group in question. A convention counts only in a game whose
Phase 2 reached the window and only where one of its own members described
the tangram there, since otherwise its fate cannot be read, and a group whose
shared words another group all shares too has no convention for that
tangram. The coding is descriptive and not preregistered. The shares are
pooled over the condition's conventions with no interval; in refer_separated,
whose groups never meet, the shared share is the floor the coding has when a
convention word is an ordinary word another group uses on its own.

Unlike the article's panels, this is a complete figure for the SI: the four
fates are named beside the last bar, each capitalized with its gloss in
parentheses below it and a line to its segment, and a segment tall enough to
hold it carries its share; the condition names are the tick labels. The bars stack the two fates in which the group still uses the
convention at the bottom (shared, then kept) and the two in which it has
dropped it above (taken over, then gone, in ABSENT gray)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.figures._common import (
    content_words,
    convention_vocabulary,
    panel_figure,
    speaker_utterances,
)
from report.plot_style import (
    ABSENT,
    COLUMNS_PANEL_AXES,
    CONDITION_COLORS,
    CONDITION_LABELS,
    CONDITION_ORDER,
    INK,
    MUTED,
    NOTE_SIZE,
    REFERENCE_GRAY,
    SMALL_SIZE,
    STACK_ALPHAS,
)

LATE_BLOCKS = (10, 12)  # the final three blocks of Phase 2, the window the fates are read over
FATES = ["shared", "kept", "taken over", "gone"]  # bottom to top
GLOSSES = {
    "shared": "own group and another group use it",
    "kept": "own group uses it, no other group uses it",
    "taken over": "another group uses it, own group dropped it",
    "gone": "nobody uses it any more",
}
BAR_HALF = 0.31  # half a bar's width, of a column
MIN_SHARE_FOR_TEXT = 0.04  # a segment shorter than this carries no share
LABEL_GAP = 0.2  # the least distance between two fate labels, as a share of the axis
LEADER = dict(color=REFERENCE_GRAY, linewidth=0.5, clip_on=False)  # the line from a fate's label to its segment
DIVIDER = dict(color="white", linewidth=0.6)  # the line between two stacked fates
MARGINS = dict(left=0.62, right=2.0, top=0.12, bottom=0.42)  # the y label, the fate labels, nothing, the condition names


def data(dirs) -> pd.DataFrame:
    utterances = speaker_utterances(dirs)
    vocabulary = convention_vocabulary(utterances)
    late = utterances[(utterances["phaseNum"] == 2) & utterances["gameBlock"].between(*LATE_BLOCKS)].copy()
    late["words"] = late["utterance"].map(content_words)
    rows = []
    for (game, group, target), words in vocabulary.items():
        described = late[(late["gameId"] == game) & (late["target"] == target)]
        if not words or described.empty:
            continue
        own = described["originalGroup"] == group
        if not own.any():
            continue
        uses = described["words"].map(lambda w: bool(w & words))
        own_uses, other_uses = bool((uses & own).any()), bool((uses & ~own).any())
        fate = "shared" if own_uses and other_uses else "kept" if own_uses else "taken over" if other_uses else "gone"
        rows.append({"gameId": game, "condition": described["condition"].iloc[0], "fate": fate})
    conventions = pd.DataFrame(rows)
    out = []
    for condition, part in conventions.groupby("condition"):
        shares = part["fate"].value_counts(normalize=True)
        for fate in FATES:
            out.append({
                "condition": condition, "fate": fate, "value": float(shares.get(fate, 0.0)),
                "n_obs": len(part), "n_games": part["gameId"].nunique(), "level": "condition",
            })
    return pd.DataFrame(out)


def plot(df: pd.DataFrame):
    fig, ax = panel_figure(axes=COLUMNS_PANEL_AXES, **MARGINS)
    centers = {}
    for i, condition in enumerate(CONDITION_ORDER):
        table = df[df["condition"] == condition].set_index("fate")["value"]
        bottom = 0.0
        for j, fate in enumerate(FATES):
            share = float(table.get(fate, 0.0))
            face = dict(color=ABSENT) if fate == "gone" else dict(color=CONDITION_COLORS[condition], alpha=STACK_ALPHAS[j])
            ax.bar(i, share, bottom=bottom, width=2 * BAR_HALF, linewidth=0, zorder=1, **face)
            if j:
                ax.plot([i - BAR_HALF, i + BAR_HALF], [bottom] * 2, zorder=2, **DIVIDER)
            if share >= MIN_SHARE_FOR_TEXT:
                ax.text(i, bottom + share / 2, f"{100 * share:.0f}%", ha="center", va="center", fontsize=NOTE_SIZE,
                        color="white" if j == 0 else INK, zorder=3)
            centers[fate] = bottom + share / 2
            bottom += share
    # The fate labels beside the last bar: pushed up from the bottom so none overlap, then walked down from
    # the top so the lowest stays beside its segment; a line from each to the center of its segment
    ys = [centers[f] for f in FATES]
    placed = []
    for y in ys:
        placed.append(y if not placed else max(y, placed[-1] + LABEL_GAP))
    placed[-1] = min(placed[-1], 1.0 - LABEL_GAP / 2)
    for j in range(len(placed) - 2, -1, -1):
        placed[j] = min(placed[j], placed[j + 1] - LABEL_GAP)
    last = len(CONDITION_ORDER) - 1
    x_text = last + BAR_HALF + 0.3
    for fate, y_segment, y_label in zip(FATES, ys, placed):
        ax.plot([last + BAR_HALF + 0.03, x_text - 0.05], [y_segment, y_label], zorder=2, **LEADER)
        ax.text(x_text, y_label, fate.capitalize(), fontsize=SMALL_SIZE, fontweight="bold", color=INK, ha="left",
                va="center", clip_on=False)
        ax.text(x_text, y_label - 0.06, f"({GLOSSES[fate]})", fontsize=NOTE_SIZE, color=MUTED, ha="left", va="top",
                clip_on=False)
    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels([CONDITION_LABELS[c].replace(" ", "\n", 1).replace("-", "-\n", 1) for c in CONDITION_ORDER])
    ax.set_xlim(-0.6, len(CONDITION_ORDER) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Share of Phase 1\nconventions")
    return fig
