"""The same group specificity under two similarity measures: each game's
estimate from the sentence embeddings against its estimate from content-word
overlap (Jaccard), at the end of each phase, filled for Phase 1 and hollow
for Phase 2, with the line of equality. The points hug the line, so the
result does not depend on the embedding model. The estimates are those of
stats/primary/group_specificity.R on each measure (primary/group_specificity
and primary/group_specificity_jaccard); a game appears at a phase when both
are estimable. The robustness contrasts in the SI report the same agreement
as numbers.

Each condition's points keep its color. The panel is placed by hand in the article
in the square plotting area, carries no letter or title, and leaves the game
counts to the caption."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.figures._common import panel_figure, specificity_endpoints
from report.plot_style import CONDITION_COLORS, CONDITION_ORDER, NOTE_SIZE, PHASE_POINT, REFERENCE_GRAY, REFERENCE_LINE, SQUARE_PANEL_AXES

MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.55)  # the two-line y label, nothing, nothing, the two-line x label
PHASES = {1: "phase1", 2: "phase2"}


def data(dirs) -> pd.DataFrame:
    sbert = specificity_endpoints(dirs)
    jaccard = specificity_endpoints(dirs, table="primary/group_specificity_jaccard", measure="jaccard")
    both = sbert.merge(jaccard, on=["gameId", "condition"], suffixes=("_sbert", "_jaccard"))
    rows = []
    for phase, window in PHASES.items():
        part = both.dropna(subset=[f"{window}_sbert", f"{window}_jaccard"])
        rows.append(pd.DataFrame({
            "gameId": part["gameId"], "condition": part["condition"], "phase": phase,
            "x": part[f"{window}_sbert"], "y": part[f"{window}_jaccard"], "level": "game",
        }))
    return pd.concat(rows, ignore_index=True)


def plot(df: pd.DataFrame):
    fig, ax = panel_figure(axes=SQUARE_PANEL_AXES, **MARGINS)
    lo = np.floor(min(df["x"].min(), df["y"].min()) * 10) / 10
    hi = np.ceil(max(df["x"].max(), df["y"].max()) * 10) / 10
    ax.plot([lo, hi], [lo, hi], zorder=0, **REFERENCE_LINE)
    ax.annotate("equal", xy=(hi, hi), xytext=(-1, 1), textcoords="offset points", ha="right", va="bottom",
                rotation=45, rotation_mode="anchor", fontsize=NOTE_SIZE, color=REFERENCE_GRAY)
    for condition in CONDITION_ORDER:
        color = CONDITION_COLORS[condition]
        for phase, face in ((1, color), (2, "white")):
            g = df[(df["condition"] == condition) & (df["phase"] == phase)]
            ax.scatter(g["x"], g["y"], facecolors=face, edgecolors=color, zorder=2, **PHASE_POINT)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [t for t in np.arange(0, hi + 1e-9, 0.4) if t >= lo]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal")
    ax.set_xlabel("Group specificity using\nsentence embeddings")
    ax.set_ylabel("Group specificity using\nword overlap")
    return fig
