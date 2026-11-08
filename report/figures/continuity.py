"""How much of a speaker's own convention survives Phase 2: for every speaker
and tangram, the similarity between the speaker's final Phase 2 description
and their final Phase 1 description of it (`simPhase1Phase2`), as the
cumulative distribution over speaker-tangram pairs in each condition, with
every game's own distribution as a thin line behind. The pooled
distribution stands without an interval; the game lines show the spread. The
share of pairs below 0.5, which the semantic-change analysis
(stats/secondary/semantic_change.R) reports as switched, is left to the text:
pairs near 0.5 are mostly the same convention shortened or extended, so the
panel marks no cutoff.

The panel is placed by hand in the article, carries no letter or title, and
leaves the game count to the caption."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.figures._common import in_conditions, numeric, panel_figure, read_measure
from report.plot_style import CONDITION_COLORS, CONDITION_ORDER, GAME_LINE, MEAN_LINE, PANEL_AXES

MARGINS = dict(left=0.5, right=0.15, top=0.12, bottom=0.55)  # the y label, nothing, nothing, the two-line x label


def data(dirs) -> pd.DataFrame:
    pairs = read_measure(dirs, "phase_change_similarities.csv")
    pairs["value"] = numeric(pairs["simPhase1Phase2"])
    pairs = pairs[pairs["value"].notna()]
    return in_conditions(pairs)[["gameId", "playerId", "target", "condition", "value"]].reset_index(drop=True)


def _ecdf(values) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    return x, np.arange(1, len(x) + 1) / len(x)


def plot(df: pd.DataFrame):
    fig, ax = panel_figure(axes=PANEL_AXES, **MARGINS)
    for c in CONDITION_ORDER:
        part = df[df["condition"] == c]
        for _, game in part.groupby("gameId"):
            ax.plot(*_ecdf(game["value"]), color=CONDITION_COLORS[c], drawstyle="steps-post", zorder=1, **GAME_LINE)
        ax.plot(*_ecdf(part["value"]), color=CONDITION_COLORS[c], drawstyle="steps-post", zorder=3, **MEAN_LINE)
    ax.set_xlim(-0.1, 1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("Similarity of a speaker's final\nPhase 2 description to their\nfinal Phase 1 description")
    ax.set_ylabel("Share of descriptions")
    return fig
