"""The preregistered test of H1 and H2 (stats/primary/h1_h2.R): Phase 2 group
specificity by condition, adjusted for Phase 1, over the three conditions that
share the Phase 1 procedure, by ordinary least squares on the per-game
estimates of stats/primary/group_specificity.R, with the planned contrasts
H1 refer_separated > refer_mixed and H2 social_mixed > refer_mixed. Three
panels.

adjusted: each game's Phase 2 specificity adjusted to the mean Phase 1
specificity of the model's games with the model's Phase 1 slope
(phase2 - slope * (phase1 - mean)), a column per condition, under the
condition's mean as a black line and its bootstrap interval over games as an
outlined box; the two contrasts as brackets marked by their significance
(significance_stars()). For the three conditions in the model the column's
mean is the model's adjusted mean. social_first stands on the right for
comparison, adjusted in the same way but outside the model, since its Phase 1
follows the manipulation; its bracket with social_mixed is the exploratory
test of stats/exploratory/social_phase2_specificity.R, which fits its own
Phase 1 slope on the two social conditions, so the bracket's test and the
column's adjustment differ slightly.

contrasts: the two planned contrasts with their 95% intervals from the model,
labelled by their predicted direction.

before_after: each game's specificity at the end of Phase 1 and at the end of
Phase 2 joined by a thin line, the condition means bold, filled for Phase 1
and hollow for Phase 2, the conditions side by side, social_first on the
right as in the adjusted panel; unadjusted, the raw data of the test.

Under a restriction (`restriction`, a rerun of stats/primary/robustness.R
named in _common.RESTRICTIONS; the SI's intact_games.py draws the panels on
the complete games) the games are those the rerun keeps, the brackets and the
contrasts panel are the rerun's, and the adjustment uses the primary model's
Phase 1 slope about the kept games' mean, since the rerun writes no slope;
social_first's bracket is the exploratory test's own rerun on the same games
(exploratory/social_phase2_specificity_robustness.csv).

The panels are written to results/plots/confirmatory_tests/ (FOLDER), carry no
letter or title, and leave the game counts and the contrast values to the
caption."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.figures._common import (
    condition_summary,
    draw_bracket,
    draw_columns,
    draw_contrasts,
    estimates,
    keep_games,
    panel_figure,
    restricted_games,
    significance_stars,
    specificity_endpoints,
    contrast_estimates,
)
from report.plot_style import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    GAME_LINE,
    GAME_POINT,
    MEAN_LINE,
    PHASE_POINT,
    SMALL_SIZE,
    columns_panel_axes,
)

FOLDER = "confirmatory_tests"
CONDITIONS = ["refer_separated", "refer_mixed", "social_mixed"]
SHOWN = CONDITIONS + ["social_first"]  # the conditions drawn in the adjusted and before_after panels
SOCIAL = ["social_mixed", "social_first"]  # the games of the exploratory test
CONTRASTS = {  # term in primary/h1_h2.csv: (label, the two columns it compares)
    "H1_sep_vs_mixed": ("Separated > mixed", ("refer_separated", "refer_mixed")),
    "H2_social_vs_mixed": ("Social mixed > refer mixed", ("social_mixed", "refer_mixed")),
}
EXPLORATORY = {  # term in exploratory/social_phase2_specificity.csv: the two columns it compares
    "socialfirst_vs_socialmixed": ("social_mixed", "social_first"),
}
GAME_JITTER = (0.18, 0.0)
SEED = 20261101
MEAN_HALF = 0.26
MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.42)


def data(dirs, restriction: str | None = None) -> pd.DataFrame:
    games = specificity_endpoints(dirs).dropna(subset=["phase1", "phase2"])
    games = keep_games(games[games["condition"].isin(SHOWN)], restricted_games(dirs, restriction)).reset_index(drop=True)
    in_model = games["condition"].isin(CONDITIONS)
    coefficients = estimates(dirs, "primary/h1_h2_coefficients")
    slope = coefficients.loc[(coefficients["analysis"] == "h1_h2") & (coefficients["term"] == "gs_phase1"), "estimate"].item()
    tests = contrast_estimates(dirs, "primary/h1_h2", "h1_h2", restriction)
    tests = tests[tests["term"].isin(CONTRASTS)]
    if int(tests["n_games"].iloc[0]) != in_model.sum():
        raise ValueError("The H1/H2 panels need the same games as the model.")
    games["adjusted"] = games["phase2"] - slope * (games["phase1"] - games.loc[in_model, "phase1"].mean())
    adjusted = games.assign(value=games["adjusted"], level="game")
    summary = condition_summary(adjusted, "value", SHOWN)
    contrasts = pd.DataFrame({
        "term": tests["term"], "label": [CONTRASTS[t][0] for t in tests["term"]], "estimate": tests["estimate"],
        "lower": tests["lower"], "upper": tests["upper"], "p_value": tests["p_value"],
    })
    exploratory = contrast_estimates(
        dirs, "exploratory/social_phase2_specificity", "social_phase2_specificity", restriction,
        reruns="exploratory/social_phase2_specificity_robustness",
    )
    exploratory = exploratory[exploratory["term"].isin(EXPLORATORY)]
    if int(exploratory["n_games"].iloc[0]) != games["condition"].isin(SOCIAL).sum():
        raise ValueError("The social_first bracket needs the same games as the exploratory model.")
    return pd.concat([
        adjusted[["gameId", "condition", "phase1", "phase2", "value", "level"]].assign(panel="games"),
        summary.assign(panel="games"),
        contrasts.assign(panel="contrasts"),
        exploratory[["term", "estimate", "lower", "upper", "p_value"]].assign(panel="exploratory"),
    ], ignore_index=True)


def plot(df: pd.DataFrame) -> dict:
    columns = df[df["panel"] == "games"]
    contrasts = df[df["panel"] == "contrasts"].set_index("term").loc[list(CONTRASTS)].reset_index()
    exploratory = df[df["panel"] == "exploratory"]
    return {
        "adjusted": _adjusted(columns, contrasts, exploratory),
        "contrasts": _contrasts(contrasts),
        "before_after": _before_after(columns[columns["level"] == "game"]),
    }


def _adjusted(table: pd.DataFrame, contrasts: pd.DataFrame, exploratory: pd.DataFrame):
    fig, ax = panel_figure(axes=columns_panel_axes(len(SHOWN)), **MARGINS)
    draw_columns(ax, table, GAME_POINT, GAME_JITTER, SEED, MEAN_HALF, SHOWN, level="game")
    top = table["value"].max()
    ax.set_ylim(-0.05, top + 0.3)
    ax.set_yticks([0, 0.4, 0.8])
    for k, row in enumerate(contrasts.itertuples()):
        a, b = (SHOWN.index(c) for c in CONTRASTS[row.term][1])
        draw_bracket(ax, a, b, top + 0.07 + 0.12 * k, significance_stars(row.p_value))
    for row in exploratory.itertuples():  # beside H1's bracket, at its height
        a, b = (SHOWN.index(c) for c in EXPLORATORY[row.term])
        draw_bracket(ax, a, b, top + 0.07, significance_stars(row.p_value))
    ax.set_ylabel("Group specificity, end of\nPhase 2 (adjusted)")
    return fig


def _contrasts(contrasts: pd.DataFrame):
    fig, ax = panel_figure(axes=(1.4, 0.7), left=1.3, right=0.12, top=0.12, bottom=0.42)
    draw_contrasts(ax, contrasts)
    ax.set_xlim(-0.05, 0.45)
    ax.set_xticks([0, 0.2, 0.4])
    ax.set_xlabel("Difference in Phase 2 group\nspecificity (adjusted)")
    return fig


def _before_after(games: pd.DataFrame):
    fig, ax = panel_figure(axes=(0.45 * len(SHOWN), 1.2), **MARGINS)
    for i, c in enumerate(SHOWN):
        part = games[games["condition"] == c]
        x = np.array([2 * i, 2 * i + 1])
        for r in part.itertuples():
            ax.plot(x, [r.phase1, r.phase2], color=CONDITION_COLORS[c], zorder=1, **GAME_LINE)
        means = [part["phase1"].mean(), part["phase2"].mean()]
        ax.plot(x, means, color=CONDITION_COLORS[c], zorder=3, **MEAN_LINE)
        ax.scatter(x, means, facecolors=[CONDITION_COLORS[c], "white"], edgecolors=CONDITION_COLORS[c], zorder=4,
                   **PHASE_POINT)
        ax.text(2 * i + 0.5, -0.17, CONDITION_LABELS[c].replace(" ", "\n", 1).replace("-", "-\n", 1), ha="center", va="top",
                fontsize=SMALL_SIZE, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(2 * len(SHOWN)))
    ax.set_xticklabels(["P1", "P2"] * len(SHOWN))
    ax.set_xlim(-0.5, 2 * len(SHOWN) - 0.5)
    ax.set_ylim(-0.1, 0.9)
    ax.set_yticks([0, 0.4, 0.8])
    ax.set_ylabel("Group specificity")
    return fig
