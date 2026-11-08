"""The preregistered tests of H4, that anticipating the social-identification
incentive changes Phase 2 communicative success (stats/primary/h4.R): logistic
mixed models of each Phase 2 response opportunity in the two social
conditions, with H4a social_first > social_mixed in social-identification
accuracy and H4b social_mixed > social_first in referential accuracy. Three
panels.

h4a and h4b: each game's accuracy over its Phase 2 response opportunities, as
the pipeline coded them (a correct on-time answer 1, an incorrect or late
answer or a timeout 0), a column per condition under the condition's mean as a
black line and its bootstrap interval over games as an outlined box, with the
preregistered test's significance as a bracket (significance_stars()).

contrasts: the two planned contrasts as odds ratios with their 95% intervals
from the models, on a logarithmic axis with the null at 1, labelled by their
predicted direction.

Under a restriction (`restriction`, a rerun of stats/primary/robustness.R
named in _common.RESTRICTIONS; the SI's intact_games.py draws the panels on
the complete games) the games are those the rerun keeps and the brackets and
contrasts are the rerun's.

The panels are written to results/plots/confirmatory_tests/ (FOLDER), carry no
letter or title, and leave the game counts and the test statistics to the
caption."""

from __future__ import annotations

import pandas as pd

from report.figures._common import (
    condition_summary,
    draw_bracket,
    draw_columns,
    draw_contrasts,
    game_means,
    keep_games,
    listener_trials,
    panel_figure,
    restricted_games,
    significance_stars,
    social_guesses,
    contrast_estimates,
)
from report.plot_style import GAME_POINT, columns_panel_axes

FOLDER = "confirmatory_tests"
CONDITIONS = ["social_mixed", "social_first"]
TESTS = {  # panel: (analysis in primary/h4.csv, label of the contrast, y label)
    "h4a": ("h4a", "Social-first > social mixed", "Group recognition\naccuracy"),
    "h4b": ("h4b", "Social mixed > social-first", "Referential accuracy"),
}
GAME_JITTER = (0.18, 0.0)
SEED = 20261101
MEAN_HALF = 0.26
MARGINS = dict(left=0.62, right=0.12, top=0.12, bottom=0.42)


def data(dirs, restriction: str | None = None) -> pd.DataFrame:
    kept = restricted_games(dirs, restriction)
    sources = {
        "h4a": social_guesses(dirs),
        "h4b": listener_trials(dirs, phase=2),
    }
    rows = []
    for panel, trials in sources.items():
        trials = keep_games(trials[trials["condition"].isin(CONDITIONS)], kept)
        analysis, label, _ = TESTS[panel]
        test = contrast_estimates(dirs, "primary/h4", analysis, restriction)
        if len(test) != 1:
            raise ValueError(f"One {analysis} contrast expected, found {len(test)}")
        test = test.iloc[0]
        if int(test["n_games"]) != trials["gameId"].nunique():
            raise ValueError(f"The {panel} panel needs the same games as the model.")
        rows += [
            game_means(trials, "correct").assign(panel=panel),
            condition_summary(trials, "correct", CONDITIONS).assign(panel=panel, p_value=test["p_value"]),
            pd.DataFrame([{"panel": "contrasts", "term": panel, "label": label, "estimate": test["estimate"],
                           "lower": test["lower"], "upper": test["upper"], "p_value": test["p_value"]}]),
        ]
    return pd.concat(rows, ignore_index=True)


def plot(df: pd.DataFrame) -> dict:
    panels = {}
    for panel, (_, _, ylabel) in TESTS.items():
        table = df[df["panel"] == panel]
        fig, ax = panel_figure(axes=columns_panel_axes(len(CONDITIONS)), **MARGINS)
        draw_columns(ax, table, GAME_POINT, GAME_JITTER, SEED, MEAN_HALF, CONDITIONS, level="game")
        ax.set_ylim(0.5, 1.1)
        ax.set_yticks([0.5, 0.75, 1.0])
        p = table.loc[table["level"] == "condition", "p_value"].iloc[0]
        draw_bracket(ax, 0, 1, 1.03, significance_stars(p))
        ax.set_ylabel(ylabel)
        panels[panel] = fig
    contrasts = df[df["panel"] == "contrasts"].set_index("term").loc[list(TESTS)].reset_index()
    fig, ax = panel_figure(axes=(1.4, 0.7), left=1.3, right=0.12, top=0.12, bottom=0.42)
    draw_contrasts(ax, contrasts, null=1.0, log=True)
    ax.set_xlim(0.6, 2.0)
    ax.set_xticks([0.75, 1.0, 1.5])
    ax.set_xticklabels(["0.75", "1", "1.5"])
    ax.minorticks_off()
    ax.set_xlabel("Odds ratio")
    panels["contrasts"] = fig
    return panels
