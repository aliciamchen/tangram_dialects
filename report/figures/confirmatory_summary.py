"""Every preregistered test on one scale: H1 and H2 (refer_separated and
social_mixed against refer_mixed, on Phase 2 group specificity adjusted for
Phase 1) and the eight tests of H3 and H4 (social_first against social_mixed),
each as the standardized difference between the per-game values of its two
conditions, the difference in means over the SD of the games pooled within
the two conditions, signed so that positive is the preregistered prediction
(for concreteness, word frequency and referential accuracy the prediction is
lower in social_first). The per-game values are those of the
confirmatory_h1_h2, confirmatory_h3 and confirmatory_h4 panels, with one
addition for the H3b rate: each game's least-squares slope of within-group
similarity over Phase 1 blocks 2 to 6. The significance marks are the
preregistered tests' (significance_stars() on the estimates' p-values); the
standardized differences are descriptive, the common scale on which tests of
different models can sit together.

standardized_differences: a row per test, the standardized difference with
its percentile-bootstrap interval over games (games resampled within each
condition) and the test's significance at the right.

standardized_columns: every game of every test in standardized units, its
value minus the comparison condition's mean over the pooled within-condition
SD, signed as above, a pair of columns per test with the comparison condition
at 0 on the left and the other condition on the right, where its mean is the
standardized difference.

Under a restriction (`restriction`, a rerun of stats/primary/robustness.R
named in _common.RESTRICTIONS; the SI's intact_games.py draws the panels on
the complete games) the per-game values are those of the games the rerun
keeps and the marks are the rerun's (the robustness script refits every test
on the complete games); a test the rerun does not refit carries no mark.

The panels are written to results/plots/confirmatory_tests/ (FOLDER), carry no
letter or title, and leave the game counts to the caption."""

from __future__ import annotations

import numpy as np
import pandas as pd

from report.figures import confirmatory_h1_h2, confirmatory_h3, confirmatory_h4
from report.figures._common import draw_contrasts, keep_games, numeric, panel_figure, read_model_input, restricted_games, significance_stars
from report.plot_style import CONDITION_COLORS, GAME_POINT, INK, REFERENCE_LINE, SMALL_SIZE

FOLDER = "confirmatory_tests"
FIRST, MIXED = "social_first", "social_mixed"
# label: (per-game source, comparison condition, other condition, sign so that positive is the prediction)
TESTS = {
    "H1  Separated > mixed": ("h1", "refer_mixed", "refer_separated", +1),
    "H2  Social mixed > refer mixed": ("h2", "refer_mixed", "social_mixed", +1),
    "H3a  Phase 1 group specificity": ("h3a", MIXED, FIRST, +1),
    "H3b  Within-group similarity, block 3": ("h3b_block3", MIXED, FIRST, +1),
    "H3b  Rate of within-group convergence": ("h3b_slope", MIXED, FIRST, +1),
    "H3c  Concreteness (lower)": ("h3c_concreteness", MIXED, FIRST, -1),
    "H3c  Lexical uniqueness": ("h3c_uniqueness", MIXED, FIRST, +1),
    "H3c  Word frequency (lower)": ("h3c_word_frequency", MIXED, FIRST, -1),
    "H4a  Group recognition accuracy": ("h4a", MIXED, FIRST, +1),
    "H4b  Referential accuracy (lower)": ("h4b", MIXED, FIRST, -1),
}
RATE_BLOCKS = confirmatory_h3.RATE_BLOCKS
N_BOOT = 10000
SEED = 20261101
COLUMN_STEP = 2.6  # the distance between the pairs of columns, in columns


def _rate_slopes(dirs) -> pd.DataFrame:
    """Each social game's least-squares slope of within-group similarity over Phase 1 blocks 2 to 6."""
    w = read_model_input(dirs, "within_group_block_means.csv")
    w = w[w["condition"].isin([FIRST, MIXED]) & (numeric(w["phaseNum"]) == 1)].copy()
    w["gameBlock"] = numeric(w["gameBlock"]).astype(int)
    w["similarity"] = numeric(w["similarity"])
    w = w[w["gameBlock"].isin(RATE_BLOCKS)]
    rows = []
    for (game, condition), part in w.groupby(["gameId", "condition"]):
        means = part.groupby("gameBlock")["similarity"].mean()
        rows.append({"gameId": game, "condition": condition, "value": np.polyfit(means.index, means.to_numpy(), 1)[0]})
    return pd.DataFrame(rows)


def _pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))


def data(dirs, restriction: str | None = None) -> pd.DataFrame:
    d12, d3, d4 = (module.data(dirs, restriction) for module in (confirmatory_h1_h2, confirmatory_h3, confirmatory_h4))
    adjusted = d12[(d12["panel"] == "games") & (d12["level"] == "game")][["gameId", "condition", "value"]]
    contrasts = d12[d12["panel"] == "contrasts"].set_index("term")["p_value"]
    slopes = keep_games(_rate_slopes(dirs), restricted_games(dirs, restriction))
    per_game = {"h1": adjusted, "h2": adjusted, "h3b_slope": slopes}
    p_values = {"h1": contrasts["H1_sep_vs_mixed"], "h2": contrasts["H2_social_vs_mixed"]}
    for d in (d3, d4):
        for panel, part in d.groupby("panel"):
            per_game[panel] = part.loc[part["level"] == "game", ["gameId", "condition", "value"]]
            p_values[panel] = part["p_value"].dropna().iloc[0] if part["p_value"].notna().any() else np.nan
    p_values["h3b_slope"] = p_values.pop("h3b_rate")
    rng = np.random.default_rng(SEED)
    rows = []
    for label, (source, base, other, sign) in TESTS.items():
        g = per_game[source]
        a = g.loc[g["condition"] == base, "value"].to_numpy()
        b = g.loc[g["condition"] == other, "value"].to_numpy()
        sd = _pooled_sd(a, b)
        boot = np.array([(rb.mean() - ra.mean()) / _pooled_sd(ra, rb)
                         for ra, rb in ((rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(N_BOOT))])
        lower, upper = np.sort(sign * np.percentile(boot, [2.5, 97.5]))
        rows.append({"part": "test", "label": label, "estimate": sign * (b.mean() - a.mean()) / sd, "lower": lower,
                     "upper": upper, "p_value": p_values[source], "n_games": len(a) + len(b)})
        for condition, values in ((base, a), (other, b)):
            rows += [{"part": "game", "label": label, "condition": condition, "value": sign * (v - a.mean()) / sd}
                     for v in values]
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> dict:
    tests = df[df["part"] == "test"].set_index("label").loc[list(TESTS)].reset_index()
    return {"standardized_differences": _differences(tests), "standardized_columns": _columns(df[df["part"] == "game"], tests)}


def _differences(tests: pd.DataFrame):
    fig, ax = panel_figure(axes=(1.6, 2.2), left=2.1, right=0.4, top=0.12, bottom=0.42)
    draw_contrasts(ax, tests)
    for y, row in zip(np.arange(len(tests))[::-1], tests.itertuples()):
        if np.isfinite(row.p_value):  # no mark for a test not run under the restriction
            ax.text(1.02, y, significance_stars(row.p_value), transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=SMALL_SIZE, color=INK)
    right = max(7.0, tests["upper"].max() + 0.3)  # the article's range, widened where a subset's interval runs past it
    ax.set_xlim(-1.2, right)
    ax.set_xticks(np.arange(0, right, 2))
    ax.set_xlabel("Standardized difference\n(positive = predicted)")
    return fig


def _columns(games: pd.DataFrame, tests: pd.DataFrame):
    fig, ax = panel_figure(axes=(3.9, 1.6), left=0.55, right=0.12, top=0.12, bottom=1.2)
    rng = np.random.default_rng(SEED)
    for k, row in enumerate(tests.itertuples()):
        _, base, other, _ = TESTS[row.label]
        g = games[games["label"] == row.label]
        for j, condition in enumerate((base, other)):
            z = g.loc[g["condition"] == condition, "value"].to_numpy()
            x = COLUMN_STEP * k + j
            ax.scatter(x + rng.uniform(-0.25, 0.25, len(z)), z, color=CONDITION_COLORS[condition], zorder=1, **GAME_POINT)
            ax.plot([x - 0.32, x + 0.32], [z.mean()] * 2, color=INK, linewidth=1.0, zorder=3)
        if np.isfinite(row.p_value):
            ax.text(COLUMN_STEP * k + 0.5, max(g["value"].max(), 2.4) + 0.4, significance_stars(row.p_value), ha="center",
                    fontsize=SMALL_SIZE, color=INK)
    ax.axhline(0, **REFERENCE_LINE)
    ax.set_xticks([COLUMN_STEP * k + 0.5 for k in range(len(tests))])
    ax.set_xticklabels(tests["label"], rotation=40, ha="right", fontsize=SMALL_SIZE)
    ax.set_xlim(-0.8, COLUMN_STEP * (len(tests) - 1) + 1.8)
    ax.set_ylim(-3.5, 10.5)
    ax.set_yticks([0, 4, 8])
    ax.set_ylabel("Standardized difference from the\ncomparison condition (higher = predicted)")
    return fig
