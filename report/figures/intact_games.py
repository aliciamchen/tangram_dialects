"""The article's main panels drawn from the games that kept all three original
groups throughout Phase 2 alone, the preregistered sensitivity analysis of
the Phase 2 results, for the SI's check that the loss of a group does not
carry them: the within- and between-group similarity trajectories
(group_specificity.py), the before/after, adjusted and contrast panels of H1
and H2 (confirmatory_h1_h2.py), referential accuracy by block
(block_trajectories.py), group recognition by block (group_recognition.py),
the H4 columns and contrasts (confirmatory_h4.py), and the two
confirmatory-summary panels, the standardized differences of the tests and
the standardized columns (confirmatory_summary.py).

Each panel is its source module's, drawn under the three-groups restriction
of stats/primary/robustness.R (RESTRICTION, named as primary/robustness.csv
names the rerun): the per-game values and the bootstrap intervals are over the
games the rerun keeps, which _common.three_group_games() reads from the
pipeline's activeGroupsMin as the rerun does, and the brackets and contrasts
of H1, H2, H4a and H4b are the rerun's. The social_first bracket of the
adjusted panel is the exploratory test's own rerun on the same games
(stats/exploratory/social_phase2_specificity.R). The restriction concerns the
Phase 2 analyses, so no script refits the H3 tests under it and the summary
panels hold the four Phase 2 tests alone.

The module, its folder and its panels are named intact_games, the name under
which the article's Illustrator file links the panels, so that the links
hold; the games drawn are those that kept three groups, of which the games
that kept all nine players are a part. The panels are written to
results/plots/intact_games/ (FOLDER) as intact_games_<source>_<panel>.pdf,
carry no letter or title, and leave the game counts to the caption."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from report.figures import (
    block_trajectories,
    confirmatory_h1_h2,
    confirmatory_h4,
    confirmatory_summary,
    group_recognition,
    group_specificity,
)

FOLDER = "intact_games"
RESTRICTION = "three_groups"
# source module: the panels of its figure drawn here, or None for a module whose plot() is one figure
SOURCES = {
    group_specificity: None,
    confirmatory_h1_h2: ["before_after", "adjusted", "contrasts"],
    block_trajectories: ["accuracy"],
    group_recognition: ["by_block"],
    confirmatory_h4: ["h4a", "h4b", "contrasts"],
    confirmatory_summary: ["standardized_differences", "standardized_columns"],
}


def _name(module) -> str:
    return module.__name__.rsplit(".", 1)[-1]


def data(dirs) -> pd.DataFrame:
    """Every source module's table under the restriction, stacked, with `figure` naming the module each row belongs to."""
    return pd.concat(
        [module.data(dirs, restriction=RESTRICTION).assign(figure=_name(module)) for module in SOURCES],
        ignore_index=True,
    )


def plot(df: pd.DataFrame) -> dict:
    panels = {}
    for module, keep in SOURCES.items():
        name = _name(module)
        result = module.plot(df[df["figure"] == name])
        if not isinstance(result, dict):
            panels[name] = result
            continue
        for key, fig in result.items():
            if key in keep:
                panels[f"{name}_{key}"] = fig
            else:
                plt.close(fig)
    return panels
