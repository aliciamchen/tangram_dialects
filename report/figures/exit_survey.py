"""What participants said about their Phase 1 group in the exit survey: every
respondent's rating of how much they felt part of the group and how close
they felt to its members, on the 1 to 7 scale from "not at all" to "very
much", one jittered column per condition with the condition's mean as a black
line across it and its bootstrap interval over games as a box outlined around
the line. Feeling part of the
group piles up at 7 in every condition and closeness spreads more; neither
differs by condition, so mixing did not weaken group feeling. The respondents
are every player who answered the survey, those removed before the end
included, as the survey macros count them.

Each panel's y axis is the question as the survey asked it, with the scale's
two ends named at 1 and 7. Players in the mixed conditions were asked about
"your Phase 1 group" and those in refer_separated about "your group"; the axis
gives the first wording, and the legend should say. The wording is copied from
the client source, and data() refuses to run when a string no longer appears
there.

Two panels, keyed identification and closeness, on the COLUMNS_PANEL_AXES
plotting area, the narrowest under which the four condition names fit; no letter, no title, and
the respondent counts left to the caption."""

from __future__ import annotations

import textwrap

import pandas as pd

from report.descriptives import exit_survey_responses
from report.figures._common import (
    PROJECT_ROOT,
    condition_summary,
    draw_columns,
    in_conditions,
    numeric,
    panel_figure,
    read_table,
)
from report.plot_style import COLUMNS_PANEL_AXES, RESPONDENT_POINT

EXIT_SURVEY = "experiment/client/src/intro-exit/ExitSurvey.jsx"
# Each item's question as ExitSurvey.jsx renders it, {groupLabel} filled in with
# the mixed conditions' wording; data() checks every string against the file.
ITEMS = {
    "identification": ("groupIdentification", "How much did you feel a sense of being a part of {groupLabel}?"),
    "closeness": ("groupCloseness", "How close did you feel to the people in {groupLabel}?"),
}
GROUP_LABEL = "your Phase 1 group"
ANCHORS = {1: "Not at all", 7: "Very much"}
SCALE = (1, 7)
QUESTION_CHARS = 22  # the question wraps to lines this wide on the y axis
JITTER = (0.2, 0.16)  # of a column's width and of a scale step
MEAN_HALF = 0.26  # half the width of the mean's line and box, of a column's width
SEED = 20261031
MARGINS = dict(left=1.25, right=0.12, top=0.12, bottom=0.42)  # the question and the anchors, nothing, nothing, the two-line condition names


def _check_wording() -> None:
    source = (PROJECT_ROOT / EXIT_SURVEY).read_text()
    quoted = [q for _, q in ITEMS.values()] + [f'"{GROUP_LABEL}"'] + list(ANCHORS.values())
    for literal in quoted:
        if literal not in source:
            raise ValueError(f"The exit survey figure quotes {literal!r}, which {EXIT_SURVEY} no longer contains.")


def data(dirs) -> pd.DataFrame:
    _check_wording()
    survey = in_conditions(exit_survey_responses(read_table(dirs, "players.csv"), read_table(dirs, "games.csv")))
    frames = []
    for panel, (field, _) in ITEMS.items():
        part = survey.assign(value=numeric(survey[field])).dropna(subset=["value"])
        frames.append(part[["gameId", "condition", "value"]].assign(level="respondent", panel=panel))
        frames.append(condition_summary(part, "value").assign(panel=panel))
    return pd.concat(frames, ignore_index=True)


def plot(df: pd.DataFrame) -> dict:
    panels = {}
    for panel, (_, question) in ITEMS.items():
        table = df[df["panel"] == panel]
        fig, ax = panel_figure(axes=COLUMNS_PANEL_AXES, **MARGINS)
        draw_columns(ax, table, RESPONDENT_POINT, JITTER, SEED, MEAN_HALF)
        ax.set_ylim(SCALE[0] - 0.5, SCALE[1] + 0.5)
        ticks = range(SCALE[0], SCALE[1] + 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{ANCHORS[t]}  {t}" if t in ANCHORS else str(t) for t in ticks])
        ax.set_ylabel(textwrap.fill(question.format(groupLabel=GROUP_LABEL), QUESTION_CHARS))
        panels[panel] = fig
    return panels
