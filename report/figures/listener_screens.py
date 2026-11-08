"""What a listener sees in Phase 2, simplified: the referential listener's
screen beside the social listener's. Each screen keeps what the listener acts
on, the instruction above the 4 by 4 grid of shapes and the speaker's
description in the chat beside it, and the social listener's adds the group
question with its two answers below the grid, so the social manipulation reads
as an added task on the same game. The rest of the app (the header with the
timer and the score, the group card, the avatars, the chat input) is left out,
and the layout follows the app's only roughly.

The instruction, the question and its answers are copied from the experiment's
client, and data() refuses to run when a string no longer appears there. The
grid and the description come from a real trial: the first Phase 2 trial of the
example target in the first convention-examples pair, the referential screen
from its refer_mixed game and the social one from its social_mixed game, each
the screen of the speaker's in-group listener in the trio that was current
group B (its chat is a single message in both games). The grid is in the order
the server dealt that listener, and the target is outlined in ink, as the app
outlines the shape a listener clicks.

Two panels, keyed referential and social, each the screen with no axes, letter
or title, for assembly by hand; which condition a panel shows goes in the
legend."""

from __future__ import annotations

import json
import textwrap

from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.textpath import TextToPath
import pandas as pd

from report.figures._common import (
    PROJECT_ROOT,
    draw_stimulus,
    is_true,
    numeric,
    panel_figure,
    read_table,
    stimulus_polygons,
)
from report.figures.convention_examples import EXAMPLES
from report.plot_style import INK, MUTED, NOTE_SIZE, PANEL_EDGE, PANEL_FACE, SCREEN_PANEL_AXES, SMALL_SIZE

EXAMPLE = EXAMPLES[0]
SCREENS = {"referential": "refer_mixed", "social": "social_mixed"}
BLOCK = 1  # the first Phase 2 block
TRIO = "B"  # the current group of the trio whose screen is shown

REFGAME = "experiment/client/src/stages/Refgame.jsx"
# What the listener is asked, as Refgame.jsx renders it; data() checks every
# entry against the file.
WORDING = {
    "prompt": "You are a listener. Please click on the picture that the speaker describes.",
    "prompt_social": " Also guess whether the speaker was in your original group.",
    "question": "Was the speaker in your original group (from Phase 1)?",
    "yes": "Yes, same group",
    "no": "No, different group",
}


# ── Data ─────────────────────────────────────────────────────


def _check_wording() -> None:
    source = (PROJECT_ROOT / REFGAME).read_text()
    for key, literal in WORDING.items():
        if literal not in source:
            raise ValueError(f"The listener screens quote {literal!r} ({key}), which {REFGAME} no longer contains.")


def data(dirs) -> pd.DataFrame:
    _check_wording()
    trials = read_table(dirs, "trials.csv")
    for col in ("phaseNum", "block"):
        trials[col] = numeric(trials[col])
    messages = read_table(dirs, "messages.csv")
    players = read_table(dirs, "players.csv")
    rows = []
    for screen, condition in SCREENS.items():
        game = EXAMPLE[condition]
        social = condition.startswith("social")
        trio = trials[
            (trials["gameId"] == game) & (trials["phaseNum"] == 2) & (trials["block"] == BLOCK)
            & (trials["target"] == EXAMPLE["target"]) & (trials["currentGroup"] == TRIO)
        ]
        listener = trio[(trio["role"] == "listener") & is_true(trio["inGroupSpeaker"])]
        if len(trio) != 3 or len(listener) != 1:
            raise ValueError(f"The listener screens need the full trio {TRIO} of the example trial in the {condition} game.")
        listener = listener.iloc[0]
        chat = messages[
            (messages["gameId"] == game) & (messages["roundId"] == listener["roundId"]) & (messages["group"] == TRIO)
        ].sort_values("timestamp")
        grid = json.loads(players.loc[players["playerId"] == listener["playerId"], "shuffledTangrams"].item())
        if len(grid) != 16 or len(set(grid)) != 16 or EXAMPLE["target"] not in grid:
            raise ValueError("A listener's grid holds the 16 distinct shapes, the target among them.")

        def add(part, text="", slot=0, label="", target="", polygons="", source=REFGAME):
            rows.append(dict(screen=screen, condition=condition, part=part, slot=slot, label=label,
                             text=text, target=target, polygons=polygons, source=source))

        add("prompt", WORDING["prompt"] + (WORDING["prompt_social"] if social else ""))
        for i, target in enumerate(grid):
            add("stimulus", slot=i, label="target" if target == EXAMPLE["target"] else "", target=target,
                polygons=json.dumps(stimulus_polygons(target)), source="players.csv shuffledTangrams")
        for i, m in enumerate(chat.itertuples()):
            add("message", m.text, slot=i, label="Speaker" if m.senderRole == "speaker" else "Listener",
                source="messages.csv")
        if social:
            add("question", WORDING["question"])
            for i, key in enumerate(("yes", "no")):
                add("answer", WORDING[key], slot=i)
    return pd.DataFrame(rows)


# ── Drawing ──────────────────────────────────────────────────

# The screen is laid out in its own units, W across, y running downward as on
# the screen; the height follows from the panel's proportions.
W = 100.0
H = W * SCREEN_PANEL_AXES[1] / SCREEN_PANEL_AXES[0]
GAME_COLUMN = 64.0  # the chat column takes the rest
CENTER = GAME_COLUMN / 2
CHAR_EM = 0.5  # the average width of a character, in em, for wrapping
LINE = 1.25  # line spacing, in em
GRID_CELL = 8.5
GRID_TOP = 18.0
QUESTION_TOP = GRID_TOP + 4 * GRID_CELL + 3


def _units(points: float) -> float:
    """A length in points as screen units."""
    return points * W / (SCREEN_PANEL_AXES[0] * 72)


def _width(s: str, size: float) -> float:
    """The width of `s` set in `size`-point type, in screen units."""
    return _units(TextToPath().get_text_width_height_descent(s, FontProperties(size=size), ismath=False)[0])


def _wrap(s: str, width: float, size: float) -> str:
    return textwrap.fill(s, max(1, int(width / _units(CHAR_EM * size))))


def _text(ax, x, y, s, size=NOTE_SIZE, ha="left", color=INK, weight="normal", width=None):
    if width is not None:
        s = _wrap(s, width, size)
    ax.text(x, y, s, fontsize=size, ha=ha, va="top", color=color, weight=weight, linespacing=LINE)
    return s.count("\n") + 1


def _box(ax, x, y, w, h, face="white", edge=PANEL_EDGE, linewidth=0.6, radius=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=face, edgecolor=edge, linewidth=linewidth))


def _screen(rows: pd.DataFrame):
    fig, ax = panel_figure(0, 0, 0, 0, axes=SCREEN_PANEL_AXES)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    _box(ax, 0.4, 0.4, W - 0.8, H - 0.8, linewidth=0.8, radius=2)

    # The instruction and the listener's grid, the target outlined.
    _text(ax, CENTER, 4, rows.loc[rows["part"] == "prompt", "text"].item(), ha="center", width=GAME_COLUMN - 8)
    left = CENTER - 2 * GRID_CELL
    for s in rows[rows["part"] == "stimulus"].itertuples():
        x = left + int(s.slot) % 4 * GRID_CELL
        y = GRID_TOP + int(s.slot) // 4 * GRID_CELL
        draw_stimulus(ax, json.loads(s.polygons), (x + GRID_CELL / 2, y + GRID_CELL / 2), 0.78 * GRID_CELL)
        if s.label == "target":
            ax.add_patch(Rectangle((x + 0.4, y + 0.4), GRID_CELL - 0.8, GRID_CELL - 0.8, facecolor="none",
                                   edgecolor=INK, linewidth=1.0))

    # The group question, on the social listener's screen only.
    if (rows["part"] == "question").any():
        lines = _wrap(rows.loc[rows["part"] == "question", "text"].item(), GAME_COLUMN - 16, SMALL_SIZE)
        n = lines.count("\n") + 1
        answer_top = QUESTION_TOP + 2.5 + _units(n * LINE * SMALL_SIZE) + 2
        answer_height = _units(2.4 * NOTE_SIZE)
        answers = rows[rows["part"] == "answer"].sort_values("slot")["text"].tolist()
        widths = [_width(a, NOTE_SIZE) + 4 for a in answers]
        gap = 2.5
        # The box holds the answers side by side with a margin, and the question
        box = max(sum(widths) + gap + 6, GAME_COLUMN - 12)
        _box(ax, CENTER - box / 2, QUESTION_TOP, box, answer_top + answer_height + 2.5 - QUESTION_TOP, edge=INK,
             linewidth=0.8)
        _text(ax, CENTER, QUESTION_TOP + 2.5, lines, size=SMALL_SIZE, ha="center", weight="bold")
        x = CENTER - (sum(widths) + gap) / 2
        for answer, w in zip(answers, widths):
            _box(ax, x, answer_top, w, answer_height, face=PANEL_FACE, edge="none", radius=0.8)
            ax.text(x + w / 2, answer_top + answer_height / 2, answer, fontsize=NOTE_SIZE, ha="center",
                    va="center", color=INK)
            x += w + gap

    # The chat: the trio's messages, each under its sender's role.
    ax.plot([GAME_COLUMN, GAME_COLUMN], [0.4, H - 0.4], color=PANEL_EDGE, linewidth=0.6)
    y = 4.0
    for m in rows[rows["part"] == "message"].sort_values("slot").itertuples():
        _text(ax, GAME_COLUMN + 4, y, m.label, color=MUTED)
        y += _units(LINE * NOTE_SIZE) + 1
        body = _wrap(m.text, W - GAME_COLUMN - 12, NOTE_SIZE)
        height = _units((body.count("\n") + 1) * LINE * NOTE_SIZE) + 3
        _box(ax, GAME_COLUMN + 3, y, W - GAME_COLUMN - 7, height, face=PANEL_FACE, edge="none", radius=1.5)
        _text(ax, GAME_COLUMN + 5, y + 1.5, body)
        y += height + 3
    return fig


def plot(df: pd.DataFrame) -> dict:
    return {screen: _screen(df[df["screen"] == screen]) for screen in SCREENS}
