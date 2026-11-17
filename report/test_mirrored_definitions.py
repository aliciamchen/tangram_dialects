"""
The definitions that R, Python and the Makefile each write out, held to one
another and to the experiment.

The design is fixed, so each layer states the conditions, the blocks and the
dataset layout where it uses them, which keeps them readable in place: the
experiment in experiment/shared/constants.js, the models in stats/config.R and
stats/R/, the pipeline in preprocessing/, the figures in report/ and the
Makefile for itself. Nothing reads a shared file, so nothing but these tests
would notice one copy changing. They read every copy and fail when two
disagree.

Run with:
    uv run pytest report/test_mirrored_definitions.py -v
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from preprocessing import preprocessing
from preprocessing.dataset_paths import DEFAULT_DATASET, PROJECT_ROOT, DatasetDirs

CONSTANTS_JS = (PROJECT_ROOT / "experiment" / "shared" / "constants.js").read_text()
CONFIG_R = (PROJECT_ROOT / "stats" / "config.R").read_text()
CONTRASTS_R = (PROJECT_ROOT / "stats" / "R" / "contrasts.R").read_text()
CONVERGENCE_R = (PROJECT_ROOT / "stats" / "R" / "convergence.R").read_text()


# ── Reading a definition out of its source ───────────────────


def r_strings(source: str, name: str) -> list[str]:
    """The strings of `NAME <- c("a", "b")` in an R file."""
    match = re.search(rf"^{name}\s*<-\s*c\((.*?)\)", source, re.M | re.S)
    assert match, f"no `{name} <- c(...)` found"
    return re.findall(r'"([^"]+)"', match.group(1))


def r_pairs(source: str, name: str) -> dict[str, tuple[str, ...]]:
    """The named pairs of `NAME <- list(a = c("x", "y"), ...)` in an R file."""
    match = re.search(rf"^{name}\s*<-\s*list\((.*?)\n\)", source, re.M | re.S)
    assert match, f"no `{name} <- list(...)` found"
    pairs = re.findall(r"(\w+)\s*=\s*c\(([^)]*)\)", match.group(1))
    return {key: tuple(re.findall(r'"([^"]+)"', values)) for key, values in pairs}


def r_integer(source: str, name: str) -> int:
    match = re.search(rf"^{name}\s*<-\s*(\d+)L?\s*$", source, re.M)
    assert match, f"no `{name} <- <integer>` found"
    return int(match.group(1))


def r_range(source: str, name: str) -> range:
    match = re.search(rf"^{name}\s*<-\s*(\d+):(\d+)\s*$", source, re.M)
    assert match, f"no `{name} <- a:b` found"
    return range(int(match.group(1)), int(match.group(2)) + 1)


def js_strings(name: str) -> list[str]:
    """The strings of `export const NAME = ["a", "b"];` in constants.js."""
    match = re.search(rf"^export const {name}\s*=\s*\[(.*?)\]", CONSTANTS_JS, re.M | re.S)
    assert match, f"no `export const {name} = [...]` in constants.js"
    return re.findall(r'"([^"]+)"', match.group(1))


def js_number(name: str) -> int:
    """`export const NAME = 3;`, or the production value of `NAME = TEST_MODE ? 3 : 6;`."""
    match = re.search(rf"^export const {name}\s*=\s*(?:TEST_MODE\s*\?\s*\d+\s*:\s*)?(\d+)\s*;", CONSTANTS_JS, re.M)
    assert match, f"no numeric `export const {name}` in constants.js"
    return int(match.group(1))


def js_condition_test(function: str) -> list[str]:
    """The conditions a helper such as hasSocialGuessing() names."""
    match = re.search(rf"function {function}\(condition\)\s*\{{\s*return \[(.*?)\]\.includes", CONSTANTS_JS, re.S)
    assert match, f"no {function}() in constants.js"
    return re.findall(r'"([^"]+)"', match.group(1))


# ── The conditions ───────────────────────────────────────────


def test_the_four_conditions_are_the_experiments_in_one_order():
    from report.plot_style import CONDITION_COLORS, CONDITION_LABELS, CONDITION_MARKERS, CONDITION_ORDER, DRAW_ORDER

    conditions = js_strings("conditions")
    assert r_strings(CONFIG_R, "CONDITION_ORDER") == conditions
    assert CONDITION_ORDER == conditions
    for other in (DRAW_ORDER, CONDITION_LABELS, CONDITION_COLORS, CONDITION_MARKERS):
        assert set(other) == set(conditions)


def test_the_social_conditions_are_those_with_the_social_guess():
    from report.figures._common import SOCIAL_CONDITIONS

    social = js_condition_test("hasSocialGuessing")
    assert r_strings(CONFIG_R, "SOCIAL_CONDITIONS") == social
    assert list(preprocessing.SOCIAL_GUESSING_CONDITIONS) == social
    assert SOCIAL_CONDITIONS == social


def test_the_h1_h2_conditions_and_contrasts_agree_between_the_model_and_its_figure():
    from report.figures import confirmatory_h1_h2

    assert confirmatory_h1_h2.CONDITIONS == r_strings(CONFIG_R, "H12_CONDITIONS")
    figure_pairs = {term: pair for term, (_, pair) in confirmatory_h1_h2.CONTRASTS.items()}
    assert figure_pairs == r_pairs(CONTRASTS_R, "H12_CONTRASTS")


def test_the_h3a_contrast_is_between_the_two_social_conditions():
    assert r_pairs(CONTRASTS_R, "H3A_CONTRASTS") == {"H3a_socialfirst_gt_socialmixed": ("social_first", "social_mixed")}


# ── The blocks and the groups ────────────────────────────────


def test_the_block_counts_are_the_experiments_production_values():
    from report.plot_style import GAME_BLOCKS, PHASE1_BLOCKS, PHASE2_BLOCKS

    assert PHASE1_BLOCKS == js_number("PHASE_1_BLOCKS")
    assert PHASE2_BLOCKS == js_number("PHASE_2_BLOCKS")
    assert GAME_BLOCKS == PHASE1_BLOCKS + PHASE2_BLOCKS


def test_the_h3b_blocks_agree_between_the_model_and_its_figure():
    from report.figures import confirmatory_h3

    assert list(confirmatory_h3.RATE_BLOCKS) == list(r_range(CONVERGENCE_R, "H3B_BLOCKS"))
    assert confirmatory_h3.EARLY_BLOCK == r_integer(CONVERGENCE_R, "H3B_BLOCK")


def test_the_groups_are_the_experiments():
    from report.descriptives import MIN_GROUP_SIZE
    from report.plot_style import GROUP_ORDER

    assert GROUP_ORDER == js_strings("GROUP_NAMES")
    assert preprocessing.GROUP_SIZE == js_number("GROUP_SIZE")
    assert preprocessing.FULL_ROSTER == js_number("GROUP_SIZE") * len(js_strings("GROUP_NAMES"))
    assert MIN_GROUP_SIZE == js_number("MIN_GROUP_SIZE")


# ── The dataset layout ───────────────────────────────────────

MAKE_VARIABLES = {
    "data": "DATA_DIR",
    "tables": "TABLES_DIR",
    "measures": "MEASURES_DIR",
    "model_inputs": "MODEL_INPUTS_DIR",
    "results": "RESULTS_DIR",
    "estimates": "ESTIMATES_DIR",
    "macros": "MACROS_DIR",
}


def python_layout(name: str) -> dict[str, str]:
    dirs = DatasetDirs(name)
    paths = {"cache": dirs.cache} | {key: getattr(dirs, key) for key in MAKE_VARIABLES}
    return {key: str(path.relative_to(PROJECT_ROOT)) for key, path in paths.items()}


def r_layout(name: str) -> dict[str, str]:
    """The directories use_dataset() in stats/config.R sets for a dataset name."""
    assert re.search(r'^DEFAULT_DATASET\s*<-\s*"(\w+)"', CONFIG_R, re.M).group(1) == DEFAULT_DATASET
    assert 'dataset_dir <<- here("data", name)' in CONFIG_R
    results = re.search(
        r'results_dir <<- if \(name == DEFAULT_DATASET\) \{\s*here\("(\w+)"\)\s*\} else \{\s*file\.path\(dataset_dir, "(\w+)"\)\s*\}',
        CONFIG_R,
    )
    assert results, "use_dataset() no longer sets results_dir as this test reads it"
    layout = {"data": f"data/{name}"}
    layout["results"] = results.group(1) if name == DEFAULT_DATASET else f"{layout['data']}/{results.group(2)}"
    parents = {"dataset": layout["data"], "results": layout["results"]}
    for child, parent, folder in re.findall(r'(\w+)_dir <<- file\.path\((\w+)_dir, "(\w+)"\)', CONFIG_R):
        layout[child] = f"{parents[parent]}/{folder}"
    return layout


def make_layout(name: str, tmp_path) -> dict[str, str]:
    """The Makefile's directories for a dataset name, as Make expands them."""
    printer = tmp_path / "print-layout.mk"
    lines = "".join(f"\t@echo {key}=$({variable})\n" for key, variable in MAKE_VARIABLES.items())
    printer.write_text(f"print-layout:\n{lines}")
    out = subprocess.run(
        ["make", "-s", "-f", "Makefile", "-f", str(printer), "print-layout", f"DATASET={name}"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


@pytest.mark.parametrize("name", [DEFAULT_DATASET, "smoke"])
def test_python_r_and_the_makefile_put_a_dataset_in_the_same_places(name, tmp_path):
    python = python_layout(name)
    r = r_layout(name)
    assert {key: python[key] for key in r} == r, "stats/config.R and preprocessing/dataset_paths.py disagree"
    assert set(r) >= {"data", "tables", "measures", "model_inputs", "cache", "results", "estimates"}
    if shutil.which("make") is None:
        pytest.skip("make is not installed")
    assert make_layout(name, tmp_path) == {key: python[key] for key in MAKE_VARIABLES}, (
        "the Makefile and preprocessing/dataset_paths.py disagree"
    )
