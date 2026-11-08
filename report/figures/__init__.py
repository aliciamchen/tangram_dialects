"""
The article's figures, one module per figure.

Each figure is a module in this package named for what it shows
(group_specificity_by_window.py, not figure_two.py, because figure numbers
change while the article is written and the manuscript's \\includegraphics
maps the number to the name). A module defines two functions:

    def data(dirs) -> pd.DataFrame
        The table the figure plots, read from the model inputs in
        data/<dataset>/model_inputs/ and the estimates in
        results/estimates/. Nothing is aggregated or re-indexed
        here beyond what the figure needs to draw; an aggregate that a
        model also uses belongs in preprocessing, and a bootstrap interval
        over games comes from report.bootstrap. A module whose panels the
        SI draws again on a subset of games also takes `restriction`, the
        name of a rerun of stats/primary/robustness.R (RESTRICTIONS in
        _common.py); the runner itself always calls data(dirs).

    def plot(df: pd.DataFrame) -> matplotlib.figure.Figure | dict[str, Figure]
        The figure, drawn from that table alone with the palette, labels
        and sizes of report.plot_style and nothing of its own. A figure
        assembled by hand from separate panels returns a dict instead, one
        figure per panel keyed by what the panel shows.

The runner (`uv run python -m report.figures --dataset full`, or `make
figures`) calls both for every module and saves the figure as
results/plots/<name>.pdf, or each panel of a dict as
results/plots/<name>_<key>.pdf. `make sync` copies the PDFs into the
manuscript's project.

Tests: report/test_figures.py
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from report.plot_style import apply_style


def figure_names() -> list[str]:
    """The figure modules in this package, in name order (private modules excluded)."""
    package_dir = Path(__file__).parent
    return sorted(
        info.name for info in pkgutil.iter_modules([str(package_dir)]) if not info.name.startswith("_")
    )


def save_figure(fig, name: str, figures_dir: Path) -> Path:
    """Save a figure as <name>.pdf in figures_dir, and close it."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pdf = figures_dir / f"{name}.pdf"
    # The tight bounding box counts an x label's height but not its width (a y label's width but not its
    # height), so a label longer than its axis is cut unless it is named in full.
    labels = [label for ax in fig.axes for label in (ax.xaxis.label, ax.yaxis.label) if label.get_text()]
    # Trimmed to what the figure draws with a thin pad, for placing in Illustrator, which also covers the
    # glyph overshoot matplotlib's text metrics leave out; no creation date in the PDF, so an unchanged figure
    # is byte-identical from run to run.
    fig.savefig(pdf, metadata={"CreationDate": None}, bbox_inches="tight", pad_inches=0.05,
                bbox_extra_artists=fig.get_default_bbox_extra_artists() + labels)
    plt.close(fig)
    return pdf


def run_figure(module, dirs) -> Path | list[Path]:
    """Build one figure module's table and figure for a dataset and save the figure.

    Returns the PDF written, or the list of PDFs when the module's plot()
    returns a dict of panels. A module that sets FOLDER writes into that
    subfolder of the plots directory (the confirmatory tests).
    """
    name = module.__name__.rsplit(".", 1)[-1]
    df = module.data(dirs)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name}.data() must return a DataFrame, not {type(df).__name__}")
    result = module.plot(df)
    out = Path(dirs.plots) / getattr(module, "FOLDER", "")
    if isinstance(result, dict):
        return [save_figure(fig, f"{name}_{key}", out) for key, fig in result.items()]
    return save_figure(result, name, out)


def run_figures(dirs, names: list[str] | None = None) -> list[Path]:
    """Run every figure module (or the named ones) for a dataset; returns the files written."""
    apply_style()
    selected = names or figure_names()
    unknown = sorted(set(selected) - set(figure_names()))
    if unknown:
        raise SystemExit(f"Error: no figure module named {', '.join(unknown)} in report/figures/")
    written = []
    for name in selected:
        module = importlib.import_module(f"report.figures.{name}")
        pdfs = run_figure(module, dirs)
        for pdf in pdfs if isinstance(pdfs, list) else [pdfs]:
            print(f"Wrote {pdf}")
            written.append(pdf)
    if not written:
        print("No figure modules in report/figures/ yet; nothing written.")
    return written
