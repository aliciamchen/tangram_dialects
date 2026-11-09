"""
Tests for the figure runner (report/figures/): module discovery, and that a
figure module's figure is saved under its module name.

Run with:
    uv run pytest report/test_figures.py -v
"""

import importlib
import types

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from report.figures import figure_names, run_figure, save_figure  # noqa: E402


def test_figure_names_lists_only_public_modules():
    names = figure_names()
    assert all(not n.startswith("_") for n in names)
    assert names == sorted(names)


def test_every_figure_module_imports_and_defines_the_contract():
    """Each module in report/figures/ imports without data and defines data() and plot(), with FOLDER a string when set."""
    for name in figure_names():
        module = importlib.import_module(f"report.figures.{name}")
        assert callable(getattr(module, "data", None)), f"{name} defines no data()"
        assert callable(getattr(module, "plot", None)), f"{name} defines no plot()"
        assert isinstance(getattr(module, "FOLDER", ""), str), f"{name}.FOLDER is not a string"


def test_save_figure_writes_the_pdf(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    pdf = save_figure(fig, "example", tmp_path / "figures")
    assert pdf == tmp_path / "figures" / "example.pdf" and pdf.stat().st_size > 0


def test_run_figure_calls_data_then_plot_and_names_the_file_after_the_module(tmp_path):
    module = types.ModuleType("report.figures.block_means")

    def data(dirs):
        return pd.DataFrame({"block": [1, 2], "mean": [0.4, 0.5]})

    def plot(df):
        fig, ax = plt.subplots()
        ax.plot(df["block"], df["mean"])
        return fig

    module.data, module.plot = data, plot
    dirs = types.SimpleNamespace(plots=tmp_path / "figures")
    pdf = run_figure(module, dirs)
    assert pdf.name == "block_means.pdf"


def test_run_figure_saves_each_panel_of_a_dict_under_the_module_name_and_its_key(tmp_path):
    module = types.ModuleType("report.figures.strands")
    module.data = lambda dirs: pd.DataFrame({"x": [1, 2]})

    def plot(df):
        panels = {}
        for key in ("pair1_similarity", "pair1_refer_mixed"):
            fig, ax = plt.subplots()
            ax.plot(df["x"])
            panels[key] = fig
        return panels

    module.plot = plot
    pdfs = run_figure(module, types.SimpleNamespace(plots=tmp_path / "figures"))
    assert [p.name for p in pdfs] == ["strands_pair1_similarity.pdf", "strands_pair1_refer_mixed.pdf"]
    assert all(p.stat().st_size > 0 for p in pdfs)


def test_run_figure_writes_into_the_module_folder(tmp_path):
    module = types.ModuleType("report.figures.confirmatory_h0")
    module.FOLDER = "confirmatory_tests"
    module.data = lambda dirs: pd.DataFrame({"x": [1, 2]})
    module.plot = lambda df: {"columns": plt.figure()}
    pdfs = run_figure(module, types.SimpleNamespace(plots=tmp_path / "figures"))
    assert pdfs == [tmp_path / "figures" / "confirmatory_tests" / "confirmatory_h0_columns.pdf"]


def test_run_figure_refuses_a_module_whose_data_is_not_a_table(tmp_path):
    module = types.ModuleType("report.figures.bad")
    module.data = lambda dirs: [1, 2]
    module.plot = lambda df: plt.figure()
    with pytest.raises(TypeError, match="must return a DataFrame"):
        run_figure(module, types.SimpleNamespace(plots=tmp_path))


def test_save_figure_is_byte_identical_from_run_to_run(tmp_path):
    def make():
        fig, ax = plt.subplots()
        ax.plot([1, 2], [3, 4])
        return fig

    first = save_figure(make(), "same", tmp_path / "a")
    second = save_figure(make(), "same", tmp_path / "b")
    assert first.read_bytes() == second.read_bytes()
