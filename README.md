# Testing the roles of passive isolation and active social motivation in lexical variation using experimental microsocieties

This repository holds the experiment, the data pipeline and the analyses for a preregistered study of how linguistic conventions diverge between groups. Nine participants play a tangram reference game in three groups of three. In Phase 1 each group describes and identifies the same tangrams among themselves. In Phase 2 the groups are either kept apart (`refer_separated`), mixed so that listeners hear speakers from other groups (`refer_mixed`), mixed and rewarded for identifying whether a speaker came from their own group (`social_mixed`), or told about that reward before the game begins (`social_first`). The preregistration is a Stage 1 manuscript and is frozen.

The full sample, 120 started games collected in 31 Prolific sessions between 2026-09-22 and 2026-10-08, is not in the repository: the processed data exist only in the author's checkout, so the analyses can be read here but not run from a clone. The results are: every estimates table, macro file and figure under `results/` is committed, and each records the git commit of the code that produced it. The pilot the preregistration reports on and the LLM benchmark behind its check for AI-assisted play are at the git tag `pilot-archive`.

## How the pieces fit together

The work runs in one direction, from the game to the article, and each step has its own directory and its own `make` verb:

```
experiment/   the Empirica game, which writes an export zip per session
     │   make data        preprocessing/ (Python) turns the exports into one dataset
     ▼
data/full/    the dataset: hand-written inputs at the top, generated files one level down
     │   make test        the integrity suite checks the dataset against the design and the raw tables
     │   make estimates   stats/ (R) fits one script per analysis and writes its estimates
     ▼
results/      the estimates tables, and from them
     │   make report      report/ (Python) writes the macros and draws the figures
     │   make sync        copies the macros and figures into the Overleaf project
     ▼
writing/manuscript/   the article reads the macros and the figures
```

The analyses are R scripts in `stats/`, one per preregistered analysis, grouped as the preregistration groups them (`checks/`, `primary/`, `secondary/`, and `exploratory/` for a comparison the article reports that the preregistration does not name). Each fits its models, prints them and writes what it estimated as a tidy table to `results/estimates/`; a script that builds on another's estimates reads them from there rather than refitting. `report/` then formats the results: two writers turn the estimates and the processed tables into the `\newcommand` macros the manuscript reads, so that no statistic is ever typed in, and one module per figure in `report/figures/` draws the panels that are assembled into the article's figures in Illustrator.

Each part has its own documentation. [`data/README.md`](data/README.md) describes the layout of a dataset and every column of its tables. The header of `stats/config.R` describes how the analysis scripts are built, and the docstring of `report/figures/__init__.py` the contract a figure module follows. [`experiment/README.md`](experiment/README.md) covers running, testing and deploying the game, [`operations/procedures.md`](operations/procedures.md) how the Prolific sessions were run, and [`simulations/power_analysis/README.md`](simulations/power_analysis/README.md) the sample-size justification. `make help` lists every target.

## Setup

Node 20 and [Empirica](https://empirica.ly/) run the experiment. Python 3.11 or later with [uv](https://docs.astral.sh/uv/) and R 4.5 with renv run the analyses.

```bash
curl -fsS https://install.empirica.dev | sh   # Empirica
cd experiment && npm install && cd ..          # the game and its Playwright tests
uv sync                                        # Python
Rscript -e 'renv::restore()'                   # R
```

The analyses need no credentials. Running or deploying the experiment and the Prolific tooling read a `.env` file at the repository root, created with `cp .env.example .env` and filled in, and running the game locally needs `experiment/.empirica/empirica.toml`, created from its example with fresh random values; both are gitignored because they hold a hostname, an organization name and tokens. Contributors should also run `git config core.hooksPath .githooks`: the pre-commit hook refuses a commit that stages a participant identifier and runs the tests for whatever the commit touches. GitHub Actions runs the server unit tests and the Python unit tests on every push.

## Reproducing the results

```bash
make test                             # the integrity suite on the dataset, the Python unit tests and the R tests
make model-inputs estimates report    # from the processed tables: the model inputs, every estimates table, the macros and the figures
```

The results reproduce from the processed tables alone. With a dataset's `tables/`, `classifier/` and `measures/` directories under `data/full/`, the second command rebuilds everything downstream into `results/`, to be compared with the committed versions. Only the model fitting needs R; `make report` on its own, which writes the macros and draws the figures from the committed estimates, needs just the Python environment. The steps before the tables need the raw exports under `experiment/data/`, which identify participants and are not committed, and Vertex AI credit for the message filter; `make measures`, which recomputes the similarity tables and description properties, needs the sentence-transformer model and takes a few minutes. The scripts take `--dataset <name>` and `make` takes `DATASET=<name>`, both defaulting to `full`; any other dataset keeps its results inside its own directory, so a trial run never touches the committed ones.
