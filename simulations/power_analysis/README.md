# Power analysis

The sample size was justified before data collection by simulating the study's main contrast from the six-player, full-feedback games of Boyce et al. (2024). `power_analysis_setup.Qmd` (Python) downloads those games, builds the two conditions from them, whole six-player games standing in for a mixed condition and two trios drawn from different games for a separated one, embeds the descriptions with the same sentence-transformer model the pipeline uses, and writes the pairwise similarity tables `pairwise_mixed.csv` and `pairwise_separate.csv` beside it. `fit_models.R` then fits the per-game group-specificity model to each table and writes `results_mixed.csv` and `results_separate.csv`, and `power_analysis_plots.Qmd` (R) reads both and draws the power curves. The intermediate tables are committed, so the plots render without the setup step. `test_power_analysis.py` checks the subgroup construction.

```bash
quarto render simulations/power_analysis/power_analysis_setup.Qmd   # simulate the data (downloads the Boyce et al. tables)
Rscript simulations/power_analysis/fit_models.R                     # fit the per-game models
quarto render simulations/power_analysis/power_analysis_plots.Qmd   # plot the power curves
```

The rendered documents go to `_output/`, which is not committed. The notebooks predate the pipeline and are the record of the preregistered justification rather than a step of `make`.
