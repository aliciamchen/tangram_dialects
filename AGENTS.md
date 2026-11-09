# Agent guide

Durable context for coding agents working in this repository. `CLAUDE.md` is a symlink to this file; keep it agent-neutral, because Codex and other tools read it too. `README.md` is the public overview and run instructions for reviewers, `make help` lists the pipeline commands, `.claude/rules/` holds area facts that load when you work in that area, and `.claude/skills/` holds procedures. `full-sample-todos.md` is the author's log: a Now section at the top, a state summary of about ten bullets edited in place, then the record, dated sections oldest first, one per piece of work. `decisions.md` is the index of the decisions taken, one dated entry each with the choice, the alternative, the reason, where it is applied and the record section that holds the detail, grouped by area with the open questions last. `reviews/` holds the audit notes, the builders of the quick-look and mockup pages (`reviews/atlas-build/<topic>/`), meeting slides, backups and patches. Both are gitignored and must not be referenced from public docs. `explore/` is the author's private workspace for analyses the preregistration does not name, gitignored and a local git repository of its own, with its own `CLAUDE.md`: exploratory work for a later paper goes there, and nothing from it enters `stats/`, `report/`, `results/` or a manuscript unless the author asks for it. A comparison the preregistration does not name that the author asks the article to report is a script in `stats/exploratory/`, which writes estimates for the figures and, so far, no macros.

## Status

A nine-player Empirica reference game for studying lexical variation and social signaling: three groups of three describe tangrams, and four between-subjects conditions (`refer_separated`, `refer_mixed`, `social_mixed`, `social_first`) vary whether the groups are mixed in Phase 2 and whether players are rewarded for identifying each other's group. `writing/preregistration/main.tex` is the frozen preregistration. The full sample, dataset `full`, is collected and frozen: 120 started games in 31 sessions between 2026-09-22 and 2026-10-08, 63 of them intact, reproducing byte for byte from the raw exports, with two participants excluded for automated assistance (`data/full/participant_exclusions.csv`). No production server exists. The work now is the article in `writing/manuscript/` and the public release in the RefBank format; where that work stands, what is under review and what comes next are in the Now section of `full-sample-todos.md`, not here.

## Commands

```bash
uv sync && uv run python <script>     # Python; never bare python or pip
make help                             # data, test, estimates, report, sync, and the finer steps underneath (extract, combine, process, measures, model-inputs, bayes-factors, macros, figures, smoke, payments)
make test DATASET=full                # integrity suite plus the Python and R unit tests
make estimates && make report             # fit the analysis scripts that are out of date, then write the macros and figures into results/
cd experiment && npm run test:unit    # server unit tests; Playwright only through the npm scripts
```

## Hard rules

- Never read or print `.env` or `experiment/.empirica/empirica.toml`, never print a participant (Prolific) id, and never put a hostname, organization name or token in a committed file. `.env.example` documents the variables.
- `experiment/data/<timestamp>/` holds the raw exports and is read-only input: never edit, move or delete anything there. `data/full/`, `backups/`, `data/runs/` and `data/payments/` stay local until the RefBank release; `results/` (estimates, macros, plots) is committed and is the regression check for any analysis change. The pilot dataset, its SI notebook and the LLM Phase 1 benchmark were retired from the working tree and remain at the git tag `pilot-archive`.
- `writing/` is gitignored, one Overleaf project per directory, synced through Dropbox. The preregistration is frozen: do not change its text, models or figures. Statistics reach a manuscript only through the generated `\newcommand` macros in `results/macros/`, copied in by `make sync`, never typed in.
- In an interactive session, commit only when asked, on `main`, and do not push: the repository is public and the author publishes. Work the author will not watch (a cloud or background session, a worktree, or a self-contained piece such as the RefBank converter) goes on a branch named `agent/<topic>`: commit freely there, push it, and open a draft pull request for the author to review and merge. Never push to `main`, force-push or merge a pull request; a hook blocks these. Before committing, say what is uncommitted and leave out hunks you did not make: the author edits by hand and runs parallel sessions, so re-read a file before editing it and never revert an edit that is not yours.
- A worktree or a cloud clone has none of the gitignored inputs (`.env` and `empirica.toml` are copied in through `.worktreeinclude`; the raw exports and `data/full/` are not), so data processing, renders and manuscript work stay in the main checkout.
- The message filter spends Vertex AI credit: state the number of calls and get a yes first.
- The experiment code is the record of what ran. Change it for tests, documentation or a future deployment only, and never edit a test to make it pass or without saying so.
- Hooks enforce the sharp edges (`.claude/settings.json`, mirrored in `.codex/hooks.json`): recursive deletes are blocked, the pre-commit check scans for identifiers and runs the test suites, and pushes to main, force pushes and pull-request merges are denied 

## Working with the author

- Present decisions as a numbered list, each with the recommended option and the reason, and mark what is optional; the author answers by number.
- Verify at the source before stating how something behaves or how many of something there are: Empirica from the installed `@empirica/core` (treat Empirica as unmaintained), Prolific from its records, counts from the exports. Findings relayed from another session are leads, not facts, and the current data beat anything remembered from an earlier session. Say which statements are inferences.
- When you find one error, search the session's work for others of the same kind before reporting.
- Explain mechanisms in plain language when asked why. The maintainers are researchers, not software engineers, and "I don't understand" means simplify, not elaborate.
- Report every change beyond the one asked for. When the author defers something, record it in `full-sample-todos.md` and move on. After a task, say what comes next.
- Progress, open questions and what comes next go in the Now section of `full-sample-todos.md`, edited in place so that its bullets stay true, never by adding a dated bullet; a session's account of what it did, what it checked and what it left open goes at the end of the record as a dated section. A decision the author takes is appended to its group in `decisions.md` in the same session, and a decision found recorded anywhere else (the record, a review note, a comment) is moved into `decisions.md` with a pointer back, so that the log stays the one index. This guide carries only what stays true across sessions, so a change in the project's state is not a reason to edit it.
- A figure idea is shown first as a quick look drawn from `data/full` and the committed estimates on an artifact page (builder in `reviews/atlas-build/<topic>/`), the candidates numbered; the author picks by number, and a chosen one becomes a module under the figure rule. Nothing from a quick look enters `results/` or a manuscript on its own.
- At the end of a session that changed code, say what is uncommitted and, when it spans several areas, propose commits grouped by area: `results/` is a regression check only against a committed baseline.
- The preregistered models are frozen: do not re-audit random effects, priors or inference policies, and do not propose refinements. The author wants simple, interpretable models that are not wrong; the accepted and declined suggestions from the revision are in `reviews/prereg-revision-decisions.md`, and the implementation and reporting decisions since are in `decisions.md`.
- Estimate money (participant pay, API calls) from actual spend with fees included, and default to the minimal scope.
- For long runs (Playwright, renders, Bayes factors) give a time estimate, pipe output through `tee` to a stable file, launch anything over ten minutes detached with `nohup`, and propose work to do meanwhile.

## Style

- Sentence case for headings and titles; US spelling in prose (library API names keep theirs); the native R pipe `|>`; no digits in LaTeX macro names.
- Comments, docstrings and docs describe current behavior only, never history or before-and-after comparisons, which belong in the record of `full-sample-todos.md`, with the choice itself in `decisions.md`. Public docs (`README.md`, `data/README.md`, `experiment/README.md`) are natural prose written for reviewers and the author's future self; they describe what the repository is and how to run it, never the project's state, plans or open tasks, which belong in the Now section of `full-sample-todos.md`.
- Commit messages follow conventional commits, `type(scope): subject`, lower case, imperative, no period, with a body only when the why is not in the diff.
- Manuscript text is short and conceptual, with no statistical minutiae and no new SI sections unless asked; apply agreed edits in the `.tex`.

## Sources of truth

- The article's figures are the Illustrator exports in `figures/paper_files/`, which the author assembles from the single-panel PDFs the figure modules write to `results/plots/`; `make sync` copies the macros and the exports into the manuscript's project, and the panels never go there. The conventions are in `.claude/rules/figures.md`.
- The per-dataset inputs (`runs.txt`, `exclude_games.txt`, `participant_exclusions.csv`, `integrity_exceptions.csv`, `payment_adjustments.csv`) are defined in `.claude/rules/data-pipeline.md`.
- Fetch current library documentation with Context7 for R packages, Python libraries, Empirica and Playwright instead of working from memory.

## Rules and skills

A rule holds what is true across the files in its area; what one file does, and why, goes in that file's docstring or the comments on its constants.

| Rule | Covers |
|------|--------|
| `.claude/rules/architecture.md` | the experiment: server callbacks, client, identity masking, exit routing |
| `.claude/rules/testing.md` | the Playwright suite and server unit tests |
| `.claude/rules/sentry.md` | error monitoring in the client |
| `.claude/rules/data-pipeline.md` | datasets, pipeline steps, the analysis scripts and the report layer, the definitions that are easy to get wrong |
| `.claude/rules/figures.md` | the article's figures: the module contract, the look, panels assembled by hand and the settled conventions |
| `.claude/rules/message-filter.md` | the LLM message filter |
| `.claude/rules/operations.md` | payments, blocklist and the other Prolific tooling |
| `.claude/rules/python-env.md` | uv and the Python environment |

Skills are in `.claude/skills/<name>/SKILL.md`, with a symlink per skill in `.agents/skills/` for Codex: `run-playwright-tests`, `rebuild-dataset`, `run-analysis` (the model scripts and the results) and `build-report`. An agent whose harness does not load rules by path should read the rule before changing files in its area.
