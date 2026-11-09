---
name: build-report
description: Use when compiling, checking, or updating a manuscript in writing/ (main.tex or the cover letter), syncing figures or statistics into it, or diagnosing a LaTeX build with undefined citations, an empty .bbl, or a truncated PDF.
allowed-tools: Bash, Read, Grep, Glob
---

# Build the manuscript

Manuscripts live under `writing/`, one directory per Overleaf project, gitignored and mirrored to Overleaf through Dropbox, so edits there are never committed to this repository. `writing/preregistration/` is the preregistration, written as a Stage 1 registered report and frozen since 2026-09-21: never change its text, tense or sections. `writing/manuscript/` is the full-sample article: `main.tex`, `figure_captions.tex`, `references.bib`, a `latexmkrc` naming `main.tex`, `macros/` (filled by `make sync` from `results/macros/`) and `figures/` (the Illustrator exports in `figures/paper_files/`, copied by `make sync`). The author writes the article's prose; agent work there is the scaffolding (structure, numbers through the macros, verified citations, legends as raw material, checks and builds). The figure panels are drawn by the modules in `report/figures/` under `.claude/rules/figures.md` and assembled by hand in Illustrator; the panels in `results/plots/` never go into the project. Framing and venue are the author's decisions; offer options with a recommendation.

1. **Sync inputs.** `make sync` copies `results/macros/*.tex` and the exports in `figures/paper_files/` into the article's `macros/` and `figures/`; it only ever writes to `writing/manuscript/`, never to the frozen preregistration. If the models or the report layer were rerun, confirm the macro files carry the new values (`git diff results/macros/`); every such file has a `% AUTO-GENERATED` header naming its writer and the git commit, and must not be hand-edited. The preregistration reads `writing/preregistration/stats/pilot.tex` and `llm.tex`, which are frozen with it; the notebooks that wrote them are at the git tag `pilot-archive`. The article reads `writing/manuscript/macros/`, one file per family (`checks`, `primary`, `secondary` from `report/inferential.py`; `overview`, `exploratory`, `survey` from `report/descriptives.py`) with macro names prefixed by file; those hold real values only from a run on `DATASET=full`. Check that every macro used in the manuscript is defined:

   ```bash
   cd writing/manuscript && grep -oh '\\[a-zA-Z]*' main.tex figure_captions.tex | sort -u > /tmp/used.txt \
     && grep -oh 'newcommand{\\[a-zA-Z]*' macros/*.tex | sed 's/newcommand{//' | sort -u > /tmp/defined.txt \
     && comm -13 /tmp/used.txt /tmp/defined.txt | wc -l   # macros defined and not yet used
   ```

2. **Build**: `cd writing/manuscript && latexmk -pdf main.tex` (the preregistration and its cover letter build the same way from their own directories).

3. **Recognize a Dropbox race.** If every citation is undefined, `main.bbl` is empty, `.aux` files contain NUL bytes, or the PDF is suddenly short, Dropbox has raced the build. Do not debug the bibliography. Copy the project directory to a directory outside Dropbox (the scratchpad works), run `latexmk -C` there, remove the stale biber PAR cache (`find /var/folders -maxdepth 4 -type d -name 'par-*' -user "$(whoami)" 2>/dev/null`, then `rm -r` that directory), rebuild, and copy back only the PDF.

4. **Check the result**: page count against the previous build, `grep -c 'Warning--' main.blg` for bibliography warnings, and `grep -n 'undefined' main.log` for references. Report warnings; never resolve them by deleting content.
