# Manuscript build

The canonical editable manuscript is `manuscript.md`. The PDF uses vector figures, a single-column scientific layout and clickable DOI references. `Cellular_Age_Resetting_AI.tex` is the generated, editable LaTeX version. `style.tex` contains print formatting.

Install Pandoc and a LaTeX distribution with pdfLaTeX, then run:

```bash
python scripts/make_figures.py
python scripts/build_paper.py
```

Run commands from the repository root. Scientific text is edited in the Markdown source; the synthetic numerical table is refreshed from the saved results during the build. Temporary compilation files are excluded from Git.
