"""Build the scientific PDF and editable LaTeX source using Pandoc and pdfLaTeX."""
import re
import shutil
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD = PAPER / "build"
BUILD.mkdir(exist_ok=True)

summary = pd.read_csv(ROOT / "results/synthetic_summary.csv")
labels = {"additive": "Additive", "synergy": "Synergy", "redundant": "Redundant",
          "stress_interaction": "Stress dependent", "unseen_synergy": "Unseen synergy",
          "no_causal_effect": "No causal effect"}
rows = ["| Scenario | Linear functional | Interaction functional | Neural functional | Clock trained |",
        "|---|---:|---:|---:|---:|"]
for scenario, label in labels.items():
    sub = summary[summary.scenario == scenario].set_index("model")
    cells = [f'{sub.loc[m, "functional_mean_rmse_mean"]:.3f} ± {sub.loc[m, "functional_mean_rmse_std"]:.3f}'
             for m in ["linear_function", "interaction_function", "neural_function", "interaction_clock"]]
    rows.append("| " + label + " | " + " | ".join(cells) + " |")
source_path = PAPER / "manuscript.md"
source = source_path.read_text()
source = re.sub(r"\| Scenario \| Linear functional.*?(?=\n\nTable 3)", "\n".join(rows), source, flags=re.S)
if "{{" in source:
    raise ValueError("Unresolved manuscript placeholder")
source_path.write_text(source)
# Preserve PNG links in the GitHub manuscript but use vector figures in print.
(BUILD / "manuscript_render.md").write_text(source.replace(".png){width", ".pdf){width"))
tex = "Cellular_Age_Resetting_AI.tex"
cmd = ["pandoc", "build/manuscript_render.md", "-s", "--from=markdown+tex_math_dollars",
       "--resource-path=.", "-V", "documentclass=article", "-V", "fontsize=11pt",
       "-V", "papersize=letter", "-V", "geometry:margin=0.85in", "-V", "fontfamily=mathptmx",
       "--include-in-header=style.tex", "-o", tex]
subprocess.run(cmd, cwd=PAPER, check=True)
for _ in range(2):
    p = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "-output-directory=build", tex], cwd=PAPER, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stdout[-6000:] + p.stderr)
pdf = PAPER / "Cellular_Age_Resetting_AI.pdf"
shutil.copy2(BUILD / pdf.name, pdf)
print(f"Built {pdf} ({pdf.stat().st_size:,} bytes)")
