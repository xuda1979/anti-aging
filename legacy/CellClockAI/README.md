# CellClockAI

A research framework and reproducible computational pilots for discovering the regulatory mechanisms of cellular age resetting.

**This release does not identify a validated human rejuvenation switch or demonstrate lifespan extension.** It separates executed public-data analysis, explicitly synthetic causal benchmarks, and a prospective large-scale biological study.

## Paper and results

- [Paper PDF](paper/Cellular_Age_Resetting_AI.pdf)
- [Manuscript source](paper/manuscript.md)
- [Public-data results](results/public_summary.json)
- [Synthetic results](results/synthetic_summary.csv)
- [Dataset catalogue](data/datasets.csv)
- [Evidence and limitations](docs/evidence_and_scope.md)

The English manuscript defines the hypotheses, data resources, multimodal model, causal estimands, evaluation splits, controls, power calculation and prospective biological confirmation. It includes 25 references checked against primary publication records, including 2026 literature.

## What was actually run

| Component | Evidence | Scope |
|---|---|---|
| GSE165177 RNA-seq pilot | Public observed data | 34 recovered cultures, three donors, 35,723 aggregated gene features |
| Treatment prediction | Logistic regression | Donor holdout AUROC 1.00; this predicts treatment status, not biological age |
| Label permutations | 100 within-context permutations | Descriptive association test; not a randomized intervention trial |
| Causal benchmark | Synthetic data | Six mechanisms, 20 seeds, 900 training and 600 test observations per setting |
| Predictors | Linear ridge, interaction ridge, neural network | Functional versus manipulable-clock objectives |
| Large multimodal graph model | Optional reference code | No biological training or scientific performance claim |
| Donor power calculation | Prospective numerical design | Not a biological result |

## Reproduce

Python 3.10 or newer is required. Python 3.12 was used for the release. No API keys, hosted LLMs, or private datasets are needed.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/run_all.py --only all --seeds 20 --permutations 100
python scripts/make_figures.py
python scripts/power_analysis.py
python -m unittest discover -s tests -v
```

The public-data run downloads two processed NCBI files, approximately 16 MB combined, and verifies their SHA256 digests. Raw input matrices are excluded from Git. A changed checksum fails explicitly rather than silently analysing a different file. Exact release package versions are recorded in `requirements-lock.txt` and runtime JSON files. Full per-seed results are included; figures can be regenerated without downloading biological data.

For the optional architecture:

```bash
python -m pip install -e '.[deep]'
python scripts/smoke_deep.py
python -m unittest discover -s tests -v
```

The architecture smoke test uses invented tensors only. Its execution status and framework version are recorded in `results/verification.json`. It does not add a biological model result.

## Repository layout

```text
paper/           English manuscript, PDF, LaTeX and rendering source
src/cellclock/   Acquisition, public analysis, simulations and reference model
scripts/         Reproduction, figures, prospective power and manuscript build
data/            Dataset catalogue, provenance and included-sample manifest
results/         Summaries, per-sample predictions, per-seed metrics and logs
figures/         Original generated figures in PNG and vector PDF
tests/           Data-selection, leakage, simulator and optional model checks
docs/            Data schema, evidence scope and literature metadata
configs/         Prospective study defaults
```

## Interpretation

The project tests the possibility of a small control set against distributed or context-dependent control and readout artifacts. Changes in methylation, telomeres or transcription are not assumed to be mere surface phenomena. Their causal roles require intervention and independent functional outcomes. Clock reduction alone is not a validated longevity intervention.

Public association gene lists are labelled `treatment_association_only`. Reg00--Reg11 in simulations are invented names, not human genes. K562 perturbation resources are useful method references but do not establish effects in normal ageing cells. The prospective study requires independent biological collaborators, appropriate oversight and new functional evidence.

## License and attribution

Original code is released under MIT. The original manuscript and original figures are released under CC BY 4.0. Public source datasets retain their original rights and terms. No third-party article text, figure artwork or source matrix is redistributed. See [data attribution](data/README.md) and [license](LICENSE).

Prepared for Da Xu with AI assistance. No journal submission, institutional endorsement, funding award or clinical validation is asserted.
