# Recovered CellClockAI project and fresh reproduction

Current project authors: **David Xu, PhD; Peter Lee, PhD, MD; and Zhimin Zheng, PhD**.
Reproduction completed **21 October 2026**. Historical source and attribution remain unchanged.

The original `CellClockAI_Paper_Code_Results.zip` was recovered and its scientific contents were restored under [`CellClockAI/`](CellClockAI/). All **59 files (1,416,422 uncompressed bytes)** preserve their original bytes, including the historical manuscript, figures, code, outputs, license and citation. The original 58-entry `MANIFEST.sha256` verifies completely. A new file-by-file source manifest is in [`reproduction/archive_manifest.json`](reproduction/archive_manifest.json).

Archive SHA-256: `8e0b99ec397c59b3ea58fe9f64e19e6aa8fed3dfb41f0eca726742b4761d2ace` (1,095,386 bytes).

The historical license explicitly permits reuse: original code is MIT; original manuscript and figures are CC BY 4.0; public source datasets retain their own rights. Original copyright and historical `Da Xu` attribution were preserved rather than retrospectively rewritten. The archived prospective study differs from the current illustrative business plan and does not replace it. The nested `.github/workflows/ci.yml` is retained as an inert historical file and is not activated as a repository-root workflow. Review found no credential, private-key, token or email-address patterns and no private correspondence. The acquisition code only fetches two fixed, hash-verified public NCBI matrices.

## What the fresh replay established

| Component | Fresh result | Interpretation |
|---|---|---|
| Public GSE165177 recovered fibroblast pilot | 34 cultures, three donor labels, 35,723 aggregated gene features | Observed public-data treatment association |
| Leave-one-donor-out classifier | AUROC 1.00; balanced accuracy 1.00 | Separates treatment labels; not biological age |
| Leave-one-experiment-out classifier | AUROC 1.00; balanced accuracy 1.00 | Reproduced robustness calculation; donors are shared across experiment folds |
| Remove selected cycle/pluripotency markers | AUROC 1.00 after removing 19 features | Incomplete adjustment; does not exclude all cell-state confounding |
| Within-context label permutations | 100; mean null AUROC 0.5376923; reported p=0.009901 | Descriptive under exchangeability assumptions; not a randomized efficacy trial |
| Synthetic benchmark | Six scenarios × 20 seeds; 480 model fits, 900 train/600 test observations per setting | Invented variables and simulator outcomes only |
| Prospective donor power illustration | Numerical output reproduced | A design calculation, not measured biology |
| Original unit tests | Six passed, one optional Torch test skipped | Optional graph architecture unavailable in this runtime |
| Optional graph smoke | Skipped; Torch unavailable; no installation | Historical smoke claim was not newly verified |

All **11 substantive result files** compared with the archive are **byte-identical**, including the public predictions and permutation null, treatment-association and marker tables, synthetic per-run and ranking outputs, aggregate summaries and prospective power table. New results were generated in a separate scratch copy; the archived outputs were not overwritten. Fresh evidence is in [`reproduction/outputs/`](reproduction/outputs/), [`comparison.json`](reproduction/comparison.json) and [`reproduction_manifest.json`](reproduction/reproduction_manifest.json). Independent review is recorded separately under [`verification/`](verification/).

The observed treatment separation may reflect treatment-associated cell state, selection, proliferation, identity or other correlated changes. Only three donor labels are represented, and cultures within a donor are not independent donors. The public pilot has no prospective gene-specific intervention, rescue, independent functional reversal, safety or lifespan test. Its association gene list is explicitly **not a validated target list**.

In the synthetic examples, clock-trained policies selected the invented clock-only decoy in every setting. Interaction models worked well when the relevant pairs were observed but failed on the deliberately unobserved synergy setting. These are useful software falsification examples; their causal truth was supplied by the simulator and is not evidence about human regulators.

## Exact execution and environment

A separate working copy was used with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `NUMEXPR_NUM_THREADS=1`. The available environment exactly matched the archived requirements: NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0 and Matplotlib 3.10.8. Python and platform details are recorded in the fresh manifest. No dependencies were installed.

Commands executed from that separate copy:

```bash
python -m unittest discover -s tests -v
python scripts/run_all.py --only all --seeds 20 --permutations 100
python scripts/power_analysis.py
```

The scientific replay, including two public processed downloads totalling **16,052,256 bytes**, took **107.39 seconds**. The test stage took 4.13 seconds including process startup; the prospective power calculation took 2.90 seconds. Direct cloud spending was **$0** and no GPU was used. Downloaded source matrices remain outside the public repository; hashes and exact public URLs are recorded for reproduction.

The historical downloader writes a hardcoded `accessed: 2026-10-13` provenance field. This original source was preserved. Fresh UTC execution records and the new reproduction manifest establish the actual **21 October 2026** replay; the historical field is not presented as its acquisition date.

For another replay, use the included wrapper to copy the archive into a temporary directory and write fresh evidence to a new destination:

```bash
python legacy/reproduction/reproduce.py --output /tmp/cellclock-reproduction-new
```

A pre-existing output destination is rejected to avoid overwriting evidence. The wrapper runs the original code and never edits the archive. It uses the current Python environment; install the pinned historical requirements into an isolated environment first if needed. Optional Torch absence remains a recorded skip.

## Claims still unsupported or not rerun

This reproduction does not establish biological rejuvenation, a complete reset of cellular aging, a causal human mechanism, a proprietary target, lifespan extension, clinical efficacy, institutional support or committed founder roles. The optional graph model has not been trained on the biological data. Its old tensor-only smoke output remains historical and was not executed here. Paper compilation, page count and literature-reference checking were not rerun; the original manuscript and figures are preserved for inspection. The new adult chronological-age pilot is a separate study under the repository's current `results/` directory.
