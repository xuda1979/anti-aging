# Independent verification of the public-data pilot

**Software/data verification: PASS.** The publication run is `pilot_20261021_v2`, using frozen protocol `GSE113957-ADULT-v1-20261021`. Independent calculations and artifact checks passed; all seven contract tests passed, including deliberate corruption of an actual prediction CSV. This is an internal computational verification, not independent biological replication or peer review.

## Verified results

GEO contains 143 samples (133 Normal and 10 HGPS). The primary analysis correctly selects 111 healthy adults aged 19–96 years, with 88 public cell lines for training and 23 for the frozen holdout. The 22 pediatric Normal and 10 HGPS samples are excluded. Public cell IDs are unique and disjoint across the split, but are only a proxy for donor identity.

| Model | Holdout MAE, years | 95% sample-bootstrap interval | R² | Improvement versus training median |
|---|---:|---:|---:|---:|
| Training-median reference | 25.0652 | 20.5870–29.5870 | -0.1897 | reference |
| Ridge | 12.7902 | 9.3894–16.1458 | 0.6262 | 48.97% lower MAE |
| Compact MLP | 14.2414 | 9.5271–19.1692 | 0.4614 | 43.18% lower MAE |

Both fitted models pass the predeclared 20% baseline-improvement criterion with positive paired-bootstrap lower bounds. The MLP does **not** establish superiority over Ridge: MLP improvement in MAE is -1.4513 years, with a paired 95% interval of -5.0545 to 1.9761 years. These intervals condition on one fixed 23-sample holdout and do not incorporate training-cohort or study-to-study uncertainty.

## Verification scope

The independent verifier parses the original GEO matrices without using the producer's parsing helper, validates source SHA256 hashes, recreates the pre-fit group split, compares sample metadata/labels, validates all five inner training folds, recomputes inner-CV alpha selection, and checks final training-only feature selection/scaling/PCA statistics against raw expression. It loads both checkpoints and reproduces all saved holdout predictions. It independently recomputes MAE, RMSE, median absolute error, R², Pearson correlation, paired confidence intervals, subgroup counts/errors, and the reported 30-shuffle diagnostic summary. Recorded code, result and checkpoint hashes match the publication run. The verifier does not fit another model or rerun each shuffled-control fit.

The first run was superseded after its inference CLI default path was corrected. The verifier detected this source drift. Its failed provenance record remains in `v1_source_drift_detected.json`; the corrected v2 run passes the same frozen protocol and has identical prediction-file hashes to v1. There was no model/threshold/holdout change.

## Scientific limits

The study predicts chronological age only. It provides no causal mechanism, rejuvenation, restoration of cell function, lifespan extension, or clinical result. One small cohort leaves passage/culture, platform, sex, ancestry and other confounding unresolved. Among the 14 held-out adults below60, Ridge has R²=-2.4100 and MLP R²=-4.0722; pooled correlation must not be interpreted as uniformly accurate prediction across age ranges. Only one held-out sample uses platform GPL16791, preventing a useful platform-generalization conclusion. No external-cohort validation was performed.

## Replay

Run from the repository root after installing the locked dependencies and downloading the pinned processed data:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_study.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

Detailed machine-readable evidence is in `report.json`, pre-fit sample expectations in `independent_preflight.json`, and test output in `test_output.txt`.
