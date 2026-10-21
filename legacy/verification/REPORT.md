# Independent verification of recovered CellClockAI

**Archived artifact verification: PASS. Fresh computational replay comparison: PASS.**

The recovered `CellClockAI_Paper_Code_Results.zip` has SHA256 `8e0b99ec397c59b3ea58fe9f64e19e6aa8fed3dfb41f0eca726742b4761d2ace`. All 58 files listed in its original manifest match their recorded hashes. The unchanged original tree is preserved in `legacy/CellClockAI`; fresh execution records and copied outputs are separate in `legacy/reproduction`. Archive-local hashes establish internal integrity, not the original execution date or author identity.

## Archived observed-data results

Independent pairwise scoring of saved probabilities reproduces pooled and per-donor AUROC **1.00** and balanced accuracy **1.00** for treatment classification. There are 34 recovered cultures from three donors: 21 controls and 13 transiently reprogrammed samples. Held-out donors O1/O2/O3 have 9/11/14 samples; the corresponding training folds have 25/23/20. Donor labels are disjoint in each fold. There are 13 matched donor/experiment/duration blocks. The 100 saved within-context permutation scores reproduce mean AUROC 0.5376923 and descriptive p=0.00990099.

These labels describe treatment history, not chronological age, biological age, function or rejuvenation. Only three donors are independent biological units. Post-treatment sorting can select responsive cells, and treatment-associated cell state is an alternative explanation. The permutation procedure depends on exchangeability within donor/experiment/duration blocks; its p-value is not evidence from a randomized biological efficacy trial.

## Archived synthetic results

All six scenarios contain 20 seeds, with 900 training and 600 test observations per seed. The archive contains 480 model-run records and 600 ranking records. The verifier independently recomputes every reported mean and sample standard deviation from these rows. It also recomputes every saved policy regret and clock-decoy choice from the chosen action and the simulator's specified truth at the policy-evaluation context. The absent-combination scenario correctly has no training support for the true pair.

These checks validate arithmetic and consistency within invented simulations. Reg00–Reg11 are invented regulators; the simulator provides randomized actions, known functional truth and an oracle identity constraint. None of these outcomes nominates a real human gene or proves a biological mechanism. Per-sample synthetic prediction vectors and ranking orders were not archived, so RMSE inputs and precision rankings cannot be independently recalculated from summary rows alone.

## Fresh replay on 21 October 2026

The builder reran the original public analysis and all six synthetic scenarios with 20 seeds and 100 public-data permutations, using unchanged source and pinned scientific dependencies. The full scientific script exited successfully in **107.39 seconds**, with no GPU or paid cloud computation. Fresh public source matrices match the two original SHA256 digests; both contain 35,723 unique gene IDs. Independently selected source headers exactly recover the 34 included cultures.

The independent comparison found all **11** checked result files byte-identical to the archive: public probabilities, public permutation scores and summary, marker changes, association genes, synthetic per-run metrics, rankings and both summary tables, synthetic metadata, and prospective power calculations. All **12** Python/TOML source files are byte-identical. Public probability differences are exactly zero. This reproduces the old experiment and marker-ablation summary outputs too, but their per-sample probabilities were not saved, so those two metrics still do not have separate independent numeric recomputation.

The prospective power file is a calculation under assumed effect sizes, not an observed biological result. Historical runtime logs, figures and optional deep-model smoke claims remain historical unless explicitly covered by the fresh replay evidence.

## Tests and review

- **585** independent checks pass on archived result records.
- **590** checks pass on the completed fresh replay, including available raw-source checks.
- **3** independent verifier tests pass, including rejection of a deliberately corrupted saved probability CSV.
- Fresh original invariant suite: **6 passed, 1 optional Torch test skipped**.
- Optional graph-model smoke: **SKIPPED**, because Torch is unavailable; no new biological deep-model training is claimed.

Static source review and pattern scans found no embedded API keys, passwords, private mail, investor lists or campaign records. Downloads use two fixed HTTPS NCBI URLs with hash checks. Packaging uses standard setuptools metadata without custom install hooks; the nested workflow is inactive in the parent repository. PDF inspection found no JavaScript. The manuscript's opening action only selects a page view. This is a bounded review, not a comprehensive security audit.

## Replay the verifier

From the main repository root, after installing its locked scientific dependencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python legacy/verification/verify_legacy.py --output /tmp/legacy_archive_recheck.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s legacy/verification -p 'test_legacy*' -v
```

The first command checks the preserved archive; fresh-replay status is UNKNOWN unless `--replay-root` points to a completed full replay tree containing source, results, metadata and raw data. Its report should be written separately from the recorded release evidence. Detailed release evidence is in `legacy/verification/report.json`, and the scientific replay command, environment and timestamps are recorded under `legacy/reproduction`.
