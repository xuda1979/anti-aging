# Public-data pilot: adult fibroblast chronological-age prediction

David Xu, PhD; Peter Lee, PhD, MD; and Zhimin Zheng, PhD  
21 October 2026 · Exploratory computational result · Protocol GSE113957-ADULT-v1-20261021

## Result and investor interpretation

A reproducible CPU experiment trained a regularized linear model and a compact neural model using free public fibroblast RNA-sequencing data. On a fixed 23-cell-line holdout, Ridge achieved **12.79 years mean absolute error (MAE)**, compared with **25.07 years** for the training-median baseline. A compact neural network achieved **14.24 years MAE** and did **not** establish superiority over Ridge. This is evidence that the public-data pipeline can learn chronological-age associations. It does not discover a rejuvenation mechanism, show that any cells became younger, or establish proprietary therapeutic value.

| Model | Holdout MAE, years | Conditional 95% bootstrap CI | RMSE, years | R² | Predeclared comparison with median baseline |
|---|---:|---:|---:|---:|---|
| Training-median age | 25.07 | 20.59–29.59 | 27.49 | −0.190 | Reference |
| Ridge | 12.79 | 9.39–16.15 | 15.41 | 0.626 | PASS |
| Compact neural network | 14.24 | 9.53–19.17 | 18.50 | 0.461 | PASS |

Both learned models passed the frozen requirement of at least 20% lower MAE than the median baseline and a positive lower bound of the paired bootstrap improvement interval. Ridge's paired improvement was 12.28 years (95% CI 7.92–17.00). The neural network's improvement over Ridge was −1.45 years (95% CI −5.05–1.98); neural superiority therefore **FAILS** the predeclared comparison. Neither result is a clinical validation or a claim to outperform published aging clocks.

![Holdout predictions and errors](holdout_results.png)

## Public data and exact sample accounting

Source: [NCBI GEO GSE113957](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113957), associated with Fleischer et al., [*Predicting age from the transcriptome of human dermal fibroblasts*](https://doi.org/10.1186/s13059-018-1599-6). Only three freely downloadable processed files were used: a 7.53 MB compressed published FPKM table and two small platform-specific GEO series matrices. No paid data, restricted-access data, new human sampling or raw sequencing downloads were required. Public availability is not an assertion of unrestricted commercial licensing; retain source attribution and review upstream rights before commercial deployment.

The published expression table contains **143 samples × 27,142 RefSeq transcripts**. GEO characteristics identify 133 normal samples and 10 HGPS samples. The primary analysis excludes all 10 HGPS samples and 22 normal samples younger than 18, leaving **111 healthy adult cell lines**. A single frozen group split used **88 training and 23 holdout samples**. There is no overlap in public cell-line IDs. These IDs are a donor proxy; independent donor identity, family relationships and provenance have not been established by this project.

Ages and disease labels were parsed from GEO characteristics, not inferred from column names. Current GEO metadata include one normal donor age **96**, whereas the paper's summary gives a healthy-cohort range of 1–94. This analysis preserves the downloaded metadata rather than silently changing the label. Source SHA-256 hashes, accession mappings, exclusions and split assignments are recorded in the machine-readable manifests.

## Frozen methods

The protocol was recorded before fitting. Eligible samples were sorted by GSM accession; `GroupShuffleSplit(test_size=0.2, random_state=20261021)` grouped by public cell ID fixed the holdout before feature processing. For every training fold, transcripts with maximum training FPKM ≥5 were retained, `log1p` transformed, and the 1,000 highest-variance transcripts selected using training samples only. Standardization was also fitted on training samples only. Age, sex, platform and ancestry metadata were not expression-model predictors.

Ridge selected alpha from {1, 10, 100, 1000} using five-fold grouped inner cross-validation and training-only MAE; alpha **100** was selected. The fixed neural model applied 16 principal components, an eight-unit tanh hidden layer, L-BFGS optimization, alpha 10 and training-only target standardization. The neural network has **145 fitted neural weights and biases**, in addition to fitted PCA and preprocessing parameters. Its architecture was not tuned against the holdout. The final fit converged in 429 iterations without convergence warnings.

The baseline always predicts the **training median age of 62.5 years**. The holdout median is 46 years, so this particular random split has a substantial age-distribution difference. The reported 2,000 paired sample-bootstrap replicates condition on this split and these fitted models; they do not capture split-selection, training-cohort, platform or external-population uncertainty.

Thirty training-label shuffles at the selected Ridge alpha gave holdout MAE averaging **24.08 years**, ranging from **18.71 to 30.71**. This is a diagnostic control, not a confirmatory permutation p-value: hyperparameter search was not repeated for every shuffled dataset.

## Limitations visible in the results

- This is a single small cohort and one random split, with no independent external validation. There are no treatment or cell-function outcomes.
- The held-out female group has seven samples and the male group 16. One platform has only one held-out sample; no platform-generalization claim is supported.
- Ridge MAE is **15.84 years** in held-out ages 18–59 and **8.04 years** at ages 60+. In the younger subgroup its R² is **−2.41**. The network's younger-subgroup R² is **−4.07**. Broad-range association therefore does not imply useful fine-grained age prediction across the adult lifespan.
- Chronological age, sex, ancestry, culture conditions, passage and sequencing batch may be correlated. A gene's fitted coefficient is not a causal mechanism or an intervention target.
- The published FPKM matrix already reflects upstream processing decisions. This pilot did not realign raw reads or audit source laboratory procedures.
- A score shift after reprogramming could reflect cell identity, proliferation, stress, batch or distribution shift. This model must not be used to assert rejuvenation from such a shift.

## Reproducibility and validation

Run: `pilot_20261021_v2`. Environment, exact dependency versions, source/data/result/model SHA-256 hashes, CPU runtime, fixed seeds, inner-CV memberships, per-sample predictions, subgroup scores, selected features and model files are included. Direct cloud spending was **$0**; local CPU wall time was 2.19 seconds, excluding code development and downloads. No GPU was used.

Before the full fit, focused tests checked schema validation, age parsing, exclusion and split integrity, training-only feature selection, failure on corrupt numeric inputs, and checkpoint reload. A tiny optimizer lifecycle exercised update → save → load → resume and compared the resumed model with uninterrupted training. A 24-sample real-data canary passed. Both final fitted models were saved and reloaded, reproducing holdout predictions within 1e−12. Independent verification is recorded separately in the repository; this report does not substitute for that review.

The earlier internal run `pilot_20261021_v1` has identical scientific predictions. Version 2 only corrects the example prediction script's default checkpoint path and reruns the unchanged frozen protocol to keep source hashes consistent. This was a packaging correction, not a search over holdout performance. Both records are retained.

## Next scientific milestone

Freeze an independent adult fibroblast cohort and a harmonization protocol before testing generalization. Add age/sex/platform-aware baselines and a prospectively specified set of repeated group splits on the training resource, without presenting those as untouched holdouts. Only after credible external age prediction should a distinct perturbation study ask whether scores track independent function while preserving cell identity. Causal mechanism claims require perturbation, rescue, independent donors, replication and safety evidence.

The wider venture remains proposed. Biological efficacy, validated proprietary targets, institutional sponsorship, founder role commitments and clinical results have not been established. The white paper's $8 million, 24-month program remains illustrative.
