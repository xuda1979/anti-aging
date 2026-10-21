# Research model and data card

**Authors:** David Xu, PhD; Peter Lee, PhD, MD; Zhimin Zheng, PhD.  
**Date:** 21 October 2026. **Run:** `pilot_20261021_v2`.

The released models are a small Ridge regression pipeline and a fixed compact neural regression pipeline. They predict **chronological age in years** from the published GSE113957 cultured human fibroblast FPKM feature space. They are intended for transparent research and pipeline reproduction. Neither model measures biological rejuvenation, diagnoses disease, estimates lifespan, establishes a causal target or supports a treatment decision.

The source dataset is freely accessible at [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113957); the source paper is [Fleischer et al.](https://doi.org/10.1186/s13059-018-1599-6). Download scripts pin the exact compressed data and metadata hashes. Source data are fetched on demand, attributed to their original investigators and not relabeled as proprietary. Public access does not establish unrestricted commercial licensing. No new specimens or restricted patient information were collected.

The training resource is 88 normal adult cell lines; a fixed set of 23 normal adult cell lines is reserved for the primary holdout. Ten HGPS and 22 pediatric normal samples were excluded before training. Cell-line IDs are the available grouping proxy; independent donor identity and kinship were not verified. Ages are current GEO characteristics, including age 96. The full inclusion/exclusion manifest is published.

`ridge.joblib` includes training-fitted transcript selection, log transformation, feature scaling and Ridge with alpha 100. `compact_mlp.joblib` additionally includes 16 fitted principal components, an eight-unit tanh hidden layer and training-fitted target scaling. The neural network has 145 fitted neural weights/biases; the full pipeline also contains fitted PCA and preprocessing parameters. Inference must retain the exact input transcript order and normalization used by the downloadable source table. The supplied prediction script reads that exact feature space rather than accepting an unvalidated arbitrary matrix.

On the frozen holdout, Ridge MAE is 12.79 years (conditional 95% bootstrap CI 9.39–16.15), the neural model MAE is 14.24 years (9.53–19.17), and the training-median baseline MAE is 25.07 years. The neural model did not demonstrate superiority over Ridge. These scores are conditional on one small split with different training/holdout age distributions. Younger-adult discrimination is weak; Ridge R² in the held-out 18–59 subgroup is −2.41. No external validation, ancestry fairness claim or platform-generalization claim is supported.

Culture, passage, sequencing batch, sex, ancestry and developmental history can confound associations. In particular, reprogrammed cells can shift identity or leave the training distribution: a younger predicted age would not prove improved cell function or reversal of aging. Fitted transcript weights must not be treated as intervention recommendations.

Both final pipelines were serialized and reloaded with predictions matching within 1e−12. Exact dependency versions, scoring output, per-sample predictions, split membership, source/model hashes and independent verification are available alongside this card. Use the pinned runtime for reproduction. Joblib files use Python serialization and should only be loaded from a trusted, hash-verified source.

The next validation requirement is an independently frozen adult cohort with a prespecified normalization and harmonization protocol, followed separately by prospective perturbation and functional studies. The program has not established biological efficacy, a proprietary therapeutic target, institutional sponsorship, founder role commitments or clinical results.
