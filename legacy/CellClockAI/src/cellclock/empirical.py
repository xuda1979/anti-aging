"""Association-only pilot on Gill et al. GSE165177 recovered fibroblasts.

No age labels, intervention effects on lifespan, or causal gene rankings are inferred.
All prediction transforms are fitted on the training fold only.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .download import FILES, download

PANELS = {
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "FBLN1"],
    "pluripotency": ["POU5F1", "NANOG", "SOX2", "LIN28A", "DPPA4", "UTF1"],
    "cell_cycle": ["MKI67", "PCNA", "TOP2A", "CCNB1", "CCNB2", "CDK1", "CDC20",
                   "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7"],
}


def parse_sample(name):
    m = re.fullmatch(r"(O[123])_(negative_control|transiently_reprogrammed)_(\d+)days_exp([12])", name)
    if not m:
        return None
    donor, status, day, batch = m.groups()
    return {"sample": name, "donor": donor, "status": status, "duration": int(day),
            "experiment": int(batch), "treated": int(status == "transiently_reprogrammed")}


def load_public(root):
    download(root)
    tables, symbols = [], None
    for name in FILES:
        d = pd.read_csv(Path(root) / "data/raw" / name, sep="\t", low_memory=False)
        cols = [c for c in d.columns if parse_sample(c)]
        if symbols is None:
            symbols = d.groupby("ID", sort=True)["Probe"].first()
        # Duplicate transcript/probe rows are averaged in the supplied log2 scale.
        # This is an explicit feature reduction, not reconstruction of count data.
        tables.append(d.groupby("ID", sort=True)[cols].mean())
    matrix = pd.concat(tables, axis=1).T
    if not matrix.index.is_unique or not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Duplicate sample labels or non-finite public matrix")
    meta = pd.DataFrame([parse_sample(s) for s in matrix.index]).set_index("sample")
    return matrix, meta, symbols


def train_transform(train, test, max_features=500):
    x = np.asarray(train, dtype=float)
    # At least 20% of training samples must have supplied log2 RPM > 0.
    expressed = (x > 0).mean(axis=0) >= .2
    variances = x.var(axis=0)
    eligible = np.flatnonzero(expressed & (variances > 1e-12))
    idx = eligible[np.argsort(-variances[eligible], kind="stable")[:max_features]]
    if not len(idx):
        raise ValueError("No expressed varying features in training fold")
    scaler = StandardScaler().fit(x[:, idx])
    return scaler.transform(x[:, idx]), scaler.transform(np.asarray(test)[:, idx]), idx


def evaluate(matrix, meta, labels=None, group="donor"):
    y = meta.treated.to_numpy() if labels is None else np.asarray(labels)
    scores = np.full(len(y), np.nan)
    fold_records = []
    for heldout in sorted(meta[group].unique()):
        test = (meta[group] == heldout).to_numpy()
        train = ~test
        a, b, selected = train_transform(matrix[train], matrix[test])
        clf = LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", random_state=17)
        clf.fit(a, y[train])
        prob = clf.predict_proba(b)[:, 1]
        scores[test] = prob
        fold_records.append({"heldout": str(heldout), "n_train": int(train.sum()),
                             "n_test": int(test.sum()), "n_features": len(selected),
                             "auroc": float(roc_auc_score(y[test], prob)),
                             "balanced_accuracy": float(balanced_accuracy_score(y[test], prob >= .5))})
    return {"auroc": float(roc_auc_score(y, scores)),
            "balanced_accuracy": float(balanced_accuracy_score(y, scores >= .5)),
            "folds": fold_records}, scores


def matched_changes(matrix, meta, symbols):
    changes, block_ids = [], []
    for block, rows in meta.groupby(["donor", "experiment", "duration"], sort=True):
        controls = rows.index[rows.treated == 0]
        treated = rows.index[rows.treated == 1]
        if len(controls) and len(treated):
            changes.append(matrix.loc[treated].mean() - matrix.loc[controls].mean())
            block_ids.append(block)
    delta = pd.DataFrame(changes)
    delta.index = pd.MultiIndex.from_tuples(block_ids, names=["donor", "experiment", "duration"])
    donor_mean = delta.groupby(level="donor").mean()
    effects = pd.DataFrame({"ensembl_id": matrix.columns, "symbol": symbols.reindex(matrix.columns).values,
                            "mean_log2_difference": donor_mean.mean().to_numpy(),
                            "donor_sd": donor_mean.std(ddof=1).to_numpy(),
                            "same_direction_all_donors": ((donor_mean > 0).all() | (donor_mean < 0).all()).to_numpy()})
    effects["evidence_level"] = "treatment_association_only"
    effects["absolute_difference"] = effects.mean_log2_difference.abs()
    effects = effects.sort_values("absolute_difference", ascending=False, kind="stable")
    return delta, donor_mean, effects


def run(root, permutations=100):
    root = Path(root)
    matrix, meta, symbols = load_public(root)
    meta.to_csv(root / "data/processed/public_sample_manifest.csv")
    donor, scores = evaluate(matrix.to_numpy(), meta)
    batch, _ = evaluate(matrix.to_numpy(), meta, group="experiment")
    banned = set(PANELS["cell_cycle"] + PANELS["pluripotency"])
    keep = ~symbols.reindex(matrix.columns).isin(banned).to_numpy()
    ablated, _ = evaluate(matrix.to_numpy()[:, keep], meta)
    rng = np.random.default_rng(20261013)
    null = []
    # Label exchangeability is assumed only within donor/experiment/duration.
    blocks = list(meta.groupby(["donor", "experiment", "duration"]).indices.values())
    for _ in range(permutations):
        labels = meta.treated.to_numpy().copy()
        for idx in blocks:
            labels[idx] = rng.permutation(labels[idx])
        result, _ = evaluate(matrix.to_numpy(), meta, labels)
        null.append(result["auroc"])
    delta, donor_mean, effects = matched_changes(matrix, meta, symbols)
    effects.head(100).to_csv(root / "results/public_association_genes.csv", index=False)
    panel_records = []
    for panel, genes in PANELS.items():
        for gene in genes:
            ids = symbols.index[symbols == gene].intersection(donor_mean.columns)
            if len(ids):
                for donor_id, values in donor_mean[ids].mean(axis=1).items():
                    panel_records.append({"panel": panel, "gene": gene, "donor": donor_id,
                                          "mean_log2_difference": float(values)})
    pd.DataFrame(panel_records).to_csv(root / "results/public_marker_changes.csv", index=False)
    prediction = meta.copy()
    prediction["leave_donor_out_probability"] = scores
    prediction.to_csv(root / "results/public_predictions.csv")
    pd.DataFrame({"null_auroc": null}).to_csv(root / "results/public_permutation_null.csv", index=False)
    summary = {"evidence_type": "public_data_association_pilot", "accession": "GSE165177",
               "n_samples": len(meta), "n_genes_after_aggregation": matrix.shape[1],
               "n_donors": meta.donor.nunique(), "class_counts": meta.treated.value_counts().to_dict(),
               "n_matched_blocks": len(delta), "leave_one_donor_out": donor,
               "leave_one_experiment_out": batch, "remove_selected_marker_genes": ablated,
               "removed_marker_features": int((~keep).sum()),
               "permutations": permutations, "permutation_null_mean_auroc": float(np.mean(null)),
               "permutation_p": float((1 + np.sum(np.array(null) >= donor["auroc"])) / (permutations + 1)),
               "limitations": ["Three independent donors only", "Treatment classification is not age measurement",
                               "Post-treatment sorting may select responsive cells", "No gene-specific interventions",
                               "Marker removal is not complete cell-cycle adjustment"]}
    (root / "results/public_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
