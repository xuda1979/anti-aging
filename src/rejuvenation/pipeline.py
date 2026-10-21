"""Leakage-resistant data preparation and small CPU models for GSE113957."""
from __future__ import annotations
import csv
import gzip
import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

SEED = 20261021


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_age(value: str) -> float:
    """Parse exact GEO characteristic, including HGPS years/months; never use title."""
    value = value.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return float(value)
    m = re.fullmatch(r"(\d+)yr(?:(\d+)mos)?", value)
    if not m:
        raise ValueError(f"Unrecognized GEO age characteristic: {value!r}")
    return int(m.group(1)) + int(m.group(2) or 0) / 12


def read_metadata(raw: str | Path) -> pd.DataFrame:
    records = []
    for path in sorted(Path(raw).glob("GSE113957-GPL*_series_matrix.txt.gz")):
        platform = re.search(r"GPL\d+", path.name).group(0)
        with gzip.open(path, "rt") as stream:
            rows = [r for r in csv.reader(stream, delimiter="\t") if r and r[0].startswith("!Sample_")]
        fields = {r[0]: r[1:] for r in rows if r[0] != "!Sample_characteristics_ch1"}
        characteristics = [r[1:] for r in rows if r[0] == "!Sample_characteristics_ch1"]
        for idx, title in enumerate(fields["!Sample_title"]):
            row = {"sample_title": title, "gsm": fields["!Sample_geo_accession"][idx], "platform": platform}
            for char in characteristics:
                key, value = char[idx].split(":", 1)
                row[key.strip().lower().replace(" ", "_")] = value.strip()
            row["age_years"] = parse_age(row["age"])
            row["sex"] = row["sex"].lower()
            row["eligible"] = row["disease"].lower() == "normal" and row["age_years"] >= 18
            row["exclusion_reason"] = "" if row["eligible"] else ("HGPS" if row["disease"].lower() != "normal" else "pediatric")
            records.append(row)
    meta = pd.DataFrame(records).sort_values("gsm", kind="stable").reset_index(drop=True)
    if len(meta) != 143 or meta["gsm"].duplicated().any() or meta["sample_title"].duplicated().any():
        raise ValueError("Unexpected/missing/duplicate GEO samples")
    if meta["cell_id"].isna().any() or meta["cell_id"].duplicated().any():
        raise ValueError("Missing or duplicate public cell IDs require donor adjudication")
    return meta


def read_dataset(raw: str | Path):
    raw = Path(raw)
    meta = read_metadata(raw)
    table = pd.read_csv(raw / "GSE113957_fpkm.txt.gz", sep="\t")
    expected = set(meta.sample_title)
    if set(table.columns[8:]) != expected or table["Transcript ID"].duplicated().any():
        raise ValueError("Expression/metadata alignment or feature identity mismatch")
    values = table[meta.sample_title].to_numpy(dtype=np.float64).T
    if values.shape != (143, 27142) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Unexpected expression schema/nonfinite/negative FPKM")
    annotation = table[["Transcript ID", "Annotation/Divergence"]].copy()
    return values, meta, annotation


def frozen_split(meta: pd.DataFrame):
    """Split whole public cell-line groups before any feature operation."""
    eligible = np.flatnonzero(meta.eligible.to_numpy())
    train_rel, test_rel = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(eligible, groups=meta.iloc[eligible].cell_id))
    train, test = eligible[train_rel], eligible[test_rel]
    if set(meta.iloc[train].cell_id) & set(meta.iloc[test].cell_id):
        raise ValueError("Training/holdout public cell ID overlap")
    return train, test


class TrainingFeatures(TransformerMixin, BaseEstimator):
    """Fit eligible/high-variance transcript selection using training samples only."""
    def __init__(self, max_features=1000, minimum_max_fpkm=5.0):
        self.max_features = max_features
        self.minimum_max_fpkm = minimum_max_fpkm

    def fit(self, X, y=None):
        X = check_array(X, ensure_all_finite=True)
        if (X < 0).any():
            raise ValueError("FPKM cannot be negative")
        self.n_features_in_ = X.shape[1]
        eligible = np.flatnonzero(X.max(axis=0) >= self.minimum_max_fpkm)
        if not len(eligible):
            raise ValueError("No eligible transcripts")
        variance = np.log1p(X).var(axis=0)
        order = np.lexsort((eligible, -variance[eligible]))[: self.max_features]
        self.selected_indices_ = eligible[order]
        return self

    def transform(self, X):
        check_is_fitted(self, "selected_indices_")
        X = check_array(X, ensure_all_finite=True)
        if X.shape[1] != self.n_features_in_ or (X < 0).any():
            raise ValueError("Invalid FPKM feature schema")
        return np.log1p(X[:, self.selected_indices_])


def ridge_pipeline(alpha):
    return Pipeline([("features", TrainingFeatures()), ("scale", StandardScaler()), ("regressor", Ridge(alpha=alpha, solver="lsqr", tol=1e-6))])


def mlp_pipeline():
    network = Pipeline([
        ("features", TrainingFeatures()),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=16, svd_solver="full")),
        ("regressor", MLPRegressor(hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs", alpha=10.0,
                                   max_iter=2000, random_state=SEED, tol=1e-6))])
    return TransformedTargetRegressor(regressor=network, transformer=StandardScaler())
