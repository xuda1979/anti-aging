#!/usr/bin/env python3
"""Independent study verifier. Does not import producer parsing/scoring helpers.

Only checkpoint deserialization imports the custom transformer class. No model
is fitted here. Failure emits a failed report and nonzero exit status.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROTOCOL_SHA = "a1adcf752795eb6ce261a1473efff2d49d9c10c52664e5b4ed9935b671859060"
RAW_SHA = {
    "GSE113957_fpkm.txt.gz": "999c9d720ff7f0dfe59675f347803d3a0181231313aace2f8406f08528aa123d",
    "GSE113957-GPL16791_series_matrix.txt.gz": "3144bef9ef61ca818e16aaf9a7a70e68b082a70a81f5737517cd009d307ea90a",
    "GSE113957-GPL18573_series_matrix.txt.gz": "1764a9926545a7c4aecc68f423e9338245e859dfe14eb38a8f4bfb88fb0c80ec",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata_independent(root):
    records = []
    for path in sorted((root / "data/raw").glob("*series_matrix.txt.gz")):
        with gzip.open(path, "rt") as handle:
            rows = list(csv.reader(handle, delimiter="\t"))
        fields = {r[0]: r[1:] for r in rows if r and r[0] in
                  ["!Sample_geo_accession", "!Sample_title"]}
        chars = [r[1:] for r in rows if r and r[0] == "!Sample_characteristics_ch1"]
        for i, gsm in enumerate(fields["!Sample_geo_accession"]):
            obj = {r[i].partition(": ")[0].lower(): r[i].partition(": ")[2] for r in chars}
            obj.update(gsm=gsm, sample_title=fields["!Sample_title"][i],
                       platform=re.search(r"GPL\d+", path.name).group())
            obj["cell_id"] = obj.pop("cell id")
            if obj["disease"] == "Normal":
                obj["age_years"] = float(obj["age"])
            else:
                parts = re.fullmatch(r"(\d+)yr(?:(\d+)mos)?", obj["age"])
                obj["age_years"] = int(parts[1]) + int(parts[2] or 0) / 12
            obj["sex"] = obj["sex"].lower()
            records.append(obj)
    return pd.DataFrame(records).sort_values("gsm").reset_index(drop=True)


def scores_independent(actual, prediction):
    actual, prediction = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    assert actual.shape == prediction.shape and actual.ndim == 1
    assert np.isfinite(actual).all() and np.isfinite(prediction).all()
    residual = actual - prediction
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return {"mae": float(np.mean(np.abs(residual))),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "r2": float(1 - np.sum(residual ** 2) / total) if total > 0 else None,
            "median_absolute_error": float(np.median(np.abs(residual))),
            "pearson_correlation": float(np.corrcoef(actual, prediction)[0, 1])
                if np.std(prediction) > 0 else None}


def intervals_independent(actual, prediction, baseline, seed=20261022):
    actual, prediction, baseline = map(np.asarray, (actual, prediction, baseline))
    samples = np.random.default_rng(seed).integers(0, len(actual), size=(2000, len(actual)))
    err = np.abs(actual - prediction)
    base = np.abs(actual - baseline)
    return {"mae_ci95": np.percentile(np.mean(err[samples], axis=1), [2.5, 97.5]).tolist(),
            "improvement_ci95": np.percentile(np.mean((base - err)[samples], axis=1), [2.5, 97.5]).tolist()}


def assert_predictions_match_record(actual, recorded, atol=1e-9):
    assert np.allclose(np.asarray(actual), np.asarray(recorded), rtol=0, atol=atol), "Checkpoint predictions differ from persisted predictions"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run", default="pilot_20261021_v2")
    args = parser.parse_args()
    root, run = args.root.resolve(), args.run
    report = {"verifier": "independent implementation; no producer scoring/parsing helpers",
              "checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "run": run, "verdict": "UNKNOWN", "checks": [], "metrics": {}}
    def check(name, expression):
        assert expression, name
        report["checks"].append({"check": name, "verdict": "PASS"})
    try:
        check("Frozen protocol hash", digest(root / "configs/protocol.json") == PROTOCOL_SHA)
        for filename, expected in RAW_SHA.items():
            check("Raw source hash: " + filename, digest(root / "data/raw" / filename) == expected)
        meta = metadata_independent(root)
        check("143 unique public GSMs/titles/cell IDs", len(meta) == 143 and
              all(meta[k].nunique() == 143 for k in ["gsm", "sample_title", "cell_id"]))
        check("133 Normal and 10 HGPS", sum(meta.disease == "Normal") == 133 and sum(meta.disease == "HGPS") == 10)
        adult = meta.loc[(meta.disease == "Normal") & (meta.age_years >= 18)].copy()
        check("111 eligible adults age19-96", len(adult) == 111 and adult.age_years.min() == 19 and adult.age_years.max() == 96)
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20261021).split(adult, groups=adult.cell_id))
        train, test = adult.iloc[tr], adult.iloc[te]
        check("88 training/23 holdout, disjoint cell IDs", len(train) == 88 and len(test) == 23 and not(set(train.cell_id) & set(test.cell_id)))
        preflight = json.loads((root / "verification/independent_preflight.json").read_text())
        check("Split agrees with independent pre-fit manifest", train.gsm.tolist() == preflight["expected_train_gsm"] and test.gsm.tolist() == preflight["expected_test_gsm"])
        runpath = root / "results" / run
        reported = json.loads((runpath / "metrics.json").read_text())
        predictions = pd.read_csv(runpath / "predictions.csv").set_index("gsm").loc[test.gsm]
        check("All and only holdout GSMs", len(predictions) == 23 and set(pd.read_csv(runpath / "predictions.csv").gsm) == set(test.gsm))
        for key in ["cell_id", "age_years", "sex", "platform"]:
            check("Prediction labels/metadata: " + key, predictions[key].tolist() == test[key].tolist())
        baseline = np.repeat(np.median(train.age_years), len(test))
        check("Training-median baseline", np.array_equal(baseline, predictions.prediction_median_baseline.to_numpy()))
        split_record = pd.read_csv(runpath / "split_manifest.csv").set_index("gsm")
        check("Public split manifest contains all143 exactly once", len(split_record) == 143 and split_record.index.is_unique and set(split_record.index) == set(meta.gsm))
        for split_name, expected_ids in [("train", train.gsm), ("holdout", test.gsm), ("excluded", meta.loc[~meta.gsm.isin(adult.gsm), "gsm"])]:
            check("Public split manifest: " + split_name, set(split_record.index[split_record.split == split_name]) == set(expected_ids))
        inner = json.loads((runpath / "inner_cv_manifest.json").read_text())
        check("Exactly five inner training folds", len(inner) == 5)
        validation_ids = []
        for fold in inner:
            a, b = set(fold["train_gsm"]), set(fold["validation_gsm"])
            check("Inner fold disjoint, partition of training, no holdout: " + str(fold["fold"]), not(a & b) and a | b == set(train.gsm) and not((a | b) & set(test.gsm)))
            validation_ids.extend(fold["validation_gsm"])
        check("Inner validation covers training exactly once", len(validation_ids) == len(set(validation_ids)) == 88 and set(validation_ids) == set(train.gsm))
        cv_results = pd.read_csv(runpath / "ridge_inner_cv.csv")
        check("Predeclared four Ridge alphas only", sorted(cv_results.param_regressor__alpha.astype(float)) == [1, 10, 100, 1000])
        scores = cv_results[["split" + str(i) + "_test_score" for i in range(5)]].to_numpy(float)
        check("Inner-CV mean scores recompute", np.allclose(scores.mean(axis=1), cv_results.mean_test_score.to_numpy(), rtol=0, atol=1e-10))
        expected_alpha = float(cv_results.iloc[int(np.argmax(scores.mean(axis=1)))].param_regressor__alpha)
        check("Ridge alpha selected from training CV only", expected_alpha == reported["ridge_selected_alpha"])
        data = pd.read_csv(root / "data/raw/GSE113957_fpkm.txt.gz", sep="\t")
        check("27142 unique features and exact 143 expression titles", len(data) == 27142 and data["Transcript ID"].nunique() == 27142 and set(data.columns[8:]) == set(meta.sample_title))
        Xtrain = data[train.sample_title].to_numpy(float).T
        Xtest = data[test.sample_title].to_numpy(float).T
        eligible = np.flatnonzero(np.max(Xtrain, axis=0) >= 5)
        variances = np.var(np.log1p(Xtrain), axis=0)
        selected = eligible[np.lexsort((eligible, -variances[eligible]))[:1000]]
        check("Expression is finite/nonnegative", np.isfinite(Xtrain).all() and np.isfinite(Xtest).all() and np.min(Xtrain) >= 0 and np.min(Xtest) >= 0)
        for modelname in ["median_baseline", "ridge", "compact_mlp"]:
            recorded = predictions["prediction_" + modelname].to_numpy()
            if modelname != "median_baseline":
                loaded = joblib.load(root / "models" / run / (modelname + ".joblib"))
                predicted = loaded.predict(Xtest)
                assert_predictions_match_record(predicted, recorded)
                check("Reloaded checkpoint matches predictions: " + modelname, True)
                pipeline = loaded.regressor_ if modelname == "compact_mlp" else loaded
                check("Training-only feature selection: " + modelname, np.array_equal(selected, pipeline.named_steps["features"].selected_indices_))
                transformed = np.log1p(Xtrain[:, selected])
                check("Training-only scale mean/variance: " + modelname,
                      np.allclose(np.mean(transformed, axis=0), pipeline.named_steps["scale"].mean_, rtol=0, atol=1e-10) and
                      np.allclose(np.var(transformed, axis=0), pipeline.named_steps["scale"].var_, rtol=0, atol=1e-10))
                if modelname == "compact_mlp":
                    check("Training-only age scaler", np.allclose(loaded.transformer_.mean_, [np.mean(train.age_years)]))
                    standardized = (transformed - transformed.mean(axis=0)) / transformed.std(axis=0)
                    pca = pipeline.named_steps["pca"]
                    check("PCA training mean", np.allclose(pca.mean_, standardized.mean(axis=0), rtol=0, atol=1e-10))
                    singular = np.linalg.svd(standardized - standardized.mean(axis=0), compute_uv=False)
                    check("PCA training singular values", np.allclose(pca.singular_values_, singular[:16], rtol=1e-9, atol=1e-9))
                    mlp = pipeline.named_steps["regressor"]
                    report["neural_trainable_parameters"] = int(sum(x.size for x in mlp.coefs_) + sum(x.size for x in mlp.intercepts_))
                    check("Declared 16PC/8hidden architecture", pipeline.named_steps["pca"].n_components_ == 16 and mlp.hidden_layer_sizes == (8,) and report["neural_trainable_parameters"] == 145)
            vals = scores_independent(test.age_years.to_numpy(), recorded)
            vals.update(intervals_independent(test.age_years.to_numpy(), recorded, baseline))
            vals["relative_mae_reduction"] = float(1 - vals["mae"] / np.mean(np.abs(test.age_years.to_numpy() - baseline)))
            vals["criterion"] = None if modelname == "median_baseline" else ("PASS" if vals["relative_mae_reduction"] >= .2 and vals["improvement_ci95"][0] > 0 else "FAIL")
            report["metrics"][modelname] = vals
            rec = reported["models"][modelname]
            for own_key, recorded_key in [("mae", "mae_years"), ("rmse", "rmse_years"), ("r2", "r2"), ("median_absolute_error", "median_absolute_error_years"), ("pearson_correlation", "pearson_r"), ("mae_ci95", "bootstrap_mae_ci95")]:
                left, right = vals[own_key], rec[recorded_key]
                check("Reported numeric result: " + modelname + "/" + recorded_key,
                      left is None and right is None or left is not None and right is not None and np.allclose(left, right, rtol=0, atol=1e-9))
            if modelname != "median_baseline":
                check("Reported paired CI and criterion: " + modelname,
                      np.allclose(vals["improvement_ci95"], rec["paired_improvement_vs_baseline_ci95"], rtol=0, atol=1e-9) and vals["criterion"] == rec["criterion"])
        contrast = intervals_independent(test.age_years.to_numpy(), predictions.prediction_compact_mlp.to_numpy(), predictions.prediction_ridge.to_numpy())
        improvement = float(np.mean(np.abs(test.age_years.to_numpy() - predictions.prediction_ridge.to_numpy()) - np.abs(test.age_years.to_numpy() - predictions.prediction_compact_mlp.to_numpy())))
        report["mlp_vs_ridge"] = {"mean_mae_improvement": improvement, "ci95": contrast["improvement_ci95"], "superiority": "PASS" if contrast["improvement_ci95"][0] > 0 else "FAIL"}
        check("Neural vs Ridge improvement/CI and failed superiority recompute", np.allclose(improvement, reported["mlp_vs_ridge"]["paired_mae_improvement_years"], rtol=0, atol=1e-9) and np.allclose(contrast["improvement_ci95"], reported["mlp_vs_ridge"]["paired_improvement_ci95"], rtol=0, atol=1e-9) and report["mlp_vs_ridge"]["superiority"] == reported["mlp_vs_ridge"]["superiority_criterion"])
        for subgroup_name, subgroup_results in reported["subgroups"].items():
            attr, value = subgroup_name.split("=", 1)
            if attr == "age_band":
                selected_rows = predictions.age_years.to_numpy() < 60 if value == "18-59" else predictions.age_years.to_numpy() >= 60
            else:
                selected_rows = predictions[attr].to_numpy() == value
            for name, rec in subgroup_results.items():
                own = scores_independent(predictions.age_years.to_numpy()[selected_rows], predictions["prediction_" + name].to_numpy()[selected_rows])
                check("Subgroup counts/errors: " + subgroup_name + "/" + name, int(sum(selected_rows)) == rec["n"] and np.allclose([own["mae"], own["rmse"]], [rec["mae_years"], rec["rmse_years"]], rtol=0, atol=1e-9))
        controls = pd.read_csv(runpath / "negative_control.csv")
        diagnostic = reported["training_label_shuffle_diagnostic"]
        check("Thirty distinct training-shuffle records and summary", len(controls) == controls.permutation.nunique() == 30 and diagnostic["permutations"] == 30 and np.allclose([controls.mae_years.mean(), controls.mae_years.min(), controls.mae_years.max()], [diagnostic["mae_mean"], diagnostic["mae_min"], diagnostic["mae_max"]], rtol=0, atol=1e-9))
        provenance = json.loads((runpath / "run_manifest.json").read_text())
        for section in ["source_sha256", "result_sha256", "model_sha256"]:
            for relative, expected in provenance[section].items():
                check("Run provenance hash: " + relative, digest(root / relative) == expected)
        original = predictions.prediction_ridge.to_numpy().copy()
        corrupted = original.copy(); corrupted[0] += 1.0
        try:
            assert_predictions_match_record(original, corrupted)
            raise AssertionError("Tamper went undetected")
        except AssertionError as error:
            check("Deliberately corrupted prediction is rejected", str(error) != "Tamper went undetected")
        report["verdict"] = "PASS"
    except Exception as error:
        report["verdict"] = "FAIL"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    report["limitations"] = ["Independent numeric/split/software verification is not external biological validation.",
                             "Public cell IDs are a donor proxy; unrelatedness and complete culture provenance are unknown.",
                             "One small adult cohort; no intervention or rejuvenation measured."]
    report["verifier_source_sha256"] = digest(Path(__file__))
    output = root / "verification/report.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verdict": report["verdict"], "checks": len(report["checks"]), "metrics": report["metrics"], "failure": report.get("failure")}, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
