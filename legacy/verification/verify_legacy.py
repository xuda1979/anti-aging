#!/usr/bin/env python3
"""Independently check recovered CellClockAI artifacts without fitting models.

Saved-probability scoring uses pair comparisons and class-wise counts, never
producer scoring helpers. Synthetic truth/regret is recomputed from saved actions.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCENARIOS = ["additive", "synergy", "redundant", "stress_interaction", "unseen_synergy", "no_causal_effect"]
MODELS = ["linear_function", "interaction_function", "neural_function", "interaction_clock"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def auc_and_balanced_accuracy(labels, probabilities):
    y, p = np.asarray(labels, int), np.asarray(probabilities, float)
    assert set(y) == {0, 1} and np.isfinite(p).all()
    assert np.all((p >= 0) & (p <= 1))
    positives, negatives = p[y == 1], p[y == 0]
    differences = positives[:, None] - negatives[None, :]
    auc = float(np.mean((differences > 0) + .5 * (differences == 0)))
    balanced = float(.5 * (np.mean(positives >= .5) + np.mean(negatives < .5)))
    return {"auroc": auc, "balanced_accuracy": balanced}


def check_close(a, b):
    assert np.allclose(a, b, rtol=0, atol=1e-10), "Independent numeric result differs"


def verify_results(root):
    report = {"source": str(root), "verdict": "UNKNOWN", "checks": [], "public": {}, "synthetic": {}}
    def check(name, condition):
        assert condition, name
        report["checks"].append({"check": name, "verdict": "PASS"})
    try:
        result = root / "results"
        public = pd.read_csv(result / "public_predictions.csv")
        manifest = pd.read_csv(root / "data/processed/public_sample_manifest.csv")
        summary = json.loads((result / "public_summary.json").read_text())
        check("34 unique saved samples and three donors", len(public) == public["sample"].nunique() == 34 and set(public.donor) == {"O1", "O2", "O3"})
        check("Saved sample manifest matches predictions", manifest.to_dict("records") == public[manifest.columns].to_dict("records"))
        for row in public.itertuples():
            match = re.fullmatch(r"(O[123])_(negative_control|transiently_reprogrammed)_(\d+)days_exp([12])", row.sample)
            check("Public sample label: " + row.sample, match is not None and match[1] == row.donor and match[2] == row.status and int(match[3]) == row.duration and int(match[4]) == row.experiment and row.treated == int(match[2] == "transiently_reprogrammed"))
        check("Class counts agree", dict(public.treated.value_counts()) == {0: 21, 1: 13} and {int(k): v for k, v in summary["class_counts"].items()} == {0: 21, 1: 13})
        own = auc_and_balanced_accuracy(public.treated, public.leave_donor_out_probability)
        check_close([own["auroc"], own["balanced_accuracy"]], [summary["leave_one_donor_out"]["auroc"], summary["leave_one_donor_out"]["balanced_accuracy"]])
        check("Pooled donor-holdout metrics recompute", True)
        folds = []
        for donor in ["O1", "O2", "O3"]:
            held = public.loc[public.donor == donor]
            train = public.loc[public.donor != donor]
            rec = next(x for x in summary["leave_one_donor_out"]["folds"] if x["heldout"] == donor)
            ownfold = auc_and_balanced_accuracy(held.treated, held.leave_donor_out_probability)
            check_close([ownfold["auroc"], ownfold["balanced_accuracy"]], [rec["auroc"], rec["balanced_accuracy"]])
            check("Donor fold counts/disjointness: " + donor, len(train) == rec["n_train"] and len(held) == rec["n_test"] and not(set(train.donor) & set(held.donor)))
            folds.append({"heldout": donor, "n_train": len(train), "n_test": len(held), **ownfold})
        blocks = sum(public.groupby(["donor", "experiment", "duration"]).treated.nunique() == 2)
        check("13 matched donor/experiment/duration blocks", int(blocks) == summary["n_matched_blocks"] == 13)
        null = pd.read_csv(result / "public_permutation_null.csv").null_auroc.to_numpy(float)
        check("100 finite saved null scores", len(null) == summary["permutations"] == 100 and np.isfinite(null).all() and np.all((null >= 0) & (null <= 1)))
        null_mean = float(sum(null) / len(null))
        pvalue = float((1 + sum(null >= own["auroc"])) / (len(null) + 1))
        check_close([null_mean, pvalue], [summary["permutation_null_mean_auroc"], summary["permutation_p"]])
        check("Saved null summary and descriptive p-value recompute", True)
        raw_headers = []
        source_checks = []
        for provenance in json.loads((root / "data/provenance.json").read_text()):
            file = root / "data/raw" / provenance["file"]
            if file.exists():
                check("Raw file hash: " + file.name, sha(file) == provenance["sha256"])
                table = pd.read_csv(file, sep="\t", nrows=0)
                raw_headers.extend(c for c in table.columns if re.fullmatch(r"O[123]_(?:negative_control|transiently_reprogrammed)_\d+days_exp[12]", c))
                feature_ids = pd.read_csv(file, sep="\t", usecols=["ID"])["ID"]
                check("35723 unique aggregated gene IDs in source: " + file.name, feature_ids.nunique() == summary["n_genes_after_aggregation"] == 35723)
                source_checks.append({"file": file.name, "verdict": "PASS", "sha256": sha(file)})
            else:
                source_checks.append({"file": file.name, "verdict": "UNKNOWN", "reason": "Raw source matrix not bundled in original archive"})
        if raw_headers:
            check("Sample selection exactly matches fetched raw headers", len(raw_headers) == len(set(raw_headers)) == 34 and set(raw_headers) == set(public["sample"]))
        report["public"] = {"sample_count": 34, "donors": 3, "class_counts": {"control": 21, "treated": 13}, "pooled": own,
                            "folds": folds, "matched_blocks": int(blocks), "null_mean_auroc": null_mean,
                            "descriptive_permutation_p": pvalue, "raw_source_checks": source_checks,
                            "unrecomputable_from_saved_probabilities": ["Experiment-holdout probabilities were not archived", "Marker-removal probabilities were not archived", "Archived fitted checkpoints and fold-specific transforms were not saved"]}
        runs = pd.read_csv(result / "synthetic_runs.csv")
        ranked = pd.read_csv(result / "synthetic_rankings.csv")
        synthetic_summary = pd.read_csv(result / "synthetic_summary.csv").set_index(["scenario", "model"])
        rank_summary = pd.read_csv(result / "synthetic_ranking_summary.csv", header=[0, 1], index_col=[0, 1])
        check("480 distinct scenario/seed/model rows", len(runs) == 480 and not runs.duplicated(["scenario", "seed", "model"]).any() and set(runs.scenario) == set(SCENARIOS) and set(runs.model) == set(MODELS))
        check("Six scenarios each20 seeds per model", all(set(g.seed) == set(range(20)) for _, g in runs.groupby(["scenario", "model"])) and np.all(runs.n_train == 900) and np.all(runs.n_test == 600))
        numeric = ["functional_mean_rmse", "policy_regret", "clock_decoy_selected", "pair01_supported", "converged"]
        check("Synthetic metrics finite and in declared ranges", np.isfinite(runs[numeric].to_numpy(float)).all() and np.all(runs.functional_mean_rmse >= 0) and np.all(runs.policy_regret >= 0) and all(set(runs[k]) <= {0, 1} for k in numeric[2:]))
        for key, subset in runs.groupby(["scenario", "model"]):
            rec = synthetic_summary.loc[key]
            for name in numeric:
                values = subset[name].to_numpy(float)
                mean = float(sum(values) / len(values))
                sd = float(np.sqrt(np.sum((values - mean) ** 2) / (len(values) - 1)))
                check_close([mean, sd], [rec[name + "_mean"], rec[name + "_std"]])
            check("Synthetic summary recomputes: " + "/".join(key), True)
        for row in runs.itertuples():
            action = json.loads(row.chosen_action)
            check("Synthetic action valid: " + row.scenario + "/" + str(row.seed) + "/" + row.model,
                  len(action) <= 2 and len(set(action)) == len(action) and all(isinstance(i, int) and 0 <= i < 12 for i in action) and 3 not in action and 4 not in action)
            a, b = int(0 in action), int(1 in action)
            if row.scenario in ["additive", "stress_interaction"]:
                value, best = a + .7 * b, 1.7
            elif row.scenario in ["synergy", "unseen_synergy"]:
                value, best = 1.8 * a * b, 1.8
            elif row.scenario == "redundant":
                value, best = 1.2 * max(a, b), 1.2
            else:
                value, best = 0., 0.
            check_close([best - value, int(2 in action)], [row.policy_regret, row.clock_decoy_selected])
        check("Every saved policy regret and clock-decoy choice matches simulator truth", True)
        check("Unseen synergy support is absent", np.all(runs.loc[runs.scenario == "unseen_synergy", "pair01_supported"] == 0))
        check("600 distinct ranking records", len(ranked) == 600 and not ranked.duplicated(["scenario", "seed", "method"]).any())
        for key, subset in ranked.groupby(["scenario", "method"]):
            check("Ranking contains20 seeds: " + "/".join(key), set(subset.seed) == set(range(20)))
            for name in ["precision_at_2", "reporter_in_top2"]:
                values = subset[name].to_numpy(float)
                mean = float(sum(values) / len(values))
                sd = float(np.sqrt(np.sum((values - mean) ** 2) / (len(values) - 1)))
                check_close([mean, sd], [rank_summary.loc[key, (name, "mean")], rank_summary.loc[key, (name, "std")]])
        check("All synthetic ranking summary means/sampleSDs recompute", True)
        report["synthetic"] = {"scenarios": SCENARIOS, "seeds_per_scenario": 20, "model_rows": 480, "ranking_rows": 600,
                               "all_summary_cells_match": True, "saved_policy_regret_recomputed_from_action_truth": True,
                               "units": "arbitrary simulator units", "limitations": ["Per-sample synthetic model predictions were not archived, so RMSE inputs and ranking orders cannot be independently reconstructed from saved summary rows alone.", "Identity eligibility uses simulator truth; results cannot identify biological targets."]}
        report["verdict"] = "PASS"
    except Exception as error:
        report["verdict"] = "FAIL"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    return report


def static_review(root):
    entries = []
    for line in (root / "MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        entries.append({"path": relative, "match": path.is_file() and sha(path) == expected})
    credential_patterns = {
        "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "github_token": r"gh[pousr]_[A-Za-z0-9]{20,}",
        "aws_access_key": r"AKIA[0-9A-Z]{16}",
        "bearer_token": r"Bearer\s+[A-Za-z0-9_.-]{24,}",
        "openai_key": r"sk-[A-Za-z0-9_-]{24,}",
        "literal_credential_assignment": r"(?i)(?:api[_-]?key|password|secret|access[_-]?token)\s*[=:]\s*[\"\'][^\"\']{8,}",
    }
    findings = []
    for file in root.rglob("*"):
        if not file.is_file() or file.suffix.lower() in [".png", ".pdf"]:
            continue
        try:
            content = file.read_text()
        except UnicodeDecodeError:
            continue
        for name, pattern in credential_patterns.items():
            if re.search(pattern, content):
                findings.append({"path": str(file.relative_to(root)), "category": name})
    return {"verdict": "PASS" if all(e["match"] for e in entries) and not findings else "FAIL",
            "manifest_entries": len(entries), "manifest_entries_match": sum(e["match"] for e in entries),
            "manifest_failures": [e["path"] for e in entries if not e["match"]],
            "credential_pattern_findings": findings,
            "manual_source_review": ["Downloader restricted to two fixed HTTPS NCBI GEO matrices and pinned hashes.", "No remote model service or credentials needed.", "Setuptools metadata declares standard scientific dependencies; no custom install hooks found.", "Nested GitHub workflow installs dependencies and runs tests/simulations; no credential exfiltration action found.", "CSV sample metadata uses public O1/O2/O3 culture labels; no private campaign or mail records found."],
            "limits": ["Static review is a bounded inspection, not a comprehensive security audit.", "Archive-local hash consistency does not authenticate the original author or historical execution date."]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive-root", type=Path, default=HERE.parent / "CellClockAI")
    p.add_argument("--replay-root", type=Path)
    p.add_argument("--original-zip", type=Path)
    p.add_argument("--output", type=Path, default=HERE / "report.json")
    a = p.parse_args()
    report = {"checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "evidence_scope": "Recovered archive and optional fresh computational replay; no biological efficacy or mechanism verification",
              "static_archive_review": static_review(a.archive_root), "archived_results": verify_results(a.archive_root)}
    if a.original_zip:
        with zipfile.ZipFile(a.original_zip) as z:
            unsafe = [n for n in z.namelist() if Path(n).is_absolute() or ".." in Path(n).parts]
        report["original_archive"] = {"sha256": sha(a.original_zip), "bytes": a.original_zip.stat().st_size, "unsafe_paths": unsafe}
    if a.replay_root:
        report["fresh_replay"] = verify_results(a.replay_root)
        left = pd.read_csv(a.archive_root / "results/public_predictions.csv").sort_values("sample")
        right = pd.read_csv(a.replay_root / "results/public_predictions.csv").sort_values("sample")
        report["archive_vs_replay"] = {"same_public_samples": left["sample"].tolist() == right["sample"].tolist(),
                                       "max_abs_probability_difference": float(np.max(np.abs(left.leave_donor_out_probability.to_numpy() - right.leave_donor_out_probability.to_numpy()))),
                                       "identical_public_prediction_sha256": sha(a.archive_root / "results/public_predictions.csv") == sha(a.replay_root / "results/public_predictions.csv")}
        compared_files = ["public_association_genes.csv", "public_marker_changes.csv", "public_predictions.csv", "public_permutation_null.csv", "synthetic_runs.csv", "synthetic_rankings.csv", "synthetic_summary.csv", "synthetic_ranking_summary.csv", "prospective_power.csv", "public_summary.json", "synthetic_metadata.json"]
        report["archive_vs_replay"]["result_files"] = [{"path": f, "byte_identical": sha(a.archive_root / "results" / f) == sha(a.replay_root / "results" / f)} for f in compared_files]
        original_sources = [p for p in a.archive_root.rglob("*") if p.is_file() and p.suffix in [".py", ".toml"]]
        report["archive_vs_replay"]["source_files"] = [{"path": str(p.relative_to(a.archive_root)), "byte_identical": (a.replay_root / p.relative_to(a.archive_root)).is_file() and sha(p) == sha(a.replay_root / p.relative_to(a.archive_root))} for p in original_sources]
        report["archive_vs_replay"]["verdict"] = "PASS" if all(x["byte_identical"] for x in report["archive_vs_replay"]["result_files"] + report["archive_vs_replay"]["source_files"]) else "FAIL"
    else:
        report["fresh_replay"] = {"verdict": "UNKNOWN", "reason": "No completed fresh replay supplied to verifier"}
    report["verifier_source_sha256"] = sha(Path(__file__))
    report["verdict"] = "PASS" if report["static_archive_review"]["verdict"] == report["archived_results"]["verdict"] == "PASS" and report["fresh_replay"]["verdict"] in ["PASS", "UNKNOWN"] and report.get("archive_vs_replay", {}).get("verdict", "PASS") == "PASS" else "FAIL"
    a.output.parent.mkdir(exist_ok=True, parents=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verdict": report["verdict"], "archive_integrity": report["static_archive_review"]["verdict"], "archive_metrics": report["archived_results"]["verdict"], "archive_checks": len(report["archived_results"]["checks"]), "fresh_replay": report["fresh_replay"]["verdict"], "failure": report["archived_results"].get("failure")}, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
