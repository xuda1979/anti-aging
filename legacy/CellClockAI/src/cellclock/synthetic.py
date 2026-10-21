"""Synthetic falsification benchmark. Reg00--Reg11 are invented variables.

The generator encodes the causal truth by construction. Performance here cannot
establish any biological mechanism, gene target, rejuvenation, or lifespan effect.
"""
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_REG = 12
SCENARIOS = ["additive", "synergy", "redundant", "stress_interaction", "unseen_synergy", "no_causal_effect"]


def benefit(u, z, scenario):
    a, b = u[:, 0], u[:, 1]
    if scenario == "additive":
        return 1.0 * a + .7 * b
    if scenario in {"synergy", "unseen_synergy"}:
        return 1.8 * a * b
    if scenario == "redundant":
        return 1.2 * np.maximum(a, b)
    if scenario == "stress_interaction":
        return (1.0 + .4 * z[:, 0]) * a + .7 * b
    if scenario == "no_causal_effect":
        return np.zeros(len(u))
    raise ValueError(scenario)


def generate(seed, n, scenario, training=False):
    rng = np.random.default_rng(seed)
    u = rng.binomial(1, .2, size=(n, N_REG)).astype(float)
    z = rng.normal(0 if training else .5, 1, size=(n, 2))
    if training and scenario == "unseen_synergy":
        # Deliberately remove every coactivation of the two true gates.
        u[(u[:, 0] * u[:, 1]) == 1, 1] = 0
    baseline = 2.0 + .5 * z[:, 0] + .2 * z[:, 1]
    functional_mean = -baseline + benefit(u, z, scenario)
    functional = functional_mean + rng.normal(0, .15, n)
    # A clock reporter can be manipulated without changing functional burden.
    clock = functional_mean + 2.0 * u[:, 2] + rng.normal(0, .15, n)
    observed = .6 * u + rng.normal(0, .5, (n, N_REG))
    observed[:, 2] = functional_mean + rng.normal(0, .03, n)  # downstream reporter
    identity_loss = .35 * u[:, 3] + .5 * u[:, 4]
    return {"u": u, "z": z, "x": np.column_stack([u, z]), "function": functional,
            "function_mean": functional_mean, "clock": clock, "observed": observed,
            "identity_loss": identity_loss}


def interaction_features(x):
    x = np.asarray(x)
    u, z = x[:, :N_REG], x[:, N_REG:]
    pairs = np.column_stack([u[:, i] * u[:, j] for i, j in itertools.combinations(range(N_REG), 2)])
    uz = np.column_stack([u[:, i] * z[:, j] for i in range(N_REG) for j in range(z.shape[1])])
    return np.column_stack([x, pairs, uz])


class InteractionRidge:
    def fit(self, x, y):
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        self.model.fit(interaction_features(x), y)
        return self

    def predict(self, x):
        return self.model.predict(interaction_features(x))


def marginal_effects(model, test):
    x = test["x"][:200].copy()
    effects = []
    for j in range(N_REG):
        off, on = x.copy(), x.copy()
        off[:, j], on[:, j] = 0, 1
        effects.append(float((model.predict(on) - model.predict(off)).mean()))
    return np.asarray(effects)


def supported_pair(u, pair, minimum=20):
    return bool(np.sum((u[:, pair[0]] == 1) & (u[:, pair[1]] == 1)) >= minimum)


def policy_regret(model, scenario):
    actions = [np.zeros(N_REG)]
    for k in [1, 2]:
        for ids in itertools.combinations(range(N_REG), k):
            a = np.zeros(N_REG)
            a[list(ids)] = 1
            actions.append(a)
    u = np.asarray(actions)
    z = np.zeros((len(u), 2))
    # In this pilot the identity constraint is supplied from simulator truth.
    # The biological study must measure/predict it independently.
    valid = .35 * u[:, 3] + .5 * u[:, 4] <= .15
    candidates = np.flatnonzero(valid)
    values = model.predict(np.column_stack([u, z]))
    choice = candidates[np.argmax(values[candidates])]
    true = benefit(u, z, scenario)
    return float(true[candidates].max() - true[choice]), bool(u[choice, 2]), np.flatnonzero(u[choice]).tolist()


def run(root, seeds=20, n_train=900, n_test=600):
    root = Path(root)
    records, ranking_records = [], []
    for scenario in SCENARIOS:
        for seed in range(seeds):
            train = generate(10000 + seed, n_train, scenario, training=True)
            test = generate(20000 + seed, n_test, scenario)
            correlations = np.array([np.corrcoef(train["observed"][:, j], train["function"])[0, 1]
                                     for j in range(N_REG)])
            ranking = np.argsort(-np.abs(correlations), kind="stable")
            true_gates = {0, 1} if scenario != "no_causal_effect" else set()
            ranking_records.append({"scenario": scenario, "seed": seed, "method": "observational_correlation",
                                    "precision_at_2": len(set(ranking[:2]) & true_gates) / 2,
                                    "reporter_in_top2": int(2 in ranking[:2])})
            models = {
                "linear_function": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                "interaction_function": InteractionRidge(),
                "neural_function": make_pipeline(StandardScaler(), MLPRegressor(
                    hidden_layer_sizes=(32,), activation="tanh", solver="adam", alpha=.01,
                    max_iter=1000, learning_rate_init=.003, early_stopping=True, n_iter_no_change=30,
                    validation_fraction=.15, random_state=seed)),
                "interaction_clock": InteractionRidge(),
            }
            for name, model in models.items():
                target = train["clock"] if name.endswith("clock") else train["function"]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(train["x"], target)
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                effects = marginal_effects(model, test)
                ranking = np.argsort(-effects, kind="stable")
                regret, decoy, action = policy_regret(model, scenario)
                rmse = float(np.sqrt(np.mean((model.predict(test["x"]) - test["function_mean"]) ** 2)))
                records.append({"scenario": scenario, "seed": seed, "model": name,
                                "functional_mean_rmse": rmse, "policy_regret": regret,
                                "clock_decoy_selected": int(decoy), "chosen_action": json.dumps(action),
                                "pair01_supported": int(supported_pair(train["u"], (0, 1))),
                                "converged": int(converged), "n_train": n_train, "n_test": n_test})
                ranking_records.append({"scenario": scenario, "seed": seed, "method": name,
                                        "precision_at_2": len(set(ranking[:2]) & true_gates) / 2,
                                        "reporter_in_top2": int(2 in ranking[:2])})
            print(f"synthetic {scenario} seed {seed + 1}/{seeds}", flush=True)
    metrics, ranks = pd.DataFrame(records), pd.DataFrame(ranking_records)
    metrics.to_csv(root / "results/synthetic_runs.csv", index=False)
    ranks.to_csv(root / "results/synthetic_rankings.csv", index=False)
    summary = metrics.groupby(["scenario", "model"])[["functional_mean_rmse", "policy_regret",
                "clock_decoy_selected", "pair01_supported", "converged"]].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(root / "results/synthetic_summary.csv")
    ranks.groupby(["scenario", "method"])[["precision_at_2", "reporter_in_top2"]].agg(
        ["mean", "std"]).to_csv(root / "results/synthetic_ranking_summary.csv")
    metadata = {"evidence_type": "synthetic_benchmark_only", "seeds": seeds,
                "scenarios": SCENARIOS, "n_regulators": N_REG, "n_train": n_train,
                "n_test": n_test, "units": "arbitrary simulator units",
                "support_minimum_coactivation": 20,
                "causal_truth": "Reg00 and Reg01, except no_causal_effect; Reg02 is a clock-only decoy",
                "caveats": ["Identity constraint uses simulator truth", "All actions are randomized",
                            "Measured outcomes and upstream action assignment are available",
                            "Pair support check is a conservative heuristic, not an identification theorem"]}
    (root / "results/synthetic_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
