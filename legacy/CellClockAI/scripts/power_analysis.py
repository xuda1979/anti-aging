"""Prospective paired donor power illustration, not an executed biological study."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import nct, t

ROOT = Path(__file__).resolve().parents[1]


def power(n, standardized_effect=.8, alpha=.05 / 20):
    critical = t.ppf(1 - alpha / 2, n - 1)
    noncentrality = standardized_effect * np.sqrt(n)
    return float(nct.sf(critical, n - 1, noncentrality) + nct.cdf(-critical, n - 1, noncentrality))


if __name__ == "__main__":
    rows = [{"independent_donors": n, "effect_sd": d, "alpha": .0025,
             "power": power(n, d), "evidence_type": "prospective_design_calculation"}
            for d in [.5, .8, 1.] for n in [8, 16, 24, 30, 34, 40, 60]]
    pd.DataFrame(rows).to_csv(ROOT / "results/prospective_power.csv", index=False)
    print(f"30 donors, effect 0.8 SD, alpha 0.0025: {power(30):.4f}")
