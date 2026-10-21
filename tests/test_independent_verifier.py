"""Independent verifier contracts, including an actual corrupted CSV record."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("independent_verification", ROOT / "scripts/verify_study.py")
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)


class IndependentContracts(unittest.TestCase):
    def test_known_numeric_fixture(self):
        got = independent.scores_independent([20, 40, 60], [22, 38, 66])
        self.assertAlmostEqual(got["mae"], 10 / 3)
        self.assertAlmostEqual(got["rmse"], np.sqrt(44 / 3))
        self.assertAlmostEqual(got["r2"], 1 - 44 / 800)
        self.assertEqual(got["median_absolute_error"], 2)

    def test_nonfinite_measurement_rejected(self):
        with self.assertRaises(AssertionError):
            independent.scores_independent([20, 40], [22, np.nan])

    @unittest.skipUnless((ROOT / "results/pilot_20261021_v2/predictions.csv").exists(), "Real run not yet available")
    def test_corrupt_prediction_csv_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Path(tmp)
            for directory in ["configs", "data", "models"]:
                (fixture / directory).symlink_to(ROOT / directory, target_is_directory=True)
            (fixture / "verification").mkdir()
            shutil.copy2(ROOT / "verification/independent_preflight.json", fixture / "verification/independent_preflight.json")
            destination = fixture / "results/pilot_20261021_v2"
            shutil.copytree(ROOT / "results/pilot_20261021_v2", destination)
            table = pd.read_csv(destination / "predictions.csv")
            table.loc[0, "prediction_ridge"] += 10
            table.to_csv(destination / "predictions.csv", index=False)
            completed = subprocess.run([sys.executable, str(ROOT / "scripts/verify_study.py"), "--root", str(fixture)], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            report = json.loads((fixture / "verification/report.json").read_text())
            self.assertEqual(report["verdict"], "FAIL")
            self.assertIn("Checkpoint predictions differ", report["failure"]["message"])


if __name__ == "__main__":
    unittest.main()
