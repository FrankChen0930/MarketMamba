import tempfile
import unittest
from pathlib import Path

from v7_stability_suite import (
    make_window_experiment_spec,
    run_window_sequence,
    window_training_settings,
)


class StabilitySuiteTests(unittest.TestCase):
    def test_window_spec_and_settings_freeze_e5_seed17_without_test_selection(self):
        window = {
            "id": "W2024",
            "train": ["2013-01-02", "2022-11-17"],
            "selection": ["2023-01-03", "2023-12-29"],
            "evaluation": ["2024-01-02", "2024-12-31"],
            "evaluation_used_for_selection": False,
        }
        spec = make_window_experiment_spec(window)
        settings = window_training_settings()
        self.assertEqual(spec["id"], "W2024-E5-seed17")
        self.assertEqual(spec["config"]["d_model"], 64)
        self.assertEqual(spec["config"]["d_state"], 32)
        self.assertEqual(spec["config"]["temporal_layers"], 1)
        self.assertEqual(spec["config"]["forward_layers"], 1)
        self.assertEqual(spec["config"]["reverse_layers"], 1)
        self.assertEqual(spec["config"]["train_cutoff"], "2022-11-17")
        self.assertEqual(spec["config"]["validation_end"], "2023-12-29")
        self.assertNotIn("2024-12-31", repr(spec))
        self.assertEqual(settings["seed"], 17)
        self.assertEqual(settings["epochs"], 20)

    def test_window_sequence_resumes_paused_and_skips_completed(self):
        windows = [{"id": name} for name in ("W2024", "W2025", "W2026")]
        calls = []
        pause_once = {"value": True}

        def run_one(window):
            calls.append(window["id"])
            if window["id"] == "W2024" and pause_once["value"]:
                pause_once["value"] = False
                return {"status": "paused", "step": 3}
            return {"status": "complete", "experiment": window["id"]}

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            first = run_window_sequence(windows, output, "suite-id", run_one)
            self.assertEqual(first["status"], "paused")
            second = run_window_sequence(windows, output, "suite-id", run_one)
            self.assertEqual(second["status"], "complete")
            self.assertEqual(calls, ["W2024", "W2024", "W2025", "W2026"])
            verified = []
            third = run_window_sequence(
                windows, output, "suite-id", run_one,
                verify_completed=lambda window, result: verified.append((window["id"], result["experiment"])),
            )
            self.assertEqual(third["status"], "skipped")
            self.assertEqual(verified, [("W2024", "W2024"), ("W2025", "W2025"), ("W2026", "W2026")])
            self.assertEqual(calls, ["W2024", "W2024", "W2025", "W2026"])
            with self.assertRaisesRegex(ValueError, "identity"):
                run_window_sequence(windows, output, "changed-id", run_one)


if __name__ == "__main__":
    unittest.main()
