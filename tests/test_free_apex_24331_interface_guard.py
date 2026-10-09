"""Negative CLI regression for mapped 24331 owner moves in the 1138 assembler."""
import json
import pathlib
import subprocess
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
EVIDENCE = pathlib.Path("/Users/n/numi-human-resting-evidence-20261005")
ASSEMBLER = REPO / "tools/evidence/native-lung-free-apex-composition-1159/compose_selected_free_apices.py"
ORIGINAL_ASSEMBLER = EVIDENCE / "compose_selected_free_apices_1138.py"
OPERATIONS = REPO / "tests/fixtures/free_apex_24331_registered_interface_guard.json"
CANDIDATE_REPORT = EVIDENCE / "native-lung-left-cluster1-freeapex-trials-1143/report.json"
TRIAL_REPORT = EVIDENCE / "native-lung-left-star-opening-trials-1131/report.json"
CLI_PYTHON = pathlib.Path("/Users/n/numi-human-prep-venv-20261005/bin/python3")

def sha(path):
    import hashlib
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

@unittest.skipUnless(
    ASSEMBLER.is_file() and ORIGINAL_ASSEMBLER.is_file() and OPERATIONS.is_file()
    and CANDIDATE_REPORT.is_file() and TRIAL_REPORT.is_file() and CLI_PYTHON.is_file(),
    "pinned Mac-mini 1138 assembler evidence is unavailable",
)
class FreeApex24331InterfaceGuardTests(unittest.TestCase):
    def test_owner_accepted_mapped_move_fails_cli_guard_before_final_asset(self):
        # Both operations are retained owner-accepted trials, not synthetic points.
        fixture = json.loads(OPERATIONS.read_text())
        self.assertEqual(len(fixture), 2)
        first, mapped = fixture
        self.assertEqual(first["label"], "AB24331-precondition-305-21508")
        self.assertEqual(mapped["label"], "AB24331-mapped-305-308-interface-negative")
        self.assertEqual(first["expected_owner_ids"], [305, 310])
        self.assertEqual(mapped["expected_owner_ids"], [305, 308, 310])
        self.assertEqual(
            mapped["arguments"]["point"],
            [-0.0019057018216699362, 0.04490205645561218, -0.06111486628651619],
        )
        self.assertEqual(mapped["arguments"]["displacement_m"], 1e-6)

        source_report = json.loads(CANDIDATE_REPORT.read_text())
        target = source_report["operation2_target"]
        accepted = source_report["second_magnitude_trials"][0]
        self.assertEqual(source_report["source_nha"]["sha256"],
                         "1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc")
        self.assertEqual(target["vertex_id"], 24331)
        self.assertEqual(target["exact_source_owners"], [
            {"stable_id": 305, "vertex_index": 21512},
            {"stable_id": 308, "vertex_index": 24331},
            {"stable_id": 310, "vertex_index": 21514},
        ])
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(accepted["operation"]["owner_surface_ids"], [305, 308, 310])
        self.assertEqual(mapped["expected_new_point"], accepted["operation"]["new_point"])

        trial_report = json.loads(TRIAL_REPORT.read_text())
        first_trial = next(
            x for x in trial_report["trials"]
            if x["cluster"] == "1" and x["requested_displacement_m"] == 1e-6
        )
        self.assertEqual(first_trial["source_vertex_id"], 21508)
        self.assertEqual(first["arguments"]["point"], first_trial["point_f32"])
        self.assertEqual(first["arguments"]["direction"], first_trial["selected_direction"])
        self.assertEqual(first["expected_new_point"], first_trial["operation"]["new_point"])

        self.assertEqual(
            sha(ASSEMBLER),
            "955b4d501287294a632198feda478bf8eb431f0c882cdb12bf315f196315ea2f",
            "the tested CLI copy must remain the exact retained 1138 assembler",
        )
        self.assertEqual(sha(ORIGINAL_ASSEMBLER), sha(ASSEMBLER))
        self.assertEqual(
            sha(CANDIDATE_REPORT),
            "83ed67747faab6a51e7c79606febe48182e1682dd9420284b1fd288b07b2fc58",
        )
        self.assertEqual(
            sha(TRIAL_REPORT),
            "ed77d30f17a6ef222745065d4826905aef3b6382b0557d775891aa143666f4c9",
        )

        with tempfile.TemporaryDirectory(prefix="free-apex-24331-negative-") as tmp:
            # Control: the preceding 305/310 move alone is admitted by this
            # real CLI and completes. This isolates the next rejection to the
            # mapped 24331 move.
            control_operations = pathlib.Path(tmp) / "control-operations.json"
            control_operations.write_text(json.dumps([first], indent=2) + "\n")
            control_out = pathlib.Path(tmp) / "control-output"
            control = subprocess.run(
                [str(CLI_PYTHON), str(ASSEMBLER), "--operations", str(control_operations), "--out", str(control_out)],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(control.returncode, 0, control.stdout + control.stderr)
            self.assertIn("owner operation 0 AB24331-precondition-305-21508 accepted", control.stdout)
            self.assertTrue((control_out / "final/resting-thorax.nhanatomy").is_file())

            out = pathlib.Path(tmp) / "fresh-negative-output"
            result = subprocess.run(
                [str(CLI_PYTHON), str(ASSEMBLER), "--operations", str(OPERATIONS), "--out", str(out)],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            # These lines print only after each existing-owner call returns the
            # evidence-matched owner set and Float32 point.
            self.assertIn("owner operation 0 AB24331-precondition-305-21508 accepted", result.stdout)
            self.assertIn(
                "owner operation 1 AB24331-mapped-305-308-interface-negative accepted",
                result.stdout,
            )
            self.assertIn("ValueError: registered interface moved", result.stderr)
            self.assertTrue(out.is_dir())
            progress = json.loads((out / "operation-progress.json").read_text())
            self.assertEqual(len(progress), 2)
            self.assertEqual(progress[1]["result"]["owner_surface_ids"], [305, 308, 310])
            # The immutable source-map guard rejects before pre-pleura/final emission.
            self.assertFalse((out / "pre-pleura").exists())
            self.assertFalse((out / "final").exists())
            self.assertEqual(list(out.rglob("*.nhanatomy")), [])

if __name__ == "__main__":
    unittest.main()
