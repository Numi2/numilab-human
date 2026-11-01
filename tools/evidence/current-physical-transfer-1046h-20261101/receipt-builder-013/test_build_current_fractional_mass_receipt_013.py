import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("build_current_fractional_mass_receipt_013.py")
SPEC = importlib.util.spec_from_file_location("receipt013", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


class ReceiptBuilder013Tests(unittest.TestCase):
    def fixture(self, root):
        source_mass = root / "source-mass.csv"
        labels = root / "source-labels.csv"
        tets = root / "source-tets.csv"
        child_labels = root / "child-labels.csv"
        write_csv(source_mass, MOD.MASS_HEADER, [
            [0, 15, "0.6"], [1, 16, "0.4"],
        ])
        write_csv(labels, MOD.LABEL_HEADER, [
            [0, 15, 5, 7, 100], [1, 16, 9, 10, 101],
        ])
        write_csv(tets, MOD.TET_HEADER, [
            [0, 15, 5, 0, 1, 2, 3, "0.0006", "0.6"],
            [1, 16, 9, 4, 5, 6, 7, "0.0004444444444444444", "0.4"],
        ])
        write_csv(child_labels, MOD.CHILD_LABEL_HEADER, [
            [0, 15, 12, 7, 100], [1, 16, 13, 10, 101],
        ])
        refs = {
            "source330_mass": {"path": str(source_mass), "sha256": MOD.sha_file(source_mass)},
            "source330_labels": {"path": str(labels), "sha256": MOD.sha_file(labels)},
            "source330_tets": {"path": str(tets), "sha256": MOD.sha_file(tets)},
            "child_skin_labels": {"path": str(child_labels), "sha256": MOD.sha_file(child_labels)},
        }
        out = root / "outputs"
        out.mkdir()
        write_csv(out / "flow.csv", MOD.FLOW_HEADER, [
            [0, 0, "0.6", "0.0006", "1.0", "0.01"],
            [1, 1, "0.4", "0.0004444444444444444", "1.0", "0.012"],
        ])
        write_csv(out / "semantic-groups.csv", MOD.GROUP_HEADER, [
            [0, 0, 15, 5, 7, "1.0", "0.6"],
            [1, 1, 16, 9, 10, "1.0", "0.4"],
        ])
        write_csv(out / "source-donors.csv", MOD.DONOR_HEADER, [
            [15, 5, "0.6", "0.6", "0"],
            [16, 9, "0.4", "0.4", "0"],
        ])
        write_csv(out / "carrier-trace.csv", MOD.TRACE_HEADER, [
            [0, 0, 0, 0, 0, "0.6", "0.0006", "0.01"],
            [1, 1, 1, 1, 0, "0.4", "0.0004444444444444444", "0.012"],
        ])
        output_log = {}
        keys = {
            "flow": ("flow_path", "flow_sha256", "flow.csv"),
            "semantic_groups": ("semantic_groups_path", "semantic_groups_sha256", "semantic-groups.csv"),
            "source_donors": ("source_donors_path", "source_donors_sha256", "source-donors.csv"),
            "carrier_trace": ("carrier_trace_path", "carrier_trace_sha256", "carrier-trace.csv"),
        }
        for key, (pk, hk, name) in keys.items():
            path = out / name
            output_log[pk] = str(path.resolve())
            output_log[hk] = MOD.sha_file(path)
        return refs, out, output_log

    def test_synthetic_four_csv_emitter_derives_all_closures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs, out, outputs_log = self.fixture(root)
            derived = MOD.derive_outputs(out, refs, {}, outputs_log, MOD.MASS_LIMIT)
            self.assertEqual(derived["parent_count"], 2)
            self.assertEqual(derived["group_count"], 2)
            self.assertEqual(derived["donor_row_count"], 2)
            self.assertEqual(derived["flow_edge_count"], 2)
            self.assertEqual(derived["source_mass_kg"], 1.0)
            self.assertEqual(derived["parent_mass_closure"], 0.0)
            self.assertEqual(derived["parent_measure_closure"], 0.0)
            self.assertEqual(derived["group_mass_closure"], 0.0)
            self.assertEqual(derived["shared_child_capacity_violation"], 0.0)
            self.assertEqual(derived["source_donor_closure"], 0.0)
            self.assertEqual(derived["density_by_material"]["15"]["min_kgm3"], 1000.0)
            self.assertAlmostEqual(derived["density_by_material"]["16"]["min_kgm3"], 900.0)

    def test_trace_gap_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs, out, outputs_log = self.fixture(root)
            write_csv(out / "carrier-trace.csv", MOD.TRACE_HEADER, [
                [0, 0, 0, 0, 0, "0.6", "0.0006", "0.01"],
            ])
            outputs_log["carrier_trace_sha256"] = MOD.sha_file(out / "carrier-trace.csv")
            with self.assertRaisesRegex(MOD.ReceiptError, "does not cover every flow edge"):
                MOD.derive_outputs(out, refs, {}, outputs_log, MOD.MASS_LIMIT)

    def test_exact_tuple_mismatch_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs, out, outputs_log = self.fixture(root)
            write_csv(out / "semantic-groups.csv", MOD.GROUP_HEADER, [
                [0, 0, 15, 5, 99, "1.0", "0.6"],
                [1, 1, 16, 9, 10, "1.0", "0.4"],
            ])
            outputs_log["semantic_groups_sha256"] = MOD.sha_file(out / "semantic-groups.csv")
            with self.assertRaises(MOD.ReceiptError):
                MOD.derive_outputs(out, refs, {}, outputs_log, MOD.MASS_LIMIT)

    def test_all_supplemental_file_pins_are_rehashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.bin"
            path.write_bytes(b"pinned")
            good = {"file_pins": [{"path": str(path.resolve()), "sha256": MOD.sha_file(path),
                                  "bytes": path.stat().st_size}]}
            MOD.validate_file_pins(good)
            bad = {"file_pins": [{"path": str(path.resolve()), "sha256": "0" * 64}]}
            with self.assertRaisesRegex(MOD.ReceiptError, "file pin mismatch"):
                MOD.validate_file_pins(bad)
            with self.assertRaisesRegex(MOD.ReceiptError, "lacks its supplemental file_pins"):
                MOD.validate_file_pins({})

    def test_source_contribution_pin_requires_exact_sibling_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mass = root / "parent-mass.csv"
            contribution = root / "parent-mass.csv.source.csv"
            mass.write_text("mass", encoding="utf-8")
            contribution.write_text("donor contributions", encoding="utf-8")
            refs = {"source330_mass": {"path": str(mass.resolve()), "sha256": MOD.sha_file(mass)}}
            good_entry = {"path": str(contribution.resolve()), "sha256": MOD.sha_file(contribution),
                          "bytes": contribution.stat().st_size}
            run = {"file_pins": [good_entry]}
            result = MOD.validate_parent_source_contributions_pin(run, refs)
            self.assertEqual(result, {"path": str(contribution.resolve()), "sha256": MOD.sha_file(contribution)})
            bad = {"file_pins": [{"path": str(mass.resolve()), "sha256": MOD.sha_file(mass)}]}
            with self.assertRaisesRegex(MOD.ReceiptError, "exactly one source330 mass contribution sibling"):
                MOD.validate_parent_source_contributions_pin(bad, refs)
            bad_sha = {"file_pins": [{"path": str(contribution.resolve()), "sha256": "0" * 64}]}
            with self.assertRaisesRegex(MOD.ReceiptError, "file pin mismatch"):
                MOD.validate_parent_source_contributions_pin(bad_sha, refs)
            duplicate = {"file_pins": [good_entry, good_entry]}
            with self.assertRaisesRegex(MOD.ReceiptError, "exactly one source330 mass contribution sibling"):
                MOD.validate_parent_source_contributions_pin(duplicate, refs)
            legacy = {"extra_file_pin": good_entry, "file_pins": [good_entry]}
            with self.assertRaisesRegex(MOD.ReceiptError, "must be pinned in run.file_pins"):
                MOD.validate_parent_source_contributions_pin(legacy, refs)
            with self.assertRaisesRegex(MOD.ReceiptError, "lacks its supplemental file_pins"):
                MOD.validate_parent_source_contributions_pin({}, refs)

    def test_actual_failed_007_is_rejected_and_failure_witness_written(self):
        base = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-fractional-semantic-native-1047/actual-007-all-parent-coverage")
        run = base / "run-manifest.json"
        execution = base / "execution.json"
        stdout = base / "run.log"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "should-not-exist.json"
            failure = Path(tmp) / "failure.json"
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--run-manifest", str(run),
                "--execution", str(execution), "--stdout-log", str(stdout),
                "--identity-config", "/absent/identity.json",
                "--control-manifest", "/absent/controls.json",
                "--output", str(out), "--failure-report", str(failure),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(out.exists())
            report = json.loads(failure.read_text())
            self.assertEqual(report["status"], "REJECTED")
            self.assertIn("exit code 6", report["reason"])


if __name__ == "__main__":
    unittest.main()

