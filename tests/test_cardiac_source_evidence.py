"""Tamper checks for retained evidence; no native execution or physical stepping."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import json
import math
import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cardiac_source_verifier", ROOT / "tools/verify_cardiac_source_20260912.py")
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class CardiacNumericalVerifierTests(unittest.TestCase):
    """Small arithmetic/shape fixtures test the verifier, not cardiovascular physics."""

    def test_first_order_gate_cannot_be_satisfied_by_one_decreasing_metric(self) -> None:
        runs = {}
        for index, name in enumerate(["refinement_2ms", "refinement_1ms", "refinement_05ms"]):
            runs[name] = {"duration_seconds": 1., **{key: 1./(2**index) for key in VERIFY.METRICS}}
        self.assertEqual(VERIFY.audit_native_refinement(runs)[VERIFY.METRICS[0]], [2., 2.])
        runs["refinement_05ms"]["max_flow_error_m3_per_s"] = .4
        with self.assertRaisesRegex(VERIFY.EvidenceError, "first-order timestep refinement failed"):
            VERIFY.audit_native_refinement(runs)

    def test_equal_step_count_cannot_replace_equal_elapsed_duration(self) -> None:
        runs = {name: {"duration_seconds": 1., **{key: 1./(2**i) for key in VERIFY.METRICS}}
                for i, name in enumerate(["refinement_2ms", "refinement_1ms", "refinement_05ms"])}
        runs["refinement_05ms"]["duration_seconds"] = .5
        with self.assertRaisesRegex(VERIFY.EvidenceError, "same accepted duration"):
            VERIFY.audit_native_refinement(runs)

    def test_nominal_decimal_time_is_distinct_from_cooked_clock(self) -> None:
        self.assertNotEqual(500*.002, 500*VERIFY.float32(.002))
        self.assertEqual(500*VERIFY.float32(.002), 2000*VERIFY.float32(.0005))

    def test_nonfinite_or_wrong_coordinate_arity_is_rejected(self) -> None:
        with self.assertRaisesRegex(VERIFY.EvidenceError, "nonfinite trace"):
            VERIFY.numbers(["0", "nan"], 2)
        with self.assertRaisesRegex(VERIFY.EvidenceError, "coordinate arity"):
            VERIFY.numbers(["0", "1"], 3)

    def test_trace_parser_rejects_resealed_numeric_tampering(self) -> None:
        # Two artificial arithmetic rows exercise the validator in isolation.
        # The patched fixture run is never admissible to the full receipt check.
        initial = [.0001]*10 + [0.]*10
        expected = initial.copy()
        expected[0] -= .000002
        expected[2] -= .000001
        expected[10] += .000003
        header = ["time_seconds"] + [name+str(i) for i in range(20) for name in ("native_", "cellml_")]
        rows = [[index*VERIFY.float32(.002)] + [value for pair in zip(initial, expected) for value in pair]
                for index in [1, 2]]
        log = ("cardiac_device=Apple M4 Pro abi=27 world_fingerprint=1\n"
               "cardiac_source_run=pass accepted_steps=2 environments=2 failed_steps=0 "
               f"dt_seconds={VERIFY.float32(.002)} duration_seconds={2*VERIFY.float32(.002)} "
               "clock=exact_binary_128 replay_pair=bitwise max_chamber_volume_error_m3=.000002 "
               "max_storage_error_m3=.000001 max_flow_error_m3_per_s=.000003 "
               f"scaled_state_rms_error={math.sqrt(.00007)} relative_storage_invariant_error=0 "
               "reference_accepted_steps=2 reference_rejected_steps=0 wall_seconds=1 "
               "qualification=source_model_numerical_comparison absolute_vascular_blood_volume=unqualified "
               "biological_calibration=unqualified\n")
        fixture_limits = dict(zip(VERIFY.METRICS, [3e-6, 2e-6, 4e-6, .01]))
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(VERIFY.RUNS, {"fixture": (.002, 2)}, clear=True), \
                mock.patch.dict(VERIFY.ACCURACY_LIMITS, {"fixture": fixture_limits}, clear=True):
            trace = Path(temp) / "arithmetic-fixture.csv.gz"

            def check(data, summary=log):
                stream = io.StringIO()
                csv.writer(stream, lineterminator="\n").writerows([header, *data])
                trace.write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))
                return VERIFY.audit_native_run(trace, summary, {"id": "fixture", "dt_seconds": .002, "steps": 2},
                                               [.0001]*20, initial)

            self.assertEqual(check(rows)["steps"], 2)
            with self.assertRaisesRegex(VERIFY.EvidenceError, "trace is truncated"):
                check(rows[:1])
            altered = [row.copy() for row in rows]
            altered[0][0] = .002
            with self.assertRaisesRegex(VERIFY.EvidenceError, "exact cooked clock"):
                check(altered)
            altered = [row.copy() for row in rows]
            altered[0][1] = .001
            with self.assertRaisesRegex(VERIFY.EvidenceError, "storage conservation"):
                check(altered)
            with self.assertRaisesRegex(VERIFY.EvidenceError, "max_flow_error_m3_per_s disagrees"):
                check(rows, log.replace("max_flow_error_m3_per_s=.000003", "max_flow_error_m3_per_s=.000001"))
            with self.assertRaisesRegex(VERIFY.EvidenceError, "native source run does not meet"):
                check(rows, log.replace("replay_pair=bitwise", "replay_pair=diverged"))
            altered = [row.copy() for row in rows]
            altered[0][1] = -.0001
            altered[0][5] += .0002
            with self.assertRaisesRegex(VERIFY.EvidenceError, "nonpositive absolute chamber volume"):
                check(altered)
            altered = [row.copy() for row in rows]
            altered[0][21] = -.000001
            with self.assertRaisesRegex(VERIFY.EvidenceError, "negative one-way valve flow"):
                check(altered)
            altered = [row.copy() for row in rows]
            for row in altered:
                row[2] = initial[0]-.000004
            worse = log.replace("max_chamber_volume_error_m3=.000002", "max_chamber_volume_error_m3=.000004")
            worse = worse.replace(str(math.sqrt(.00007)), str(math.sqrt(.00013)))
            with self.assertRaisesRegex(VERIFY.EvidenceError, "numerical regression envelope exceeded"):
                check(altered, worse)


class CardiacSourceEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="cardiac-evidence-test-")
        self.addCleanup(temporary.cleanup)
        self.repository = Path(temporary.name) / "repository"
        self.bundle = self.repository / "evidence"
        source = ROOT / "Docs/media/cardiac-source-20260912"
        self.assertTrue((source / "receipt.json").is_file(), "Full cardiac receipt is required; partial evidence cannot pass")
        shutil.copytree(source, self.bundle)
        self.receipt_path = self.bundle / "receipt.json"
        self.receipt = VERIFY.read_json(self.receipt_path)
        for relative in self.receipt["source_sha256"]:
            target = self.repository / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)

    def verify(self) -> dict:
        self.receipt_path.write_text(json.dumps(self.receipt))
        return VERIFY.verify(self.receipt_path, repository_root=self.repository)

    def rejected(self, message: str) -> None:
        with self.assertRaisesRegex(VERIFY.EvidenceError, message):
            self.verify()

    def reseal(self, name: str) -> None:
        matches = [row for row in self.receipt["artifacts"] if row["path"] == name]
        self.assertEqual(len(matches), 1)
        matches[0]["sha256"] = VERIFY.file_hash(self.bundle / name)

    def run_record(self, name: str) -> dict:
        return next(row for row in self.receipt["native_runs"] if row["id"] == name)

    def reseal_run_output(self, name: str) -> None:
        """Reseal output metadata too, to exercise independent numeric checks."""
        self.reseal(name)
        for run in self.receipt["native_runs"]:
            if name not in {run["log"], run["trace"]}:
                continue
            path = self.bundle / run["identity"]
            identity = VERIFY.read_json(path)
            if name == run["log"]:
                identity["log_sha256"] = VERIFY.file_hash(self.bundle / name)
            else:
                identity["trace_raw_sha256"] = VERIFY.csv_data(self.bundle / name)[2]
            path.write_text(json.dumps(identity))
            self.reseal(run["identity"])

    def edit_trace(self, run_id: str, edit) -> str:
        name = self.run_record(run_id)["trace"]
        path = self.bundle / name
        with gzip.open(path, "rt") as stream:
            rows = list(csv.reader(stream))
        edit(rows)
        stream = io.StringIO()
        csv.writer(stream, lineterminator="\n").writerows(rows)
        path.write_bytes(gzip.compress(stream.getvalue().encode(), mtime=0))
        self.reseal_run_output(name)
        return name

    def test_retained_evidence_has_only_bounded_source_reproduction_status(self) -> None:
        result = self.verify()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["native_tests_passed"], 13)
        self.assertEqual(result["native_runs"]["ten_cycles_2ms"]["steps"], 5000)
        self.assertEqual(result["scientific_status"], "unqualified")
        self.assertEqual(result["absolute_vascular_blood_volume"], "unqualified")
        self.assertEqual(result["oracle"]["cycles"], 20)

    def test_artifact_and_live_source_drift_are_rejected(self) -> None:
        artifact = self.bundle / self.run_record("refinement_2ms")["log"]
        artifact.write_text(artifact.read_text()+"uncaptured modification\n")
        self.rejected("artifact hash drift")
        self.reseal(artifact.relative_to(self.bundle).as_posix())
        source = self.repository / "src/numilab_human/shi_hose.py"
        source.write_text(source.read_text()+"\n# changed source\n")
        self.rejected("Human source hash drift")

    def test_resealed_summary_cannot_replace_trace_metrics(self) -> None:
        name = self.run_record("refinement_2ms")["log"]
        path = self.bundle / name
        log = path.read_text()
        summary = VERIFY.records(log, "cardiac_source_run")[0]
        old = "max_flow_error_m3_per_s="+summary["max_flow_error_m3_per_s"]
        path.write_text(log.replace(old, "max_flow_error_m3_per_s=0.0000001"))
        self.reseal_run_output(name)
        self.rejected("max_flow_error_m3_per_s disagrees")

    def test_resealed_trace_cannot_lose_last_accepted_step(self) -> None:
        self.edit_trace("ten_cycles_2ms", lambda rows: rows.pop())
        self.rejected("native trace is truncated")

    def test_one_cycle_refinement_cannot_replace_the_ten_cycle_gate(self) -> None:
        self.receipt["native_runs"] = [row for row in self.receipt["native_runs"] if row["id"] != "ten_cycles_2ms"]
        self.rejected("successful ten-cycle run")

    def test_archived_before_fix_run_cannot_be_promoted_to_final_cohort(self) -> None:
        self.run_record("refinement_2ms")["log"] = "retained-before-fix-cycle-final-2ms.log"
        self.rejected("before-fix evidence cannot qualify")

    def test_renamed_old_success_cannot_use_new_execution_identity(self) -> None:
        name = "renamed-before-fix-success.log"
        shutil.copy2(self.bundle / "retained-before-fix-cycle-final-2ms.log", self.bundle / name)
        self.receipt["artifacts"].append({"path": name, "sha256": VERIFY.file_hash(self.bundle / name)})
        self.run_record("refinement_2ms")["log"] = name
        self.rejected("execution identity is not bound to the selected log/trace")

    def test_binary_change_during_execution_cannot_be_resealed(self) -> None:
        name = self.run_record("refinement_2ms")["identity"]
        path = self.bundle / name
        identity = VERIFY.read_json(path)
        identity["built_artifact_sha256_after"]["matter/numi-matter-cardiac-check"] = "1"*64
        path.write_text(json.dumps(identity))
        self.reseal(name)
        self.rejected("code/binary changed during")

    def test_resealed_trace_cannot_use_nominal_instead_of_cooked_time(self) -> None:
        self.edit_trace("refinement_2ms", lambda rows: rows[1].__setitem__(0, ".002"))
        self.rejected("exact cooked clock")

    def test_resealed_trace_cannot_change_shape_or_hide_conservation_failure(self) -> None:
        name = self.edit_trace("refinement_2ms", lambda rows: rows[1].pop())
        self.rejected("trace coordinate arity")
        original = ROOT / "Docs/media/cardiac-source-20260912" / name
        shutil.copy2(original, self.bundle / name)
        self.reseal_run_output(name)
        self.edit_trace("refinement_2ms", lambda rows: rows[1].__setitem__(1, "0.01"))
        self.rejected("closed hydraulic storage conservation")

    def test_resealed_nonfinite_reference_coordinate_is_rejected(self) -> None:
        self.edit_trace("refinement_05ms", lambda rows: rows[1].__setitem__(2, "nan"))
        self.rejected("nonfinite trace coordinate")

    def test_native_test_summary_cannot_hide_missing_cardiac_case(self) -> None:
        path = self.bundle / "native-ctest.txt"
        path.write_text("\n".join(line for line in path.read_text().splitlines()
                                  if not ("/13 Test " in line and "matter.metal.cardiac_transaction" in line))+"\n")
        self.reseal(path.name)
        self.rejected("CTest case coverage")

    def test_new_native_owner_and_input_provenance_cannot_be_dropped(self) -> None:
        path = self.bundle / "native-identity.json"
        identity = VERIFY.read_json(path)
        del identity["built_artifact_sha256"]["matter/numi-matter-cardiac-check"]
        path.write_text(json.dumps(identity))
        self.reseal(path.name)
        self.rejected("source/binary provenance incomplete")

    def test_resealing_receipt_and_identity_cannot_change_the_native_revision(self) -> None:
        self.receipt["native_commit"] = "1"*40
        path = self.bundle / "native-identity.json"
        identity = VERIFY.read_json(path)
        identity["native_commit"] = self.receipt["native_commit"]
        path.write_text(json.dumps(identity))
        self.reseal(path.name)
        self.rejected("native commit differs from the frozen qualified revision")

    def test_source_reference_trace_cannot_be_resealed_after_modification(self) -> None:
        row = self.receipt["oracle"]["runs"][0]
        path = self.bundle / row["trace"]
        with gzip.open(path, "rb") as stream:
            raw = stream.read()
        path.write_bytes(gzip.compress(raw+b"\n", mtime=0))
        self.reseal(row["trace"])
        self.rejected(r"raw hash differs from C\+\+ run provenance")

    def test_failure_history_cannot_be_removed_from_receipt(self) -> None:
        self.assertTrue(self.receipt["retained_failures"], "Discovered ten-cycle failure must remain retained")
        self.receipt["retained_failures"] = []
        self.rejected("known failed ten-cycle evidence cannot be omitted")

    def test_known_failure_cannot_be_deleted_with_its_inventory_entries(self) -> None:
        for name in [VERIFY.KNOWN_FAILURE_LOG, VERIFY.KNOWN_FAILURE_TRACE]:
            (self.bundle / name).unlink()
        self.receipt["artifacts"] = [row for row in self.receipt["artifacts"] if row["path"] not in
                                     {VERIFY.KNOWN_FAILURE_LOG, VERIFY.KNOWN_FAILURE_TRACE}]
        self.receipt["retained_failures"] = []
        self.rejected("required cardiac artifact is missing")

    def test_biological_or_absolute_blood_promotion_is_rejected(self) -> None:
        self.receipt["scientific_status"] = "qualified"
        self.rejected("receipt does not meet")
        self.receipt["scientific_status"] = "unqualified"
        self.receipt["absolute_vascular_blood_volume"] = "qualified"
        self.rejected("receipt does not meet")


if __name__ == "__main__":
    unittest.main()
