"""Synthetic contract fixtures only. These tests contain no Human runtime evidence."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from numilab_human import behavior_qualification as q


def hashed(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n")
    return q.digest(path)


def artifact(path):
    return {"path": str(path), "sha256": q.digest(path), "bytes": path.stat().st_size}


class BehaviorQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.protocol_path = self.root / "protocol.json"
        self.stack_path = self.root / "stack.json"
        self.bundle_path = self.root / "bundle.json"
        self.cases = []
        for task, speed, count in (("standing", None, 20), ("recovery", None, 100),
                                   *(("walking", speed, 100) for speed in q.SPEEDS)):
            for seed in range(count):
                key = f"{task}-{speed}-{seed}"
                impulse = {"body_semantic_id": "pelvis", "frame": "world", "at_step": 0,
                           "point_m": [0, 0, 1], "linear_impulse_ns": [seed + 1, 0, 0]}
                self.cases.append({"trial_id": key, "task": task, "seed": seed,
                                   "reset_state_sha256": hashed(f"reset-{seed}"),
                                   "target_speed_mps": speed,
                                   "impulse": impulse if task == "recovery" else None})
        task_pack = self.root / "task.pack"
        task_pack.write_bytes(b"SYNTHETIC TEST TaskPack, not a compiled native task")
        rule = {"minimum_root_height_m": 0.7, "maximum_trunk_tilt_rad": 0.3,
                "maximum_planar_speed_mps": 0.1, "forbidden_contact_semantic_ids": ["pelvis"],
                "root_body_semantic_id": "pelvis", "trunk_body_semantic_id": "thorax",
                "world_reference_origin_m": [0, 0, 0], "world_up_axis": [0, 0, 1],
                "world_forward_axis": [1, 0, 0], "trunk_up_axis_body": [0, 0, 1],
                "velocity_observable": "whole_human_com_linear_velocity",
                "velocity_body_semantic_id": None}
        self.protocol = {"schema": q.PROTOCOL_SCHEMA, "metric_contract": q.METRIC_CONTRACT,
                         "step_ns": 1_000_000, "recovery_hold_ns": q.SECOND_NS,
                         "task_pack_sha256": q.digest(task_pack),
                         "task_criteria": {task: dict(rule) for task in ("standing", "recovery", "walking")},
                         "cases": self.cases}
        self.protocol_hash = write_json(self.protocol_path, self.protocol)
        entries = {}
        for role in q.ARTIFACT_ROLES:
            path = self.root / (role + ".artifact")
            path.write_bytes(("SYNTHETIC TEST " + role).encode())
            entries[role] = artifact(path)
        entries["task_pack"] = artifact(task_pack)
        entries["qualifier"] = artifact(Path(q.__file__))
        self.stack = {"schema": q.STACK_SCHEMA, "source_state": "clean", "backend": "Apple Metal",
                      "device": "Apple test fixture", "os_build": "test fixture",
                      "revisions": {owner: "a" * 40 for owner in ("human", "native", "brain")},
                      "artifacts": entries}
        receipt_path = Path(entries["task_lowering_receipt"]["path"])
        write_json(receipt_path, {
            "schema": q.LOWERING_SCHEMA, "metric_contract": q.METRIC_CONTRACT,
            "protocol_sha256": self.protocol_hash, "native_revision": "a" * 40,
            "step_ns": self.protocol["step_ns"], "task_criteria": self.protocol["task_criteria"],
            "artifacts": {role: entries[role]["sha256"] for role in q.LOWERING_ARTIFACT_ROLES},
        })
        entries["task_lowering_receipt"] = artifact(receipt_path)
        self.stack_hash = write_json(self.stack_path, self.stack)
        self.bundle = {"schema": q.BUNDLE_SCHEMA, "protocol_sha256": self.protocol_hash,
                       "stack_sha256": self.stack_hash, "trials": []}
        for case in self.cases:
            path = self.root / (case["trial_id"] + ".jsonl")
            self.write_trace(path, self.trace(case))
            self.bundle["trials"].append({"trial_id": case["trial_id"], **artifact(path)})
        write_json(self.bundle_path, self.bundle)

    def trace(self, case):
        duration = {"standing": 60, "recovery": 5, "walking": 120}[case["task"]]
        count = duration * q.SECOND_NS // self.protocol["step_ns"]
        initial, final = hashed(case["trial_id"] + "initial"), hashed(case["trial_id"] + "final")
        header = {"schema": q.TRACE_SCHEMA, **case, "protocol_sha256": self.protocol_hash,
                  "stack_sha256": self.stack_hash, "device": self.stack["device"],
                  "metric_contract": q.METRIC_CONTRACT, "execution_id": case["trial_id"],
                  "initial_root_sha256": initial, "initial_posture_valid": True, "initial_settled": True,
                  "os_build": self.stack["os_build"], "step_ns": self.protocol["step_ns"],
                  **{role + "_sha256": self.stack["artifacts"][role]["sha256"]
                     for role in ("metric_program", "accepted_root_proof_schema", "task_lowering_receipt")}}
        span = {"kind": "accepted_span", "first_step": 1, "last_step": count,
                "start_ns": 0, "end_ns": duration * q.SECOND_NS, "accepted_steps": count,
                "start_root_sha256": initial, "end_root_sha256": final,
                "attempt_count": count, "rejected_attempt_count": 0,
                "metric_sample_count": count, "audit_covered_root_count": count,
                "audit_covered_attempt_count": count,
                "speed_error_sample_count": count if case["task"] == "walking" else 0,
                "audit": {key: 0 for key in q.AUDIT_COUNTERS}, "posture_violation_steps": 0,
                "settled_steps": count, "settled_suffix_steps": count,
                "speed_squared_error_sum_m2_per_s2": 0.0}
        footer = {"kind": "completed", "accepted_steps": count,
                  "final_root_sha256": final, "exit_code": 0}
        return [header, span, footer]

    @staticmethod
    def write_trace(path, entries):
        path.write_text("".join(json.dumps(entry, allow_nan=False) + "\n" for entry in entries))

    def mutate_trial(self, index, edit):
        descriptor = self.bundle["trials"][index]
        path = Path(descriptor["path"])
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        edit(entries)
        self.write_trace(path, entries)
        descriptor.update(artifact(path))
        write_json(self.bundle_path, self.bundle)

    def evaluate(self):
        return q.evaluate(self.protocol_path, self.stack_path, self.bundle_path,
                          expected_protocol_sha256=self.protocol_hash,
                          expected_stack_sha256=self.stack_hash)

    def test_complete_frozen_populations_pass_with_exact_horizons(self):
        report = self.evaluate()
        self.assertEqual(report["status"], "passed")
        self.assertEqual([g["trials"] for g in report["groups"]], [20, 100, 100, 100, 100])
        self.assertEqual(len(report["trials"]), 420)
        self.assertEqual(report["trials"][-1]["accepted_seconds"], 120)

    def test_all_twenty_standing_seeds_must_succeed(self):
        self.mutate_trial(0, lambda rows: rows[1].update(settled_steps=0, settled_suffix_steps=0))
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["groups"][0]["successful_trials"], 19)

    def test_recovery_requires_95_complete_trials_and_frozen_hold(self):
        def not_recovered(rows):
            rows[1].update(settled_steps=999, settled_suffix_steps=999)
        for index in range(20, 25):
            self.mutate_trial(index, not_recovered)
        self.assertEqual(self.evaluate()["status"], "passed")
        self.mutate_trial(25, not_recovered)
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["groups"][1]["successful_trials"], 94)

    def test_recovery_hold_uses_first_settled_sample_time(self):
        case = self.cases[20]
        rows = self.trace(case)
        rows[0]["initial_settled"] = False
        path = self.root / "single.jsonl"
        for suffix, expected in ((0, False), (1000, False), (1001, True)):
            with self.subTest(suffix=suffix):
                rows[1].update(settled_steps=suffix, settled_suffix_steps=suffix)
                self.write_trace(path, rows)
                result, _ = q._trace(path, case, self.protocol, self.protocol_hash,
                                      self.stack_hash, self.stack)
                self.assertEqual(result["success"], expected)
                if suffix == 1001:
                    self.assertEqual(result["recovery_seconds"], 4.0)

    def test_standing_requires_settled_reset_and_suffixes_compose_across_spans(self):
        case = self.cases[0]
        rows = self.trace(case)
        first = rows[1]
        second = copy.deepcopy(first)
        midpoint = hashed("midpoint")
        half = first["accepted_steps"] // 2
        first.update(last_step=half, accepted_steps=half, attempt_count=half,
                     end_ns=30 * q.SECOND_NS, end_root_sha256=midpoint,
                     settled_steps=half, settled_suffix_steps=half, metric_sample_count=half,
                     audit_covered_root_count=half, audit_covered_attempt_count=half)
        second.update(first_step=half + 1, start_ns=30 * q.SECOND_NS,
                      start_root_sha256=midpoint, accepted_steps=half, attempt_count=half,
                      settled_steps=half, settled_suffix_steps=half, metric_sample_count=half,
                     audit_covered_root_count=half, audit_covered_attempt_count=half)
        rows.insert(2, second)
        path = self.root / "single.jsonl"
        self.write_trace(path, rows)
        result, _ = q._trace(path, case, self.protocol, self.protocol_hash,
                            self.stack_hash, self.stack)
        self.assertTrue(result["success"])
        rows[0]["initial_settled"] = False
        self.write_trace(path, rows)
        result, _ = q._trace(path, case, self.protocol, self.protocol_hash,
                            self.stack_hash, self.stack)
        self.assertFalse(result["success"])

    def test_walking_rmse_is_per_trial_and_per_speed(self):
        def fails_speed(rows):
            rows[1]["speed_squared_error_sum_m2_per_s2"] = (0.15001 ** 2) * rows[1]["accepted_steps"]
        for index in range(120, 125):
            self.mutate_trial(index, fails_speed)
        self.assertEqual(self.evaluate()["status"], "passed")
        self.mutate_trial(125, fails_speed)
        report = self.evaluate()
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["groups"][2]["successful_trials"], 94)
        self.assertEqual(report["groups"][3]["successful_trials"], 100)

    def test_assistance_and_unaccepted_or_omitted_physics_invalidate_entire_evidence(self):
        original = self.trace(self.cases[0])
        path = self.root / "single.jsonl"
        for counter in q.AUDIT_COUNTERS:
            with self.subTest(counter=counter):
                rows = copy.deepcopy(original)
                rows[1]["audit"][counter] = 1
                self.write_trace(path, rows)
                with self.assertRaisesRegex(q.EvidenceError, "inadmissible native audit"):
                    q._trace(path, self.cases[0], self.protocol, self.protocol_hash,
                             self.stack_hash, self.stack)

    def test_truncation_duplicate_spans_chain_and_mixed_identity_are_rejected(self):
        original = self.trace(self.cases[0])
        edits = [lambda r: r.pop(),
                 lambda r: r.insert(2, copy.deepcopy(r[1])),
                 lambda r: r[1].update(first_step=2),
                 lambda r: r[1].update(start_ns=1),
                 lambda r: r[1].update(start_root_sha256="f" * 64),
                 lambda r: r[0].update(stack_sha256="f" * 64),
                 lambda r: r[0].update(protocol_sha256="f" * 64),
                 lambda r: r[0].update(seed=True),
                 lambda r: r[1]["audit"].update(root_assistance_steps=False),
                 lambda r: r[1].update(attempt_count=r[1]["accepted_steps"] + 1),
                 lambda r: r[-1].update(exit_code=1)]
        for edit in edits:
            rows = copy.deepcopy(original)
            edit(rows)
            path = self.root / "single.jsonl"
            self.write_trace(path, rows)
            with self.subTest(rows=rows), self.assertRaises(q.EvidenceError):
                q._trace(path, self.cases[0], self.protocol, self.protocol_hash,
                         self.stack_hash, self.stack)

    def test_short_but_self_consistent_trace_cannot_pass(self):
        def shortened(rows):
            rows[1].update(last_step=1000, end_ns=q.SECOND_NS, accepted_steps=1000,
                           attempt_count=1000, settled_steps=1000, settled_suffix_steps=1000,
                           metric_sample_count=1000, audit_covered_root_count=1000,
                           audit_covered_attempt_count=1000)
            rows[-1]["accepted_steps"] = 1000
        self.mutate_trial(20, shortened)
        with self.assertRaisesRegex(q.EvidenceError, "truncated trial"):
            self.evaluate()

    def test_duplicate_trial_execution_and_frozen_impulse_are_rejected(self):
        self.mutate_trial(1, lambda rows: rows[0].update(execution_id=self.cases[0]["trial_id"]))
        with self.assertRaisesRegex(q.EvidenceError, "reused execution_id"):
            self.evaluate()
        protocol = copy.deepcopy(self.protocol)
        protocol["cases"][21]["impulse"] = protocol["cases"][20]["impulse"]
        with self.assertRaisesRegex(q.EvidenceError, "duplicate frozen impulse"):
            q._protocol(protocol)
        protocol = copy.deepcopy(self.protocol)
        protocol["cases"][1]["seed"] = protocol["cases"][0]["seed"]
        with self.assertRaisesRegex(q.EvidenceError, "duplicate seed"):
            q._protocol(protocol)

    def test_impulse_uniqueness_cannot_be_evaded_with_metadata_or_number_spelling(self):
        protocol = copy.deepcopy(self.protocol)
        impulse = copy.deepcopy(protocol["cases"][20]["impulse"])
        for index in range(20, 120):
            protocol["cases"][index]["impulse"] = {
                **impulse, "metadata_only_trial_label": str(index)}
        with self.assertRaisesRegex(q.EvidenceError, "five physical descriptor fields"):
            q._protocol(protocol)
        protocol = copy.deepcopy(self.protocol)
        protocol["cases"][21]["impulse"] = copy.deepcopy(protocol["cases"][20]["impulse"])
        protocol["cases"][21]["impulse"]["linear_impulse_ns"] = [1.0, -0.0, 0.0]
        with self.assertRaisesRegex(q.EvidenceError, "duplicate frozen impulse"):
            q._protocol(protocol)

    def test_trace_booleans_do_not_match_frozen_numeric_speed_or_impulse(self):
        for case_index, field, value in ((220, "target_speed_mps", True),
                                        (20, "linear_impulse_ns", [True, False, False]),
                                        (20, "point_m", [False, False, True]),
                                        (20, "at_step", False), (20, "at_step", 0.0)):
            with self.subTest(field=field):
                case = self.cases[case_index]
                rows = copy.deepcopy(self.trace(case))
                if field == "target_speed_mps":
                    rows[0][field] = value
                else:
                    rows[0]["impulse"][field] = value
                path = self.root / "single.jsonl"
                self.write_trace(path, rows)
                with self.assertRaisesRegex(q.EvidenceError, "differs from frozen case|at_step must be integer"):
                    q._trace(path, case, self.protocol, self.protocol_hash,
                             self.stack_hash, self.stack)

    def test_walking_cannot_report_unsettled_posture_valid_roots(self):
        case = self.cases[120]
        rows = self.trace(case)
        rows[1].update(settled_steps=0, settled_suffix_steps=0)
        path = self.root / "single.jsonl"
        self.write_trace(path, rows)
        with self.assertRaisesRegex(q.EvidenceError, "walking settled count"):
            q._trace(path, case, self.protocol, self.protocol_hash,
                     self.stack_hash, self.stack)

    def test_native_reduction_must_cover_every_root_attempt_and_speed_sample(self):
        case = self.cases[120]
        original = self.trace(case)
        path = self.root / "single.jsonl"
        for field in ("metric_sample_count", "audit_covered_root_count",
                      "audit_covered_attempt_count", "speed_error_sample_count"):
            for mode in ("missing", "zero", "decimated"):
                with self.subTest(field=field, mode=mode):
                    rows = copy.deepcopy(original)
                    if mode == "missing":
                        rows[1].pop(field)
                    else:
                        rows[1][field] = 0 if mode == "zero" else rows[1][field] // 2
                    self.write_trace(path, rows)
                    with self.assertRaises(q.EvidenceError):
                        q._trace(path, case, self.protocol, self.protocol_hash,
                                 self.stack_hash, self.stack)
        rows = copy.deepcopy(original)
        rows[1].update(attempt_count=rows[1]["accepted_steps"] + 1, rejected_attempt_count=1)
        self.write_trace(path, rows)
        with self.assertRaisesRegex(q.EvidenceError, "audit_covered_attempt_count"):
            q._trace(path, case, self.protocol, self.protocol_hash, self.stack_hash, self.stack)
        rows[1]["audit_covered_attempt_count"] += 1
        self.write_trace(path, rows)
        self.assertTrue(q._trace(path, case, self.protocol, self.protocol_hash,
                                self.stack_hash, self.stack)[0]["success"])

    def test_frozen_observables_require_registered_bodies_frames_and_velocity(self):
        for field in ("root_body_semantic_id", "trunk_body_semantic_id", "world_reference_origin_m",
                      "world_up_axis", "world_forward_axis", "trunk_up_axis_body",
                      "velocity_observable", "velocity_body_semantic_id"):
            with self.subTest(field=field):
                protocol = copy.deepcopy(self.protocol)
                protocol["task_criteria"]["walking"].pop(field)
                with self.assertRaises(q.EvidenceError):
                    q._protocol(protocol)
        for field, value in (("world_up_axis", [0, 0, 0]),
                             ("world_forward_axis", [0, 0, 1]),
                             ("trunk_up_axis_body", [False, False, True]),
                             ("world_reference_origin_m", [0, 0, 10 ** 1000])):
            with self.subTest(field=field, value=value):
                protocol = copy.deepcopy(self.protocol)
                protocol["task_criteria"]["walking"][field] = value
                with self.assertRaises(q.EvidenceError):
                    q._protocol(protocol)

    def test_header_binds_native_timestep_os_program_and_proof_schema(self):
        case = self.cases[0]
        original = self.trace(case)
        path = self.root / "single.jsonl"
        for field, value in (("step_ns", 10_000_000), ("os_build", "different OS"),
                             ("metric_program_sha256", "f" * 64),
                             ("accepted_root_proof_schema_sha256", "f" * 64),
                             ("task_lowering_receipt_sha256", "f" * 64)):
            with self.subTest(field=field):
                rows = copy.deepcopy(original)
                rows[0][field] = value
                self.write_trace(path, rows)
                with self.assertRaises(q.EvidenceError):
                    q._trace(path, case, self.protocol, self.protocol_hash,
                             self.stack_hash, self.stack)

    def test_rehashed_stack_cannot_hide_a_lowering_receipt_for_different_task_or_native_program(self):
        receipt_path = Path(self.stack["artifacts"]["task_lowering_receipt"]["path"])
        original = json.loads(receipt_path.read_text())
        for field, value in (("native_revision", "f" * 40), ("protocol_sha256", "f" * 64),
                             ("step_ns", 10_000_000), ("task_criteria", {}),
                             ("artifacts", {})):
            with self.subTest(field=field):
                changed = {**original, field: value}
                write_json(receipt_path, changed)
                self.stack["artifacts"]["task_lowering_receipt"] = artifact(receipt_path)
                self.stack_hash = write_json(self.stack_path, self.stack)
                with self.assertRaisesRegex(q.EvidenceError, "task lowering receipt"):
                    self.evaluate()

    def test_current_stack_pins_and_artifact_bytes_cannot_be_relabelled(self):
        with self.assertRaisesRegex(q.EvidenceError, "externally pinned hash mismatch"):
            q.evaluate(self.protocol_path, self.stack_path, self.bundle_path,
                       expected_protocol_sha256="f" * 64, expected_stack_sha256=self.stack_hash)
        Path(self.stack["artifacts"]["native_library"]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(q.EvidenceError, "artifact size mismatch|artifact hash mismatch"):
            self.evaluate()

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
        for raw in (b'{"x": 0, "x": 1}', b'{"x": NaN}', b'{"x": Infinity}', b'{"x": 1e999}'):
            with self.subTest(raw=raw), self.assertRaises(q.EvidenceError):
                q._decode(raw, "fixture")

    def test_missing_trial_cannot_be_hidden_by_success_percentage(self):
        self.bundle["trials"].pop()
        write_json(self.bundle_path, self.bundle)
        with self.assertRaisesRegex(q.EvidenceError, "complete frozen trial population"):
            self.evaluate()

    def test_cli_writes_failure_receipt_without_overwriting_existing_evidence(self):
        output = self.root / "assessment.json"
        arguments = ["--protocol", str(self.protocol_path), "--protocol-sha256", "f" * 64,
                     "--stack", str(self.stack_path), "--stack-sha256", self.stack_hash,
                     "--bundle", str(self.bundle_path), "--output", str(output)]
        self.assertEqual(q.main(arguments), 1)
        self.assertEqual(json.loads(output.read_text())["status"], "invalid")
        with self.assertRaises(FileExistsError):
            q.main(arguments)


if __name__ == "__main__":
    unittest.main()
