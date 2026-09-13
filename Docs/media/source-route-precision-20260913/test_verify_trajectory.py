"""Manufactured corruption controls for the retained-log verifier; no physics."""
import base64
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import verify_trajectory as verifier


def records():
    rows = []
    for scenario in ("zero", "replay"):
        for root in (1, 2):
            rows.append(("mrnx_accepted_root_translation=", {
                "schema": "numi.human.accepted-root-translation.v1", "valid": True,
                "root": root, "physics_generation": root,
                "timestamp_microseconds": 100 + root * 25,
                "transaction_fingerprint": f"{root:016x}",
                "words": ["3f800000", "00000000", "00000000", "00000000"] + ["00000000"] * 8,
            }))
            for kind, count in verifier.COUNTS.items():
                values = [0.0] * count
                if kind == "qv":
                    values[0] = values[6] = 1.0
                rows.append(("prepared_recruitment=", {
                    "schema": "numi.human.prepared-recruitment-trace.v3", "scenario": scenario,
                    "root": root, "elapsed_microseconds": root * 25, "kind": kind,
                    "transaction_fingerprint": f"{root:016x}", "physics_generation": root,
                    "accepted_timestamp_microseconds": 100 + root * 25,
                    "fp32_le_base64": base64.b64encode(struct.pack("<" + "f" * count, *values)).decode(),
                }))
    return rows


def log(rows):
    return "".join(prefix + json.dumps(row) + "\n" for prefix, row in rows) + (
        "prepared_timestep=observed roots_per_scenario=2 timestep_us=25 duration_us=50 replay=bitwise\n"
        "Executed 1 test, with 0 failures (0 unexpected)\n"
    )


class TrajectoryVerifierTests(unittest.TestCase):
    def audit(self, rows):
        return verifier.audit(log(rows), timestep_us=25, duration_us=50)

    def test_complete_manufactured_records(self):
        self.assertEqual(self.audit(records())["accepted_roots_per_scenario"], 2)

    def test_reordered_root_blocks_rejected(self):
        rows = records()
        size = len(verifier.COUNTS) + 1
        rows[:2 * size] = rows[size:2 * size] + rows[:size]
        with self.assertRaisesRegex(ValueError, "order"):
            self.audit(rows)

    def test_reordered_kinds_rejected(self):
        rows = records()
        rows[1], rows[2] = rows[2], rows[1]
        with self.assertRaisesRegex(ValueError, "order"):
            self.audit(rows)

    def test_reordered_scenarios_rejected(self):
        rows = records()
        split = len(rows) // 2
        with self.assertRaisesRegex(ValueError, "order"):
            self.audit(rows[split:] + rows[:split])

    def test_negative_zero_correction_rejected(self):
        rows = records()
        rows[0][1]["words"][8] = "80000000"
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.audit(rows)

    def test_non_normalized_displacement_rejected(self):
        rows = records()
        rows[0][1]["words"][4] = "33800000"  # 2^-24, exactly halfway at reference 1
        rows[0][1]["words"][8] = "33800000"  # another 2^-24 must carry into displacement
        with self.assertRaises(ValueError):
            self.audit(rows)

    def test_duplicate_transaction_rejected(self):
        rows = records()
        rows[len(verifier.COUNTS) + 1][1]["transaction_fingerprint"] = "0000000000000001"
        with self.assertRaisesRegex(ValueError, "repeated"):
            self.audit(rows)

    def test_boolean_clock_rejected(self):
        rows = records()
        rows[0][1]["physics_generation"] = True
        with self.assertRaisesRegex(ValueError, "generation"):
            self.audit(rows)

    def test_every_trace_kind_joins_actual_committed_identity(self):
        for kind_index in range(1, len(verifier.COUNTS) + 1):
            for field, invalid in (("transaction_fingerprint", "000000000000000f"),
                                   ("physics_generation", 2), ("accepted_timestamp_microseconds", 126)):
                rows = records()
                rows[kind_index][1][field] = invalid
                with self.subTest(kind=rows[kind_index][1]["kind"], field=field):
                    with self.assertRaisesRegex(ValueError, "committed publication"):
                        self.audit(rows)

    def test_old_unbound_trace_schema_rejected(self):
        rows = records()
        rows[1][1]["schema"] = "numi.human.prepared-recruitment-trace.v2"
        with self.assertRaisesRegex(ValueError, "schema"):
            self.audit(rows)

    def test_execution_flags_are_strict_booleans(self):
        raw = log(records()).encode()
        execution = {"status": "pass", "returncode": 0, "source_unchanged": True,
                     "artifacts_unchanged": True,
                     "log_identity": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.log").write_bytes(raw)
            arguments = ["verify_trajectory.py", str(root / "execution.json"), str(root / "run.log"),
                         "--timestep-us", "25", "--duration-us", "50"]
            for field, invalid in (("source_unchanged", "false"), ("artifacts_unchanged", 1), ("returncode", False)):
                mutated = copy.deepcopy(execution)
                mutated[field] = invalid
                (root / "execution.json").write_text(json.dumps(mutated))
                with self.subTest(field=field), mock.patch.object(sys, "argv", arguments):
                    with self.assertRaisesRegex(ValueError, "qualified"), contextlib.redirect_stdout(io.StringIO()):
                        verifier.main()


if __name__ == "__main__":
    unittest.main()
