from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from numilab_human.model import (
    ImportError, _myosim_joint_equality_artifacts, _myosim_joint_equality_bundle,
)
from numilab_human.cli import myosim_build


class JointEqualityPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {"archive_sha256": "ab" * 32, "mujoco_version": "3.12.0"}
        self.policy = {
            "schema": "numi.human.mujoco-scalar-equality-solver.v1",
            "mujoco_version": "3.12.0", "integrator": "Euler",
            "refsafe": True, "diagexact": False,
        }
        self.mapping = {
            1: {"core_q_index": 7, "core_v_index": 6, "source_name": "dependent"},
            2: {"core_q_index": 8, "core_v_index": 7, "source_name": "master"},
        }
        self.row = {
            "id": 0, "name": "quartic", "dependent_joint": 1, "master_joint": 2,
            "dependent_reference": 0.25, "master_reference": -0.5,
            "polycoef": [0.05, 0.2, -0.3, 0.1, -0.01],
            "solref": [0.02, 1.0], "solimp": [0.9999, 0.9999, 0.001, 0.5, 2.0],
        }

    def compile(self, row=None, policy=None):
        model = {} if policy is None else {"joint_equality_solver": policy}
        return _myosim_joint_equality_artifacts(
            self.source, model, [self.row if row is None else row], self.mapping, 9, 8
        )

    def compliant_row(self):
        return {**self.row, "dependent_dof_invweight0": 3.25, "master_dof_invweight0": 4.5}

    def test_legacy_export_is_byte_identical_nheq1(self) -> None:
        _, payload, abi = self.compile()
        expected = struct.pack("<8s10I32s", b"NHEQ1\0\0\0", 1, 9, 8, 1, 96, 1,
                               0, 0, 0, 0, bytes.fromhex("ab" * 32))
        expected += struct.pack("<4I20f", 7, 6, 8, 7, 0.25, -0.5,
                                *self.row["polycoef"], 0.0,
                                *self.row["solref"], 0.0, 0.0,
                                *self.row["solimp"], 0.0, 0.0, 0.0)
        self.assertEqual(abi, 1)
        self.assertEqual(payload, expected)

    def test_nheq2_retains_entire_legacy_record_and_source_weights(self) -> None:
        _, legacy, _ = self.compile()
        manifest, payload, abi = self.compile(self.compliant_row(), self.policy)
        self.assertEqual(abi, 2)
        self.assertEqual(len(payload), 80 + 112)
        header = struct.unpack_from("<8s10I32s", payload)
        self.assertEqual(header[:11], (b"NHEQ2\0\0\0", 2, 9, 8, 1, 112, 1, 1, 1, 0, 0))
        self.assertEqual(payload[80:176], legacy[80:])
        self.assertEqual(struct.unpack_from("<4f", payload, 176), (3.25, 4.5, 0.0, 0.0))
        self.assertEqual(manifest[0]["dependent_dof_invweight0"], 3.25)
        self.assertEqual(self.compile(self.compliant_row(), self.policy)[1], payload)

    def test_explicit_refsafe_disabled_is_preserved(self) -> None:
        _, payload, _ = self.compile(self.compliant_row(), {**self.policy, "refsafe": False})
        self.assertEqual(struct.unpack_from("<8s10I32s", payload)[8], 0)

    def test_bundle_emits_explicit_legacy_and_compliant_programs(self) -> None:
        _, expected_legacy, _ = self.compile()
        _, expected_compliant, _ = self.compile(self.compliant_row(), self.policy)
        records, legacy, compliant = _myosim_joint_equality_bundle(
            self.source, {"joint_equality_solver": self.policy},
            [self.compliant_row()], self.mapping, 9, 8,
        )
        self.assertEqual(legacy, expected_legacy)
        self.assertEqual(compliant, expected_compliant)
        self.assertEqual(records[0]["master_dof_invweight0"], 4.5)
        _, old_legacy, old_compliant = _myosim_joint_equality_bundle(
            self.source, {}, [self.row], self.mapping, 9, 8,
        )
        self.assertEqual(old_legacy, expected_legacy)
        self.assertIsNone(old_compliant)

    def test_build_writes_both_payloads_with_separate_manifest_entries(self) -> None:
        _, legacy, compliant = _myosim_joint_equality_bundle(
            self.source, {"joint_equality_solver": self.policy},
            [self.compliant_row()], self.mapping, 9, 8,
        )
        names = {
            "rigid": "fixture.nhrigid", "muscles": "fixture.nhmyo",
            "support_contact": "fixture.nhcnt", "extensor_hood": "fixture.nhhood",
            "joint_equalities": "myosim-fullbody-joint-equalities.nheq",
            "joint_equalities_source_compliance": "myosim-fullbody-joint-equalities-source-compliance.nheq",
        }
        manifest = {"payloads": {key: {"file": value} for key, value in names.items()}}
        for key, payload in [("joint_equalities", legacy),
                             ("joint_equalities_source_compliance", compliant)]:
            manifest["payloads"][key]["sha256"] = hashlib.sha256(payload).hexdigest()
        def fake_export(command, **_kwargs):
            Path(command[-1]).write_text("{}")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = argparse.Namespace(sources=root, output=root / "output", python=Path(sys.executable))
            with patch("numilab_human.cli.subprocess.run", side_effect=fake_export), patch(
                "numilab_human.cli.myosim_fullbody_reference_artifacts",
                return_value=(manifest, b"rigid", b"muscle", b"support", legacy, b"hood", compliant),
            ), patch("numilab_human.support_primitives.compile_support_primitives",
                     return_value=({"file": "fixture-primitives.nhcnt"}, b"primitives")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(myosim_build(args), 0)
            self.assertEqual((args.output / "fixture-primitives.nhcnt").read_bytes(), b"primitives")
            self.assertEqual((args.output / names["joint_equalities"]).read_bytes(), legacy)
            self.assertEqual((args.output / names["joint_equalities_source_compliance"]).read_bytes(), compliant)
            written = json.loads((args.output / "myosim-fullbody-reference.manifest.json").read_text())
            self.assertEqual(written, manifest)

    def test_fixed_master_has_no_inverse_weight(self) -> None:
        row = {**self.compliant_row(), "master_joint": -1, "master_dof_invweight0": 0.0}
        _, payload, _ = self.compile(row, self.policy)
        self.assertEqual(struct.unpack_from("<4I", payload, 80), (7, 6, 0xFFFFFFFF, 0xFFFFFFFF))
        row["master_dof_invweight0"] = 1.0
        with self.assertRaises(ImportError):
            self.compile(row, self.policy)

    def test_missing_partial_or_unrepresentable_weights_fail_closed(self) -> None:
        bad_rows = [self.row, {**self.row, "dependent_dof_invweight0": 1.0}]
        bad_rows += [{**self.compliant_row(), "dependent_dof_invweight0": value}
                     for value in [-1.0, float("nan"), float("inf"), 1e100, 1e-100]]
        bad_rows += [{**self.compliant_row(), "dependent_dof_invweight0": 0.0,
                      "master_dof_invweight0": 0.0}]
        for row in bad_rows:
            with self.subTest(row=row), self.assertRaises(ImportError):
                self.compile(row, self.policy)

    def test_solver_metadata_cannot_be_dropped_to_downgrade_payload(self) -> None:
        with self.assertRaises(ImportError):
            self.compile(self.compliant_row())

    def test_unsupported_solver_semantics_fail_closed(self) -> None:
        for key, value in [("schema", "unknown"), ("mujoco_version", "3.13.0"),
                           ("integrator", "discrete"), ("diagexact", True), ("refsafe", 1)]:
            policy = {**self.policy, key: value}
            with self.subTest(key=key), self.assertRaises(ImportError):
                self.compile(self.compliant_row(), policy)
        self.source["mujoco_version"] = "3.11.0"
        with self.assertRaises(ImportError):
            self.compile(self.compliant_row(), self.policy)


if __name__ == "__main__":
    unittest.main()
