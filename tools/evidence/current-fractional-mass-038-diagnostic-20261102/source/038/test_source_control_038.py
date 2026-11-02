#!/usr/bin/env python3
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILDER_PATH = HERE / "build_source_control_038.py"
READER_PATH = HERE / "recorded_flow_contract_010_030d.py"
OUTER_SCHEMA = "numi.h706.current-fractional-mass-cli-run.v1"

def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

BUILDER = load(BUILDER_PATH, "test_builder_038")
READER = load(READER_PATH, "test_reader_010_030d")

def pin(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def synthetic_mass_receipt() -> dict:
    return {
        "schema": READER.BASE.SCHEMA_MASS,
        "status": "PASS_SOURCE_ONLY_REPLAY",
        "physical_admission": False,
        "native_steps": 0,
        "allocation": "NOT_NATIVE_EXECUTED",
        "claims": {"source_controls": True, "deterministic_replay": True},
        "solver": {
            "method": "deterministic_dinic_feasibility",
            "status": "PASS_FEASIBLE_NOT_OPTIMAL",
            "optimality_certified": False,
            "scale_factor": 1.0,
            "postsolve_rescaling": False,
        },
        "replay": {"status": "PASS_DETERMINISTIC_REPLAY", "flow_array_equality": True},
    }

class SourceControl038Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="h706-038-tests-")
        cls.root = Path(cls.temp.name)
        cls.control, cls.closure = BUILDER.build(cls.root / "generated")
        cls.control_path = cls.root / "generated" / "source-control-manifest-038.json"
        cls.closure_path = cls.root / "generated" / "source-closure-038.json"
        cls.control_json = json.loads(cls.control_path.read_text(encoding="utf-8"))
        cls.closure_json = json.loads(cls.closure_path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_generated_json_and_resolved_closure_paths(self):
        self.assertEqual(self.control_json["status"], "PASS_SOURCE_CONTROLS_FULL_ALLOCATOR_NOT_RUN")
        self.assertEqual(self.control_json["solver_contract"]["status"], "NOT_RUN_FULL_ALLOCATOR")
        self.assertEqual(self.closure_json["schema"], "numi.h706.allocator-source-closure.v1")
        for item in self.closure_json["sources"]:
            self.assertEqual(item["path"], str(Path(item["path"]).resolve()))
            self.assertTrue(Path(item["path"]).is_file())
            self.assertEqual(item["bytes"], Path(item["path"]).stat().st_size)
            self.assertEqual(item["sha256"], hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest())
        READER.verify_nested_control_refs(self.control_json)
        READER.BASE.verify_source_closure(
            self.closure_path, pin(self.closure_path)["sha256"],
            self.control_path.resolve(), pin(self.control_path)["sha256"], self.control_json,
        )

    def test_030d_postpass_projection_and_full_reader(self):
        mass = synthetic_mass_receipt()
        projected = READER._control_projection_for_013(self.control_json, mass_receipt=mass)
        self.assertEqual(projected["source"]["path"], self.control_json["source"]["path"])
        self.assertEqual(projected["binary"]["path"], self.control_json["binary"]["path"])
        self.assertEqual(projected["solver_contract"]["status"], "PASS_FEASIBLE_NOT_OPTIMAL")
        self.assertEqual(len(projected["focused_controls"]), 4)
        self.assertTrue(all(row["status"] == "PASS" for row in projected["focused_controls"]))

        mass_path = self.root / "synthetic-mass-receipt.json"
        mass_path.write_text(json.dumps(mass, sort_keys=True) + "\n", encoding="utf-8")
        bundle = {
            "schema": READER.BASE.SCHEMA,
            "unified_mass_receipt": pin(mass_path),
            "source_control_manifest": pin(self.control_path),
            "source_closure": pin(self.closure_path),
            "paired_solver_run_sha256": "0" * 64,
            "roles": {},
        }
        run = {
            "source_control_manifest": pin(self.control_path),
            "source": self.control_json["source"],
            "binary": self.control_json["binary"],
        }
        READER.verify_source_control(run, bundle, mass)

    def test_030d_does_not_masquerade_030c_or_accept_mutation(self):
        wrong = copy.deepcopy(self.control_json)
        wrong["binary"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "independent pinned-input reconstruction"):
            READER.verify_nested_control_refs(wrong)
        mislabeled = copy.deepcopy(self.control_json)
        mislabeled["variant"] = "030c-gzip-fullcut-witness-closed-include"
        with self.assertRaises(ValueError):
            READER.verify_nested_control_refs(mislabeled)

    def test_frozen_029_projection_is_bytewise_unchanged(self):
        parent_path = Path(self.control_json["parent_manifests"][0]["path"])
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        old = READER.legacy_projection(parent)
        delegated = READER._control_projection_for_013(parent)
        self.assertEqual(delegated, old)
        READER.verify_nested_control_refs(parent)

if __name__ == "__main__":
    unittest.main()

