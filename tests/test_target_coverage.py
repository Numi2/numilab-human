from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path

import tempfile
import unittest
from unittest.mock import patch

from numilab_human.model import ImportError as HumanImportError
from numilab_human.target_coverage import (
    Inventory, MANDATORY, _myosim_ir, _xml_declarations, canonical_bytes, command,
    digest, materialize, validate_manifest, validate_transition,
)


def _lock(directory: Path, entries: dict) -> Path:
    path = directory / "sources.lock.json"
    path.write_text(json.dumps({"schema": "numi.human.source-lock.v1", "sources": entries}))
    return path


def _model(directory: Path, xml: str, name: str = "model.osim") -> dict:
    path = directory / name
    path.write_text(xml)
    return {"model_file": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "revision": "fixture-1", "license": "fixture-only"}


def _seal(value: dict) -> None:
    value["manifest_sha256"] = digest({key: item for key, item in value.items() if key != "manifest_sha256"})


def _build(directory: Path, entries: dict, previous: dict | None = None) -> dict:
    return materialize(sources=directory, source_lock=_lock(directory, entries),
                       supplements=False, previous=previous)


class TargetCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_content_stability_and_complete_mandatory_catalog(self) -> None:
        tmp_path = self.directory
        metadata = _model(tmp_path, '<Model name="M"><Body name="bone"/><Sensor name="novel_sensor"/></Model>')
        first = _build(tmp_path, {"fixture": metadata})
        second = _build(tmp_path, {"fixture": metadata})
        assert canonical_bytes(first) == canonical_bytes(second)
        validate_manifest(first)
        assert len(MANDATORY) == 8
        assert first["counts"]["mandatory_leaves"] == 95
        assert {f"mandatory:{domain}/{name}" for domain, names in MANDATORY.items() for name in names} <= {
            leaf["semantic_id"] for leaf in first["leaves"]
        }
        assert first["integrated_qualification"] == "unknown"
        assert all(leaf["evidence_status"] == "unknown" for leaf in first["leaves"])


    def test_source_scope_is_not_merged_by_similar_names(self) -> None:
        tmp_path = self.directory
        metadata = _model(tmp_path, '<Model name="M"><Body name="same"/><Muscle name="same"/>\n'
                          '<Body name="other"><Joint name="same"/></Body></Model>')
        value = _build(tmp_path, {"left_source": metadata, "right_source": metadata})
        same = [leaf for leaf in value["leaves"] if leaf["name"] == "same"]
        assert len(same) == 6
        assert len({leaf["semantic_id"] for leaf in same}) == 6

    def test_historical_eighty_leaf_manifest_extends_without_rewriting_old_targets(self) -> None:
        before = _build(self.directory, {})
        before["compiler"] = "numilab-human.target-coverage.1"
        before["leaves"] = [leaf for leaf in before["leaves"] if leaf["domain"] != "systemic_physiology"]
        before["counts"].update(leaves=80, mandatory_leaves=80, by_kind={"mandatory_target": 80})
        _seal(before)
        validate_manifest(before, allow_legacy=True)
        with self.assertRaisesRegex(HumanImportError, "historical"):
            validate_manifest(before)
        after = _build(self.directory, {}, previous=before)
        validate_transition(before, after)
        assert after["compiler"] == "numilab-human.target-coverage.2"
        assert after["counts"]["mandatory_leaves"] == 95
        old = {leaf["leaf_sha256"]: leaf for leaf in before["leaves"]}
        current = {leaf["leaf_sha256"]: leaf for leaf in after["leaves"]}
        assert old.keys() < current.keys()
        assert all(current[key] == leaf for key, leaf in old.items())
        with self.assertRaisesRegex(HumanImportError, "historical"):
            validate_transition(after, before)


    def test_unnamed_routes_constraints_and_new_named_kinds_are_retained(self) -> None:
        tmp_path = self.directory
        inventory = Inventory()
        key = inventory.source("source", {"license": "fixture"})
        data = b'<mujoco><tendon><spatial name="route"><site site="a"/><geom geom="wrap"/></spatial></tendon><equality><joint joint1="dependent" joint2="master"/></equality><sensor><user objname="b"/></sensor><novel name="unknown_physiology"/></mujoco>'
        assert _xml_declarations(io.BytesIO(data), inventory, key, "model.xml") == 6
        kinds = {leaf["kind"] for leaf in inventory.leaves.values()}
        assert {"xml:spatial", "xml:site", "xml:geom", "xml:joint", "xml:user", "xml:novel"} == kinds


    def test_changed_revision_and_missing_sources_cannot_erase_old_targets(self) -> None:
        tmp_path = self.directory
        original = _model(tmp_path, '<Model name="M"><Body name="original"/></Model>')
        before = _build(tmp_path, {"fixture": original})
        updated = _model(tmp_path, '<Model name="M"><Body name="replacement"/></Model>')
        updated["revision"] = "fixture-2"
        after = _build(tmp_path, {"fixture": updated}, previous=before)
        validate_transition(before, after)
        assert {leaf["leaf_sha256"] for leaf in before["leaves"]} < {
            leaf["leaf_sha256"] for leaf in after["leaves"]
        }
        assert {"original", "replacement"} <= {leaf["name"] for leaf in after["leaves"]}
        absent = _build(tmp_path, {}, previous=after)
        validate_transition(after, absent)
        assert len(absent["leaves"]) > len(after["leaves"])
        assert absent["scope_status"] == "blocked"
        assert len(absent["register_history"]) == 2


    def test_unknown_source_inventory_is_a_blocking_leaf(self) -> None:
        tmp_path = self.directory
        value = _build(tmp_path, {"restricted": {"manual_download_required": True, "license": "restricted"}})
        assert value["scope_status"] == "blocked"
        assert value["counts"]["unresolved_current_registers"] == 1
        assert any(leaf["kind"] == "unresolved_source_register" for leaf in value["leaves"])


    def test_deleted_prior_leaf_rejected_even_after_rehash(self) -> None:
        tmp_path = self.directory
        before = _build(tmp_path, {"fixture": _model(tmp_path, '<Model name="M"/>')})
        after = copy.deepcopy(before)
        after["leaves"] = [leaf for leaf in after["leaves"] if leaf["name"] != "M"]
        after["counts"]["leaves"] -= 1
        del after["counts"]["by_kind"]["xml:Model"]
        _seal(after)
        validate_manifest(after)
        with self.assertRaisesRegex(HumanImportError, "deletes or changes prior leaves"):
            validate_transition(before, after)


    def test_manifest_tampering_is_rejected(self) -> None:
        tmp_path = self.directory
        for change in ("source", "leaf", "summary", "promotion", "mandatory"):
            with self.subTest(change=change):
                value = _build(tmp_path, {})
                if change == "source":
                    value["source_records"][0]["metadata"]["license"] = "changed"
                elif change == "leaf":
                    value["leaves"][0]["name"] = "changed"
                elif change == "summary":
                    value["counts"]["leaves"] = 0
                elif change == "promotion":
                    value["integrated_qualification"] = "qualified"
                else:
                    leaf = value["leaves"].pop()
                    value["counts"]["leaves"] -= 1
                    value["counts"]["by_kind"][leaf["kind"]] -= 1
                _seal(value)
                with self.assertRaises(HumanImportError):
                    validate_manifest(value)

    def test_hash_mismatch_rejected_before_source_enumeration(self) -> None:
        tmp_path = self.directory
        metadata = _model(tmp_path, '<Model name="M"/>')
        (tmp_path / "model.osim").write_text('<Model name="changed"/>')
        with self.assertRaisesRegex(HumanImportError, "source hash mismatch"):
            _build(tmp_path, {"fixture": metadata})


    def test_archive_inventory_uses_pinned_bytes_not_dirty_checkout(self) -> None:
        tmp_path = self.directory
        (tmp_path / "source").mkdir()
        archive_path = tmp_path / "source/archive.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for name in ("variants/a.xml", "variants/b.xml"):
                content = b'<mujoco><body name="retained"/></mujoco>'
                item = tarfile.TarInfo("root/" + name)
                item.size = len(content)
                archive.addfile(item, io.BytesIO(content))
        metadata = {"storage_dir": "source", "archive_file": "archive.tar.gz",
                    "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}
        value = _build(tmp_path, {"fixture": metadata})
        assert sum(leaf["name"] == "retained" for leaf in value["leaves"]) == 2
        assert all(register["status"] == "materialized" for register in value["registers"])


    def test_composed_ir_requires_explicit_pin_and_retains_every_actuator_route(self) -> None:
        tmp_path = self.directory
        metadata = {"archive_sha256": "a" * 64, "revision": "pinned"}
        value = {"schema": "numi.human.myosim-mujoco-export.v1", "source": metadata,
                 "model": {"body_count_with_world": 1, "joint_count": 0, "nu": 2, "tendon_count": 2},
                 **{key: [] for key in ("bodies", "joints", "joint_equalities", "sites", "wrap_geometries")},
                 "muscles": [{"name": "right", "id": 0, "tendon": 0, "route": [{"source_id": 1}, {"source_id": 2}]},
                             {"name": "left", "id": 1, "tendon": 1, "route": [{"source_id": 3}, {"source_id": 4}]}]}
        path = tmp_path / "ir.json"
        path.write_bytes(canonical_bytes(value))
        inventory = Inventory()
        key = inventory.source("myosim_fullbody", metadata)
        with self.assertRaisesRegex(HumanImportError, "explicit SHA-256"):
            _myosim_ir(inventory, key, path, None, metadata)
        _myosim_ir(inventory, key, path, hashlib.sha256(path.read_bytes()).hexdigest(), metadata)
        assert len(inventory.leaves) == 6
        assert {leaf["name"] for leaf in inventory.leaves.values()} >= {"left", "right"}
        value["source"]["revision"] = "drifted"
        path.write_bytes(canonical_bytes(value))
        with self.assertRaisesRegex(HumanImportError, "pinned source identity"):
            _myosim_ir(inventory, key, path, hashlib.sha256(path.read_bytes()).hexdigest(),
                       {"archive_sha256": "a" * 64, "revision": "pinned"})


    def test_malformed_pinned_legacy_xml_remains_an_explicit_gap(self) -> None:
        tmp_path = self.directory
        value = _build(tmp_path, {"legacy": _model(tmp_path, '<Model><Body name="retained"/><Body name="bad&name"/></Model>')})
        assert value["scope_status"] == "blocked"
        assert value["registers"][0]["status"] == "invalid_source_xml"
        assert any(leaf["name"] == "retained" for leaf in value["leaves"])
        assert any(leaf["kind"] == "unresolved_source_register" for leaf in value["leaves"])


    def test_nonmuscle_inventory_retains_routes_dependencies_and_source_only_status(self) -> None:
        metadata = {"archive_sha256": "a" * 64, "revision": "pinned"}
        parameters = {
            "tendon_stiffness": 0.0, "tendon_damping": 0.0,
            "tendon_lengthspring": [0.04, 0.04], "tendon_limited": False,
            "tendon_range": [0.0, 0.0], "tendon_frictionloss": 0.0,
            "tendon_solref_lim": [0.02, 1.0], "tendon_solimp_lim": [0.9, 0.95, 0.001, 0.5, 2.0],
            "tendon_solref_fri": [0.02, 1.0], "tendon_solimp_fri": [0.9, 0.95, 0.001, 0.5, 2.0],
            "tendon_actuatorid": -1,
        }
        tendon = {"name": "ligament", "id": 1, "native_mechanics_status": "not_lowered",
                  "compiled_mujoco_parameters": parameters,
                  "route": [{"kind": "site", "source_id": 1, "side_site_source_id": -1},
                            {"kind": "sphere", "source_id": 10, "side_site_source_id": 3},
                            {"kind": "site", "source_id": 2, "side_site_source_id": -1}],
                  "sites": [{"id": index, "name": "site" + str(index)} for index in (1, 2, 3)],
                  "wrap_geometries": [{"id": 10, "name": "source_wrap"}]}
        value = {"schema": "numi.human.myosim-mujoco-export.v1", "source": metadata,
                 "model": {"body_count_with_world": 1, "joint_count": 0, "nu": 1, "tendon_count": 2},
                 **{key: [] for key in ("bodies", "joints", "joint_equalities", "sites", "wrap_geometries")},
                 "muscles": [{"name": "muscle", "id": 0, "tendon": 0, "route": []}],
                 "nonmuscle_tendons": [tendon]}
        path = self.directory / "ir.json"

        def load(candidate: dict) -> Inventory:
            path.write_bytes(canonical_bytes(candidate))
            inventory = Inventory()
            key = inventory.source("myosim_fullbody", metadata)
            _myosim_ir(inventory, key, path, hashlib.sha256(path.read_bytes()).hexdigest(), metadata)
            return inventory

        inventory = load(value)
        self.assertEqual(len(inventory.leaves), 9)
        self.assertEqual(sum(leaf["kind"] == "composed:route_element" for leaf in inventory.leaves.values()), 3)
        self.assertTrue(all(leaf["evidence_status"] == "unknown" for leaf in inventory.leaves.values()))
        self.assertTrue(all(register["status"] == "materialized" for register in inventory.registers))
        self.assertEqual(next(leaf for leaf in inventory.leaves.values() if leaf["name"] == "ligament")["declaration_sha256"], digest(tendon))

        absent = copy.deepcopy(value)
        del absent["nonmuscle_tendons"]
        self.assertTrue(any(register["status"] == "incomplete_composed_inventory" for register in load(absent).registers))

        for mutation in ("missing", "overlap", "bool_id", "duplicate", "dangling", "missing_parameter", "promoted", "conflicting_dependency"):
            with self.subTest(mutation=mutation):
                bad = copy.deepcopy(value)
                entry = bad["nonmuscle_tendons"][0]
                if mutation == "missing":
                    bad["nonmuscle_tendons"] = []
                elif mutation == "overlap":
                    entry["id"] = 0
                elif mutation == "bool_id":
                    entry["id"] = True
                elif mutation == "duplicate":
                    bad["nonmuscle_tendons"].append(copy.deepcopy(entry))
                elif mutation == "dangling":
                    entry["route"][1]["side_site_source_id"] = 99
                elif mutation == "missing_parameter":
                    del entry["compiled_mujoco_parameters"]["tendon_stiffness"]
                elif mutation == "promoted":
                    entry["native_mechanics_status"] = "qualified"
                elif mutation == "conflicting_dependency":
                    bad["sites"] = [{"id": 1, "name": "different source site"}]
                with self.assertRaises(HumanImportError):
                    load(bad)


    def test_cli_is_exposed_and_output_is_immutable(self) -> None:
        tmp_path = self.directory
        from numilab_human.cli import parser
        arguments = parser().parse_args(["target-coverage", "--output", str(tmp_path / "coverage.json")])
        assert arguments.handler is command
        arguments.sources = tmp_path
        arguments.source_lock = _lock(tmp_path, {})
        # Exercise the command's real builder without unrelated source acquisition.
        import numilab_human.target_coverage as coverage
        original = coverage.materialize
        patcher = patch.object(coverage, "materialize", lambda **kwargs: original(**kwargs, supplements=False))
        patcher.start()
        self.addCleanup(patcher.stop)
        assert command(arguments) == 0
        original_bytes = arguments.output.read_bytes()
        assert command(arguments) == 0
        assert arguments.output.read_bytes() == original_bytes
        arguments.source_lock = _lock(tmp_path, {"new": {"license": "unknown"}})
        with self.assertRaisesRegex(HumanImportError, "immutable"):
            command(arguments)
        assert arguments.output.read_bytes() == original_bytes
