import copy

import pytest

from numilab_human.resting_brachiocephalic_joint_parent_composition import (
    extend_joint_source_inventory,
    merge_vascular_parent_maps,
)
from tools import prepare_brachiocephalic_joint_parent_bindings as prepare


def _hash(letter):
    return letter * 64


def _common():
    return {
        "anatomy_payload_sha256": _hash("a"),
        "child_payload_sha256": _hash("b"),
        "source_payloads": [
            {"path": f"assets/common-{i}.nhtissue", "sha256": _hash(letter)}
            for i, letter in enumerate("123456")
        ],
        "surfaces": [],
        "anatomy_surfaces": [
            {"stable_id": 475, "vertices": [[[5, 1, 0, 0.5], [5, 2, 0, 0.5]]]},
            {"stable_id": 477, "vertices": [[[5, 3, 0, 0.5], [5, 4, 0, 0.5]]]},
        ],
    }


def _joint():
    return {
        "anatomy_payload_sha256": _hash("a"),
        "child_payload_sha256": _hash("c"),
        "source_payloads": [
            {"path": f"assets/joint-{i}.nhtissue", "sha256": _hash(letter)}
            for i, letter in enumerate("789abc")
        ],
        "surfaces": [
            {"stable_id": 132, "vertices": [[[0, 1, 0, 1.0]]]},
            {"stable_id": 134, "vertices": [[[5, 2, 0, 1.0]]]},
            {"stable_id": 153, "vertices": [[[2, 3, 0, 1.0]]]},
            {"stable_id": 154, "vertices": [[[3, 4, 0, 1.0]]]},
        ],
        "anatomy_surfaces": [
            {"stable_id": 477, "vertices": [[[4, 3, 0, 1.0]]], "kept": "joint-owner"}
        ],
    }


def test_vascular_map_remap_uses_source_payload_identity_and_preserves_joint_owners():
    common = _common()
    joint = _joint()
    before_sources = copy.deepcopy(joint["source_payloads"])
    before_surfaces = copy.deepcopy(joint["surfaces"])
    extended, source_map, inventory_audit = extend_joint_source_inventory(
        common, joint,
        append_paths_by_common_index={5: "assets/route110-carriers.nhtissue"},
    )

    merged, audit = merge_vascular_parent_maps(
        common, extended,
        final_anatomy_sha256=_hash("d"),
        final_child_sha256=_hash("c"),
        common_source_to_joint=source_map,
    )

    rows = {row["stable_id"]: row for row in merged["anatomy_surfaces"]}
    assert rows[475]["vertices"] == [[[6, 1, 0, 0.5], [6, 2, 0, 0.5]]]
    assert rows[477]["vertices"] == [[[6, 3, 0, 0.5], [6, 4, 0, 0.5]]]
    assert rows[477]["kept"] == "joint-owner"
    assert merged["source_payloads"][:6] == before_sources
    assert merged["source_payloads"][5]["sha256"] == _hash("c")
    assert merged["source_payloads"][6]["sha256"] == _hash("6")
    assert source_map == {5: 6}
    assert inventory_audit["source_payloads_appended"] == [{
        "common_source_index": 5,
        "joint_source_index": 6,
        "sha256": _hash("6"),
        "path": "assets/route110-carriers.nhtissue",
    }]
    assert merged["surfaces"] == before_surfaces
    assert merged["anatomy_payload_sha256"] == _hash("d")
    assert merged["child_payload_sha256"] == _hash("c")
    assert audit["source_hash_to_joint_index"] == {_hash("6"): 6}
    assert audit["remapped_parent_term_count"] == 4


def test_vascular_map_rejects_index_map_without_matching_hash():
    with pytest.raises(ValueError, match="preserve payload identity"):
        merge_vascular_parent_maps(
            _common(), _joint(),
            final_anatomy_sha256=_hash("d"),
            final_child_sha256=_hash("c"),
            common_source_to_joint={5: 0},
        )


def test_vascular_map_rejects_nonpositive_or_nonunit_parent_weights():
    common = _common()
    common["anatomy_surfaces"][0]["vertices"][0][0][3] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        merge_vascular_parent_maps(
            common, _joint(),
            final_anatomy_sha256=_hash("d"),
            final_child_sha256=_hash("c"),
            common_source_to_joint={5: 6},
        )


def test_vascular_map_rejects_unmapped_referenced_source():
    with pytest.raises(ValueError, match="every referenced common source"):
        merge_vascular_parent_maps(
            _common(), _joint(),
            final_anatomy_sha256=_hash("d"),
            final_child_sha256=_hash("c"),
            common_source_to_joint={},
        )



def test_source_inventory_reuses_existing_payload_index_without_reordering():
    common = _common()
    joint = _joint()
    joint["source_payloads"][3]["sha256"] = _hash("6")
    extended, source_map, audit = extend_joint_source_inventory(
        common, joint, append_paths_by_common_index={})

    assert source_map == {5: 3}
    assert extended["source_payloads"] == joint["source_payloads"]
    assert audit["source_payloads_appended"] == []


def test_source_inventory_requires_a_path_for_missing_parent_hash():
    with pytest.raises(ValueError, match="missing output path"):
        extend_joint_source_inventory(_common(), _joint(), append_paths_by_common_index={})


def _owner_fixture(row_157_indices=(2, 3)):
    source_payloads = [
        {"path": f"source-{index}.nhtissue", "sha256": _hash(str(index))}
        for index in range(6)
    ]
    source_indices = {
        132: (0,), 134: (5,), 153: (2,), 155: (2,),
        154: (3,), 156: (3,), 157: row_157_indices, 158: (3,),
    }
    surfaces = [
        {"stable_id": stable_id,
         "vertices": [[[index, index + 1, 0, 1.0]
                       for index in indices]]}
        for stable_id, indices in source_indices.items()
    ]
    return {"source_payloads": source_payloads, "surfaces": surfaces}


def test_joint_owner_inventory_preserves_exact_dual_parent_row_157():
    audit = prepare._verify_joint_owner_roles(_owner_fixture())

    assert audit["157"]["role"] == "back-parent-blend"
    assert audit["157"]["source_indices"] == [2, 3]
    assert audit["157"]["source_sha256_by_index"] == {
        "2": _hash("2"), "3": _hash("3")}


def test_joint_owner_inventory_rejects_collapsing_row_157_to_one_parent():
    with pytest.raises(ValueError, match="source ownership changed for row 157"):
        prepare._verify_joint_owner_roles(_owner_fixture(row_157_indices=(3,)))


def test_vascular_map_accepts_only_roundoff_bounded_float32_unit_partitions():
    import struct

    common = _common()
    joint = _joint()
    joint["source_payloads"].append(
        {"path": "assets/bridge.nhtissue", "sha256": _hash("6")})
    down = struct.unpack("<f", struct.pack("<f", 0.1))[0]
    up = struct.unpack("<f", struct.pack("<f", 0.9))[0]
    common["anatomy_surfaces"][0]["vertices"][0] = [
        [5, 1, 0, down], [5, 2, 0, up]]
    merged, _ = merge_vascular_parent_maps(
        common, joint,
        final_anatomy_sha256=_hash("d"),
        final_child_sha256=_hash("c"),
        common_source_to_joint={5: 6},
    )
    assert merged["anatomy_surfaces"][-1]["stable_id"] == 477


def test_vascular_map_rejects_nonfloat32_or_excessively_nonnormalized_weights():
    common = _common()
    joint = _joint()
    joint["source_payloads"].append(
        {"path": "assets/bridge.nhtissue", "sha256": _hash("6")})
    common["anatomy_surfaces"][0]["vertices"][0] = [
        [5, 1, 0, 0.1], [5, 2, 0, 0.9]]
    with pytest.raises(ValueError, match="not exact Float32"):
        merge_vascular_parent_maps(
            common, joint,
            final_anatomy_sha256=_hash("d"),
            final_child_sha256=_hash("c"),
            common_source_to_joint={5: 6},
        )

    common = _common()
    common["anatomy_surfaces"][0]["vertices"][0] = [
        [5, 1, 0, 0.5], [5, 2, 0, 0.5000001192092896]]
    with pytest.raises(ValueError, match="Float32 unit partition"):
        merge_vascular_parent_maps(
            common, joint,
            final_anatomy_sha256=_hash("d"),
            final_child_sha256=_hash("c"),
            common_source_to_joint={5: 6},
        )


def test_preflight_snapshot_matches_json_object_key_normalization():
    snapshot = prepare._json_snapshot({"source_map": {5: 6}})
    assert snapshot == {"source_map": {"5": 6}}


def _write_json(path, value):
    import json
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return path


def _serialized_output_fixture(root):
    import argparse
    import hashlib
    from copy import deepcopy
    from pathlib import Path

    root = Path(root)
    common_dir = root / "common"
    input_dir = root / "joint-input"
    output = root / "composition"
    for directory in (common_dir, input_dir, output / "back", output / "joint"):
        directory.mkdir(parents=True, exist_ok=True)

    def asset(path, contents):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()}

    nha = asset(common_dir / "thorax.nhanatomy", b"fixture NHA bytes")
    common_owner = asset(common_dir / "common-muscles.nhtissue", b"common owner")
    common_manifest = asset(common_dir / "common-muscles.manifest.json", b"common manifest")
    back_child = asset(root / "back.nhtissue", b"back child")
    joint_child = asset(root / "joint.nhtissue", b"joint child")
    back_manifest_path = root / "back.manifest.json"
    joint_manifest_path = root / "joint.manifest.json"
    back_manifest = _write_json(back_manifest_path, {
        "payload": {"sha256": back_child["sha256"]},
        "source": {"passive_surface_appends": [
            {"source_payload_sha256": common_owner["sha256"]}]},
    })
    joint_manifest = _write_json(joint_manifest_path, {
        "payload": {"sha256": joint_child["sha256"]},
        "source": {"passive_surface_appends": [
            {"source_payload_sha256": back_child["sha256"]}]},
    })
    back_manifest_sha = hashlib.sha256(back_manifest.read_bytes()).hexdigest()
    joint_manifest_sha = hashlib.sha256(joint_manifest.read_bytes()).hexdigest()
    common_sources = []
    joint_sources = []
    for index in range(6):
        common_sources.append(asset(
            common_dir / "assets" / f"common-{index}.nhtissue",
            f"common carrier {index}".encode()))
        joint_sources.append(asset(
            input_dir / "assets" / f"joint-{index}.nhtissue",
            f"joint carrier {index}".encode()))

    common_extension = {
        "anatomy_payload_sha256": nha["sha256"],
        "child_payload_sha256": common_owner["sha256"],
        "source_payloads": common_sources,
        "surfaces": [],
        "anatomy_surfaces": [
            {"stable_id": 475, "vertices": [[[5, 475, i, 1.0]] for i in range(902)]},
            {"stable_id": 477, "vertices": [[[5, 477, i, 1.0]] for i in range(1089)]},
        ],
    }
    joint_extension = {
        "anatomy_payload_sha256": nha["sha256"],
        "child_payload_sha256": joint_child["sha256"],
        "source_payloads": joint_sources,
        "surfaces": [
            {"stable_id": stable_id,
             "vertices": [[[index, 0, 0, 1.0]] for _ in range(1)]}
            for stable_id, index in ((132, 0), (134, 5), (153, 2), (155, 2),
                                     (154, 3), (156, 3), (158, 3))
        ],
        "anatomy_surfaces": [
            {"stable_id": 477, "vertices": [[[4, 477, 0, 1.0]]], "kept": "joint-owner"}
        ],
    }
    # Row 157 is the existing dual-parent blend, not a single-parent surface.
    joint_extension["surfaces"].append({
        "stable_id": 157, "vertices": [[[2, 0, 0, 0.5], [3, 0, 0, 0.5]]]
    })
    joint_bindings = _write_json(input_dir / "passive-tissue-parent-bindings.json",
                                 joint_extension)
    common_receipt_path = _write_json(common_dir / "resting-anatomy-receipt.json", {})
    common_report_path = _write_json(common_dir / "common-preparation.json", {})
    joint_preparation = _write_json(input_dir / "joint-preparation.json", {})
    joint_report = _write_json(input_dir / "joint-report.json", {})

    def native_owner(payload, manifest, manifest_sha):
        return {
            "payload_path": payload["path"], "sha256": payload["sha256"],
            "manifest_path": str(manifest), "manifest_sha256": manifest_sha,
        }

    base_provenance = {
        "source_id_map": {"525": {"source_member": "FJ3482"}},
        "rigid_payload_sha256": _hash("r"),
        "bodyparts_registration_sha256": _hash("g"),
        "bodyparts3d_archives": [{"hierarchy": "part_of", "sha256": _hash("s")}],
        "cardiac_geometry_binding": {"common_field": {"map": {"record_count": 621369}}},
        "native_muscle_surfaces": native_owner(
            common_owner, Path(common_manifest["path"]), common_manifest["sha256"]),
        "reference_surface_composition_binding_history": [],
    }
    common_payload = {
        "path": nha["path"], "sha256": nha["sha256"],
        "registration_fingerprint32": 7, "vertex_count": 1,
    }
    common_receipt = {
        "schema": "fixture-receipt-v1", "payload": common_payload,
        "provenance": base_provenance,
        "passive_tissue_parent_bindings": common_extension,
        "physical_state": {"unchanged": True},
    }
    _write_json(common_receipt_path, common_receipt)

    back_binding = {
        "prior_receipt_path": str(common_receipt_path),
        "composition_manifest_sha256": back_manifest_sha,
    }
    back_receipt = deepcopy(common_receipt)
    back_receipt["provenance"]["native_muscle_surfaces"] = native_owner(
        back_child, back_manifest_path, back_manifest_sha)
    back_receipt["provenance"]["reference_surface_composition_binding"] = back_binding
    back_receipt["provenance"]["reference_surface_composition_binding_history"] = [back_binding]
    back_receipt_path = _write_json(output / "back" / "resting-anatomy-receipt.json",
                                    back_receipt)

    joint_binding = {
        "prior_receipt_path": str(back_receipt_path),
        "composition_manifest_sha256": joint_manifest_sha,
    }
    final_receipt = deepcopy(back_receipt)
    final_receipt["provenance"]["native_muscle_surfaces"] = native_owner(
        joint_child, joint_manifest_path, joint_manifest_sha)
    final_receipt["provenance"]["reference_surface_composition_binding"] = joint_binding
    final_receipt["provenance"]["reference_surface_composition_binding_history"].append(
        joint_binding)

    route_carrier = common_sources[5]
    append_asset = output / "joint" / "assets" / "route110-carriers.nhtissue"
    append_asset.parent.mkdir(parents=True, exist_ok=True)
    append_asset.write_bytes(Path(route_carrier["path"]).read_bytes())
    extended, source_map, inventory_audit = extend_joint_source_inventory(
        common_extension, joint_extension,
        append_paths_by_common_index={5: "assets/route110-carriers.nhtissue"})
    merged, map_audit = merge_vascular_parent_maps(
        common_extension, extended,
        final_anatomy_sha256=nha["sha256"],
        final_child_sha256=joint_child["sha256"],
        common_source_to_joint=source_map,
    )
    final_receipt["passive_tissue_parent_bindings"] = merged

    # The alternate path is a symlink alias, exercising /Users/n vs /Users/numi
    # style canonical path binding without changing the frozen source identity.
    alias = root / "alternate-common-path"
    alias.symlink_to(common_dir, target_is_directory=True)
    final_receipt["payload"]["path"] = str(alias / "thorax.nhanatomy")
    final_receipt["provenance"]["brachiocephalic_joint_parent_binding_composition"] = {
        "scope": "Existing ABI5 parent map composition; source preparation only.",
        "common_receipt_path": str(common_receipt_path),
        "common_receipt_sha256": hashlib.sha256(common_receipt_path.read_bytes()).hexdigest(),
        "common_preparation_report_path": str(common_report_path),
        "common_preparation_report_sha256": hashlib.sha256(common_report_path.read_bytes()).hexdigest(),
        "back_child_path": back_child["path"], "back_child_sha256": back_child["sha256"],
        "back_manifest_path": str(back_manifest_path), "back_manifest_sha256": back_manifest_sha,
        "joint_preparation_path": str(joint_preparation),
        "joint_preparation_sha256": hashlib.sha256(joint_preparation.read_bytes()).hexdigest(),
        "joint_parent_binding_sha256": hashlib.sha256(joint_bindings.read_bytes()).hexdigest(),
        "joint_child_path": joint_child["path"], "joint_child_sha256": joint_child["sha256"],
        "joint_manifest_path": str(joint_manifest_path), "joint_manifest_sha256": joint_manifest_sha,
        "source_inventory": inventory_audit,
        "vascular_parent_maps": map_audit,
        "native_or_gpu_run": False, "geometry_or_anatomical_admission": "not evaluated",
        "physical_mass_state_or_force_added": False,
    }
    final_path = output / "joint" / "resting-anatomy-receipt.json"
    _write_json(final_path, final_receipt)

    args = argparse.Namespace(
        common_receipt=common_receipt_path, common_report=common_report_path,
        back_child=Path(back_child["path"]), back_manifest=back_manifest_path,
        back_report=root / "unused-back-report.json", joint_preparation=joint_preparation,
        joint_report=joint_report, joint_bindings=joint_bindings,
        joint_child=Path(joint_child["path"]), joint_manifest=joint_manifest_path,
        output=output,
    )
    roles = prepare._verify_joint_owner_roles(joint_extension)
    chain = {
        "common_receipt_muscle_owner_sha256": common_owner["sha256"],
        "back_manifest_direct_parent_sha256": common_owner["sha256"],
        "back_payload_sha256": back_child["sha256"],
        "joint_manifest_direct_parent_sha256": back_child["sha256"],
        "joint_child_sha256": joint_child["sha256"],
    }
    source_validation = {"status": "PASS_FIXTURE_SOURCE_BINDINGS"}
    frozen = {
        "ancestor_chain": chain, "joint_muscle_owner_indices": roles,
        "common_validation": {"source_asset_validation": source_validation},
    }
    proof = prepare._validate_final_composition_output(
        args, frozen, back_receipt_path=back_receipt_path,
        final_receipt_path=final_path)
    report = {
        "status": "source_parent_binding_composition_only_not_admission",
        "common_source_candidate_validation": {
            "nha_payload_sha256": nha["sha256"],
            "source_asset_validation": source_validation,
            "common_map_record_count": 621369,
            "geometry_predicates_run": False, "native_or_gpu_run": False,
        },
        "ancestor_chain": chain, "joint_muscle_owner_indices": roles,
        "common_to_joint_source_inventory": inventory_audit,
        "vascular_parent_maps": map_audit,
        "final_source_provenance_preservation": proof["source_provenance_preservation"],
        "final_receipt_path": str(final_path),
        "final_receipt_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "final_muscle_payload_path": joint_child["path"],
        "final_muscle_payload_sha256": joint_child["sha256"],
        "physical_mass_state_or_force_added": False,
        "geometry_or_anatomical_admission": "not evaluated",
        "all_pose_or_native_admission": "not evaluated",
        "final_output_structure_validation": proof,
    }
    report_path = output / "source-parent-composition-report.json"
    _write_json(report_path, report)
    prepare._assert_composition_report(
        args, frozen, final_receipt_path=final_path,
        report_path=report_path, report=report, output_proof=proof)
    return args, frozen, back_receipt_path, final_path, report_path


def test_serialized_output_validator_checks_full_receipt_and_resolves_path_alias(tmp_path):
    _serialized_output_fixture(tmp_path)


def test_serialized_output_validator_rejects_drift_in_an_unrelated_provenance_field(tmp_path):
    args, frozen, back_path, final_path, _report_path = _serialized_output_fixture(tmp_path)
    import json
    final = json.loads(final_path.read_text())
    final["provenance"]["source_id_map"]["525"]["source_member"] = "wrong-source"
    final_path.write_text(json.dumps(final, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="source-bound provenance"):
        prepare._validate_final_composition_output(
            args, frozen, back_receipt_path=back_path,
            final_receipt_path=final_path)


def test_serialized_output_validator_rejects_appended_source_path_escape(tmp_path):
    args, frozen, back_path, final_path, _report_path = _serialized_output_fixture(tmp_path)
    import json
    final = json.loads(final_path.read_text())
    final["passive_tissue_parent_bindings"]["source_payloads"][-1]["path"] = "../../escape.nhtissue"
    final_path.write_text(json.dumps(final, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="source inventory"):
        prepare._validate_final_composition_output(
            args, frozen, back_receipt_path=back_path,
            final_receipt_path=final_path)
