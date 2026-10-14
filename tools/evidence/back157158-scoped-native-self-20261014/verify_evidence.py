#!/usr/bin/env python3
"""Verify remote artifact pins and the scoped back157/158 report contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def verify(manifest_path: Path, check_hashes: bool = True) -> list[str]:
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "numi.human.back157158.scoped-evidence.v1",
            "manifest schema mismatch")
    pins = manifest.get("evidence_pins")
    require(isinstance(pins, list), "evidence_pins must be a list")
    pin_by_id = {item["id"]: item for item in pins}
    require(len(pin_by_id) == len(pins), "duplicate evidence pin id")
    loaded: dict[str, dict] = {}
    for item in pins:
        artifact = Path(item["path"])
        require(artifact.is_file(), f"missing evidence pin {item['id']}: {artifact}")
        if check_hashes:
            actual = sha256_file(artifact)
            require(actual == item["sha256"],
                    f"hash mismatch {item['id']}: expected {item['sha256']}, got {actual}")
        if artifact.suffix == ".json":
            loaded[item["id"]] = read_json(artifact)

    def get(pin_id: str) -> dict:
        require(pin_id in loaded, f"JSON evidence not loaded: {pin_id}")
        return loaded[pin_id]

    receipt = get("integrated_receipt")
    current = manifest["current_receipt"]
    require(receipt["payload"]["sha256"] == pin_by_id[current["nha_payload_pin"]]["sha256"],
            "receipt NHA child hash mismatch")
    muscle = receipt["provenance"]["native_muscle_surfaces"]
    require(muscle["sha256"] == pin_by_id[current["nhtiss_payload_pin"]]["sha256"],
            "receipt NHTISS provenance hash mismatch")
    require(muscle["manifest_sha256"] == pin_by_id[current["nhtiss_manifest_pin"]]["sha256"],
            "receipt NHTISS manifest hash mismatch")
    require(receipt["passive_tissue_parent_bindings"]["child_payload_sha256"] == muscle["sha256"],
            "receipt passive child binding differs from NHTISS provenance")

    composition = manifest["composition_chain"]
    comp158 = get("row158_composition_report")
    comp157 = get("row157_composition_report")
    require(comp158["payload_sha256"] == pin_by_id[composition["row158_output_payload_pin"]]["sha256"],
            "row158 composition payload pin mismatch")
    require(comp157["payload_sha256"] == pin_by_id[composition["row157_output_payload_pin"]]["sha256"],
            "row157 composition payload pin mismatch")
    require(comp158["input_sha256"].get(pin_by_id[composition["row158_input_payload_pin"]]["path"])
            == pin_by_id[composition["row158_input_payload_pin"]]["sha256"],
            "row158 composition input pin mismatch")
    require(comp158["input_sha256"].get(pin_by_id[composition["row158_input_manifest_pin"]]["path"])
            == pin_by_id[composition["row158_input_manifest_pin"]]["sha256"],
            "row158 composition input manifest pin mismatch")
    require(comp157["input_sha256"].get(pin_by_id[composition["row157_input_payload_pin"]]["path"])
            == pin_by_id[composition["row157_input_payload_pin"]]["sha256"],
            "row157 composition input does not consume row158 output")
    require(comp157["input_sha256"].get(pin_by_id["row158_composed_manifest"]["path"])
            == pin_by_id["row158_composed_manifest"]["sha256"],
            "row157 composition input manifest does not consume row158 output")
    for row, report in ((157, comp157), (158, comp158)):
        require(report["inputs_unchanged"] is True, f"row{row} composition input changed")
        require(report["binding_table_byte_exact"] is True,
                f"row{row} composition changed the binding table")
        changed = report["changed_rows"]
        require(len(changed) == 1 and changed[0]["stable_id"] == row,
                f"row{row} composition changed an unexpected row")
        require(report["reference_surface_composition"]["physical_route_mass_and_force_state_unchanged"] is True,
                f"row{row} composition changed physical route state")

    tissue_manifest = get("integrated_nhtiss_manifest")
    surfaces = tissue_manifest["source"]["surfaces"]
    by_id = {int(item["stable_id"]): item for item in surfaces}
    for row in (157, 158):
        record = by_id[row]
        expected = manifest["rows"][str(row)]
        require(record["label"] == expected["label"], f"row{row} label mismatch")
        require(record["member"] == expected["source_member"], f"row{row} source member mismatch")
        require(record["member_sha256"] == expected["source_sha256"], f"row{row} source hash mismatch")
        require(record["passive_visual_binding"]["mechanics_changed"] is False,
                f"row{row} passive binding changed mechanics")
        require(record["passive_visual_binding"]["myosim_route_added"] is False,
                f"row{row} unexpectedly added a MyoSim route")
        require(record["reference_surface_reconstruction"]["candidate_sha256"]
                == pin_by_id[expected["candidate_pin"]]["sha256"],
                f"row{row} manifest candidate pin mismatch")

    row157_prep = get("row157_preparation")
    require(row157_prep["status"] == "changed_face_delta_zero_not_full_self_admitted",
            "row157 cheap report status mismatch")
    require(row157_prep["actual_step0_delta"]["count"] == 0,
            "row157 changed-face delta is nonzero")
    row157_lineage = get("row157_lineage")
    require(row157_lineage["typed_identity"] == [51005, 157], "row157 lineage identity mismatch")
    require(row157_lineage["protected_412_face_ids"] == manifest["protected_back412"]["face_ids"],
            "protected back412 face list mismatch")
    require(row157_lineage["protected_412_faces_unchanged"] is True,
            "protected back412 faces changed")
    require(row157_lineage["retired_input_face_ids"] == [9349, 9382],
            "row157 retired face lineage mismatch")
    row157_self = get("row157_step0_self")
    require(row157_self["status"] == "step0_full_self_clear", "row157 full-self status mismatch")
    require(row157_self["candidate_admission"] is False and row157_self["native_admission"] is False,
            "row157 scoped result was promoted to admission")
    require(row157_self["stages"][-1]["count"] == 0, "row157 step0 self pairs are nonzero")
    require(row157_self["typed_row_mapping"]["typed_identity"] == [51005, 157],
            "row157 self report identity mismatch")
    row157_equiv = get("row157_fresh_capture_equivalence")
    require(row157_equiv["exact_equal"] is True, "row157 fresh capture equivalence failed")
    require(row157_equiv["fresh_pack_sha256"] == pin_by_id["fresh_step0_pack_ba3e"]["sha256"],
            "row157 equivalence does not pin the current step0 pack")
    require(row157_equiv["fields"]["positions_m"]["byte_exact"] is True
            and row157_equiv["fields"]["normals"]["byte_exact"] is True
            and row157_equiv["fields"]["face_indices"]["byte_exact"] is True
            and row157_equiv["fields"]["global_gids"]["byte_exact"] is True,
            "row157 source capture fields are not byte-exact across captures")

    row158_id = get("row158_identity_correction")
    require(row158_id["correct_typed_identity"]["semantic"] == 51005
            and row158_id["correct_typed_identity"]["stable_id"] == 158
            and row158_id["correct_typed_identity"]["source_member_id"] == "FJ1521M",
            "row158 identity correction mismatch")
    row158_lineage = get("row158_retired_face_lineage")
    require(row158_lineage["typed_row"] == [51005, 158], "row158 lineage identity mismatch")
    require(len(row158_lineage["retired_faces"]) == manifest["rows"]["158"]["retired_face_count"],
            "row158 retired face count mismatch")
    row158_self = get("row158_step0_self")
    require(row158_self["status"] == "pass_row158_candidate_actual_step0_self",
            "row158 step0 self status mismatch")
    require(row158_self["candidate_admission"] is False,
            "row158 scoped result was promoted to candidate admission")
    require(row158_self["stages"][-1]["count"] == 0, "row158 step0 self pairs are nonzero")
    require(row158_self["pins"].get(pin_by_id["self_capture_pack_92c"]["path"])
            == pin_by_id["self_capture_pack_92c"]["sha256"],
            "row158 self report capture pin mismatch")

    placement = manifest["retained_placement_inventory"]
    fb, provenance = receipt["functional_bindings"], receipt["provenance"]
    require(len(fb["head_structure_stable_ids"]) == placement["head_and_eyes"]["bound_stable_id_count"],
            "head/eye bound stable ID count mismatch")
    require(provenance["head_eye_boundary_repair"]["status"]
            == placement["head_and_eyes"]["eye_boundary_status"], "eye status differs from receipt")
    require(provenance["head_eye_387_contralateral_reference_replacement"]["status"]
            == placement["head_and_eyes"]["contralateral_lens_candidate_status"],
            "contralateral lens status differs from receipt")
    eye_report = placement["head_and_eyes"]["existing_test_report"]
    eye_provenance = provenance["head_eye_387_contralateral_reference_replacement"]
    require(eye_provenance["test_report_path"] == eye_report["path"]
            and eye_provenance["test_report_sha256"] == eye_report["sha256_recorded_in_current_receipt"],
            "eye test report reference differs from receipt")
    require(eye_report["artifact_retrieved_from_snapshot"] is False,
            "unavailable eye report was incorrectly marked retrieved")
    require(provenance["head_eye_source_exact_cleanup_candidate"]["status"]
            == placement["head_and_eyes"]["source_cleanup_status"],
            "head source cleanup status differs from receipt")
    require(provenance["pons_bounded_local_repair"]["stable_id"] == placement["pons_brainstem"]["stable_id"]
            and provenance["pons_bounded_local_repair"]["status"] == placement["pons_brainstem"]["status"],
            "pons status differs from receipt")
    require(fb["lung_stable_ids"] == placement["pleura_and_lungs"]["functional_lung_stable_ids"],
            "lung binding IDs differ from receipt")
    require(fb["pleura_stable_ids"] == [placement["pleura_and_lungs"]["functional_pleura_stable_id"]],
            "pleura binding ID differs from receipt")
    require(provenance["lung_final_shared_interface_rebuild"]["status"]
            == placement["pleura_and_lungs"]["lung_rebuild_status"], "lung status differs from receipt")
    require(fb["liver_surface_identity"]["stable_ids"] == placement["liver"]["functional_stable_ids"],
            "liver functional identity differs from receipt")
    require(fb["liver_surface_identity"]["retired_source_aliases"]
            == placement["liver"]["retired_alias"], "liver retired alias differs from receipt")
    require(provenance["retired_liver_alias_native_replay"]["native_qualification"]
            == placement["liver"]["alias_replay_status_in_receipt"],
            "liver replay status differs from receipt")
    require(len(fb["passive_viscera_geometry_binding"]["stable_ids"])
            == placement["passive_organs_and_viscera"]["functional_stable_id_count"],
            "passive viscera ID count differs from receipt")
    require(provenance["passive_viscera_extension"]["scope"]
            == placement["passive_organs_and_viscera"]["extension_scope"],
            "passive viscera scope differs from receipt")

    liver = get("liver_main_shell_eight_capture_report")
    require(liver["status"] == "pass" and len(liver["captures"]) == 8,
            "retained liver report scope mismatch")
    require(all(capture["self_pair_count"] == 0 and capture["closed_oriented_single_shell"]
                for capture in liver["captures"]), "retained liver shell gate is not clear")
    skin = get("skin_021_audit")
    require(skin["complete"] is True and skin["completed_capture_count"] == 8,
            "retained skin report scope mismatch")
    current_gate = manifest["current_all_neighbor_gate"]
    first_failure = get(current_gate["first_failure_pin"])
    require(first_failure["audit"]["identities"] == current_gate["first_failure_typed_pair"],
            "current all-neighbor first-failure identities mismatch")
    require(first_failure["audit"]["triangle_pair_count"]
            == current_gate["unexpected_triangle_pair_count"],
            "current all-neighbor first-failure count mismatch")
    require(first_failure["audit"]["passed"] is False, "current all-neighbor failure was overwritten")
    return [item["id"] for item in pins]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-only", action="store_true",
                        help="check report structure but skip external artifact SHA-256 reads")
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).with_name("manifest.json"))
    args = parser.parse_args()
    try:
        pin_ids = verify(args.manifest, check_hashes=not args.metadata_only)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    mode = "metadata" if args.metadata_only else "sha256+metadata"
    print(f"PASS: {len(pin_ids)} evidence pins and scoped report contract ({mode})")
    print("No geometry predicates were rerun; this is an integrity check only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
