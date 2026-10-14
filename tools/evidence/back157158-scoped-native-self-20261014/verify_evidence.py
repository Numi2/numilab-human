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


def canonical_json_sha256(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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

    # The corrected v2 report maps the fresh emitted candidate to the already
    # self-tested candidate. The native invocation receipt and later prepared
    # integrated receipt are intentionally different artifacts.
    row158_contract = manifest["rows"]["158"]
    row158_v2 = get(row158_contract["fresh_capture_equivalence_pin"])
    row158_erratum = get(row158_contract["fresh_capture_equivalence_v1_erratum_pin"])
    require(row158_v2["schema"] == "numi.human.row158.tested-candidate-fresh-native-step0-equivalence.v2",
            "row158 fresh-capture v2 schema mismatch")
    require(row158_v2["typed_identity"] == [51005, 158], "row158 v2 typed identity mismatch")
    require(row158_v2["status"] == row158_contract["fresh_capture_equivalence_status"],
            "row158 v2 status mismatch")
    require(row158_v2["candidate_admission"] is False
            and row158_v2["native_admission"] is False
            and row158_v2["geometry_predicates_called"] is False,
            "row158 v2 was promoted beyond metadata correspondence")
    capture_map = row158_v2["capture_mapping"]
    require(capture_map["ordered_candidate_face_count"] == 8934
            and capture_map["candidate_active_source_vertex_count"] == 4467
            and capture_map["candidate_unused_source_slots_retained"] == 20
            and capture_map["retired_source_face_count"] == row158_contract["retired_face_count"],
            "row158 v2 active/retired geometry counts mismatch")
    require(capture_map["candidate_faces_match_old_and_fresh_exactly"] is True,
            "row158 ordered candidate faces differ between tested and fresh captures")
    equivalence = row158_v2["step0_capture_equivalence"]
    require(equivalence["active_vertex_count"] == capture_map["candidate_active_source_vertex_count"]
            and equivalence["position_changed_scalars"] == 0
            and equivalence["position_max_ulp"] == 0
            and equivalence["normal_changed_scalars"] == 0
            and equivalence["normal_max_ulp"] == 0,
            "row158 active position/normal bytes are not exact")
    composition_v2 = row158_v2["source_and_composition"]
    require(composition_v2["candidate_vertices6_binding_indices_weights_equal_to_base_row"] is True
            and composition_v2["candidate_binding_table_byte_exact"] is True
            and composition_v2["row158_binding_vertex_and_ordered_face_bytes_equal_fresh_source"] is True,
            "row158 candidate binding/source correspondence is not byte-exact")

    def require_v2_input(pin_id: str, report_key: str) -> None:
        pinned = pin_by_id[pin_id]
        basename_matches = [digest for path, digest in row158_v2["input_pins"].items()
                            if Path(path).name == Path(pinned["path"]).name]
        require(basename_matches.count(pinned["sha256"]) == 1,
                f"row158 v2 input pin mismatch: {report_key}")

    for pin_id, report_key in (
        ("row158_candidate_base", "candidate base"),
        ("row158_candidate", "candidate"),
        ("row158_old_tested_source_nhtiss", "old tested source"),
        ("row158_composed_nhtiss", "fresh emission source"),
        ("integrated_nhtiss_payload", "later integrated source"),
        ("self_capture_pack_92c", "old self-test capture"),
        ("row158_step0_self", "candidate self report"),
        ("fresh_step0_pack_ba3e", "fresh native capture"),
        ("fresh_step0_receipt_ba3e", "fresh native capture receipt"),
        ("row157_composition_report", "final composition report"),
    ):
        require_v2_input(pin_id, report_key)
    require(row158_v2["nhtiss_source_row"]["candidate_base_npz_sha256"]
            == pin_by_id["row158_candidate_base"]["sha256"]
            and row158_v2["nhtiss_source_row"]["candidate_npz_sha256"]
            == pin_by_id["row158_candidate"]["sha256"],
            "row158 v2 candidate/base hashes mismatch")
    require(row158_v2["nhtiss_source_row"]["old_tested_source_sha256"]
            == pin_by_id["row158_old_tested_source_nhtiss"]["sha256"]
            and row158_v2["nhtiss_source_row"]["fresh_emission_source_sha256"]
            == pin_by_id["row158_composed_nhtiss"]["sha256"]
            and row158_v2["nhtiss_source_row"]["final_integrated_source_sha256"]
            == pin_by_id["integrated_nhtiss_payload"]["sha256"],
            "row158 source payload correspondence mismatch")
    reused_gate = row158_v2["reused_candidate_self_gate"]
    require(reused_gate["status"] == row158_self["status"]
            and reused_gate["exact_pair_count"] == 0
            and reused_gate["sha256"] == pin_by_id["row158_step0_self"]["sha256"]
            and reused_gate["scope"] == "row158 retirement candidate only, evaluated on previous actual step0 capture",
            "row158 reused self result scope or pin mismatch")

    v1_pin = pin_by_id["row158_fresh_capture_equivalence_v1"]
    require(row158_v2["correction_of"]["sha256"] == v1_pin["sha256"]
            and row158_erratum["original_report_sha256"] == v1_pin["sha256"]
            and row158_v2["correction_of"]["path"] == v1_pin["path"]
            and row158_erratum["original_report_preserved"] is True
            and row158_erratum["corrected_report_path"]
            == row158_contract["fresh_capture_equivalence_remote_path"],
            "row158 v1/v2 correction chain mismatch")
    require(row158_v2["prior_old_full_row_comparison"]["sha256"]
            == pin_by_id["row158_old_full_row_capture_comparison"]["sha256"]
            and row158_v2["prior_old_full_row_comparison"]["retained_as_distinct_old_full_row_result"] is True,
            "row158 prior full-row comparison was not preserved distinctly")

    invocation = get(row158_contract["native_invocation_pin"])
    actual_receipt_pin = pin_by_id[row158_contract["native_invocation_receipt_pin"]]
    prepared_receipt_pin = pin_by_id[row158_contract["prepared_integrated_receipt_pin"]]
    invocation_receipt = get(row158_contract["native_invocation_receipt_pin"])
    prepared_receipt = get(current["pin"])
    require(current["prepared_integrated_receipt_was_passed_to_row158_native"] is False,
            "prepared integrated receipt was incorrectly labeled as the native invocation receipt")
    require(current["prepared_integrated_receipt_pin"] == current["pin"]
            and current["row158_native_invocation_pin"] == row158_contract["native_invocation_pin"]
            and current["row158_native_invocation_receipt_pin"] == row158_contract["native_invocation_receipt_pin"],
            "manifest actual/prepared receipt roles mismatch")
    native_receipt_v2 = row158_v2["nhtiss_source_row"]["native_invocation_receipt"]
    prepared_receipt_v2 = row158_v2["nhtiss_source_row"]["final_prepared_receipt"]
    require(native_receipt_v2["actually_passed_to_native"] is True
            and native_receipt_v2["sha256"] == actual_receipt_pin["sha256"]
            and native_receipt_v2["path"] == actual_receipt_pin["path"],
            "row158 v2 actual native receipt pin mismatch")
    require(prepared_receipt_v2["passed_to_native"] is False
            and prepared_receipt_v2["sha256"] == prepared_receipt_pin["sha256"]
            and prepared_receipt_v2["path"] == prepared_receipt_pin["path"]
            and actual_receipt_pin["sha256"] != prepared_receipt_pin["sha256"],
            "row158 prepared receipt was confused with the actual invocation receipt")
    argv = invocation["argv"]

    def option_value(option: str) -> str:
        require(option in argv, f"native invocation omitted {option}")
        index = argv.index(option)
        require(index + 1 < len(argv), f"native invocation has no value for {option}")
        return argv[index + 1]

    require(Path(option_value("--resting-anatomy-receipt")).resolve()
            == Path(actual_receipt_pin["path"]).resolve(),
            "native argv did not consume the pinned a779 receipt")
    require(Path(option_value("--soft-tissue-payload")).resolve()
            == Path(pin_by_id["row158_composed_nhtiss"]["path"]).resolve(),
            "native argv did not consume the row158 source payload")
    require(Path(option_value("--skin-payload")).resolve()
            == Path(pin_by_id["skin_payload_b010"]["path"]).resolve(),
            "native argv skin payload path mismatch")
    for artifact_pin_id in ("row158_composed_nhtiss", "skin_payload_b010"):
        artifact_pin = pin_by_id[artifact_pin_id]
        asset_hashes = [digest for path, digest in invocation["asset_sha256"].items()
                        if Path(path).resolve() == Path(artifact_pin["path"]).resolve()]
        require(asset_hashes == [artifact_pin["sha256"]],
                f"native invocation asset hash mismatch: {artifact_pin_id}")
    require(row158_v2["nhtiss_source_row"]["final_integrated_source_sha256"]
            == pin_by_id[current["nhtiss_payload_pin"]]["sha256"],
            "prepared receipt NHTISS is not the manifest's later integrated payload")

    # The skin is an intentionally open shell: the two retained boundary loops
    # are the two ocular interfaces, not missing volume caps.
    actual_accounting = invocation_receipt["mass_geometry_accounting"]
    prepared_accounting = prepared_receipt["mass_geometry_accounting"]
    accounting_digest = canonical_json_sha256(actual_accounting)
    skin_contract = row158_v2["skin_scope_equivalence"]
    skin_manifest = manifest["retained_placement_inventory"]["skin"]
    require(actual_accounting == prepared_accounting
            and accounting_digest == skin_contract["skin_accounting_canonical_sha256"]
            and accounting_digest == skin_manifest["skin_accounting_sha256"],
            "skin accounting differs between actual and prepared receipts")
    boundary = actual_accounting["skin_boundary_repair"]
    openings = [item["interface"] for item in boundary["retained_anatomical_interfaces"]]
    expected_opening = "ocular surface opening; passive eyeball interface"
    interfaces = boundary["retained_anatomical_interfaces"]
    opening_bounds = [item["bounds_world_m"] for item in interfaces]
    opening_vertex_counts = [len(item["source_vertex_ids"]) for item in interfaces]
    require(openings == [expected_opening, expected_opening]
            and openings == skin_contract["intended_openings"]
            and openings == skin_manifest["declared_openings"]
            and opening_bounds == skin_manifest["declared_opening_bounds_world_m"]
            and opening_vertex_counts == skin_manifest["declared_opening_source_vertex_counts"],
            "skin does not retain exactly the two declared ocular openings")
    require(actual_accounting["skin_boundary_edge_count"] == 115
            and actual_accounting["skin_boundary_loop_count"] == 2
            and actual_accounting["skin_boundary_branch_vertex_count"] == 0
            and actual_accounting["closed_skin_volume_m3"] is None
            and boundary["qualification"] == skin_contract["boundary_repair_qualification"]
            and skin_contract["native_skin_qualification"] == "pending"
            and skin_contract["ocular_source_vertices_unchanged"] is True,
            "skin open-boundary scope or qualification mismatch")
    skin_pin = pin_by_id[skin_manifest["current_payload_pin"]]
    require(skin_pin["sha256"] == skin_contract["actual_invocation_skin_payload_sha256"]
            and actual_accounting["medial_thigh_reference_correction"]["candidate_sha256"]
            == skin_pin["sha256"],
            "skin payload/candidate hash mismatch")

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
