from __future__ import annotations
import copy, hashlib, json, struct, sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/n/numi-human-common-skin-multipose-001")
E = Path("/Users/n/numi-human-resting-evidence-20261005")
RUN = E / "native-common-skin-multipose-clearance-candidate-927/candidate-run-1"
OUT = E / "native-common-skin-multipose-clearance-candidate-927/asset-candidate-001"
RECEIPT_OUT = E / "native-common-skin-multipose-clearance-candidate-927/registered-composed-001"
BASE_SKIN = E / "common-atlas-skin-composition-907/bodyparts3d-myosim-skinned-shell.nhskin"
BASE_SKIN_MANIFEST = E / "common-atlas-skin-composition-907/common-atlas-skin-geometry-registration.manifest.json"
BASE_NHA_RECEIPT = E / "lung-choroid-composition-924/final-v2/resting-anatomy-receipt.json"
BASE_NHA_MANIFEST = E / "lung-choroid-composition-924/final-v2/resting-anatomy-manifest.json"
DRIVER = E / "native-common-skin-multipose-clearance-candidate-927.py"
SOLVER = ROOT / "src/numilab_human/common_atlas_skin_clearance.py"
COMPOSER = ROOT / "src/numilab_human/common_atlas_skin_geometry_registration.py"

sys.path.insert(0, str(ROOT / "src"))
from numilab_human.skin_source_payload_preflight import decode_payload
from numilab_human.common_atlas_skin_geometry_registration import compose_disjoint_skin_position_corrections
from numilab_human import resting_anatomy


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())

def record(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}

def json_file(path: Path) -> dict:
    return json.loads(path.read_text())

def ancestry_run(stage: int) -> dict:
    run = E / f"native-common-skin-multipose-clearance-candidate-{stage}/candidate-run-1"
    result = {"stage": stage, "run_path": str(run), "artifacts": {}}
    for name in ("preflight.json", "solve-start.json", "progress-state.json", "solve-result.json", "candidate-report.json", "attempts.jsonl"):
        p = run / name
        if p.is_file():
            result["artifacts"][name] = record(p)
    attempts_path = run / "attempts.jsonl"
    attempts = []
    if attempts_path.is_file():
        for line in attempts_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            entry = {k: row.get(k) for k in ("attempt", "iteration", "backtrack", "scale", "status", "reason", "source_positions_f32_sha256", "nonocular_pair_counts_before_by_pose", "nonocular_pair_counts_by_pose", "nonocular_pair_counts_candidate_by_pose") if k in row}
            for key in ("candidate_npy_path", "target_audits_path"):
                p = Path(row[key]) if row.get(key) else None
                if p is not None:
                    if not p.is_file():
                        raise RuntimeError(f"historical attempt artifact is missing: {p}")
                    entry[key] = record(p)
            attempts.append(entry)
    result["attempts"] = attempts
    solve = json_file(run / "solve-result.json")
    result["status"] = solve.get("status")
    result["error_type"] = solve.get("error_type")
    result["error"] = solve.get("error")
    return result

for p in (BASE_SKIN, BASE_SKIN_MANIFEST, BASE_NHA_RECEIPT, BASE_NHA_MANIFEST, RUN / "candidate-source-positions-f32.npy", RUN / "candidate-report.json", RUN / "solve-result.json", RUN / "preflight.json", DRIVER, SOLVER, COMPOSER):
    if not p.is_file():
        raise FileNotFoundError(p)
if OUT.exists() or RECEIPT_OUT.exists():
    raise FileExistsError("refusing to overwrite an earlier candidate or receipt")

base_raw = BASE_SKIN.read_bytes()
base_sha = sha_bytes(base_raw)
base_manifest = json_file(BASE_SKIN_MANIFEST)
base_receipt = json_file(BASE_NHA_RECEIPT)
base_nha_manifest = json_file(BASE_NHA_MANIFEST)
if base_manifest.get("output_payload", {}).get("path") != str(BASE_SKIN) or base_manifest.get("output_payload", {}).get("sha256") != base_sha:
    raise RuntimeError("907 registration manifest does not bind exact 907 NHSKIN bytes")
if base_receipt.get("mass_geometry_accounting", {}).get("skin_payload_path") != str(BASE_SKIN) or base_receipt.get("mass_geometry_accounting", {}).get("skin_payload_sha256") != base_sha:
    raise RuntimeError("924 base receipt does not account for exact 907 NHSKIN bytes")
if base_nha_manifest.get("receipt", {}).get("path") != str(BASE_NHA_RECEIPT) or base_nha_manifest.get("receipt", {}).get("sha256") != sha_file(BASE_NHA_RECEIPT):
    raise RuntimeError("924 anatomy manifest does not bind its selected receipt")

report_path = RUN / "candidate-report.json"
report = json_file(report_path)
result_path = RUN / "solve-result.json"
solve_result = json_file(result_path)
preflight_path = RUN / "preflight.json"
preflight = json_file(preflight_path)
positions_path = RUN / "candidate-source-positions-f32.npy"
positions_npy_sha = sha_file(positions_path)
positions = np.load(positions_path, allow_pickle=False)
if positions.dtype != np.dtype("<f4") or positions.shape != (54949, 3):
    raise RuntimeError(f"unexpected candidate positions layout {positions.shape}/{positions.dtype}")
position_bytes = np.ascontiguousarray(positions, dtype="<f4").tobytes()
positions_data_sha = sha_bytes(position_bytes)
if positions_data_sha != report.get("candidate_source_positions_f32_sha256") or positions_npy_sha != report.get("candidate_source_positions_npy_sha256"):
    raise RuntimeError("candidate F32 position array differs from the accepted 927 report")
if report.get("status") != "inferred_engineering_clearance_candidate_pending_native_replay" or solve_result.get("status") != report.get("status"):
    raise RuntimeError("927 did not finish as the expected inferred pending-native candidate")
if report.get("final_nonocular_pair_count_by_pose") != [0] * 8 or report.get("final_ocular_pair_count_by_pose") != [0] * 8:
    raise RuntimeError("927 final exact accepted-pose pair gates are not all zero")
if not np.isfinite(positions).all():
    raise RuntimeError("candidate positions contain non-finite values")

decoded = decode_payload(base_raw)
if decoded["vertex_count"] != len(positions) or decoded["abi"] != 5 or decoded["binding_count"] != 86:
    raise RuntimeError("candidate dimensions do not match exact base ABI 5 payload")
vertex_offset = 60 + 36 * decoded["binding_count"]
vertex_end = vertex_offset + 56 * decoded["vertex_count"]
index_offset = vertex_end
weight_offset = index_offset + 4 * decoded["index_count"]
base_positions = decoded["vertices_f"][:, :3]
source_position_bits = base_positions.view("<u4")
candidate_position_bits = np.asarray(positions, dtype="<f4").view("<u4")
changed = np.any(source_position_bits != candidate_position_bits, axis=1)
referenced = np.zeros(decoded["vertex_count"], dtype=bool)
referenced[np.unique(decoded["indices"])] = True
if np.any(changed & ~referenced):
    raise RuntimeError("candidate changes unreferenced source geometry")
changed_ids = np.flatnonzero(changed).astype("<u4")

delta = positions.astype(np.float64) - base_positions.astype(np.float64)
displacement = np.linalg.norm(delta, axis=1)
if abs(float(displacement.max()) - float(report["maximum_source_displacement_m"])) > 1e-9:
    raise RuntimeError("candidate displacement does not reproduce report")

correction_raw = bytearray(base_raw)
for vertex in changed_ids:
    i = int(vertex)
    struct.pack_into("<3f", correction_raw, vertex_offset + i * 56, *map(float, positions[i]))
correction_raw = bytes(correction_raw)
registration_path = Path(base_manifest["inputs"]["registration_path"])
registration = json_file(registration_path)
composed_raw, composition = compose_disjoint_skin_position_corrections(
    base_raw, [correction_raw],
    global_source_matrix=registration["coordinate_system"]["global_source_mm_to_myosim_world_m"],
)
composed_decoded = decode_payload(composed_raw)
if not np.array_equal(composed_decoded["vertices_f"][:, :3].view("<u4"), candidate_position_bits):
    raise RuntimeError("owner composition changed candidate source positions")
if composed_raw[:vertex_offset] != base_raw[:vertex_offset] or composed_raw[vertex_end:] != base_raw[vertex_end:]:
    raise RuntimeError("owner composition changed header, bindings, indices, or full weights")
if not np.array_equal(composed_decoded["full_weights"].view("<u4"), decoded["full_weights"].view("<u4")):
    raise RuntimeError("owner composition changed ABI 5 full weights")
if not np.array_equal(composed_decoded["indices"], decoded["indices"]):
    raise RuntimeError("owner composition changed triangle topology/order")
if not np.array_equal(composed_decoded["bindings_u"], decoded["bindings_u"]):
    raise RuntimeError("owner composition changed binding record bits")
if not np.array_equal(composed_decoded["vertices_u"][:, 6:], decoded["vertices_u"][:, 6:]):
    raise RuntimeError("owner composition changed per-vertex influence records")
if not np.array_equal(composed_decoded["vertices_u"][~referenced, :6], decoded["vertices_u"][~referenced, :6]):
    raise RuntimeError("owner composition changed unreferenced xyz/normal rows")
if not np.isfinite(composed_decoded["vertices_f"][:, :6]).all():
    raise RuntimeError("owner-composed positions/normals contain non-finite values")

OUT.mkdir(parents=True)
asset_path = OUT / BASE_SKIN.name
asset_path.write_bytes(composed_raw)
changed_ids_path = OUT / "changed-referenced-source-vertex-ids-u32.npy"
np.save(changed_ids_path, changed_ids, allow_pickle=False)

# Preserve 003/907 ancestry as historical provenance while making the exact 907
# payload the immediate source required by the 924 receipt composer.
manifest = copy.deepcopy(base_manifest)
manifest_path = OUT / "common-atlas-skin-geometry-registration.manifest.json"
manifest["inputs"]["source_payload"] = {
    **record(BASE_SKIN),
    "route": "exact 907 common-atlas skin plus one 927 multi-pose inferred source-position clearance correction; all prior 003/018/906 provenance retained below",
}
manifest["inputs"]["immediate_predecessor_registration_manifest"] = record(BASE_SKIN_MANIFEST)
manifest["output_payload"] = record(asset_path)
manifest["output_manifest"] = str(manifest_path)
manifest["method"] = (
    "Applied the frozen 927 inferred source-position field to the exact 907 common-atlas NHSKIN payload; "
    "preserved the complete ABI 5 bindings, weights, indices, and unreferenced source rows; recomputed indexed "
    "rest-world normals with the existing skin geometry owner. This is pending native replay."
)
manifest["evidence_boundary"] = (
    "The selected 0.25 mm value is a directional target used by the offline clearance solver, not a certified minimum clearance or measured skin thickness. "
    "The candidate passed exact self and 859-target scans on eight owner-equivalent predicted poses (steps 0, 4991, 5375, 5759, 6111, 6495, 7743, 9999). "
    "Native replay of this recomposed payload, later breathing poses, and the 300 s study remain pending; geometry is inferred and not admitted as physical/collision truth."
)
manifest["geometry_registration"]["multipose_clearance_correction"] = {
    "schema": "numi.human.common-atlas-multipose-clearance-correction.v1",
    "status": "inferred_candidate_pending_native_replay",
    "source_payload_path": str(BASE_SKIN),
    "source_payload_sha256": base_sha,
    "solver_report": record(report_path),
    "solver_result": record(result_path),
    "solver_preflight": record(preflight_path),
    "candidate_positions_npy": record(positions_path),
    "candidate_positions_f32_sha256": positions_data_sha,
    "selected_margin_mm": report["selected_margin_mm"],
    "selected_margin_interpretation": "target value only; not certified achieved clearance",
    "accepted_pose_steps": report["accepted_pose_steps"],
    "exact_nonocular_pair_counts_by_pose": report["final_nonocular_pair_count_by_pose"],
    "exact_ocular_pair_counts_by_pose": report["final_ocular_pair_count_by_pose"],
    "exact_skin_self_intersection_gate": "passed on every accepted trial and all eight resumed owner-equivalent predicted poses; exact predicate _audit_pair(same_surface=True)",
    "orientation_and_shape_gates": {
        "all_source_faces_connected_and_oriented": report["source_orientation"]["all_source_faces_connected_and_oriented"],
        "minimum_face_normal_alignment": report["surface_shape_diagnostics"]["minimum_face_normal_dot_to_accepted_across_poses"],
        "minimum_triangle_area_ratio": report["surface_shape_diagnostics"]["minimum_face_area_ratio_across_poses"],
        "zero_area_triangles": 0,
    },
    "source_geometry_changes": {
        "referenced_vertex_count": int(referenced.sum()),
        "changed_referenced_vertex_count": int(changed.sum()),
        "changed_unreferenced_vertex_count": int(np.count_nonzero(changed & ~referenced)),
        "changed_vertex_ids_path": str(changed_ids_path),
        "changed_vertex_ids_sha256": sha_file(changed_ids_path),
        "maximum_source_displacement_m": float(displacement.max()),
        "displacement_p50_m": float(np.percentile(displacement[referenced], 50)),
        "displacement_p95_m": float(np.percentile(displacement[referenced], 95)),
        "displacement_p99_m": float(np.percentile(displacement[referenced], 99)),
        "maximum_world_displacement_m_by_pose": report["maximum_world_displacement_m_by_pose"],
        "baseline_and_candidate_bed_gaps_m_by_pose": report["candidate_minimum_bed_gap_m_by_pose"],
        "baseline_negative_bed_vertex_count_by_pose": report["baseline_negative_bed_vertex_count_by_pose"],
        "candidate_negative_bed_vertex_count_by_pose": report["candidate_negative_bed_vertex_count_by_pose"],
    },
    "preservation": {
        "source_payload_derived_from_exact_907": True,
        "all_86_binding_records_byte_identical": True,
        "all_86_binding_records_match_canonical_reference": True,
        "full_86_column_weight_matrix_byte_identical": True,
        "per_vertex_influence_records_byte_identical": True,
        "triangle_indices_and_order_byte_identical": True,
        "unreferenced_vertex_xyz_and_normal_bytes_byte_identical": True,
        "referenced_positions_are_exact_927_f32_candidate": True,
        "referenced_rest_world_normals_recomputed_by_existing_skin_owner": True,
        "NHTISS_tendon_muscle_or_physics_payloads_touched": False,
        "skin_contact_or_physics_owner_changed": False,
    },
    "composition_owner": composition,
    "history": {
        "registration_003": record(E / "common-atlas-skin-registration-003/common-atlas-skin-geometry-registration.manifest.json"),
        "registration_907": record(BASE_SKIN_MANIFEST),
        "base_receipt_924": record(BASE_NHA_RECEIPT),
        "base_manifest_924": record(BASE_NHA_MANIFEST),
        "runs": [ancestry_run(i) for i in (925, 926, 927)],
        "previously_failed_or_incomplete_runs_retained": [925, 926],
        "source_original_positions_f32_sha256": sha_bytes(np.ascontiguousarray(base_positions, dtype="<f4").tobytes()),
        "solver_source": {"worktree": str(ROOT), "revision": "a2eeb789e1b2f147b045e5383df1780e6ef7f3ac", "module": record(SOLVER)},
        "driver": record(DRIVER),
    },
}
manifest["qualification"]["native_pose_clearance"] = "pending native GPU replay of this recomposed payload and later accepted poses"
manifest["qualification"]["registered_skeleton_and_organ_clearance"] = "eight owner-equivalent offline predicted poses passed exact 859-target scans; native replay and longer-duration coverage pending"
manifest["qualification"]["skin_self_intersection"] = "exact self-pair gate passed during all accepted candidate trials on eight offline predicted poses; native replay pending"
manifest["qualification"]["physical_or_collision_use"] = "not admitted"
manifest["code"]["clearance_solver"] = {"path": str(SOLVER), "sha256": sha_file(SOLVER), "revision": "a2eeb789e1b2f147b045e5383df1780e6ef7f3ac"}
manifest["code"]["clearance_driver"] = record(DRIVER)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

# Re-read on-disk identities before using the existing receipt owner.
if sha_file(asset_path) != manifest["output_payload"]["sha256"]:
    raise RuntimeError("saved candidate payload hash mismatch")
if manifest.get("inputs", {}).get("source_payload", {}).get("sha256") != base_sha:
    raise RuntimeError("candidate manifest immediate source identity mismatch")
receipt_result = resting_anatomy.compose_skin_binding_candidate(
    BASE_NHA_RECEIPT, asset_path, manifest_path, RECEIPT_OUT)

summary = {
    "status": "candidate_payload_manifest_and_receipt_prepared_pending_native_replay",
    "candidate_payload": record(asset_path),
    "candidate_registration_manifest": record(manifest_path),
    "candidate_receipt_composition": receipt_result,
    "source_payload_907": record(BASE_SKIN),
    "source_manifest_907": record(BASE_SKIN_MANIFEST),
    "base_receipt_924": record(BASE_NHA_RECEIPT),
    "base_manifest_924": record(BASE_NHA_MANIFEST),
    "candidate_report_927": record(report_path),
    "candidate_run_927_result": record(result_path),
    "candidate_run_927_preflight": record(preflight_path),
    "candidate_positions_array_sha256": positions_npy_sha,
    "candidate_positions_f32_sha256": positions_data_sha,
    "changed_referenced_vertices": int(changed.sum()),
    "changed_unreferenced_vertices": int(np.count_nonzero(changed & ~referenced)),
    "maximum_source_displacement_m": float(displacement.max()),
    "final_exact_nonocular_pairs_by_pose": report["final_nonocular_pair_count_by_pose"],
    "final_exact_ocular_pairs_by_pose": report["final_ocular_pair_count_by_pose"],
    "selected_margin_mm_is_target_not_certified_minimum": report["selected_margin_mm"],
    "failed_incomplete_history_preserved": [925, 926],
}
OUT.mkdir(exist_ok=True)
summary_path = OUT / "candidate-asset-composition-report.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
