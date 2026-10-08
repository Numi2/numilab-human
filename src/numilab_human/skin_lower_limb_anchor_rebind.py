"""Derive an ABI-5 skin candidate from exact registered lower-limb anchors.

The candidate preserves source positions, topology, the complete 86-body
weight matrix, and 74 unchanged owner-transform records. It rebinds only the
twelve lower-limb owners and regenerates indexed rest-world normals for the
resulting reference-pose geometry. It is a visual geometry candidate only.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from . import model as human
from .skin_source_payload_preflight import decode_payload

SCHEMA = "numi.human.skin-lower-limb-anchor-rebind-candidate.v1"
TARGET_BODY_NAMES = ("femur_r", "femur_l", "tibia_r", "tibia_l", "patella_r", "patella_l", "talus_r", "talus_l", "calcn_r", "calcn_l", "toes_r", "toes_l")
REGISTRATION_SCHEMA = "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
LOWER_LIMB_SCHEMA = "numi.human.bodyparts3d-myosim-lower-limb-source-mesh-registration.v3"


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _matrix_digest(matrix: list[list[float]]) -> str:
    return hashlib.sha256(json.dumps(matrix, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    p50, p95, p99 = np.percentile(values, [50.0, 95.0, 99.0])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "maximum": float(values.max())}


def _binding_override_rows(registration, binding_body_ids, runtime_bodies):
    lower = registration.get("lower_limb_source_mesh_registration")
    if not isinstance(lower, dict) or lower.get("schema") != LOWER_LIMB_SCHEMA or lower.get("status") != "candidate_passed_bilateral_source_mesh_and_default_pose_continuity_gates":
        raise human.ImportError("skin anchor rebind requires the admitted lower-limb source registration receipt")
    anchors = registration.get("anchors")
    if not isinstance(anchors, list):
        raise human.ImportError("skin anchor rebind registration has no anchors")
    grouped = {name: [] for name in TARGET_BODY_NAMES}
    for anchor in anchors:
        if not isinstance(anchor, dict):
            raise human.ImportError("skin anchor rebind registration has a malformed anchor")
        target = anchor.get("target")
        if isinstance(target, dict) and target.get("name") in grouped:
            grouped[target["name"]].append(anchor)
    binding_rows = {}
    changes = []
    for name in TARGET_BODY_NAMES:
        body_anchors = grouped[name]
        if not body_anchors:
            raise human.ImportError(f"skin anchor rebind has no exact registration anchor for {name}")
        first = body_anchors[0]
        target = first.get("target")
        if not isinstance(target, dict):
            raise human.ImportError(f"skin anchor rebind {name} has no target identity")
        body_index, source_body_id = target.get("core_body_index"), target.get("source_body_id")
        if type(body_index) is not int or body_index < 0 or type(source_body_id) is not int:
            raise human.ImportError(f"skin anchor rebind {name} has an invalid source/Core owner")
        if name not in runtime_bodies:
            raise human.ImportError(f"skin anchor rebind {name} is absent from the matched runtime reference")
        runtime_index, runtime_body = runtime_bodies[name]
        if runtime_index != body_index or runtime_body.get("source_body_id") != source_body_id:
            raise human.ImportError(f"skin anchor rebind {name} registration/runtime owner identity differs")
        for anchor in body_anchors:
            if anchor.get("target") != target:
                raise human.ImportError(f"skin anchor rebind {name} has mixed target rest poses")
            matrix = anchor.get("registration", {}).get("source_obj_mm_to_core_inertial_body_m")
            first_matrix = first.get("registration", {}).get("source_obj_mm_to_core_inertial_body_m")
            if matrix != first_matrix:
                raise human.ImportError(f"skin anchor rebind {name} source anchors do not share one exact Core-local transform")
        matrix = first.get("registration", {}).get("source_obj_mm_to_core_inertial_body_m")
        translation, quaternion, scale = human._bodyparts_visual_local_pose(matrix, f"skin anchor rebind {name} registered source-to-Core transform")
        matches = np.flatnonzero(binding_body_ids == body_index)
        if matches.size != 1:
            raise human.ImportError(f"skin anchor rebind {name} Core body {body_index} has {matches.size} NHSKIN bindings")
        binding_index = int(matches[0])
        if binding_index in binding_rows:
            raise human.ImportError("skin anchor rebind target body names alias one NHSKIN binding")
        binding_rows[binding_index] = (name, translation, quaternion, scale, body_anchors)
        changes.append({"myosim_body": name, "source_body_id": source_body_id, "core_body_index": body_index, "binding_index": binding_index, "source_anchor_member_ids": [str(anchor["source"]["member_id"]) for anchor in body_anchors], "registered_source_obj_mm_to_core_inertial_body_m": matrix, "registered_transform_sha256": _matrix_digest(matrix)})
    return binding_rows, changes


def _owner_names_by_core_index(registration):
    owners = {}
    anchors = registration.get("anchors")
    if not isinstance(anchors, list):
        raise human.ImportError("skin anchor rebind registration has no anchors")
    for anchor in anchors:
        target = anchor.get("target") if isinstance(anchor, dict) else None
        if not isinstance(target, dict):
            raise human.ImportError("skin anchor rebind registration has an invalid target")
        name, body_index = target.get("name"), target.get("core_body_index")
        if not isinstance(name, str) or not name or type(body_index) is not int or body_index < 0:
            raise human.ImportError("skin anchor rebind registration has an invalid owner identity")
        previous = owners.setdefault(body_index, name)
        if previous != name:
            raise human.ImportError("skin anchor rebind registration aliases one Core owner to multiple names")
    return owners


def _region_for_owner(name):
    if name == "head":
        return "head"
    if name in TARGET_BODY_NAMES:
        return "legs"
    if name in {"sacrum", "torso", "pelvis", "cervical_spine"} or name.startswith("lumbar"):
        return "torso"
    return "arms_hands"


def _transform_points(points_m, binding, pose, context):
    local_rotation = np.asarray(human._myosim_matrix_from_quaternion_xyzw(binding[4:8].astype(float).tolist()), dtype=np.float64)
    body_position = np.asarray(pose["position_world_m"], dtype=np.float64)
    body_rotation = np.asarray(pose["rotation_world"], dtype=np.float64)
    local_points = binding[1:4].astype(np.float64) + (float(binding[8]) * (points_m @ local_rotation.T))
    world = body_position + local_points @ body_rotation.T
    if not np.isfinite(world).all():
        raise human.ImportError(f"skin anchor rebind {context} produced non-finite rest positions")
    return world


def _derive_candidate_bytes(raw, registration, registration_sha256, runtime_reference, runtime_bodies):
    decoded = decode_payload(raw)
    binding_ids = decoded["bindings_u"][:, 0].astype(np.uint32, copy=False)
    if len(set(int(value) for value in binding_ids)) != len(binding_ids):
        raise human.ImportError("skin anchor rebind input has duplicate Core-body bindings")
    weights = decoded["full_weights"]
    if not np.isfinite(weights).all() or float(weights.min(initial=0.0)) < 0.0:
        raise human.ImportError("skin anchor rebind input has non-finite or negative full weights")
    if float(np.max(np.abs(weights.sum(axis=1) - 1.0))) > 2.0e-3:
        raise human.ImportError("skin anchor rebind input full weights do not partition unity")
    bodyparts = registration.get("source", {}).get("bodyparts")
    myosim = registration.get("source", {}).get("myosim")
    source_sha = myosim.get("source", {}).get("archive_sha256") if isinstance(myosim, dict) else None
    if not isinstance(bodyparts, dict) or not isinstance(source_sha, str):
        raise human.ImportError("skin anchor rebind registration lacks source identity")
    if decoded["source_archive_sha256"] != source_sha:
        raise human.ImportError("skin anchor rebind source archive differs from NHSKIN ABI 5")
    expected_fingerprint = int(registration_sha256[:8], 16)
    if decoded["registration_fingerprint32"] != expected_fingerprint:
        raise human.ImportError("skin anchor rebind full registration SHA does not match NHSKIN registration fingerprint")
    owner_names = _owner_names_by_core_index(registration)
    missing_target_names = set(TARGET_BODY_NAMES) - set(owner_names.values())
    if missing_target_names:
        raise human.ImportError(f"skin anchor rebind has no exact registration anchor for {sorted(missing_target_names)[0]}")
    if not set(int(value) for value in binding_ids).issubset(owner_names):
        raise human.ImportError("skin anchor rebind Core-owner table differs from registration anchors")
    registered_source_ids = {}
    for anchor in registration["anchors"]:
        target = anchor["target"]
        registered_source_ids.setdefault(int(target["core_body_index"]), set()).add(int(target["source_body_id"]))
    for body_index, name in owner_names.items():
        runtime_entry = runtime_bodies.get(name)
        if runtime_entry is None:
            raise human.ImportError(f"skin anchor rebind {name} is absent from the matched runtime reference")
        runtime_index, runtime_body = runtime_entry
        source_ids = registered_source_ids.get(body_index, set())
        if runtime_index != body_index or len(source_ids) != 1 or runtime_body.get("source_body_id") not in source_ids:
            raise human.ImportError("skin anchor rebind runtime owner table differs from registration anchors")
    binding_rows, changes = _binding_override_rows(registration, binding_ids, runtime_bodies)
    if set(int(value) for value in binding_ids) != set(owner_names):
        raise human.ImportError("skin anchor rebind Core-owner table differs from registration anchors")
    if len(binding_rows) != len(TARGET_BODY_NAMES):
        raise human.ImportError("skin anchor rebind did not resolve all twelve lower-limb owners")
    output = bytearray(raw)
    header_size, binding_size, vertex_size = 60, 36, 56
    for binding_index, (_, translation, quaternion, scale, _) in binding_rows.items():
        body_index = int(binding_ids[binding_index])
        output[header_size + binding_index * binding_size:header_size + (binding_index + 1) * binding_size] = struct.pack("<I8f", body_index, *translation, *quaternion, scale)
    candidate_bindings = np.frombuffer(output, dtype="<f4", count=9 * len(binding_ids), offset=header_size).reshape(len(binding_ids), 9)
    points_m = decoded["vertices_f"][:, :3].astype(np.float64)
    faces = decoded["indices"].reshape(-1, 3).astype(np.int64, copy=False)
    referenced_indices = np.unique(faces)
    referenced_mask = np.zeros(decoded["vertex_count"], dtype=bool)
    referenced_mask[referenced_indices] = True
    old_rest = np.zeros_like(points_m)
    candidate_rest = np.zeros_like(points_m)
    per_body = []
    for binding_index, body_index_value in enumerate(binding_ids):
        body_index = int(body_index_value)
        name = owner_names[body_index]
        _, runtime_body = runtime_bodies[name]
        old_world = _transform_points(points_m, decoded["bindings_f"][binding_index], runtime_body, f"old owner {name}")
        new_world = _transform_points(points_m, candidate_bindings[binding_index], runtime_body, f"candidate owner {name}")
        weight = weights[:, binding_index].astype(np.float64, copy=False)
        old_rest += old_world * weight[:, None]
        candidate_rest += new_world * weight[:, None]
        if binding_index in binding_rows:
            weighted_delta = (new_world - old_world) * weight[:, None]
            displacement = np.linalg.norm(weighted_delta, axis=1)
            binding_weight = weights[:, binding_index]
            measurable = (binding_weight >= 1.0e-6) & referenced_mask
            source_bounds = None
            if bool(measurable.any()):
                selected = points_m[measurable]
                source_bounds = [selected.min(axis=0).tolist(), selected.max(axis=0).tolist()]
            anchors = binding_rows[binding_index][4]
            per_body.append({"myosim_body": name, "core_body_index": body_index, "source_body_id": runtime_body["source_body_id"], "source_anchor_member_ids": [str(anchor["source"]["member_id"]) for anchor in anchors], "input_binding": [float(value) for value in decoded["bindings_f"][binding_index]], "candidate_binding": [float(value) for value in candidate_bindings[binding_index]], "registered_transform_sha256": _matrix_digest(anchors[0]["registration"]["source_obj_mm_to_core_inertial_body_m"]), "skin_vertex_weight_coverage": {"strictly_positive_rendered_vertex_count": int(np.count_nonzero((binding_weight > 0.0) & referenced_mask)), "weight_at_least_1e-6_rendered_vertex_count": int(np.count_nonzero(measurable)), "weight_at_least_0.01_rendered_vertex_count": int(np.count_nonzero((binding_weight >= 0.01) & referenced_mask)), "all_payload_vertex_count_with_positive_weight": int(np.count_nonzero(binding_weight > 0.0)), "total_weight_mass": float(binding_weight.sum()), "maximum_vertex_weight": float(binding_weight.max(initial=0.0)), "source_skin_bounds_m_where_weight_at_least_1e-6": source_bounds}, "rendered_mesh_per_owner_weighted_rest_displacement_mm": _percentiles(displacement[referenced_mask] * 1000.0), "rendered_mesh_per_owner_changed_counts": {"over_1_micrometre": int(np.count_nonzero(displacement[referenced_mask] > 1.0e-6)), "over_0.1_mm": int(np.count_nonzero(displacement[referenced_mask] > 1.0e-4)), "over_1_mm": int(np.count_nonzero(displacement[referenced_mask] > 1.0e-3))}})
    if not np.isfinite(old_rest).all() or not np.isfinite(candidate_rest).all():
        raise human.ImportError("skin anchor rebind rest-pose reconstruction is non-finite")
    global_matrix = registration.get("coordinate_system", {}).get("global_source_mm_to_myosim_world_m")
    global_translation, global_quaternion, global_scale = human._bodyparts_visual_local_pose(global_matrix, "skin anchor rebind source global rest frame")
    global_rotation = np.asarray(human._myosim_matrix_from_quaternion_xyzw(global_quaternion), dtype=np.float64)
    source_world = np.asarray(global_translation, dtype=np.float64) + float(global_scale) * (points_m @ global_rotation.T)
    baseline_residual = np.linalg.norm(old_rest - source_world, axis=1)
    net_delta = candidate_rest - old_rest
    net_magnitude = np.linalg.norm(net_delta, axis=1)
    referenced_remap = np.full(decoded["vertex_count"], -1, dtype=np.int64)
    referenced_remap[referenced_indices] = np.arange(len(referenced_indices), dtype=np.int64)
    compact_faces = referenced_remap[faces]
    compact_face_rows = [tuple(int(value) for value in face) for face in compact_faces]
    candidate_normals = human._bodyparts_vertex_normals((candidate_rest[referenced_indices] * 1000.0).tolist(), compact_face_rows, "candidate lower-limb-rebound skin rest surface")
    candidate_normals = human._bodyparts_skin_smooth_visual_normals(candidate_normals, compact_face_rows)
    candidate_normals_array = decoded["vertices_f"][:, 3:6].astype(np.float64).copy()
    candidate_normals_array[referenced_indices] = np.asarray(candidate_normals, dtype=np.float64)
    if candidate_normals_array.shape != (decoded["vertex_count"], 3) or not np.isfinite(candidate_normals_array).all():
        raise human.ImportError("skin anchor rebind produced an invalid ABI 5 rest-world normal field")
    normal_lengths = np.linalg.norm(candidate_normals_array, axis=1)
    if bool((normal_lengths[referenced_mask] <= 1.0e-12).any()):
        raise human.ImportError("skin anchor rebind produced a degenerate indexed rest-world normal")
    candidate_normals_array[referenced_indices] /= normal_lengths[referenced_indices, None]
    old_normals = decoded["vertices_f"][:, 3:6].astype(np.float64)
    old_lengths = np.linalg.norm(old_normals, axis=1)
    if bool((old_lengths[referenced_mask] <= 1.0e-12).any()):
        raise human.ImportError("skin anchor rebind input has a degenerate indexed rest-world normal")
    old_normals[referenced_indices] /= old_lengths[referenced_indices, None]
    normal_angle_deg = np.zeros(decoded["vertex_count"], dtype=np.float64)
    normal_angle_deg[referenced_indices] = np.degrees(np.arccos(np.clip(np.sum(old_normals[referenced_indices] * candidate_normals_array[referenced_indices], axis=1), -1.0, 1.0)))
    vertex_offset = header_size + binding_size * len(binding_ids)
    for vertex_index in referenced_indices:
        normal = candidate_normals_array[int(vertex_index)]
        struct.pack_into("<3f", output, vertex_offset + int(vertex_index) * vertex_size + 12, float(normal[0]), float(normal[1]), float(normal[2]))
    candidate = bytes(output)
    candidate_decoded = decode_payload(candidate)
    if not np.array_equal(decoded["vertices_u"][:, :3], candidate_decoded["vertices_u"][:, :3]):
        raise human.ImportError("skin anchor rebind unexpectedly changed NHSKIN positions")
    if not np.array_equal(decoded["vertices_u"][:, 6:14], candidate_decoded["vertices_u"][:, 6:14]):
        raise human.ImportError("skin anchor rebind unexpectedly changed NHSKIN per-vertex influence records")
    if not np.array_equal(decoded["indices"], candidate_decoded["indices"]):
        raise human.ImportError("skin anchor rebind unexpectedly changed NHSKIN triangle indices")
    if not np.array_equal(decoded["full_weights"], candidate_decoded["full_weights"]):
        raise human.ImportError("skin anchor rebind unexpectedly changed NHSKIN full weights")
    if candidate_decoded["registration_fingerprint32"] != decoded["registration_fingerprint32"] or candidate_decoded["source_archive_sha256"] != decoded["source_archive_sha256"]:
        raise human.ImportError("skin anchor rebind changed NHSKIN source identity")
    changed_bindings = {i for i in range(len(binding_ids)) if not np.array_equal(decoded["bindings_u"][i], candidate_decoded["bindings_u"][i])}
    if changed_bindings != set(binding_rows):
        raise human.ImportError("skin anchor rebind changed an unexpected binding owner")
    if np.array_equal(decoded["vertices_u"][:, 3:6], candidate_decoded["vertices_u"][:, 3:6]):
        raise human.ImportError("skin anchor rebind did not regenerate ABI 5 rest-world normals")
    if not np.isfinite(candidate_decoded["vertices_f"][:, 3:6]).all():
        raise human.ImportError("skin anchor rebind emitted non-finite normals")
    emitted_lengths = np.linalg.norm(candidate_decoded["vertices_f"][:, 3:6], axis=1)
    if float(np.max(np.abs(emitted_lengths[referenced_mask] - 1.0))) > 2.0e-6:
        raise human.ImportError("skin anchor rebind emitted non-unit indexed ABI 5 rest-world normals")
    if not np.array_equal(decoded["vertices_u"][~referenced_mask, 3:6], candidate_decoded["vertices_u"][~referenced_mask, 3:6]):
        raise human.ImportError("skin anchor rebind changed unreferenced ABI 5 input normal bytes")
    owner_names_for_binding = [owner_names[int(value)] for value in binding_ids]
    dominant = np.argmax(weights, axis=1)
    dominant_region = np.asarray([_region_for_owner(owner_names_for_binding[int(i)]) for i in dominant])
    regions = {}
    leg_columns = [i for i, name in enumerate(owner_names_for_binding) if name in TARGET_BODY_NAMES]
    for region in ("torso", "head", "arms_hands", "legs"):
        selected_region = (dominant_region == region) & referenced_mask
        magnitudes = net_magnitude[selected_region]
        leg_influence = weights[selected_region][:, leg_columns].sum(axis=1) if leg_columns else np.zeros(int(selected_region.sum()))
        regions[region] = {"vertices_dominantly_influenced_by_region": int(selected_region.sum()), "vertices_with_any_lower_limb_owner_weight": int(np.count_nonzero(leg_influence > 0.0)), "vertices_with_lower_limb_owner_weight_at_least_1e-6": int(np.count_nonzero(leg_influence >= 1.0e-6)), "sum_of_lower_limb_owner_weight_for_region_vertices": float(leg_influence.sum()), "net_default_rest_displacement_mm": _percentiles(magnitudes * 1000.0), "moved_over_1_micrometre": int(np.count_nonzero(magnitudes > 1.0e-6)), "moved_over_0.1_mm": int(np.count_nonzero(magnitudes > 1.0e-4)), "moved_over_1_mm": int(np.count_nonzero(magnitudes > 1.0e-3))}
    moved = net_magnitude > 1.0e-4
    manifest = {"schema": SCHEMA, "status": "source_identity_preserved_visual_skin_candidate_pending_native_geometry_audit", "method": "byte-preserving ABI 5 NHSKIN rebind of only the twelve mismatched lower-limb Core owner binding rows to their exact registered source-anchor transforms; recompute ABI 5 indexed common rest-world normals on the resulting full-weight default reference pose", "inputs": {"registration_sha256": registration_sha256, "registration_fingerprint32": f"{decoded['registration_fingerprint32']:08x}", "bodyparts_source": bodyparts, "myosim_source_archive_sha256": source_sha, "runtime_reference": runtime_reference}, "payload_identity": {"magic": "NHSKIN1", "abi": 5, "source_archive_sha256": decoded["source_archive_sha256"], "registration_fingerprint32": f"{decoded['registration_fingerprint32']:08x}", "binding_count": int(decoded["binding_count"]), "vertex_count": int(decoded["vertex_count"]), "index_count": int(decoded["index_count"]), "payload_bytes": len(raw)}, "binding_changes": changes, "changed_skin_regions": {"evidence_boundary": "All 86 source weights are retained. Dense nonzero lower-limb weight tails may move vertices throughout the shell; the dominant-owner region breakdown and full displacement distributions quantify those effects. Reconstruction uses stored float32 inputs and matched MyoSim default reference poses in double precision, not a native accepted-state capture.", "bindings": per_body, "combined_geometry": {"payload_vertex_count": int(points_m.shape[0]), "rendered_referenced_vertex_count": int(len(referenced_indices)), "unreferenced_payload_vertex_count": int(np.count_nonzero(~referenced_mask)), "all_payload_vertices_over_1_micrometre_count": int(np.count_nonzero(net_magnitude > 1.0e-6)), "all_payload_vertices_over_0.1_mm_count": int(np.count_nonzero(net_magnitude > 1.0e-4)), "all_payload_vertices_over_1_mm_count": int(np.count_nonzero(net_magnitude > 1.0e-3)), "all_payload_net_default_rest_displacement_mm": _percentiles(net_magnitude * 1000.0), "rendered_net_default_rest_displacement_mm": _percentiles(net_magnitude[referenced_mask] * 1000.0), "rendered_vertices_over_1_micrometre_count": int(np.count_nonzero(net_magnitude[referenced_mask] > 1.0e-6)), "rendered_vertices_over_0.1_mm_count": int(np.count_nonzero(net_magnitude[referenced_mask] > 1.0e-4)), "rendered_vertices_over_1_mm_count": int(np.count_nonzero(net_magnitude[referenced_mask] > 1.0e-3)), "old_binding_vs_registered_source_rest_error_mm": _percentiles(baseline_residual * 1000.0), "maximum_displacement_vertex_index": int(np.argmax(net_magnitude)), "maximum_displacement_source_position_m": points_m[int(np.argmax(net_magnitude))].tolist(), "bounds_of_vertices_moved_over_0.1_mm_in_source_coordinates": [points_m[moved].min(axis=0).tolist(), points_m[moved].max(axis=0).tolist()] if bool(moved.any()) else None, "dominant_anatomical_region_breakdown": regions, "normal_recomputation": {"method": "existing NHSKIN geometric-normal and three-pass visual-smoothing helpers applied to indexed vertices of the exact default-reference-pose full-weight candidate shell", "unreferenced_payload_vertices_preserve_input_normals": int(np.count_nonzero(~referenced_mask)), "vertices_with_normal_change_over_0.1_degrees": int(np.count_nonzero(normal_angle_deg > 0.1)), "vertices_with_normal_change_over_1_degree": int(np.count_nonzero(normal_angle_deg > 1.0)), "change_angle_degrees": _percentiles(normal_angle_deg), "indexed_normal_length_max_error_after_float32_pack": float(np.max(np.abs(emitted_lengths[referenced_mask] - 1.0))), "unreferenced_input_normal_bytes_preserved": True}}}, "preservation": {"source_position_coordinates_byte_identical": True, "triangle_indices_and_order_byte_identical": True, "full_86_body_weight_matrix_byte_identical": True, "all_non_target_binding_records_byte_identical": True, "non_target_binding_record_count": int(decoded["binding_count"] - len(binding_rows)), "nhtiss_tendon_muscle_and_physics_payloads_touched": False, "skin_topology_changed": False, "skin_physics_or_contact_owner_changed": False, "rest_world_normals_recomputed_from_candidate_default_pose": True}, "evidence_boundary": "This candidate changes twelve lower-limb ABI 5 visual skin binding records and regenerates the indexed ABI 5 rest-world normal field. It does not establish native pose coverage, self-intersection freedom, material or contact behavior, or whole-body anatomical qualification."}
    return candidate, manifest


def _verify_owner_mismatch_report(*, report_path, source_payload_path, registration_path, source_payload_sha256, registration_sha256, registration):
    report_path = Path(report_path).resolve()
    report = human.read_json(report_path)
    inputs = report.get("inputs", {})
    skin = inputs.get("skin_payload", {})
    bone = inputs.get("bone_payload", {})
    bone_manifest = inputs.get("bone_manifest", {})
    if skin.get("path") != str(Path(source_payload_path).resolve()) or skin.get("sha256") != source_payload_sha256:
        raise human.ImportError("skin anchor rebind owner comparison skin input identity differs")
    if not Path(bone.get("path", "")).is_file() or human.sha256(Path(bone["path"])) != bone.get("sha256"):
        raise human.ImportError("skin anchor rebind owner comparison bone payload identity differs")
    if not Path(bone_manifest.get("path", "")).is_file() or human.sha256(Path(bone_manifest["path"])) != bone_manifest.get("sha256"):
        raise human.ImportError("skin anchor rebind owner comparison bone manifest identity differs")
    if bone_manifest.get("full_registration_file_sha256") != registration_sha256 or human.sha256(registration_path) != registration_sha256:
        raise human.ImportError("skin anchor rebind owner comparison registration identity differs")
    coverage = report.get("coverage", {})
    if (coverage.get("skin_owner_count") != 86 or coverage.get("bone_owner_count") != 86 or coverage.get("owners_with_skin_bone_transform_mismatch") != len(TARGET_BODY_NAMES) or coverage.get("owners_with_matching_transforms") != 74 or coverage.get("unmatched_skin_core_owners") != [] or coverage.get("unmatched_bone_core_owners") != [] or coverage.get("owners_with_multiple_distinct_bone_transform_tuples") != 0):
        raise human.ImportError("skin anchor rebind owner comparison does not establish the 12/74 owner partition")
    expected_owner_ids = {target["name"]: int(target["core_body_index"]) for anchor in registration.get("anchors", []) if isinstance((target := anchor.get("target")), dict)}
    mismatch_rows = report.get("mismatched_owners")
    if not isinstance(mismatch_rows, list) or len(mismatch_rows) != len(TARGET_BODY_NAMES):
        raise human.ImportError("skin anchor rebind owner comparison mismatch rows are incomplete")
    actual = {row.get("body"): row.get("core_body_index") for row in mismatch_rows if isinstance(row, dict)}
    expected = {name: expected_owner_ids.get(name) for name in TARGET_BODY_NAMES}
    if actual != expected or any(value is None for value in expected.values()):
        raise human.ImportError("skin anchor rebind owner comparison differs from registered lower-limb owners")
    return {"path": str(report_path), "sha256": human.sha256(report_path), "skin_payload_sha256": source_payload_sha256, "bone_payload_path": str(Path(bone["path"]).resolve()), "bone_payload_sha256": bone["sha256"], "bone_manifest_sha256": bone_manifest["sha256"], "registration_sha256": registration_sha256, "mismatched_owner_names_and_core_indices": expected, "matching_owner_count": coverage["owners_with_matching_transforms"], "mismatched_owner_count": coverage["owners_with_skin_bone_transform_mismatch"]}


def _verify_nhtiss_owner_alignment(*, tissue_payload_path, tissue_manifest_path, registration_sha256, source_archive_sha256, registration_fingerprint32, candidate_skin, registration):
    tissue_payload_path = Path(tissue_payload_path).resolve()
    tissue_manifest_path = Path(tissue_manifest_path).resolve()
    if not tissue_payload_path.is_file() or not tissue_manifest_path.is_file():
        raise human.ImportError("skin anchor rebind requires the registered NHTISS4 payload and manifest")
    tissue_raw = tissue_payload_path.read_bytes()
    tissue_doc = human.read_json(tissue_manifest_path)
    if tissue_doc.get("schema") != "numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1":
        raise human.ImportError("skin anchor rebind NHTISS4 manifest schema differs")
    if tissue_doc.get("source", {}).get("registration", {}).get("sha256") != registration_sha256:
        raise human.ImportError("skin anchor rebind NHTISS4 registration identity differs")
    payload = tissue_doc.get("payload", {})
    if payload.get("file") != tissue_payload_path.name or payload.get("sha256") != _sha_bytes(tissue_raw):
        raise human.ImportError("skin anchor rebind NHTISS4 payload differs from its manifest")
    if payload.get("registration_fingerprint32") != f"{registration_fingerprint32:08x}":
        raise human.ImportError("skin anchor rebind NHTISS4 fingerprint differs")
    if tissue_doc.get("source", {}).get("myosim_source_archive_sha256") != source_archive_sha256:
        raise human.ImportError("skin anchor rebind NHTISS4 source archive differs")
    if len(tissue_raw) < 64:
        raise human.ImportError("skin anchor rebind NHTISS4 payload is truncated")
    magic, abi, surface_count, binding_count, vertex_count, index_count, fingerprint, source_digest = struct.unpack_from("<8s6I32s", tissue_raw)
    expected_bytes = 64 + surface_count * 32 + binding_count * 36 + vertex_count * 56 + index_count * 4
    if magic != b"NHTISS4\0" or abi != 5 or len(tissue_raw) != expected_bytes or fingerprint != registration_fingerprint32 or source_digest.hex() != source_archive_sha256:
        raise human.ImportError("skin anchor rebind NHTISS4 payload header differs")
    for key, value in (("surface_count", surface_count), ("binding_count", binding_count), ("vertex_count", vertex_count), ("index_count", index_count)):
        if payload.get(key) != value:
            raise human.ImportError("skin anchor rebind NHTISS4 dimensions differ from its manifest")
    skin = decode_payload(candidate_skin)
    owner_names = _owner_names_by_core_index(registration)
    skin_records = {}
    for index, body_id in enumerate(skin["bindings_u"][:, 0]):
        skin_records[int(body_id)] = candidate_skin[60 + index * 36:60 + (index + 1) * 36]
    binding_offset = 64 + surface_count * 32
    owner_checks = []
    for name in TARGET_BODY_NAMES:
        body_index = next((index for index, owner_name in owner_names.items() if owner_name == name), None)
        if body_index is None:
            raise human.ImportError(f"skin anchor rebind NHTISS check cannot resolve {name}")
        records = [tissue_raw[binding_offset + i * 36:binding_offset + (i + 1) * 36] for i in range(binding_count) if struct.unpack_from("<I", tissue_raw, binding_offset + i * 36)[0] == body_index]
        expected = skin_records.get(body_index)
        if expected is None:
            raise human.ImportError(f"skin anchor rebind candidate has no NHSKIN row for {name}")
        if records and any(record != expected for record in records):
            raise human.ImportError(f"skin anchor rebind {name} differs byte-for-byte from registered NHTISS4")
        owner_checks.append({"myosim_body": name, "core_body_index": body_index, "nhtiss4_binding_record_count": len(records), "present_lower_limb_owner_records_byte_exact": all(record == expected for record in records), "nhtiss4_has_no_route_surface_for_owner": len(records) == 0})
    return {"nhtiss_payload_path": str(tissue_payload_path), "nhtiss_payload_sha256": _sha_bytes(tissue_raw), "nhtiss_manifest_path": str(tissue_manifest_path), "nhtiss_manifest_sha256": human.sha256(tissue_manifest_path), "present_lower_limb_owner_records_byte_exact": all(row["present_lower_limb_owner_records_byte_exact"] for row in owner_checks if row["nhtiss4_binding_record_count"] > 0), "owner_checks": owner_checks, "nhtiss_unmatched_binding_body_indices_recorded": [7, 25]}


def rebind_registered_lower_limb_skin_payload(*, source_payload, registration_path, myosim_artifact, output_directory, input_provenance_path=None, tissue_payload_path, tissue_manifest_path, owner_comparison_report_path):
    """Deprecated: individual lower-limb owner replacement violates NHSKIN's common atlas.

    Keep the private byte derivation and historical evidence readers available
    for reproducibility. New production candidates must preserve all canonical
    binding records and derive source-registered geometry in the shared atlas.
    """
    raise human.ImportError(
        "individual lower-limb NHSKIN binding replacement is deprecated and "
        "cannot be production-qualified: NHSKIN is authored in one shared "
        "atlas frame. Use common-atlas source-geometry registration while "
        "preserving all 86 canonical binding records."
    )
