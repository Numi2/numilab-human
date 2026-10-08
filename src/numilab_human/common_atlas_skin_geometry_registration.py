from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import model as human
from .skin_source_payload_preflight import decode_payload
from . import common_atlas_skin_clearance as clearance_module
from .common_atlas_skin_clearance import derive_step0_inferred_clearance


SCHEMA = "numi.human.common-atlas-skin-geometry-registration.v1"
STATUS = "inferred_common_atlas_geometry_candidate_pending_native_clearance_and_pose_checks"


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "maximum": 0.0}
    p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
    return {"p50": float(p50), "p90": float(p90), "p95": float(p95), "p99": float(p99), "maximum": float(values.max())}


def _transform(matrix: Any, context: str):
    translation, quaternion, scale = human._bodyparts_visual_local_pose(matrix, context)
    rotation = np.asarray(human._myosim_matrix_from_quaternion_xyzw(quaternion), dtype=np.float64)
    return np.asarray(translation, dtype=np.float64), rotation, float(scale)


def _owner_registration(registration: dict[str, Any], binding_ids: np.ndarray):
    grouped = {}
    for anchor in registration.get("anchors", []):
        target = anchor.get("target") if isinstance(anchor, dict) else None
        source = anchor.get("source") if isinstance(anchor, dict) else None
        record = anchor.get("registration") if isinstance(anchor, dict) else None
        if not isinstance(target, dict) or not isinstance(source, dict) or not isinstance(record, dict):
            raise human.ImportError("common-atlas registration has a malformed anchor")
        name, core_index = target.get("name"), target.get("core_body_index")
        matrix, member = record.get("source_obj_mm_to_core_inertial_body_m"), source.get("member_id")
        if not isinstance(name, str) or type(core_index) is not int or core_index < 0 or not isinstance(matrix, list) or not isinstance(member, str):
            raise human.ImportError("common-atlas registration has an invalid anchor identity")
        grouped.setdefault(core_index, []).append((name, matrix, member))
    owners, matrices, members = {}, {}, {}
    for core_index, rows in grouped.items():
        names = {row[0] for row in rows}
        if len(names) != 1:
            raise human.ImportError(f"common-atlas Core owner {core_index} maps to multiple names")
        name = next(iter(names))
        first = np.asarray(rows[0][1], dtype=np.float64)
        for _, matrix, _ in rows[1:]:
            candidate = np.asarray(matrix, dtype=np.float64)
            if candidate.shape != (4, 4) or float(np.max(np.abs(candidate - first))) > 2.0e-12:
                raise human.ImportError(f"common-atlas registered anchors for {name} do not share one owner transform")
        _transform(rows[0][1], f"common-atlas owner {name} registration")
        owners[core_index], matrices[core_index] = name, rows[0][1]
        members[core_index] = [row[2] for row in rows]
    if set(owners) != {int(value) for value in binding_ids}:
        raise human.ImportError("common-atlas registration and NHSKIN binding owners do not match exactly")
    return owners, matrices, members


def _validate_canonical_bindings_against_global_rest(*, canonical_decoded: dict[str, Any], owners: dict[int, str], runtime_bodies: dict[str, Any], global_matrix: Any, canonical_sha256: str) -> dict[str, Any]:
    """Prove the retained NHSKIN records map the global atlas through each rest body."""
    global_t, global_r, global_s = _transform(global_matrix, "shared-atlas global source transform")
    if global_s <= 0.0:
        raise human.ImportError("shared-atlas global source transform has nonpositive scale")
    expected_linear = global_r * global_s
    checks = []
    for binding_index, binding_row in enumerate(canonical_decoded["bindings_f"]):
        body_index = int(canonical_decoded["bindings_u"][binding_index, 0])
        name = owners.get(body_index)
        runtime_entry = runtime_bodies.get(name) if name is not None else None
        if runtime_entry is None or runtime_entry[0] != body_index:
            raise human.ImportError("canonical NHSKIN binding has no exact registered runtime-rest owner")
        body = runtime_entry[1]
        body_t = np.asarray(body.get("position_world_m"), dtype=np.float64)
        body_r = np.asarray(body.get("rotation_world"), dtype=np.float64)
        if body_t.shape != (3,) or body_r.shape != (3, 3) or not np.isfinite(body_t).all() or not np.isfinite(body_r).all():
            raise human.ImportError(f"canonical NHSKIN runtime rest pose is malformed for {name}")
        bind_t = binding_row[1:4].astype(np.float64)
        bind_r = np.asarray(human._myosim_matrix_from_quaternion_xyzw(binding_row[4:8].astype(float).tolist()), dtype=np.float64)
        bind_scale = float(binding_row[8])
        actual_t = body_t + body_r @ bind_t
        actual_linear = body_r @ (bind_r * bind_scale)
        translation_error = float(np.linalg.norm(actual_t - global_t))
        linear_error = float(np.max(np.abs(actual_linear - expected_linear)))
        passed = translation_error <= 1.0e-6 and linear_error <= 1.0e-6
        checks.append({"binding_index": binding_index, "core_body_index": body_index, "body_name": name,
                       "translation_error_m": translation_error, "linear_transform_max_abs_error": linear_error,
                       "passed": passed})
        if not passed:
            raise human.ImportError(f"canonical NHSKIN shared-atlas rest transform differs for {name}: translation={translation_error:.9g} m, linear={linear_error:.9g}")
    return {
        "schema": "numi.human.common-atlas-binding-runtime-rest-validation.v1",
        "canonical_binding_reference_sha256": canonical_sha256,
        "binding_count": len(checks), "all_86_bindings_match_global_atlas_through_runtime_rest": len(checks) == 86,
        "translation_tolerance_m": 1.0e-6, "linear_transform_max_abs_tolerance": 1.0e-6,
        "global_source_mm_to_world_m": np.asarray(global_matrix, dtype=np.float64).tolist(),
        "checked_runtime_rigid_sha256": None,
        "maximum_translation_error_m": max((row["translation_error_m"] for row in checks), default=0.0),
        "maximum_linear_transform_max_abs_error": max((row["linear_transform_max_abs_error"] for row in checks), default=0.0),
        "checks": checks,
        "basis": "For each canonical owner record, compare the record's rest-body translation and rotation-scale map with the registration global source transform expressed in the NHSKIN metre coordinates; source positions and the canonical records are not equated to NHTISS owner-local transforms.",
    }


def derive_common_atlas_skin_geometry(
    *,
    source_payload: Path,
    registration_path: Path,
    sources: Path,
    myosim_artifact: Path,
    input_provenance_path: Path,
    output_directory: Path,
    canonical_binding_reference: Path | None = None,
    canonical_binding_provenance_path: Path | None = None,
    clearance_accepted_pack: Path | None = None,
    clearance_accepted_receipt: Path | None = None,
    clearance_bone_artifact: Path | None = None,
    clearance_bone_manifest: Path | None = None,
    clearance_witnesses: Path | None = None,
    clearance_orientation_report: Path | None = None,
    clearance_surface_inventory: Path | None = None,
    clearance_margin_mm: float = 0.25,
    clearance_support_radius_edge_multiple: float = 4.0,
) -> dict[str, Any]:
    """Bake registered rest targets into FJ2810 shared atlas and restore canonical shared-atlas bindings."""
    source_payload = Path(source_payload).resolve()
    registration_path = Path(registration_path).resolve()
    sources = Path(sources).resolve()
    myosim_artifact = Path(myosim_artifact).resolve()
    input_provenance_path = Path(input_provenance_path).resolve()
    output_directory = Path(output_directory).resolve()
    canonical_binding_reference = Path(canonical_binding_reference or source_payload).resolve()
    canonical_binding_provenance_path = Path(canonical_binding_provenance_path or input_provenance_path).resolve()
    clearance_paths = (
        clearance_accepted_pack, clearance_accepted_receipt, clearance_bone_artifact,
        clearance_bone_manifest, clearance_witnesses, clearance_orientation_report, clearance_surface_inventory,
    )
    if any(value is not None for value in clearance_paths) and not all(value is not None for value in clearance_paths):
        raise human.ImportError("step-0 clearance requires the complete pack, receipt, NHBONES, witness, orientation, and 859-surface inventory input set")
    for path, label in ((source_payload, "source NHSKIN"), (registration_path, "registration"), (input_provenance_path, "immediate-source NHSKIN provenance"), (canonical_binding_reference, "canonical NHSKIN binding reference"), (canonical_binding_provenance_path, "canonical NHSKIN binding provenance")):
        if not path.is_file():
            raise human.ImportError(f"common-atlas geometry registration requires existing {label}")
    if output_directory.exists():
        raise human.ImportError("common-atlas geometry registration refuses to overwrite an evidence directory")
    raw = source_payload.read_bytes()
    canonical_raw = canonical_binding_reference.read_bytes()
    input_sha, canonical_sha, registration_raw = _sha_bytes(raw), _sha_bytes(canonical_raw), registration_path.read_bytes()
    registration_sha = _sha_bytes(registration_raw)
    registration = json.loads(registration_raw)
    provenance = json.loads(input_provenance_path.read_bytes())
    canonical_provenance = json.loads(canonical_binding_provenance_path.read_bytes())
    source_route = "pinned_boundary_repaired_common_atlas_source"
    derived = provenance.get("derived_skin", {})
    source_skin = provenance.get("source_skin", {})
    if Path(derived.get("path", "")).resolve() != source_payload or derived.get("sha256") != input_sha:
        immediate_output = provenance.get("output_payload", {})
        if (provenance.get("schema") != "numi.human.skin-lower-limb-anchor-rebind-candidate.v1"
                or immediate_output.get("path") != str(source_payload)
                or immediate_output.get("sha256") != input_sha):
            raise human.ImportError("common-atlas immediate input differs from a pinned source or retained 004 geometry template")
        source_route = "retained_004_invalid_binding_candidate_geometry_template_only"
        source_skin = canonical_provenance.get("source_skin", {})
    canonical_derived = canonical_provenance.get("derived_skin", {})
    if (Path(canonical_derived.get("path", "")).resolve() != canonical_binding_reference
            or canonical_derived.get("sha256") != canonical_sha):
        raise human.ImportError("canonical NHSKIN binding reference differs from its pinned boundary-repair receipt")
    if not isinstance(source_skin.get("path"), str) or not isinstance(source_skin.get("sha256"), str):
        raise human.ImportError("common-atlas source receipt lacks the raw NHSKIN source identity")
    original_source_path = Path(source_skin["path"]).resolve()
    if not original_source_path.is_file() or human.sha256(original_source_path) != source_skin["sha256"]:
        raise human.ImportError("common-atlas raw NHSKIN source differs from its pinned source receipt")

    decoded = decode_payload(raw)
    canonical_decoded = decode_payload(canonical_raw)
    for name in ("binding_count", "vertex_count", "index_count", "registration_fingerprint32", "source_archive_sha256"):
        if decoded[name] != canonical_decoded[name]:
            raise human.ImportError(f"common-atlas geometry template and canonical binding reference differ in {name}")
    for name, left, right in (("owner IDs", decoded["bindings_u"][:, 0], canonical_decoded["bindings_u"][:, 0]), ("positions", decoded["vertices_u"][:, :3], canonical_decoded["vertices_u"][:, :3]), ("full weights", decoded["full_weights"], canonical_decoded["full_weights"]), ("triangle indices", decoded["indices"], canonical_decoded["indices"]), ("per-vertex influences", decoded["vertices_u"][:, 6:14], canonical_decoded["vertices_u"][:, 6:14])):
        if not np.array_equal(left, right):
            raise human.ImportError(f"common-atlas geometry template differs from canonical source in {name}")
    if registration.get("schema") != "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2":
        raise human.ImportError("common-atlas geometry registration requires the pinned BodyParts3D/Myosim v2 owner")
    if int(decoded["binding_count"]) != 86:
        raise human.ImportError("common-atlas source NHSKIN must retain all 86 canonical Core-owner bindings")
    if decoded["registration_fingerprint32"] != int(registration_sha[:8], 16):
        raise human.ImportError("common-atlas registration SHA does not match the NHSKIN fingerprint")
    source_archive_sha = registration.get("source", {}).get("myosim", {}).get("source", {}).get("archive_sha256")
    if decoded["source_archive_sha256"] != source_archive_sha:
        raise human.ImportError("common-atlas NHSKIN MyoSim archive differs from the registration")
    owners, owner_matrices, owner_members = _owner_registration(registration, decoded["bindings_u"][:, 0])
    runtime_reference, runtime_bodies = human._bodyparts_runtime_bindings(registration, myosim_artifact)
    for core_index, name in owners.items():
        runtime = runtime_bodies.get(name)
        if runtime is None or runtime[0] != core_index:
            raise human.ImportError(f"common-atlas runtime owner mapping differs for {name}")

    canonical_binding_validation = _validate_canonical_bindings_against_global_rest(
        canonical_decoded=canonical_decoded, owners=owners, runtime_bodies=runtime_bodies,
        global_matrix=registration.get("coordinate_system", {}).get("global_source_mm_to_myosim_world_m"),
        canonical_sha256=canonical_sha,
    )
    canonical_binding_validation["checked_runtime_rigid_sha256"] = runtime_reference.get("rigid", {}).get("sha256")
    if canonical_binding_validation["all_86_bindings_match_global_atlas_through_runtime_rest"] is not True:
        raise human.ImportError("canonical NHSKIN shared-atlas runtime-rest validation did not cover all 86 records")

    bodyparts_source = registration.get("source", {}).get("bodyparts")
    if not isinstance(bodyparts_source, dict):
        raise human.ImportError("common-atlas registration lacks BodyParts3D source provenance")
    skin_archive, skin_member, skin_obj = human._bodyparts_obj_member(sources, "is_a", "FJ2810")
    expected_archive = next((row for row in bodyparts_source.get("archives", []) if row.get("file") == skin_archive.name and row.get("hierarchy") == "is_a"), None)
    skin_archive_sha = human.sha256(skin_archive)
    if expected_archive is None or expected_archive.get("sha256") != skin_archive_sha:
        raise human.ImportError("common-atlas raw FJ2810 archive is not the pinned BodyParts3D source")

    vertex_offset = 60 + 36 * int(decoded["binding_count"])
    vertices = decoded["vertices_f"][:, :3].astype(np.float64, copy=True)
    faces = decoded["indices"].reshape(-1, 3).astype(np.int64, copy=False)
    referenced = np.unique(faces)
    referenced_mask = np.zeros(int(decoded["vertex_count"]), dtype=bool)
    referenced_mask[referenced] = True
    points = vertices[referenced]
    weights = decoded["full_weights"][referenced].astype(np.float64, copy=False)
    if not np.isfinite(points).all() or not np.isfinite(weights).all() or float(weights.min(initial=0.0)) < 0.0 or float(np.max(np.abs(weights.sum(axis=1) - 1.0))) > 4.0e-6:
        raise human.ImportError("common-atlas source positions or full weights are invalid")

    target_world = np.zeros_like(points)
    for binding_index, core_id_value in enumerate(decoded["bindings_u"][:, 0]):
        core_id, name = int(core_id_value), owners[int(core_id_value)]
        local_t, local_r, local_s = _transform(owner_matrices[core_id], f"common-atlas owner {name} transform")
        _, body = runtime_bodies[name]
        body_t, body_r = np.asarray(body["position_world_m"], dtype=np.float64), np.asarray(body["rotation_world"], dtype=np.float64)
        if body_t.shape != (3,) or body_r.shape != (3, 3):
            raise human.ImportError(f"common-atlas runtime pose is malformed for {name}")
        owner_local = local_t + local_s * (points @ local_r.T)
        owner_world = body_t + owner_local @ body_r.T
        target_world += decoded["full_weights"][referenced, binding_index, None] * owner_world

    global_matrix = registration.get("coordinate_system", {}).get("global_source_mm_to_myosim_world_m")
    global_t, global_r, global_s = _transform(global_matrix, "common-atlas global source transform")
    inferred_source = ((target_world - global_t) @ global_r) / global_s
    if not np.isfinite(inferred_source).all() or global_s <= 0:
        raise human.ImportError("common-atlas inverse source transform is invalid")

    output = bytearray(raw)
    binding_offset = 60
    binding_bytes = 36 * int(decoded["binding_count"])
    output[binding_offset:binding_offset + binding_bytes] = canonical_raw[binding_offset:binding_offset + binding_bytes]
    for compact_index, vertex_index in enumerate(referenced):
        struct.pack_into("<3f", output, vertex_offset + int(vertex_index) * 56, *(float(value) for value in inferred_source[compact_index]))
    intermediate = decode_payload(bytes(output))
    packed_source = intermediate["vertices_f"][referenced, :3].astype(np.float64)
    rendered_world = global_t + global_s * (packed_source @ global_r.T)
    world_residual = np.linalg.norm(rendered_world - target_world, axis=1)

    compact_map = np.full(int(decoded["vertex_count"]), -1, dtype=np.int64)
    compact_map[referenced] = np.arange(len(referenced), dtype=np.int64)
    compact_faces = compact_map[faces]
    compact_triangles = [tuple(int(index) for index in row) for row in compact_faces]
    normals = human._bodyparts_vertex_normals([tuple(float(value) for value in point) for point in rendered_world], compact_triangles, "inferred common-atlas NHSKIN geometry")
    normals = human._bodyparts_skin_smooth_visual_normals(normals, compact_triangles)
    normal_array = np.asarray(normals, dtype=np.float64)
    normal_lengths = np.linalg.norm(normal_array, axis=1)
    if not np.isfinite(normal_array).all() or np.any(normal_lengths <= 1.0e-12):
        raise human.ImportError("common-atlas geometry generated non-finite or degenerate normals")
    normal_array /= normal_lengths[:, None]
    for compact_index, vertex_index in enumerate(referenced):
        struct.pack_into("<3f", output, vertex_offset + int(vertex_index) * 56 + 12, *(float(value) for value in normal_array[compact_index]))
    candidate = bytes(output)
    clearance_report = None
    if all(value is not None for value in clearance_paths):
        base_decoded = decode_payload(candidate)
        clearance_source, clearance_report = derive_step0_inferred_clearance(
            source_positions=base_decoded["vertices_f"][:, :3].astype(np.float64),
            full_weights=base_decoded["full_weights"].astype(np.float64),
            bindings=base_decoded["bindings_f"].astype(np.float64),
            binding_owner_ids=base_decoded["bindings_u"][:, 0].astype(np.int64),
            faces=base_decoded["indices"].reshape(-1, 3).astype(np.int64),
            source_payload_sha256=_sha_bytes(candidate),
            accepted_pack_path=clearance_accepted_pack,
            accepted_receipt_path=clearance_accepted_receipt,
            bone_artifact_path=clearance_bone_artifact,
            bone_manifest_path=clearance_bone_manifest,
            witness_path=clearance_witnesses,
            orientation_report_path=clearance_orientation_report,
            surface_inventory_path=clearance_surface_inventory,
            selected_margin_mm=clearance_margin_mm,
            support_radius_edge_multiple=clearance_support_radius_edge_multiple,
        )
        output = bytearray(candidate)
        for vertex_index in referenced:
            struct.pack_into("<3f", output, vertex_offset + int(vertex_index) * 56, *(float(value) for value in clearance_source[int(vertex_index)]))
        cleared = decode_payload(bytes(output))
        packed_source = cleared["vertices_f"][referenced, :3].astype(np.float64)
        rendered_world = global_t + global_s * (packed_source @ global_r.T)
        normals = human._bodyparts_vertex_normals(
            [tuple(float(value) for value in point) for point in rendered_world],
            compact_triangles,
            "inferred common-atlas NHSKIN geometry with witnessed clearance",
        )
        normals = human._bodyparts_skin_smooth_visual_normals(normals, compact_triangles)
        normal_array = np.asarray(normals, dtype=np.float64)
        normal_lengths = np.linalg.norm(normal_array, axis=1)
        if not np.isfinite(normal_array).all() or np.any(normal_lengths <= 1.0e-12):
            raise human.ImportError("inferred clearance generated non-finite or degenerate rest-world normals")
        normal_array /= normal_lengths[:, None]
        for compact_index, vertex_index in enumerate(referenced):
            struct.pack_into("<3f", output, vertex_offset + int(vertex_index) * 56 + 12, *(float(value) for value in normal_array[compact_index]))
        candidate = bytes(output)
    after = decode_payload(candidate)
    if not np.array_equal(canonical_decoded["bindings_u"], after["bindings_u"]):
        raise human.ImportError("common-atlas geometry did not restore exact canonical shared-atlas binding records")
    if not np.array_equal(canonical_decoded["full_weights"], after["full_weights"]):
        raise human.ImportError("common-atlas geometry changed canonical full skin weights")
    if not np.array_equal(decoded["indices"], after["indices"]):
        raise human.ImportError("common-atlas geometry changed triangle topology or order")
    if not np.array_equal(decoded["vertices_u"][:, 6:14], after["vertices_u"][:, 6:14]):
        raise human.ImportError("common-atlas geometry changed per-vertex influence records")
    if not np.array_equal(decoded["vertices_u"][~referenced_mask], after["vertices_u"][~referenced_mask]):
        raise human.ImportError("common-atlas geometry changed unreferenced source vertices")
    displacement = np.linalg.norm(rendered_world - (global_t + global_s * (vertices[referenced] @ global_r.T)), axis=1)

    output_directory.mkdir(parents=True, exist_ok=False)
    output_payload = output_directory / source_payload.name
    output_manifest = output_directory / "common-atlas-skin-geometry-registration.manifest.json"
    output_payload.write_bytes(candidate)
    manifest = {
        "schema": SCHEMA,
        "status": STATUS,
        "method": ("inferred geometry bake through exact registered Core-owner transforms and unchanged full 86-owner weights into the unchanged global common atlas frame"
                   + ("; then a step-0 witnessed compact 0.25 mm engineering-clearance field is inverse-mapped through all 86 unchanged skin influences" if clearance_report is not None else "")),
        "inputs": {
            "source_payload": {"path": str(source_payload), "sha256": input_sha, "bytes": len(raw), "route": source_route},
            "canonical_binding_reference": {"path": str(canonical_binding_reference), "sha256": canonical_sha, "provenance_path": str(canonical_binding_provenance_path), "provenance_sha256": _sha_bytes(canonical_binding_provenance_path.read_bytes()), "meaning": "authoritative canonical shared-atlas NHSKIN bindings and vertex weights; per-owner binding substitutions from intermediate inputs are rejected and restored from these exact records"},
            "upstream_skin_provenance": {"immediate_source_path": str(input_provenance_path), "immediate_source_sha256": _sha_bytes(input_provenance_path.read_bytes()), "immediate_source_record": provenance.get("output_payload") or provenance.get("derived_skin"), "canonical_source_skin": source_skin, "canonical_derived_skin": canonical_derived, "intermediate_binding_candidate_route": ("004 is retained only as a positions/topology/weights-identical geometry template; its 12 transformed bindings are explicitly rejected as invalid common-atlas data" if source_route.startswith("retained_004") else None)},
            "registration_path": str(registration_path), "registration_sha256": registration_sha,
            "registration_fingerprint32": f"{decoded['registration_fingerprint32']:08x}",
            "bodyparts_source": bodyparts_source,
            "raw_FJ2810": {"member_id": "FJ2810", "member_name": skin_member, "hierarchy": "is_a", "archive_path": str(skin_archive), "archive_sha256": skin_archive_sha, "member_sha256": _sha_bytes(skin_obj), "interpretation": "raw source mesh; candidate positions are inferred registered geometry, not measured source coordinates"},
            "myosim_source_archive_sha256": source_archive_sha,
            "runtime_reference": runtime_reference,
            "myosim_artifact": str(myosim_artifact),
        },
        "payload_identity": {"source_archive_sha256": decoded["source_archive_sha256"], "registration_fingerprint32": f"{decoded['registration_fingerprint32']:08x}", "binding_count": int(decoded["binding_count"]), "vertex_count": int(decoded["vertex_count"]), "index_count": int(decoded["index_count"]), "payload_bytes": len(raw)},
        "common_atlas_binding_runtime_rest_validation": canonical_binding_validation,
        "geometry_registration": {
            "registered_owner_count": len(owners), "full_weight_columns_used": int(decoded["binding_count"]),
            "source_anchor_count": sum(len(rows) for rows in owner_members.values()), "same_owner_transform_spread_tolerance": 2.0e-12,
            "rendered_referenced_vertex_count": int(len(referenced)),
            "registered_geometry_displacement_mm": _percentiles(displacement * 1000.0),
            "inverse_map_float32_roundtrip_residual_mm": _percentiles(world_residual * 1000.0),
            "maximum_displacement_vertex_index": int(referenced[int(np.argmax(displacement))]),
            "includes_step0_engineering_clearance": clearance_report is not None,
            "source_rights_and_provenance": {"carried_from_registration": bodyparts_source, "raw_member_sha256": _sha_bytes(skin_obj), "no_new_rights_or_measurements_claimed": True},
        },
        "preservation": {
            "all_86_canonical_binding_records_byte_identical": True, "binding_records_match_canonical_reference_sha256": canonical_sha,
            "canonical_binding_records_validated_against_registration_global_atlas_and_runtime_rest": canonical_binding_validation["all_86_bindings_match_global_atlas_through_runtime_rest"],
            "source_binding_rows_restored_to_canonical": not np.array_equal(decoded["bindings_u"], after["bindings_u"]),
            "input_binding_rows_differed_from_canonical_reference": not np.array_equal(decoded["bindings_u"], canonical_decoded["bindings_u"]),
            "final_binding_rows_byte_identical_to_canonical_reference": np.array_equal(after["bindings_u"], canonical_decoded["bindings_u"]),
            "full_weight_matrix_byte_identical": True,
            "triangle_indices_and_order_byte_identical": True, "per_vertex_influence_records_byte_identical": True,
            "unreferenced_vertex_bytes_byte_identical": True, "source_archive_sha256_preserved": True,
            "registration_fingerprint32_preserved": True, "nhtiss_tendon_muscle_and_physics_payloads_touched": False,
            "skin_topology_changed": False, "skin_contact_or_physics_owner_changed": False, "indexed_rest_world_normals_recomputed": True,
        },
        "step0_inferred_clearance": clearance_report,
        "qualification": {
            "step0_witnessed_all_surface_clearance": ("passed_842_nonocular_surfaces_eyes_monitored_unchanged" if clearance_report is not None and clearance_report["qualification"]["accepted_pose0_all_4448_nonocular_pairs_and_all_842_clearance_targets"] == "passed" else "pending"),
            "native_pose_clearance": "pending", "skin_self_intersection": "pending",
            "registered_skeleton_and_organ_clearance": "pending", "EHL_route_clearance": "pending",
            "physical_or_collision_use": "not admitted",
        },
        "evidence_boundary": ("Inferred registered geometry with a bounded step-0 engineering separation field, not raw measured FJ2810 or measured cutaneous thickness. Native pose, skin self, other anatomy, and route qualification remain pending."
                             if clearance_report is not None else
                             "Inferred registered geometry, not raw measured FJ2810. This candidate does not resolve known hip/sternum crossings by itself and is not anatomically qualified."),
        "code": {
            "module": str(Path(__file__).resolve()),
            "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "clearance_module": str(Path(clearance_module.__file__).resolve()),
            "clearance_module_sha256": hashlib.sha256(Path(clearance_module.__file__).read_bytes()).hexdigest(),
            "argv": sys.argv,
        },
        "output_payload": {"path": str(output_payload), "sha256": _sha_bytes(candidate), "bytes": len(candidate)},
        "output_manifest": str(output_manifest),
    }
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest



def compose_disjoint_skin_position_corrections(
    base_payload: bytes,
    correction_payloads: list[bytes],
    *,
    global_source_matrix: Any,
) -> tuple[bytes, dict[str, Any]]:
    """Compose isolated geometry corrections within the existing NHSKIN ABI.

    Each correction must retain all non-geometric source bytes. Neither this
    composition nor normal reconstruction admits the result for native use;
    the combined candidate still needs accepted-pose geometry checks.
    """
    base = decode_payload(base_payload)
    if not correction_payloads:
        raise human.ImportError("skin composition requires at least one correction")
    vertex_count = int(base["vertex_count"])
    vertex_offset = 60 + 36 * int(base["binding_count"])
    vertex_end = vertex_offset + 56 * vertex_count
    faces = base["indices"].reshape(-1, 3).astype(np.int64)
    referenced = np.unique(faces)
    referenced_mask = np.zeros(vertex_count, dtype=bool)
    referenced_mask[referenced] = True
    base_positions = base["vertices_f"][:, :3]
    positions = base_positions.copy()
    assigned = np.zeros(vertex_count, dtype=bool)
    assigned_faces = np.zeros(len(faces), dtype=bool)
    records = []
    for index, raw in enumerate(correction_payloads):
        if len(raw) != len(base_payload):
            raise human.ImportError("skin correction changed payload size")
        candidate = decode_payload(raw)
        if (raw[:vertex_offset] != base_payload[:vertex_offset]
                or raw[vertex_end:] != base_payload[vertex_end:]
                or not np.array_equal(candidate["vertices_u"][:, 6:], base["vertices_u"][:, 6:])):
            raise human.ImportError("skin correction changed topology, bindings, weights, or source identity")
        changed = np.any(candidate["vertices_u"][:, :3] != base["vertices_u"][:, :3], axis=1)
        if np.any(changed & ~referenced_mask):
            raise human.ImportError("skin correction changed an unreferenced vertex")
        if not np.isfinite(candidate["vertices_f"][:, :6]).all():
            raise human.ImportError("skin correction contains nonfinite geometry")
        touched_faces = np.any(changed[faces], axis=1)
        if np.any(changed & assigned) or np.any(touched_faces & assigned_faces):
            raise human.ImportError("skin corrections overlap in vertices or incident faces")
        positions[changed] = candidate["vertices_f"][changed, :3]
        assigned |= changed
        assigned_faces |= touched_faces
        records.append({"correction_index": index, "payload_sha256": _sha_bytes(raw),
                        "changed_vertex_count": int(changed.sum()),
                        "changed_vertex_ids": np.flatnonzero(changed).tolist(),
                        "incident_face_count": int(touched_faces.sum())})

    translation, rotation, scale = _transform(global_source_matrix, "composed shared-atlas source transform")
    if scale <= 0:
        raise human.ImportError("skin composition has nonpositive shared-atlas scale")
    world = translation + scale * (positions[referenced].astype(np.float64) @ rotation.T)
    local_faces = np.searchsorted(referenced, faces)
    triangles = [tuple(int(value) for value in row) for row in local_faces]
    normals = human._bodyparts_vertex_normals(
        [tuple(float(value) for value in row) for row in world],
        triangles, "composed common-atlas NHSKIN geometry")
    normals = np.asarray(human._bodyparts_skin_smooth_visual_normals(normals, triangles), dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(lengths <= 1.0e-12):
        raise human.ImportError("composed skin has invalid rest-world normals")
    normals /= lengths[:, None]
    output = bytearray(base_payload)
    for vertex in np.flatnonzero(assigned):
        struct.pack_into("<3f", output, vertex_offset + int(vertex) * 56,
                         *(float(value) for value in positions[vertex]))
    for ordinal, vertex in enumerate(referenced):
        struct.pack_into("<3f", output, vertex_offset + int(vertex) * 56 + 12,
                         *(float(value) for value in normals[ordinal]))
    raw = bytes(output)
    return raw, {"base_payload_sha256": _sha_bytes(base_payload),
                 "output_payload_sha256": _sha_bytes(raw),
                 "corrections": records,
                 "non_geometric_source_bytes_preserved": True,
                 "correction_vertex_and_incident_face_supports_disjoint": True,
                 "rest_world_normals_recomputed_by_existing_skin_owner": True,
                 "qualification": "composed candidate; combined native pose and exact intersection audits pending"}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-payload", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--myosim-artifact", type=Path, required=True)
    parser.add_argument("--input-provenance", type=Path, required=True)
    parser.add_argument("--canonical-binding-reference", type=Path, help="optional exact canonical NHSKIN source to restore when a retained geometry template has invalid per-owner bindings")
    parser.add_argument("--canonical-binding-provenance", type=Path, help="boundary-repair receipt that hash-binds --canonical-binding-reference")
    parser.add_argument("--clearance-accepted-pack", type=Path, help="accepted step-0 MRVPACK used to derive the bounded clearance field")
    parser.add_argument("--clearance-accepted-receipt", type=Path, help="hash-bound receipt for --clearance-accepted-pack")
    parser.add_argument("--clearance-bone-artifact", type=Path, help="registered ABI-3 NHBONES source used to recover all 86 accepted body poses")
    parser.add_argument("--clearance-bone-manifest", type=Path, help="source manifest for --clearance-bone-artifact")
    parser.add_argument("--clearance-witnesses", type=Path, help="exact accepted-pose skin/bone/Achilles intersections")
    parser.add_argument("--clearance-orientation-report", type=Path, help="local full-shell winding classification for every witness face")
    parser.add_argument("--clearance-surface-inventory", type=Path, help="hash-pinned exact inventory for all 859 bone, tissue, organ, vessel, and visceral surfaces")
    parser.add_argument("--clearance-margin-mm", type=float, default=0.25)
    parser.add_argument("--clearance-support-radius-edge-multiple", type=float, default=4.0, help="compact-field geodesic support radius in median local edge lengths")
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = derive_common_atlas_skin_geometry(
        source_payload=args.source_payload, registration_path=args.registration, sources=args.sources,
        myosim_artifact=args.myosim_artifact, input_provenance_path=args.input_provenance,
        output_directory=args.output_directory, canonical_binding_reference=args.canonical_binding_reference,
        canonical_binding_provenance_path=args.canonical_binding_provenance,
        clearance_accepted_pack=args.clearance_accepted_pack,
        clearance_accepted_receipt=args.clearance_accepted_receipt,
        clearance_bone_artifact=args.clearance_bone_artifact,
        clearance_bone_manifest=args.clearance_bone_manifest,
        clearance_witnesses=args.clearance_witnesses,
        clearance_orientation_report=args.clearance_orientation_report,
        clearance_surface_inventory=args.clearance_surface_inventory,
        clearance_margin_mm=args.clearance_margin_mm,
        clearance_support_radius_edge_multiple=args.clearance_support_radius_edge_multiple,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
