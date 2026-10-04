"""Source-bone-seeded motion screen for the complete FJ2810 skin solid."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from . import model as human
from .skin_full_solid_crossing_classes import classify_pairs
from .skin_full_solid_motion import (
    HELDOUT_DIR,
    PLAN_PATH as HARMONIC_PLAN_PATH,
    SOLID_DIR,
    _deform_lbs,
    _load_skin,
    _load_solid,
    _native_skin_positions,
    _outer_to_full,
    _write_binary,
    require,
)
from .skin_dual_quaternion_audit import POSES
from .compiled_quotient_embeddedness import coordinate_quotient
from .surface_topology_audit import exact_embedding
from .whole_body_embeddedness import atomic_json

ROOT = human.REPOSITORY_ROOT
PLAN_PATH = ROOT / "Docs/media/skin-full-solid-motion-20261004/source-seeded-plan.json"
REGISTRATION_PATH = ROOT / "Docs/media/skin-patella-rebase-source-preflight-20261003/registration.json"
SOURCE_PREFLIGHT_PATH = ROOT / "Docs/media/skin-source-fit-recovery-20261004/receipt.json"
CORE_MANIFEST_PATH = HELDOUT_DIR / "inputs/native/myosim-fullbody-reference.manifest.json"
OUTPUT_SCHEMA = "numi.human.skin-full-solid-source-seeded-screen.v1"


def _source_bone_records(registration: dict[str, Any], matrix: np.ndarray,
                         bindings: list[tuple]) -> list[dict[str, Any]]:
    binding_by_core = {int(row[0]): index for index, row in enumerate(bindings)}
    records = []
    for anchor_index, anchor in enumerate(registration["anchors"]):
        source = anchor["source"]
        target = anchor["target"]
        core_id = int(target["core_body_index"])
        require(core_id in binding_by_core, "registered bone anchor maps to ABI 5 binding")
        archive_path, member, obj = human._bodyparts_obj_member(
            ROOT / "Sources", source["hierarchy"], source["member_id"],
        )
        require(human.sha256(archive_path) == source["archive_sha256"]
                and member == source["member"]
                and hashlib.sha256(obj).hexdigest() == source["member_sha256"],
                f"registered bone source identity {anchor_index}")
        bone_vertices_mm, _ = human._bodyparts_obj_triangles(obj, member)
        bone_vertices_mm = np.asarray(bone_vertices_mm, dtype=np.float64)
        count = min(64, len(bone_vertices_mm))
        sample_ids = (np.arange(count, dtype=np.int64) * (len(bone_vertices_mm)-1)
                      // (count-1) if count > 1 else np.array([0], dtype=np.int64))
        bone_world = bone_vertices_mm @ matrix[:3, :3].T + matrix[:3, 3]
        centroid = bone_world.mean(axis=0)
        radius = 2.0 * float(np.linalg.norm(bone_world-centroid, axis=1).max())
        require(np.isfinite(radius) and radius > 0, "registered bone envelope radius")
        records.append({
            "anchor_index": anchor_index,
            "core_body_index": core_id,
            "binding_index": binding_by_core[core_id],
            "centroid_world_m": centroid,
            "sample_world_m": bone_world[sample_ids],
            "sample_source_vertex_ids": sample_ids,
            "diameter_bound_m": radius,
        })
    require({row["core_body_index"] for row in records} == set(binding_by_core),
            "registered source bones cover all ABI 5 bindings")
    return records


def source_seeded_extend(
    vertices: np.ndarray, faces: np.ndarray, fixed_ids: np.ndarray,
    fixed_weights: np.ndarray, bones: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a screened graph extension in the registered source world frame."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    fixed_ids = np.asarray(fixed_ids, dtype=np.int64)
    fixed_weights = np.asarray(fixed_weights, dtype=np.float64)
    require(vertices.ndim == 2 and vertices.shape[1] == 3
            and faces.ndim == 2 and faces.shape[1] == 3
            and fixed_weights.ndim == 2
            and fixed_weights.shape[0] == len(fixed_ids)
            and bool((faces >= 0).all() and (faces < len(vertices)).all())
            and bool((fixed_ids >= 0).all() and (fixed_ids < len(vertices)).all()),
            "source-seeded extension shapes and indices")
    require(len(np.unique(fixed_ids)) == len(fixed_ids)
            and bool(np.isfinite(vertices).all() and np.isfinite(fixed_weights).all())
            and bool((fixed_weights >= 0).all())
            and float(np.max(np.abs(fixed_weights.sum(axis=1)-1))) <= 1e-6,
            "fixed outer source rows are valid")
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]],
                                               faces[:, [1, 2]],
                                               faces[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]]-vertices[edges[:, 1]], axis=1)
    require(bool(np.isfinite(lengths).all() and (lengths > 0).all()),
            "source graph edge lengths")
    conductance = 1.0/lengths
    adjacency = coo_matrix((np.r_[conductance, conductance],
                            (np.r_[edges[:, 0], edges[:, 1]],
                             np.r_[edges[:, 1], edges[:, 0]])),
                           shape=(len(vertices), len(vertices))).tocsr()
    from scipy.sparse.csgraph import connected_components
    components = int(connected_components(adjacency, directed=False,
                                          return_labels=False))
    require(components == 1, "complete source-skin graph is connected")
    distance_graph = coo_matrix((np.r_[lengths, lengths],
                                 (np.r_[edges[:, 0], edges[:, 1]],
                                  np.r_[edges[:, 1], edges[:, 0]])),
                                shape=(len(vertices), len(vertices))).tocsr()
    fixed = np.zeros(len(vertices), dtype=bool)
    fixed[fixed_ids] = True
    unknown_ids = np.flatnonzero(~fixed)
    binding_count = fixed_weights.shape[1]
    tree = cKDTree(vertices)
    owners: dict[int, dict[int, list[float]]] = {}
    admitted_count = 0
    admitted_bodies: set[int] = set()
    fixed_seed_count = 0
    nearest_seed_ids = []
    projection_gaps = []

    def closest(point: np.ndarray) -> tuple[int, float]:
        distance, index = tree.query(point)
        nearby = tree.query_ball_point(point, float(distance)+1e-12)
        if len(nearby) > 1:
            delta = vertices[nearby]-point
            squared = np.einsum("ij,ij->i", delta, delta)
            index = nearby[int(np.argmin(squared))]
        require(np.isfinite(distance) and distance > 0,
                "bone-to-source projection gap is positive and finite")
        return int(index), float(distance)

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = (diags(degree)-adjacency).tocsr()
    center_cache: dict[int, np.ndarray] = {}
    for bone in bones:
        center, center_gap = closest(bone["centroid_world_m"])
        radius = float(bone["diameter_bound_m"]+2.0*center_gap)
        if center not in center_cache:
            center_cache[center] = dijkstra(distance_graph, directed=False,
                                            indices=center)
        graph_distance = center_cache[center]
        points = np.vstack((bone["centroid_world_m"], bone["sample_world_m"]))
        ids = np.r_[np.uint32(0xFFFFFFFF),
                    np.asarray(bone["sample_source_vertex_ids"], dtype=np.uint32)]
        for source_id, point in zip(ids, points, strict=True):
            vertex, gap = closest(point)
            projection_gaps.append(gap)
            nearest_seed_ids.append(vertex)
            admitted = bool(graph_distance[vertex] <= radius)
            if not admitted:
                continue
            admitted_count += 1
            admitted_bodies.add(int(bone["binding_index"]))
            if fixed[vertex]:
                fixed_seed_count += 1
                continue
            owners.setdefault(vertex, {}).setdefault(
                int(bone["binding_index"]), [],
            ).append(1.0/gap)

    seeded_bodies = {body for per_vertex in owners.values() for body in per_vertex}
    all_bodies = {int(bone["binding_index"]) for bone in bones}
    require(admitted_count > 0 and bool(owners)
            and admitted_bodies == all_bodies
            and seeded_bodies.issubset(all_bodies)
            and len(all_bodies) == binding_count,
            "source projection coverage: "
            f"admitted={admitted_count}, fixed={fixed_seed_count}, "
            f"unknown_seed_vertices={len(owners)}, "
            f"admitted_bindings={sorted(admitted_bodies)}, "
            f"missing_bindings={sorted(all_bodies-admitted_bodies)}, "
            f"binding_count={binding_count}")

    confidence = np.zeros(len(vertices), dtype=np.float64)
    source_rhs = np.zeros((len(vertices), binding_count), dtype=np.float64)
    for vertex, body_gaps in owners.items():
        per_body = {body: float(np.mean(gaps)) for body, gaps in body_gaps.items()}
        confidence[vertex] = float(np.mean(list(per_body.values())))
        total = sum(per_body.values())
        for body, value in per_body.items():
            source_rhs[vertex, body] = confidence[vertex]*value/total
    result = np.empty((len(vertices), binding_count), dtype=np.float64)
    result[fixed_ids] = fixed_weights
    operator = (laplacian[unknown_ids][:, unknown_ids]
                + diags(confidence[unknown_ids])).tocsc()
    rhs = (-laplacian[unknown_ids][:, fixed_ids] @ result[fixed_ids]
           + source_rhs[unknown_ids])
    extension = splu(operator).solve(np.asarray(rhs))
    minimum_before_clamp = float(extension.min())
    require(bool(np.isfinite(extension).all()) and minimum_before_clamp >= -1e-12,
            "screened Dirichlet solve positivity and finiteness")
    extension[extension < 0] = 0.0
    partitions = extension.sum(axis=1)
    require(bool(np.isfinite(partitions).all() and (partitions > 1e-12).all()),
            "screened unknown rows have positive partition")
    extension /= partitions[:, None]
    result[unknown_ids] = extension
    require(bool(np.isfinite(result).all() and (result >= 0).all())
            and np.array_equal(result[fixed_ids], fixed_weights)
            and float(np.max(np.abs(result[unknown_ids].sum(axis=1)-1))) <= 1e-10,
            "source-seeded partition and exact outer constraints")
    return result, {
        "graph_vertex_count": len(vertices),
        "graph_edge_count": len(edges),
        "graph_component_count": components,
        "fixed_outer_vertex_count": len(fixed_ids),
        "unknown_inner_and_connector_vertex_count": len(unknown_ids),
        "bone_anchor_count": len(bones),
        "bone_projection_candidate_count": len(projection_gaps),
        "admitted_bone_projection_count": admitted_count,
        "rejected_bone_projection_count": len(projection_gaps)-admitted_count,
        "admitted_fixed_outer_projection_count": fixed_seed_count,
        "admitted_binding_indices": sorted(admitted_bodies),
        "screened_unknown_seed_vertex_count": len(owners),
        "screened_unknown_seed_body_count": len(seeded_bodies),
        "screened_unknown_seed_body_indices": sorted(seeded_bodies),
        "minimum_projection_gap_m": float(min(projection_gaps)),
        "maximum_projection_gap_m": float(max(projection_gaps)),
        "minimum_weight_before_roundoff_clamp": minimum_before_clamp,
        "fixed_outer_maximum_partition_error": float(
            np.max(np.abs(fixed_weights.sum(axis=1)-1))),
        "unknown_maximum_partition_error": float(
            np.max(np.abs(result[unknown_ids].sum(axis=1)-1))),
        "weight_field_sha256": hashlib.sha256(result.astype("<f4").tobytes()).hexdigest(),
    }


def audit(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = human.read_json(PLAN_PATH)
    plan_sha = human.sha256(PLAN_PATH)
    solid_path = SOLID_DIR / "bodyparts3d-full-source-skin-solid.nhsolid"
    solid_manifest_path = SOLID_DIR / "manifest.json"
    solid_audit_path = SOLID_DIR / "audit.json"
    payload_path = HELDOUT_DIR / "inputs/base/bodyparts3d-myosim-skinned-shell.nhskin"
    manifest_path = HELDOUT_DIR / "inputs/base/bodyparts3d-myosim-skinned-shell.manifest.json"
    source_preflight_path = SOURCE_PREFLIGHT_PATH
    registration_path = REGISTRATION_PATH
    require(human.sha256(solid_path) == plan["model"]["full_solid_sha256"]
            and human.sha256(solid_manifest_path) == plan["model"]["full_solid_manifest_sha256"]
            and human.sha256(solid_audit_path) == plan["model"]["full_solid_audit_sha256"],
            "pinned source-solid identity")
    require(human.sha256(payload_path) == plan["model"]["base_skin_payload_sha256"]
            and human.sha256(manifest_path) == plan["model"]["base_skin_manifest_sha256"]
            and human.sha256(source_preflight_path)
            == plan["model"]["source_preflight_receipt_sha256"]
            and human.sha256(registration_path) == plan["model"]["registration_sha256"]
            and human.sha256(CORE_MANIFEST_PATH) == plan["model"]["core_manifest_sha256"],
            "pinned ABI 5 source inputs")
    require(human.sha256(ROOT / "Sources/myosim/myo_sim-33c89c2b.tar.gz")
            == plan["model"]["myosim_archive_sha256"], "pinned MyoSim source archive")
    base_manifest, _, nb, outer_nv, outer_ni, _, outer, outer_faces, bindings, outer_weights = (
        _load_skin(payload_path, manifest_path)
    )
    source_manifest = human.read_json(source_preflight_path)
    require(source_manifest["status"]
            == "source_reproduction_and_independent_preflight_passed"
            and source_manifest["artifacts"]["skin_payload"]["sha256"]
            == plan["model"]["base_skin_payload_sha256"],
            "current ABI 5 source preflight")
    _, _, full_vertices, full_faces = _load_solid(
        solid_path, solid_manifest_path, solid_audit_path,
    )
    outer_to_full, fixed_ids, fixed_weights = _outer_to_full(
        outer, full_vertices, outer_weights,
    )
    registration = human.read_json(registration_path)
    frame = np.asarray(base_manifest["coverage"]["source_common_frame"]
                       ["global_source_mm_to_myosim_world_m"], dtype=np.float64)
    registered_frame = np.asarray(registration["coordinate_system"]
                                  ["global_source_mm_to_myosim_world_m"], dtype=np.float64)
    require(float(np.max(np.abs(frame-registered_frame))) <= 1e-9,
            "registered bone and source solid coordinate frames")
    bones = _source_bone_records(registration, frame, bindings)
    full_world = full_vertices @ (frame[:3, :3]*1000.0).T + frame[:3, 3]
    try:
        candidate_weights, extension_metrics = source_seeded_extend(
            full_world, full_faces, fixed_ids, fixed_weights, bones,
        )
    except human.ImportError as error:
        atomic_json(output_dir / "source-seed-attempt.json", {
            "schema": "numi.human.skin-full-solid-source-seeded-attempt.v1",
            "status": "stopped_before_candidate_field_or_pose_screen",
            "plan_sha256": plan_sha,
            "registration_sha256": human.sha256(registration_path),
            "base_skin_payload_sha256": human.sha256(payload_path),
            "reason": str(error),
            "candidate_weights_written": False,
            "pose_predicates_run": False,
            "prediction_or_conditions_changed": False,
        })
        raise
    candidate_weights_f32 = candidate_weights.astype("<f4")
    require(np.array_equal(candidate_weights_f32[fixed_ids], fixed_weights.astype("<f4"))
            and float(np.max(np.abs(candidate_weights_f32.sum(axis=1)-1))) <= 1e-6,
            "Float32 source-seeded weights retain exact outer rows")
    weight_path = output_dir / "source-seeded-full-solid-weights.f32.z"
    weights_sha = _write_binary(weight_path,
                                zlib.compress(candidate_weights_f32.tobytes(), 9))

    core_manifest = human.read_json(CORE_MANIFEST_PATH)
    body_order = core_manifest["core_tree"]["body_order"]
    require(len(body_order) == 157 and nb == plan["fixed_settings"]["binding_count"],
            "pinned source core and binding count")
    import mujoco
    from myo_sim.build.compose import build_model
    model = build_model("myofullbody")
    _empty_override_pose, rest_pose, rest_metrics = _deform_lbs(
        model, body_order, (), bindings, full_vertices, candidate_weights_f32, mujoco,
    )
    source_rest = full_vertices @ (frame[:3, :3]*1000.0).T + frame[:3, 3]
    rest_error_by_vertex = np.linalg.norm(rest_pose-source_rest, axis=1)
    rest_error = float(rest_error_by_vertex.max())
    outer_rest_error = float(rest_error_by_vertex[fixed_ids].max())
    hidden = np.ones(len(full_vertices), dtype=bool)
    hidden[fixed_ids] = False
    hidden_rest_error = float(rest_error_by_vertex[hidden].max())
    if rest_error > 2e-5:
        atomic_json(output_dir / "rest-reconstruction.json", {
            "schema": "numi.human.skin-full-solid-source-seeded-rest.v1",
            "status": "failed_source_rest_gate_before_pose_screen",
            "plan_sha256": plan_sha,
            "maximum_all_vertex_error_m": rest_error,
            "maximum_outer_vertex_error_m": outer_rest_error,
            "maximum_inner_or_connector_error_m": hidden_rest_error,
            "allowed_error_m": 2e-5,
            "pose_predicates_run": False,
            "candidate_adopted": False,
        })
        raise human.ImportError(
            "source-seeded skin rest reconstruction failed: "
            f"all={rest_error} m outer={outer_rest_error} m hidden={hidden_rest_error} m"
        )

    baseline_dir = HELDOUT_DIR / "audits"
    pack_dir = HELDOUT_DIR / "packs"
    outer_face_keys = {tuple(sorted(map(int, outer_to_full[face]))) for face in outer_faces}
    require(len(outer_face_keys) == len(outer_faces)
            == plan["fixed_settings"]["outer_triangle_count"],
            "exact source outer-face subset")
    case_rows = []
    total_pairs = 0
    for case_name, pose in POSES.items():
        condition = next(row for row in plan["conditions"] if row["id"] == case_name)
        suffix = case_name.rsplit("_", 1)[1]
        pack_path = pack_dir / f"base-q_{suffix}.mrvpack"
        baseline_path = baseline_dir / f"base-q_{suffix}.json"
        require(human.sha256(pack_path) == condition["baseline_pack_sha256"]
                and human.sha256(baseline_path)
                == plan["model"]["baseline_audit_sha256"][case_name],
                f"{case_name} native baseline identity")
        baseline = human.read_json(baseline_path)
        require(baseline["exact_intersection_pairs"]
                == condition["outer_baseline_pairs"], f"{case_name} outer control pairs")
        current, _, transform_metrics = _deform_lbs(
            model, body_order, pose, bindings, full_vertices, candidate_weights_f32, mujoco,
        )
        current_f32 = current.astype("<f4")
        outer_current = current_f32[outer_to_full].astype(np.float64)
        native_outer = _native_skin_positions(pack_path, outer_faces, outer_nv)
        outer_error = float(np.max(np.linalg.norm(outer_current-native_outer, axis=1)))
        require(outer_error <= 2e-6,
                f"{case_name} source-seeded outer/native parity {outer_error}")
        quotient_vertices, quotient_faces, _ = coordinate_quotient(
            current_f32.astype(np.float64).tolist(), full_faces.tolist(),
        )
        exact = exact_embedding(quotient_vertices, quotient_faces)
        classes = dict(sorted(classify_pairs(exact["triangle_pairs"],
                                              full_faces, outer_face_keys).items()))
        pair_path = output_dir / f"{case_name}-source-seeded-intersections.json.gz"
        pair_raw = json.dumps(exact["triangle_pairs"], separators=(",", ":")).encode()
        pair_sha = _write_binary(pair_path, gzip.compress(pair_raw, mtime=0))
        vertex_path = output_dir / f"{case_name}-source-seeded-vertices.f32"
        vertex_sha = _write_binary(vertex_path, current_f32.tobytes())
        row = {
            "schema": "numi.human.skin-full-solid-source-seeded-row.v1",
            "case": case_name,
            "joint_overrides": [list(item) for item in pose],
            "baseline_outer_intersection_pair_count": int(condition["outer_baseline_pairs"]),
            "maximum_outer_to_native_position_error_m": outer_error,
            "full_solid_exact_intersection_pair_count": int(exact["count"]),
            "full_solid_intersection_pair_classes": classes,
            "candidate_topology": exact["topology"],
            "full_solid_intersection_pairs_file": pair_path.name,
            "full_solid_intersection_pairs_sha256": pair_sha,
            "full_solid_vertex_positions_sha256": vertex_sha,
            "registered_signed_volume_m3": float(np.einsum(
                "ij,ij->", current_f32[full_faces[:, 0]],
                np.cross(current_f32[full_faces[:, 1]],
                         current_f32[full_faces[:, 2]]),
            )/6.0),
            "maximum_source_joint_equality_residual": transform_metrics[
                "maximum_source_joint_equality_residual"],
            "gpu_used": False,
            "candidate_adopted": False,
            "physical_skin_mechanics": False,
            "clinical_anatomy": False,
        }
        row_path = output_dir / f"{case_name}-source-seeded.json"
        atomic_json(row_path, row)
        case_rows.append(row)
        total_pairs += int(exact["count"])
        print(json.dumps({
            "case": case_name,
            "outer_parity_max_um": round(1e6*outer_error, 3),
            "full_solid_exact_intersection_pairs": int(exact["count"]),
            "pair_classes": classes,
        }), flush=True)

    passed = all(
        row["full_solid_intersection_pair_classes"].get(
            "inner_or_connector_to_outer", 0) == 0
        and row["full_solid_intersection_pair_classes"].get(
            "inner_or_connector_to_inner", 0) == 0
        and row["full_solid_exact_intersection_pair_count"]
        == row["baseline_outer_intersection_pair_count"]
        and row["candidate_topology"]["closed_oriented_manifold_candidate"]
        for row in case_rows
    )
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "passed_source_seeded_full_solid_screen" if passed
        else "failed_source_seeded_full_solid_screen",
        "plan_sha256": plan_sha,
        "harmonic_control_plan_sha256": human.sha256(HARMONIC_PLAN_PATH),
        "full_solid_sha256": human.sha256(solid_path),
        "full_solid_manifest_sha256": human.sha256(solid_manifest_path),
        "base_skin_payload_sha256": human.sha256(payload_path),
        "base_skin_manifest_sha256": human.sha256(manifest_path),
        "registration_sha256": human.sha256(registration_path),
        "source_preflight_sha256": human.sha256(source_preflight_path),
        "source_code_sha256": {
            name: human.sha256(Path(__file__).with_name(name))
            for name in ("skin_full_solid_source_seeded_motion.py",
                         "skin_full_solid_motion.py",
                         "skin_full_solid_crossing_classes.py",
                         "skin_dual_quaternion_audit.py",
                         "skin_embeddedness_gate.py",
                         "compiled_quotient_embeddedness.py",
                         "surface_topology_audit.py",
                         "cardiac_cavity_intersections.py")
        },
        "source_rest_reconstruction_maximum_error_m": rest_error,
        "outer_source_rest_reconstruction_maximum_error_m": outer_rest_error,
        "inner_source_rest_reconstruction_maximum_error_m": hidden_rest_error,
        "rest_joint_equality_residual": rest_metrics["maximum_source_joint_equality_residual"],
        "source_seeded_extension": extension_metrics,
        "source_seeded_weight_field_sha256": extension_metrics["weight_field_sha256"],
        "source_seeded_weight_field_compressed_sha256": weights_sha,
        "case_count": len(case_rows),
        "full_solid_intersection_pair_total": total_pairs,
        "outer_native_pack_parity_maximum_error_m": max(
            row["maximum_outer_to_native_position_error_m"] for row in case_rows),
        "rows_sha256": {
            row["case"]+"-source-seeded.json": human.sha256(
                output_dir/(row["case"]+"-source-seeded.json"))
            for row in case_rows
        },
        "candidate_adopted": False,
        "native_renderer_implemented": False,
        "continuous_motion_qualified": False,
        "physical_skin_mechanics": False,
        "clinical_anatomy": False,
        "boundary": plan["candidate"]["boundary"],
    }
    atomic_json(output_dir / "summary.json", result)
    return result


def summarize_existing(output_dir: Path) -> dict[str, Any]:
    """Verify and aggregate retained rows without rerunning exact predicates."""
    output_dir = Path(output_dir).resolve()
    plan = human.read_json(PLAN_PATH)
    plan_sha = human.sha256(PLAN_PATH)
    attempt_path = ROOT / "Docs/media/skin-full-solid-motion-20261004/source-seeded-attempts/attempt-002.json"
    attempt = human.read_json(attempt_path)
    solid_path = SOLID_DIR / "bodyparts3d-full-source-skin-solid.nhsolid"
    solid_manifest_path = SOLID_DIR / "manifest.json"
    solid_audit_path = SOLID_DIR / "audit.json"
    payload_path = HELDOUT_DIR / "inputs/base/bodyparts3d-myosim-skinned-shell.nhskin"
    manifest_path = HELDOUT_DIR / "inputs/base/bodyparts3d-myosim-skinned-shell.manifest.json"
    require(attempt["status"] == "exact_pose_rows_retained_summary_not_written"
            and attempt["plan_sha256"] == plan_sha,
            "retained source-seeded pose attempt identity")
    require(human.sha256(solid_path) == plan["model"]["full_solid_sha256"]
            and human.sha256(solid_manifest_path)
            == plan["model"]["full_solid_manifest_sha256"]
            and human.sha256(solid_audit_path) == plan["model"]["full_solid_audit_sha256"]
            and human.sha256(payload_path) == plan["model"]["base_skin_payload_sha256"]
            and human.sha256(manifest_path) == plan["model"]["base_skin_manifest_sha256"]
            and human.sha256(REGISTRATION_PATH) == plan["model"]["registration_sha256"]
            and human.sha256(SOURCE_PREFLIGHT_PATH)
            == plan["model"]["source_preflight_receipt_sha256"]
            and human.sha256(CORE_MANIFEST_PATH) == plan["model"]["core_manifest_sha256"]
            and human.sha256(ROOT / "Sources/myosim/myo_sim-33c89c2b.tar.gz")
            == plan["model"]["myosim_archive_sha256"],
            "pinned source identities for retained pose rows")

    base_manifest, _, _, outer_nv, _, _, outer, outer_faces, bindings, outer_weights = (
        _load_skin(payload_path, manifest_path)
    )
    _, _, full_vertices, full_faces = _load_solid(
        solid_path, solid_manifest_path, solid_audit_path,
    )
    outer_to_full, fixed_ids, fixed_weights = _outer_to_full(
        outer, full_vertices, outer_weights,
    )
    frame = np.asarray(base_manifest["coverage"]["source_common_frame"]
                       ["global_source_mm_to_myosim_world_m"], dtype=np.float64)
    registration = human.read_json(REGISTRATION_PATH)
    registration_frame = np.asarray(registration["coordinate_system"]
                                    ["global_source_mm_to_myosim_world_m"], dtype=np.float64)
    require(float(np.max(np.abs(frame-registration_frame))) <= 1e-9,
            "registered source frame for retained source-seed solution")
    bones = _source_bone_records(registration, frame, bindings)
    full_world = full_vertices @ (frame[:3, :3]*1000.0).T + frame[:3, 3]
    weights, extension_metrics = source_seeded_extend(
        full_world, full_faces, fixed_ids, fixed_weights, bones,
    )
    weights_f32 = weights.astype("<f4")
    weight_path = output_dir / "source-seeded-full-solid-weights.f32.z"
    stored_weight_bytes = zlib.decompress(weight_path.read_bytes())
    require(stored_weight_bytes == weights_f32.tobytes(),
            "recomputed source-seeded field matches retained Float32 candidate")
    require(np.array_equal(weights_f32[fixed_ids], fixed_weights.astype("<f4")),
            "recomputed source-seeded outer rows are unchanged")

    core_manifest = human.read_json(CORE_MANIFEST_PATH)
    body_order = core_manifest["core_tree"]["body_order"]
    import mujoco
    from myo_sim.build.compose import build_model
    model = build_model("myofullbody")
    _, rest_positions, rest_metrics = _deform_lbs(
        model, body_order, (), bindings, full_vertices, weights_f32, mujoco,
    )
    source_rest = full_vertices @ (frame[:3, :3]*1000.0).T + frame[:3, 3]
    rest_error = float(np.linalg.norm(rest_positions-source_rest, axis=1).max())
    outer_rest_error = float(np.linalg.norm(
        rest_positions[fixed_ids]-source_rest[fixed_ids], axis=1).max())
    hidden = np.ones(len(full_vertices), dtype=bool)
    hidden[fixed_ids] = False
    hidden_rest_error = float(np.linalg.norm(
        rest_positions[hidden]-source_rest[hidden], axis=1).max())
    require(rest_error <= 2e-5, "retained source-seeded candidate rest gate")

    outer_face_keys = {
        tuple(sorted(map(int, outer_to_full[face]))) for face in outer_faces
    }
    case_rows = []
    for condition in plan["conditions"]:
        case = condition["id"]
        row_path = output_dir / f"{case}-source-seeded.json"
        row = human.read_json(row_path)
        pairs_path = output_dir / row["full_solid_intersection_pairs_file"]
        vertex_path = output_dir / f"{case}-source-seeded-vertices.f32"
        suffix = case.rsplit("_", 1)[1]
        pack_path = HELDOUT_DIR / "packs" / f"base-q_{suffix}.mrvpack"
        baseline_path = HELDOUT_DIR / "audits" / f"base-q_{suffix}.json"
        expected_baseline_sha = plan["model"]["baseline_audit_sha256"][case]
        require(row["schema"] == "numi.human.skin-full-solid-source-seeded-row.v1"
                and row["joint_overrides"] == condition["joint_overrides"]
                and row["baseline_outer_intersection_pair_count"]
                == condition["outer_baseline_pairs"]
                and human.sha256(pack_path) == condition["baseline_pack_sha256"]
                and human.sha256(baseline_path) == expected_baseline_sha
                and human.sha256(pairs_path)
                == row["full_solid_intersection_pairs_sha256"]
                and human.sha256(vertex_path)
                == row["full_solid_vertex_positions_sha256"],
                f"{case} retained exact row identity and controls")
        baseline = human.read_json(baseline_path)
        require(baseline["exact_intersection_pairs"]
                == condition["outer_baseline_pairs"],
                f"{case} retained outer baseline pair count")
        saved_positions = np.fromfile(vertex_path, dtype="<f4")
        require(saved_positions.size == len(full_vertices)*3,
                f"{case} retained emitted vertex-position shape")
        saved_positions = saved_positions.reshape((-1, 3))
        case_pose = tuple((int(joint), float(value))
                          for joint, value in condition["joint_overrides"])
        recomputed_positions, _, _ = _deform_lbs(
            model, body_order, case_pose, bindings, full_vertices,
            weights_f32, mujoco,
        )
        recomputed_f32 = recomputed_positions.astype("<f4")
        require(np.array_equal(saved_positions, recomputed_f32),
                f"{case} retained Float32 positions match regenerated LBS")
        outer_current = saved_positions[outer_to_full].astype(np.float64)
        native_outer = _native_skin_positions(
            pack_path, outer_faces, outer_nv,
        )
        recomputed_outer_error = float(np.max(np.linalg.norm(
            outer_current-native_outer, axis=1,
        )))
        require(recomputed_outer_error <= 2e-6
                and abs(recomputed_outer_error
                        - row["maximum_outer_to_native_position_error_m"])
                <= 1e-15,
                f"{case} recomputed outer/native parity {recomputed_outer_error}")
        with gzip.open(pairs_path, "rt", encoding="utf-8") as stream:
            pairs = json.load(stream)
        classes = dict(sorted(classify_pairs(pairs, full_faces, outer_face_keys).items()))
        require(len(pairs) == row["full_solid_exact_intersection_pair_count"]
                and classes == row["full_solid_intersection_pair_classes"],
                f"{case} retained exact-pair classification closure")
        row["retained_row_sha256"] = human.sha256(row_path)
        row["retained_positions_match_regenerated_lbs"] = True
        row["recomputed_outer_native_position_error_m"] = recomputed_outer_error
        row["pair_classes_reverified_without_predicate"] = classes
        case_rows.append(row)

    passed = all(
        row["full_solid_intersection_pair_classes"].get(
            "inner_or_connector_to_outer", 0) == 0
        and row["full_solid_intersection_pair_classes"].get(
            "inner_or_connector_to_inner", 0) == 0
        and row["full_solid_exact_intersection_pair_count"]
        == row["baseline_outer_intersection_pair_count"]
        and row["candidate_topology"]["closed_oriented_manifold_candidate"]
        for row in case_rows
    )
    predicate_sources = {
        name: human.sha256(Path(__file__).with_name(name))
        for name in ("skin_full_solid_motion.py",
                     "skin_full_solid_crossing_classes.py",
                     "skin_dual_quaternion_audit.py",
                     "skin_embeddedness_gate.py",
                     "compiled_quotient_embeddedness.py",
                     "surface_topology_audit.py",
                     "cardiac_cavity_intersections.py")
    }
    predicate_sources["skin_full_solid_source_seeded_motion.py"] = attempt[
        "predicate_source_sha256"]
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "passed_source_seeded_full_solid_screen" if passed
        else "failed_source_seeded_full_solid_screen",
        "summary_recovered_from_retained_rows": True,
        "exact_predicates_reexecuted": False,
        "plan_sha256": plan_sha,
        "harmonic_control_plan_sha256": human.sha256(HARMONIC_PLAN_PATH),
        "full_solid_sha256": human.sha256(solid_path),
        "full_solid_manifest_sha256": human.sha256(solid_manifest_path),
        "base_skin_payload_sha256": human.sha256(payload_path),
        "base_skin_manifest_sha256": human.sha256(manifest_path),
        "registration_sha256": human.sha256(REGISTRATION_PATH),
        "source_preflight_sha256": human.sha256(SOURCE_PREFLIGHT_PATH),
        "source_code_sha256": predicate_sources,
        "summary_recovery_source_sha256": human.sha256(Path(__file__)),
        "source_rest_reconstruction_maximum_error_m": rest_error,
        "outer_source_rest_reconstruction_maximum_error_m": outer_rest_error,
        "inner_source_rest_reconstruction_maximum_error_m": hidden_rest_error,
        "rest_joint_equality_residual": rest_metrics[
            "maximum_source_joint_equality_residual"],
        "source_seeded_extension": extension_metrics,
        "source_seeded_weight_field_sha256": extension_metrics["weight_field_sha256"],
        "source_seeded_weight_field_compressed_sha256": human.sha256(weight_path),
        "case_count": len(case_rows),
        "full_solid_intersection_pair_total": sum(
            row["full_solid_exact_intersection_pair_count"] for row in case_rows),
        "outer_native_pack_parity_maximum_error_m": max(
            row["maximum_outer_to_native_position_error_m"] for row in case_rows),
        "rows_sha256": {row["case"]+"-source-seeded.json": row[
            "retained_row_sha256"] for row in case_rows},
        "case_rows": case_rows,
        "candidate_adopted": False,
        "native_renderer_implemented": False,
        "continuous_motion_qualified": False,
        "physical_skin_mechanics": False,
        "clinical_anatomy": False,
        "boundary": plan["candidate"]["boundary"],
    }
    atomic_json(output_dir / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summarize-existing", action="store_true",
                        help="validate retained pose rows and recover summary without exact-predicate reruns")
    args = parser.parse_args()
    result = summarize_existing(args.output) if args.summarize_existing else audit(args.output)
    print(json.dumps({
        "status": result["status"],
        "full_solid_intersection_pair_total": result[
            "full_solid_intersection_pair_total"],
        "outer_native_pack_parity_maximum_error_m": result[
            "outer_native_pack_parity_maximum_error_m"],
    }))
    return 0 if result["status"] == "passed_source_seeded_full_solid_screen" else 2


if __name__ == "__main__":
    raise SystemExit(main())
