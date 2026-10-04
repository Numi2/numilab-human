"""CPU-only articulated embeddedness screen for the complete FJ2810 solid."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation

from . import model as human
from .compiled_quotient_embeddedness import coordinate_quotient
from .skin_dual_quaternion_audit import (
    POSES,
    _load_skin,
    _native_skin_positions,
    _pose_transforms,
)
from .surface_topology_audit import exact_embedding
from .whole_body_embeddedness import atomic_json

SCHEMA = "numi.human.skin-full-solid-motion-screen.v1"
PLAN_PATH = human.REPOSITORY_ROOT / "Docs/media/skin-full-solid-motion-20261004/plan.json"
SOLID_DIR = human.REPOSITORY_ROOT / "Docs/media/skin-full-source-solid-20260930"
HELDOUT_DIR = human.REPOSITORY_ROOT / "Docs/media/skin-weight-heldout-20261004"
SOLID_HEADER = struct.Struct("<8sIII32s32s")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise human.ImportError("skin full-solid motion screen: " + message)


def harmonic_extend(
    vertices: np.ndarray, faces: np.ndarray, fixed_ids: np.ndarray,
    fixed_weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extend fixed source weights over a connected triangle-surface graph."""
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
            "harmonic extension array shapes and indices")
    require(len(np.unique(fixed_ids)) == len(fixed_ids)
            and bool(np.isfinite(vertices).all() and np.isfinite(fixed_weights).all())
            and bool((fixed_weights >= 0).all())
            and float(np.max(np.abs(fixed_weights.sum(axis=1)-1))) <= 1e-6,
            "fixed source weights are unique, finite and normalized")
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]],
                                               faces[:, [1, 2]],
                                               faces[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]]-vertices[edges[:, 1]], axis=1)
    require(bool(np.isfinite(lengths).all() and (lengths > 0).all()),
            "source graph has positive finite edge lengths")
    conductance = 1.0/lengths
    adjacency = coo_matrix((np.r_[conductance, conductance],
                            (np.r_[edges[:, 0], edges[:, 1]],
                             np.r_[edges[:, 1], edges[:, 0]])),
                           shape=(len(vertices), len(vertices))).tocsr()
    component_count = int(connected_components(adjacency, directed=False,
                                               return_labels=False))
    require(component_count == 1, "complete source skin graph is not connected")
    fixed = np.zeros(len(vertices), dtype=bool)
    fixed[fixed_ids] = True
    unknown_ids = np.flatnonzero(~fixed)
    result = np.empty((len(vertices), fixed_weights.shape[1]), dtype=np.float64)
    result[fixed_ids] = fixed_weights
    if len(unknown_ids):
        known_ids = np.flatnonzero(fixed)
        laplacian = diags(np.asarray(adjacency.sum(axis=1)).ravel())-adjacency
        unknown_operator = laplacian[unknown_ids][:, unknown_ids].tocsc()
        rhs = -laplacian[unknown_ids][:, known_ids] @ result[known_ids]
        extension = splu(unknown_operator).solve(np.asarray(rhs))
        require(bool(np.isfinite(extension).all())
                and float(extension.min()) >= -1e-12,
                "harmonic solve produced nonfinite or materially negative weights")
        extension[extension < 0] = 0.0
        partition = extension.sum(axis=1)
        require(bool(np.isfinite(partition).all() and (partition > 1e-12).all()),
                "harmonic rows have zero or nonfinite weight mass")
        extension /= partition[:, None]
        result[unknown_ids] = extension
    fixed_partition_error = float(np.max(np.abs(fixed_weights.sum(axis=1)-1)))
    harmonic_partition_error = (float(np.max(np.abs(result[unknown_ids].sum(axis=1)-1)))
                                if len(unknown_ids) else 0.0)
    require(bool(np.isfinite(result).all() and (result >= 0).all())
            and fixed_partition_error <= 1e-6
            and harmonic_partition_error <= 1e-10
            and np.array_equal(result[fixed_ids], fixed_weights),
            "harmonic source constraints and partition")
    return result, {
        "graph_vertex_count": len(vertices), "graph_edge_count": len(edges),
        "graph_component_count": component_count,
        "fixed_outer_vertex_count": len(fixed_ids),
        "extended_inner_and_connector_vertex_count": len(unknown_ids),
        "fixed_outer_maximum_partition_error": fixed_partition_error,
        "harmonic_extension_maximum_partition_error": harmonic_partition_error,
        "minimum_weight": float(result.min()),
        "maximum_weight": float(result.max()),
    }


def _load_solid(artifact: Path, manifest_path: Path, audit_path: Path):
    manifest = human.read_json(manifest_path)
    audit = human.read_json(audit_path)
    raw = artifact.read_bytes()
    require(manifest["artifact_sha256"] == human.sha256(artifact)
            and audit["artifact_sha256"] == human.sha256(artifact)
            and audit["status"] == "passed_exact_source_and_float32_geometric_skin_solid"
            and not audit["deformation_or_native_execution"],
            "retained complete source-solid identity")
    magic, abi, nv, nf, member_sha, registration_sha = SOLID_HEADER.unpack_from(raw)
    require(magic == b"NHSOLID1" and abi == 1 and nv == 101691
            and nf == 203382
            and member_sha.hex() == manifest["source_member_sha256"]
            and registration_sha.hex() == manifest["registration_sha256"]
            and len(raw) == SOLID_HEADER.size+12*nv+12*nf,
            "NHSOLID1 complete surface header")
    vertices = np.frombuffer(raw, dtype="<f4", count=nv*3,
                             offset=SOLID_HEADER.size).reshape(nv, 3).astype(np.float64)
    faces = np.frombuffer(raw, dtype="<u4", count=nf*3,
                          offset=SOLID_HEADER.size+12*nv).reshape(nf, 3)
    return manifest, audit, vertices, faces


def _outer_to_full(outer: np.ndarray, full: np.ndarray, weights: np.ndarray):
    lookup = {tuple(row): index for index, row in enumerate(full)}
    try:
        mapping = np.fromiter((lookup[tuple(row.astype("<f4"))] for row in outer),
                              dtype=np.int64, count=len(outer))
    except KeyError as error:
        raise human.ImportError(
            "skin full-solid motion screen: outer shell vertex is absent from full source solid"
        ) from error
    unique_outer, first, inverse = np.unique(
        outer.astype("<f4"), axis=0, return_index=True, return_inverse=True,
    )
    require(len(unique_outer) == 54663 and np.array_equal(mapping, mapping[first][inverse]),
            "exact outer-to-full coordinate quotient")
    require(np.array_equal(weights, weights[first][inverse]),
            "exact outer seam source-weight equality")
    fixed_ids = mapping[first]
    require(len(np.unique(fixed_ids)) == len(fixed_ids),
            "unique fixed full-solid vertex identities")
    return mapping, fixed_ids, weights[first]


def _inertial_rotations(model: Any, data: Any, body_ids: np.ndarray) -> Rotation:
    body = Rotation.from_quat(np.asarray(data.xquat[body_ids][:, [1, 2, 3, 0]],
                                         dtype=np.float64))
    inertial = Rotation.from_quat(np.asarray(model.body_iquat[body_ids][:, [1, 2, 3, 0]],
                                             dtype=np.float64))
    return body*inertial


def _deform_lbs(
    model: Any, body_order: list[str], pose: tuple[tuple[int, float], ...],
    bindings: list[tuple], vertices: np.ndarray, weights: np.ndarray,
    mujoco: Any,
):
    rest_data, current_data, equality = _pose_transforms(model, pose, mujoco)
    body_ids = []
    translations, quaternions, scales = [], [], []
    for row in bindings:
        core_id = int(row[0])
        require(0 <= core_id < len(body_order), "skin body index within pinned order")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    body_order[core_id])
        require(body_id >= 0, "skin body maps to an articulated source body")
        body_ids.append(body_id)
        translations.append(row[1:4])
        quaternions.append(row[4:8])
        scales.append(row[8])
    body_ids = np.asarray(body_ids, dtype=np.int32)
    local_translations = np.asarray(translations, dtype=np.float64)
    local_rotations = Rotation.from_quat(np.asarray(quaternions, dtype=np.float64))
    local_scales = np.asarray(scales, dtype=np.float64)
    rest_positions = np.asarray(rest_data.xipos[body_ids], dtype=np.float64)
    current_positions = np.asarray(current_data.xipos[body_ids], dtype=np.float64)
    rest_rotations = _inertial_rotations(model, rest_data, body_ids)
    current_rotations = _inertial_rotations(model, current_data, body_ids)

    posed = np.zeros_like(vertices)
    rested = np.zeros_like(vertices)
    for influence in range(len(bindings)):
        local_vertices = (local_translations[influence]
                          + local_scales[influence]
                          * local_rotations[influence].apply(vertices))
        current_vertices = (current_positions[influence]
                            + current_rotations[influence].apply(local_vertices))
        rest_vertices = (rest_positions[influence]
                         + rest_rotations[influence].apply(local_vertices))
        posed += weights[:, influence, None]*current_vertices
        rested += weights[:, influence, None]*rest_vertices
    return posed, rested, {
        "maximum_source_joint_equality_residual": equality["maximum_absolute_residual"],
        "projected_equality_count": equality["measured_joint_equalities"],
    }


def _write_binary(path: Path, raw: bytes) -> str:
    require(not path.is_symlink(), "candidate geometry output is not a symlink")
    if path.exists():
        require(path.read_bytes() == raw, "candidate geometry output identity")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def audit(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = human.REPOSITORY_ROOT
    plan = human.read_json(PLAN_PATH)
    solid_path = SOLID_DIR/"bodyparts3d-full-source-skin-solid.nhsolid"
    solid_manifest_path = SOLID_DIR/"manifest.json"
    solid_audit_path = SOLID_DIR/"audit.json"
    base_payload_path = HELDOUT_DIR/"inputs/base/bodyparts3d-myosim-skinned-shell.nhskin"
    base_manifest_path = HELDOUT_DIR/"inputs/base/bodyparts3d-myosim-skinned-shell.manifest.json"
    source_preflight_path = root/"Docs/media/skin-source-fit-recovery-20261004/receipt.json"
    core_manifest_path = HELDOUT_DIR/"inputs/native/myosim-fullbody-reference.manifest.json"
    baseline_dir = HELDOUT_DIR/"audits"
    pack_dir = HELDOUT_DIR/"packs"
    require(human.sha256(solid_path) == plan["model"]["full_solid_sha256"]
            and human.sha256(solid_manifest_path) == plan["model"]["full_solid_manifest_sha256"]
            and human.sha256(solid_audit_path) == plan["model"]["full_solid_audit_sha256"],
            "preregistered source solid hashes")
    require(human.sha256(base_payload_path) == plan["model"]["base_skin_payload_sha256"]
            and human.sha256(base_manifest_path) == plan["model"]["base_skin_manifest_sha256"]
            and human.sha256(source_preflight_path)
            == plan["model"]["source_preflight_receipt_sha256"]
            and human.sha256(core_manifest_path) == plan["model"]["core_manifest_sha256"],
            "preregistered outer payload and source preflight")
    require(human.sha256(root/"Sources/myosim/myo_sim-33c89c2b.tar.gz")
            == plan["model"]["myosim_archive_sha256"], "preregistered MyoSim archive")
    source_preflight = human.read_json(source_preflight_path)
    require(source_preflight["status"] == "source_reproduction_and_independent_preflight_passed"
            and source_preflight["artifacts"]["skin_payload"]["sha256"]
            == plan["model"]["base_skin_payload_sha256"],
            "current ABI5 source preflight")
    base_manifest, _, nb, outer_nv, outer_ni, _, outer, outer_faces, bindings, outer_weights = (
        _load_skin(base_payload_path, base_manifest_path)
    )
    solid_manifest, _, full_vertices, full_faces = _load_solid(
        solid_path, solid_manifest_path, solid_audit_path,
    )
    require(base_manifest["source"]["skin"]["member_sha256"]
            == solid_manifest["source_member_sha256"]
            == plan["model"].get("source_skin_member_sha256",
                                  "682f402206f15592acdeaae8ffb6b34c3e5c3267fa4685e63d2e4920ef2a80e0")
            and base_manifest["source"]["bodyparts"]["archives"][0]["sha256"]
            == solid_manifest["source_archive_sha256"],
            "shared BodyParts3D source member and archive")
    require(nb == plan["fixed_settings"]["binding_count"]
            and outer_nv == plan["fixed_settings"]["outer_vertex_count"]
            and outer_ni//3 == plan["fixed_settings"]["outer_triangle_count"]
            and len(full_vertices) == plan["fixed_settings"]["full_unique_vertex_count"]
            and len(full_faces) == plan["fixed_settings"]["full_triangle_count"],
            "preregistered full and outer source geometry counts")
    outer_to_full, fixed_ids, fixed_weights = _outer_to_full(
        outer, full_vertices, outer_weights,
    )
    full_weights, extension_metrics = harmonic_extend(
        full_vertices, full_faces, fixed_ids, fixed_weights,
    )
    full_weights_f32 = full_weights.astype("<f4")
    require(float(np.max(np.abs(full_weights_f32.sum(axis=1)-1))) <= 1.0e-6
            and np.array_equal(full_weights_f32[fixed_ids], fixed_weights.astype("<f4")),
            "Float32 full-surface weights and fixed outer rows")
    weights_bytes = zlib.compress(full_weights_f32.tobytes(), level=9)

    core_manifest = human.read_json(core_manifest_path)
    body_order = core_manifest["core_tree"]["body_order"]
    require(len(body_order) == 157, "pinned source core body order")
    import mujoco
    from myo_sim.build.compose import build_model

    model = build_model("myofullbody")
    # Use the explicit qpos0/rest-data reconstruction returned second. The
    # first value is the empty-override pose-data result, which can differ
    # slightly from qpos0 in constrained source models.
    _empty_override_pose, rest_pose, rest_transform_metrics = _deform_lbs(
        model, body_order, (), bindings, full_vertices, full_weights_f32, mujoco,
    )
    frame = np.asarray(base_manifest["coverage"]["source_common_frame"]
                       ["global_source_mm_to_myosim_world_m"], dtype=np.float64)
    registered_source = full_vertices @ (frame[:3, :3]*1000.0).T + frame[:3, 3]
    rest_error_per_vertex = np.linalg.norm(rest_pose-registered_source, axis=1)
    rest_error = float(rest_error_per_vertex.max())
    outer_rest_error = float(rest_error_per_vertex[fixed_ids].max())
    inner_rest_error = float(rest_error_per_vertex[~np.isin(
        np.arange(len(full_vertices)), fixed_ids)].max())
    if rest_error > 2.0e-5:
        atomic_json(output_dir/"rest-reconstruction-attempt.json", {
            "schema": "numi.human.skin-full-solid-rest-attempt.v1",
            "status": "failed_source_rest_reconstruction_before_pose_screen",
            "plan_sha256": human.sha256(PLAN_PATH),
            "full_solid_sha256": human.sha256(solid_path),
            "base_skin_payload_sha256": human.sha256(base_payload_path),
            "maximum_all_vertex_error_m": rest_error,
            "maximum_outer_fixed_vertex_error_m": outer_rest_error,
            "maximum_inner_or_connector_extension_error_m": inner_rest_error,
            "allowed_error_m": 2.0e-5,
            "pose_predicates_run": False,
            "candidate_adopted": False,
            "interpretation": "The source LBS binding frame must first reproduce the exact registered rest surface. The harmonic outer-weight extension did not pass this gate; no pose result is inferred.",
        })
        raise human.ImportError(
            "skin full-solid motion screen: full source-rest reconstruction "
            f"all={rest_error} m outer={outer_rest_error} m "
            f"inner_or_connector={inner_rest_error} m; allowed=2e-5 m"
        )

    weights_sha = _write_binary(output_dir/"full-solid-weights.f32.z", weights_bytes)
    case_rows = []
    pair_total = 0
    for case, pose in POSES.items():
        condition = next(row for row in plan["conditions"] if row["id"] == case)
        suffix = case.rsplit("_", 1)[1]
        pack_path = pack_dir/f"base-q_{suffix}.mrvpack"
        baseline_path = baseline_dir/f"base-q_{suffix}.json"
        require(human.sha256(pack_path) == condition["baseline_pack_sha256"]
                and human.sha256(baseline_path)
                == plan["model"]["baseline_audit_sha256"][case],
                f"{case} native control identity")
        baseline_receipt = human.read_json(baseline_path)
        require(baseline_receipt["exact_intersection_pairs"]
                == condition["outer_baseline_pairs"], f"{case} outer control count")
        current, _, transform_metrics = _deform_lbs(
            model, body_order, pose, bindings, full_vertices, full_weights_f32, mujoco,
        )
        current_f32 = current.astype("<f4")
        outer_current = current_f32[outer_to_full].astype(np.float64)
        native_outer = _native_skin_positions(pack_path, outer_faces, outer_nv)
        outer_error = float(np.max(np.linalg.norm(outer_current-native_outer, axis=1)))
        require(outer_error <= 2.0e-6,
                f"{case} full-solid outer field differs from native LBS: {outer_error}")
        quotient_vertices, quotient_faces, _ = coordinate_quotient(
            current_f32.astype(np.float64).tolist(), full_faces.tolist(),
        )
        exact = exact_embedding(quotient_vertices, quotient_faces)
        pairs = exact["triangle_pairs"]
        pairs_path = output_dir/f"{case}-intersections.json.gz"
        pairs_raw = json.dumps(pairs, separators=(",", ":")).encode()
        pairs_sha = _write_binary(pairs_path, gzip.compress(pairs_raw, mtime=0))
        position_sha = _write_binary(
            output_dir/f"{case}-vertices.f32", current_f32.tobytes(),
        )
        row = {
            "schema": "numi.human.skin-full-solid-motion-screen-row.v1",
            "case": case,
            "joint_overrides": [list(item) for item in pose],
            "baseline_pack_sha256": human.sha256(pack_path),
            "baseline_outer_intersection_pair_count": condition["outer_baseline_pairs"],
            "maximum_outer_to_native_position_error_m": outer_error,
            "maximum_source_joint_equality_residual": transform_metrics[
                "maximum_source_joint_equality_residual"],
            "full_solid_exact_intersection_pair_count": exact["count"],
            "full_solid_intersection_pairs_sha256": pairs_sha,
            "full_solid_intersection_pairs_file": pairs_path.name,
            "full_solid_vertex_positions_sha256": position_sha,
            "candidate_topology": exact["topology"],
            "registered_signed_volume_m3": float(np.einsum(
                "ij,ij->", current_f32[full_faces[:, 0]],
                np.cross(current_f32[full_faces[:, 1]],
                         current_f32[full_faces[:, 2]]),
            )/6.0),
            "gpu_used": False,
            "physical_skin_mechanics": False,
            "clinical_anatomy": False,
        }
        atomic_json(output_dir/f"{case}.json", row)
        case_rows.append(row)
        pair_total += exact["count"]
        print(json.dumps({"case": case,
                          "outer_parity_max_um": round(1e6*outer_error, 3),
                          "full_solid_intersection_pairs": exact["count"],
                          "closed_manifold": exact["topology"][
                              "closed_oriented_manifold_candidate"]}), flush=True)

    passed = all(row["full_solid_exact_intersection_pair_count"] == 0
                 and row["candidate_topology"]["closed_oriented_manifold_candidate"]
                 for row in case_rows)
    result = {
        "schema": SCHEMA,
        "status": "passed_sampled_full_solid_embeddedness" if passed
        else "failed_sampled_full_solid_embeddedness",
        "plan_sha256": human.sha256(PLAN_PATH),
        "full_solid_sha256": human.sha256(solid_path),
        "full_solid_manifest_sha256": human.sha256(solid_manifest_path),
        "base_skin_payload_sha256": human.sha256(base_payload_path),
        "base_skin_manifest_sha256": human.sha256(base_manifest_path),
        "source_preflight_sha256": human.sha256(source_preflight_path),
        "candidate_source_sha256": human.sha256(Path(__file__)),
        "predicate_source_sha256": {
            name: human.sha256(Path(__file__).with_name(name)) for name in (
                "skin_full_solid_motion.py", "skin_dual_quaternion_audit.py",
                "skin_embeddedness_gate.py", "compiled_quotient_embeddedness.py",
                "surface_topology_audit.py", "cardiac_cavity_intersections.py",
            )
        },
        "source_rest_reconstruction_maximum_error_m": rest_error,
        "outer_source_rest_reconstruction_maximum_error_m": outer_rest_error,
        "inner_source_rest_reconstruction_maximum_error_m": inner_rest_error,
        "rest_joint_equality_residual": rest_transform_metrics[
            "maximum_source_joint_equality_residual"],
        "harmonic_extension": extension_metrics,
        "full_solid_weight_field_sha256": hashlib.sha256(
            full_weights_f32.tobytes()).hexdigest(),
        "full_solid_weight_field_compressed_sha256": weights_sha,
        "case_count": len(case_rows),
        "full_solid_intersection_pair_total": pair_total,
        "outer_native_pack_parity_maximum_error_m": max(
            row["maximum_outer_to_native_position_error_m"] for row in case_rows),
        "rows_sha256": {row["case"]+".json": human.sha256(output_dir/(row["case"]+".json"))
                        for row in case_rows},
        "candidate_adopted": False,
        "native_renderer_implemented": False,
        "continuous_motion_qualified": False,
        "physical_skin_mechanics": False,
        "clinical_anatomy": False,
        "boundary": plan["boundary"],
    }
    atomic_json(output_dir/"summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output)
    print(json.dumps({"status": result["status"],
                      "full_solid_intersection_pair_total": result[
                          "full_solid_intersection_pair_total"],
                      "outer_native_pack_parity_maximum_error_m": result[
                          "outer_native_pack_parity_maximum_error_m"]}))
    return 0 if result["status"] == "passed_sampled_full_solid_embeddedness" else 2


if __name__ == "__main__":
    raise SystemExit(main())
