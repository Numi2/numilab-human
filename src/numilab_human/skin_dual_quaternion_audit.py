"""CPU-only dual-quaternion screen for the retained ABI5 source skin.

This tests an alternate visual deformation formulation. It does not qualify
the native renderer, continuous motion, tissue mechanics, shell closure, or
clinical anatomy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from . import model as human
from .compiled_quotient_embeddedness import coordinate_quotient
from .joint_constraint_consistency import source_equality_projection_oracle
from .skin_embeddedness_gate import _source_mesh
from .surface_topology_audit import exact_embedding
from .torso_anatomy_audit import _pack_sections
from .upper_limb_pose_audit import _pose_qpos
from .whole_body_embeddedness import atomic_json

SCHEMA = "numi.human.skin-dual-quaternion-screen.v1"
PLAN_PATH = human.REPOSITORY_ROOT / "Docs/media/skin-dual-quaternion-20261004/plan.json"
ARTIFACT_ROOT = human.REPOSITORY_ROOT / "Docs/media/skin-weight-heldout-20261004"
POSES = {
    "bilateral_knee_q_0p4": ((106, 0.4), (120, 0.4)),
    "bilateral_knee_q_0p8": ((106, 0.8), (120, 0.8)),
    "bilateral_knee_q_1p2": ((106, 1.2), (120, 1.2)),
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise human.ImportError("skin dual-quaternion audit: " + message)


def quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for XYZW quaternions, supporting broadcast arrays."""
    lx, ly, lz, lw = np.moveaxis(np.asarray(left, dtype=np.float64), -1, 0)
    rx, ry, rz, rw = np.moveaxis(np.asarray(right, dtype=np.float64), -1, 0)
    return np.stack((lw*rx + lx*rw + ly*rz - lz*ry,
                     lw*ry - lx*rz + ly*rw + lz*rx,
                     lw*rz + lx*ry - ly*rx + lz*rw,
                     lw*rw - lx*rx - ly*ry - lz*rz), axis=-1)


def blend_dual_quaternion_points(
    points: np.ndarray, weights: np.ndarray, rotations_xyzw: np.ndarray,
    translations: np.ndarray,
) -> np.ndarray:
    """Blend rigid transforms with hemisphere alignment and apply them."""
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    rotations = np.asarray(rotations_xyzw, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    require(points.ndim == 2 and points.shape[1] == 3
            and weights.ndim == 2 and weights.shape[0] == len(points)
            and rotations.shape == (weights.shape[1], 4)
            and translations.shape == (weights.shape[1], 3), "dual-quaternion shapes")
    require(bool(np.isfinite(points).all() and np.isfinite(weights).all()
                 and np.isfinite(rotations).all() and np.isfinite(translations).all())
            and bool((weights >= 0).all())
            and float(np.max(np.abs(weights.sum(axis=1)-1))) <= 1.0e-6,
            "dual-quaternion finite normalized source weights")

    rotation_norm = np.linalg.norm(rotations, axis=1)
    require(bool((rotation_norm > 1.0e-12).all()), "nonzero real transform quaternions")
    rotations = rotations / rotation_norm[:, None]
    translation_quaternions = np.column_stack((translations, np.zeros(len(translations))))
    dual = 0.5 * quat_multiply_xyzw(translation_quaternions, rotations)

    reference = rotations[np.argmax(weights, axis=1)]
    real_sum = np.zeros((len(points), 4), dtype=np.float64)
    dual_sum = np.zeros_like(real_sum)
    for influence in range(weights.shape[1]):
        sign = np.where(np.sum(reference*rotations[influence], axis=1) < 0.0, -1.0, 1.0)
        factor = (weights[:, influence]*sign)[:, None]
        real_sum += factor*rotations[influence]
        dual_sum += factor*dual[influence]

    magnitude = np.linalg.norm(real_sum, axis=1)
    require(bool(np.isfinite(magnitude).all() and (magnitude > 1.0e-12).all()),
            "dual-quaternion blend has nonzero real part")
    real = real_sum/magnitude[:, None]
    dual_normalized = dual_sum/magnitude[:, None]
    dual_normalized -= real*np.sum(real*dual_normalized, axis=1)[:, None]
    conjugate = real.copy()
    conjugate[:, :3] *= -1.0
    displacement = 2.0*quat_multiply_xyzw(dual_normalized, conjugate)[:, :3]
    return Rotation.from_quat(real).apply(points)+displacement


def _load_skin(payload: Path, manifest_path: Path):
    manifest = human.read_json(manifest_path)
    nb, nv, ni, fingerprint, vertices, faces = _source_mesh(payload, manifest)
    raw = payload.read_bytes()
    bindings = [struct.unpack_from("<I8f", raw, 60+36*i) for i in range(nb)]
    weight_offset = 60+36*nb+56*nv+4*ni
    weights = np.frombuffer(raw, dtype="<f4", count=nv*nb,
                            offset=weight_offset).reshape(nv, nb).astype(np.float64)
    require(float(np.max(np.abs(weights.sum(axis=1)-1))) <= 1.0e-6,
            "ABI5 source weights partition unity")
    return manifest, raw, nb, nv, ni, fingerprint, vertices, faces, bindings, weights


def _native_skin_positions(pack_path: Path, faces: np.ndarray,
                           nv: int) -> np.ndarray:
    sections = _pack_sections(pack_path)
    require(all(kind in sections and sections[kind][2] == stride
                and len(sections[kind][0]) == sections[kind][1]*stride
                for kind, stride in ((2, 80), (3, 4), (4, 64), (5, 80))),
            "pinned native visual pack layout")
    packed_vertices = np.frombuffer(sections[2][0], "<f4").reshape(-1, 20)
    packed_indices = np.frombuffer(sections[3][0], "<u4")
    primitives = np.frombuffer(sections[4][0], "<u4").reshape(-1, 16)
    selected = primitives[primitives[:, 4] == 51007]
    require(len(selected) == 1, "one native ABI5 skin surface")
    first, count, _, _ = map(int, selected[0, :4])
    require(count == faces.size and first+count <= len(packed_indices),
            "native skin face count")
    emitted = packed_indices[first:first+count]
    base = int(emitted.min())
    require(base+nv <= len(packed_vertices)
            and np.array_equal(emitted-base, faces.ravel()),
            "native source face order")
    positions = packed_vertices[base:base+nv, :3].astype(np.float64)
    require(bool(np.isfinite(positions).all()), "finite native skin vertices")
    return positions


def _pose_transforms(model: Any, pose: tuple[tuple[int, float], ...],
                     mujoco: Any):
    rest_data = mujoco.MjData(model)
    rest_data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, rest_data)
    pose_data = mujoco.MjData(model)
    pose_data.qpos[:] = _pose_qpos(model, pose, mujoco, np)[0]
    mujoco.mj_forward(model, pose_data)
    equality = source_equality_projection_oracle(model, pose_data, mujoco, 1.0e-9)
    require(equality["passed"], "complete source joint-equality pose projection")
    return rest_data, pose_data, equality


def _positions_for_condition(
    model: Any, body_order: list[str], pose: tuple[tuple[int, float], ...],
    bindings: list[tuple], source_vertices: np.ndarray, weights: np.ndarray,
    mujoco: Any,
):
    rest_data, current_data, equality = _pose_transforms(model, pose, mujoco)
    body_ids = []
    binding_translations, binding_quaternions, binding_scales = [], [], []
    for record in bindings:
        core_body_id = int(record[0])
        require(0 <= core_body_id < len(body_order), "binding core-body range")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                    body_order[core_body_id])
        require(body_id >= 0, "source model body name exists")
        body_ids.append(body_id)
        binding_translations.append(record[1:4])
        binding_quaternions.append(record[4:8])
        binding_scales.append(record[8])

    body_ids = np.asarray(body_ids, dtype=np.int32)
    local_translation = np.asarray(binding_translations, dtype=np.float64)
    local_rotation = Rotation.from_quat(np.asarray(binding_quaternions, dtype=np.float64))
    local_scale = np.asarray(binding_scales, dtype=np.float64)
    require(bool(np.isfinite(local_translation).all() and np.isfinite(local_scale).all()
                 and (local_scale > 0).all()), "source binding transforms")
    body_rest_position = np.asarray(rest_data.xipos[body_ids], dtype=np.float64)
    body_current_position = np.asarray(current_data.xipos[body_ids], dtype=np.float64)
    body_inertial_rotation = Rotation.from_quat(
        np.asarray(model.body_iquat[body_ids][:, [1, 2, 3, 0]], dtype=np.float64)
    )
    body_rest_rotation = (
        Rotation.from_quat(np.asarray(rest_data.xquat[body_ids][:, [1, 2, 3, 0]], dtype=np.float64))
        * body_inertial_rotation
    ).as_quat()
    body_current_rotation = (
        Rotation.from_quat(np.asarray(current_data.xquat[body_ids][:, [1, 2, 3, 0]], dtype=np.float64))
        * body_inertial_rotation
    ).as_quat()
    relative_rotation = (Rotation.from_quat(body_current_rotation)
                         * Rotation.from_quat(body_rest_rotation).inv())
    relative_quaternion = relative_rotation.as_quat()
    relative_translation = body_current_position-relative_rotation.apply(body_rest_position)

    direct_lbs = np.zeros_like(source_vertices)
    rest_per_binding = []
    for influence in range(len(bindings)):
        local_vertices = (local_translation[influence]
                          + local_scale[influence]
                          * local_rotation[influence].apply(source_vertices))
        rest_vertices = body_rest_position[influence] + Rotation.from_quat(
            body_rest_rotation[influence]).apply(local_vertices)
        current_vertices = body_current_position[influence] + Rotation.from_quat(
            body_current_rotation[influence]).apply(local_vertices)
        rest_per_binding.append(rest_vertices)
        direct_lbs += weights[:, influence, None]*current_vertices
    rest_per_binding = np.stack(rest_per_binding, axis=1)
    rest_world = np.einsum("vb,vbc->vc", weights, rest_per_binding, optimize=True)
    maximum_rest_binding_spread = float(np.max(np.linalg.norm(
        rest_per_binding-rest_world[:, None, :], axis=2)))
    candidate = blend_dual_quaternion_points(
        rest_world, weights, relative_quaternion, relative_translation,
    )
    return direct_lbs, candidate.astype("<f4").astype(np.float64), {
        "maximum_rest_binding_spread_m": maximum_rest_binding_spread,
        "maximum_source_joint_equality_residual": equality["maximum_absolute_residual"],
        "projected_equality_count": equality["measured_joint_equalities"],
    }


def audit(
    payload: Path, manifest_path: Path, pack_dir: Path, baseline_audit_dir: Path,
    core_manifest_path: Path, source_preflight_path: Path, output_dir: Path,
) -> dict[str, Any]:
    paths = [Path(value).resolve() for value in (
        payload, manifest_path, pack_dir, baseline_audit_dir, core_manifest_path,
        source_preflight_path, PLAN_PATH,
    )]
    payload, manifest_path, pack_dir, baseline_audit_dir, core_manifest_path, source_preflight_path, plan_path = paths
    require(payload.is_file() and not payload.is_symlink()
            and manifest_path.is_file() and not manifest_path.is_symlink(),
            "retained ABI5 payload and manifest")
    plan = human.read_json(plan_path)
    source_preflight = human.read_json(source_preflight_path)
    prior_evidence_manifest_path = ARTIFACT_ROOT / "evidence-manifest.json"
    prior_evidence_manifest = human.read_json(prior_evidence_manifest_path)
    require(source_preflight["status"] == "source_reproduction_and_independent_preflight_passed"
            and source_preflight["artifacts"]["skin_payload"]["sha256"] == human.sha256(payload)
            and source_preflight["independent_preflight"]["maximum_rest_reconstruction_error_m"] <= 2e-5,
            "pinned source preflight admission")
    require(human.sha256(payload) == plan["model"]["base_payload_sha256"]
            and human.sha256(manifest_path) == plan["model"]["base_manifest_sha256"],
            "preregistered skin payload identity")
    payload_manifest = human.read_json(manifest_path)
    require(payload_manifest["source"]["registration"]["sha256"]
            == plan["model"]["source_registration_sha256"]
            and source_preflight["source"]["registration_sha256"]
            == plan["model"]["source_registration_sha256"]
            and payload_manifest["source"]["skin"]["member_sha256"]
            == plan["model"]["source_skin_member_sha256"],
            "preregistered registration and atlas skin source")
    require(human.sha256(core_manifest_path) == plan["model"]["source_core_manifest_sha256"]
            and human.sha256(source_preflight_path)
            == plan["model"]["source_preflight_receipt_sha256"],
            "preregistered source-model records")
    rigid = source_preflight["artifacts"]["rigid"]
    rigid_path = human.REPOSITORY_ROOT / rigid["path"]
    require(human.sha256(rigid_path) == plan["model"]["source_core_rigid_sha256"]
            and rigid["sha256"] == plan["model"]["source_core_rigid_sha256"],
            "preregistered source core rigid")
    require(human.sha256(human.REPOSITORY_ROOT / "Sources/myosim/myo_sim-33c89c2b.tar.gz")
            == plan["model"]["source_myo_sim_archive_sha256"],
            "preregistered source archive")

    _, _, nb, nv, ni, _, vertices, faces, bindings, weights = _load_skin(
        payload, manifest_path,
    )
    require(nb == plan["fixed_settings"]["binding_count"]
            and nv == plan["fixed_settings"]["skin_vertex_count"]
            and ni//3 == plan["fixed_settings"]["triangle_count"],
            "preregistered ABI5 geometry counts")
    core_manifest = human.read_json(core_manifest_path)
    body_order = core_manifest["core_tree"]["body_order"]
    require(len(body_order) == 157, "pinned source core body order")

    import mujoco
    from myo_sim.build.compose import build_model

    model = build_model("myofullbody")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = []
    baseline_pairs_total = 0
    candidate_pairs_total = 0
    new_pairs_total = 0
    for case, pose in POSES.items():
        condition = next(row for row in plan["conditions"] if row["id"] == case)
        if case.endswith("0p4"):
            pack_path = pack_dir / "base-q_0p4.mrvpack"
            baseline_path = baseline_audit_dir / "base-q_0p4.json"
        elif case.endswith("0p8"):
            pack_path = pack_dir / "base-q_0p8.mrvpack"
            baseline_path = baseline_audit_dir / "base-q_0p8.json"
        else:
            pack_path = pack_dir / "base-q_1p2.mrvpack"
            baseline_path = baseline_audit_dir / "base-q_1p2.json"
        baseline_receipt = human.read_json(baseline_path)
        require(human.sha256(baseline_path)
                == plan["model"]["baseline_audit_sha256"][case]
                and baseline_receipt["native_pack_sha256"] == condition["baseline_pack_sha256"],
                f"{case} baseline audit identity")
        require(pack_path.is_file() and not pack_path.is_symlink()
                and human.sha256(pack_path) == condition["baseline_pack_sha256"],
                f"{case} native control pack identity")
        control_vertices = _native_skin_positions(pack_path, faces, nv)
        lbs_vertices, candidate_vertices, transform_metrics = _positions_for_condition(
            model, body_order, pose, bindings, vertices, weights, mujoco,
        )
        parity_error = float(np.max(np.linalg.norm(lbs_vertices-control_vertices, axis=1)))
        require(parity_error <= 2.0e-6,
                f"{case} reconstructed LBS differs from retained native control: {parity_error}")
        require(transform_metrics["maximum_rest_binding_spread_m"] <= 2.0e-5,
                f"{case} per-binding rest transforms do not describe one rest shell")
        baseline_pair_rows = baseline_receipt.get("triangle_pairs")
        if baseline_pair_rows is None:
            if condition["baseline_pairs"] == 0:
                baseline_pair_rows = []
            else:
                complete_path = baseline_audit_dir / "complete-source-q_1p2-pairs.json"
                complete_record = prior_evidence_manifest["files"][
                    "audits/complete-source-q_1p2-pairs.json"
                ]
                require(human.sha256(complete_path) == complete_record["sha256"],
                        "complete baseline pair-set evidence identity")
                complete = human.read_json(
                    complete_path
                )
                matches = [item for item in complete["cases"]
                           if item["case"] == "skin-flexion-right"
                           and item["native_pack_sha256"] == condition["baseline_pack_sha256"]]
                require(len(matches) == 1, f"{case} complete baseline pair identity")
                baseline_pair_rows = matches[0]["exact_intersection_pairs"]
        require(len(baseline_pair_rows) == condition["baseline_pairs"],
                f"{case} baseline exact-pair count")
        quotient_vertices, quotient_faces, _ = coordinate_quotient(
            candidate_vertices.tolist(), faces.tolist(),
        )
        exact = exact_embedding(quotient_vertices, quotient_faces)
        baseline_pairs = {tuple(map(int, pair)) for pair in baseline_pair_rows}
        candidate_pairs = {tuple(map(int, pair)) for pair in exact["triangle_pairs"]}
        new_pairs = sorted(candidate_pairs-baseline_pairs)
        removed_pairs = sorted(baseline_pairs-candidate_pairs)
        row = {
            "schema": "numi.human.skin-dual-quaternion-screen-row.v1",
            "case": case,
            "joint_overrides": [list(item) for item in pose],
            "baseline_native_pack_sha256": human.sha256(pack_path),
            "baseline_receipt_sha256": human.sha256(baseline_path),
            "baseline_lbs_exact_intersection_pairs": len(baseline_pairs),
            "candidate_dq_exact_intersection_pairs": exact["count"],
            "new_intersecting_face_pairs": [list(pair) for pair in new_pairs],
            "removed_intersecting_face_pairs": [list(pair) for pair in removed_pairs],
            "candidate_intersecting_face_pairs": exact["triangle_pairs"],
            "maximum_lbs_to_native_position_error_m": parity_error,
            "maximum_rest_binding_spread_m": transform_metrics["maximum_rest_binding_spread_m"],
            "maximum_source_joint_equality_residual": transform_metrics["maximum_source_joint_equality_residual"],
            "candidate_geometry_sha256": hashlib.sha256(
                candidate_vertices.astype("<f4").tobytes()+faces.astype("<u4").tobytes()
            ).hexdigest(),
            "candidate_topology": exact["topology"],
            "gpu_used": False,
            "clinical_anatomy": False,
            "physical_skin_mechanics": False,
        }
        atomic_json(output_dir / f"{case}.json", row)
        candidate_rows.append(row)
        baseline_pairs_total += len(baseline_pairs)
        candidate_pairs_total += exact["count"]
        new_pairs_total += len(new_pairs)
        print(json.dumps({"case": case,
                          "lbs_to_native_max_mm": round(1000*parity_error, 6),
                          "baseline_pairs": len(baseline_pairs),
                          "dq_pairs": exact["count"],
                          "new_pairs": len(new_pairs)}), flush=True)

    no_regression = new_pairs_total == 0 and all(
        row["candidate_dq_exact_intersection_pairs"]
        <= row["baseline_lbs_exact_intersection_pairs"] for row in candidate_rows
    )
    result = {
        "schema": SCHEMA,
        "status": "passed_sampled_geometric_nonregression" if no_regression
        else "failed_sampled_geometric_nonregression",
        "plan_sha256": human.sha256(plan_path),
        "payload_sha256": human.sha256(payload),
        "payload_manifest_sha256": human.sha256(manifest_path),
        "source_preflight_receipt_sha256": human.sha256(source_preflight_path),
        "prior_heldout_evidence_manifest_sha256": human.sha256(prior_evidence_manifest_path),
        "core_manifest_sha256": human.sha256(core_manifest_path),
        "candidate_code_sha256": human.sha256(Path(__file__)),
        "predicate_source_sha256": {
            name: human.sha256(Path(__file__).with_name(name)) for name in (
                "skin_dual_quaternion_audit.py", "skin_embeddedness_gate.py",
                "compiled_quotient_embeddedness.py", "surface_topology_audit.py",
                "cardiac_cavity_intersections.py",
            )
        },
        "case_count": len(candidate_rows),
        "baseline_intersection_pairs": baseline_pairs_total,
        "candidate_intersection_pairs": candidate_pairs_total,
        "new_intersection_pair_count": new_pairs_total,
        "rows_sha256": {row["case"]+".json": human.sha256(output_dir/(row["case"]+".json"))
                        for row in candidate_rows},
        "candidate_adopted": False,
        "native_renderer_implemented": False,
        "continuous_motion_qualified": False,
        "physical_skin_mechanics": False,
        "clinical_anatomy": False,
        "boundary": plan["boundary"],
    }
    atomic_json(output_dir / "summary.json", result)
    return result


def main() -> int:
    root = human.REPOSITORY_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path,
                        default=ARTIFACT_ROOT/"inputs/base/bodyparts3d-myosim-skinned-shell.nhskin")
    parser.add_argument("--manifest", type=Path,
                        default=ARTIFACT_ROOT/"inputs/base/bodyparts3d-myosim-skinned-shell.manifest.json")
    parser.add_argument("--packs", type=Path, default=ARTIFACT_ROOT/"packs")
    parser.add_argument("--baseline-audits", type=Path, default=ARTIFACT_ROOT/"audits")
    parser.add_argument("--core-manifest", type=Path,
                        default=ARTIFACT_ROOT/"inputs/native/myosim-fullbody-reference.manifest.json")
    parser.add_argument("--source-preflight", type=Path,
                        default=root/"Docs/media/skin-source-fit-recovery-20261004/receipt.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.payload, args.manifest, args.packs, args.baseline_audits,
                   args.core_manifest, args.source_preflight, args.output)
    print(json.dumps({"status": result["status"],
                      "baseline_pairs": result["baseline_intersection_pairs"],
                      "candidate_pairs": result["candidate_intersection_pairs"],
                      "new_pairs": result["new_intersection_pair_count"]}))
    return 0 if result["status"] == "passed_sampled_geometric_nonregression" else 2


if __name__ == "__main__":
    raise SystemExit(main())
