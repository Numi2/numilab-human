"""Register the paired BodyParts3D hip bones to pinned MyoSim pelvis meshes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from . import model as human_model
from .myosim_bone_proximity import AUDIT_SCHEMA, _compiled_meshes_by_body
from .myosim_export import export_fullbody
from .upper_limb_registration import (
    _body_frame_to_core,
    _core_to_world,
    _endpoint_surface_distances,
    _fit_candidates,
    _minimum_gap,
    _transform_points,
)


SCHEMA = "numi.human.bodyparts3d-myosim-pelvis-source-mesh-registration.v1"
REGISTRATION_SCHEMA = "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
TENDON_SCHEMAS = {
    "numi.human.tendon-attachment-envelope-payload.v2",
    "numi.human.tendon-attachment-envelope-payload.v3",
}
ENDPOINT_MAXIMUM_DISTANCE_M = 0.012
HELD_OUT_P90_MAXIMUM_M = 0.015
ROTATION_MAXIMUM_RAD = math.radians(10.0)
TRANSLATION_MAXIMUM_M = 0.020
BILATERAL_ENDPOINT_PARITY_MAXIMUM_M = 0.002
BILATERAL_SACRAL_GAP_PARITY_MAXIMUM_M = 0.002


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_angle(rotation: Any, np: Any) -> float:
    cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) * 0.5))
    return math.acos(cosine)


def propose_pelvis_registration(
    *, sources: Path, registration_path: Path, source_audit_path: Path,
    tendon_manifest_path: Path,
) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - source environment only
        raise RuntimeError(
            "pelvis registration requires the pinned MyoSim/MuJoCo environment"
        ) from error

    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    tendon_manifest = json.loads(tendon_manifest_path.read_text(encoding="utf-8"))
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise RuntimeError("pelvis registration requires registration candidate v2")
    if source_audit.get("schema") != AUDIT_SCHEMA:
        raise RuntimeError("pelvis registration requires source-bone audit v1")
    if tendon_manifest.get("schema") not in TENDON_SCHEMAS:
        raise RuntimeError("pelvis registration requires NHTENDON2 or NHTENDON3")
    source_hashes = {
        registration.get("source", {}).get("myosim", {}).get("source", {}).get("archive_sha256"),
        source_audit.get("source", {}).get("archive_sha256"),
        tendon_manifest.get("source", {}).get("myosim_archive_sha256"),
    }
    if len(source_hashes) != 1 or None in source_hashes:
        raise RuntimeError("pelvis registration inputs do not share one MyoSim source")

    source_bodies = {int(body["id"]): body for body in export_fullbody(sources)["bodies"]}
    meshes_by_body = _compiled_meshes_by_body(build_model("myofullbody"), mujoco, np)
    anchors = {
        anchor["source"]["member_id"]: anchor for anchor in registration.get("anchors", [])
    }
    specifications = {
        "r": ("FJ3152", "r_pelvis"),
        "l": ("FJ3288", "l_pelvis"),
    }
    records: dict[str, dict[str, Any]] = {}
    for side, (member_id, mesh_name) in specifications.items():
        anchor = anchors.get(member_id)
        if anchor is None or anchor.get("target", {}).get("name") != "pelvis":
            raise RuntimeError(f"pelvis registration has no {side} hip anchor")
        source = anchor["source"]
        _, member, obj = human_model._bodyparts_obj_member(
            sources, source["hierarchy"], member_id
        )
        raw_vertices, raw_triangles = human_model._bodyparts_obj_triangles(obj, member)
        matrix = np.asarray(
            anchor["registration"]["source_obj_mm_to_core_inertial_body_m"], dtype=float
        )
        vertices = np.einsum(
            "ki,ji->kj", np.asarray(raw_vertices, dtype=float), matrix[:3, :3]
        ) + matrix[:3, 3]
        source_body_id = int(anchor["target"]["source_body_id"])
        source_body = source_bodies.get(source_body_id)
        matching = [
            mesh for mesh in meshes_by_body.get(source_body_id, [])
            if mesh.get("mesh_name") == mesh_name
        ]
        if source_body is None or len(matching) != 1:
            raise RuntimeError(f"pelvis registration cannot resolve exact {mesh_name}")
        source_vertices = _body_frame_to_core(
            np.asarray(matching[0]["vertices"], dtype=float), source_body, np
        )
        records[side] = {
            "member_id": member_id, "anchor": anchor, "vertices": vertices,
            "triangles": np.asarray(raw_triangles, dtype=int),
            "source_body": source_body, "source_vertices": source_vertices,
            "fit_candidates": _fit_candidates(vertices, source_vertices, np),
            "endpoints": [],
        }

    audit_index = {
        (int(endpoint["source_actuator_index"]), str(endpoint["endpoint"])): endpoint
        for endpoint in source_audit.get("endpoints", [])
    }
    for endpoint in tendon_manifest.get("endpoints", []):
        surface = endpoint.get("surface")
        member_id = surface.get("bone_member_id") if isinstance(surface, dict) else None
        side = next((side for side, spec in specifications.items() if spec[0] == member_id), None)
        is_iliacus = endpoint.get("muscle") in {"iliacus_r", "iliacus_l"} and endpoint.get("endpoint") == "origin"
        if side is None and is_iliacus:
            side = str(endpoint["muscle"])[-1]
            member_id = specifications[side][0]
        if side is None:
            continue
        source_endpoint = audit_index.get((int(endpoint["muscle_index"]), str(endpoint["endpoint"])))
        if source_endpoint is None:
            raise RuntimeError("pelvis registration cannot join a named endpoint to source audit")
        if is_iliacus and source_endpoint.get("classification") != "source_model_bone_adjacent":
            raise RuntimeError(f"pelvis registration lacks source-bone authority for iliacus_{side}")
        point = _body_frame_to_core(
            np.asarray([source_endpoint["source_site_position_body_m"]], dtype=float),
            records[side]["source_body"], np,
        )[0]
        records[side]["endpoints"].append({
            "muscle_index": int(endpoint["muscle_index"]),
            "muscle": str(endpoint["muscle"]),
            "endpoint": str(endpoint["endpoint"]),
            "attachment_mode_before": str(endpoint["attachment_mode"]),
            "point": point,
        })
    if any(len(record["endpoints"]) != 22 for record in records.values()):
        raise RuntimeError("pelvis registration expected 22 named endpoints per hip")

    chosen: dict[str, dict[str, Any]] = {}
    metrics = []
    for side, record in records.items():
        points = np.asarray([item["point"] for item in record["endpoints"]], dtype=float)
        selected = None
        selected_distances = None
        for fit in record["fit_candidates"]:
            if not (
                float(np.linalg.det(fit["rotation"])) > 0.999999
                and fit["held_out_metrics"]["p90_m"] <= HELD_OUT_P90_MAXIMUM_M
                and _fit_angle(fit["rotation"], np) <= ROTATION_MAXIMUM_RAD
                and float(np.linalg.norm(fit["translation"])) <= TRANSLATION_MAXIMUM_M
            ):
                continue
            distances = _endpoint_surface_distances(
                points, record["vertices"], record["triangles"],
                fit["rotation"], fit["translation"], np,
            )
            if bool(np.all(distances <= ENDPOINT_MAXIMUM_DISTANCE_M + 1.0e-12)):
                selected, selected_distances = fit, distances
                break
        if selected is None:
            raise RuntimeError(f"pelvis {side} has no surface-fit candidate preserving every endpoint")
        chosen[side] = selected
        before = _endpoint_surface_distances(
            points, record["vertices"], record["triangles"], np.eye(3), np.zeros(3), np
        )
        for item, initial, final in zip(record["endpoints"], before, selected_distances, strict=True):
            metrics.append({
                "side": side,
                **{key: item[key] for key in (
                    "muscle_index", "muscle", "endpoint", "attachment_mode_before",
                )},
                "member_id": record["member_id"],
                "distance_before_m": float(initial), "distance_after_m": float(final),
                "passed_12mm_gate": bool(final <= ENDPOINT_MAXIMUM_DISTANCE_M),
            })

    paired = {}
    for item in metrics:
        key = (item["muscle"][:-2], item["endpoint"])
        paired.setdefault(key, {})[item["side"]] = item["distance_after_m"]
    pair_differences = [
        abs(pair["r"] - pair["l"]) for pair in paired.values()
        if set(pair) == {"r", "l"}
    ]
    if len(pair_differences) != 22 or max(pair_differences) > BILATERAL_ENDPOINT_PARITY_MAXIMUM_M:
        raise RuntimeError("pelvis registration violates bilateral endpoint parity")

    sacral_anchor = anchors["FJ3393"]
    source = sacral_anchor["source"]
    _, sacral_member, sacral_obj = human_model._bodyparts_obj_member(
        sources, source["hierarchy"], "FJ3393"
    )
    raw_sacrum, _ = human_model._bodyparts_obj_triangles(sacral_obj, sacral_member)
    sacral_matrix = np.asarray(
        sacral_anchor["registration"]["source_obj_mm_to_core_inertial_body_m"], dtype=float
    )
    sacrum_core = np.einsum(
        "ki,ji->kj", np.asarray(raw_sacrum, dtype=float), sacral_matrix[:3, :3]
    ) + sacral_matrix[:3, 3]
    sacrum_world = _core_to_world(sacrum_core, sacral_anchor["target"], np)
    continuity = []
    for side, record in records.items():
        hip_world = _core_to_world(
            _transform_points(record["vertices"], chosen[side], np),
            record["anchor"]["target"], np,
        )
        gap = _minimum_gap(sacrum_world, hip_world, np)[0]
        continuity.append({
            "name": f"sacrum_to_{'right' if side == 'r' else 'left'}_hip",
            "source_member_ids": ["FJ3393", record["member_id"]],
            "minimum_vertex_gap_m": gap,
            "maximum_allowed_gap_m": human_model._NUMI_HUMAN_AXIAL_CONTINUITY_MAXIMUM_GAP_M,
            "passed": gap <= human_model._NUMI_HUMAN_AXIAL_CONTINUITY_MAXIMUM_GAP_M,
        })
    if (
        any(not item["passed"] for item in continuity)
        or abs(continuity[0]["minimum_vertex_gap_m"] - continuity[1]["minimum_vertex_gap_m"])
        > BILATERAL_SACRAL_GAP_PARITY_MAXIMUM_M
    ):
        raise RuntimeError("pelvis registration violates sacroiliac continuity or parity")

    output = json.loads(json.dumps(registration))
    output_anchors = {
        anchor["source"]["member_id"]: anchor for anchor in output["anchors"]
    }
    fits = []
    for side, record in records.items():
        fit = chosen[side]
        receipt = {
            "side": "right" if side == "r" else "left",
            "method": "exact_side_pca_seeded_trimmed_symmetric_rigid_icp_to_compiled_myosim_pelvis_mesh",
            "selected_start": fit["start"], "iterations": int(fit["iterations"]),
            "proper_rotation_determinant": float(np.linalg.det(fit["rotation"])),
            "rotation_angle_rad": _fit_angle(fit["rotation"], np),
            "rigid_translation_core_m": [float(value) for value in fit["translation"]],
            "training_metrics": fit["training_metrics"], "held_out_metrics": fit["held_out_metrics"],
            "training_vertex_count": int(fit["training_vertex_count"]),
            "held_out_vertex_count": int(fit["held_out_vertex_count"]),
            "independent_articulation_count": 0,
        }
        anchor = output_anchors[record["member_id"]]
        matrix = np.asarray(
            anchor["registration"]["source_obj_mm_to_core_inertial_body_m"], dtype=float
        )
        matrix[:3, :3] = fit["rotation"] @ matrix[:3, :3]
        matrix[:3, 3] = fit["rotation"] @ matrix[:3, 3] + fit["translation"]
        anchor["registration"]["source_obj_mm_to_core_inertial_body_m"] = [
            [float(value) for value in row] for row in matrix
        ]
        centroid_mm = np.asarray(anchor["source"]["vertex_centroid_mm"], dtype=float)
        centroid_core = np.einsum("i,ji->j", centroid_mm, matrix[:3, :3]) + matrix[:3, 3]
        centroid_world = _core_to_world(np.asarray([centroid_core]), anchor["target"], np)[0]
        anchor["registration"]["default_pose_vertex_centroid_world_m"] = [
            float(value) for value in centroid_world
        ]
        anchor["registration"]["status"] = "provisional_pelvis_source_mesh_rigid_registration"
        anchor["registration"]["pelvis_source_mesh_registration"] = receipt
        fits.append({"member_id": record["member_id"], **receipt})

    output["pelvis_source_mesh_registration"] = {
        "schema": SCHEMA,
        "status": "candidate_passed_bilateral_source_mesh_enthesis_and_sacroiliac_gates",
        "inputs": {
            "registration": {"file": registration_path.name, "sha256": _sha256(registration_path)},
            "source_bone_audit": {"file": source_audit_path.name, "sha256": _sha256(source_audit_path)},
            "tendon_manifest": {"file": tendon_manifest_path.name, "sha256": _sha256(tendon_manifest_path)},
            "myosim_archive_sha256": next(iter(source_hashes)),
        },
        "hip_member_count": 2, "named_endpoint_count": len(metrics),
        "named_endpoint_gate_pass_count": sum(item["passed_12mm_gate"] for item in metrics),
        "prior_distributed_endpoint_count": sum(
            item["attachment_mode_before"] == "registered_bone_distributed_envelope" for item in metrics
        ),
        "point_enthesis_recovery_candidate_count": sum(
            item["muscle"] in {"iliacus_r", "iliacus_l"} for item in metrics
        ),
        "maximum_endpoint_distance_before_m": max(item["distance_before_m"] for item in metrics),
        "maximum_endpoint_distance_after_m": max(item["distance_after_m"] for item in metrics),
        "maximum_bilateral_endpoint_distance_difference_m": max(pair_differences),
        "body_fits": fits, "continuity": continuity, "named_endpoints": metrics,
        "new_joint_count": 0, "endpoint_migration_m": 0.0,
        "promotion_requirement": (
            "Recompile exact paired NHBONES1/NHTENDON3 artifacts, require both iliacus origins "
            "and all 42 prior hip envelopes to remain distributed, then inspect bilateral pelvis views."
        ),
    }
    output["status"] = "provisional_visual_registration_not_admitted_to_collision_or_physics"
    output["evidence_boundary"] = (
        "This candidate corrects two exact BodyParts3D hip-bone rest meshes inside the existing "
        "MyoSim pelvis body. It adds no articulation, moves no route site, and does not qualify "
        "sacroiliac cartilage, contact, fascia, or clinical registration."
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--tendon-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = propose_pelvis_registration(
            sources=arguments.sources.resolve(), registration_path=arguments.registration.resolve(),
            source_audit_path=arguments.source_audit.resolve(),
            tendon_manifest_path=arguments.tendon_manifest.resolve(),
        )
    except (RuntimeError, OSError, ValueError) as error:
        print(f"numilab-human pelvis registration: {error}", file=sys.stderr)
        return 2
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
