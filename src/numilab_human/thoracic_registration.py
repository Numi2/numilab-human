"""Register the twelve BodyParts3D thoracic vertebrae to pinned MyoSim meshes.

Each vertebra receives a rest-geometry proper-rigid correction inside the one
authored MyoSim torso segment.  This adds no articulation.  A candidate is
emitted only when every named thoracic enthesis, the complete T1--T12 chain,
and both neighbouring C7/T1 and T12/L1 transitions pass unchanged gates.
"""

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


SCHEMA = "numi.human.bodyparts3d-myosim-thoracic-source-mesh-registration.v1"
REGISTRATION_SCHEMA = "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
TENDON_SCHEMAS = {
    "numi.human.tendon-attachment-envelope-payload.v2",
    "numi.human.tendon-attachment-envelope-payload.v3",
}
ENDPOINT_MAXIMUM_DISTANCE_M = 0.012
HELD_OUT_P90_MAXIMUM_M = 0.015
ROTATION_MAXIMUM_RAD = math.radians(20.0)
TRANSLATION_MAXIMUM_M = 0.050


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_angle(rotation: Any, np: Any) -> float:
    cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) * 0.5))
    return math.acos(cosine)


def _fit_passes(fit: dict[str, Any], np: Any) -> bool:
    return bool(
        float(np.linalg.det(fit["rotation"])) > 0.999999
        and float(fit["held_out_metrics"]["p90_m"]) <= HELD_OUT_P90_MAXIMUM_M
        and _fit_angle(fit["rotation"], np) <= ROTATION_MAXIMUM_RAD + 1.0e-12
        and float(np.linalg.norm(fit["translation"])) <= TRANSLATION_MAXIMUM_M + 1.0e-12
    )


def propose_thoracic_registration(
    *,
    sources: Path,
    registration_path: Path,
    source_audit_path: Path,
    tendon_manifest_path: Path,
) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - source environment only
        raise RuntimeError(
            "thoracic registration requires the pinned MyoSim/MuJoCo environment"
        ) from error

    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    tendon_manifest = json.loads(tendon_manifest_path.read_text(encoding="utf-8"))
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise RuntimeError("thoracic registration requires registration candidate v2")
    if source_audit.get("schema") != AUDIT_SCHEMA:
        raise RuntimeError("thoracic registration requires source-bone audit v1")
    if tendon_manifest.get("schema") not in TENDON_SCHEMAS:
        raise RuntimeError("thoracic registration requires NHTENDON2 or NHTENDON3")
    source_hashes = {
        registration.get("source", {}).get("myosim", {}).get("source", {}).get(
            "archive_sha256"
        ),
        source_audit.get("source", {}).get("archive_sha256"),
        tendon_manifest.get("source", {}).get("myosim_archive_sha256"),
    }
    if len(source_hashes) != 1 or None in source_hashes:
        raise RuntimeError("thoracic registration inputs do not share one MyoSim source")

    exported = export_fullbody(sources)
    model = build_model("myofullbody")
    meshes_by_body = _compiled_meshes_by_body(model, mujoco, np)
    source_bodies = {int(body["id"]): body for body in exported["bodies"]}
    anchors_by_member: dict[str, dict[str, Any]] = {}
    for anchor in registration.get("anchors", []):
        member = anchor.get("source", {}).get("member_id")
        if not isinstance(member, str) or member in anchors_by_member:
            raise RuntimeError("thoracic registration contains an invalid anchor")
        anchors_by_member[member] = anchor

    member_by_level = human_model._NUMI_HUMAN_THORACIC_VERTEBRA_ENTHESIS_MEMBER_IDS
    records: dict[str, dict[str, Any]] = {}
    for level, member_id in sorted(member_by_level.items()):
        anchor = anchors_by_member.get(member_id)
        if anchor is None or anchor.get("target", {}).get("name") != "torso":
            raise RuntimeError(f"thoracic registration has no torso anchor for T{level}")
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
        target = anchor["target"]
        source_body_id = int(target["source_body_id"])
        source_body = source_bodies.get(source_body_id)
        matching_meshes = [
            mesh for mesh in meshes_by_body.get(source_body_id, [])
            if f"thoracic{level}_" in str(mesh.get("mesh_name"))
        ]
        if source_body is None or len(matching_meshes) != 1:
            raise RuntimeError(f"thoracic registration cannot resolve exact MyoSim T{level}")
        source_vertices = _body_frame_to_core(
            np.asarray(matching_meshes[0]["vertices"], dtype=float), source_body, np
        )
        records[member_id] = {
            "level": level,
            "anchor": anchor,
            "target": target,
            "source_body": source_body,
            "vertices": vertices,
            "triangles": np.asarray(raw_triangles, dtype=int),
            "source_vertices": source_vertices,
            "fit_candidates": _fit_candidates(vertices, source_vertices, np),
            "endpoints": [],
        }

    audit_by_key = {
        (int(endpoint["source_actuator_index"]), str(endpoint["endpoint"])): endpoint
        for endpoint in source_audit.get("endpoints", [])
    }
    endpoint_records = []
    for endpoint in tendon_manifest.get("endpoints", []):
        muscle = endpoint.get("muscle")
        endpoint_name = endpoint.get("endpoint")
        if not isinstance(muscle, str) or endpoint_name not in {"origin", "insertion"}:
            continue
        ordinal = 0 if endpoint_name == "origin" else 1
        member_ids = human_model._NUMI_HUMAN_AXIAL_ENTHESIS_MEMBERS.get((muscle, ordinal))
        if not member_ids or member_ids[0] not in records:
            continue
        key = (int(endpoint["muscle_index"]), endpoint_name)
        source_endpoint = audit_by_key.get(key)
        if (
            source_endpoint is None
            or source_endpoint.get("classification") != "source_model_bone_adjacent"
            or source_endpoint.get("source_body_name") != "torso"
        ):
            raise RuntimeError(f"thoracic enthesis lacks source-bone authority: {muscle}:{endpoint_name}")
        record = records[member_ids[0]]
        point = _body_frame_to_core(
            np.asarray([source_endpoint["source_site_position_body_m"]], dtype=float),
            record["source_body"], np,
        )[0]
        item = {
            "source_actuator_index": key[0],
            "muscle": muscle,
            "endpoint": endpoint_name,
            "member_id": member_ids[0],
            "attachment_mode_before": endpoint.get("attachment_mode"),
            "point": point,
        }
        record["endpoints"].append(item)
        endpoint_records.append(item)
    if len(endpoint_records) != 28:
        raise RuntimeError(
            f"thoracic registration expected 28 named entheses, found {len(endpoint_records)}"
        )

    chosen: dict[str, dict[str, Any]] = {}
    endpoint_metrics = []
    for member_id, record in records.items():
        points = np.asarray([item["point"] for item in record["endpoints"]], dtype=float)
        selected = None
        selected_distances = None
        for fit in record["fit_candidates"]:
            if not _fit_passes(fit, np):
                continue
            distances = _endpoint_surface_distances(
                points, record["vertices"], record["triangles"],
                fit["rotation"], fit["translation"], np,
            ) if len(points) else np.asarray([], dtype=float)
            if bool(np.all(distances <= ENDPOINT_MAXIMUM_DISTANCE_M + 1.0e-12)):
                selected, selected_distances = fit, distances
                break
        if selected is None:
            raise RuntimeError(
                f"thoracic T{record['level']} has no candidate satisfying surface and enthesis gates"
            )
        chosen[member_id] = selected
        for item, distance in zip(record["endpoints"], selected_distances, strict=True):
            before = _endpoint_surface_distances(
                np.asarray([item["point"]]), record["vertices"], record["triangles"],
                np.eye(3), np.zeros(3), np,
            )[0]
            endpoint_metrics.append({
                **{key: item[key] for key in (
                    "source_actuator_index", "muscle", "endpoint", "member_id",
                    "attachment_mode_before",
                )},
                "distance_before_m": float(before),
                "distance_after_m": float(distance),
                "passed_12mm_gate": bool(distance <= ENDPOINT_MAXIMUM_DISTANCE_M),
            })

    transformed = {
        member_id: _transform_points(record["vertices"], chosen[member_id], np)
        for member_id, record in records.items()
    }
    continuity = []
    continuity_pairs = [
        (f"thoracic{level}_to_thoracic{level + 1}", member_by_level[level], member_by_level[level + 1])
        for level in range(1, 12)
    ]
    for name, first, second in continuity_pairs:
        gap = _minimum_gap(transformed[first], transformed[second], np)[0]
        continuity.append({
            "name": name,
            "source_member_ids": [first, second],
            "minimum_vertex_gap_m": gap,
            "maximum_allowed_gap_m": human_model._NUMI_HUMAN_AXIAL_CONTINUITY_MAXIMUM_GAP_M,
            "passed": gap <= human_model._NUMI_HUMAN_AXIAL_CONTINUITY_MAXIMUM_GAP_M,
        })
    for name, first, second in (
        ("cervical7_to_thoracic1", "FJ3172", member_by_level[1]),
        ("thoracic12_to_lumbar1", member_by_level[12], "FJ3157"),
    ):
        other_member = first if first not in transformed else second
        other_anchor = anchors_by_member[other_member]
        source = other_anchor["source"]
        _, member, obj = human_model._bodyparts_obj_member(
            sources, source["hierarchy"], other_member
        )
        raw_vertices, _ = human_model._bodyparts_obj_triangles(obj, member)
        matrix = np.asarray(
            other_anchor["registration"]["source_obj_mm_to_core_inertial_body_m"], dtype=float
        )
        other_core = np.einsum(
            "ki,ji->kj", np.asarray(raw_vertices, dtype=float), matrix[:3, :3]
        ) + matrix[:3, 3]
        first_world = _core_to_world(
            transformed[first] if first in transformed else other_core,
            records[first]["target"] if first in records else other_anchor["target"], np,
        )
        second_world = _core_to_world(
            transformed[second] if second in transformed else other_core,
            records[second]["target"] if second in records else other_anchor["target"], np,
        )
        gap = _minimum_gap(first_world, second_world, np)[0]
        continuity.append({
            "name": name,
            "source_member_ids": [first, second],
            "minimum_vertex_gap_m": gap,
            "maximum_allowed_gap_m": human_model._NUMI_HUMAN_AXIAL_CONTINUITY_MAXIMUM_GAP_M,
            "passed": gap <= human_model._NUMI_HUMAN_AXIAL_CONTINUITY_MAXIMUM_GAP_M,
        })
    failed = [item["name"] for item in continuity if not item["passed"]]
    if failed:
        raise RuntimeError("thoracic registration violates continuity: " + ", ".join(failed))

    output = json.loads(json.dumps(registration))
    output_anchors = {
        anchor["source"]["member_id"]: anchor for anchor in output["anchors"]
    }
    body_fits = []
    for member_id, record in records.items():
        fit = chosen[member_id]
        receipt = {
            "thoracic_level": int(record["level"]),
            "method": "exact_level_pca_seeded_trimmed_symmetric_rigid_icp_to_compiled_myosim_mesh",
            "selected_start": fit["start"],
            "iterations": int(fit["iterations"]),
            "proper_rotation_determinant": float(np.linalg.det(fit["rotation"])),
            "rotation_angle_rad": _fit_angle(fit["rotation"], np),
            "rigid_translation_core_m": [float(value) for value in fit["translation"]],
            "training_metrics": fit["training_metrics"],
            "held_out_metrics": fit["held_out_metrics"],
            "training_vertex_count": int(fit["training_vertex_count"]),
            "held_out_vertex_count": int(fit["held_out_vertex_count"]),
            "independent_articulation_count": 0,
        }
        anchor = output_anchors[member_id]
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
        centroid_world = _core_to_world(
            np.asarray([centroid_core]), anchor["target"], np
        )[0]
        anchor["registration"]["default_pose_vertex_centroid_world_m"] = [
            float(value) for value in centroid_world
        ]
        anchor["registration"]["status"] = "provisional_thoracic_source_mesh_rigid_registration"
        anchor["registration"]["thoracic_source_mesh_registration"] = receipt
        body_fits.append({"member_id": member_id, **receipt})

    output["thoracic_source_mesh_registration"] = {
        "schema": SCHEMA,
        "status": "candidate_passed_all_named_enthesis_and_axial_continuity_gates",
        "inputs": {
            "registration": {"file": registration_path.name, "sha256": _sha256(registration_path)},
            "source_bone_audit": {"file": source_audit_path.name, "sha256": _sha256(source_audit_path)},
            "tendon_manifest": {"file": tendon_manifest_path.name, "sha256": _sha256(tendon_manifest_path)},
            "myosim_archive_sha256": next(iter(source_hashes)),
        },
        "thoracic_member_count": len(body_fits),
        "named_enthesis_count": len(endpoint_metrics),
        "named_enthesis_gate_pass_count": sum(item["passed_12mm_gate"] for item in endpoint_metrics),
        "prior_distributed_enthesis_count": sum(
            item["attachment_mode_before"] == "registered_bone_distributed_envelope"
            for item in endpoint_metrics
        ),
        "point_enthesis_recovery_candidate_count": sum(
            item["attachment_mode_before"] == "source_site_point" for item in endpoint_metrics
        ),
        "maximum_enthesis_distance_before_m": max(item["distance_before_m"] for item in endpoint_metrics),
        "maximum_enthesis_distance_after_m": max(item["distance_after_m"] for item in endpoint_metrics),
        "body_fits": body_fits,
        "continuity": continuity,
        "maximum_continuity_gap_m": max(item["minimum_vertex_gap_m"] for item in continuity),
        "named_entheses": endpoint_metrics,
        "new_joint_count": 0,
        "endpoint_migration_m": 0.0,
        "promotion_requirement": (
            "Recompile paired NHBONES1/NHTENDON3 artifacts, require all 28 named thoracic "
            "entheses to remain distributed, and inspect the complete rib-spine-tendon complex "
            "from anterior, posterior, and both lateral M4 Pro views."
        ),
    }
    output["status"] = "provisional_visual_registration_not_admitted_to_collision_or_physics"
    output["evidence_boundary"] = (
        "This candidate corrects the rest geometry of twelve exact BodyParts3D vertebrae inside "
        "one existing MyoSim torso rigid body. It adds no joint, moves no route site, and does not "
        "qualify discs, cartilage, contact, fascia, or clinical registration."
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
        result = propose_thoracic_registration(
            sources=arguments.sources.resolve(),
            registration_path=arguments.registration.resolve(),
            source_audit_path=arguments.source_audit.resolve(),
            tendon_manifest_path=arguments.tendon_manifest.resolve(),
        )
    except (RuntimeError, OSError, ValueError) as error:
        print(f"numilab-human thoracic registration: {error}", file=sys.stderr)
        return 2
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
