"""Register 24 exact BodyParts3D ribs to topology-resolved MyoSim rib meshes.

The pinned MyoSim ribcage mesh contains 24 disconnected, bilaterally ordered
rib components.  This tool derives that correspondence from topology and rest
pose, applies one proper-rigid rest placement per exact BodyParts3D rib, and
selects left/right candidates jointly.  It adds no rib articulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
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
    _surface_split_metrics,
    _transform_points,
)


SCHEMA = "numi.human.bodyparts3d-myosim-rib-source-component-registration.v1"
REGISTRATION_SCHEMA = "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
TENDON_SCHEMAS = {
    "numi.human.tendon-attachment-envelope-payload.v2",
    "numi.human.tendon-attachment-envelope-payload.v3",
}
ENDPOINT_MAXIMUM_DISTANCE_M = 0.012
COSTOVERTEBRAL_MAXIMUM_GAP_M = 0.008
BILATERAL_GAP_PARITY_MAXIMUM_M = 0.0025
BILATERAL_ENDPOINT_PARITY_MAXIMUM_M = 0.0025
ROTATION_MAXIMUM_RAD = math.radians(35.0)
REFINEMENT_MAXIMUM_TRANSLATION_M = 0.004
REFINEMENT_STEPS_M = (0.001, 0.0005, 0.00025, 0.000125, 0.0000625)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit_angle(rotation: Any, np: Any) -> float:
    cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) * 0.5))
    return math.acos(cosine)


def _held_out_gate(level: int) -> float:
    # BodyParts3D's left fifth and both twelfth ribs are longer than the low-
    # polygon MyoSim components.  Their mean fit remains below 8 mm; these p90
    # gates admit that documented atlas shape difference without allowing a
    # reflected or misplaced rib.
    return 0.023 if level == 12 else (0.020 if level == 5 else 0.015)


def _translation_gate(level: int) -> float:
    if level <= 8:
        return 0.040
    if level == 9:
        return 0.055
    if level <= 11:
        return 0.070
    return 0.090


def _components(vertices: Any, faces: Any) -> list[list[int]]:
    parents = list(range(len(vertices)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def merge(first: int, second: int) -> None:
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first, second, third in faces:
        merge(int(first), int(second))
        merge(int(second), int(third))
        merge(int(third), int(first))
    result: dict[int, list[int]] = defaultdict(list)
    for index in range(len(vertices)):
        result[root(index)].append(index)
    return sorted(result.values(), key=lambda value: (-len(value), value[0]))


def _dense_component_surface(
    vertices: Any, faces: Any, component: list[int], np: Any,
) -> Any:
    members = set(component)
    component_faces = np.asarray([
        face for face in faces if all(int(index) in members for index in face)
    ], dtype=int)
    if len(component_faces) < 4:
        raise RuntimeError("rib source component has too few triangles")
    triangles = vertices[component_faces]
    return np.concatenate((
        vertices[component],
        np.mean(triangles, axis=1),
        (triangles[:, 0] + triangles[:, 1]) * 0.5,
        (triangles[:, 1] + triangles[:, 2]) * 0.5,
        (triangles[:, 2] + triangles[:, 0]) * 0.5,
    ))


def _refine_translation_to_gates(
    *, fit: dict[str, Any], vertices: Any, triangles: Any, endpoints: Any,
    vertebra_vertices: Any, np: Any,
) -> tuple[dict[str, Any], Any, float]:
    rotation = fit["rotation"]
    initial = fit["translation"].copy()
    translation = initial.copy()

    def evaluate(candidate: Any) -> tuple[tuple[float, float, float], Any, float]:
        distances = _endpoint_surface_distances(
            endpoints, vertices, triangles, rotation, candidate, np,
        ) if len(endpoints) else np.asarray([], dtype=float)
        gap = _minimum_gap(
            np.einsum("ki,ji->kj", vertices, rotation) + candidate,
            vertebra_vertices, np,
        )[0]
        endpoint_ratio = float(max(distances, default=0.0)) / ENDPOINT_MAXIMUM_DISTANCE_M
        gap_ratio = gap / COSTOVERTEBRAL_MAXIMUM_GAP_M
        return (
            (max(endpoint_ratio, gap_ratio), endpoint_ratio + gap_ratio,
             float(np.linalg.norm(candidate - initial))),
            distances, gap,
        )

    objective, distances, gap = evaluate(translation)
    if objective[0] <= 1.0 + 1.0e-12:
        return fit, distances, gap
    for step in REFINEMENT_STEPS_M:
        while True:
            best = (objective, translation, distances, gap)
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    candidate = translation.copy()
                    candidate[axis] += sign * step
                    if (
                        float(np.linalg.norm(candidate - initial))
                        > REFINEMENT_MAXIMUM_TRANSLATION_M + 1.0e-12
                    ):
                        continue
                    candidate_objective, candidate_distances, candidate_gap = evaluate(candidate)
                    if candidate_objective < best[0]:
                        best = (
                            candidate_objective, candidate,
                            candidate_distances, candidate_gap,
                        )
            if bool(np.array_equal(best[1], translation)):
                break
            objective, translation, distances, gap = best
            if objective[0] <= 1.0 + 1.0e-12:
                refined = dict(fit)
                refined["translation"] = translation
                refined["gate_refinement_translation_m"] = float(
                    np.linalg.norm(translation - initial)
                )
                return refined, distances, gap
    return fit, distances, gap


def propose_rib_registration(
    *, sources: Path, registration_path: Path, source_audit_path: Path,
    tendon_manifest_path: Path,
) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - source environment only
        raise RuntimeError(
            "rib registration requires the pinned MyoSim/MuJoCo environment"
        ) from error

    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    tendon_manifest = json.loads(tendon_manifest_path.read_text(encoding="utf-8"))
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise RuntimeError("rib registration requires registration candidate v2")
    if source_audit.get("schema") != AUDIT_SCHEMA:
        raise RuntimeError("rib registration requires source-bone audit v1")
    if tendon_manifest.get("schema") not in TENDON_SCHEMAS:
        raise RuntimeError("rib registration requires NHTENDON2 or NHTENDON3")
    source_hashes = {
        registration.get("source", {}).get("myosim", {}).get("source", {}).get("archive_sha256"),
        source_audit.get("source", {}).get("archive_sha256"),
        tendon_manifest.get("source", {}).get("myosim_archive_sha256"),
    }
    if len(source_hashes) != 1 or None in source_hashes:
        raise RuntimeError("rib registration inputs do not share one MyoSim source")

    exported = export_fullbody(sources)
    source_bodies = {int(body["id"]): body for body in exported["bodies"]}
    meshes_by_body = _compiled_meshes_by_body(build_model("myofullbody"), mujoco, np)
    anchors = {
        anchor["source"]["member_id"]: anchor for anchor in registration.get("anchors", [])
    }
    torso_anchor = anchors[human_model._NUMI_HUMAN_THORACIC_VERTEBRA_ENTHESIS_MEMBER_IDS[1]]
    torso_body_id = int(torso_anchor["target"]["source_body_id"])
    torso_body = source_bodies[torso_body_id]
    ribcage_meshes = [
        mesh for mesh in meshes_by_body.get(torso_body_id, [])
        if "ribcage" in str(mesh.get("mesh_name"))
    ]
    if len(ribcage_meshes) != 1:
        raise RuntimeError("rib registration cannot resolve one pinned ribcage mesh")
    ribcage = ribcage_meshes[0]
    source_vertices = _body_frame_to_core(
        np.asarray(ribcage["vertices"], dtype=float), torso_body, np
    )
    source_faces = np.asarray(ribcage["faces"], dtype=int)
    all_components = _components(source_vertices, source_faces)
    rib_components = [
        component for component in all_components
        if float(np.mean(source_vertices[component], axis=0)[0]) < 0.0
    ]
    if len(all_components) != 36 or len(rib_components) != 24:
        raise RuntimeError(
            "ribcage topology drifted; expected 36 components including 24 ribs"
        )
    rib_components.sort(key=lambda component: (
        -float(np.mean(source_vertices[component], axis=0)[1]),
        -float(np.mean(source_vertices[component], axis=0)[2]),
    ))
    component_by_side_level: dict[tuple[str, int], list[int]] = {}
    for level in range(1, 13):
        pair = rib_components[(level - 1) * 2:level * 2]
        if len(pair) != 2:
            raise RuntimeError(f"rib registration cannot form source pair at level {level}")
        for component in pair:
            centroid = np.mean(source_vertices[component], axis=0)
            side = "r" if float(centroid[2]) > 0.0 else "l"
            key = (side, level)
            if key in component_by_side_level:
                raise RuntimeError(f"rib registration source pair is not bilateral at level {level}")
            component_by_side_level[key] = component

    audit_index = {
        (int(endpoint["source_actuator_index"]), str(endpoint["endpoint"])): endpoint
        for endpoint in source_audit.get("endpoints", [])
    }
    rib_members = {
        member for side_members in human_model._NUMI_HUMAN_RIB_ENTHESIS_MEMBER_IDS.values()
        for member in side_members.values()
    }
    endpoints_by_member: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for endpoint in tendon_manifest.get("endpoints", []):
        endpoint_name = endpoint.get("endpoint")
        if endpoint_name not in {"origin", "insertion"}:
            continue
        member_ids = human_model._NUMI_HUMAN_AXIAL_ENTHESIS_MEMBERS.get((
            str(endpoint.get("muscle")), 0 if endpoint_name == "origin" else 1,
        ))
        if not member_ids or member_ids[0] not in rib_members:
            continue
        source_endpoint = audit_index.get((int(endpoint["muscle_index"]), endpoint_name))
        if source_endpoint is None:
            raise RuntimeError(
                f"rib enthesis lacks source-bone authority: {endpoint.get('muscle')}:{endpoint_name}"
            )
        if (
            source_endpoint.get("classification") != "source_model_bone_adjacent"
            or source_endpoint.get("source_body_name") != "torso"
        ):
            # Named routes can terminate in soft tissue even when their label
            # contains a rib level.  Preserve those source points; this
            # registration is only allowed to recover source-bone-adjacent
            # endpoints.
            continue
        point = _body_frame_to_core(
            np.asarray([source_endpoint["source_site_position_body_m"]], dtype=float),
            torso_body, np,
        )[0]
        endpoints_by_member[member_ids[0]].append({
            "muscle_index": int(endpoint["muscle_index"]),
            "muscle": str(endpoint["muscle"]), "endpoint": endpoint_name,
            "attachment_mode_before": str(endpoint["attachment_mode"]),
            "point": point,
        })
    if sum(len(items) for items in endpoints_by_member.values()) != 44:
        raise RuntimeError("rib registration expected 44 named rib entheses")

    vertebrae: dict[int, Any] = {}
    source_vertebrae: dict[int, Any] = {}
    for level, member_id in human_model._NUMI_HUMAN_THORACIC_VERTEBRA_ENTHESIS_MEMBER_IDS.items():
        anchor = anchors[member_id]
        source = anchor["source"]
        _, member, obj = human_model._bodyparts_obj_member(
            sources, source["hierarchy"], member_id
        )
        raw_vertices, _ = human_model._bodyparts_obj_triangles(obj, member)
        matrix = np.asarray(
            anchor["registration"]["source_obj_mm_to_core_inertial_body_m"], dtype=float
        )
        vertebrae[level] = np.einsum(
            "ki,ji->kj", np.asarray(raw_vertices, dtype=float), matrix[:3, :3]
        ) + matrix[:3, 3]
        matching = [
            mesh for mesh in meshes_by_body[torso_body_id]
            if f"thoracic{level}_" in str(mesh.get("mesh_name"))
        ]
        if len(matching) != 1:
            raise RuntimeError(f"rib registration cannot resolve source T{level}")
        source_vertebrae[level] = _body_frame_to_core(
            np.asarray(matching[0]["vertices"], dtype=float), torso_body, np
        )

    records: dict[tuple[str, int], dict[str, Any]] = {}
    for side in ("r", "l"):
        for level in range(1, 13):
            member_id = human_model._NUMI_HUMAN_RIB_ENTHESIS_MEMBER_IDS[side][level]
            anchor = anchors[member_id]
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
            component = component_by_side_level[(side, level)]
            dense_source = _dense_component_surface(
                source_vertices, source_faces, component, np
            )
            points = np.asarray(
                [item["point"] for item in endpoints_by_member[member_id]], dtype=float
            )
            candidates = []
            for ordinal, fit in enumerate(_fit_candidates(vertices, dense_source, np, 6)):
                if (
                    _fit_angle(fit["rotation"], np) > ROTATION_MAXIMUM_RAD
                    or float(np.linalg.norm(fit["translation"])) > _translation_gate(level)
                    or float(fit["held_out_metrics"]["p90_m"]) > _held_out_gate(level)
                ):
                    continue
                refined, distances, gap = _refine_translation_to_gates(
                    fit=fit, vertices=vertices,
                    triangles=np.asarray(raw_triangles, dtype=int), endpoints=points,
                    vertebra_vertices=vertebrae[level], np=np,
                )
                if (
                    float(max(distances, default=0.0)) > ENDPOINT_MAXIMUM_DISTANCE_M + 1.0e-12
                    or gap > COSTOVERTEBRAL_MAXIMUM_GAP_M + 1.0e-12
                ):
                    continue
                training_metrics, held_out_metrics, training_count, held_out_count = (
                    _surface_split_metrics(
                        vertices, dense_source, refined["rotation"], refined["translation"], np
                    )
                )
                if held_out_metrics["p90_m"] > _held_out_gate(level) + 1.0e-12:
                    continue
                candidates.append({
                    "ordinal": ordinal, "fit": refined, "distances": distances,
                    "costovertebral_gap_m": gap,
                    "training_metrics": training_metrics,
                    "held_out_metrics": held_out_metrics,
                    "training_vertex_count": training_count,
                    "held_out_vertex_count": held_out_count,
                })
            if not candidates:
                raise RuntimeError(f"rib registration has no acceptable {side} rib {level} fit")
            records[(side, level)] = {
                "member_id": member_id, "anchor": anchor, "vertices": vertices,
                "triangles": np.asarray(raw_triangles, dtype=int),
                "dense_source": dense_source, "source_component": component,
                "endpoints": endpoints_by_member[member_id], "candidates": candidates,
            }

    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    pair_receipts = []
    for level in range(1, 13):
        best = None
        right_record, left_record = records[("r", level)], records[("l", level)]
        right_source_gap = _minimum_gap(
            right_record["dense_source"], source_vertebrae[level], np
        )[0]
        left_source_gap = _minimum_gap(
            left_record["dense_source"], source_vertebrae[level], np
        )[0]
        for right in right_record["candidates"]:
            for left in left_record["candidates"]:
                gap_parity = abs(
                    right["costovertebral_gap_m"] - left["costovertebral_gap_m"]
                )
                right_endpoint = float(max(right["distances"], default=0.0))
                left_endpoint = float(max(left["distances"], default=0.0))
                endpoint_parity = abs(right_endpoint - left_endpoint)
                if (
                    gap_parity > BILATERAL_GAP_PARITY_MAXIMUM_M + 1.0e-12
                    or endpoint_parity > BILATERAL_ENDPOINT_PARITY_MAXIMUM_M + 1.0e-12
                ):
                    continue
                score = (
                    right["held_out_metrics"]["mean_m"]
                    + left["held_out_metrics"]["mean_m"]
                    + abs(right["costovertebral_gap_m"] - right_source_gap)
                    + abs(left["costovertebral_gap_m"] - left_source_gap)
                    + gap_parity + endpoint_parity
                )
                candidate = (
                    score, right["ordinal"], left["ordinal"], right, left,
                    gap_parity, endpoint_parity,
                )
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is None:
            raise RuntimeError(f"rib registration has no bilateral pair at level {level}")
        chosen[("r", level)], chosen[("l", level)] = best[3], best[4]
        pair_receipts.append({
            "thoracic_level": level,
            "right_candidate_ordinal": best[1], "left_candidate_ordinal": best[2],
            "right_costovertebral_gap_m": best[3]["costovertebral_gap_m"],
            "left_costovertebral_gap_m": best[4]["costovertebral_gap_m"],
            "costovertebral_gap_parity_m": best[5],
            "maximum_endpoint_distance_parity_m": best[6],
            "passed": True,
        })

    centroid_order = []
    for side in ("r", "l"):
        previous = None
        for level in range(1, 13):
            record, selected = records[(side, level)], chosen[(side, level)]
            centroid_y = float(np.mean(
                _transform_points(record["vertices"], selected["fit"], np), axis=0
            )[1])
            if previous is not None and not centroid_y < previous:
                raise RuntimeError(f"rib registration reverses {side} rib order at level {level}")
            centroid_order.append({
                "side": "right" if side == "r" else "left",
                "thoracic_level": level, "centroid_core_y_m": centroid_y,
                "inferior_to_previous": previous is None or centroid_y < previous,
            })
            previous = centroid_y

    output = json.loads(json.dumps(registration))
    output_anchors = {
        anchor["source"]["member_id"]: anchor for anchor in output["anchors"]
    }
    fit_receipts = []
    endpoint_metrics = []
    for side in ("r", "l"):
        for level in range(1, 13):
            record, selected = records[(side, level)], chosen[(side, level)]
            fit = selected["fit"]
            receipt = {
                "side": "right" if side == "r" else "left",
                "thoracic_level": level,
                "method": "topology_resolved_dense_component_trimmed_symmetric_rigid_icp",
                "source_component_vertex_count": len(record["source_component"]),
                "source_dense_surface_point_count": len(record["dense_source"]),
                "selected_start": fit["start"], "iterations": int(fit["iterations"]),
                "proper_rotation_determinant": float(np.linalg.det(fit["rotation"])),
                "rotation_angle_rad": _fit_angle(fit["rotation"], np),
                "rigid_translation_core_m": [float(value) for value in fit["translation"]],
                "gate_refinement_translation_m": float(
                    fit.get("gate_refinement_translation_m", 0.0)
                ),
                "training_metrics": selected["training_metrics"],
                "held_out_metrics": selected["held_out_metrics"],
                "training_vertex_count": int(selected["training_vertex_count"]),
                "held_out_vertex_count": int(selected["held_out_vertex_count"]),
                "costovertebral_gap_m": selected["costovertebral_gap_m"],
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
            centroid_world = _core_to_world(
                np.asarray([centroid_core]), anchor["target"], np
            )[0]
            anchor["registration"]["default_pose_vertex_centroid_world_m"] = [
                float(value) for value in centroid_world
            ]
            anchor["registration"]["status"] = "provisional_rib_source_component_rigid_registration"
            anchor["registration"]["rib_source_component_registration"] = receipt
            fit_receipts.append({"member_id": record["member_id"], **receipt})
            before_points = np.asarray(
                [item["point"] for item in record["endpoints"]], dtype=float
            )
            before_distances = _endpoint_surface_distances(
                before_points, record["vertices"], record["triangles"],
                np.eye(3), np.zeros(3), np,
            ) if len(before_points) else []
            for item, before, after in zip(
                record["endpoints"], before_distances, selected["distances"], strict=True
            ):
                endpoint_metrics.append({
                    **{key: item[key] for key in (
                        "muscle_index", "muscle", "endpoint", "attachment_mode_before",
                    )},
                    "member_id": record["member_id"],
                    "distance_before_m": float(before), "distance_after_m": float(after),
                    "passed_12mm_gate": bool(after <= ENDPOINT_MAXIMUM_DISTANCE_M),
                })

    output["rib_source_component_registration"] = {
        "schema": SCHEMA,
        "status": "candidate_passed_topology_enthesis_costovertebral_bilateral_and_order_gates",
        "inputs": {
            "registration": {"file": registration_path.name, "sha256": _sha256(registration_path)},
            "source_bone_audit": {"file": source_audit_path.name, "sha256": _sha256(source_audit_path)},
            "tendon_manifest": {"file": tendon_manifest_path.name, "sha256": _sha256(tendon_manifest_path)},
            "myosim_archive_sha256": next(iter(source_hashes)),
        },
        "ribcage_connected_component_count": len(all_components),
        "topology_resolved_rib_component_count": len(rib_components),
        "bodyparts_rib_member_count": len(fit_receipts),
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
        "maximum_costovertebral_gap_m": max(
            item["costovertebral_gap_m"] for item in fit_receipts
        ),
        "maximum_costovertebral_gap_parity_m": max(
            item["costovertebral_gap_parity_m"] for item in pair_receipts
        ),
        "maximum_endpoint_distance_parity_m": max(
            item["maximum_endpoint_distance_parity_m"] for item in pair_receipts
        ),
        "body_fits": fit_receipts, "bilateral_pairs": pair_receipts,
        "centroid_order": centroid_order, "named_entheses": endpoint_metrics,
        "new_joint_count": 0, "endpoint_migration_m": 0.0,
        "promotion_requirement": (
            "Recompile paired NHBONES1/NHTENDON3 artifacts, require all 44 named rib entheses "
            "to be distributed, and inspect the complete rib-spine-tendon cage from anterior, "
            "posterior, both lateral, superior, and inferior M4 Pro views."
        ),
    }
    output["status"] = "provisional_visual_registration_not_admitted_to_collision_or_physics"
    output["evidence_boundary"] = (
        "This candidate derives 24 rib correspondences from the disconnected topology and bilateral "
        "ordering of one pinned MyoSim ribcage mesh, then corrects exact BodyParts3D rib rest placement "
        "inside the existing torso body. It adds no rib joints, moves no route sites, and does not "
        "qualify costal cartilage, contact, breathing mechanics, fascia, or clinical registration."
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
        result = propose_rib_registration(
            sources=arguments.sources.resolve(), registration_path=arguments.registration.resolve(),
            source_audit_path=arguments.source_audit.resolve(),
            tendon_manifest_path=arguments.tendon_manifest.resolve(),
        )
    except (RuntimeError, OSError, ValueError) as error:
        print(f"numilab-human rib registration: {error}", file=sys.stderr)
        return 2
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
