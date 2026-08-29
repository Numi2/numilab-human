"""Restore the source-owned manubrium and neutral sternal-girdle continuity.

The BodyParts3D meshes share one anatomical coordinate system.  The initial
registration preserves that common frame, but a later soft-tissue site
translation moved the sternum body independently and the visual skeleton did
not include the manubrium at all.  This pass restores the exact sternum body to
the pinned common frame and adds the exact manubrium on the existing MyoSim
torso body.  It does not move either clavicle, add a joint, or invent geometry.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from . import model as human_model


SCHEMA = "numi.human.bodyparts3d-sternal-girdle-registration.v1"
MANUBRIUM_MEMBER_ID = "FJ3290"
STERNUM_BODY_MEMBER_ID = "FJ3178"
CLAVICLE_MEMBER_IDS = {"right": "FJ3362", "left": "FJ3237"}
MANUBRIOSTERNAL_MAXIMUM_GAP_M = 0.002
STERNOCLAVICULAR_MAXIMUM_GAP_M = 0.004
MAXIMUM_STERNUM_CENTROID_CORRECTION_M = 0.060


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transform(points: list[list[float]] | list[tuple[float, float, float]],
               matrix: list[list[float]]) -> list[list[float]]:
    return [
        [
            sum(matrix[row][axis] * float(point[axis]) for axis in range(3))
            + matrix[row][3]
            for row in range(3)
        ]
        for point in points
    ]


def _anchor_world_vertices(
    sources: Path, anchor: dict[str, Any],
) -> list[list[float]]:
    source = anchor["source"]
    _, member, obj = human_model._bodyparts_obj_member(
        sources, str(source["hierarchy"]), str(source["member_id"])
    )
    vertices, _ = human_model._bodyparts_obj_triangles(obj, member)
    local = _transform(
        vertices,
        anchor["registration"]["source_obj_mm_to_core_inertial_body_m"],
    )
    target = anchor["target"]
    rotation = human_model._myosim_matrix_from_quaternion_xyzw(
        target["default_inertial_quaternion_world_xyzw"]
    )
    com = target["default_com_position_world_m"]
    return [
        [
            sum(rotation[row][axis] * point[axis] for axis in range(3))
            + com[row]
            for row in range(3)
        ]
        for point in local
    ]


def _source_identity(sources: Path, member_id: str) -> tuple[str, str]:
    relation = sources / "isa_element_parts.txt"
    if not relation.is_file():
        raise RuntimeError("sternal-girdle registration requires BodyParts3D element labels")
    matches = []
    for raw in relation.read_text(encoding="utf-8").splitlines():
        columns = raw.split("\t")
        if (
            len(columns) == 3
            and columns[2] == member_id
            and columns[1] == "manubrium"
        ):
            matches.append((columns[0], columns[1]))
    if len(matches) != 1:
        raise RuntimeError(f"sternal-girdle registration cannot resolve {member_id}")
    return matches[0]


def register_sternal_girdle(
    *, sources: Path, registration_path: Path,
) -> dict[str, Any]:
    registration_path = registration_path.resolve()
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("schema") != (
        "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
    ):
        raise RuntimeError("sternal-girdle registration requires a v2 bone registration")
    anchors = registration.get("anchors")
    coordinate = registration.get("coordinate_system")
    global_matrix = coordinate.get("global_source_mm_to_myosim_world_m") \
        if isinstance(coordinate, dict) else None
    if (
        not isinstance(anchors, list)
        or not isinstance(global_matrix, list)
        or len(global_matrix) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in global_matrix)
    ):
        raise RuntimeError("sternal-girdle registration has no pinned common frame")
    if any(
        not math.isfinite(float(value))
        for row in global_matrix for value in row
    ):
        raise RuntimeError("sternal-girdle common frame is non-finite")

    by_member = {
        str(anchor.get("source", {}).get("member_id")): anchor
        for anchor in anchors if isinstance(anchor, dict)
    }
    input_has_manubrium = MANUBRIUM_MEMBER_ID in by_member
    sternum = by_member.get(STERNUM_BODY_MEMBER_ID)
    clavicles = {
        side: by_member.get(member_id)
        for side, member_id in CLAVICLE_MEMBER_IDS.items()
    }
    if sternum is None or any(anchor is None for anchor in clavicles.values()):
        raise RuntimeError("sternal-girdle registration is missing sternum or clavicle authority")

    concept_id, source_name = _source_identity(sources, MANUBRIUM_MEMBER_ID)
    if source_name != "manubrium":
        raise RuntimeError("BodyParts3D FJ3290 is no longer the manubrium")
    archive_path, member, obj = human_model._bodyparts_obj_member(
        sources, "is_a", MANUBRIUM_MEMBER_ID
    )
    manubrium_vertices, manubrium_triangles = human_model._bodyparts_obj_triangles(
        obj, member
    )
    centroid_mm = [
        sum(vertex[axis] for vertex in manubrium_vertices) / len(manubrium_vertices)
        for axis in range(3)
    ]
    manubrium_world = _transform(manubrium_vertices, global_matrix)
    sternum_archive = str(sternum["source"]["archive"])
    sternum_archive_sha = str(sternum["source"]["archive_sha256"])
    if archive_path.name != sternum_archive or _sha256(archive_path) != sternum_archive_sha:
        raise RuntimeError("sternal-girdle BodyParts3D archive provenance drifted")

    output = copy.deepcopy(registration)
    output_by_member = {
        str(anchor["source"]["member_id"]): anchor for anchor in output["anchors"]
    }
    output_sternum = output_by_member[STERNUM_BODY_MEMBER_ID]
    target = output_sternum["target"]
    common_local = human_model._bodyparts_local_registration_matrix(
        global_matrix,
        target["default_com_position_world_m"],
        target["default_inertial_quaternion_world_xyzw"],
    )
    old_sternum_centroid = output_sternum["registration"][
        "default_pose_vertex_centroid_world_m"
    ]
    sternum_centroid_mm = output_sternum["source"]["vertex_centroid_mm"]
    new_sternum_centroid = _transform([sternum_centroid_mm], global_matrix)[0]
    correction = math.sqrt(sum(
        (new_sternum_centroid[axis] - old_sternum_centroid[axis]) ** 2
        for axis in range(3)
    ))
    if correction > MAXIMUM_STERNUM_CENTROID_CORRECTION_M:
        raise RuntimeError("sternal-girdle sternum correction exceeds its bounded gate")
    superseded_refinement = output_sternum["registration"].pop(
        "attachment_surface_refinement", None
    )
    output_sternum["registration"].update({
        "source_obj_mm_to_core_inertial_body_m": common_local,
        "default_pose_vertex_centroid_world_m": new_sternum_centroid,
        "vertex_centroid_to_source_com_residual_m": None,
        "status": "source_common_frame_sternal_continuity_restoration",
    })

    manubrium_record = {
        "source": {
            "archive": archive_path.name,
            "archive_sha256": sternum_archive_sha,
            "hierarchy": "is_a",
            "member": member,
            "member_id": MANUBRIUM_MEMBER_ID,
            "member_sha256": hashlib.sha256(obj).hexdigest(),
            "concept_id": concept_id,
            "name": source_name,
            "vertex_count": len(manubrium_vertices),
            "triangle_count": len(manubrium_triangles),
            "vertex_centroid_mm": centroid_mm,
        },
        "target": copy.deepcopy(target),
        "registration": {
            "source_obj_mm_to_core_inertial_body_m": common_local,
            "default_pose_vertex_centroid_world_m": _transform(
                [centroid_mm], global_matrix
            )[0],
            "vertex_centroid_to_source_com_residual_m": None,
            "status": "source_common_frame_visual_binding",
        },
    }
    if input_has_manubrium:
        output_manubrium = output_by_member[MANUBRIUM_MEMBER_ID]
        output_manubrium.clear()
        output_manubrium.update(manubrium_record)
    else:
        sternum_index = next(
            index for index, anchor in enumerate(output["anchors"])
            if anchor["source"]["member_id"] == STERNUM_BODY_MEMBER_ID
        )
        output["anchors"].insert(sternum_index + 1, manubrium_record)

    sternum_world = _anchor_world_vertices(sources, output_sternum)
    manubriosternal_gap = human_model._bodyparts_bounded_vertex_gap(
        manubrium_world, sternum_world, MANUBRIOSTERNAL_MAXIMUM_GAP_M,
        "manubrium to sternum body", "source sternal continuity",
    )
    sternoclavicular: dict[str, float] = {}
    for side, clavicle in clavicles.items():
        assert clavicle is not None
        gap = human_model._bodyparts_bounded_vertex_gap(
            manubrium_world,
            _anchor_world_vertices(sources, clavicle),
            STERNOCLAVICULAR_MAXIMUM_GAP_M,
            f"manubrium to {side} clavicle",
            "source sternoclavicular continuity",
        )
        sternoclavicular[side] = gap
    if abs(sternoclavicular["right"] - sternoclavicular["left"]) > 0.002:
        raise RuntimeError("sternal-girdle sternoclavicular gaps lose bilateral parity")

    output["sternal_girdle_source_registration"] = {
        "schema": SCHEMA,
        "status": "passed_exact_source_geometry_and_neutral_continuity_gates",
        "input": {
            "registration": registration_path.name,
            "sha256": _sha256(registration_path),
        },
        "source": {
            "manubrium_member_id": MANUBRIUM_MEMBER_ID,
            "manubrium_member_sha256": hashlib.sha256(obj).hexdigest(),
            "bodyparts_archive_sha256": sternum_archive_sha,
        },
        "ownership": {
            "manubrium_body": "torso",
            "sternum_body": "torso",
            "manubrium_added_to_input": not input_has_manubrium,
            "new_joint_count": 0,
            "clavicle_transform_count": 0,
        },
        "sternum_centroid_common_frame_correction_m": correction,
        "manubriosternal_gap_m": manubriosternal_gap,
        "sternoclavicular_gap_m": sternoclavicular,
        "superseded_soft_tissue_site_refinement": superseded_refinement,
        "evidence_boundary": (
            "Exact BodyParts3D bone geometry and neutral source-surface continuity; "
            "not sternoclavicular cartilage, disc, ligament, or contact mechanics."
        ),
    }
    output["status"] = "provisional_visual_registration_not_admitted_to_collision_or_physics"
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = register_sternal_girdle(
            sources=arguments.sources,
            registration_path=arguments.registration,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"numilab-human sternal-girdle registration: {error}", file=sys.stderr)
        return 1
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
