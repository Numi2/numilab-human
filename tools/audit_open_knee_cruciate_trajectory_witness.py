"""Bind a failed native PCL/ACL step to pinned source and candidate faces."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from numilab_human.open_knee import (
    HEADER_STRUCT, NODE_SET_STRUCT, NODE_STRUCT, REGION_STRUCT,
    SURFACE_PAIR_STRUCT, SURFACE_STRUCT,
)
from tools.audit_open_knee_cruciate_initialization_candidate import (
    CENTER, OUTER_M, PLATEAU_M, SHIFT_M,
)
from tools.audit_open_knee_cruciate_source_contact import (
    SELECTED, decode, strictly_crosses,
)
from tools.audit_patellofemoral_tetrahedral_separation import (
    SCALE, exact_lattice,
)


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Docs/media/cruciate-source-contact-20260930"
LOG = FOLDER / "ptl-trajectory-eight-step.log"
OUTPUT = FOLDER / "ptl-trajectory-contact-witness.json"
SURFACES = {
    "PCL": "PCL_@_ACL_ContactFaces",
    "ACL": "ACL_@_PCL_ContactFaces",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("cruciate trajectory witness: " + message)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def triangle(nodes: np.ndarray, positions: np.ndarray):
    return tuple(tuple(int(value) for value in positions[int(node)])
                 for node in nodes)


def run() -> dict:
    source_bytes, regions, surfaces, _, source, faces = decode()
    source_audit = json.loads((FOLDER / "receipt.json").read_text())
    candidate_audit = json.loads(
        (FOLDER / "initialization-candidate.json").read_text())
    require(source_audit["payload_sha256"] == sha(source_bytes) and
            candidate_audit["source_payload_sha256"] == sha(source_bytes) and
            candidate_audit["shifted_free_acl_nodes"] == 4448,
            "pinned source or candidate identity drifted")

    log_bytes = LOG.read_bytes()
    text = log_bytes.decode()
    stand = re.search(
        r"human_joint_failure completed_steps=(\d+) matter_status=(\d+) "
        r"matter_failing_index=(\d+)", text)
    contact = re.search(
        r"deformable_contact_failure step=(\d+) microtick=(\d+) "
        r"slot=(\d+) primitive_a=(\d+) primitive_b=(\d+) "
        r"thickness=([^ ]+) dt_a=([^ ]+) dt_b=([^\n]+)", text)
    require(stand is not None and contact is not None and
            int(stand[1]) == int(contact[1]) == 5 and
            int(stand[2]) == 6,
            "native accepted-step or contact status drifted")
    primitive_rows = re.findall(
        r"contact_primitive side=(\d) object=(\d+) kind=(\d+) "
        r"source=(\d+)", text)
    vertex_rows = re.findall(
        r"contact_vertex side=(\d) corner=(\d) node=(\d+) "
        r"start=([^ ]+) finish=([^\n]+)", text)
    require(len(primitive_rows) == 2 and len(vertex_rows) == 6,
            "native contact triangles are incomplete")
    selected = [region for region in regions if region[0] in SELECTED]
    require(tuple(region[0] for region in selected) == SELECTED,
            "selected runtime object order drifted")

    native_start = np.zeros((6, 3), dtype=np.float32)
    native_finish = np.zeros((6, 3), dtype=np.float32)
    local_nodes: dict[int, list[int]] = {0: [0] * 3, 1: [0] * 3}
    for side, corner, node, start, finish in vertex_rows:
        index = 3 * int(side) + int(corner)
        local_nodes[int(side)][int(corner)] = int(node)
        native_start[index] = [float(value) for value in start.split(",")]
        native_finish[index] = [float(value) for value in finish.split(",")]

    source_faces: dict[str, int] = {}
    source_triangles: dict[str, np.ndarray] = {}
    for side_text, object_text, kind_text, primitive_text in primitive_rows:
        side, object_index = int(side_text), int(object_text)
        require(side in (0, 1) and object_index in (1, 2) and
                int(kind_text) == 0 and
                int(primitive_text) == int(contact[4 + side]),
                "native contact object or primitive kind drifted")
        region_name = selected[object_index][0]
        require(region_name == ("PCL" if side == 0 else "ACL"),
                "native contact no longer belongs to PCL/ACL")
        payload_nodes = []
        for runtime_node in local_nodes[side]:
            remaining = runtime_node
            for owner, first, count in selected:
                if remaining < count:
                    require(owner == region_name,
                            "native node belongs to another region")
                    payload_nodes.append(first + remaining)
                    break
                remaining -= count
            else:
                raise RuntimeError("native contact node escaped FEM payload")
        first_face, face_count = surfaces[SURFACES[region_name]]
        matching = [face for face in range(face_count)
                    if sorted(int(node) for node in faces[first_face + face]) ==
                    sorted(payload_nodes)]
        require(len(matching) == 1,
                "native face is outside its source-authored contact surface")
        source_faces[region_name] = matching[0]
        source_triangles[region_name] = faces[first_face + matching[0]]

    header = HEADER_STRUCT.unpack_from(source_bytes)
    node_offset = (HEADER_STRUCT.size + header[3] * REGION_STRUCT.size +
                   header[6] * SURFACE_STRUCT.size +
                   header[8] * NODE_SET_STRUCT.size +
                   header[10] * SURFACE_PAIR_STRUCT.size)
    acl_first, acl_count = next((first, count)
                                for name, first, count in regions
                                if name == "ACL")
    anchored = np.array([
        bool(NODE_STRUCT.unpack_from(
            source_bytes, node_offset + node * NODE_STRUCT.size)[-1] & 1)
        for node in range(acl_first, acl_first + acl_count)], dtype=bool)
    candidate = source.astype(np.float64) / SCALE
    radius = np.linalg.norm(candidate[acl_first:acl_first + acl_count] -
                            CENTER, axis=1)
    u = np.clip((radius - PLATEAU_M) / (OUTER_M - PLATEAU_M), 0.0, 1.0)
    weight = 1.0 - u * u * (3.0 - 2.0 * u)
    weight[anchored] = 0.0
    candidate[acl_first:acl_first + acl_count, 0] += SHIFT_M * weight
    candidate_exact = exact_lattice(candidate.astype(np.float32))

    pcl, acl = source_triangles["PCL"], source_triangles["ACL"]
    source_crossing = strictly_crosses(triangle(pcl, source),
                                       triangle(acl, source))
    candidate_crossing = strictly_crosses(triangle(pcl, candidate_exact),
                                          triangle(acl, candidate_exact))
    native_initial = exact_lattice(native_start)
    native_final = exact_lattice(native_finish)
    native_start_crossing = strictly_crosses(
        triangle(np.arange(3), native_initial),
        triangle(np.arange(3, 6), native_initial))
    native_finish_crossing = strictly_crosses(
        triangle(np.arange(3), native_final),
        triangle(np.arange(3, 6), native_final))
    require(not source_crossing and not candidate_crossing and
            not native_start_crossing and native_finish_crossing,
            "source-to-native crossing transition changed")

    return {
        "schema": "numi.human.cruciate-trajectory-contact-witness.v1",
        "source_payload_sha256": sha(source_bytes),
        "initialization_candidate_sha256": sha(
            (FOLDER / "initialization-candidate.json").read_bytes()),
        "native_log_sha256": sha(log_bytes),
        "accepted_steps_before_rejection": int(stand[1]),
        "failed_zero_based_step": int(contact[1]),
        "matter_status": int(stand[2]),
        "matter_first_failing_index": int(stand[3]),
        "deformable_contact_slot": int(contact[3]),
        "source_contact_pair": "PCL_To_ACL",
        "source_contact_faces": source_faces,
        "source_strict_crossing": source_crossing,
        "candidate_strict_crossing": candidate_crossing,
        "native_step_start_strict_crossing": native_start_crossing,
        "native_step_finish_strict_crossing": native_finish_crossing,
        "maximum_native_vertex_move_m": float(np.linalg.norm(
            native_finish.astype(np.float64) -
            native_start.astype(np.float64), axis=1).max()),
        "qualified": False,
        "boundary": "The local ACL initialization clears this face pair initially, "
                    "but native motion produces a strict PCL/ACL crossing on "
                    "attempted step six. This is diagnostic geometry, not a "
                    "source-consistent contact resolution or loaded-knee proof.",
    }


if __name__ == "__main__":
    result = run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in (
        "accepted_steps_before_rejection", "source_contact_faces",
        "native_step_finish_strict_crossing")}, sort_keys=True))
