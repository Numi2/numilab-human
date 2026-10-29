#!/usr/bin/env python3
"""Emit the existing registered-volume CSVs for a locally refined skin/fat reference.

Captured source nodes, cells, labels and mass stay separately identified.
Inserted nodes and changed cells are derived anatomy with explicit source
ancestry. This is an asset compiler, not a runtime integrator or mass admission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import struct
import sys
from collections import defaultdict
from typing import Any


ROLE_ORDER = ("O", "I", "D")
ROLE_DOMAIN = {
    "O": "derived_connected_refinement433_O",
    "I": "D37_coarse_parent",
    "D": "D37_coarse_parent",
}
MATERIAL = {"O": 15, "I": 15, "D": 16}
CORNER_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
IDENTITY_KIND = "source_parent_tet_reference_registration"
FRAME_METHOD = "captured_native_current_rest_local_to_world_f32"
OPERATIONAL_REFERENCE_MAP_FIELDS = [
    "source_parent_node_row", "operational_reference_tet_row",
    "operational_reference_bary0", "operational_reference_bary1",
    "operational_reference_bary2", "operational_reference_bary3",
    "operational_reference_residual_x_m", "operational_reference_residual_y_m",
    "operational_reference_residual_z_m", "operational_reference_distance_m",
    "source_parent_geometric_donor_body", "source_parent_anatomical_region_id",
]


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: pathlib.Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def candidate_operational_reference_fields(
        node: int, source_parent_node_row: int, source_parent_tet_row: int,
        parent_labels: list[dict[str, Any]], child_tets: list[tuple[int, ...]],
        child_labels: list[dict[str, Any]], incident_cells: list[list[int]]) -> list[Any]:
    """Create exact self-tet registration while keeping source ancestry separate."""
    if node < 0 or node >= len(incident_cells):
        raise ValueError("candidate operational reference node is outside the child mesh")
    if source_parent_node_row < -1 or source_parent_tet_row < 0 or source_parent_tet_row >= len(parent_labels):
        raise ValueError("candidate operational reference lacks immutable source-parent ancestry")
    parent_label = parent_labels[source_parent_tet_row]
    donor = int(parent_label["geometric_donor_body"])
    region = int(parent_label["anatomical_region_id"])
    if donor < 0 or region < 0:
        raise ValueError("candidate operational reference source-parent tuple is invalid")
    eligible = [cell for cell in incident_cells[node]
                if 0 <= cell < len(child_tets) and
                int(child_labels[cell]["geometric_donor_body"]) == donor and
                int(child_labels[cell]["anatomical_region_id"]) == region]
    if not eligible:
        raise ValueError("candidate node has no incident operational tet in its source-parent donor/region")
    cell = min(eligible)
    tet = child_tets[cell]
    corners = [corner for corner, volume_node in enumerate(tet) if int(volume_node) == node]
    if len(corners) != 1 or len(set(tet)) != 4:
        raise ValueError("candidate operational reference tet does not uniquely contain its node")
    bary = [0.0, 0.0, 0.0, 0.0]
    bary[corners[0]] = 1.0
    return [source_parent_node_row, cell, *bary, 0.0, 0.0, 0.0, 0.0, donor, region]


def validate_candidate_operational_reference_fields(
        fields: list[Any], node: int, expected_source_parent_node_row: int,
        expected_source_parent_tet_row: int, parent_labels: list[dict[str, Any]],
        child_tets: list[tuple[int, ...]], child_labels: list[dict[str, Any]],
        incident_cells: list[list[int]]) -> None:
    if len(fields) != len(OPERATIONAL_REFERENCE_MAP_FIELDS):
        raise ValueError("candidate operational reference map row has the wrong width")
    source_node, cell = int(fields[0]), int(fields[1])
    if source_node != expected_source_parent_node_row:
        raise ValueError("candidate map changed the immutable source-parent node identity")
    if cell < 0 or cell >= len(child_tets) or cell not in incident_cells[node]:
        raise ValueError("candidate operational support tet is not incident to its node")
    parent_label = parent_labels[expected_source_parent_tet_row]
    donor = int(parent_label["geometric_donor_body"])
    region = int(parent_label["anatomical_region_id"])
    if int(fields[10]) != donor or int(fields[11]) != region:
        raise ValueError("candidate map changed immutable source-parent donor/region ancestry")
    child_label = child_labels[cell]
    if int(child_label["geometric_donor_body"]) != donor or int(child_label["anatomical_region_id"]) != region:
        raise ValueError("candidate operational support tet is outside its typed donor/region domain")
    bary = [float(value) for value in fields[2:6]]
    tet = child_tets[cell]
    if len(set(tet)) != 4 or node not in tet or bary != [1.0 if int(volume_node) == node else 0.0 for volume_node in tet]:
        raise ValueError("candidate operational registration is not an exact one-hot self-tet reference")
    residual = [float(value) for value in fields[6:9]]
    distance = float(fields[9])
    if residual != [0.0, 0.0, 0.0] or distance != 0.0:
        raise ValueError("candidate operational self-tet registration must have exact zero residual and distance")


def f32(x: float) -> float:
    try:
        y = struct.unpack("<f", struct.pack("<f", x))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"coordinate is not representable as Float32: {x}") from exc
    if not math.isfinite(y):
        raise ValueError(f"coordinate is not finite Float32: {x}")
    return y


def norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def sub(a: tuple[float, float, float], b: tuple[float, float, float]):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a: tuple[float, float, float], b: tuple[float, float, float]):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def orient_sign(tri: tuple[int, int, int]) -> int:
    inversions = sum(tri[i] > tri[j] for i in range(3) for j in range(i + 1, 3))
    return -1 if inversions & 1 else 1


def v3(row: dict[str, str], names: tuple[str, str, str]) -> tuple[float, float, float]:
    return tuple(float(row[k]) for k in names)


def f32op(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f32_add(a: float, b: float) -> float:
    return f32op(a + b)


def f32_sub(a: float, b: float) -> float:
    return f32op(a - b)


def f32_mul(a: float, b: float) -> float:
    return f32op(a * b)


def inverse3_row_major(matrix: tuple[float, ...]) -> tuple[float, ...]:
    if len(matrix) != 9 or any(not math.isfinite(value) for value in matrix):
        raise ValueError("captured native frame matrix is incomplete or nonfinite")
    a, b, c, d, e, f, g, h, i = matrix
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ValueError("captured native frame matrix is singular")
    inverse = (
        (e * i - f * h) / determinant,
        (c * h - b * i) / determinant,
        (b * f - c * e) / determinant,
        (f * g - d * i) / determinant,
        (a * i - c * g) / determinant,
        (c * d - a * f) / determinant,
        (d * h - e * g) / determinant,
        (b * g - a * h) / determinant,
        (a * e - b * d) / determinant,
    )
    if any(not math.isfinite(value) for value in inverse):
        raise ValueError("captured native frame inverse is nonfinite")
    return inverse


def matrix_vector3(matrix: tuple[float, ...], vector: tuple[float, float, float]):
    return tuple(sum(matrix[3 * row + column] * vector[column] for column in range(3))
                 for row in range(3))


def rotate_point_native(q, point):
    x, y, z, w = q
    px, py, pz = point
    twice = (
        f32_mul(2.0, f32_sub(f32_mul(y, pz), f32_mul(z, py))),
        f32_mul(2.0, f32_sub(f32_mul(z, px), f32_mul(x, pz))),
        f32_mul(2.0, f32_sub(f32_mul(x, py), f32_mul(y, px))),
    )
    correction = (
        f32_sub(f32_add(f32_mul(w, twice[0]), f32_mul(y, twice[2])), f32_mul(z, twice[1])),
        f32_sub(f32_add(f32_mul(w, twice[1]), f32_mul(z, twice[0])), f32_mul(x, twice[2])),
        f32_sub(f32_add(f32_mul(w, twice[2]), f32_mul(x, twice[1])), f32_mul(y, twice[0])),
    )
    return tuple(f32_add(point[i], correction[i]) for i in range(3))


def load_frame_table(poses):
    by_body = {}
    body_to_binding = {}
    for binding, pose_row in enumerate(poses["bodies"]):
        body = int(pose_row["body_index"])
        if body in by_body:
            raise ValueError("native pose table repeats a bound body")
        native_pose = poses["all_bodies"][body]
        current = native_pose["current"]
        rest = native_pose["rest"]
        current_R = tuple(float(x) for x in current.get("local_to_world_row_major_f32", []))
        rest_R = tuple(float(x) for x in rest.get("local_to_world_row_major_f32", []))
        if len(current_R) != 9 or len(rest_R) != 9:
            raise ValueError("capture pose snapshot lacks native current/rest local_to_world_row_major_f32 matrices")
        if any(not math.isfinite(x) or f32op(x) != x for x in (*current_R, *rest_R)):
            raise ValueError("captured native local-to-world matrix contains a non-Float32 or nonfinite value")
        by_body[body] = {
            "binding": binding,
            "origin": tuple(f32op(x) for x in current["position_world_m"]),
            "current_R": current_R,
            "current_invR": inverse3_row_major(current_R),
            "rest_origin": tuple(f32op(x) for x in rest["position_world_m"]),
            "rest_R": rest_R,
        }
        body_to_binding[body] = binding
    if len(by_body) != 86 or int(poses["binding_count"]) != 86:
        raise ValueError(f"expected exact parent capture A native 86-binding pose table, got {len(by_body)}")
    return by_body, body_to_binding

def write_frame_table(path: pathlib.Path, frames_by_body):
    header = ["binding", "donor", "current_x", "current_y", "current_z"]
    header += [f"current_R{i}{j}" for i in range(3) for j in range(3)]
    header += ["rest_x", "rest_y", "rest_z"]
    header += [f"rest_R{i}{j}" for i in range(3) for j in range(3)]
    rows = []
    for body, frame in sorted(frames_by_body.items(), key=lambda item: item[1]["binding"]):
        rows.append([frame["binding"], body,
                     *[format(x, ".17g") for x in frame["origin"]],
                     *[format(x, ".17g") for x in frame["current_R"]],
                     *[format(x, ".17g") for x in frame["rest_origin"]],
                     *[format(x, ".17g") for x in frame["rest_R"]]])
    write_csv(path, header, rows)


def validate_o_boundary_retirement_pins(document, expected_pins):
    if document.get("pins") != expected_pins:
        raise ValueError("O-boundary retirement map source/candidate pins do not match these inputs")


def collect_parent_o_faces(parent_tets, source_labels, source_role_by_node):
    uses = defaultdict(list)
    for tet_row, tet in enumerate(parent_tets):
        for corner in CORNER_FACES:
            tri = tuple(int(tet[k]) for k in corner)
            uses[tuple(sorted(tri))].append((tet_row, tri))
    records = []
    for key, face_uses in uses.items():
        if len(face_uses) > 2:
            raise ValueError("parent capture has a nonmanifold source face")
        if len(face_uses) != 1:
            continue
        parent_tet_row, tri = face_uses[0]
        label = source_labels[parent_tet_row]
        if int(label["material_index"]) != 15:
            continue
        if any(source_role_by_node.get(node, (None, None))[0] != "O" for node in tri):
            continue
        records.append({
            "parent_o_face_row": -1,
            "node_rows": tuple(sorted(tri)),
            "oriented_node_rows": tri,
            "parent_tet_row": parent_tet_row,
            "material_index": int(label["material_index"]),
            "geometric_donor_body": int(label["geometric_donor_body"]),
            "anatomical_region_id": int(label["anatomical_region_id"]),
            "stable_source_face_id": int(label["stable_source_face_id"]),
        })
    records.sort(key=lambda row: row["node_rows"])
    for row, record in enumerate(records):
        record["parent_o_face_row"] = row
    return records


def validate_o_boundary_retirement_nodes(document, source_role_by_node,
                                          parent_node_by_child,
                                          output_o_role_by_child_node):
    if document.get("schema") != "numi-human.explicit-o-boundary-retirement.v1":
        raise ValueError("unsupported O-boundary retirement schema")
    retired_rows = document.get("retired_parent_o_nodes")
    if not isinstance(retired_rows, list) or not retired_rows:
        raise ValueError("retirement map must explicitly name retired parent O nodes")
    retired = {}
    for row in retired_rows:
        try:
            node = int(row["volume_node_row"])
            role_row = int(row["parent_o_role_row"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("retired O node row is malformed") from exc
        if node in retired:
            raise ValueError("retirement map repeats a parent O source node")
        source_role=source_role_by_node.get(node)
        if source_role is None or source_role[0]!="O":
            raise ValueError("retirement map names a non-O source node")
        if source_role != ("O", role_row):
            raise ValueError("retired node parent role row does not match the exact source O row")
        retired[node] = role_row

    parent_to_child = {}
    for child_node, parent_node in enumerate(map(int, parent_node_by_child)):
        if parent_node >= 0:
            if parent_node in parent_to_child:
                raise ValueError("a parent node appears more than once in the refined node map")
            parent_to_child[parent_node] = child_node
    source_o = {node: role_row for node, (role, role_row) in source_role_by_node.items()
                if role == "O"}
    if set(retired) - set(source_o):
        raise ValueError("retirement map names a non-O source node")
    if any(node not in parent_to_child for node in source_o):
        raise ValueError("refined node map dropped an original O source node")
    if any(node not in parent_to_child for node in retired):
        raise ValueError("retired O source node is not retained in the volume")
    if set(output_o_role_by_child_node) & {parent_to_child[node] for node in retired}:
        raise ValueError("retired parent O node remains in the candidate O boundary")
    for parent_node in source_o:
        child_node = parent_to_child[parent_node]
        output_role_row = output_o_role_by_child_node.get(child_node)
        if parent_node in retired:
            if output_role_row is not None:
                raise ValueError("retired parent O node still has an output boundary role")
        elif output_role_row is None:
            raise ValueError("an unretired parent O node lost its output boundary role")
    for child_node in output_o_role_by_child_node:
        parent_node = int(parent_node_by_child[child_node])
        if parent_node >= 0 and source_o.get(parent_node) is None:
            raise ValueError("candidate O boundary reassigns a non-O parent node")
    return set(retired)


def collect_parent_d_faces(parent_tets, source_labels, source_role_by_node):
    uses = defaultdict(list)
    for tet_row, tet in enumerate(parent_tets):
        for corner in CORNER_FACES:
            tri = tuple(int(tet[k]) for k in corner)
            uses[tuple(sorted(tri))].append((tet_row, tri))
    records = []
    for key, face_uses in uses.items():
        if len(face_uses) > 2:
            raise ValueError("parent capture has a nonmanifold source face")
        if len(face_uses) != 1:
            continue
        parent_tet_row, tri = face_uses[0]
        label = source_labels[parent_tet_row]
        if int(label["material_index"]) != 16:
            continue
        if any(source_role_by_node.get(node, (None, None))[0] != "D" for node in tri):
            continue
        records.append({
            "parent_d_face_row": -1,
            "node_rows": tuple(sorted(tri)),
            "oriented_node_rows": tri,
            "parent_tet_row": parent_tet_row,
            "material_index": int(label["material_index"]),
            "geometric_donor_body": int(label["geometric_donor_body"]),
            "anatomical_region_id": int(label["anatomical_region_id"]),
            "stable_source_face_id": int(label["stable_source_face_id"]),
        })
    records.sort(key=lambda row: row["node_rows"])
    for row, record in enumerate(records):
        record["parent_d_face_row"] = row
    return records


def collect_parent_role_boundary_faces(parent_tets, source_labels, source_role_by_node):
    """Return exact source O/I/D surface triangles used to protect the head loops."""
    uses = defaultdict(list)
    for tet_row, tet in enumerate(parent_tets):
        for corner in CORNER_FACES:
            tri = tuple(int(tet[k]) for k in corner)
            uses[tuple(sorted(tri))].append((tet_row, tri))
    result = {role: [] for role in ROLE_ORDER}
    for key, face_uses in uses.items():
        if len(face_uses) > 2:
            raise ValueError("parent capture has a nonmanifold source face")
        labels = [source_labels[row] for row, _ in face_uses]
        roles = {source_role_by_node.get(n, (None, None))[0] for n in key}
        if len(face_uses) == 1:
            label = labels[0]
            if int(label["material_index"]) == 15 and roles == {"O"}:
                result["O"].append(face_uses[0][1])
            elif int(label["material_index"]) == 16 and roles == {"D"}:
                result["D"].append(face_uses[0][1])
        elif (int(labels[0]["material_index"]) != int(labels[1]["material_index"])
              and roles == {"I"}):
            result["I"].append(face_uses[0][1])
    return result


def derive_protected_parent_nodes(parent_tets, source_labels, source_role_by_node,
                                  deep_attachment_rows, deep_attachment_sha256):
    expected_attachment_sha = "a0bad61ae21a6fb952c589ab30a69c866866996c23842929d88d91576b29ba3f"
    if deep_attachment_sha256 != expected_attachment_sha:
        raise ValueError("pinned parent deep-attachment source hash mismatch")
    surfaces = collect_parent_role_boundary_faces(parent_tets, source_labels, source_role_by_node)
    head_nodes = {}
    head_loop_counts = {}
    for role in ROLE_ORDER:
        edge_count = defaultdict(int)
        for tri in surfaces[role]:
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge_count[tuple(sorted((int(a), int(b))))] += 1
        boundary = [edge for edge, count in edge_count.items() if count == 1]
        if any(count not in (1, 2) for count in edge_count.values()):
            raise ValueError("source role surface has nonmanifold edge incidence")
        nodes = {n for edge in boundary for n in edge}
        if len(boundary) != 45 or len(nodes) != 45:
            raise ValueError("source head-loop topology differs from pinned 45-node role boundary")
        degree = defaultdict(int)
        adjacency = defaultdict(set)
        for a, b in boundary:
            degree[a] += 1; degree[b] += 1
            adjacency[a].add(b); adjacency[b].add(a)
        if any(degree[n] != 2 for n in nodes):
            raise ValueError("source head boundary is not a single cycle")
        seen = set(); loop_count = 0
        for start in sorted(nodes):
            if start in seen:
                continue
            loop_count += 1; stack = [start]
            while stack:
                node = stack.pop()
                if node in seen:
                    continue
                seen.add(node); stack.extend(adjacency[node] - seen)
        if seen != nodes:
            raise ValueError("source role head boundary is not fully connected by its edges")
        if any(source_role_by_node.get(n, (None, None))[0] != role for n in nodes):
            raise ValueError("source head loop contains a node from another role")
        head_nodes[role] = nodes
        head_loop_counts[role] = loop_count
    if any(head_nodes[a] & head_nodes[b] for a in ROLE_ORDER for b in ROLE_ORDER if a < b):
        raise ValueError("source O/I/D head loops are not disjoint")
    if len(deep_attachment_rows) != 14:
        raise ValueError("pinned source deep-attachment inventory is not 14 rows")
    attachment_nodes = []
    for row in deep_attachment_rows:
        try:
            node = int(row["volume_node_row"])
            role = row["role"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("deep-attachment source row is malformed") from exc
        if role != "D" or source_role_by_node.get(node, (None, None))[0] != "D":
            raise ValueError("deep attachment is not an exact original D source node")
        attachment_nodes.append(node)
    if len(set(attachment_nodes)) != 14 or set(attachment_nodes) & set().union(*head_nodes.values()):
        raise ValueError("deep attachments repeat or overlap the protected head loops")
    protected = set().union(*head_nodes.values(), set(attachment_nodes))
    if len(protected) != 149:
        raise ValueError("derived original head/deep protected set is not exactly 149 nodes")
    return protected, {"head_boundary_nodes_by_role": {r: len(head_nodes[r]) for r in ROLE_ORDER},
                       "head_boundary_loops_by_role": head_loop_counts,
                       "deep_attachment_nodes": len(attachment_nodes),
                       "protected_parent_node_count": len(protected),
                       "deep_attachment_sha256": deep_attachment_sha256}


def validate_d_boundary_retirement_pins(document, expected_pins):
    if document.get("pins") != expected_pins:
        raise ValueError("D-boundary retirement map source/candidate/Boolean pins do not match these inputs")


def validate_d_boundary_retirement_nodes(document, source_role_by_node,
                                          parent_node_by_child,
                                          output_d_role_by_child_node,
                                          protected_parent_nodes):
    if document.get("schema") != "numi-human.explicit-d-boundary-retirement.v1":
        raise ValueError("unsupported D-boundary retirement schema")
    retired_rows = document.get("retired_parent_d_nodes")
    if not isinstance(retired_rows, list) or not retired_rows:
        raise ValueError("D retirement map must explicitly name retired parent D nodes")
    retired = {}
    for row in retired_rows:
        try:
            node = int(row["volume_node_row"])
            role_row = int(row["parent_d_role_row"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("retired D node row is malformed") from exc
        if node in retired:
            raise ValueError("D retirement map repeats a parent D source node")
        source_role = source_role_by_node.get(node)
        if source_role is None or source_role[0] != "D":
            raise ValueError("D retirement map names a non-D source node")
        if source_role != ("D", role_row):
            raise ValueError("retired node parent role row does not match the exact source D row")
        if node in protected_parent_nodes:
            raise ValueError("D retirement map names a protected head/deep attachment node")
        retired[node] = role_row

    parent_to_child = {}
    for child_node, parent_node in enumerate(map(int, parent_node_by_child)):
        if parent_node >= 0:
            if parent_node in parent_to_child:
                raise ValueError("a parent node appears more than once in the refined node map")
            parent_to_child[parent_node] = child_node
    source_d = {node: role_row for node, (role, role_row) in source_role_by_node.items()
                if role == "D"}
    missing = set(source_role_by_node) - set(parent_to_child)
    if missing != set(retired):
        if missing - set(retired):
            raise ValueError("refined node map dropped a non-retired original source node")
        raise ValueError("D retirement map names a node that is still retained in the volume")
    if not set(retired).issubset(source_d):
        raise ValueError("D retirement map names a non-D source node")
    if set(retired) & set(protected_parent_nodes):
        raise ValueError("D retirement map includes a protected source node")
    for parent_node in set(source_d) - set(retired):
        child_node = parent_to_child[parent_node]
        if child_node not in output_d_role_by_child_node:
            raise ValueError("a retained original D source node lost its output D boundary role")
    for child_node in output_d_role_by_child_node:
        if child_node < 0 or child_node >= len(parent_node_by_child):
            raise ValueError("candidate D role row is outside the child node domain")
        parent_node = int(parent_node_by_child[child_node])
        if parent_node >= 0 and source_role_by_node.get(parent_node, (None, None))[0] != "D":
            raise ValueError("candidate D boundary reassigns a non-D source node")
    return set(retired)


def validate_tagged_output_triangle_geometry(output_row, output_points, input_points, input_triangle):
    """Require a Boolean output triangle to lie on and inside its pinned tagged input triangle."""
    import numpy as np
    output_tri = np.asarray(output_points, dtype=np.float64)
    input_points = np.asarray(input_points, dtype=np.float64)
    input_nodes = np.asarray(input_triangle, dtype=np.int64)
    if (output_tri.shape != (3, 3) or input_points.ndim != 2 or input_points.shape[1] != 3 or
        input_nodes.shape != (3,) or np.any(input_nodes < 0) or np.any(input_nodes >= len(input_points))):
        raise ValueError("Boolean tagged input/output triangle geometry is malformed")
    input_tri = input_points[input_nodes]
    if not np.isfinite(output_tri).all() or not np.isfinite(input_tri).all():
        raise ValueError("Boolean tagged input/output triangle geometry is malformed")
    u = input_tri[1] - input_tri[0]
    v = input_tri[2] - input_tri[0]
    normal = np.cross(u, v)
    normal_sq = float(np.dot(normal, normal))
    if not math.isfinite(normal_sq) or normal_sq <= 0.0:
        raise ValueError("Boolean source input contains a degenerate tagged triangle")
    d00 = float(np.dot(u, u)); d01 = float(np.dot(u, v)); d11 = float(np.dot(v, v))
    gram_det = d00 * d11 - d01 * d01
    if not math.isfinite(gram_det) or gram_det <= 0.0:
        raise ValueError("Boolean source input has an invalid tagged triangle basis")
    condition = (d00 * d11) / gram_det
    if not math.isfinite(condition) or condition > 1.0e12:
        raise ValueError("Boolean source input tagged triangle is numerically ill-conditioned")
    coordinate_scale = max(float(np.max(np.abs(input_tri))),
                           float(np.max(np.linalg.norm(input_tri[1:] - input_tri[0], axis=1))),
                           np.finfo(np.float64).tiny)
    plane_tolerance = 256.0 * np.finfo(np.float64).eps * coordinate_scale
    bary_tolerance = max(1.0e-12, 256.0 * np.finfo(np.float64).eps * condition)
    max_plane = 0.0
    min_bary = 1.0
    for point in output_tri:
        delta = point - input_tri[0]
        plane = abs(float(np.dot(delta, normal))) / math.sqrt(normal_sq)
        d20 = float(np.dot(delta, u)); d21 = float(np.dot(delta, v))
        b1 = (d11 * d20 - d01 * d21) / gram_det
        b2 = (d00 * d21 - d01 * d20) / gram_det
        b0 = 1.0 - b1 - b2
        max_plane = max(max_plane, plane)
        min_bary = min(min_bary, b0, b1, b2)
    if (not math.isfinite(max_plane) or max_plane > plane_tolerance or
        not math.isfinite(min_bary) or min_bary < -bary_tolerance):
        raise ValueError(f"Boolean output face {output_row} is not geometrically supported by its tagged input face")
    return {"maximum_plane_residual_m": max_plane,
            "minimum_input_barycentric": min_bary,
            "plane_tolerance_m": plane_tolerance,
            "barycentric_tolerance": bary_tolerance}


def validate_d_boolean_face_lineage(document, boolean_npz, source_shell_npz,
                                    bank_input_npz, baseline_npz, expected_pins,
                                    source_d_faces, source_labels, source_components,
                                    candidate_points):
    """Validate tagged CSG output-face lineage and its explicit 330-D support CSR."""
    if document.get("schema") != "numi-human.d-boolean-output-face-lineage.v1":
        raise ValueError("unsupported D Boolean face-lineage schema")
    if document.get("pins") != expected_pins:
        raise ValueError("D Boolean face-lineage pins do not match these exact inputs")
    if digest(pathlib.Path(baseline_npz)) != expected_pins.get("frozen_current006_baseline_sha256"):
        raise ValueError("D Boolean face-lineage frozen006 source-row pin mismatch")
    import numpy as np
    with np.load(boolean_npz, allow_pickle=False) as z:
        required = {"points_world_m_f64", "faces_f64", "run_index", "run_original_id", "face_id"}
        if not required.issubset(z.files):
            raise ValueError("Boolean output lacks the tagged source-face arrays")
        bp = np.asarray(z["points_world_m_f64"], dtype=np.float64)
        bf = np.asarray(z["faces_f64"], dtype=np.int64)
        run_index = np.asarray(z["run_index"], dtype=np.int64)
        run_ids = np.asarray(z["run_original_id"], dtype=np.int64)
        face_id = np.asarray(z["face_id"], dtype=np.int64)
    if bp.ndim != 2 or bp.shape[1] != 3 or not np.isfinite(bp).all():
        raise ValueError("Boolean tagged vertex domain is malformed")
    if bf.ndim != 2 or bf.shape[1] != 3 or len(bf) != len(face_id) or (len(bf) and (bf.min() < 0 or bf.max() >= len(bp))):
        raise ValueError("Boolean tagged face domain is malformed")
    if run_index.shape != (len(run_ids) + 1,) or len(run_index) == 0 or run_index[0] != 0 or run_index[-1] != 3 * len(bf):
        raise ValueError("Boolean run-index domain does not cover its face tags")
    if np.any(np.diff(run_index) < 0) or np.any(np.diff(run_index) % 3):
        raise ValueError("Boolean run-index offsets are not triangle-aligned")
    face_run_ids = np.repeat(run_ids, np.diff(run_index) // 3)
    if face_run_ids.shape != face_id.shape:
        raise ValueError("Boolean run tags do not cover every output face")
    with np.load(source_shell_npz, allow_pickle=False) as shell_z:
        shell = {k: np.asarray(shell_z[k]) for k in shell_z.files}
    shell_points_key = "points_world_m_f64" if "points_world_m_f64" in shell else None
    if shell_points_key is None or not {"faces_f64", "source_face_global_nodes", "source_tet_rows", "source_face_kind"}.issubset(shell):
        raise ValueError("pinned source shell lacks geometric faces and original 330 ancestry")
    shell_points = np.asarray(shell[shell_points_key], dtype=np.float64)
    shell_faces = np.asarray(shell["faces_f64"], dtype=np.int64)
    shell_source_nodes = np.asarray(shell["source_face_global_nodes"], dtype=np.int64)
    shell_source_tets = np.asarray(shell["source_tet_rows"], dtype=np.int64)
    shell_kinds = np.asarray(shell["source_face_kind"]).astype(str)
    if (shell_points.ndim != 2 or shell_points.shape[1] != 3 or not np.isfinite(shell_points).all() or
        shell_faces.ndim != 2 or shell_faces.shape[1] != 3 or
        (len(shell_faces) and (shell_faces.min() < 0 or shell_faces.max() >= len(shell_points))) or
        shell_source_nodes.shape != (len(shell_faces), 3) or
        shell_source_tets.shape != (len(shell_faces),) or
        shell_kinds.shape != (len(shell_faces),) or np.any(shell_source_nodes < 0)):
        raise ValueError("pinned source-shell triangle geometry or original ancestry is malformed")
    shell_face_count = len(shell_faces)
    candidate_points = np.asarray(candidate_points, dtype=np.float64)
    if candidate_points.ndim != 2 or candidate_points.shape[1] != 3 or not np.isfinite(candidate_points).all():
        raise ValueError("candidate emitted-F32 coordinate domain is malformed")
    candidate_f32_widened = np.asarray(candidate_points, dtype=np.float32).astype(np.float64)
    if not np.array_equal(candidate_points, candidate_f32_widened):
        raise ValueError("candidate coordinates are not the exact widened emitted-F32 values")
    def oriented_f32_triangle_key(points, tri):
        rows = np.asarray(points[np.asarray(tri, dtype=np.int64)], dtype=np.float32)
        if rows.shape != (3, 3) or not np.isfinite(rows).all():
            raise ValueError("oriented F32 triangle has invalid coordinates")
        bits = rows.view(np.uint32)
        ordered = [tuple(int(x) for x in row) for row in bits]
        rotations = [tuple(ordered[i:] + ordered[:i]) for i in range(3)]
        return min(rotations)
    with np.load(baseline_npz, allow_pickle=False) as baseline_z:
        baseline_child_parent_tet = np.asarray(baseline_z["child_parent_tet_row"], dtype=np.int64)
    if (baseline_child_parent_tet.ndim != 1 or
        np.any(baseline_child_parent_tet < 0) or np.any(baseline_child_parent_tet >= len(source_labels)) or
        (len(shell_source_tets) and (shell_source_tets.min() < 0 or shell_source_tets.max() >= len(baseline_child_parent_tet)))):
        raise ValueError("frozen current006 source-tet crosswalk is malformed")
    with np.load(bank_input_npz, allow_pickle=False) as bank_z:
        bank = {k: np.asarray(bank_z[k]) for k in bank_z.files}
    bank_face_key = "faces_f64" if "faces_f64" in bank else "faces" if "faces" in bank else None
    if bank_face_key is None and not {"bank_face_offsets", "bank_source_id"}.issubset(bank):
        raise ValueError("pinned Boolean bank input has no source face array")
    bank_points_key = "points_world_f32" if "points_world_f32" in bank else "points_world_m_f64" if "points_world_m_f64" in bank else None
    if bank_face_key is not None:
        bank_faces = np.asarray(bank[bank_face_key], dtype=np.int64)
        if bank_faces.ndim != 2 or bank_faces.shape[1] != 3:
            raise ValueError("pinned bank input triangle domain is malformed")
        if bank_points_key is None:
            raise ValueError("pinned replay bank lacks its actual input coordinates")
        bank_points = np.asarray(bank[bank_points_key], dtype=np.float64)
        if (bank_points.ndim != 2 or bank_points.shape[1] != 3 or not np.isfinite(bank_points).all() or
            (len(bank_faces) and (bank_faces.min() < 0 or bank_faces.max() >= len(bank_points)))):
            raise ValueError("pinned replay bank input geometry is malformed")
    run_sources = document.get("run_sources")
    face_rows = document.get("boolean_faces")
    if not isinstance(run_sources, list) or not isinstance(face_rows, list) or len(face_rows) != len(bf):
        raise ValueError("Boolean face-lineage table does not cover every output triangle")
    source_by_run = {}
    for row in run_sources:
        try:
            run_id = int(row["run_original_id"])
            domain = row["source_domain"]
            source_sha = row["source_mesh_sha256"]
            face_count = int(row["input_face_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Boolean run-source row is malformed") from exc
        if run_id in source_by_run:
            raise ValueError("Boolean run-source table repeats a source ID")
        if domain not in ("source_shell", "bone_bank"):
            raise ValueError("Boolean run-source domain is unsupported")
        expected_sha = expected_pins["source_shell_npz_sha256"] if domain == "source_shell" else expected_pins["bank_input_npz_sha256"]
        if domain == "source_shell":
            actual_count = shell_face_count
        elif "bank_face_offsets" in bank:
            bank_ids = np.asarray(bank["bank_source_id"], dtype=np.int64)
            matches = np.flatnonzero(bank_ids == run_id)
            if len(matches) != 1:
                raise ValueError("Boolean bank run ID does not select exactly one pinned source bank")
            bi = int(matches[0]); bs, be = map(int, bank["bank_face_offsets"][bi:bi+2])
            actual_count = be - bs
            if "bank_body" in bank and "source_body" in row and int(bank["bank_body"][bi]) != int(row["source_body"]):
                raise ValueError("Boolean bank body identity differs from the pinned replay")
            if "bank_bone" in bank and "source_bone" in row and int(bank["bank_bone"][bi]) != int(row["source_bone"]):
                raise ValueError("Boolean bank bone identity differs from the pinned replay")
            if "face_source_operand_face_row" not in bank:
                raise ValueError("pinned replay bank lacks its exact input face-tag mapping")
            bank_tag_rows = np.asarray(bank["face_source_operand_face_row"][bs:be], dtype=np.int64)
            if len(bank_tag_rows) != actual_count or len(np.unique(bank_tag_rows)) != len(bank_tag_rows):
                raise ValueError("pinned replay bank input face tags are missing or repeated")
            row["_bank_face_start"] = bs
            row["_bank_tag_to_local"] = {int(tag): local for local, tag in enumerate(bank_tag_rows)}
        else:
            actual_count = len(bank[bank_face_key])
        if source_sha != expected_sha or face_count != actual_count:
            raise ValueError("Boolean run-source table differs from its pinned input mesh")
        source_by_run[run_id] = row
    if set(source_by_run) != set(map(int, run_ids)):
        raise ValueError("Boolean run-source table does not exactly cover the CSG run IDs")
    source_d_by_row = {int(row["parent_d_face_row"]): row for row in source_d_faces}
    if len(source_d_by_row) != len(source_d_faces):
        raise ValueError("original 330 D source-face rows are repeated")
    boolean_parent_d_rows = []
    maximum_tagged_plane_residual = 0.0
    maximum_tagged_plane_tolerance = 0.0
    maximum_tagged_barycentric_tolerance = 0.0
    minimum_tagged_input_barycentric = 1.0
    for row, actual_run, actual_face in zip(face_rows, face_run_ids, face_id):
        try:
            row_id = int(row["boolean_face_row"])
            declared_run = int(row["run_original_id"])
            input_face = int(row["input_face_row"])
            parent_rows = list(map(int, row["parent_d_face_rows"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Boolean output face ancestry row is malformed") from exc
        if row_id != len(boolean_parent_d_rows) or declared_run != int(actual_run) or input_face != int(actual_face):
            raise ValueError("Boolean output face row/tag differs from the actual CSG NPZ arrays")
        if not parent_rows or len(set(parent_rows)) != len(parent_rows) or any(r not in source_d_by_row for r in parent_rows):
            raise ValueError("Boolean output face has invalid or repeated 330 D source-face ancestry")
        for parent_row in parent_rows:
            parent_tet_row = int(source_d_by_row[parent_row]["parent_tet_row"])
            if parent_tet_row < 0 or parent_tet_row >= len(source_labels) or parent_tet_row >= len(source_components):
                raise ValueError("Boolean output face ancestry has an out-of-domain 330 source tet row")
        run_source = source_by_run[declared_run]
        if run_source["source_domain"] == "source_shell":
            if input_face < 0 or input_face >= int(run_source["input_face_count"]):
                raise ValueError("Boolean source face row is outside its exact input mesh")
            source_input_face_row = input_face
            source_input_tri = shell_points[shell_faces[source_input_face_row]]
        else:
            local_input_face = run_source["_bank_tag_to_local"].get(input_face)
            if local_input_face is None:
                raise ValueError("Boolean bank face tag is absent from the pinned replay input face map")
            source_input_face_row = int(run_source["_bank_face_start"]) + int(local_input_face)
            if bank_face_key is None:
                raise ValueError("pinned replay bank lacks source triangles")
            if source_input_face_row < 0 or source_input_face_row >= len(bank_faces):
                raise ValueError("Boolean bank source-face row is outside the pinned triangle array")
            source_input_tri = bank_points[bank_faces[source_input_face_row]]
        if run_source["source_domain"] == "source_shell":
            geometry_input_points = shell_points
        else:
            geometry_input_points = bank_points
        geometry_support = validate_tagged_output_triangle_geometry(
            row_id, bp[bf[row_id]], geometry_input_points,
            (shell_faces[input_face] if run_source["source_domain"] == "source_shell"
             else bank_faces[source_input_face_row]))
        maximum_tagged_plane_residual = max(maximum_tagged_plane_residual,
                                             geometry_support["maximum_plane_residual_m"])
        maximum_tagged_plane_tolerance = max(maximum_tagged_plane_tolerance,
                                              geometry_support["plane_tolerance_m"])
        maximum_tagged_barycentric_tolerance = max(maximum_tagged_barycentric_tolerance,
                                                    geometry_support["barycentric_tolerance"])
        minimum_tagged_input_barycentric = min(minimum_tagged_input_barycentric,
                                                geometry_support["minimum_input_barycentric"])
        if run_source["source_domain"] == "source_shell":
            required_shell = {"source_face_kind", "source_face_global_nodes", "source_tet_rows"}
            if not required_shell.issubset(shell):
                raise ValueError("pinned source shell lacks original 330 face/tet ancestry")
            shell_kind = str(shell_kinds[input_face])
            shell_nodes = tuple(sorted(map(int, shell_source_nodes[input_face])))
            shell_child_tet = int(shell_source_tets[input_face])
            if shell_child_tet < 0 or shell_child_tet >= len(baseline_child_parent_tet):
                raise ValueError("Boolean source-shell face lacks a valid frozen006 child tet row")
            shell_tet = int(baseline_child_parent_tet[shell_child_tet])
            if shell_tet < 0 or shell_tet >= len(source_labels):
                raise ValueError("Boolean source-shell face lacks a valid original 330 source tet row")
            if any(source_tet_label_key(source_labels[source_d_by_row[r]["parent_tet_row"]]) != source_tet_label_key(source_labels[shell_tet]) or
                   source_components[int(source_d_by_row[r]["parent_tet_row"])] != source_components[shell_tet]
                   for r in parent_rows):
                raise ValueError("Boolean shell-to-D ancestry is outside the input source tet label/component")
            if shell_kind == "existing_D_material_boundary":
                matching = [r for r, face in source_d_by_row.items()
                            if tuple(face["node_rows"]) == shell_nodes]
                if len(matching) != 1:
                    raise ValueError("Boolean D-boundary shell tag is not an exact original 330 D face")
                if matching[0] not in parent_rows:
                    raise ValueError("Boolean D-boundary shell face omits its exact original 330 D-face ancestry")
        else:
            source_row = source_input_face_row
            stable_key = "face_source_stable_row" if "face_source_stable_row" in bank else "bank_stable_face_row" if "bank_stable_face_row" in bank else None
            if stable_key is None:
                raise ValueError("pinned bank input lacks stable source face identity")
            if int(row.get("source_stable_face_row", -1)) != int(bank[stable_key][source_row]):
                raise ValueError("Boolean bank face stable-row ancestry differs from pinned input")
            if "face_source_record_index" in bank and "_bank_face_start" in run_source:
                if int(row.get("source_record_index", -1)) != int(bank["face_source_record_index"][source_row]):
                    raise ValueError("Boolean bank face record-index ancestry differs from pinned replay")
            if "face_source_operand_face_row" in bank and "_bank_face_start" in run_source:
                if int(row.get("source_operand_face_row", -1)) != int(bank["face_source_operand_face_row"][source_row]):
                    raise ValueError("Boolean bank face operand-row ancestry differs from pinned replay")
        if len({source_tet_label_key(source_labels[source_d_by_row[r]["parent_tet_row"]]) for r in parent_rows}) != 1:
            raise ValueError("Boolean output face ancestry spans incompatible source labels")
        if len({source_components[int(source_d_by_row[r]["parent_tet_row"])] for r in parent_rows}) != 1:
            raise ValueError("Boolean output face ancestry spans disconnected 330 source components")
        boolean_parent_d_rows.append(set(parent_rows))
    child_rows = document.get("child_face_lineage")
    if not isinstance(child_rows, list):
        raise ValueError("Boolean face-lineage table lacks candidate child-face mapping")
    child_lineage = {}
    used_boolean_rows = set()
    for row in child_rows:
        try:
            child_node_order = tuple(map(int, row["child_node_rows"]))
            tri = tuple(sorted(child_node_order))
            bool_rows = list(map(int, row["boolean_face_rows"]))
            parent_rows = list(map(int, row["parent_d_face_rows"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("child D face Boolean ancestry row is malformed") from exc
        if (len(tri) != 3 or len(set(tri)) != 3 or tri in child_lineage or
            min(tri) < 0 or max(tri) >= len(candidate_points)):
            raise ValueError("child D face Boolean ancestry repeats or malforms a target face")
        if not bool_rows or len(set(bool_rows)) != len(bool_rows) or any(r < 0 or r >= len(boolean_parent_d_rows) for r in bool_rows):
            raise ValueError("child D face lacks valid actual Boolean output-face ancestry")
        if len(bool_rows) != 1:
            raise ValueError("D retirement requires one exact facet-preserving Boolean face per changed child face")
        boolean_row = bool_rows[0]
        if boolean_row in used_boolean_rows:
            raise ValueError("one Boolean output face is reused by multiple child D facets")
        if oriented_f32_triangle_key(candidate_points, child_node_order) != oriented_f32_triangle_key(bp, bf[boolean_row]):
            raise ValueError("candidate D child face is not the exact oriented emitted-F32 Boolean facet")
        used_boolean_rows.add(boolean_row)
        supported = set().union(*(boolean_parent_d_rows[r] for r in bool_rows))
        if not parent_rows or len(set(parent_rows)) != len(parent_rows) or set(parent_rows) != supported:
            raise ValueError("child D face parent CSR differs from its actual Boolean face support")
        if any(r not in source_d_by_row for r in parent_rows):
            raise ValueError("child D face ancestry names an unknown original 330 D face")
        child_lineage[tri] = {"boolean_face_rows": tuple(bool_rows),
                              "parent_d_face_rows": frozenset(parent_rows),
                              "child_node_order": child_node_order}
    return child_lineage, {"boolean_output_face_count": len(bf),
                           "boolean_lineage_face_rows": len(boolean_parent_d_rows),
                           "child_d_face_lineage_rows": len(child_lineage),
                           "boolean_output_faces_checked_against_tagged_input_geometry": len(boolean_parent_d_rows),
                           "child_faces_bound_to_exact_oriented_f32_boolean_facets": len(child_lineage),
                           "source_face_support_plane_tolerance_policy": "256*float64_epsilon*max(abs(world_coordinate),input_edge_length)",
                           "source_face_support_maximum_plane_residual_m": maximum_tagged_plane_residual,
                           "source_face_support_maximum_plane_tolerance_m": maximum_tagged_plane_tolerance,
                           "source_face_support_minimum_input_barycentric": minimum_tagged_input_barycentric,
                           "source_face_support_maximum_barycentric_tolerance": maximum_tagged_barycentric_tolerance,
                           "candidate_coordinates_exactly_widened_float32": True,
                           "source_connected_material_donor_region_proof": True,
                           "physical_admission": False}


D_RETIREMENT_MAX_PARENT_RESIDUAL_M = 4.0e-4


def validate_parent_registration_distance_m(value):
    try:
        distance = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("maximum parent registration distance must be a positive finite value in meters") from exc
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("maximum parent registration distance must be a positive finite value in meters")
    return distance


def parse_parent_registration_distance_m(value):
    try:
        return validate_parent_registration_distance_m(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def add_parent_registration_distance_argument(parser):
    parser.add_argument(
        "--maximum-parent-registration-distance-m",
        type=parse_parent_registration_distance_m,
        default=D_RETIREMENT_MAX_PARENT_RESIDUAL_M,
        metavar="METERS",
        help="maximum original-330 parent registration residual for newly constructed D nodes (default: 0.0004)",
    )


def unchanged_preexisting_derived_node_rows(baseline_parent_node_by_child, baseline_points_f32,
                                             baseline_ancestry_rows, candidate_parent_node_by_child,
                                             candidate_points, candidate_node_ancestry):
    """Return only current006 inserted nodes preserved byte-for-byte in the candidate."""
    import numpy as np
    baseline_parent=np.asarray(baseline_parent_node_by_child,dtype=np.int64)
    baseline_points=np.asarray(baseline_points_f32)
    candidate_parent=np.asarray(candidate_parent_node_by_child,dtype=np.int64)
    candidate_points=np.asarray(candidate_points,dtype=np.float64)
    if (baseline_parent.ndim!=1 or baseline_points.shape!=(len(baseline_parent),3) or
        baseline_points.dtype!=np.float32 or candidate_parent.ndim!=1 or
        candidate_points.shape!=(len(candidate_parent),3) or len(candidate_parent)<len(baseline_parent) or
        not np.isfinite(baseline_points).all() or not np.isfinite(candidate_points).all()):
        raise ValueError("preexisting derived-node comparison has invalid array domains: "+
                         repr((baseline_parent.shape,baseline_points.shape,str(baseline_points.dtype),candidate_parent.shape,candidate_points.shape)))
    if not np.array_equal(candidate_points,candidate_points.astype(np.float32).astype(np.float64)):
        delta=float(np.max(np.abs(candidate_points-candidate_points.astype(np.float32).astype(np.float64))))
        raise ValueError("candidate derived-node points are not widened F32: max_delta="+repr(delta))
    baseline_ancestry={}
    for row in baseline_ancestry_rows:
        try:
            node=int(row["volume_node_row"])
            value=(int(row["parent_tet_row"]),tuple(float(row["parent_bary"+str(k)]) for k in range(4)))
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError("baseline derived-node ancestry row is malformed") from exc
        if node in baseline_ancestry:
            raise ValueError("baseline derived-node ancestry repeats a volume row")
        baseline_ancestry[node]=value
    unchanged=set()
    for node in np.flatnonzero(baseline_parent<0):
        node=int(node)
        if node>=len(candidate_parent) or node not in baseline_ancestry:
            continue
        if candidate_parent[node]>=0 or node not in candidate_node_ancestry:
            continue
        if not np.array_equal(baseline_points[node].view(np.uint32),
                              candidate_points[node].astype(np.float32).view(np.uint32)):
            continue
        try:
            candidate_value=(int(candidate_node_ancestry[node][0]),
                             tuple(map(float,candidate_node_ancestry[node][1])))
        except (IndexError,TypeError,ValueError) as exc:
            raise ValueError("candidate derived-node ancestry row is malformed") from exc
        if candidate_value==baseline_ancestry[node]:
            unchanged.add(node)
    return unchanged


def validate_d_new_node_source_residuals(points, parent_points, parent_tets,
                                         parent_node_by_child, selected_parent,
                                         role_by_node, maximum_m=D_RETIREMENT_MAX_PARENT_RESIDUAL_M,
                                         preexisting_derived_node_rows=()):
    """Keep newly constructed D boundary nodes on the existing 330 source map."""
    maximum_m = validate_parent_registration_distance_m(maximum_m)
    preexisting=set(map(int,preexisting_derived_node_rows))
    node_rows = [int(node) for node, (role, _role_row) in role_by_node.items()
                 if role == "D" and int(parent_node_by_child[int(node)]) < 0 and int(node) not in preexisting]
    maximum = 0.0
    for node in node_rows:
        if node < 0 or node >= len(points) or node >= len(selected_parent):
            raise ValueError("new D source-registration node is outside the candidate domain")
        parent_tet_row, beta = selected_parent[node]
        parent_tet_row = int(parent_tet_row)
        beta = tuple(map(float, beta))
        if (parent_tet_row < 0 or parent_tet_row >= len(parent_tets) or len(beta) != 4 or
            not all(math.isfinite(x) for x in beta) or min(beta) < -1e-8 or
            max(beta) > 1.0 + 1e-8 or abs(sum(beta) - 1.0) > 1e-8):
            raise ValueError("new D node lacks valid original 330 tet/barycentric ancestry")
        tet = parent_tets[parent_tet_row]
        source_position = tuple(sum(beta[k] * parent_points[int(tet[k])][axis]
                                    for k in range(4)) for axis in range(3))
        distance = norm(sub(points[node], source_position))
        if not math.isfinite(distance) or distance > maximum_m:
            if maximum_m == D_RETIREMENT_MAX_PARENT_RESIDUAL_M:
                raise ValueError(f"new D node {node} exceeds the 0.4 mm original-source registration bound")
            raise ValueError(
                f"new D node {node} exceeds the declared original-source registration bound ({maximum_m:.17g} m)")
        maximum = max(maximum, distance)
    return {"new_d_boundary_node_count": len(node_rows),
            "new_d_boundary_node_maximum_source_residual_m": maximum,
            "new_d_boundary_node_maximum_source_residual_limit_m": maximum_m,
            "declared_maximum_parent_registration_distance_m": maximum_m,
            "preexisting_current006_derived_node_count_excluded_from_new_node_gate": len(
                preexisting & {int(node) for node,(role,_row) in role_by_node.items() if role=="D"}),
            "new_d_boundary_nodes_parent_tet_bary_and_full86_prevalidated": True}


def _oriented_triangle_cycle_key(nodes):
    tri = tuple(map(int, nodes))
    if len(tri) != 3 or len(set(tri)) != 3:
        raise ValueError("oriented D boundary facet is malformed")
    return min((tri[i:] + tri[:i]) for i in range(3))


FROZEN006_RECEIPT_SHA256 = "74fcbaf6a6776067d8d2fd90158afa3b1b9c815226ce3d60ba612ab5e9625fbf"
REFINEMENT530_SHA256 = "bc3831a56663cd561d5465d5943eb7cab275ca82b3c0be92561d62c42e8e8606"
LEDGER534_SHA256 = "ed320a79755132a7418cd7dafbb97d034b491ee5ccba587e33324bc7e61b0210"
FROZEN006_CANDIDATE_SHA256 = "888eb12450d0e69fb0ad3f5998159ab7a0a20d3e70b13c42ba416c61cbbede63"


def oriented_f32_coordinate_triangle_key(points, nodes):
    import numpy as np
    tri=np.asarray(nodes,dtype=np.int64)
    coords=np.asarray(points,dtype=np.float32)
    if (coords.ndim!=2 or coords.shape[1]!=3 or tri.shape!=(3,) or
        np.any(tri<0) or np.any(tri>=len(coords)) or len(set(map(int,tri)))!=3):
        raise ValueError("oriented F32 coordinate triangle is malformed")
    raw=coords[tri].view(np.uint32)
    if not np.isfinite(coords[tri]).all():
        raise ValueError("oriented F32 coordinate triangle is nonfinite")
    rows=[tuple(int(v) for v in row) for row in raw]
    return min(tuple(rows[i:]+rows[:i]) for i in range(3))


def collect_frozen006_d_face_lineage(baseline_path, source_d_faces, source_labels):
    """Authenticate unchanged current006 D facets through pinned 530/534 lineage."""
    import numpy as np
    from collections import defaultdict
    baseline_path=pathlib.Path(baseline_path).resolve()
    if digest(baseline_path)!=FROZEN006_CANDIDATE_SHA256:
        raise ValueError("frozen current006 D-facet baseline hash mismatch")
    receipt_path=baseline_path.parent/"receipt.json"
    if digest(receipt_path)!=FROZEN006_RECEIPT_SHA256:
        raise ValueError("frozen current006 refinement receipt hash mismatch")
    receipt=json.loads(receipt_path.read_text())
    physical_root=baseline_path.parents[3]
    refine_path=physical_root/"conforming-refinement-control-530/run-001/refinement-connectivity.npz"
    ledger_path=physical_root/"refined-reference-ledger-534/run-001/child-parent-reference-ledger.npz"
    source_path=physical_root/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz"
    if (digest(refine_path)!=REFINEMENT530_SHA256 or digest(ledger_path)!=LEDGER534_SHA256 or
        receipt.get("candidate_sha256")!=FROZEN006_CANDIDATE_SHA256 or
        receipt.get("inputs",{}).get("530_refinement_control")!=REFINEMENT530_SHA256 or
        receipt.get("inputs",{}).get("534_reference_mass_ledger")!=LEDGER534_SHA256 or
        receipt.get("inputs",{}).get("source002_candidate")!="2f547b21c6be0f3e21a021d80dad6c63c3a0e6af31d5ebe1780ca1b32c237a54" or
        digest(source_path)!=receipt.get("inputs",{}).get("source002_candidate")):
        raise ValueError("current006 is not bound to the pinned 530/534 source chain")
    with np.load(baseline_path,allow_pickle=False) as z:
        base={k:np.asarray(z[k]) for k in z.files}
    with np.load(refine_path,allow_pickle=False) as z:
        refine={k:np.asarray(z[k]) for k in z.files}
    with np.load(ledger_path,allow_pickle=False) as z:
        ledger={k:np.asarray(z[k]) for k in z.files}
    pts=base.get("points_current_world_m_f32")
    tets=base.get("volume_tetrahedra")
    child_parent=base.get("child_parent_tet_row")
    parent_node=base.get("parent_volume_node_row")
    faces=base.get("coarse_boundary_faces_D")
    offsets=base.get("refinement_node_source_offsets")
    source_rows=base.get("refinement_node_source_rows")
    source_weights=base.get("refinement_node_source_weights")
    if (pts is None or pts.ndim!=2 or pts.shape[1]!=3 or pts.dtype!=np.float32 or
        tets is None or tets.ndim!=2 or tets.shape[1]!=4 or
        child_parent is None or parent_node is None or faces is None or
        offsets is None or source_rows is None or source_weights is None):
        raise ValueError("frozen current006 lacks typed D-facet source ancestry")
    if (not np.isfinite(pts).all() or tets.min()<0 or tets.max()>=len(pts) or
        child_parent.shape!=(len(tets),) or parent_node.shape!=(len(pts),) or
        faces.ndim!=2 or faces.shape[1]!=3 or faces.min()<0 or faces.max()>=len(pts)):
        raise ValueError("frozen current006 D-facet row domains are malformed")
    if (not np.array_equal(tets,refine.get("tetrahedra")) or
        not np.array_equal(child_parent,refine.get("child_source_tet_rows")) or
        not np.array_equal(base.get("refinement_child_source_tet_rows"),refine.get("child_source_tet_rows")) or
        not np.array_equal(offsets,refine.get("node_source_offsets")) or
        not np.array_equal(source_rows,refine.get("node_source_rows")) or
        not np.array_equal(source_weights,refine.get("node_source_weights")) or
        not np.array_equal(base.get("refinement_child_source_measure_fractions"),refine.get("child_source_measure_fractions"))):
        raise ValueError("frozen current006 cell/node ancestry differs from pinned 530 refinement")
    if (not np.array_equal(ledger.get("child_parent_rows"),child_parent) or
        not np.array_equal(ledger.get("source_child_fraction"),base.get("refinement_child_source_measure_fractions"))):
        raise ValueError("frozen current006 source-tet/measure ancestry differs from pinned 534 ledger")
    for key in ("material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"):
        value=base.get(key)
        if value is None or value.shape!=(len(tets),):
            raise ValueError("frozen current006 source label domain is malformed")
        expected=np.asarray([int(row[key]) for row in source_labels],dtype=np.int64)[child_parent]
        if not np.array_equal(value,expected) or not np.array_equal(ledger.get(key),value):
            raise ValueError("frozen current006 labels differ from authenticated 330/534 source ancestry")
    if (offsets.shape!=(len(pts)+1,) or offsets[0]!=0 or offsets[-1]!=len(source_rows) or
        np.any(offsets[1:]<offsets[:-1]) or source_weights.shape!=source_rows.shape or
        np.any(source_rows<0) or np.any(source_rows>=len(source_labels)) or
        not np.isfinite(source_weights).all() or np.any(source_weights<0)):
        raise ValueError("frozen current006 node-source CSR is malformed")
    source_by_tet=defaultdict(list)
    for face in source_d_faces:
        source_by_tet[int(face["parent_tet_row"])].append(face)
    incidence=defaultdict(list)
    for child_row,tet in enumerate(tets):
        for corner in CORNER_FACES:
            tri=tuple(int(tet[k]) for k in corner)
            incidence[tuple(sorted(tri))].append((child_row,tri))
    by_geometry={};covered_source_rows=set()
    facet_records=[]
    for baseline_row,stored_tri in enumerate(faces):
        key=tuple(sorted(map(int,stored_tri)))
        uses=incidence.get(key,[])
        if len(uses)!=1:
            raise ValueError("frozen current006 D facet lacks one incident child tet")
        child_row,outward_tri=uses[0]
        if _oriented_triangle_cycle_key(stored_tri)!=_oriented_triangle_cycle_key(outward_tri):
            raise ValueError("frozen current006 D facet orientation differs from its outward child tet")
        parent_row=int(child_parent[child_row])
        support=[]
        for node in outward_tri:
            lo,hi=map(int,offsets[int(node):int(node)+2])
            if hi<=lo or abs(float(source_weights[lo:hi].sum())-1.0)>1e-10:
                raise ValueError("frozen current006 D-facet node lacks normalized 530 source weights")
            support.append(set(map(int,source_rows[lo:hi][source_weights[lo:hi]!=0])))
        matches=[face for face in source_by_tet[parent_row]
                 if all(nodes and nodes.issubset(set(face["node_rows"])) for nodes in support)]
        if len(matches)!=1:
            raise ValueError("frozen current006 D facet lacks unique 330 parent-face ancestry")
        source=matches[0]
        for key_name in ("material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"):
            if int(base[key_name][child_row])!=int(source[key_name]):
                raise ValueError("frozen current006 D-facet source label differs from its 330 parent face")
        triangle_key=oriented_f32_coordinate_triangle_key(pts,outward_tri)
        if triangle_key in by_geometry:
            raise ValueError("frozen current006 repeats an oriented D facet coordinate triangle")
        record={"baseline_d_face_row":baseline_row,"child_node_order":outward_tri,
                "child_tet_row":child_row,"parent_tet_row":parent_row,
                "parent_d_face_rows":(int(source["parent_d_face_row"]),),
                "source_label":tuple(int(source[k]) for k in
                    ("material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"))}
        by_geometry[triangle_key]=record
        facet_records.append(record)
        covered_source_rows.add(int(source["parent_d_face_row"]))
    expected_source_rows={int(face["parent_d_face_row"]) for face in source_d_faces}
    if covered_source_rows!=expected_source_rows:
        raise ValueError("authenticated current006 D facets do not cover every original 330 D source face")
    return {"by_oriented_f32_triangle":by_geometry,"facet_records":facet_records,
            "summary":{"frozen_current006_sha256":FROZEN006_CANDIDATE_SHA256,
                "frozen_current006_receipt_sha256":FROZEN006_RECEIPT_SHA256,
                "refinement530_sha256":REFINEMENT530_SHA256,"reference_ledger534_sha256":LEDGER534_SHA256,
                "frozen_current006_d_facet_count":len(facet_records),
                "frozen_current006_d_facets_with_unique_330_parent_face":len(facet_records),
                "unique_330_d_source_faces_covered":len(covered_source_rows),
                "authenticated_530_534_source_face_ancestry":True}}


def validate_d_boundary_face_retirement(document, source_d_faces, child_d_faces,
                                         parent_node_by_child, source_role_by_node,
                                         source_labels, source_components, boolean_child_lineage,
                                         candidate_points=None, frozen_baseline_d_faces=None):
    old_by_nodes = {tuple(face["node_rows"]): face for face in source_d_faces}
    if len(old_by_nodes) != len(source_d_faces):
        raise ValueError("parent D source-face rows are not unique")
    child_by_nodes = {}
    for face in child_d_faces:
        nodes = tuple(sorted(map(int, face["node_rows"])))
        if nodes in child_by_nodes:
            raise ValueError("candidate D boundary repeats a face")
        child_by_nodes[nodes] = face
    retained = set()
    retained_frozen006_rows=set()
    covered_source_rows=set()
    changed = {}
    if frozen_baseline_d_faces is not None:
        if candidate_points is None:
            raise ValueError("frozen006 D-facet retention requires emitted candidate coordinates")
        candidate_points=__import__("numpy").asarray(candidate_points,dtype=float)
        if not __import__("numpy").array_equal(candidate_points,candidate_points.astype(__import__("numpy").float32).astype(float)):
            raise ValueError("candidate D-facet retention coordinates are not widened emitted-F32")
    for nodes, face in child_by_nodes.items():
        baseline_face=None
        if frozen_baseline_d_faces is not None:
            coord_key=oriented_f32_coordinate_triangle_key(candidate_points,face["node_rows"])
            baseline_face=frozen_baseline_d_faces["by_oriented_f32_triangle"].get(coord_key)
        if baseline_face is not None:
            if int(face["parent_tet_row"])!=int(baseline_face["parent_tet_row"]):
                raise ValueError("unchanged frozen006 D facet changed its authenticated 330 parent tet")
            child_label=source_labels[int(face["parent_tet_row"])]
            if tuple(int(child_label[k]) for k in ("material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id"))!=baseline_face["source_label"]:
                raise ValueError("unchanged frozen006 D facet changed its authenticated source label")
            retained_frozen006_rows.add(int(baseline_face["baseline_d_face_row"]))
            covered_source_rows.update(baseline_face["parent_d_face_rows"])
            continue
        old_nodes = tuple(sorted(int(parent_node_by_child[node]) for node in nodes))
        if frozen_baseline_d_faces is None and all(
                parent_node >= 0 and source_role_by_node.get(parent_node, (None, None))[0] == "D"
                for parent_node in old_nodes) and old_nodes in old_by_nodes:
            source = old_by_nodes[old_nodes]
            if source_tet_label_key(source_labels[int(face["parent_tet_row"])]) != source_tet_label_key(source_labels[int(source["parent_tet_row"])]):
                raise ValueError("retained D face changed its source material/donor/region")
            retained.add(int(source["parent_d_face_row"]))
            covered_source_rows.add(int(source["parent_d_face_row"]))
        else:
            changed[nodes] = face
    required_source_rows={int(face["parent_d_face_row"]) for face in source_d_faces}
    if set(boolean_child_lineage) != set(changed):
        raise ValueError("Boolean child-face lineage does not exactly cover changed D boundary facets")
    for target, face in changed.items():
        lineage = boolean_child_lineage[target]
        if _oriented_triangle_cycle_key(face["node_rows"]) != _oriented_triangle_cycle_key(lineage["child_node_order"]):
            raise ValueError("Boolean child-face orientation differs from the actual outward D boundary facet")
        covered_source_rows.update(map(int,lineage["parent_d_face_rows"]))
    retired_by_node = {int(r["volume_node_row"]) for r in document["retired_parent_d_nodes"]}
    retired_faces_by_node = defaultdict(set)
    for face in source_d_faces:
        for node in face["node_rows"]:
            if node in retired_by_node:
                retired_faces_by_node[node].add(int(face["parent_d_face_row"]))
    required_replacement_rows=set().union(*(set(map(int,x["parent_d_face_rows"])) for x in boolean_child_lineage.values())) if boolean_child_lineage else set()
    for incident_faces in retired_faces_by_node.values():
        required_replacement_rows.update(incident_faces)
    entries = document.get("replaced_parent_d_faces")
    if not isinstance(entries, list):
        raise ValueError("D retirement map lacks replaced parent D face coverage")
    declared_rows = set(); target_rows = set(); source_by_row = {int(x["parent_d_face_row"]): x for x in source_d_faces}
    target_sources = defaultdict(list)
    for entry in entries:
        try:
            source_row = int(entry["parent_d_face_row"])
            declared_stable = int(entry["source_stable_source_face_id"])
            targets = entry["replacement_child_node_rows"]
            boolean_targets = entry["replacement_boolean_face_rows"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("replacement D face ancestry row is malformed") from exc
        if source_row in declared_rows:
            raise ValueError("parent D source face is mapped more than once")
        declared_rows.add(source_row)
        source = source_by_row.get(source_row)
        if source is None or source_row not in required_replacement_rows:
            raise ValueError("D retirement map names a parent D face absent from changed Boolean/retired-node ancestry")
        if declared_stable != source["stable_source_face_id"]:
            raise ValueError("D retirement map source stable-face history does not match the parent face")
        if (int(entry.get("source_material_index", -1)) != source["material_index"] or
            int(entry.get("source_geometric_donor_body", -1)) != source["geometric_donor_body"] or
            int(entry.get("source_anatomical_region_id", -1)) != source["anatomical_region_id"]):
            raise ValueError("D retirement map source material/donor/region does not match the parent face")
        if not isinstance(targets, list) or not targets or not isinstance(boolean_targets, list) or len(boolean_targets) != len(targets):
            raise ValueError("retired parent D face has incomplete replacement/Boolean boundary ancestry")
        for target_index,raw_target in enumerate(targets):
            if not isinstance(raw_target, list) or len(raw_target) != 3:
                raise ValueError("replacement D face must name exactly three child volume node rows")
            target = tuple(sorted(map(int, raw_target)))
            if len(set(target)) != 3 or target not in changed:
                raise ValueError("D replacement is not an actual new D boundary face")
            face = changed[target]
            lineage = boolean_child_lineage.get(target)
            if lineage is None:
                raise ValueError("new D boundary face lacks pinned actual Boolean ancestry")
            declared_boolean_rows = tuple(map(int, boolean_targets[target_index]))
            if declared_boolean_rows != lineage["boolean_face_rows"]:
                raise ValueError("D retirement map Boolean face rows differ from the pinned child-face lineage")
            target_rows.add(target); target_sources[target].append(source)
            source_key = source_tet_label_key(source_labels[int(source["parent_tet_row"])])
            child_parent_row = int(face["parent_tet_row"])
            if source_tet_label_key(source_labels[child_parent_row]) != source_key:
                raise ValueError("replacement D face changes material/donor/region source ownership")
            if source_components[child_parent_row] != source_components[int(source["parent_tet_row"])] :
                raise ValueError("replacement D face is outside the source-connected parent-tet component")
            if source_row not in lineage["parent_d_face_rows"]:
                raise ValueError("replacement D face Boolean CSR omits the retired source face")
    if declared_rows != required_replacement_rows:
        raise ValueError("D retirement map does not cover every changed Boolean/retired-node source face")
    if target_rows != set(changed):
        raise ValueError("D retirement map does not cover every new D boundary face")
    if set(boolean_child_lineage) != set(changed):
        raise ValueError("Boolean face-lineage table does not exactly cover the changed D boundary")
    if covered_source_rows != required_source_rows:
        raise ValueError("D retirement and unchanged-facet ancestry do not cover every 330 source D face")
    for node, incident_faces in retired_faces_by_node.items():
        if not incident_faces or not incident_faces.issubset(covered_source_rows):
            raise ValueError("retired parent D node has source faces absent from retained/replacement coverage")
    for target, sources in target_sources.items():
        first_key = source_tet_label_key(source_labels[int(sources[0]["parent_tet_row"])])
        first_component = source_components[int(sources[0]["parent_tet_row"])]
        if any(source_tet_label_key(source_labels[int(source["parent_tet_row"])]) != first_key or
               source_components[int(source["parent_tet_row"])] != first_component for source in sources[1:]):
            raise ValueError("replacement D face is mapped across incompatible source domains")
    return {"retired_parent_d_node_count": len(retired_by_node),
            "retired_parent_d_face_count": len(declared_rows),
            "replacement_d_face_count": len(changed),
            "retained_parent_d_face_count": len(covered_source_rows-declared_rows),
            "retained_frozen_current006_d_facet_count": len(retained_frozen006_rows),
            "covered_original_330_d_source_face_count": len(covered_source_rows),
            "source_connected_material_donor_region_proof": True,
            "actual_boolean_face_ancestry_proof": True,
            "physical_admission": False}

def source_tet_label_key(label):
    return (int(label["material_index"]),
            int(label["geometric_donor_body"]),
            int(label["anatomical_region_id"]))


def parent_tet_source_components(parent_tets, source_labels):
    face_uses = defaultdict(list)
    for tet_row, tet in enumerate(parent_tets):
        for corner in CORNER_FACES:
            tri = tuple(int(tet[k]) for k in corner)
            face_uses[tuple(sorted(tri))].append(tet_row)
    adjacency = [set() for _ in parent_tets]
    for uses in face_uses.values():
        if len(uses) > 2:
            raise ValueError("parent capture has a nonmanifold face while building source ancestry")
        if len(uses) == 2:
            a, b = uses
            if source_tet_label_key(source_labels[a]) == source_tet_label_key(source_labels[b]):
                adjacency[a].add(b)
                adjacency[b].add(a)
    component = [-1] * len(parent_tets)
    next_component = 0
    for start in range(len(parent_tets)):
        if component[start] >= 0:
            continue
        key = source_tet_label_key(source_labels[start])
        stack = [start]
        component[start] = next_component
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if component[neighbor] < 0 and source_tet_label_key(source_labels[neighbor]) == key:
                    component[neighbor] = next_component
                    stack.append(neighbor)
        next_component += 1
    return component


def validate_o_boundary_face_retirement(document, source_o_faces, child_o_faces,
                                         parent_node_by_child, source_role_by_node,
                                         source_labels, source_components):
    old_by_nodes = {tuple(face["node_rows"]): face for face in source_o_faces}
    if len(old_by_nodes) != len(source_o_faces):
        raise ValueError("parent O source-face rows are not unique")
    child_by_nodes = {}
    for face in child_o_faces:
        nodes = tuple(sorted(map(int, face["node_rows"])))
        if nodes in child_by_nodes:
            raise ValueError("candidate O boundary repeats a face")
        child_by_nodes[nodes] = face

    retained = set()
    changed = {}
    for nodes, face in child_by_nodes.items():
        old_nodes = tuple(sorted(int(parent_node_by_child[node]) for node in nodes))
        if all(parent_node >= 0 and source_role_by_node.get(parent_node, (None, None))[0] == "O"
               for parent_node in old_nodes) and old_nodes in old_by_nodes:
            source = old_by_nodes[old_nodes]
            if source_tet_label_key(source_labels[int(face["parent_tet_row"])]) != source_tet_label_key(source_labels[int(source["parent_tet_row"])]):
                raise ValueError("retained O face changed its source material/donor/region")
            retained.add(int(source["parent_o_face_row"]))
        else:
            changed[nodes] = face

    missing_source_rows = {int(face["parent_o_face_row"]) for face in source_o_faces} - retained
    entries = document.get("replaced_parent_o_faces")
    if not isinstance(entries, list):
        raise ValueError("retirement map lacks replaced parent O face coverage")
    declared_rows = set()
    target_rows = set()
    retired_by_node = {int(row["volume_node_row"]) for row in document["retired_parent_o_nodes"]}
    retired_faces_by_node = defaultdict(set)
    source_by_row = {int(face["parent_o_face_row"]): face for face in source_o_faces}
    for face in source_o_faces:
        for node in face["node_rows"]:
            if node in retired_by_node:
                retired_faces_by_node[node].add(int(face["parent_o_face_row"]))
    target_sources = defaultdict(list)
    for entry in entries:
        try:
            source_row = int(entry["parent_o_face_row"])
            declared_stable = int(entry["source_stable_source_face_id"])
            targets = entry["replacement_child_node_rows"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("replacement face ancestry row is malformed") from exc
        if source_row in declared_rows:
            raise ValueError("parent O source face is mapped more than once")
        declared_rows.add(source_row)
        source = source_by_row.get(source_row)
        if source is None or source_row not in missing_source_rows:
            raise ValueError("retirement map names a parent O face that was not retired")
        if declared_stable != source["stable_source_face_id"]:
            raise ValueError("retirement map source stable-face history does not match the parent face")
        if (int(entry.get("source_material_index", -1)) != source["material_index"] or
            int(entry.get("source_geometric_donor_body", -1)) != source["geometric_donor_body"] or
            int(entry.get("source_anatomical_region_id", -1)) != source["anatomical_region_id"]):
            raise ValueError("retirement map source material/donor/region does not match the parent face")
        if not isinstance(targets, list) or not targets:
            raise ValueError("retired parent O face has no replacement boundary ancestry")
        for raw_target in targets:
            if not isinstance(raw_target, list) or len(raw_target) != 3:
                raise ValueError("replacement O face must name exactly three child volume node rows")
            target = tuple(sorted(map(int, raw_target)))
            if len(set(target)) != 3 or target not in changed:
                raise ValueError("retirement map replacement is not an actual new O boundary face")
            face = changed[target]
            target_rows.add(target)
            target_sources[target].append(source)
            source_key = source_tet_label_key(source_labels[int(source["parent_tet_row"])])
            child_parent_row = int(face["parent_tet_row"])
            if source_tet_label_key(source_labels[child_parent_row]) != source_key:
                raise ValueError("replacement O face changes material/donor/region source ownership")
            if source_components[child_parent_row] != source_components[int(source["parent_tet_row"])]:
                raise ValueError("replacement O face is outside the source-connected parent-tet component")
    if declared_rows != missing_source_rows:
        raise ValueError("retirement map does not cover every removed parent O source face")
    if target_rows != set(changed):
        raise ValueError("retirement map does not cover every new O boundary face")
    for node, incident_faces in retired_faces_by_node.items():
        if not incident_faces or not incident_faces.issubset(missing_source_rows):
            raise ValueError("retired parent O node has source faces absent from retirement coverage")
    for target, sources in target_sources.items():
        first_key = source_tet_label_key(source_labels[int(sources[0]["parent_tet_row"])])
        if any(source_tet_label_key(source_labels[int(source["parent_tet_row"])]) != first_key
               for source in sources[1:]):
            raise ValueError("replacement O face is mapped across incompatible source domains")
    return {
        "retired_parent_o_node_count": len(retired_by_node),
        "retired_parent_o_face_count": len(missing_source_rows),
        "replacement_o_face_count": len(changed),
        "retained_parent_o_face_count": len(retained),
        "source_connected_material_donor_region_proof": True,
        "physical_admission": False,
    }

def export(args: argparse.Namespace) -> dict[str, Any]:
    d_parent_registration_limit_m = validate_parent_registration_distance_m(
        getattr(args, "maximum_parent_registration_distance_m", D_RETIREMENT_MAX_PARENT_RESIDUAL_M))
    capture = args.capture.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    packet = capture / "compiled-parent-cell-mass.csv"
    nodes_path = capture / "compiled-parent-cell-mass.csv.nodes.csv"
    tets_path = capture / "compiled-parent-cell-mass.csv.tets.csv"
    labels_path = capture / "compiled-parent-cell-mass.csv.labels.csv"
    weights_path = capture / "compiled-parent-cell-mass.csv.weights.csv"
    layer_path = capture / "parent-layer-rows.csv"
    pose_path = capture / "registered-skin-poses-100.json"
    inputs = [packet, nodes_path, tets_path, labels_path, weights_path, layer_path, pose_path]
    for p in inputs:
        if not p.is_file():
            raise FileNotFoundError(p)
    input_hashes = {str(p): digest(p) for p in inputs}
    poses = json.loads(pose_path.read_text())
    if len(poses.get("all_bodies", [])) != 157 or len(poses.get("bodies", [])) != 86:
        raise ValueError("parent capture must contain all 157 body poses and native 86 bindings")
    frame_by_body, binding_by_body = load_frame_table(poses)

    node_rows = read_csv(nodes_path)
    tets = read_csv(tets_path)
    labels = read_csv(labels_path)
    weights = read_csv(weights_path)
    layer_rows = read_csv(layer_path)
    if len(node_rows) != 7233 or len(tets) != 28632 or len(labels) != len(tets):
        raise ValueError("source-control capture dimensions differ from the exact 7233/28632 parent packet")
    if len(weights) != len(node_rows) or len(layer_rows) != len(node_rows):
        raise ValueError("parent layer or full86 weight rows do not cover all captured volume nodes")

    points = []
    for i, row in enumerate(node_rows):
        if int(row["parent_node_index"]) != i or int(row["compiled_global_node"]) < 0:
            raise ValueError(f"parent node row identity is invalid at {i}")
        p = tuple(f32(float(row[k])) for k in ("current_x_m", "current_y_m", "current_z_m"))
        points.append(p)
    # Preserve the original source packet. Derived nodes/cells are separate
    # construction records; they never become captured source measurements.
    import numpy as np
    source_node_rows=node_rows;source_tets=tets;source_labels=labels
    source_weights=weights;source_layer_rows=layer_rows
    parent_points=list(points)
    parent_tets=np.asarray([[int(r["parent_node"+str(k)]) for k in range(4)] for r in source_tets],np.int64)
    source_role_by_node={int(r["volume_node_row"]):(r["role"],int(r["role_row"])) for r in source_layer_rows}
    candidate_path=args.candidate_npz.resolve();node_path=args.candidate_node.resolve()
    node_map_path=args.candidate_node_map.resolve()
    anc_path=args.inserted_node_ancestry.resolve() if args.inserted_node_ancestry else None
    d_retirement_requested = getattr(args, "d_boundary_retirement_map", None) is not None
    candidate=np.load(candidate_path,allow_pickle=False)
    p=np.asarray(candidate["points_reference_world_m_f64"],np.float32).astype(np.float64)
    tt=np.asarray(candidate["volume_tetrahedra"],np.int64)
    cp=np.asarray(candidate["child_parent_tet_row"],np.int64)
    pn=np.asarray(candidate["parent_volume_node_row"],np.int64)
    ww=np.asarray(candidate["full86_weights"],float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)==0 or (len(p)<len(parent_points) and not d_retirement_requested) or not np.isfinite(p).all():
        raise ValueError("invalid refined coordinate domain")
    if tt.ndim!=2 or tt.shape[1]!=4 or tt.min()<0 or tt.max()>=len(p) or cp.shape!=(len(tt),) or cp.min()<0 or cp.max()>=len(source_tets):
        raise ValueError("invalid refined child/source-tet ancestry")
    if (pn.shape!=(len(p),) or np.any(pn < -1) or np.any(pn>=len(parent_points)) or
        len(np.unique(pn[pn>=0]))!=len(pn[pn>=0]) or
        (not d_retirement_requested and set(pn[pn>=0])!=set(range(len(parent_points))))):
        raise ValueError("refinement must preserve every original node exactly once unless explicit D retirement is enabled")
    child=np.loadtxt(node_path,skiprows=1)
    if child.shape!=(len(p),4) or not np.array_equal(child[:,0],np.arange(1,len(p)+1)) or not np.array_equal(child[:,1:],p):
        raise ValueError("canonical node file differs from exact Float32 candidate reference")
    source_w=np.asarray([[float(r["w"+str(k)]) for k in range(86)] for r in source_weights])
    if ww.shape!=(len(p),86) or not np.isfinite(ww).all() or ww.min()<0 or np.abs(ww.sum(1)-1).max()>max(1e-7,float(np.abs(source_w.sum(1)-1).max())+2e-9):
        raise ValueError("refined full86 weights fail inherited bounds")
    if not np.array_equal(ww[pn>=0],source_w[pn[pn>=0]]):raise ValueError("original full86 rows changed")
    keys=("material_index","geometric_donor_body","anatomical_region_id","stable_source_face_id")
    for key in keys:
        value=np.asarray(candidate[key],np.int64)
        expected=np.asarray([int(r[key]) for r in source_labels])[cp]
        if value.shape!=(len(tt),) or not np.array_equal(value,expected):
            raise ValueError("refined cell changed declared source semantics: "+key)
    original_role_metadata={int(r["volume_node_row"]):r for r in source_layer_rows}
    maps=read_csv(node_map_path);map_by_node={int(r["volume_node_row"]):r for r in maps}
    if len(map_by_node)!=len(maps):raise ValueError("duplicate derived node-source map row")
    inserted=read_csv(anc_path) if anc_path else []
    inserted_by_node={int(r["volume_node_row"]):r for r in inserted}
    if len(inserted_by_node)!=len(inserted):raise ValueError("duplicate inserted-node ancestry")
    if set(inserted_by_node)!=set(np.flatnonzero(pn<0).tolist()):raise ValueError("inserted node ancestry is incomplete")
    input_hashes.update({str(candidate_path):digest(candidate_path),str(node_path):digest(node_path),str(node_map_path):digest(node_map_path)})
    if anc_path:input_hashes[str(anc_path)]=digest(anc_path)
    retirement_path=getattr(args,"o_boundary_retirement_map",None)
    retirement_doc=None;source_o_faces=[];retired_parent_o_nodes=set();retirement_summary=None
    if retirement_path is not None:
        retirement_path=pathlib.Path(retirement_path).resolve()
        retirement_doc=json.loads(retirement_path.read_text())
        input_hashes[str(retirement_path)]=digest(retirement_path)
        expected_pins={
            "parent_packet_sha256":input_hashes[str(packet)],
            "parent_nodes_sha256":input_hashes[str(nodes_path)],
            "parent_tets_sha256":input_hashes[str(tets_path)],
            "parent_labels_sha256":input_hashes[str(labels_path)],
            "parent_weights_sha256":input_hashes[str(weights_path)],
            "parent_layer_rows_sha256":input_hashes[str(layer_path)],
            "native_poses_sha256":input_hashes[str(pose_path)],
            "candidate_npz_sha256":input_hashes[str(candidate_path)],
            "candidate_node_sha256":input_hashes[str(node_path)],
            "candidate_node_map_sha256":input_hashes[str(node_map_path)],
            "inserted_node_ancestry_sha256":input_hashes.get(str(anc_path)) if anc_path else None,
        }
        validate_o_boundary_retirement_pins(retirement_doc,expected_pins)
        source_o_faces=collect_parent_o_faces(parent_tets,source_labels,source_role_by_node)
        output_o_role_by_child_node={int(r["volume_node_row"]):int(r["role_row"]) for r in maps if r["role"]=="O"}
        if len(output_o_role_by_child_node)!=sum(r["role"]=="O" for r in maps):
            raise ValueError("candidate O role map repeats a volume node")
        retired_parent_o_nodes=validate_o_boundary_retirement_nodes(
            retirement_doc,source_role_by_node,pn,output_o_role_by_child_node)

    d_retirement_path=getattr(args,"d_boundary_retirement_map",None)
    d_retirement_doc=None;source_d_faces=[];retired_parent_d_nodes=set();d_retirement_summary=None
    d_boolean_summary=None;boolean_child_lineage={};protected_parent_nodes=set();protected_source_summary=None
    d_boolean_npz_path=None;d_shell_npz_path=None;d_bank_input_npz_path=None;d_boolean_lineage_path=None
    if d_retirement_path is not None:
        if len(source_role_by_node)!=len(parent_points) or set(source_role_by_node)!=set(range(len(parent_points))):
            raise ValueError("D retirement requires the exact complete 330 source role-node domain")
        d_retirement_path=pathlib.Path(d_retirement_path).resolve()
        d_boolean_npz_path=pathlib.Path(args.d_boolean_output_npz).resolve()
        d_shell_npz_path=pathlib.Path(args.d_boolean_source_shell_npz).resolve()
        d_bank_input_npz_path=pathlib.Path(args.d_boolean_bank_input_npz).resolve()
        d_boolean_lineage_path=pathlib.Path(args.d_boolean_face_lineage).resolve()
        d_retirement_baseline_path=pathlib.Path(args.d_retirement_baseline_npz).resolve()
        deep_attachment_path=capture/"parent-deep-attachments.csv"
        for path in (d_retirement_path,d_boolean_npz_path,d_shell_npz_path,d_bank_input_npz_path,d_boolean_lineage_path,d_retirement_baseline_path,deep_attachment_path):
            if not path.is_file():raise FileNotFoundError(path)
        d_retirement_doc=json.loads(d_retirement_path.read_text())
        boolean_lineage_doc=json.loads(d_boolean_lineage_path.read_text())
        deep_attachment_sha=digest(deep_attachment_path)
        if deep_attachment_sha!="a0bad61ae21a6fb952c589ab30a69c866866996c23842929d88d91576b29ba3f":
            raise ValueError("pinned parent deep-attachment source hash mismatch")
        deep_attachment_rows=read_csv(deep_attachment_path)
        protected_parent_nodes,protected_source_summary=derive_protected_parent_nodes(
            parent_tets,source_labels,source_role_by_node,deep_attachment_rows,deep_attachment_sha)
        baseline_sha=digest(d_retirement_baseline_path)
        if baseline_sha!="888eb12450d0e69fb0ad3f5998159ab7a0a20d3e70b13c42ba416c61cbbede63":
            raise ValueError("frozen current006 source-row baseline hash mismatch")
        with np.load(d_retirement_baseline_path,allow_pickle=False) as baseline_z:
            baseline_parent_rows=np.asarray(baseline_z["parent_volume_node_row"],np.int64)
        protected_rows=np.asarray(sorted(protected_parent_nodes),np.int64)
        baseline_source_rows = baseline_parent_rows[baseline_parent_rows >= 0]
        if (baseline_parent_rows.ndim != 1 or len(baseline_parent_rows) < len(parent_points) or
            np.any(baseline_parent_rows < -1) or np.any(baseline_parent_rows >= len(parent_points)) or
            not np.array_equal(np.sort(baseline_source_rows), np.arange(len(parent_points), dtype=np.int64)) or
            np.any(protected_rows >= len(baseline_parent_rows)) or
            not np.array_equal(baseline_parent_rows[protected_rows], protected_rows)):
            raise ValueError("frozen current006 does not preserve the complete source-node map and independently derived 149 protected rows")
        protected_source_summary["frozen_current006_sha256"]=baseline_sha
        protected_source_summary["frozen_current006_protected_rows_identity"]=True
        source_d_faces=collect_parent_d_faces(parent_tets,source_labels,source_role_by_node)
        frozen006_d_face_lineage=collect_frozen006_d_face_lineage(
            d_retirement_baseline_path,source_d_faces,source_labels)
        components=parent_tet_source_components(parent_tets,source_labels)
        input_hashes.update({str(d_retirement_path):digest(d_retirement_path),
            str(d_boolean_npz_path):digest(d_boolean_npz_path),str(d_shell_npz_path):digest(d_shell_npz_path),
            str(d_bank_input_npz_path):digest(d_bank_input_npz_path),str(d_boolean_lineage_path):digest(d_boolean_lineage_path),
            str(d_retirement_baseline_path):baseline_sha,str(deep_attachment_path):deep_attachment_sha})
        expected_d_pins={
            "parent_packet_sha256":input_hashes[str(packet)],
            "parent_nodes_sha256":input_hashes[str(nodes_path)],
            "parent_tets_sha256":input_hashes[str(tets_path)],
            "parent_labels_sha256":input_hashes[str(labels_path)],
            "parent_weights_sha256":input_hashes[str(weights_path)],
            "parent_layer_rows_sha256":input_hashes[str(layer_path)],
            "parent_deep_attachments_sha256":deep_attachment_sha,
            "frozen_current006_baseline_sha256":baseline_sha,
            "native_poses_sha256":input_hashes[str(pose_path)],
            "candidate_npz_sha256":input_hashes[str(candidate_path)],
            "candidate_node_sha256":input_hashes[str(node_path)],
            "candidate_node_map_sha256":input_hashes[str(node_map_path)],
            "inserted_node_ancestry_sha256":input_hashes.get(str(anc_path)) if anc_path else None,
            "o_boundary_retirement_map_sha256":input_hashes.get(str(retirement_path)) if retirement_path else None,
            "boolean_output_npz_sha256":input_hashes[str(d_boolean_npz_path)],
            "source_shell_npz_sha256":input_hashes[str(d_shell_npz_path)],
            "bank_input_npz_sha256":input_hashes[str(d_bank_input_npz_path)],
            "boolean_face_lineage_sha256":input_hashes[str(d_boolean_lineage_path)],
        }
        validate_d_boundary_retirement_pins(d_retirement_doc,expected_d_pins)
        expected_boolean_lineage_pins={k:v for k,v in expected_d_pins.items()
            if k not in ("boolean_face_lineage_sha256","o_boundary_retirement_map_sha256")}
        boolean_child_lineage,d_boolean_summary=validate_d_boolean_face_lineage(
            boolean_lineage_doc,d_boolean_npz_path,d_shell_npz_path,d_bank_input_npz_path,
            d_retirement_baseline_path,expected_boolean_lineage_pins,source_d_faces,source_labels,components,p)
        output_d_role_by_child_node={int(r["volume_node_row"]):int(r["role_row"]) for r in maps if r["role"]=="D"}
        if len(output_d_role_by_child_node)!=sum(r["role"]=="D" for r in maps):
            raise ValueError("candidate D role map repeats a volume node")
        retired_parent_d_nodes=validate_d_boundary_retirement_nodes(
            d_retirement_doc,source_role_by_node,pn,output_d_role_by_child_node,protected_parent_nodes)
    node_parent_ancestry={}
    for node,a in inserted_by_node.items():
        pt=int(a["parent_tet_row"]);beta=np.array([float(a["parent_bary"+str(k)]) for k in range(4)])
        if pt<0 or pt>=len(parent_tets) or not np.isfinite(beta).all() or beta.min()<-1e-8 or beta.max()>1+1e-8 or abs(beta.sum()-1)>1e-8:
            raise ValueError("invalid inserted-node original parent barycentrics")
        if not np.allclose(ww[node],beta@source_w[parent_tets[pt]],atol=2e-12,rtol=0):raise ValueError("inserted full86 weights differ from source ancestry")
        node_parent_ancestry[node]=(pt,tuple(map(float,beta)))
    points=[tuple(map(float,x)) for x in p]
    tets=[dict(parent_tet_row=i,material_index=int(candidate["material_index"][i]),**{"parent_node"+str(k):int(tt[i,k]) for k in range(4)}) for i in range(len(tt))]
    labels=[dict(parent_tet_row=i,original_parent_tet_row=int(cp[i]),**{key:int(candidate[key][i]) for key in keys}) for i in range(len(tt))]
    weights=[dict(vertex=i,**{"w"+str(k):float(ww[i,k]) for k in range(86)}) for i in range(len(p))]
    role_rows={};role_by_node={};local_points_by_node={};owner_by_node=[-1]*len(p)
    for role in ROLE_ORDER:
        rr=sorted((r for r in maps if r["role"]==role),key=lambda r:int(r["role_row"]))
        if not rr:raise ValueError("missing refined role "+role)
        enriched=[]
        for row,rec in enumerate(rr):
            node=int(rec["volume_node_row"])
            if int(rec["role_row"])!=row or node<0 or node>=len(p) or node in role_by_node:raise ValueError("invalid refined boundary row identity")
            old=int(pn[node])
            if old>=0:
                source_role=source_role_by_node.get(old)
                if source_role is None or source_role[0]!=role:
                    raise ValueError("original boundary source role changed")
                if role=="I" and source_role!=(role,row):
                    raise ValueError("original I boundary source role/row changed")
                if role=="D":
                    if source_role[0]!="D":
                        raise ValueError("original D boundary source role changed")
                    if d_retirement_doc is None and source_role!=(role,row):
                        raise ValueError("original D boundary source role/row changed without a D retirement map")
                if role=="O":
                    if retirement_doc is None and source_role!=(role,row):
                        raise ValueError("original O boundary source role/row changed without a retirement map")
                    if old in retired_parent_o_nodes:
                        raise ValueError("retired parent O node remains in the output O role map")
            body=int(rec["frame_owner_body"])
            if body not in frame_by_body:raise ValueError("boundary owner not in original86 binding frames")
            local=matrix_vector3(frame_by_body[body]["current_invR"],sub(points[node],frame_by_body[body]["origin"]))
            local_points_by_node[node]=local;owner_by_node[node]=body;role_by_node[node]=(role,row)
            enriched.append(dict(rec,compiled_global_node=int(source_node_rows[old]["compiled_global_node"]) if old>=0 else -1,
                                 donor_local_x_m=local[0],donor_local_y_m=local[1],donor_local_z_m=local[2]))
        role_rows[role]=enriched
    for node in range(len(points)):
        if node in role_by_node:continue
        old=int(pn[node])
        if old>=0:
            if old not in retired_parent_o_nodes:
                raise ValueError("an original boundary node lost its role without explicit O retirement")
            body=int(original_role_metadata[old]["frame_owner_body"])
        else:
            pt,_beta=node_parent_ancestry[node];body=int(source_labels[pt]["geometric_donor_body"])
        if body not in frame_by_body:raise ValueError("interior source frame is not in captured86 frames")
        owner_by_node[node]=body;role_by_node[node]=( "interior",-1)
        local_points_by_node[node]=matrix_vector3(frame_by_body[body]["current_invR"],sub(points[node],frame_by_body[body]["origin"]))
    derivation={"method":"explicit local reference refinement with original node injection, child-source cell ancestry and conservative mass mapped separately",
                "candidate_npz":str(candidate_path),"candidate_npz_sha256":digest(candidate_path),
                "parent_geometry_preserved":False,"parent_topology_preserved":False,
                "source_node_count":len(parent_points),"child_node_count":len(points),"source_cell_count":len(source_tets),"child_cell_count":len(tets),
                "coordinate_contract":"child current equals reference at canonical Float32 candidate reference; captured source current/reference remain separate",
                "reference_assembly_not_measured_individual":True}
    # Preserve the actual parent row identity. The exact canonical node set is
    # also recorded for future producers that re-order the same cells.
    parent_cell_by_nodes = {}
    cell_tets = []
    cell_material = []
    cell_labels = []
    incident: list[list[int]] = [[] for _ in points]
    for i, (tetrow, label) in enumerate(zip(tets, labels)):
        if int(tetrow["parent_tet_row"]) != i or int(label["parent_tet_row"]) != i:
            raise ValueError(f"parent tet/label row identity differs at {i}")
        nodes = tuple(int(tetrow[f"parent_node{k}"]) for k in range(4))
        if min(nodes) < 0 or max(nodes) >= len(points) or len(set(nodes)) != 4:
            raise ValueError(f"parent tetrahedron has invalid nodes at {i}")
        material = int(tetrow["material_index"])
        if material not in (15, 16) or int(label["material_index"]) != material:
            raise ValueError(f"parent material label mismatch at cell {i}")
        key = tuple(sorted(nodes))
        if key in parent_cell_by_nodes:
            raise ValueError(f"parent contains duplicate cell topology at row {i}")
        parent_cell_by_nodes[key] = i
        a, b, c, d = (points[n] for n in nodes)
        six = dot(sub(b, a), cross(sub(c, a), sub(d, a)))
        if not math.isfinite(six) or six <= 0.0:
            raise ValueError(f"parent cell is nonpositive in the captured Float32 target at row {i}")
        cell_tets.append(nodes)
        cell_material.append(material)
        cell_labels.append(label)
        for node in nodes:
            incident[node].append(i)

    face_uses: dict[tuple[int, int, int], list[tuple[int, tuple[int, int, int], int]]] = defaultdict(list)
    for cell, tet in enumerate(cell_tets):
        for corner in CORNER_FACES:
            tri = tuple(tet[j] for j in corner)
            face_uses[tuple(sorted(tri))].append((cell, tri, cell_material[cell]))
    role_faces = {r: [] for r in ROLE_ORDER}
    walls: list[tuple[str, int, tuple[int, int, int], int]] = []
    classified_boundary = 0
    i_interface = 0
    for key, uses in face_uses.items():
        role_pattern = tuple(role_by_node[n][0] for n in key)
        if len(uses) == 2:
            if orient_sign(uses[0][1]) + orient_sign(uses[1][1]) != 0:
                raise ValueError(f"paired cell face has equal orientation: nodes={key} uses={uses}")
            if uses[0][2] != uses[1][2]:
                skin = next(u for u in uses if u[2] == 15)
                fat = next(u for u in uses if u[2] == 16)
                if tuple(sorted(role_pattern)) != ("I", "I", "I"):
                    raise ValueError("material interface does not use exactly three captured I nodes")
                if orient_sign(skin[1]) + orient_sign(fat[1]) != 0:
                    raise ValueError("parent skin/fat I interface triangles do not have reversed orientation")
                role_faces["I"].append((skin[1], skin[0], fat[0]))
                i_interface += 1
            continue
        if len(uses) != 1:
            raise ValueError("parent source-control volume has a nonmanifold face")
        cell, tri, material = uses[0]
        classified_boundary += 1
        if material == 15:
            if set(role_pattern) == {"O"}:
                role_faces["O"].append((tri, cell, cell_labels[cell]))
            elif set(role_pattern) <= {"O", "I"} and "O" in role_pattern and "I" in role_pattern:
                walls.append(("skin", cell, tri, 15))
            else:
                raise ValueError(f"skin boundary face has unsupported role pattern {role_pattern}")
        else:
            if set(role_pattern) == {"D"}:
                role_faces["D"].append((tri, cell, cell_labels[cell]))
            elif set(role_pattern) <= {"I", "D"} and "I" in role_pattern and "D" in role_pattern:
                walls.append(("fat", cell, tri, 16))
            else:
                raise ValueError(f"fat boundary face has unsupported role pattern {role_pattern}")

    for role in ROLE_ORDER:
        role_faces[role].sort(key=lambda x: tuple(sorted(x[0] if role != "I" else x[0])))
        if not role_faces[role]:raise ValueError("missing derived role boundary "+role)
        used_nodes = {n for face in role_faces[role] for n in face[0]}
        expected_nodes = {int(row["volume_node_row"]) for row in role_rows[role]}
        if used_nodes != expected_nodes:
            raise ValueError(f"parent facet extraction does not exactly cover {role} role nodes")
    if len(walls)!=180:raise ValueError("refinement changed original headwall count")

    retirement_summary=None
    if retirement_doc is not None:
        components=parent_tet_source_components(parent_tets,source_labels)
        child_o_face_records=[]
        for tri,cell,label in role_faces["O"]:
            child_o_face_records.append({
                "node_rows":tuple(tri),"child_tet_row":int(cell),
                "parent_tet_row":int(cp[cell]),
                "material_index":int(label["material_index"]),
                "geometric_donor_body":int(label["geometric_donor_body"]),
                "anatomical_region_id":int(label["anatomical_region_id"]),
                "stable_source_face_id":int(label["stable_source_face_id"]),
            })
        retirement_summary=validate_o_boundary_face_retirement(
            retirement_doc,source_o_faces,child_o_face_records,pn,source_role_by_node,
            source_labels,components)
        retirement_summary["map_sha256"]=digest(retirement_path)
        retirement_summary["map_path"]=str(retirement_path)
        derivation["o_boundary_retirement"]={
            "schema":retirement_doc["schema"],
            "map_sha256":retirement_summary["map_sha256"],
            "retired_parent_o_node_count":retirement_summary["retired_parent_o_node_count"],
            "retired_parent_o_face_count":retirement_summary["retired_parent_o_face_count"],
            "replacement_o_face_count":retirement_summary["replacement_o_face_count"],
            "source_connected_material_donor_region_proof":True,
            "physical_admission":False,
        }
    if d_retirement_doc is not None:
        child_d_face_records=[]
        for tri,cell,label in role_faces["D"]:
            child_d_face_records.append({"node_rows":tuple(tri),"child_tet_row":int(cell),
                "parent_tet_row":int(cp[cell]),"material_index":int(label["material_index"]),
                "geometric_donor_body":int(label["geometric_donor_body"]),
                "anatomical_region_id":int(label["anatomical_region_id"]),
                "stable_source_face_id":int(label["stable_source_face_id"])})
        d_retirement_summary=validate_d_boundary_face_retirement(
            d_retirement_doc,source_d_faces,child_d_face_records,pn,source_role_by_node,
            source_labels,components,boolean_child_lineage,candidate_points=p,
            frozen_baseline_d_faces=frozen006_d_face_lineage)
        d_retirement_summary.update(frozen006_d_face_lineage["summary"])
        d_retirement_summary.update(d_boolean_summary or {})
        d_retirement_summary["map_sha256"]=digest(d_retirement_path)
        d_retirement_summary["map_path"]=str(d_retirement_path)
        d_retirement_summary["protected_parent_nodes"]=protected_source_summary
        derivation["d_boundary_retirement"]={
            "schema":d_retirement_doc["schema"],"map_sha256":d_retirement_summary["map_sha256"],
            "retired_parent_d_node_count":d_retirement_summary["retired_parent_d_node_count"],
            "retired_parent_d_face_count":d_retirement_summary["retired_parent_d_face_count"],
            "replacement_d_face_count":d_retirement_summary["replacement_d_face_count"],
            "actual_boolean_face_ancestry_proof":True,
            "source_connected_material_donor_region_proof":True,"physical_admission":False}

    # Select one exact same-material parent tet per node for the semantic tuple.
    # Every I row uses its skin-side anchor, while both node-map copies retain
    # the same O/I/D source identity and current target.
    source_incident=defaultdict(list)
    for pt,tet in enumerate(parent_tets):
        for old in tet:source_incident[int(old)].append(pt)
    selected_parent=[]
    for node in range(len(points)):
        role=role_by_node[node][0];wanted=16 if role=="D" else 15
        eligible=[c for c in incident[node] if role=="interior" or cell_material[c]==wanted]
        if not eligible:raise ValueError("node has no correctly labeled incident child cell")
        if pn[node]>=0:
            # Retriangulation changes incident child cells. Original-node
            # ancestry stays in the original source packet, never in a
            # coincident child-cell index. Keep the existing semantic gate.
            old=int(pn[node])
            allowed={tuple(int(cell_labels[c][k]) for k in keys[:3]) for c in eligible}
            source_candidates=[pt for pt in source_incident[old]
                               if tuple(int(source_labels[pt][k]) for k in keys[:3]) in allowed]
            if not source_candidates:
                raise ValueError(f"original node {old} source material/donor/region absent in refined incident cells")
            child_sources={int(cp[c]) for c in eligible}
            pt=min(source_candidates,key=lambda pt:(pt not in child_sources,pt))
            tet=parent_tets[pt];match=np.flatnonzero(tet==old)
            if len(match)!=1:raise ValueError("original node source ancestry is not a unique original cell corner")
            beta=tuple(1. if k==int(match[0]) else 0. for k in range(4))
        else:pt,beta=node_parent_ancestry[node]
        key=tuple(int(source_labels[pt][k]) for k in keys[:3])
        if not any(tuple(int(cell_labels[c][k]) for k in keys[:3])==key for c in eligible):
            raise ValueError("node source material/donor/region absent in its incident child cells")
        selected_parent.append((pt,beta))

    # Exact TetGen parent identity files.
    volume_dir = out / "volume"
    volume_dir.mkdir()
    with (volume_dir / "mesh.node").open("w") as f:
        f.write(f"{len(points)} 3 0 0\n")
        for i, p in enumerate(points):
            f.write(f"{i+1} " + " ".join(format(x, ".17g") for x in p) + "\n")
    with (volume_dir / "mesh.ele").open("w") as f:
        f.write(f"{len(cell_tets)} 4 1\n")
        for i, (tet, material) in enumerate(zip(cell_tets, cell_material)):
            f.write(f"{i+1} " + " ".join(str(n+1) for n in tet) + f" {material}\n")
    label_header = ["child_tet_row", "material_index", "geometric_donor_body",
                    "anatomical_region_id", "stable_source_face_id"]
    label_rows = []
    for i, label in enumerate(cell_labels):
        label_rows.append([i, label["material_index"], label["geometric_donor_body"],
                           label["anatomical_region_id"], label["stable_source_face_id"]])
    write_csv(volume_dir / "child.labels.csv", label_header, label_rows)
    write_csv(volume_dir / "cell-parent-map.csv",
              ["child_tet_row", "parent_tet_row", "match_method", "parent_node0", "parent_node1",
               "parent_node2", "parent_node3", "material_index", "geometric_donor_body",
               "anatomical_region_id", "stable_source_face_id"],
              [[i, int(cp[i]), "declared_original_parent_cell_reference_refinement", *parent_tets[cp[i]],
                cell_material[i], cell_labels[i]["geometric_donor_body"],
                cell_labels[i]["anatomical_region_id"], cell_labels[i]["stable_source_face_id"]]
               for i in range(len(cell_tets))])

    # Role assets and exact extracted registered surfaces.
    role_dir = out / "roles"
    registered_dir = out / "registered"
    role_dir.mkdir()
    registered_dir.mkdir()
    weights_by_node = {}
    for i, row in enumerate(weights):
        if int(row["vertex"]) != i:
            raise ValueError(f"parent full86 row index differs at {i}")
        vals = [float(row[f"w{k}"]) for k in range(86)]
        if any(not math.isfinite(x) or x < 0.0 for x in vals) or abs(sum(vals) - 1.0) > 1.0e-5:
            raise ValueError(f"parent full86 weights are invalid at node {i}")
        weights_by_node[i] = vals

    sidecar_face_maps: dict[str, dict[tuple[int, int, int], int]] = {}
    source_asset_records = []
    surface_face_entries = {}
    for role in ROLE_ORDER:
        local_for_node = {int(row["volume_node_row"]): i for i, row in enumerate(role_rows[role])}
        face_entries = []
        sidecar_faces = []
        parent_face_info = []
        for face_row, entry in enumerate(role_faces[role]):
            tri, cell, label_or_fat = entry
            local_tri = tuple(local_for_node[n] for n in tri)
            sidecar_faces.append((face_row, *local_tri, face_row))
            face_entries.append({"face_row": face_row, "surface_vertex_rows": list(local_tri),
                                 "child_node_rows":list(tri),"child_tet_row":cell,"parent_tet_row":int(cp[cell]),
                                 "parent_material": cell_material[cell],
                                 "parent_stable_source_face_id": cell_labels[cell]["stable_source_face_id"]})
            parent_face_info.append((face_row, tri, cell, cell_labels[cell]))
        source_asset = role_dir / f"{role}-source.json"
        source_doc = {
            "schema": "numi-human.parent-capture-identity-role-source-control.v1",
            "control_only": False,
            "derivation": derivation,
            "role": role,
            "source_domain": ROLE_DOMAIN[role],
            "parent_capture": str(capture),
            "parent_packet_sha256": input_hashes[str(packet)],
            "parent_nodes_sha256": input_hashes[str(nodes_path)],
            "parent_tets_sha256": input_hashes[str(tets_path)],
            "parent_labels_sha256": input_hashes[str(labels_path)],
            "vertices": [
                {"source_vertex_row": i, "child_volume_node_row": int(row["volume_node_row"]), "parent_volume_node_row":int(pn[int(row["volume_node_row"])]),
                 "compiled_global_node": int(row["compiled_global_node"]),
                 "current_world_m": list(points[int(row["volume_node_row"])]),
                 "source_frame_owner_body":int(source_node_rows[pn[int(row["volume_node_row"])]] ["node_frame_owner_body"]) if pn[int(row["volume_node_row"])]>=0 else None,
                 "source_node_ancestry":int(pn[int(row["volume_node_row"])]),
                 "registered_donor_local_m": list(local_points_by_node[int(row["volume_node_row"])]),
                 "registered_frame_owner_body": int(row["frame_owner_body"])}
                for i, row in enumerate(role_rows[role])
            ],
            "faces": face_entries,
            "physical_admission": False,
        }
        source_asset.write_text(json.dumps(source_doc, indent=2, sort_keys=True) + "\n")
        weights_out = role_dir / f"{role}-full86.csv"
        wrows = []
        for i, row in enumerate(role_rows[role]):
            node = int(row["volume_node_row"])
            wrows.append([i, *[format(v, ".17g") for v in weights_by_node[node]]])
        write_csv(weights_out, ["vertex", *[f"w{i}" for i in range(86)]], wrows)

        domain = ROLE_DOMAIN[role]
        nodes_out = registered_dir / f"registered-{role}.csv"
        faces_out = registered_dir / f"registered-{role}-faces.csv"
        nheader = ["vertex", "donor", "wx", "wy", "wz", "nx", "ny", "nz",
                   "lx", "ly", "lz", "m00", "m01", "m02", "m10", "m11",
                   "m12", "m20", "m21", "m22"]
        if role == "O":
            nheader += ["d37_source_vertex_row", "d37_source_face_row",
                        "beta0", "beta1", "beta2", "projection_distance_m",
                        "source_vertex_row", "source_face_row", "source_bary0",
                        "source_bary1", "source_bary2", "source_residual_x_m",
                        "source_residual_y_m", "source_residual_z_m",
                        "source_domain", "frame_owner_method"]
            face_header = ["face", "v0", "v1", "v2", "baseline806_source_face_row"]
        else:
            nheader += ["source_domain", "d37_parent_node_row",
                        "d37_parent_surface_face_row", "beta0", "beta1", "beta2",
                        "source_residual_x_m", "source_residual_y_m",
                        "source_residual_z_m", "frame_owner_method"]
            face_header = ["face", "v0", "v1", "v2", "d37_parent_surface_face_row",
                           "d37_parent_tet_row", "material_index", "source_role"]
        nrows = []
        normals = [[0.0, 0.0, 0.0] for _ in role_rows[role]]
        for face_row, tri, _, _ in parent_face_info:
            p0, p1, p2 = (points[n] for n in tri)
            fn = cross(sub(p1, p0), sub(p2, p0))
            for node in tri:
                local = local_for_node[node]
                normals[local] = [normals[local][a] + fn[a] for a in range(3)]
        for i, row in enumerate(role_rows[role]):
            node = int(row["volume_node_row"])
            body = int(row["frame_owner_body"])
            frame = frame_by_body[body]
            nn = tuple(normals[i])
            nnorm = norm(nn)
            if nnorm <= 1.0e-20:
                raise ValueError(f"role {role} has an unused/degenerate node normal at row {i}")
            normal = tuple(x / nnorm for x in nn)
            p = points[node]
            local = local_points_by_node[node]
            inv = frame["current_invR"]
            prefix = [i, body, *p, *normal, *local, *inv]
            if role == "O":
                nrows.append([*prefix, i, -1, 1, 0, 0, 0, i, -1, 0, 0, 0,
                              0, 0, 0, domain, FRAME_METHOD])
            else:
                nrows.append([*prefix, domain, i, -1, 0, 0, 0, 0, 0, 0, FRAME_METHOD])
        write_csv(nodes_out, nheader, nrows)

        frows = []
        face_source_map = {}
        for face_row, tri, cell, label in parent_face_info:
            local = [local_for_node[n] for n in tri]
            if role == "O":
                frows.append([face_row, *local, face_row])
            else:
                frows.append([face_row, *local, face_row, int(cp[cell]),
                              cell_material[cell], f"parent-capture-A-{role}-exact-facet"])
            face_source_map[tuple(sorted(tri))] = face_row
        write_csv(faces_out, face_header, frows)
        sidecar_face_maps[role] = face_source_map
        surface_face_entries[role] = parent_face_info
        source_asset_records.append({
            "role": role, "source_domain": domain, "source_asset": str(source_asset),
            "source_asset_sha256": digest(source_asset),
            "source_vertex_count": len(role_rows[role]),
            "source_face_count": len(parent_face_info),
            "full86_csv": str(weights_out), "full86_sha256": digest(weights_out),
            "full86_row_count": len(wrows),
            "registered_nodes_csv": str(nodes_out), "registered_nodes_sha256": digest(nodes_out),
            "registered_faces_csv": str(faces_out), "registered_faces_sha256": digest(faces_out),
        })

    # Exact donor-frame table stays in its original native producer encoding.
    frames_out = registered_dir / "donor-frames.csv"
    write_frame_table(frames_out, frame_by_body)

    # Surface-to-parent registration uses exact captured parent nodes, with
    # one-hot parent barycentrics and zero current residuals.
    reg_header = ["layer", "surface_vertex_row", "parent_registration_kind", "parent_tet_row",
                  "parent_bary0", "parent_bary1", "parent_bary2", "parent_bary3",
                  "parent_residual_x_m", "parent_residual_y_m", "parent_residual_z_m",
                  "parent_registration_distance_m", "parent_registration_tie_count",
                  "frame_owner_method", "source_domain", "source_vertex_row",
                  "source_face_row", "source_bary0", "source_bary1", "source_bary2",
                  "source_residual_x_m", "source_residual_y_m", "source_residual_z_m"]
    registration_rows = []
    parent_residual=[sub(points[n],tuple(sum(selected_parent[n][1][k]*parent_points[parent_tets[selected_parent[n][0]][k]][axis] for k in range(4)) for axis in range(3))) for n in range(len(points))]
    if d_retirement_doc is not None:
        with np.load(d_retirement_baseline_path,allow_pickle=False) as baseline_z:
            baseline_parent_rows=np.asarray(baseline_z["parent_volume_node_row"],np.int64)
            baseline_points_f32=np.asarray(baseline_z["points_current_world_m_f32"])
        baseline_ancestry_rows=read_csv(d_retirement_baseline_path.parent/"inserted-node-ancestry.csv")
        preexisting_derived_nodes=unchanged_preexisting_derived_node_rows(
            baseline_parent_rows,baseline_points_f32,baseline_ancestry_rows,pn,points,node_parent_ancestry)
        baseline_derived_d_rows={int(node) for node in np.flatnonzero(baseline_parent_rows<0)
                                 if int(node) in role_by_node and role_by_node[int(node)][0]=="D"}
        unchanged_derived_d_rows=baseline_derived_d_rows & preexisting_derived_nodes
        d_source_residual_summary=validate_d_new_node_source_residuals(
            points,parent_points,parent_tets,pn,selected_parent,role_by_node,
            maximum_m=d_parent_registration_limit_m,
            preexisting_derived_node_rows=preexisting_derived_nodes)
        d_source_residual_summary["preexisting_current006_d_nodes_unchanged_and_exempt"]=len(unchanged_derived_d_rows)
        d_source_residual_summary["preexisting_current006_d_nodes_changed_and_checked_as_new"]=len(baseline_derived_d_rows-unchanged_derived_d_rows)
        d_retirement_summary.update(d_source_residual_summary)
        derivation["d_boundary_retirement"].update(d_source_residual_summary)
    for role in ROLE_ORDER:
        for local, row in enumerate(role_rows[role]):
            node = int(row["volume_node_row"])
            cell, beta = selected_parent[node]
            registration_rows.append([role, local, IDENTITY_KIND, cell, *beta,
                                      *parent_residual[node], norm(parent_residual[node]), 0, FRAME_METHOD,
                                      ROLE_DOMAIN[role], local, -1, 0, 0, 0, 0, 0, 0])
    node_registration = registered_dir / "node-parent-registration.csv"
    write_csv(node_registration, reg_header, registration_rows)

    # O-face ancestry preserves exact parent cell and source-face identity.
    outer_rows = []
    for face_row, tri, cell, label in surface_face_entries["O"]:
        outer_rows.append([face_row, -1, label["stable_source_face_id"], face_row,
                           "parent_capture_A_exact_parent_cell_facet"])
    outer_ancestry = registered_dir / "outer-face-ancestry.csv"
    write_csv(outer_ancestry,
              ["output_face_row", "parent_face_row", "source_original_id",
               "source_face_row", "ancestry_role"], outer_rows)

    # Semantic child labels remain exact parent rows. The combined node map
    # preserves frame-owner, geometric donor, region and source namespace as
    # distinct typed fields. The candidate operational map is default-off; its
    # extra columns keep the immutable source-parent registration distinct from
    # an exact self-tet registration in the emitted candidate reference.
    candidate_operational_reference = bool(getattr(args, "candidate_operational_reference", False))
    map_header = ["joint_node_id", "source_partition", "source_tetgen_node_id",
                  "registered_I_source_vertex_row", "merge_peer_partition",
                  "merge_peer_tetgen_node_id", "registered_surface_layer",
                  "registered_surface_vertex_row", "role", "current_x_m", "current_y_m",
                  "current_z_m", "donor_body", "donor_binding", "donor_local_x_m",
                  "donor_local_y_m", "donor_local_z_m", "source_domain",
                  "source_mapping_kind", "source_vertex_row", "source_face_row",
                  "source_bary0", "source_bary1", "source_bary2", "source_residual_x_m",
                  "source_residual_y_m", "source_residual_z_m", "frame_owner_method",
                  "parent_registration_kind", "parent_tet_row", "parent_bary0",
                  "parent_bary1", "parent_bary2", "parent_bary3",
                  "parent_current_residual_x_m", "parent_current_residual_y_m",
                  "parent_current_residual_z_m", "parent_registration_distance_m",
                  "parent_registration_tie_count", "incident_material_mask",
                  "geometric_donor_body", "anatomical_region_id"]
    if candidate_operational_reference:
        map_header.extend(OPERATIONAL_REFERENCE_MAP_FIELDS)
    map_rows = []
    for node in range(len(points)):
        role, local = role_by_node[node]
        cell, beta = selected_parent[node]
        layer_row = role_rows[role][local] if role!="interior" else {"frame_owner_body":owner_by_node[node]}
        label = source_labels[cell]
        operational_fields = []
        if candidate_operational_reference:
            operational_fields = candidate_operational_reference_fields(
                node, int(pn[node]), int(cell), source_labels, cell_tets, cell_labels, incident)
            validate_candidate_operational_reference_fields(
                operational_fields, node, int(pn[node]), int(cell), source_labels,
                cell_tets, cell_labels, incident)
        mask = 0
        for incident_cell in incident[node]:
            mask |= 1 << cell_material[incident_cell]
        body = int(layer_row["frame_owner_body"])
        # TetGen node IDs are partition-local. Skin combines its disjoint O
        # and I role rows; fat combines its I and D role rows. Keep those IDs
        # unique within each partition while preserving reciprocal I peers.
        p = points[node]
        world_delta = sub(p, frame_by_body[body]["origin"])
        local_point = matrix_vector3(frame_by_body[body]["current_invR"],world_delta)
        common = [node + 1, None, None, local if role == "I" else -1,
                  None, None, role, local, role, *p, body, binding_by_body[body],
                  *local_point, ROLE_DOMAIN.get(role,"derived_volume_interior_no_surface_ancestry"), "vertex" if role!="interior" else "none", local, -1, 0, 0, 0,
                  0, 0, 0, FRAME_METHOD if role!="interior" else "derived_full_all_bodies_geometric_donor_pose", IDENTITY_KIND, cell, *beta, *parent_residual[node], norm(parent_residual[node]), 0,
                  mask, int(label["geometric_donor_body"]), int(label["anatomical_region_id"])]
        if candidate_operational_reference:
            common.extend(operational_fields)
        if role == "O":
            common[1] = "skin"; common[2] = local + 1
            map_rows.append(common)
        elif role == "D":
            fat_source_id = len(role_rows["I"]) + local + 1
            common[1] = "fat"; common[2] = fat_source_id
            map_rows.append(common)
        elif role=="interior":
            partition="skin" if cell_material[incident[node][0]]==15 else "fat"
            if len(set(cell_material[c] for c in incident[node]))!=1:raise ValueError("interior node is an unregistered material interface")
            common[1]=partition;common[2]=len(points)+node+1;common[6]="";common[7]=-1
            map_rows.append(common)
        else:
            skin_source_id = len(role_rows["O"]) + local + 1
            fat_source_id = local + 1
            skin = list(common)
            skin[1] = "skin"; skin[2] = skin_source_id; skin[4] = "fat"; skin[5] = fat_source_id
            fat = list(common)
            fat[1] = "fat"; fat[2] = fat_source_id; fat[4] = "skin"; fat[5] = skin_source_id
            map_rows.extend([skin, fat])
    node_map = volume_dir / "combined-node-map.csv"
    write_csv(node_map, map_header, map_rows)

    # All registered faces come from extracted parent tet facets. Mixed
    # O/I and I/D boundaries are explicit retained parent head-wall faces.
    facet_header = ["source_partition", "shell_face_row", "face_marker", "face_source_role",
                    "face_source_row", "face_exact_source_row", "face_source_original_id",
                    "face_source_layer", "face_source_body", "face_source_bone",
                    "face_ancestry_kind", "face_run_original_id", "face_source_face_id",
                    "face_ancestry_source_rows", "face_parent_tet_row",
                    "face_parent_material_index", "parent_source_node_rows",
                    "joint_element_id", "child_tet_row", "joint_node0", "joint_node1",
                    "joint_node2", "face_ancestry_source_original_ids",
                    "face_ancestry_parent_face_rows"]
    facet_rows = []
    shell_face_row = defaultdict(int)
    registered_counts = 0
    for role, partition, marker, face_role in [
        ("O", "skin", 1, "outer_O"),
        ("I", "skin", 2, "inner_I"),
        ("I", "fat", 4, "outer_I"),
        ("D", "fat", 5, "inner_D"),
    ]:
        for face_row, tri, cell, label_or_fat in surface_face_entries[role]:
            if role == "I":
                info = face_uses[tuple(sorted(tri))]
                cell_use = next(u for u in info if cell_material[u[0]] == (15 if partition == "skin" else 16))
                cell = cell_use[0]
                tri = cell_use[1]
            label = cell_labels[cell]
            fields = ["-1"] * 24
            fields[0] = partition
            fields[1] = shell_face_row[partition]
            shell_face_row[partition] += 1
            fields[2] = marker
            fields[3] = face_role
            fields[4] = face_row
            fields[5] = face_row
            fields[6] = label["stable_source_face_id"]
            fields[7] = role
            fields[10] = -1
            fields[12] = label["stable_source_face_id"]
            fields[13] = json.dumps([face_row])
            fields[14] = int(cp[cell])
            fields[15] = cell_material[cell]
            fields[16] = json.dumps([int(pn[n]) for n in tri], separators=(",", ":"))
            fields[17] = cell + 1
            fields[18] = cell
            fields[19:22] = [n + 1 for n in tri]
            fields[22] = json.dumps([label["stable_source_face_id"]])
            fields[23] = json.dumps([face_row])
            facet_rows.append(fields)
            registered_counts += 1

    wall_ids = []
    for index, (partition, cell, tri, material) in enumerate(walls):
        label = cell_labels[cell]
        role = "donor_skin_OI_headwall" if partition == "skin" else "donor_fat_ID_headwall"
        stable_wall = 1_000_000 + index
        wall_ids.append(stable_wall)
        fields = ["-1"] * 24
        fields[0] = partition
        fields[1] = shell_face_row[partition]
        shell_face_row[partition] += 1
        fields[2] = 3 if material == 15 else 6
        fields[3] = role
        fields[4] = stable_wall
        fields[10] = 5
        fields[13] = json.dumps([label["stable_source_face_id"]])
        fields[14] = int(cp[cell])
        fields[15] = material
        fields[16] = json.dumps([int(pn[n]) for n in tri], separators=(",", ":"))
        fields[17] = cell + 1
        fields[18] = cell
        fields[19:22] = [n + 1 for n in tri]
        fields[22] = json.dumps([label["stable_source_face_id"]])
        fields[23] = json.dumps([cell])
        facet_rows.append(fields)
    facet_csv = volume_dir / "facet-ancestry.csv"
    write_csv(facet_csv, facet_header, facet_rows)

    role_registry = {
        "schema": "numi-human.registered-volume-role-source-registry.v1",
        "control_only": False,
            "derivation": derivation,
        "roles": source_asset_records,
        "typed_role_domain_indices": {"O": 0, "I": 1, "D": 2},
        "physical_admission": False,
    }
    if candidate_operational_reference:
        role_registry["registration_domain"] = "candidateOperationalReference"
        role_registry["candidate_operational_reference_map"] = {
            "schema": "numi-human.candidate-operational-reference-node-map.v1",
            "node_map_sha256": digest(node_map),
            "candidate_npz_sha256": input_hashes[str(candidate_path)],
            "immutable_source_parent_packet_sha256": input_hashes[str(packet)],
            "coordinate_frame_sha256": input_hashes[str(pose_path)],
        }
    registry_path = out / "source-domain-registry.json"
    registry_path.write_text(json.dumps(role_registry, indent=2, sort_keys=True) + "\n")

    # Inherit nodal thickness; new face spacing must not define a new field.
    thickness_outputs=[]
    thickness_records={}
    for role in ROLE_ORDER:
        path=role_dir / f"{role}-contact-thickness.csv"
        values=[]
        for row,rec in enumerate(role_rows[role]):
            node=int(rec["volume_node_row"]);old=int(pn[node])
            if old>=0:
                pair=[source_layer_rows[old][key] for key in ("skin_thickness_m","fat_thickness_m")]
            else:
                pt,beta=selected_parent[node]
                pair=[format(sum(beta[k]*float(source_layer_rows[parent_tets[pt][k]][key])
                                 for k in range(4)),".17g")
                      for key in ("skin_thickness_m","fat_thickness_m")]
            if any(not math.isfinite(float(v)) or float(v)<=0 for v in pair):
                raise ValueError("derived contact thickness is invalid")
            values.append([row,*pair])
        write_csv(path,["role_row","skin_thickness_m","fat_thickness_m"],values)
        thickness_outputs.append(path)
        thickness_records[role]={"path":str(path),"sha256":digest(path),"rows":len(values),
            "method":"exact original nodal values; new nodes interpolate explicit original-parent barycentrics"}

    all_outputs = [volume_dir / "mesh.node", volume_dir / "mesh.ele",
                   volume_dir / "child.labels.csv", volume_dir / "cell-parent-map.csv",
                   node_map, facet_csv, node_registration, outer_ancestry, frames_out,
                   registry_path]
    if retirement_doc is not None:
        retirement_out=out/"o-boundary-retirement.json"
        retirement_out.write_text(json.dumps(retirement_doc,indent=2,sort_keys=True)+"\n")
        all_outputs.append(retirement_out)
    if d_retirement_doc is not None:
        d_retirement_out=out/"d-boundary-retirement.json"
        d_retirement_out.write_text(json.dumps(d_retirement_doc,indent=2,sort_keys=True)+"\n")
        all_outputs.append(d_retirement_out)
    all_outputs += [pathlib.Path(r[k]) for r in source_asset_records
                    for k in ("source_asset", "full86_csv", "registered_nodes_csv", "registered_faces_csv")]
    all_outputs += thickness_outputs
    output_hashes = {str(p): digest(p) for p in all_outputs}
    receipt = {
        "schema": "numi-human.registered-volume-parent-identity-source-control.v1",
        "method":"explicit connected reference refinement with source node/cell ancestry and exact child O/I/D facets",
        "parent_capture": str(capture),
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "source_parent_nodes": len(parent_points),"child_nodes":len(points),
        "source_parent_tetrahedra": len(source_tets),"child_tetrahedra":len(cell_tets),
        "contact_thickness":thickness_records,
        "role_nodes": {r: len(role_rows[r]) for r in ROLE_ORDER},
        "role_faces": {r: len(role_faces[r]) for r in ROLE_ORDER},
        "registered_facet_rows": registered_counts,
        "authored_parent_headwall_faces": len(walls),
        "node_map_rows": len(map_rows),
        "node_map_shared_I_pairs": len(role_rows["I"]),
        "parent_maximum_registration_distance_m": max(map(norm,parent_residual)),
        "derivation": derivation,
        "mass_mapping_maximum_transport_distance_m": args.maximum_mass_transport_distance_m,
        "coordinate_frame_sha256": input_hashes[str(pose_path)],
        "source_body_count": len(poses["all_bodies"]),
        "native_frame_source": "registered-skin-poses-100.json current/rest local_to_world_row_major_f32",
        "role_source_registry": str(registry_path),
        "physical_admission": False,
        "native_steps": 0,
        "claim": "derived coarse candidate sidecars only; no geometry admission or physical motion qualification",
        "rejected_alternative_preserved": str(capture.parent.parent / "coarse-transfer-mass-assets-299" / "mass-preflight.log"),
    }
    if retirement_summary is not None:
        receipt["o_boundary_retirement"]=retirement_summary
    if d_retirement_summary is not None:
        receipt["d_boundary_retirement"]=d_retirement_summary
    if candidate_operational_reference:
        receipt["registration_domain"] = "candidateOperationalReference"
        receipt["candidate_operational_reference_map"] = {
            "schema": "numi-human.candidate-operational-reference-node-map.v1",
            "node_map_sha256": output_hashes[str(node_map)],
            "candidate_npz_sha256": input_hashes[str(candidate_path)],
            "candidate_node_sha256": output_hashes[str(volume_dir / "mesh.node")],
            "candidate_element_sha256": output_hashes[str(volume_dir / "mesh.ele")],
            "candidate_labels_sha256": output_hashes[str(volume_dir / "child.labels.csv")],
            "coordinate_frame_sha256": input_hashes[str(pose_path)],
            "immutable_source_parent_packet_sha256": input_hashes[str(packet)],
            "source_parent_node_identity_rows": len(parent_points),
            "operational_registration": "exact candidate self-tet one-hot with zero residual and distance",
            "source_parent_ancestry": "existing original 330 source tet/bary/residual plus one-to-one retained source-node row and donor/region tuple",
            "physical_admission": False,
        }
    receipt_path = out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    add_parent_registration_distance_argument(parser)
    parser.add_argument("--candidate-npz",type=pathlib.Path,required=True)
    parser.add_argument("--candidate-node",type=pathlib.Path,required=True)
    parser.add_argument("--candidate-node-map",type=pathlib.Path,required=True)
    parser.add_argument("--inserted-node-ancestry",type=pathlib.Path)
    parser.add_argument("--o-boundary-retirement-map",type=pathlib.Path,
                        help="Pinned explicit O parent-node retirement and source-face replacement map")
    parser.add_argument("--d-boundary-retirement-map",type=pathlib.Path,
                        help="Pinned explicit D parent-node retirement, replacement facets and Boolean ancestry map")
    parser.add_argument("--d-boolean-output-npz",type=pathlib.Path)
    parser.add_argument("--d-boolean-source-shell-npz",type=pathlib.Path)
    parser.add_argument("--d-boolean-bank-input-npz",type=pathlib.Path)
    parser.add_argument("--d-boolean-face-lineage",type=pathlib.Path)
    parser.add_argument("--d-retirement-baseline-npz",type=pathlib.Path)
    parser.add_argument("--capture", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--maximum-mass-transport-distance-m", type=float, default=0.02)
    parser.add_argument("--candidate-operational-reference", action="store_true",
                        help="emit the opt-in 54-column map with exact candidate self-tet registration; source-mass gates remain active")
    args = parser.parse_args()
    if not math.isfinite(args.maximum_mass_transport_distance_m) or args.maximum_mass_transport_distance_m <= 0:
        parser.error("mass transport distance must be positive and finite")
    if args.d_boundary_retirement_map is not None and any(x is None for x in (
            args.d_boolean_output_npz,args.d_boolean_source_shell_npz,
            args.d_boolean_bank_input_npz,args.d_boolean_face_lineage,args.d_retirement_baseline_npz)):
        parser.error("D retirement requires its frozen006 baseline, pinned Boolean output, shell, bank input and face-lineage sidecar")
    if args.d_boundary_retirement_map is None and any(x is not None for x in (
            args.d_boolean_output_npz,args.d_boolean_source_shell_npz,
            args.d_boolean_bank_input_npz,args.d_boolean_face_lineage,args.d_retirement_baseline_npz)):
        parser.error("D Boolean inputs require --d-boundary-retirement-map")
    try:
        receipt = export(args)
    except Exception as exc:
        print(f"derived_registered_volume_sidecars=FAIL reason={exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    print("derived_registered_volume_sidecars=PASS physical_admission=0 native_steps=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
