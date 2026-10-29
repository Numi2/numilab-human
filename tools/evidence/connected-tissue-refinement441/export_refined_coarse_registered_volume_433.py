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

def export(args: argparse.Namespace) -> dict[str, Any]:
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
    candidate=np.load(candidate_path,allow_pickle=False)
    p=np.asarray(candidate["points_reference_world_m_f64"],np.float32).astype(np.float64)
    tt=np.asarray(candidate["volume_tetrahedra"],np.int64)
    cp=np.asarray(candidate["child_parent_tet_row"],np.int64)
    pn=np.asarray(candidate["parent_volume_node_row"],np.int64)
    ww=np.asarray(candidate["full86_weights"],float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<len(parent_points) or not np.isfinite(p).all():
        raise ValueError("invalid refined coordinate domain")
    if tt.ndim!=2 or tt.shape[1]!=4 or tt.min()<0 or tt.max()>=len(p) or cp.shape!=(len(tt),) or cp.min()<0 or cp.max()>=len(source_tets):
        raise ValueError("invalid refined child/source-tet ancestry")
    if pn.shape!=(len(p),) or np.any(pn < -1) or np.any(pn>=len(parent_points)) or set(pn[pn>=0])!=set(range(len(parent_points))) or len(np.unique(pn[pn>=0]))!=len(pn[pn>=0]):
        raise ValueError("refinement must preserve every original node exactly once")
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
            if old>=0 and source_role_by_node[old]!=(role,row):raise ValueError("original boundary source role/row changed")
            body=int(rec["frame_owner_body"])
            if body not in frame_by_body:raise ValueError("boundary owner not in original86 binding frames")
            local=matrix_vector3(frame_by_body[body]["current_invR"],sub(points[node],frame_by_body[body]["origin"]))
            local_points_by_node[node]=local;owner_by_node[node]=body;role_by_node[node]=(role,row)
            enriched.append(dict(rec,compiled_global_node=int(source_node_rows[old]["compiled_global_node"]) if old>=0 else -1,
                                 donor_local_x_m=local[0],donor_local_y_m=local[1],donor_local_z_m=local[2]))
        role_rows[role]=enriched
    for node in range(len(points)):
        if node in role_by_node:continue
        if pn[node]>=0:raise ValueError("an original boundary node lost its role")
        pt,_beta=node_parent_ancestry[node];body=int(source_labels[pt]["geometric_donor_body"])
        if body not in frame_by_body:raise ValueError("interior geometric-donor frame is not in captured86 frames")
        owner_by_node[node]=body;role_by_node[node]=( "interior",-1)
        local_points_by_node[node]=matrix_vector3(frame_by_body[body]["current_invR"],sub(points[node],frame_by_body[body]["origin"]))
    input_hashes.update({str(candidate_path):digest(candidate_path),str(node_path):digest(node_path),str(node_map_path):digest(node_map_path)})
    if anc_path:input_hashes[str(anc_path)]=digest(anc_path)
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

    # Select one exact same-material parent tet per node for the semantic tuple.
    # Every I row uses its skin-side anchor, while both node-map copies retain
    # the same O/I/D source identity and current target.
    selected_parent=[]
    for node in range(len(points)):
        role=role_by_node[node][0];wanted=16 if role=="D" else 15
        eligible=[c for c in incident[node] if role=="interior" or cell_material[c]==wanted]
        if not eligible:raise ValueError("node has no correctly labeled incident child cell")
        if pn[node]>=0:
            cell=eligible[0];pt=int(cp[cell]);old=int(pn[node]);tet=parent_tets[pt]
            match=np.flatnonzero(tet==old)
            if len(match)!=1:raise ValueError("original boundary node is not in its declared source cell")
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
    # distinct typed fields.
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
    map_rows = []
    for node in range(len(points)):
        role, local = role_by_node[node]
        cell, beta = selected_parent[node]
        layer_row = role_rows[role][local] if role!="interior" else {"frame_owner_body":owner_by_node[node]}
        label = source_labels[cell]
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
    receipt_path = out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-npz",type=pathlib.Path,required=True)
    parser.add_argument("--candidate-node",type=pathlib.Path,required=True)
    parser.add_argument("--candidate-node-map",type=pathlib.Path,required=True)
    parser.add_argument("--inserted-node-ancestry",type=pathlib.Path)
    parser.add_argument("--capture", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--maximum-mass-transport-distance-m", type=float, default=0.02)
    args = parser.parse_args()
    if not math.isfinite(args.maximum_mass_transport_distance_m) or args.maximum_mass_transport_distance_m <= 0:
        parser.error("mass transport distance must be positive and finite")
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
