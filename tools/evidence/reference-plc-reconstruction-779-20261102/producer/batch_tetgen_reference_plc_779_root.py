#!/usr/bin/env python3
"""Bounded local TetGen PLC quality repairs with exact F32 cavity boundaries and full ancestry."""
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
from fractions import Fraction
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
QUALITY_OWNER_PATH = HERE / "batch_interior_steiner_kernel_cone_quality_750_root.py"
TETGEN_PATH = Path(
    "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/"
    "physical-transfer-209/offline-tetgen-318/build/tetgen"
)
BASELINE_756_PATH = Path(
    "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/"
    "physical-transfer-209/fine-physical-volume-706/quality-partial-cavity-684-root/"
    "batch_tetgen_plc_quality_756_root.py"
)
EXPECTED_BASELINE_756_SHA256 = "993035088016c8b920e9be0c88bde3afa4406f2354c6affc745028430f064def"
TOPOLOGY_OWNER_PATH = HERE.parent / "remesh_complete_surface_cavities_218_root.py"
EXPECTED_TOPOLOGY_OWNER_SHA256 = "81e1128ca6f92693039d9ce53bea62e90b1d99f72bd9b9d997826b7ee2877b9a"
EXPECTED_EXACT_MESH_OWNER_SHA256 = "650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0"
EXPECTED_QUALITY_OWNER_SHA256 = "509260bb075771481aa307541b186c37389fc2d5ecbccad0b16c5a2d6bc36cc8"
EXPECTED_TETGEN_SHA256 = "a8d2a4979fe577206588db0997147fb60ebadad533a0f3d07b5b39a21741196a"
QUALITY_GATE = 1.0e-5
VOLUME_GATE = 1.0e-18
ATTEMPT_FLAGS = ("-pYACJf", "-pq1.2YACJf", "-pq1.5YACJf")
MAX_OLD_CELLS_HARD = 256

def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()

def pin(path):
    return {"path": str(path), "sha256": sha(path)}


def reference_dependency_pins():
    return {
        "source_756": pin(BASELINE_756_PATH),
        "closed_shell_topology_218": pin(TOPOLOGY_OWNER_PATH),
        "quality_readback_750": pin(QUALITY_OWNER_PATH),
        "partial_solver_684": pin(HERE / "solve_partial_oriented_cavity_milp_684.py"),
        "exact_mesh_volume_198": pin(MESH_EXACT_OWNER_PATH),
        "exact_surface_intersection_349": pin(EXACT_SURFACE_OWNER_PATH),
        "tetgen_1_6_1": pin(TETGEN_PATH),
    }

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

if sha(QUALITY_OWNER_PATH) != EXPECTED_QUALITY_OWNER_SHA256:
    raise RuntimeError("pinned 750 quality/readback helper SHA mismatch")
if sha(TETGEN_PATH) != EXPECTED_TETGEN_SHA256:
    raise RuntimeError("pinned TetGen binary SHA mismatch")
if sha(BASELINE_756_PATH) != EXPECTED_BASELINE_756_SHA256:
    raise RuntimeError("pinned 756 provenance/readback source SHA mismatch")
if sha(TOPOLOGY_OWNER_PATH) != EXPECTED_TOPOLOGY_OWNER_SHA256:
    raise RuntimeError("pinned 218 closed-shell topology owner SHA mismatch")
S = load_module("quality_owner_750_for_tetgen756", QUALITY_OWNER_PATH)
M, T = S.M, S.T
if Path(inspect.getsourcefile(T.closed)).resolve() != TOPOLOGY_OWNER_PATH.resolve():
    raise RuntimeError("reference PLC shell checks are not executing the pinned 218 owner")
MESH_EXACT_OWNER_PATH = Path(inspect.getsourcefile(M.exact_det)).resolve()
if sha(MESH_EXACT_OWNER_PATH) != EXPECTED_EXACT_MESH_OWNER_SHA256:
    raise RuntimeError("pinned exact lattice volume/chain owner SHA mismatch")

EXACT_SURFACE_OWNER_PATH = Path(
    "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/"
    "physical-transfer-209/deep-boundary-checkpoint-349/payload/exact-owner/"
    "numilab_human/cardiac_cavity_intersections.py"
)
EXPECTED_EXACT_SURFACE_OWNER_SHA256 = "f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"
if sha(EXACT_SURFACE_OWNER_PATH) != EXPECTED_EXACT_SURFACE_OWNER_SHA256:
    raise RuntimeError("pinned exact PLC intersection/point-location owner SHA mismatch")
sys.path.insert(0, str(EXACT_SURFACE_OWNER_PATH.parent.parent))
EXACT_SURFACE = __import__(
    "numilab_human.cardiac_cavity_intersections", fromlist=["_audit_pair"]
)


def _reference_shell_proof(points, boundary):
    """Validate an oriented fixed PLC independently of old interior tet signs."""
    p = np.asarray(points, np.float32)
    faces = np.asarray(boundary, np.int64).reshape((-1, 3))
    if not np.isfinite(p).all() or not len(faces):
        raise ValueError("reference PLC needs finite points and nonempty triangular boundary")
    if np.any((faces < 0) | (faces >= len(p))):
        raise ValueError("reference PLC boundary node row outside point domain")
    closed, edge_bad = T.closed(faces)
    topo = T.manifold_topology(faces)
    if not closed or topo["bad_vertex_links"] or topo["boundary_components"] != 1:
        raise ValueError("reference PLC boundary is not one closed oriented manifold shell")
    volume = int(T.exact_volume(p, faces))
    if volume <= 0:
        raise ValueError("reference PLC exact signed shell volume is not positive")
    global_rows = np.unique(faces.reshape(-1))
    fixed = p[global_rows]
    if len({_point_bits(x) for x in fixed}) != len(fixed):
        raise ValueError("reference PLC has duplicate exact-F32 boundary coordinates")
    local_index = np.full(len(p), -1, np.int64)
    local_index[global_rows] = np.arange(len(global_rows), dtype=np.int64)
    local_faces = local_index[faces]
    records, _ = EXACT_SURFACE.float32_surface_records(fixed, local_faces)
    self_pairs = EXACT_SURFACE._audit_pair(records, records, same_surface=True)
    if int(self_pairs["count"]) != 0:
        raise ValueError("reference PLC boundary self-intersects")
    prepared = EXACT_SURFACE.prepare_float32_point_location_meters(fixed, local_faces)
    return {
        "boundary_faces": faces,
        "boundary_global_point_rows": global_rows,
        "boundary_local_faces": local_faces,
        "boundary_points": fixed,
        "point_location_prepared": prepared,
        "exact_signed_volume_integer": volume,
        "edge_bad_count": int(edge_bad),
        "topology": topo,
        "self_intersection_count": 0,
    }


def _reference_region_seed(points, tets, rows, shell_proof):
    """Choose an old-cell centroid only after exact location in the PLC shell."""
    p = np.asarray(points, np.float32)
    cells = np.asarray(tets, np.int64)
    for cell_row in sorted(map(int, rows)):
        tet = cells[cell_row]
        q = tuple(
            sum(Fraction(float(v)) for v in p[tet, axis]) / 4
            for axis in range(3)
        )
        location = EXACT_SURFACE.point_location_prepared_indexed(
            q, shell_proof["point_location_prepared"]
        )
        if location["location"] == "inside":
            return np.asarray([float(v) for v in q], np.float64), int(cell_row)
    raise ValueError("no old-cell centroid is an exact interior seed for reference PLC")


def _reference_output_certificate(points, old_cells, new_cells, boundary,
                                  expected_volume_integer=None):
    """Boundary-chain/volume proof for -pY reconstruction from invalid interiors."""
    p = np.asarray(points, np.float32)
    old_cells = np.asarray(old_cells, np.int64)
    new_cells = np.asarray(new_cells, np.int64)
    faces = np.asarray(boundary, np.int64)
    proof = _reference_shell_proof(p, faces)
    if expected_volume_integer is not None and int(proof["exact_signed_volume_integer"]) != int(expected_volume_integer):
        return False, {"reason": "shell_volume_changed"}
    if not np.array_equal(M.chain(old_cells), M.chain(new_cells)):
        return False, {"reason": "oriented_cavity_chain_changed"}
    output_boundary = _oriented_boundary(new_cells)
    if not np.array_equal(T.m.boundary_identity(output_boundary), T.m.boundary_identity(faces)):
        return False, {"reason": "oriented_boundary_identity_changed"}
    ip = M.integer_points(p)
    exact_dets = [int(M.exact_det(ip, tet)) for tet in new_cells]
    if not exact_dets or min(exact_dets) <= 0:
        return False, {"reason": "new_exact_nonpositive", "minimum_det_integer": min(exact_dets, default=0)}
    if sum(exact_dets) != int(proof["exact_signed_volume_integer"]):
        return False, {
            "reason": "new_exact_volume_differs_from_boundary",
            "new_det_sum_integer": str(sum(exact_dets)),
            "boundary_volume_integer": str(proof["exact_signed_volume_integer"]),
        }
    return True, {
        "reason": "PASS_REFERENCE_PLC_BOUNDARY_VOLUME",
        "exact_signed_volume_integer": str(proof["exact_signed_volume_integer"]),
        "new_exact_det_sum_integer": str(sum(exact_dets)),
        "all_new_exact_determinants_positive": True,
        "oriented_boundary_identity_exact": True,
        "self_intersection_count": 0,
    }


def _oriented_coordinate_face_map(points, faces):
    """Return coordinate-keyed oriented triangles; reject ambiguous duplicates."""
    p = np.asarray(points, np.float32)
    result = {}
    for face in np.asarray(faces, np.int64).reshape((-1, 3)):
        seq = tuple(_point_bits(p[int(v)]) for v in face)
        if len(set(seq)) != 3:
            raise ValueError("shared interface face has duplicate F32 corner coordinates")
        ordered = tuple(sorted(seq))
        permutation = [ordered.index(v) for v in seq]
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(3) for j in range(i + 1, 3)
        )
        sign = -1 if inversions & 1 else 1
        if ordered in result:
            raise ValueError("role boundary repeats an exact coordinate triangle")
        result[ordered] = sign
    return result


def _reference_shared_interface_audit(skin_points, skin_faces, fat_points, fat_faces,
                                      expected_skin_points=None, expected_skin_faces=None,
                                      expected_fat_points=None, expected_fat_faces=None):
    """Bind the unchanged shared Skin/Fat interface by exact F32 coordinates."""
    skin = _oriented_coordinate_face_map(skin_points, skin_faces)
    fat = _oriented_coordinate_face_map(fat_points, fat_faces)
    shared = set(skin) & set(fat)
    if not shared:
        raise ValueError("reference PLC pair has no exact shared Skin/Fat interface faces")
    if any(skin[k] != -fat[k] for k in shared):
        raise ValueError("reference PLC input Skin/Fat shared face orientations are not opposite")
    if expected_skin_points is not None:
        want_skin = _oriented_coordinate_face_map(expected_skin_points, expected_skin_faces)
        want_fat = _oriented_coordinate_face_map(expected_fat_points, expected_fat_faces)
        if set(want_skin) & set(want_fat) != shared:
            raise ValueError("reference PLC output changed the exact shared interface face set")
        if any(want_skin[k] != skin[k] for k in shared):
            raise ValueError("reference PLC output changed Skin interface orientation/coordinates")
        if any(want_fat[k] != fat[k] for k in shared):
            raise ValueError("reference PLC output changed Fat interface orientation/coordinates")
        if any(want_skin[k] != -want_fat[k] for k in shared):
            raise ValueError("reference PLC output Skin/Fat shared cofaces are not opposite")
    return {
        "status": "PASS_EXACT_SHARED_SKIN_FAT_INTERFACE",
        "shared_face_count": len(shared),
        "coordinate_key": "canonical_exact_f32_xyz_bits",
        "opposite_coface_orientation": True,
        "output_interface_matches_input": expected_skin_points is not None,
    }

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def _array_copy_dict(parent):
    return {k: np.array(v, copy=True) for k, v in parent.items()}

def _point_bits(p):
    b = np.asarray(p, np.float32).reshape(3).view(np.uint32).copy()
    b[b == 0x80000000] = 0
    return tuple(map(int, b))

def _oriented_boundary(tets):
    return np.asarray(T.m.boundary(np.asarray(tets, np.int64)), np.int64)

def _boundary_keyset(faces):
    return {tuple(sorted(map(int, f))) for f in np.asarray(faces, np.int64)}

def _edge_star(seed, edge, materials, tets, max_cells, max_points,
               reserved_cells, reserved_vertices):
    u, v = map(int, edge)
    rows = np.flatnonzero(np.any(tets == u, axis=1) & np.any(tets == v, axis=1))
    rows = np.asarray(rows[materials[rows] == materials[int(seed)]], np.int64)
    if int(seed) not in set(map(int, rows)) or len(rows) > int(max_cells):
        return None
    verts = np.unique(tets[rows])
    if len(verts) > int(max_points) or set(map(int, rows)) & reserved_cells:
        return None
    if set(map(int, verts)) & reserved_vertices:
        return None
    return np.sort(rows)

def _candidate_cavities(seed, bad, materials, tets, incidence, adjacency, deficits,
                        max_cells, max_points, reserved_cells, reserved_vertices):
    """Complete same-material node/edge stars followed by bounded connected face-balls."""
    candidates = []
    seen = set()
    for vertex in np.unique(tets[int(seed)]):
        rows = S.complete_same_material_vertex_star(
            int(seed), int(vertex), materials, tets, incidence,
            int(max_cells), int(max_points), reserved_cells, reserved_vertices)
        if rows is not None:
            key = tuple(map(int, rows))
            if key not in seen:
                seen.add(key)
                candidates.append((f"complete_vertex_star_node_{int(vertex)}", rows))
    edge_rows = []
    tet = list(map(int, tets[int(seed)]))
    for i in range(4):
        for j in range(i + 1, 4):
            e = (tet[i], tet[j])
            rows = _edge_star(seed, e, materials, tets, max_cells, max_points,
                              reserved_cells, reserved_vertices)
            if rows is not None:
                key = tuple(map(int, rows))
                if key not in seen:
                    seen.add(key)
                    edge_rows.append((f"complete_edge_star_{min(e)}_{max(e)}", rows))
    edge_rows.sort(key=lambda x: (len(x[1]), x[0]))
    candidates.extend(edge_rows)
    sizes = (2, 4, 8, 12, 24, 32, 48, 64, 96, 128, 192, 256)
    for size in sizes:
        if size > int(max_cells):
            continue
        rows = S.grow_bad_cluster(seed, adjacency, bad, materials, tets,
                                  size, max_points, reserved_cells,
                                  reserved_vertices, deficits)
        if rows is None or not len(rows):
            continue
        key = tuple(map(int, rows))
        if key not in seen:
            seen.add(key)
            candidates.append((f"connected_face_ball_{size}", rows))
    candidates.sort(key=lambda x: (len(x[1]), 0 if "vertex_star" in x[0] else 1, x[0]))
    return candidates

def _write_poly(path, points, facets, region_seed, region_attr):
    lines = [
        "# Exact-F32 boundary points for bounded local source cavity.",
        f"{len(points)} 3 0 1",
    ]
    for i, p in enumerate(np.asarray(points, np.float64), start=1):
        lines.append(f"{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g} 1")
    lines.append(f"{len(facets)} 1")
    for tri in np.asarray(facets, np.int64):
        lines.append("1 0")
        lines.append("3 " + " ".join(str(int(v) + 1) for v in tri))
    lines.extend(["0", "1"])
    q = np.asarray(region_seed, np.float64)
    lines.append(f"1 {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {int(region_attr)} 0")
    Path(path).write_text("\n".join(lines) + "\n")

def _read_node(path):
    rows = [x.split("#", 1)[0].strip() for x in Path(path).read_text().splitlines()]
    rows = [x for x in rows if x]
    n, dim, attrs, markers = map(int, rows[0].split()[:4])
    if (dim, attrs) != (3, 0):
        raise ValueError(f"unexpected TetGen node header {rows[0]}")
    out = {}
    for row in rows[1:1+n]:
        q = row.split()
        out[int(q[0])] = np.asarray([float(q[1]), float(q[2]), float(q[3])], np.float64)
    if len(out) != n:
        raise ValueError("TetGen node row count mismatch")
    return out

def _read_ele(path):
    rows = [x.split("#", 1)[0].strip() for x in Path(path).read_text().splitlines()]
    rows = [x for x in rows if x]
    n, corners, attrs = map(int, rows[0].split()[:3])
    if corners != 4 or attrs < 1:
        raise ValueError(f"unexpected TetGen element header {rows[0]}")
    out = []
    for row in rows[1:1+n]:
        q = row.split()
        a = float(q[5])
        if abs(a - round(a)) > 1e-8:
            raise ValueError(f"noninteger TetGen material attribute {a}")
        out.append((int(q[0]), tuple(map(int, q[1:5])), int(round(a))))
    if len(out) != n:
        raise ValueError("TetGen element row count mismatch")
    return out

def _write_operation_attempt(attempt_dir, points, facets, seed, material, flags, timeout_s):
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    poly = attempt_dir / "cavity.poly"
    _write_poly(poly, points, facets, seed, material)
    argv = [str(TETGEN_PATH), str(flags), str(poly)]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(argv, cwd=attempt_dir, capture_output=True, text=True,
                              timeout=max(0.1, float(timeout_s)))
        (attempt_dir / "tetgen.stdout.log").write_text(proc.stdout)
        (attempt_dir / "tetgen.stderr.log").write_text(proc.stderr)
        nodep = attempt_dir / "cavity.1.node"
        elep = attempt_dir / "cavity.1.ele"
        result = {
            "argv": argv, "return_code": int(proc.returncode),
            "wall_seconds": time.monotonic() - t0, "poly_sha256": sha(poly),
            "stdout_sha256": sha(attempt_dir / "tetgen.stdout.log"),
            "stderr_sha256": sha(attempt_dir / "tetgen.stderr.log"),
            "node_path": str(nodep), "element_path": str(elep),
            "node_sha256": sha(nodep) if nodep.is_file() else None,
            "element_sha256": sha(elep) if elep.is_file() else None,
        }
        if proc.returncode != 0 or not nodep.is_file() or not elep.is_file():
            result["status"] = "TETGEN_FAILED"
            return result, None
        result["status"] = "TETGEN_OUTPUT_READY"
        return result, (_read_node(nodep), _read_ele(elep))
    except subprocess.TimeoutExpired as exc:
        (attempt_dir / "timeout.json").write_text(json.dumps({
            "argv": argv, "timeout_seconds": float(timeout_s),
            "stdout": exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout,
            "stderr": exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr,
        }, indent=2) + "\n")
        return {"argv": argv, "status": "TETGEN_TIMEOUT",
                "wall_seconds": time.monotonic() - t0,
                "poly_sha256": sha(poly)}, None

def _candidate_from_tetgen(nodes, elements, p, old_cell_rows, material, input_boundary,
                           attempt_dir, existing_key_to_row, accepted_keys):
    local_input_keys = {_point_bits(q): int(i) for i, q in enumerate(p)}
    if len(local_input_keys) != len(p):
        raise ValueError("input F32 volume has duplicate exact point keys")
    appended = []
    appended_key_to_local = {}
    node_to_global = {}
    node_records = []
    for nid, xyz64 in sorted(nodes.items()):
        xyz32 = np.asarray(xyz64, np.float32)
        if xyz32.shape != (3,) or not np.isfinite(xyz32).all():
            raise ValueError("TetGen emitted nonfinite point")
        key = _point_bits(xyz32)
        if key in existing_key_to_row:
            global_row = int(existing_key_to_row[key])
            if not np.array_equal(np.asarray(p[global_row], np.float32).view(np.uint32), xyz32.view(np.uint32)):
                raise ValueError("TetGen mapped an input point with altered F32 coordinate")
            source_kind = "exact_existing_f32_point"
        elif key in appended_key_to_local:
            raise ValueError("TetGen emitted duplicate new F32 points")
        elif key in accepted_keys:
            raise ValueError("TetGen point duplicates a point appended by an accepted cavity")
        else:
            global_row = len(p) + len(appended)
            appended_key_to_local[key] = len(appended)
            appended.append(xyz32)
            source_kind = "new_tetgen_interior_f32_point"
        node_to_global[int(nid)] = global_row
        node_records.append({"tetgen_node_id": int(nid), "output_point_row": int(global_row),
                             "point_f32": xyz32.astype(float).tolist(),
                             "source_kind": source_kind,
                             "rounding_max_abs_m": float(np.max(np.abs(xyz64 - xyz32.astype(np.float64))))})
    boundary_vertices = np.unique(np.asarray(input_boundary, np.int64).reshape(-1))
    if any(int(v) not in set(existing_key_to_row.values()) for v in boundary_vertices):
        raise ValueError("PLC boundary point row absent from parent point domain")
    out_point_rows = set(node_to_global.values())
    for v in boundary_vertices:
        if int(v) not in out_point_rows:
            raise ValueError("TetGen omitted a required exact-F32 PLC boundary point")
    new_local = []
    for eid, nids, attr in elements:
        if int(attr) != int(material):
            raise ValueError(f"TetGen material region changed: {attr} != {material}")
        if any(int(nid) not in node_to_global for nid in nids):
            raise ValueError(f"TetGen element {eid} references absent node")
        new_local.append([node_to_global[int(nid)] for nid in nids])
    new_local = M.orient(np.asarray(p, np.float32) if not appended else
                         np.vstack((np.asarray(p, np.float32), np.asarray(appended, np.float32))),
                         np.asarray(new_local, np.int64))
    new_points = np.asarray(appended, np.float32).reshape((-1, 3))
    new_global_points = np.vstack((np.asarray(p, np.float32), new_points)) if len(new_points) else np.asarray(p, np.float32)
    if not len(new_local):
        raise ValueError("TetGen returned no replacement cells")
    local_boundary = _oriented_boundary(new_local)
    if _boundary_keyset(local_boundary) != _boundary_keyset(input_boundary):
        raise ValueError("TetGen output boundary facets differ from exact PLC input facets")
    return new_points, np.asarray(new_local, np.int64), node_records, new_global_points

def _make_point_parent_csr(nold, ops):
    offsets = [0] * (int(nold) + 1)
    rows = []
    for op in ops:
        parent_rows = sorted(set(map(int, op["old_cell_rows"])))
        for _ in op["new_point_output_rows"]:
            rows.extend(parent_rows)
            offsets.append(len(rows))
    return np.asarray(offsets, np.int64), np.asarray(rows, np.int64)

def _output_payload(parent, source_path, source_sha, points, cells, materials,
                    retained, operations, desired_boundary, role, readback_path):
    nold = len(parent["points_world_m"])
    nnew = len(points) - nold
    replacement = np.vstack([np.asarray(op["new_tetrahedra"], np.int64) for op in operations])
    replacement_parent_offsets = [0]
    replacement_parent_rows = []
    replacement_tet_offsets = [0]
    operation_parent_offsets = [0]
    operation_parent_rows = []
    for op in operations:
        parent_rows = sorted(set(map(int, op["old_cell_rows"])))
        for _ in op["new_tetrahedra"]:
            replacement_parent_rows.extend(parent_rows)
            replacement_parent_offsets.append(len(replacement_parent_rows))
        operation_parent_rows.extend(parent_rows)
        operation_parent_offsets.append(len(operation_parent_rows))
        replacement_tet_offsets.append(replacement_tet_offsets[-1] + len(op["new_tetrahedra"]))
    point_offsets, point_rows = _make_point_parent_csr(nold, operations)
    payload = _array_copy_dict(parent)
    payload["points_world_m"] = np.asarray(points, np.float32)
    if "points_world_m_f32" in parent:
        payload["points_world_m_f32"] = np.asarray(points, np.float32).copy()
    payload["tetrahedra"] = np.asarray(cells, np.int64)
    payload["material_indices"] = np.asarray(materials, dtype=np.asarray(parent["material_indices"]).dtype)
    payload["boundary_faces"] = _oriented_boundary(cells)
    payload["desired_boundary_faces"] = np.asarray(desired_boundary, np.int64)
    output_to_input = np.r_[np.asarray(retained, np.int64),
                           np.full(len(replacement), -1, np.int64)]
    inverse = np.full(len(parent["tetrahedra"]), -1, np.int64)
    inverse[np.asarray(retained, np.int64)] = np.arange(len(retained), dtype=np.int64)
    payload.update({
        "output_cell_to_input_cell": output_to_input,
        "input_cell_to_output_cell": inverse,
        "retained_input_cell_rows": np.asarray(retained, np.int64),
        "replacement_tetrahedra": replacement,
        "replacement_parent_offsets": np.asarray(replacement_parent_offsets, np.int64),
        "replacement_parent_cell_rows": np.asarray(replacement_parent_rows, np.int64),
        "replacement_tet_offsets": np.asarray(replacement_tet_offsets, np.int64),
        "quality_fixed_boundary_operation_parent_offsets": np.asarray(operation_parent_offsets, np.int64),
        "quality_fixed_boundary_operation_parent_cell_rows": np.asarray(operation_parent_rows, np.int64),
        "quality_fixed_boundary_operation_count": np.asarray([len(operations)], np.int64),
        "quality_fixed_boundary_operation_methods": np.asarray([op["proposal_method"] for op in operations]),
        "quality_fixed_boundary_operation_seed_rows": np.asarray([op["seed_cell_row"] for op in operations], np.int64),
        "quality_fixed_boundary_input_candidate_path": np.asarray([str(source_path)]),
        "quality_fixed_boundary_input_candidate_sha256": np.asarray([source_sha]),
        "quality_milp_operation_parent_offsets": np.asarray([0], np.int64),
        "quality_milp_operation_parent_cell_rows": np.empty(0, np.int64),
        "quality_milp_partial_bad_replacement": np.asarray([False]),
        "quality_milp_operation_count": np.asarray([0], np.int64),
        "quality_milp_operation_seed_rows": np.empty(0, np.int64),
        "quality_milp_input_candidate_path": np.asarray([str(source_path)]),
        "quality_milp_input_candidate_sha256": np.asarray([source_sha]),
        "quality_milp_input_node_to_output_node": np.arange(nold, dtype=np.int64),
        "quality_milp_output_node_to_input_node": np.r_[np.arange(nold, dtype=np.int64), np.full(nnew, -1, np.int64)],
        "quality_steiner_point_output_rows": np.concatenate([
            np.asarray(op["new_point_output_rows"], np.int64) for op in operations
        ]) if operations else np.empty(0, np.int64),
        "quality_steiner_operation_count": np.asarray([len(operations)], np.int64),
        "quality_steiner_cavity_selection_methods": np.asarray([op["candidate_selection_method"] for op in operations]),
        "output_point_to_input_point_row": np.r_[np.arange(nold, dtype=np.int64), np.full(nnew, -1, np.int64)],
        "input_node_to_output_node": np.arange(nold, dtype=np.int64),
        "output_node_to_input_node": np.r_[np.arange(nold, dtype=np.int64), np.full(nnew, -1, np.int64)],
        "source_I_vertex_quotient": np.r_[np.arange(nold, dtype=np.int64), np.full(nnew, -1, np.int64)],
        "source_input_vertex_quotient": np.r_[np.arange(nold, dtype=np.int64), np.full(nnew, -1, np.int64)],
        "source_input_node_to_output_node": np.arange(nold, dtype=np.int64),
        "output_point_parent_offsets": point_offsets,
        "output_point_parent_input_cell_rows": point_rows,
        "current_geometry_parent_path": np.asarray([str(source_path)]),
        "current_geometry_parent_sha256": np.asarray([source_sha]),
        "canonical_parent_union_operation_order": np.asarray(
            "tetgen-reference-plc-full-original-cavity-union-v1"
            if any(op.get("cavity_certification_mode") == "reference_plc_exact_boundary_volume_v1"
                   for op in operations)
            else "tetgen-local-plc-full-original-cavity-union-v1"
        ),
        "quality_tetgen_schema": np.asarray(
            "h706.tetgen-reference-plc-quality.v1"
            if any(op.get("cavity_certification_mode") == "reference_plc_exact_boundary_volume_v1"
                   for op in operations)
            else "h706.tetgen-local-plc-quality.v1"
        ),
        "quality_tetgen_binary_path": np.asarray([str(TETGEN_PATH)]),
        "quality_tetgen_binary_sha256": np.asarray([EXPECTED_TETGEN_SHA256]),
        "quality_tetgen_independent_readback_path": np.asarray([str(readback_path)]),
        "quality_tetgen_role": np.asarray([str(role)]),
    })
    if "input_point_to_output_point_row" in parent:
        payload["input_point_to_output_point_row"] = np.arange(nold, dtype=np.int64)
    if "point_parent_offsets" in parent:
        payload["point_parent_offsets"] = point_offsets.copy()
    if "point_parent_rows" in parent:
        payload["point_parent_rows"] = point_rows.copy()
    if "output_point_parent_cell_rows" in parent:
        payload["output_point_parent_cell_rows"] = point_rows.copy()
    if "source_interior_vertex_quotient" in parent:
        payload["source_interior_vertex_quotient"] = np.r_[
            np.asarray(parent["source_interior_vertex_quotient"], np.int64),
            np.full(nnew, -1, np.int64)]
    payload.update(S._historical_point_csr_updates(parent, nold, source_path, source_sha))
    return payload

def independent_readback(parent, readback, source_path, source_sha, retained,
                         desired_boundary, operations, old_bad, new_bad):
    nold = len(parent["points_world_m"])
    old_points = np.asarray(parent["points_world_m"], np.float32)
    out_points = np.asarray(readback["points_world_m"], np.float32)
    old_tets = np.asarray(parent["tetrahedra"], np.int64)
    out_tets = np.asarray(readback["tetrahedra"], np.int64)
    out_mat = np.asarray(readback["material_indices"])
    nnew = len(out_points) - nold
    if nnew < 0 or not np.array_equal(out_points[:nold].view(np.uint32), old_points.view(np.uint32)):
        raise ValueError("independent readback changed original point prefix")
    if "points_world_m_f32" in parent and not np.array_equal(
            np.asarray(readback["points_world_m_f32"], np.float32).view(np.uint32), out_points.view(np.uint32)):
        raise ValueError("independent readback F32 point copies disagree")
    retained = np.asarray(retained, np.int64)
    new_blocks = [np.asarray(op["new_tetrahedra"], np.int64) for op in operations]
    replacement = np.vstack(new_blocks)
    expected_tets = np.vstack((old_tets[retained], replacement))
    expected_mat = np.r_[np.asarray(parent["material_indices"])[retained],
                         np.concatenate([np.full(len(b), int(op["material_index"]),
                         dtype=np.asarray(parent["material_indices"]).dtype) for op, b in zip(operations, new_blocks)])]
    if not np.array_equal(out_tets, expected_tets) or not np.array_equal(out_mat, expected_mat):
        raise ValueError("independent readback output differs from canonical parent-union journal")
    if not np.array_equal(out_tets[:len(retained)], old_tets[retained]):
        raise ValueError("outside/retained parent cells changed")
    det, q = M.measurements(out_points, out_tets)
    good = np.asarray(M.passes(det, q), bool)
    if int(np.count_nonzero(~good)) != int(new_bad) or not int(new_bad) < int(old_bad):
        raise ValueError("independent readback did not preserve strict native bad-count decrease")
    if np.any(det <= 0):
        raise ValueError("independent readback has nonpositive measured output cell")
    if not S.exact_positive(out_points, out_tets):
        raise ValueError("independent readback has a nonpositive exact-F32 output cell")
    old_det, old_q = M.measurements(old_points, old_tets)
    old_good = np.asarray(M.passes(old_det, old_q), bool)
    for out_row, in_row in enumerate(np.asarray(readback["output_cell_to_input_cell"], np.int64)):
        if in_row >= 0 and old_good[int(in_row)] and not good[out_row]:
            raise ValueError("readback regressed a previously passing retained cell")
    actual_boundary = _oriented_boundary(out_tets)
    if not np.array_equal(T.m.boundary_identity(np.asarray(readback["boundary_faces"], np.int64)),
                          T.m.boundary_identity(actual_boundary)):
        raise ValueError("readback boundary_faces differ from reconstructed volume boundary")
    if not np.array_equal(T.m.boundary_identity(actual_boundary),
                          T.m.boundary_identity(np.asarray(desired_boundary, np.int64))):
        raise ValueError("readback changed exact global target boundary")
    if not np.array_equal(np.asarray(readback["desired_boundary_faces"], np.int64), np.asarray(desired_boundary, np.int64)):
        raise ValueError("readback desired boundary rows changed")
    po = np.asarray(readback["replacement_parent_offsets"], np.int64)
    pr = np.asarray(readback["replacement_parent_cell_rows"], np.int64)
    if po.shape != (len(replacement) + 1,) or po[0] != 0 or po[-1] != len(pr) or np.any(np.diff(po) < 0):
        raise ValueError("replacement parent CSR malformed")
    exp_off = [0]
    exp_rows = []
    tet_offsets = [0]
    op_offsets = [0]
    op_rows = []
    point_offsets = [0] * (nold + 1)
    point_rows = []
    new_point_rows_all = []
    cursor = 0
    for op, block in zip(operations, new_blocks):
        rows = np.asarray(sorted(set(map(int, op["old_cell_rows"]))), np.int64)
        if not np.array_equal(rows, np.asarray(op["full_cavity_parent_rows"], np.int64)):
            raise ValueError("operation full cavity parent rows are not canonical")
        for _ in block:
            exp_rows.extend(rows.tolist())
            exp_off.append(len(exp_rows))
        op_rows.extend(rows.tolist())
        op_offsets.append(len(op_rows))
        tet_offsets.append(tet_offsets[-1] + len(block))
        for point_row in op["new_point_output_rows"]:
            point_rows.extend(rows.tolist())
            point_offsets.append(len(point_rows))
            new_point_rows_all.append(int(point_row))
        start, end = map(int, op["replacement_tet_range"])
        if start != cursor or end - start != len(block):
            raise ValueError("operation replacement range mismatch")
        cursor = end
        local_new = out_tets[len(retained) + start:len(retained) + end]
        if not np.array_equal(local_new, block):
            raise ValueError("operation replacement connectivity changed")
        old_local = old_tets[np.asarray(op["old_cell_rows"], np.int64)]
        if op.get("cavity_certification_mode") == "reference_plc_exact_boundary_volume_v1":
            cert = _reference_output_certificate(
                out_points, old_local, local_new,
                np.asarray(op["boundary_faces"], np.int64),
                int(op["cavity_exact_volume_integer"]),
            )
        else:
            cert = M.certify(out_points, old_local, local_new)
        cert_ok = bool(cert[0]) if isinstance(cert, tuple) else bool(cert)
        if not cert_ok:
            raise ValueError("operation exact chain/volume certificate failed: " + repr(cert))
        if not np.all(M.passes(*M.measurements(out_points, local_new))):
            raise ValueError("operation new cells fail unchanged native M gates")
        if not np.array_equal(M.chain(old_local), M.chain(local_new)):
            raise ValueError("operation exact oriented chain mismatch")
        for rr in range(start, end):
            if not np.array_equal(pr[po[rr]:po[rr+1]], rows):
                raise ValueError("replacement tetra parent row omits full cavity union")
        for point_row in op["new_point_output_rows"]:
            if point_row < nold or point_row >= len(out_points):
                raise ValueError("operation new-point output row is outside appended point suffix")
            lo, hi = point_offsets[point_row], point_offsets[point_row+1]
            if not np.array_equal(np.asarray(point_rows[lo:hi], np.int64), rows):
                raise ValueError("new TetGen point parent row omits full cavity union")
        source_rows = np.asarray(op["boundary_source_vertex_rows"], np.int64)
        if any(int(x) < 0 or int(x) >= nold for x in source_rows):
            raise ValueError("operation boundary source point row out of range")
        if not np.array_equal(np.asarray(op["boundary_f32_bits"], np.uint32),
                              np.asarray(out_points[source_rows], np.float32).view(np.uint32)):
            raise ValueError("operation boundary F32 point binding changed")
    if not np.array_equal(po, np.asarray(exp_off, np.int64)) or not np.array_equal(pr, np.asarray(exp_rows, np.int64)):
        raise ValueError("replacement parent CSR does not encode each full cavity union")
    point_off = np.asarray(readback["output_point_parent_offsets"], np.int64)
    point_pr = np.asarray(readback["output_point_parent_input_cell_rows"], np.int64)
    if point_off.shape != (len(out_points) + 1,) or not np.array_equal(point_off, np.asarray(point_offsets, np.int64)) or not np.array_equal(point_pr, np.asarray(point_rows, np.int64)):
        raise ValueError("point-parent CSR does not reset old prefix and map each inserted node to cavity union")
    historical = S._historical_point_csr_updates(parent, nold, source_path, source_sha)
    for k, v in historical.items():
        if k not in readback or not np.array_equal(np.asarray(readback[k]), np.asarray(v)):
            raise ValueError("historical parent CSR/provenance archive mismatch: " + k)
    for key, expected in (
        ("quality_fixed_boundary_operation_parent_offsets", np.asarray(op_offsets, np.int64)),
        ("quality_fixed_boundary_operation_parent_cell_rows", np.asarray(op_rows, np.int64)),
        ("quality_fixed_boundary_operation_count", np.asarray([len(operations)], np.int64)),
        ("quality_fixed_boundary_operation_methods", np.asarray([op["proposal_method"] for op in operations])),
        ("quality_fixed_boundary_operation_seed_rows", np.asarray([op["seed_cell_row"] for op in operations], np.int64)),
        ("quality_steiner_operation_count", np.asarray([len(operations)], np.int64)),
        ("quality_steiner_point_output_rows", np.asarray(new_point_rows_all, np.int64)),
    ):
        if key not in readback or not np.array_equal(np.asarray(readback[key]), expected):
            raise ValueError("independent operation journal mismatch: " + key)
    if not np.array_equal(np.asarray(readback["retained_input_cell_rows"], np.int64), retained):
        raise ValueError("readback retained cell rows changed")
    expected_map = np.r_[retained, np.full(len(replacement), -1, np.int64)]
    if not np.array_equal(np.asarray(readback["output_cell_to_input_cell"], np.int64), expected_map):
        raise ValueError("readback output-to-input cell map differs from canonical map")
    inverse = np.full(len(old_tets), -1, np.int64)
    inverse[retained] = np.arange(len(retained), dtype=np.int64)
    if not np.array_equal(np.asarray(readback["input_cell_to_output_cell"], np.int64), inverse):
        raise ValueError("readback inverse cell map mismatch")
    if str(np.asarray(readback["current_geometry_parent_path"]).reshape(-1)[0]) != str(source_path):
        raise ValueError("readback current parent path mismatch")
    if str(np.asarray(readback["current_geometry_parent_sha256"]).reshape(-1)[0]) != source_sha:
        raise ValueError("readback current parent SHA mismatch")
    reference_mode = any(
        op.get("cavity_certification_mode") == "reference_plc_exact_boundary_volume_v1"
        for op in operations
    )
    return {"status": ("PASS_REFERENCE_PLC_INDEPENDENT_READBACK_SOURCE_ONLY"
                        if reference_mode else "PASS_TETGEN_PLC_INDEPENDENT_READBACK"),
            "original_point_prefix_unchanged": True,
            "all_new_cells_native_M_pass": True,
            "all_new_cells_exact_positive_and_cavity_certified": True,
            "all_original_cavity_parent_unions_complete": True,
            "old_point_parent_CSR_reset_and_historically_archived": True,
            "new_point_parent_CSR_full_cavity_union": True,
            "exact_F32_global_boundary_unchanged": True,
            "reference_plc_reconstruction": reference_mode,
            "outside_parent_cells_unchanged": True,
            "strict_global_bad_count_decrease": True,
            "native_bad_before": int(old_bad), "native_bad_after": int(new_bad),
            "operations": len(operations), "new_points": int(nnew),
            "replacement_tets": int(len(replacement)),
            "physical_admission": False, "native_steps": 0}

def _candidate_attempt(role, source_path, source_sha, arrays, points, tets, materials,
                       old_det, old_q, seed, bad, adjacency, incidence, deficits,
                       selection_label, rows, out_root, attempt_index, deadline, args,
                       reserved_new_keys):
    p = np.asarray(points, np.float32)
    oldrows = np.asarray(rows, np.int64)
    oldcells = np.asarray(tets[oldrows], np.int64)
    mats = np.asarray(materials[oldrows])
    if len(np.unique(mats)) != 1:
        return None, {"status": "REJECT_MIXED_MATERIAL"}
    material = int(mats[0])
    if len(oldrows) > args.maximum_old_cells:
        return None, {"status": "REJECT_CELL_CAP"}
    if len(np.unique(oldcells)) > args.maximum_plc_points:
        return None, {"status": "REJECT_PLC_POINT_CAP"}
    boundary = _oriented_boundary(oldcells)
    closed, edge_bad = T.closed(boundary)
    topo = T.manifold_topology(boundary)
    if not closed or topo["bad_vertex_links"] or topo["boundary_components"] != 1:
        return None, {"status": "REJECT_BOUNDARY_TOPOLOGY",
                      "edge_bad": int(edge_bad), "bad_vertex_links": len(topo["bad_vertex_links"]),
                      "boundary_components": int(topo["boundary_components"])}
    shell_proof = None
    if args.reference_plc:
        try:
            shell_proof = _reference_shell_proof(p, boundary)
            region_seed, seed_cell = _reference_region_seed(p, tets, oldrows, shell_proof)
        except Exception as exc:
            return None, {"status": "REJECT_REFERENCE_PLC_SHELL_OR_SEED", "error": repr(exc)}
    else:
        if not S.exact_positive(p, oldcells):
            return None, {"status": "REJECT_OLD_CAVITY_NOT_EXACT_POSITIVE"}
        seed_cell = int(rows[0])
        region_seed = np.asarray(p[tets[seed_cell]].astype(np.float64).mean(axis=0), np.float64)
    old_bad_local = int(np.count_nonzero(~M.passes(old_det[oldrows], old_q[oldrows])))
    if old_bad_local <= 0:
        return None, {"status": "REJECT_NO_OLD_BAD_CELL"}
    boundary_vertices = np.unique(boundary.reshape(-1))
    local_index = {int(v): i for i, v in enumerate(boundary_vertices)}
    facets = np.asarray([[local_index[int(v)] for v in f] for f in boundary], np.int64)
    plc_points = p[boundary_vertices]
    if len(np.unique(plc_points.view(np.uint32).reshape((-1, 3)), axis=0)) != len(plc_points):
        return None, {"status": "REJECT_DUPLICATE_F32_PLC_BOUNDARY_POINT"}
    if args.reference_plc:
        # The chosen seed is checked by exact point-in-shell replay, even when
        # every old interior tetrahedron has invalid geometric orientation.
        location = EXACT_SURFACE.point_location_prepared_indexed(
            tuple(Fraction(float(v)) for v in region_seed),
            shell_proof["point_location_prepared"],
        )
        if location["location"] != "inside":
            return None, {"status": "REJECT_REFERENCE_PLC_REGION_SEED_OUTSIDE"}
    attempt_base = Path(out_root) / "tetgen-attempts" / f"{role.lower()}-{attempt_index:05d}-{selection_label}"
    attempt_records = []
    expected_boundary_keys = _boundary_keyset(boundary)
    for flag_index, flags in enumerate(ATTEMPT_FLAGS):
        if time.monotonic() >= deadline:
            return None, {"status": "REJECT_TIME_BUDGET", "attempts": attempt_records}
        attempt_dir = attempt_base / f"flags-{flag_index:02d}"
        try:
            tetgen_result, parsed = _write_operation_attempt(
                attempt_dir, plc_points, facets, region_seed, material,
                flags, min(args.tetgen_timeout, max(0.1, deadline-time.monotonic())))
        except Exception as exc:
            attempt_records.append({"flags": flags, "status": "OWNER_EXCEPTION", "error": repr(exc)})
            continue
        attempt_records.append(tetgen_result)
        if parsed is None:
            continue
        nodes, elements = parsed
        existing_key_to_row = {_point_bits(q): i for i, q in enumerate(p)}
        try:
            appended, local_new, node_records, trial_points = _candidate_from_tetgen(
                nodes, elements, p, oldrows, material, boundary, attempt_dir,
                existing_key_to_row, reserved_new_keys)
            new_det, new_q = M.measurements(trial_points, local_new)
            if not np.all(M.passes(new_det, new_q)):
                attempt_records[-1]["status"] = "REJECT_NATIVE_M_GATE"
                attempt_records[-1]["new_bad_count"] = int(np.count_nonzero(~M.passes(new_det,new_q)))
                continue
            if not S.exact_positive(trial_points, local_new):
                attempt_records[-1]["status"] = "REJECT_EXACT_NONPOSITIVE_NEW_CELL"
                continue
            if args.reference_plc:
                cert = _reference_output_certificate(
                    trial_points, oldcells, local_new, boundary,
                    shell_proof["exact_signed_volume_integer"],
                )
            else:
                cert = M.certify(trial_points, oldcells, local_new)
            cert_ok = bool(cert[0]) if isinstance(cert, tuple) else bool(cert)
            if not cert_ok or not np.array_equal(M.chain(oldcells), M.chain(local_new)):
                attempt_records[-1]["status"] = "REJECT_EXACT_CAVITY_CHAIN_OR_VOLUME"
                attempt_records[-1]["certificate"] = repr(cert)
                continue
            if len(set(map(int, local_new.reshape(-1)))) < len(boundary_vertices):
                attempt_records[-1]["status"] = "REJECT_BOUNDARY_NODE_USAGE"
                continue
            face_keys = _boundary_keyset(_oriented_boundary(local_new))
            if face_keys != expected_boundary_keys:
                attempt_records[-1]["status"] = "REJECT_BOUNDARY_FACET_SET"
                continue
            new_local_bad = int(np.count_nonzero(~M.passes(new_det, new_q)))
            if not new_local_bad < old_bad_local:
                attempt_records[-1]["status"] = "REJECT_NO_LOCAL_BAD_DECREASE"
                continue
            # Every generated or reused TetGen point must be mapped to an exact F32 row.
            new_rows = sorted({int(v) for v in local_new.reshape(-1) if int(v) >= len(p)})
            if any(row >= len(trial_points) for row in new_rows):
                attempt_records[-1]["status"] = "REJECT_POINT_ROW_MAP"
                continue
            operation = {
                "proposal_method": ("tetgen_reference_plc" if args.reference_plc else "tetgen_local_plc"),
                "cavity_certification_mode": (
                    "reference_plc_exact_boundary_volume_v1" if args.reference_plc
                    else "positive_old_cavity_exact_volume_v1"
                ),
                "candidate_selection_method": str(selection_label),
                "seed_cell_row": int(seed),
                "region_seed_source_cell_row": int(seed_cell),
                "material_index": material,
                "old_cell_rows": oldrows.astype(int).tolist(),
                "full_cavity_parent_rows": sorted(set(map(int, oldrows))),
                "old_vertex_rows": np.unique(oldcells).astype(int).tolist(),
                "boundary_source_vertex_rows": boundary_vertices.astype(int).tolist(),
                "boundary_faces": boundary.astype(int).tolist(),
                "boundary_f32_bits": p[boundary_vertices].view(np.uint32).astype(int).tolist(),
                "cavity_exact_volume_integer": (
                    str(shell_proof["exact_signed_volume_integer"]) if args.reference_plc else None
                ),
                "cavity_boundary_self_intersection_count": (
                    int(shell_proof["self_intersection_count"]) if args.reference_plc else None
                ),
                "cavity_boundary_topology": (
                    {"edge_bad_count": int(shell_proof["edge_bad_count"]),
                     "bad_vertex_links": list(map(int, shell_proof["topology"]["bad_vertex_links"])),
                     "boundary_components": int(shell_proof["topology"]["boundary_components"])}
                    if args.reference_plc else None
                ),
                "plc_facets_local": facets.astype(int).tolist(),
                "tetgen_flags": flags,
                "tetgen_attempt": tetgen_result,
                "tetgen_node_records": node_records,
                "new_point_output_rows": new_rows,
                "new_point_f32": trial_points[new_rows].astype(float).tolist(),
                "new_tetrahedra": local_new.astype(int).tolist(),
                "old_bad_count": old_bad_local,
                "new_bad_count": new_local_bad,
                "exact_oriented_chain_matches": True,
                "exact_signed_volume_matches": True,
                "all_new_cells_native_M_pass": True,
                "all_new_cells_exact_positive": True,
                "boundary_facets_exact": True,
            }
            return operation, {"status": "PASS_TETGEN_LOCAL_PLC", "attempts": attempt_records,
                               "operation": operation}
        except Exception as exc:
            attempt_records[-1]["status"] = "REJECT_CANDIDATE_VALIDATION"
            attempt_records[-1]["error"] = repr(exc)
    return None, {"status": "NO_TETGEN_VARIANT_PASSED", "attempts": attempt_records}

def run_role(role, entry, state, output_dir, deadline, args):
    role_started = time.monotonic()
    source_path = Path(entry["path"])
    source_sha = sha(source_path)
    if source_sha != entry["sha256"]:
        raise ValueError(f"{role} source candidate SHA mismatch")
    parent = load_npz(source_path)
    points = np.asarray(parent["points_world_m"])
    tets = np.asarray(parent["tetrahedra"], np.int64)
    materials = np.asarray(parent["material_indices"])
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{role} requires finite exact F32 world points")
    if tets.ndim != 2 or tets.shape[1] != 4 or materials.shape != (len(tets),):
        raise ValueError(f"{role} volume/material arrays malformed")
    target, target_key = S.boundary_target(parent)
    original_boundary = _oriented_boundary(tets)
    if not np.array_equal(T.m.boundary_identity(original_boundary), T.m.boundary_identity(target)):
        raise ValueError(f"{role} source boundary mismatch")
    det, quality = M.measurements(points, tets)
    input_nonpositive_count = int(np.count_nonzero(det <= 0))
    if input_nonpositive_count and not args.reference_plc:
        raise ValueError(f"{role} source has nonpositive cells; explicit reference-PLC mode is required")
    declared_bad = entry.get("native_quality_failures")
    if declared_bad is not None and int(declared_bad) != int(np.count_nonzero(~M.passes(det, quality))):
        raise ValueError(f"{role} source bad count differs from pinned entry")
    declared_nonpositive = entry.get("nonpositive", parent.get("nonpositive"))
    if declared_nonpositive is not None:
        declared_nonpositive = int(np.asarray(declared_nonpositive).reshape(-1)[0])
        if declared_nonpositive != input_nonpositive_count:
            raise ValueError(f"{role} exact nonpositive count differs from source pin")
    elif not args.reference_plc:
        raise ValueError(f"{role} source must pin nonpositive=0")
    bad = np.flatnonzero(~M.passes(det, quality))
    old_bad_count = int(len(bad))
    if old_bad_count == 0:
        return {"role": role, "status": "NO_NATIVE_BAD_CELLS",
                "input": pin(source_path), "native_bad_before": 0, "native_bad_after": 0}
    deficits = -np.log(np.maximum(np.minimum(np.minimum(quality / QUALITY_GATE,
                      det / (6.0 * VOLUME_GATE)), 1.0), 1.0e-300))
    adjacency = S.face_bad_adjacency(tets, materials, bad)
    node_counts = np.bincount(tets.reshape(-1), minlength=len(points))
    node_offsets = np.r_[0, np.cumsum(node_counts)]
    corner_order = np.argsort(tets.reshape(-1), kind="stable")
    corner_cell_rows = corner_order // 4
    bad_corner_incidence = {
        int(v): corner_cell_rows[node_offsets[int(v)]:node_offsets[int(v)+1]].astype(np.int32, copy=True)
        for v in np.unique(tets[bad].reshape(-1))
    }
    del node_counts, node_offsets, corner_order, corner_cell_rows
    bad_corner_vertices = sorted(bad_corner_incidence)
    reserved_cells, reserved_vertices = set(), set()
    operations, attempts, rejection = [], [], Counter()
    accepted_new_keys = set()
    current_points = points.copy()
    current_tets = tets.copy()
    current_materials = materials.copy()
    accepted_rows = set()
    current_bad_count = old_bad_count
    seed_order = sorted(map(int, bad), key=lambda r: (-float(deficits[r]), r))
    for seed in seed_order:
        if time.monotonic() >= deadline or len(operations) >= args.maximum_operations:
            break
        if seed in reserved_cells:
            continue
        candidates = _candidate_cavities(
            seed, bad, materials, tets, bad_corner_incidence, adjacency,
            deficits, args.maximum_old_cells, args.maximum_plc_points,
            reserved_cells, reserved_vertices)
        for label, rows in candidates:
            if time.monotonic() >= deadline or len(attempts) >= args.maximum_attempts:
                break
            key = tuple(map(int, rows))
            if key in accepted_rows or set(key) & reserved_cells:
                rejection["candidate_cell_overlap"] += 1
                continue
            old_vertices = set(map(int, np.unique(tets[rows])))
            if old_vertices & reserved_vertices:
                rejection["candidate_vertex_overlap"] += 1
                continue
            attempts.append({"seed_cell_row": int(seed), "candidate_selection_method": label,
                             "old_cell_count": int(len(rows)), "old_vertex_count": len(old_vertices)})
            op, result = _candidate_attempt(
                role, source_path, source_sha, parent, current_points, tets, materials, det, quality,
                seed, bad, adjacency, bad_corner_incidence, deficits, label, rows,
                output_dir, len(attempts), deadline, args, accepted_new_keys)
            attempts[-1]["result"] = result
            if op is None:
                rejection[str(result.get("status", "UNKNOWN"))] += 1
                continue
            # Track old/new global bad-count delta from immutable source cells.
            if not int(op["new_bad_count"]) < int(op["old_bad_count"]):
                rejection["not_strict_local_bad_drop"] += 1
                continue
            op["replacement_tet_range"] = [
                int(sum(len(x["new_tetrahedra"]) for x in operations)),
                int(sum(len(x["new_tetrahedra"]) for x in operations) + len(op["new_tetrahedra"]))
            ]
            op["new_point_output_rows"] = sorted(set(map(int, op["new_point_output_rows"])))
            operations.append(op)
            point_block = np.asarray(op["new_point_f32"], np.float32).reshape((-1, 3))
            current_points = np.vstack((current_points, point_block)) if len(point_block) else current_points
            rowsset = set(map(int, rows))
            reserved_cells.update(rowsset)
            reserved_vertices.update(old_vertices)
            accepted_rows.add(key)
            accepted_new_keys.update(_point_bits(q) for q in op["new_point_f32"])
            current_bad_count -= int(op["old_bad_count"]) - int(op["new_bad_count"])
            break
    if not operations:
        return {"role": role, "status": "NO_VALID_TETGEN_CAVITY_PROPOSAL",
                "input": pin(source_path), "native_bad_before": old_bad_count,
                "native_bad_after": old_bad_count, "attempts": attempts,
                "rejections": dict(rejection), "elapsed_s": time.monotonic()-role_started}
    retained = np.asarray([i for i in range(len(tets)) if i not in reserved_cells], np.int64)
    new_points = current_points
    new_tets = np.vstack([tets[retained]] + [np.asarray(op["new_tetrahedra"], np.int64)
                                            for op in operations])
    new_materials = np.r_[materials[retained],
                          np.concatenate([np.full(len(op["new_tetrahedra"]), int(op["material_index"]),
                          dtype=materials.dtype) for op in operations])]
    output_det, output_quality = M.measurements(new_points, new_tets)
    actual_bad_count = int(np.count_nonzero(~M.passes(output_det, output_quality)))
    if actual_bad_count != current_bad_count or not actual_bad_count < old_bad_count:
        raise ValueError(f"{role} final global bad count mismatch/nondecrease {old_bad_count}->{actual_bad_count} expected {current_bad_count}")
    output_nonpositive_count = int(np.count_nonzero(output_det <= 0))
    if output_nonpositive_count:
        raise ValueError(f"{role} final full mesh has {output_nonpositive_count} nonpositive measured cells")
    if not S.exact_positive(new_points, new_tets):
        raise ValueError(f"{role} final full mesh has a nonpositive exact-F32 cell")
    if args.reference_plc and actual_bad_count != 0:
        raise ValueError(f"{role} reference-PLC result still has {actual_bad_count} native q/det failures")
    if not np.array_equal(_boundary_keyset(_oriented_boundary(new_tets)), _boundary_keyset(target)):
        raise ValueError(f"{role} final exact boundary face set changed")
    input_after_sha = sha(source_path)
    if input_after_sha != source_sha:
        raise ValueError("source candidate changed during construction")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{role.lower()}-tetgen-plc-quality-proposal.npz"
    payload = _output_payload(parent, source_path, source_sha, new_points, new_tets,
                              new_materials, retained, operations, target, role,
                              output_dir / f"{role.lower()}-independent-readback.json")
    np.savez_compressed(output_path, **payload)
    output_sha = sha(output_path)
    readback_arrays = load_npz(output_path)
    if sha(output_path) != output_sha:
        raise ValueError("output candidate changed during readback")
    readback_result = independent_readback(
        parent, readback_arrays, source_path, source_sha, retained, target, operations,
        old_bad_count, actual_bad_count)
    readback_path = output_dir / f"{role.lower()}-independent-readback.json"
    readback_path.write_text(json.dumps(readback_result, indent=2, sort_keys=True) + "\n")
    # Re-open after journal write and verify the declared journal binding.
    readback2 = load_npz(output_path)
    if str(np.asarray(readback2["quality_tetgen_independent_readback_path"]).reshape(-1)[0]) != str(readback_path):
        raise ValueError("serialized independent readback path binding changed")
    if sha(source_path) != source_sha:
        raise ValueError("source candidate changed during independent readback")
    operations_path = output_dir / f"{role.lower()}-operations.json"
    operations_doc = {
        "schema": ("h706.tetgen-reference-plc-operation-journal.v1" if args.reference_plc
                   else "h706.tetgen-local-plc-operation-journal.v1"),
        "reference_dependency_pins": reference_dependency_pins() if args.reference_plc else None,
        "source_candidate": pin(source_path),
        "producer": pin(Path(__file__)),
        "tetgen": pin(TETGEN_PATH),
        "operations": operations,
        "attempts": attempts,
        "rejections": dict(rejection),
        "source_input_unchanged": True,
        "physical_admission": False,
        "native_steps": 0,
    }
    operations_path.write_text(json.dumps(operations_doc, indent=2, sort_keys=True) + "\n")
    return {
        "role": role,
        "status": ("PASS_REFERENCE_PLC_ROLE_SOURCE_ONLY" if args.reference_plc
                   else "PASS_TETGEN_PLC_FULL_CAVITY_BATCH"),
        "input": pin(source_path), "output": pin(output_path),
        "independent_readback": pin(readback_path), "operation_journal": pin(operations_path),
        "state_input_sha256": state.get("_state_sha256"),
        "input_points": int(len(points)), "output_points": int(len(new_points)),
        "new_points": int(len(new_points)-len(points)),
        "input_cells": int(len(tets)), "output_cells": int(len(new_tets)),
        "replaced_cells": int(len(reserved_cells)),
        "replacement_cells": int(len(new_tets)-len(retained)),
        "native_bad_before": old_bad_count, "native_bad_after": actual_bad_count,
        "input_nonpositive_cells": input_nonpositive_count,
        "output_nonpositive_cells": output_nonpositive_count,
        "strict_native_bad_count_drop": True,
        "source_exact_positive": bool(input_nonpositive_count == 0),
        "all_output_measured_positive": True, "all_new_cells_exact_positive": True,
        "all_new_cells_native_M_pass": True, "exact_chain_and_volume_per_cavity": True,
        "outside_cells_unchanged": True, "exact_boundary_faces_unchanged": True,
        "full_cavity_parent_union_point_and_cell_csr": True,
        "historical_parent_point_csr_archived": True,
        "native_thresholds_unchanged": True, "mass_mapping": "NOT_RUN",
        "native_admission": False, "physical_admission": False, "native_steps": 0,
        "reference_plc_reconstruction": bool(args.reference_plc),
        "reference_dependency_pins": reference_dependency_pins() if args.reference_plc else None,
        "target_boundary_key": target_key, "operations": len(operations),
        "attempts": attempts, "rejections": dict(rejection),
        "elapsed_s": time.monotonic()-role_started,
    }

def run_state(state_path, output_path, seconds, args):
    start = time.monotonic()
    state_path = Path(state_path)
    state_pin = pin(state_path)
    raw = json.loads(state_path.read_text())
    wrapped = "current_verified_state_20261031" in raw
    state = raw.get("current_verified_state_20261031", raw)
    state = dict(state)
    state["_state_sha256"] = state_pin["sha256"]
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=False)
    available = []
    for role in ("Skin", "Fat"):
        key = role + "_current"
        if key in state:
            nbad = int(state[key].get("native_quality_failures", -1))
            if nbad < 0:
                raise ValueError(f"{role} source state must pin nonnegative native bad count")
            available.append((role, nbad))
    available.sort(key=lambda x: (-x[1], 0 if x[0] == "Fat" else 1))
    results, schedule = [], []
    updated = dict(state)
    for i, (role, pinned_bad) in enumerate(available):
        remain = max(0.0, start + float(seconds) - time.monotonic())
        slice_s = remain / max(1, len(available)-i)
        deadline = time.monotonic() + slice_s
        schedule.append({"role": role, "pinned_native_bad_before": pinned_bad,
                         "budget_seconds": float(slice_s),
                         "policy": "equal_remaining_slice_sorted_by_bad_desc_fat_tiebreak"})
        result = run_role(role, state[role+"_current"], state, out, deadline, args) if slice_s > 0.01 else {
            "role": role, "status": "SKIPPED_GLOBAL_TIME_BUDGET"}
        results.append(result)
        if (not args.reference_plc
                and result.get("status") == "PASS_TETGEN_PLC_FULL_CAVITY_BATCH"):
            updated[role+"_current"] = {
                **state[role+"_current"], **result["output"],
                "nodes": result["output_points"], "cells": result["output_cells"],
                "native_quality_failures": result["native_bad_after"], "nonpositive": 0,
            }
    pair_audit = None
    if args.reference_plc:
        by_role = {str(r["role"]): r for r in results}
        complete = set(by_role) == {"Skin", "Fat"} and all(
            by_role[role].get("status") in {
                "PASS_REFERENCE_PLC_ROLE_SOURCE_ONLY", "NO_NATIVE_BAD_CELLS"
            } for role in ("Skin", "Fat")
        )
        if complete:
            source_arrays = {}
            output_arrays = {}
            for role in ("Skin", "Fat"):
                source_arrays[role] = load_npz(by_role[role]["input"]["path"])
                out_pin = by_role[role].get("output")
                output_arrays[role] = load_npz(out_pin["path"] if out_pin else by_role[role]["input"]["path"])
            try:
                source_skin_boundary, _ = S.boundary_target(source_arrays["Skin"])
                source_fat_boundary, _ = S.boundary_target(source_arrays["Fat"])
                output_skin_boundary, _ = S.boundary_target(output_arrays["Skin"])
                output_fat_boundary, _ = S.boundary_target(output_arrays["Fat"])
                pair_audit = _reference_shared_interface_audit(
                    source_arrays["Skin"]["points_world_m"], source_skin_boundary,
                    source_arrays["Fat"]["points_world_m"], source_fat_boundary,
                    output_arrays["Skin"]["points_world_m"], output_skin_boundary,
                    output_arrays["Fat"]["points_world_m"], output_fat_boundary,
                )
            except Exception as exc:
                pair_audit = {"status": "REJECT_SHARED_SKIN_FAT_INTERFACE", "error": repr(exc)}
        else:
            pair_audit = {
                "status": "REJECT_INCOMPLETE_ROLE_PAIR",
                "role_status": {role: row.get("status") for role, row in by_role.items()},
            }
        if pair_audit.get("status") == "PASS_EXACT_SHARED_SKIN_FAT_INTERFACE":
            for role in ("Skin", "Fat"):
                row = by_role[role]
                if row.get("status") == "PASS_REFERENCE_PLC_ROLE_SOURCE_ONLY":
                    updated[role+"_current"] = {
                        **state[role+"_current"], **row["output"],
                        "nodes": row["output_points"], "cells": row["output_cells"],
                        "native_quality_failures": row["native_bad_after"], "nonpositive": 0,
                    }
        else:
            updated = dict(state)
    updated["physical_admission"] = False
    updated["native_steps"] = 0
    updated["mass_mapping"] = "NOT_RUN"
    updated.pop("independent_replay", None)
    updated.pop("exact_context", None)
    if wrapped:
        output_state = dict(raw)
        output_state["current_verified_state_20261031"] = updated
    else:
        output_state = updated
    state_out = out / "state.json"
    state_out.write_text(json.dumps(output_state, indent=2) + "\n")
    reference_pair_pass = bool(
        args.reference_plc and pair_audit
        and pair_audit.get("status") == "PASS_EXACT_SHARED_SKIN_FAT_INTERFACE"
    )
    report = {
        "schema": ("h706.tetgen-reference-plc-quality-batch.v1" if args.reference_plc
                   else "h706.tetgen-local-plc-quality-batch.v1"),
        "status": (
            "PASS_REFERENCE_PLC_RECONSTRUCTION_SOURCE_ONLY" if reference_pair_pass
            else "REFERENCE_PLC_PAIR_NOT_QUALIFIED" if args.reference_plc
            else "PASS_WITH_PARTIAL_NATIVE_DEFECTS" if any(
                r.get("status") == "PASS_TETGEN_PLC_FULL_CAVITY_BATCH" for r in results
            ) else "NO_VALID_BOUNDED_PROGRESS"
        ),
        "source_state": state_pin, "output_state": pin(state_out),
        "producer": pin(Path(__file__)),
        "quality_owner_750": pin(QUALITY_OWNER_PATH),
        "tetgen": pin(TETGEN_PATH),
        "quality_thresholds": {"normalized_quality": QUALITY_GATE, "det6_m3": 6.0*VOLUME_GATE},
        "native_thresholds_unchanged": True,
        "local_cell_cap": int(args.maximum_old_cells),
        "attempt_flags": list(ATTEMPT_FLAGS),
        "role_schedule": schedule, "role_results": results,
        "reference_plc_reconstruction": bool(args.reference_plc),
        "paired_interface_audit": pair_audit,
        "reference_dependency_pins": reference_dependency_pins() if args.reference_plc else None,
        "all_new_cells_measured_positive": all(r.get("all_new_cells_exact_positive", True) for r in results),
        "all_new_cells_pass_native_M": all(r.get("all_new_cells_native_M_pass", True) for r in results),
        "mass_mapping": "NOT_RUN", "native_steps": 0,
        "native_admission": False, "physical_admission": False,
        "elapsed_s": time.monotonic()-start, "time_budget_s": float(seconds),
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report

def _fixture_disjoint_batch(root, args):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    tetra = np.asarray([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], np.float32)
    faces = np.asarray([[0,2,1], [0,1,3], [1,2,3], [2,0,3]], np.int64)
    point_blocks, cell_blocks = [], []
    for offset in (np.asarray([0,0,0], np.float32), np.asarray([3,0,0], np.float32)):
        base = len(point_blocks) * 5
        corners = tetra + offset
        center = np.asarray([[0.3,0.3,1.0e-7]], np.float32) + offset
        block = np.vstack((corners, center))
        point_blocks.append(block)
        cell_blocks.append(np.column_stack((
            np.full(len(faces), base + 4, np.int64), faces + base)))
    points = np.vstack(point_blocks).astype(np.float32)
    tets = M.orient(points, np.vstack(cell_blocks))
    det, quality = M.measurements(points, tets)
    bad = int(np.count_nonzero(~M.passes(det, quality)))
    if bad != 2:
        raise AssertionError(f"two-cavity fixture must start with exactly two bad cells, got {bad}")
    boundary = _oriented_boundary(tets)
    inp = root / "two-disjoint-cavity-input.npz"
    np.savez_compressed(
        inp, points_world_m=points, points_world_m_f32=points.copy(),
        tetrahedra=tets, material_indices=np.full(len(tets), 15, np.int32),
        boundary_faces=boundary, desired_boundary_faces=boundary,
        nonpositive=np.asarray([0], np.int64),
        output_point_parent_offsets=np.zeros(len(points)+1, np.int64),
        output_point_parent_input_cell_rows=np.empty(0, np.int64))
    entry = {"path": str(inp), "sha256": sha(inp),
             "native_quality_failures": bad, "nonpositive": 0}
    batch_args = argparse.Namespace(**vars(args))
    batch_args.maximum_operations = max(2, int(args.maximum_operations))
    result = run_role("Fat", entry, {}, root / "producer",
                      time.monotonic()+float(args.fixture_seconds), batch_args)
    if (result.get("status") != "PASS_TETGEN_PLC_FULL_CAVITY_BATCH"
            or result.get("native_bad_before") != 2
            or result.get("native_bad_after") != 0
            or result.get("operations") != 2):
        raise AssertionError("two-disjoint-cavity batch/append-index fixture failed: "+
                             json.dumps(result, sort_keys=True)[:3000])
    readback = json.loads(Path(result["independent_readback"]["path"]).read_text())
    return {"status": "PASS_TWO_DISJOINT_CAVITIES_2_BAD_TO_0",
            "input": pin(inp), "producer_result": result,
            "independent_readback": readback}

def fixture(output_path, args):
    if args.reference_plc:
        return fixture_reference_plc(output_path, args)
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=False)
    # Tetra shell with one near-face interior point creates a closed one-cell quality defect.
    corners = np.asarray([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1],
    ], np.float32)
    boundary = np.asarray([
        [0,2,1], [0,1,3], [1,2,3], [2,0,3],
    ], np.int64)
    center = np.asarray([[0.3,0.3,1.0e-7]], np.float32)
    points = np.vstack((corners, center))
    # Cone the fixed tetra boundary to one point. Orient keeps all source volumes positive.
    old = M.orient(points, np.column_stack((np.full(len(boundary), len(corners), np.int64), boundary)))
    if np.any(M.measurements(points, old)[0] <= 0):
        raise AssertionError("fixture source cells must be positive")
    old_det, old_q = M.measurements(points, old)
    old_bad = int(np.count_nonzero(~M.passes(old_det, old_q)))
    if old_bad != 1:
        raise AssertionError(f"fixture must contain exactly one native-bad cell, got {old_bad}")
    input_path = out / "one-bad-cavity-input.npz"
    np.savez_compressed(input_path, points_world_m=points, points_world_m_f32=points.copy(),
                        tetrahedra=old, material_indices=np.full(len(old), 15, np.int32),
                        boundary_faces=boundary, desired_boundary_faces=boundary,
                        nonpositive=np.asarray([0], np.int64),
                        output_point_parent_offsets=np.zeros(len(points)+1, np.int64),
                        output_point_parent_input_cell_rows=np.empty(0, np.int64))
    entry = {"path": str(input_path), "sha256": sha(input_path),
             "native_quality_failures": old_bad, "nonpositive": 0}
    fixture_args = argparse.Namespace(**vars(args))
    fixture_args.maximum_operations = max(1, int(args.maximum_operations))
    fixture_args.maximum_attempts = max(1, int(args.maximum_attempts))
    result = run_role("Skin", entry, {}, out / "producer", time.monotonic()+float(args.fixture_seconds),
                      fixture_args)
    if result.get("status") != "PASS_TETGEN_PLC_FULL_CAVITY_BATCH" or result.get("native_bad_after") != 0:
        raise AssertionError("tiny TetGen fixture did not complete 1-bad-to-0 exact replay: "+
                             json.dumps(result, sort_keys=True)[:3000])
    check = json.loads(Path(result["independent_readback"]["path"]).read_text())
    if check.get("status") != "PASS_TETGEN_PLC_INDEPENDENT_READBACK":
        raise AssertionError("tiny fixture independent readback did not pass")
    multi = _fixture_disjoint_batch(out / "multi-operation", args)
    fixture_report = {
        "schema": "h706.tetgen-local-plc-quality-fixture.v1",
        "status": "PASS_TINY_1_BAD_TO_0_WITH_INDEPENDENT_READBACK",
        "multi_operation_batch": multi,
        "input": pin(input_path), "producer_result": result,
        "independent_readback": check,
        "exact_F32_old_point_prefix_preserved": True,
        "physical_admission": False, "native_steps": 0,
    }
    (out / "fixture-report.json").write_text(json.dumps(fixture_report, indent=2, sort_keys=True)+"\n")
    return fixture_report


def fixture_reference_plc(output_path, args):
    """Small real-TetGen control: valid PLC boundary, one inverted old cell."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=False)
    boundary_points = np.asarray([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ], np.float32)
    boundary = np.asarray([
        [0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3],
    ], np.int64)
    # Build an initially valid coning, then move its interior point beyond one
    # face without changing the oriented cell chain or fixed PLC shell.
    good_points = np.vstack((boundary_points, np.asarray([[0.25, 0.25, 0.25]], np.float32)))
    tets = M.orient(good_points, np.column_stack((np.full(len(boundary), 4, np.int64), boundary)))
    points = good_points.copy()
    points[4] = np.asarray([0.4, 0.4, 0.4], np.float32)
    input_det, input_quality = M.measurements(points, tets)
    input_bad = int(np.count_nonzero(~M.passes(input_det, input_quality)))
    input_nonpositive = int(np.count_nonzero(input_det <= 0))
    if input_bad != 1 or input_nonpositive != 1:
        raise AssertionError("inverted-interior fixture must contain one native-bad, exact-negative cell")
    if not np.array_equal(T.m.boundary_identity(_oriented_boundary(tets)),
                          T.m.boundary_identity(boundary)):
        raise AssertionError("inverted-interior fixture changed its valid oriented PLC boundary")
    input_path = out / "inverted-interior-valid-boundary.npz"
    np.savez_compressed(
        input_path, points_world_m=points, points_world_m_f32=points.copy(),
        tetrahedra=tets, material_indices=np.full(len(tets), 15, np.int32),
        boundary_faces=boundary, desired_boundary_faces=boundary,
        nonpositive=np.asarray([input_nonpositive], np.int64),
        output_point_parent_offsets=np.zeros(len(points)+1, np.int64),
        output_point_parent_input_cell_rows=np.empty(0, np.int64),
    )
    _reference_shell_proof(points, boundary)
    entry = {"path": str(input_path), "sha256": sha(input_path),
             "native_quality_failures": input_bad, "nonpositive": input_nonpositive}
    fixture_args = argparse.Namespace(**vars(args))
    fixture_args.maximum_old_cells = max(4, int(args.maximum_old_cells))
    fixture_args.maximum_plc_points = max(4, int(args.maximum_plc_points))
    fixture_args.maximum_attempts = max(32, int(args.maximum_attempts))
    fixture_args.maximum_operations = max(1, int(args.maximum_operations))
    fixture_args.reference_plc = True
    result = run_role("Skin", entry, {}, out / "producer",
                      time.monotonic()+float(args.fixture_seconds), fixture_args)
    if (result.get("status") != "PASS_REFERENCE_PLC_ROLE_SOURCE_ONLY"
            or result.get("native_bad_before") != 1
            or result.get("native_bad_after") != 0
            or result.get("input_nonpositive_cells") != 1
            or result.get("output_nonpositive_cells") != 0):
        raise AssertionError("reference PLC did not reconstruct one inverted cell with exact gates: "+
                             json.dumps(result, sort_keys=True)[:3000])
    readback = json.loads(Path(result["independent_readback"]["path"]).read_text())
    if readback.get("status") != "PASS_REFERENCE_PLC_INDEPENDENT_READBACK_SOURCE_ONLY":
        raise AssertionError("reference PLC independent full-cavity readback failed")
    # A closed, orientable but self-crossing octahedral shell must be rejected
    # by the same exact shell audit before any TetGen invocation.
    crossing_points = np.asarray([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [-1, -1, -1.2], [0, 0, -1],
    ], np.float32)
    crossing_faces = np.asarray([
        [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],
        [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3],
    ], np.int64)[:, [0, 2, 1]]
    if not T.closed(crossing_faces)[0] or T.exact_volume(crossing_points, crossing_faces) <= 0:
        raise AssertionError("self-intersecting negative control must be closed with positive signed volume")
    try:
        _reference_shell_proof(crossing_points, crossing_faces)
    except ValueError as exc:
        if "self-intersects" not in str(exc):
            raise
        self_intersection_status = "PASS_REJECTED_EXACT_SELF_INTERSECTION"
    else:
        raise AssertionError("self-intersecting closed PLC shell was not rejected")
    interface = _reference_interface_control()
    fat_points = np.vstack((boundary_points[:3], np.asarray([[0.25, 0.25, -1]], np.float32)))
    fat_tets = M.orient(fat_points, np.asarray([[0, 1, 2, 3]], np.int64))
    fat_faces = _oriented_boundary(fat_tets)
    fat_det, fat_quality = M.measurements(fat_points, fat_tets)
    if np.any(~M.passes(fat_det, fat_quality)) or np.any(fat_det <= 0):
        raise AssertionError("paired fixture Fat reference tetra must start fully passing")
    fat_path = out / "paired-fat-valid-tetra.npz"
    np.savez_compressed(
        fat_path, points_world_m=fat_points, points_world_m_f32=fat_points.copy(),
        tetrahedra=fat_tets, material_indices=np.full(1, 16, np.int32),
        boundary_faces=fat_faces, desired_boundary_faces=fat_faces,
        nonpositive=np.asarray([0], np.int64),
        output_point_parent_offsets=np.zeros(len(fat_points)+1, np.int64),
        output_point_parent_input_cell_rows=np.empty(0, np.int64),
    )
    paired_state_path = out / "paired-role-source-state.json"
    paired_state_path.write_text(json.dumps({
        "Skin_current": {"path": str(input_path), "sha256": sha(input_path),
                         "native_quality_failures": input_bad, "nonpositive": input_nonpositive},
        "Fat_current": {"path": str(fat_path), "sha256": sha(fat_path),
                        "native_quality_failures": 0, "nonpositive": 0},
        "physical_admission": False, "native_steps": 0,
    }, indent=2, sort_keys=True)+"\n")
    paired_state_result = run_state(
        paired_state_path, out / "paired-role-run",
        float(args.fixture_seconds), fixture_args,
    )
    if paired_state_result.get("status") != "PASS_REFERENCE_PLC_RECONSTRUCTION_SOURCE_ONLY":
        raise AssertionError("paired role reference reconstruction did not pass atomically: "+
                             json.dumps(paired_state_result, sort_keys=True)[:3000])
    if paired_state_result.get("physical_admission") is not False or paired_state_result.get("native_steps") != 0:
        raise AssertionError("paired role source-only run reported physical/native admission")
    fixture_report = {
        "schema": "h706.tetgen-reference-plc-quality-fixture.v1",
        "status": "PASS_REFERENCE_PLC_TINY_CONTROLS_SOURCE_ONLY",
        "valid_boundary_inverted_cell": {
            "input": pin(input_path), "native_bad_before": 1,
            "exact_nonpositive_before": 1, "native_bad_after": 0,
            "exact_nonpositive_after": 0, "producer_result": result,
            "independent_readback": readback,
        },
        "self_intersection_negative_control": {
            "status": self_intersection_status,
            "closed_shell": True, "positive_exact_signed_volume": True,
            "physical_admission": False, "native_steps": 0,
        },
        "paired_interface_control": interface,
        "paired_role_run": {
            "status": paired_state_result["status"],
            "skin_status": next(r["status"] for r in paired_state_result["role_results"]
                                if r["role"] == "Skin"),
            "fat_status": next(r["status"] for r in paired_state_result["role_results"]
                               if r["role"] == "Fat"),
            "shared_interface_audit": paired_state_result["paired_interface_audit"],
            "source_state": pin(paired_state_path),
            "output_state": paired_state_result["output_state"],
            "report": pin(out / "paired-role-run" / "report.json"),
            "physical_admission": False, "native_steps": 0,
        },
        "reference_dependency_pins": reference_dependency_pins(),
        "tetgen": pin(TETGEN_PATH), "physical_admission": False, "native_steps": 0,
    }
    (out / "fixture-report.json").write_text(json.dumps(fixture_report, indent=2, sort_keys=True)+"\n")
    return fixture_report


def _reference_interface_control():
    base = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    skin_points = np.vstack((base, np.asarray([[0.25, 0.25, 1]], np.float32)))
    fat_points = np.vstack((base, np.asarray([[0.25, 0.25, -1]], np.float32)))
    skin_tet = M.orient(skin_points, np.asarray([[0, 1, 2, 3]], np.int64))
    fat_tet = M.orient(fat_points, np.asarray([[0, 1, 2, 3]], np.int64))
    skin_faces, fat_faces = _oriented_boundary(skin_tet), _oriented_boundary(fat_tet)
    ok = _reference_shared_interface_audit(
        skin_points, skin_faces, fat_points, fat_faces,
        skin_points.copy(), skin_faces.copy(), fat_points.copy(), fat_faces.copy(),
    )
    bad_fat_faces = fat_faces.copy()
    common = set(_oriented_coordinate_face_map(skin_points, skin_faces)) & set(
        _oriented_coordinate_face_map(fat_points, fat_faces)
    )
    target = next(iter(common))
    for i, face in enumerate(bad_fat_faces):
        if tuple(sorted(_point_bits(fat_points[int(v)]) for v in face)) == target:
            bad_fat_faces[i] = face[[0, 2, 1]]
            break
    try:
        _reference_shared_interface_audit(skin_points, skin_faces, fat_points, bad_fat_faces)
    except ValueError as exc:
        if "orientations are not opposite" not in str(exc):
            raise
        rejection = "PASS_REJECTED_SHARED_COFACE_ORIENTATION_MISMATCH"
    else:
        raise AssertionError("paired-interface control accepted same-direction cofaces")
    return {"status": "PASS_EXACT_SHARED_INTERFACE_CONTROLS",
            "valid_pair": ok, "negative_control": rejection}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--maximum-old-cells", type=int, default=128)
    ap.add_argument("--maximum-plc-points", type=int, default=1024)
    ap.add_argument("--maximum-attempts", type=int, default=4096)
    ap.add_argument("--maximum-operations", type=int, default=512)
    ap.add_argument("--tetgen-timeout", type=float, default=30.0)
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--fixture-seconds", type=float, default=30.0)
    ap.add_argument("--reference-plc", action="store_true",
                    help="explicit source-only exact-boundary reference reconstruction for inverted interior cells")
    args = ap.parse_args()
    if args.maximum_old_cells < 1 or args.maximum_old_cells > MAX_OLD_CELLS_HARD:
        ap.error(f"--maximum-old-cells must be in 1..{MAX_OLD_CELLS_HARD}")
    if args.maximum_plc_points < 4 or args.maximum_plc_points > 8192:
        ap.error("--maximum-plc-points must be in 4..8192")
    if args.maximum_operations < 1 or args.maximum_operations > 2048:
        ap.error("--maximum-operations must be in 1..2048")
    if args.maximum_attempts < 1 or args.maximum_attempts > 20000:
        ap.error("--maximum-attempts must be in 1..20000")
    if args.fixture:
        result = fixture(args.output, args)
    else:
        if args.state is None:
            ap.error("--state is required unless --fixture is selected")
        result = run_state(args.state, args.output, args.seconds, args)
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "report": pin(args.output / ("fixture-report.json" if args.fixture else "report.json")),
        "operations": (
            sum(x.get("operations", 0) for x in result.get("role_results", []))
            if not args.fixture else
            result.get("producer_result", {}).get(
                "operations", result.get("valid_boundary_inverted_cell", {}).get(
                    "producer_result", {}).get("operations", 0)
            )
        ),
        "physical_admission": False, "native_steps": 0,
    }, sort_keys=True))

if __name__ == "__main__":
    main()
