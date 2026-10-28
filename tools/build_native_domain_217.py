from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
D = ROOT / "D/connected-obstacle-207"
P = D / "physical-transfer-209"
PROPOSAL = P / "volume-worker/head-collar/proposal-001/donor-head-collar-O033.npz"
WALL_CSV = P / "volume-worker/head-collar/proposal-001/donor-sidewall-faces.csv"
PARENT_CELLS = P / "volume-worker/baseline-parent-cell-mass.csv.gz"
PARENT_NODES = P / "volume-worker/baseline-compiled-parent-nodes.csv.gz"
DEFAULT_TETGEN = Path(
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/"
    "human-resting-20261005/supported-continuation-20261020/metal-optimization-080/"
    "visual-verification-20261022/wrist-connected-repair-001/arm-local-pocket-remesh-116/"
    "local-remesh-118/tetgen-v1.6.0/tetgen"
)
MARKER = {"skin_outer_O": 1, "skin_inner_I": 2, "skin_wall_OI": 3,
          "fat_outer_I": 4, "fat_inner_D": 5, "fat_wall_ID": 6}
INVALID = -1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _npz_array(z: Any, key: str, fallback: str | None = None) -> np.ndarray:
    if key in z.files:
        return np.asarray(z[key])
    if fallback is not None and fallback in z.files:
        return np.asarray(z[fallback])
    raise ValueError(f"input NPZ is missing required array {key!r}")


def _face_ancestry(z: Any, count: int, fallback_rows: np.ndarray) -> list[list[int]]:
    if "face_ancestry_offsets" in z.files:
        offsets = np.asarray(z["face_ancestry_offsets"], dtype=np.int64)
        key = ("face_ancestry_input_cache_rows"
               if "face_ancestry_input_cache_rows" in z.files else
               "face_ancestry_source_rows")
        if len(offsets) != count + 1 or key not in z.files:
            raise ValueError("face ancestry CSR is incomplete")
        rows = np.asarray(z[key], dtype=np.int64)
        if offsets[0] != 0 or offsets[-1] != len(rows) or np.any(np.diff(offsets) < 0):
            raise ValueError("face ancestry CSR offsets are invalid")
        return [rows[offsets[i]:offsets[i + 1]].tolist() for i in range(count)]
    return [[int(row)] if int(row) >= 0 else [] for row in fallback_rows]


def _optional_face_array(z: Any, key: str, count: int, dtype: Any, fill: int) -> np.ndarray:
    if key not in z.files:
        return np.full(count, fill, dtype=dtype)
    value = np.asarray(z[key], dtype=dtype)
    if value.shape != (count,):
        raise ValueError(f"per-face array {key!r} has the wrong row count")
    return value


def load_surface(path: Path, layer: str, collar: Any | None = None) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    points = np.asarray(_npz_array(z, "points_world_m"), dtype=np.float32)
    if "face_source_identity_schema_by_original_id_json" in z.files:
        # Reuse the current typed authoring contract. These are source-face IDs
        # within pinned source namespaces, never NHSKIN vertex/face indices.
        from export_authoring_registered_surface_sidecars_230 import validate_candidate
        validate_candidate(path, layer, require_f32=True)
        faces = np.asarray(z["faces"], dtype=np.int64)
        rows = np.arange(len(faces), dtype=np.int64)
        exact = np.full(len(faces), INVALID, dtype=np.int64)
        original = np.asarray(z["face_source_original_id"], dtype=np.int64)
        face_ids = np.asarray(z["face_source_face_id"], dtype=np.int64)
        layers = np.asarray(z["face_source_layer"], dtype=np.int8)
        bodies = np.asarray(z["face_source_body"], dtype=np.int32)
        bones = np.asarray(z["face_source_bone"], dtype=np.int32)
        run_ids = original.copy()
        ancestry = [[int(v)] for v in face_ids]
        ancestry_ids = [[int(v)] for v in original]
        ancestry_parents = [[int(v)] for v in rows]
        ancestry_kind = np.full(len(faces), 7, dtype=np.int8)
    elif layer == "O":
        faces = np.asarray(_npz_array(z, "source_O_faces", "faces"), dtype=np.int64)
        rows = np.asarray(_npz_array(z, "envelope_source_face_row", "face_source_row"),
                          dtype=np.int64)
        original = _optional_face_array(z, "envelope_source_original_id", len(faces),
                                        np.int64, INVALID)
        if collar is None or "exact_candidate_face_source_rows" not in collar.files:
            raise ValueError("O candidate needs exact-candidate face ancestry from the collar packet")
        exact = np.asarray(collar["exact_candidate_face_source_rows"], dtype=np.int64)
        if len(exact) != len(faces):
            raise ValueError("O exact-source face ancestry row count differs from candidate faces")
        layers = np.full(len(faces), INVALID, np.int8)
        bodies = np.full(len(faces), INVALID, np.int32)
        bones = np.full(len(faces), INVALID, np.int32)
        run_ids = np.full(len(faces), INVALID, np.int64)
        face_ids = original.copy()
        ancestry = [[int(v)] if v >= 0 else [] for v in exact]
        ancestry_ids = [[int(original[i])] * len(v) for i,v in enumerate(ancestry)]
        ancestry_parents = [[INVALID] * len(v) for v in ancestry]
        ancestry_kind = np.where(exact >= 0, 1, 2).astype(np.int8)
        if "face_ancestry_offsets" in z.files:
            offsets=np.asarray(z["face_ancestry_offsets"],np.int64)
            keys=("face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows")
            if any(k not in z.files for k in keys):raise ValueError("conditioned O lost typed multi-source ancestry")
            arrays=[np.asarray(z[k],np.int64) for k in keys]
            if offsets.shape!=(len(faces)+1,) or offsets[0]!=0 or np.any(np.diff(offsets)<=0) or any(len(a)!=offsets[-1] or np.any(a<0) for a in arrays):raise ValueError("conditioned O typed ancestry CSR is invalid")
            ancestry,ancestry_ids,ancestry_parents=[[a[offsets[i]:offsets[i+1]].tolist() for i in range(len(faces))] for a in arrays]
            # Kind 6 retains complete conditioned O ancestry; no single parent
            # is inferred from a contraction/flip's union of source faces.
            ancestry_kind=np.full(len(faces),6,np.int8)
    else:
        faces = np.asarray(_npz_array(z, "faces"), dtype=np.int64)
        rows = np.asarray(_npz_array(z, "face_cache_row", "face_source_row"),
                          dtype=np.int64)
        original = np.full(len(faces), INVALID, np.int64)
        exact = _optional_face_array(z, "face_exact_source_row", len(faces),
                                     np.int64, INVALID)
        layers = _optional_face_array(z, "face_source_layer", len(faces),
                                      np.int8, INVALID)
        bodies = _optional_face_array(z, "face_source_body", len(faces),
                                      np.int32, INVALID)
        bones = _optional_face_array(z, "face_source_bone", len(faces),
                                     np.int32, INVALID)
        run_ids = _optional_face_array(z, "face_run_original_id", len(faces),
                                       np.int64, INVALID)
        face_ids = _optional_face_array(z, "face_id", len(faces),
                                        np.int64, INVALID)
        ancestry = _face_ancestry(z, len(faces), rows)
        ancestry_ids = [[int(original[i])] * len(v) for i,v in enumerate(ancestry)]
        ancestry_parents = [[INVALID] * len(v) for v in ancestry]
        ancestry_kind = np.where(np.fromiter((bool(v) for v in ancestry), bool),
                                 3, 0).astype(np.int8)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError(f"{layer} points must be finite F32 xyz rows")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) != len(rows):
        raise ValueError(f"{layer} faces/source rows have inconsistent shapes")
    if len(faces) == 0 or np.min(faces) < 0 or np.max(faces) >= len(points):
        raise ValueError(f"{layer} face index is outside its point domain")
    if np.any(faces[:, 0] == faces[:, 1]) or np.any(faces[:, 1] == faces[:, 2]) or np.any(faces[:, 2] == faces[:, 0]):
        raise ValueError(f"{layer} has repeated triangle point IDs")
    tri = points[faces]
    if np.any(np.all(tri[:, 0] == tri[:, 1], axis=1) |
              np.all(tri[:, 1] == tri[:, 2], axis=1) |
              np.all(tri[:, 2] == tri[:, 0], axis=1)):
        raise ValueError(f"{layer} has Float32-collapsed triangles")
    cross = np.cross(tri[:,1].astype(np.float64)-tri[:,0],
                     tri[:,2].astype(np.float64)-tri[:,0])
    if np.any(np.all(cross == 0.0, axis=1)):
        raise ValueError(f"{layer} has zero-area emitted triangles")
    if len(exact) != len(faces) or len(original) != len(faces):
        raise ValueError(f"{layer} source ancestry row counts are incomplete")
    return {"path": str(path), "points": points, "faces": faces,
            "source_rows": rows, "exact_source_rows": exact,
            "source_original_ids": original, "source_layers": layers,
            "source_bodies": bodies, "source_bones": bones,
            "source_run_original_ids": run_ids, "source_face_ids": face_ids,
            "ancestry": ancestry, "ancestry_original_ids":ancestry_ids,
            "ancestry_parent_rows":ancestry_parents,"ancestry_kind": ancestry_kind,
            "raw_npz_keys": list(z.files)}


def _edge_directions(faces: np.ndarray) -> tuple[dict[tuple[int, int], list[tuple[int, int]]], dict[tuple[int, int], int]]:
    directed: dict[tuple[int, int], list[tuple[int, int]]] = {}
    counts: dict[tuple[int, int], int] = {}
    for tri in np.asarray(faces, dtype=np.int64):
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            directed.setdefault(key, []).append((u, v))
            counts[key] = counts.get(key, 0) + 1
    return directed, counts


def edge_report(faces: np.ndarray) -> dict[str, Any]:
    directed, counts = _edge_directions(faces)
    histogram: dict[str, int] = {}
    for count in counts.values():
        histogram[str(count)] = histogram.get(str(count), 0) + 1
    same = sum(1 for key, rows in directed.items()
               if len(rows) == 2 and rows[0] == rows[1])
    return {"edges": len(counts), "incidence_histogram": histogram,
            "boundary_edges": sum(v == 1 for v in counts.values()),
            "nonmanifold_edges": sum(v > 2 for v in counts.values()),
            "same_direction_twoface_edges": same}


def _orient_surface_to_wall(surface_faces: np.ndarray, wall_faces: np.ndarray,
                            name: str) -> tuple[np.ndarray, bool]:
    surface_directed, surface_counts = _edge_directions(surface_faces)
    wall_directed, wall_counts = _edge_directions(wall_faces)
    if any(v > 2 for v in surface_counts.values()) or any(v > 2 for v in wall_counts.values()):
        raise ValueError(f"{name} surface/wall has nonmanifold edges")
    surface_boundary = {e for e, n in surface_counts.items() if n == 1}
    wall_boundary = {e for e, n in wall_counts.items() if n == 1}
    # The annular wall has two boundary loops.  This source surface must match
    # exactly one of them; the adjacent I or D surface consumes the other loop.
    if not surface_boundary or not surface_boundary.issubset(wall_boundary):
        raise ValueError(f"{name} surface boundary is not a loop of its donor annulus: "
                         f"surface={len(surface_boundary)} wall={len(wall_boundary)}")
    same = 0
    opposite = 0
    for edge in surface_boundary:
        a = surface_directed[edge]
        b = wall_directed.get(edge, [])
        if len(a) != 1 or len(b) != 1:
            raise ValueError(f"{name} seam edge has an invalid directed incidence")
        if a[0] == b[0]:
            same += 1
        elif a[0] == b[0][::-1]:
            opposite += 1
        else:
            raise ValueError(f"{name} seam edge direction is inconsistent")
    if same == len(surface_boundary):
        return np.asarray(surface_faces, dtype=np.int64)[:, [0, 2, 1]], True
    if opposite == len(surface_boundary):
        return np.asarray(surface_faces, dtype=np.int64).copy(), False
    raise ValueError(f"{name} seam requires inconsistent per-edge surface flips")


def _closed_shell_orientation(points: np.ndarray, faces: np.ndarray, name: str) -> tuple[np.ndarray, float, bool]:
    report = edge_report(faces)
    if report["boundary_edges"] or report["nonmanifold_edges"] or report["same_direction_twoface_edges"]:
        raise ValueError(f"{name} shell incidence is not a closed oriented two-manifold: {report}")
    p = points.astype(np.float64)
    f = np.asarray(faces, dtype=np.int64)
    volume = float(np.einsum("ij,ij->", p[f[:, 0]],
                             np.cross(p[f[:, 1]], p[f[:, 2]])) / 6.0)
    if not math.isfinite(volume) or volume == 0.0:
        raise ValueError(f"{name} shell has nonpositive or zero signed volume")
    flipped = volume < 0.0
    if flipped:
        f = f[:, [0, 2, 1]]
        volume = -volume
    return f, volume, flipped


def _coordinate_key(point: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(v) for v in np.asarray(point, dtype=np.float32).view(np.uint32))


def _unique_coordinate_index(points: np.ndarray, wanted: np.ndarray, layer: str) -> int:
    key = _coordinate_key(wanted)
    matches = [i for i, point in enumerate(points) if _coordinate_key(point) == key]
    if len(matches) != 1:
        raise ValueError(f"{layer} collar point has {len(matches)} exact-F32 matches in candidate surface")
    return matches[0]


def _make_face_records(surface: dict[str, Any], marker: int, role: str) -> list[dict[str, Any]]:
    rows = []
    for i in range(len(surface["faces"])):
        rows.append({
            "marker": marker, "role": role,
            "source_row": int(surface["source_rows"][i]),
            "exact_source_row": int(surface["exact_source_rows"][i]),
            "source_original_id": int(surface["source_original_ids"][i]),
            "source_layer": int(surface["source_layers"][i]),
            "source_body": int(surface["source_bodies"][i]),
            "source_bone": int(surface["source_bones"][i]),
            "run_original_id": int(surface["source_run_original_ids"][i]),
            "source_face_id": int(surface["source_face_ids"][i]),
            "ancestry": list(map(int, surface["ancestry"][i])),
            "ancestry_original_ids":list(map(int,surface["ancestry_original_ids"][i])),
            "ancestry_parent_rows":list(map(int,surface["ancestry_parent_rows"][i])),
            "ancestry_kind": int(surface["ancestry_kind"][i]),
            "parent_tet_row": INVALID, "parent_material_index": INVALID,
            "parent_source_nodes": [INVALID, INVALID, INVALID],
        })
    return rows


def _wall_records(parent_rows: np.ndarray, parent_nodes: np.ndarray,
                  parent_materials: np.ndarray, start: int,
                  role: str, marker: int) -> list[dict[str, Any]]:
    out = []
    for local, (tet, nodes, material) in enumerate(zip(parent_rows, parent_nodes, parent_materials)):
        out.append({
            "marker": marker, "role": role, "source_row": local + start,
            "exact_source_row": INVALID, "source_original_id": INVALID,
            "source_layer": INVALID, "source_body": INVALID, "source_bone": INVALID,
            "run_original_id": INVALID, "source_face_id": INVALID,
            "ancestry": [], "ancestry_original_ids":[],"ancestry_parent_rows":[],"ancestry_kind": 5,
            "parent_tet_row": int(tet), "parent_material_index": int(material),
            "parent_source_nodes": list(map(int, nodes)),
        })
    return out


def _merge_face_records(parts: list[list[dict[str, Any]]]) -> dict[str, np.ndarray]:
    records = list(itertools.chain.from_iterable(parts))
    offsets = [0]
    ancestry_rows: list[int] = []
    ancestry_ids: list[int] = [];ancestry_parents: list[int] = []
    for row in records:
        if len(row["ancestry"])!=len(row["ancestry_original_ids"]) or len(row["ancestry"])!=len(row["ancestry_parent_rows"]):raise ValueError("typed face ancestry rows are incomplete")
        ancestry_rows.extend(row["ancestry"])
        ancestry_ids.extend(row["ancestry_original_ids"]);ancestry_parents.extend(row["ancestry_parent_rows"])
        offsets.append(len(ancestry_rows))
    keys = ("marker", "source_row", "exact_source_row", "source_original_id",
            "source_layer", "source_body", "source_bone", "run_original_id",
            "source_face_id", "ancestry_kind", "parent_tet_row",
            "parent_material_index")
    merged = {f"face_{key}": np.asarray([row[key] for row in records], dtype=np.int64)
              for key in keys}
    merged["face_source_role"] = np.asarray([row["role"] for row in records], dtype="U24")
    merged["face_parent_source_node_rows"] = np.asarray(
        [row["parent_source_nodes"] for row in records], dtype=np.int64)
    merged["face_ancestry_offsets"] = np.asarray(offsets, dtype=np.int64)
    merged["face_ancestry_source_rows"] = np.asarray(ancestry_rows, dtype=np.int64)
    merged["face_ancestry_source_original_ids"]=np.asarray(ancestry_ids,np.int64)
    merged["face_ancestry_parent_face_rows"]=np.asarray(ancestry_parents,np.int64)
    return merged


def place_verified_donor_collar(collar: Any, placement: Path) -> dict[str,np.ndarray]:
    """Apply the already-declared native source registration to donor annuli."""
    packet = np.load(placement,allow_pickle=False)
    rotation = np.asarray(packet["native_placement_rotation_row_major"],np.float64)
    translation = np.asarray(packet["native_placement_translation_m"],np.float64)
    if rotation.shape!=(9,) or translation.shape!=(3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("donor collar native placement is not a finite 3D transform")
    rotation = rotation.reshape(3,3)
    if np.max(np.abs(rotation.T@rotation-np.eye(3)))>1e-6 or abs(np.linalg.det(rotation)-1.0)>1e-6:
        raise ValueError("donor collar native placement must be a proper rigid transform")
    result = {k: np.array(collar[k],copy=True) for k in collar.files}
    result["points_world_m"] = (np.asarray(result["points_world_m"],np.float64)@rotation.T+translation).astype(np.float32)
    return result


def rebind_typed_collar_outer(collar: Any, outer: dict[str, Any]) -> dict[str, np.ndarray]:
    """Keep the verified donor annuli, remapping only their exact outer rows.

    Call read_wall_provenance on the original packet first. The returned packet
    uses the current O domain; I/D donor ring coordinates and parent IDs stay exact.
    """
    old = {k: np.array(collar[k], copy=True) for k in (collar.files if hasattr(collar,"files") else collar)}
    ni, nd = len(old["donor_I_parent_node_rows"]), len(old["donor_D_parent_node_rows"])
    old_no = len(old["points_world_m"]) - ni - nd
    old_outer_rows = np.asarray(old["donor_O_to_candidate_vertex"], dtype=np.int64)
    if ni != nd or len(old_outer_rows) != ni:
        raise ValueError("typed collar donor ring counts differ")
    old_points = np.asarray(old["points_world_m"], dtype=np.float32)
    points = np.asarray(outer["points"], dtype=np.float32)
    new_no = len(points)
    new_outer_rows = np.asarray([
        _unique_coordinate_index(points, old_points[row], "O")
        for row in old_outer_rows], dtype=np.int64)
    if len(set(map(int,new_outer_rows))) != ni:
        raise ValueError("typed collar outer ring reuses a candidate row")
    remap = dict(zip(map(int,old_outer_rows),map(int,new_outer_rows)))
    for row in range(ni+nd):
        remap[old_no+row] = new_no+row
    for key in ("skin_sidewall_faces","fat_sidewall_faces"):
        source = np.asarray(old[key], dtype=np.int64)
        if any(int(v) not in remap for v in source.flat):
            raise ValueError("typed collar wall references a non-ring vertex")
        old[key] = np.asarray([[remap[int(v)] for v in face] for face in source],
                              dtype=np.int64)
    old["points_world_m"] = np.vstack((points,old_points[old_no:])).astype(np.float32)
    old["outer_faces"] = np.asarray(outer["faces"],dtype=np.int64).copy()
    old["donor_O_to_candidate_vertex"] = new_outer_rows
    old["exact_candidate_face_source_rows"] = np.full(len(outer["faces"]),INVALID,np.int64)
    return old


def _map_collar_faces(outer: dict[str, Any], inner: dict[str, Any],
                      deep: dict[str, Any], collar: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    collar_points = np.asarray(collar["points_world_m"], dtype=np.float32)
    outer_count = len(outer["points"])
    inner_ring_count = len(collar["donor_I_parent_node_rows"])
    deep_ring_count = len(collar["donor_D_parent_node_rows"])
    if len(collar_points) != outer_count + inner_ring_count + deep_ring_count:
        raise ValueError("collar point packet is not outer points followed by exact I/D donor rings")
    if not np.array_equal(np.asarray(collar["outer_faces"], dtype=np.int64), outer["faces"]):
        raise ValueError("collar outer faces differ from supplied O surface")
    if not np.array_equal(collar_points[:outer_count], outer["points"]):
        raise ValueError("collar O rows are not bitwise the supplied Float32 O domain")
    if inner_ring_count != len(collar["donor_O_parent_node_rows"]) or deep_ring_count != inner_ring_count:
        raise ValueError("collar O/I/D ring ancestry counts differ")
    inner_ids = [_unique_coordinate_index(inner["points"], p, "I")
                 for p in collar_points[outer_count:outer_count + inner_ring_count]]
    deep_ids = [_unique_coordinate_index(deep["points"], p, "D")
                for p in collar_points[outer_count + inner_ring_count:]]
    if len(set(inner_ids)) != inner_ring_count or len(set(deep_ids)) != deep_ring_count:
        raise ValueError("collar ring coordinate map repeats a candidate surface point")
    skin_source = np.asarray(collar["skin_sidewall_faces"], dtype=np.int64)
    fat_source = np.asarray(collar["fat_sidewall_faces"], dtype=np.int64)
    if np.min(skin_source) < 0 or np.max(skin_source) >= len(collar_points) or \
       np.min(fat_source) < 0 or np.max(fat_source) >= len(collar_points):
        raise ValueError("collar sidewall face references an invalid collar point")
    if not set(map(int, np.unique(skin_source[skin_source < outer_count]))) <= \
       set(map(int, collar["donor_O_to_candidate_vertex"])):
        raise ValueError("skin collar uses O vertices outside its recorded donor-O correspondence")
    if set(map(int, np.unique(skin_source[skin_source >= outer_count]))) != \
       set(range(outer_count, outer_count + inner_ring_count)):
        raise ValueError("skin collar does not use its complete exact I ring")
    if set(map(int, np.unique(fat_source[fat_source < outer_count + inner_ring_count]))) - \
       set(range(outer_count, outer_count + inner_ring_count)):
        raise ValueError("fat collar references an unexpected O point")
    if set(map(int, np.unique(fat_source[fat_source >= outer_count + inner_ring_count]))) != \
       set(range(outer_count + inner_ring_count, len(collar_points))):
        raise ValueError("fat collar does not use its complete exact D ring")
    skin = np.empty_like(skin_source)
    for i, tri in enumerate(skin_source):
        for j, node in enumerate(tri):
            v = int(node)
            skin[i, j] = v if v < outer_count else outer_count + inner_ids[v - outer_count]
    fat = np.empty_like(fat_source)
    for i, tri in enumerate(fat_source):
        for j, node in enumerate(tri):
            v = int(node)
            if outer_count <= v < outer_count + inner_ring_count:
                fat[i, j] = inner_ids[v - outer_count]
            elif v >= outer_count + inner_ring_count:
                fat[i, j] = len(inner["points"]) + deep_ids[v - outer_count - inner_ring_count]
            else:
                raise ValueError("fat I-D wall references an O vertex")
    return skin, fat, np.asarray(inner_ids, dtype=np.int64)


def assemble_shells(outer: dict[str, Any], inner: dict[str, Any],
                    deep: dict[str, Any], collar: Any,
                    wall_rows: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    skin_wall, fat_wall, inner_ring_ids = _map_collar_faces(outer, inner, deep, collar)
    no, ni, nd = len(outer["points"]), len(inner["points"]), len(deep["points"])
    outer_faces = np.asarray(outer["faces"], dtype=np.int64).copy()
    inner_skin_faces = np.asarray(inner["faces"], dtype=np.int64) + no
    inner_fat_faces = np.asarray(inner["faces"], dtype=np.int64).copy()
    deep_faces = np.asarray(deep["faces"], dtype=np.int64) + ni
    # Determine winding from exact seam edge incidence with the preserved donor wall.
    outer_faces, outer_flip = _orient_surface_to_wall(outer_faces, skin_wall, "O/skin-wall")
    inner_skin_faces, inner_skin_flip = _orient_surface_to_wall(inner_skin_faces, skin_wall, "I/skin-wall")
    inner_fat_faces, inner_fat_flip = _orient_surface_to_wall(inner_fat_faces, fat_wall, "I/fat-wall")
    deep_faces, deep_flip = _orient_surface_to_wall(deep_faces, fat_wall, "D/fat-wall")
    skin_points = np.vstack((outer["points"], inner["points"])).astype(np.float32)
    fat_points = np.vstack((inner["points"], deep["points"])).astype(np.float32)
    skin_faces = np.vstack((outer_faces, inner_skin_faces, skin_wall)).astype(np.int64)
    fat_faces = np.vstack((inner_fat_faces, deep_faces, fat_wall)).astype(np.int64)
    skin_faces, skin_volume, skin_global_flip = _closed_shell_orientation(skin_points, skin_faces, "skin")
    fat_faces, fat_volume, fat_global_flip = _closed_shell_orientation(fat_points, fat_faces, "fat")
    skin_records = _make_face_records(outer, MARKER["skin_outer_O"], "outer_O")
    skin_records += _make_face_records(inner, MARKER["skin_inner_I"], "inner_I")
    skin_records += _wall_records(wall_rows["skin_parent_tets"], wall_rows["skin_parent_nodes"],
                                  wall_rows["skin_materials"], 0, "donor_skin_OI_headwall",
                                  MARKER["skin_wall_OI"])
    fat_records = _make_face_records(inner, MARKER["fat_outer_I"], "outer_I")
    fat_records += _make_face_records(deep, MARKER["fat_inner_D"], "inner_D")
    fat_records += _wall_records(wall_rows["fat_parent_tets"], wall_rows["fat_parent_nodes"],
                                 wall_rows["fat_materials"], 90, "donor_fat_ID_headwall",
                                 MARKER["fat_wall_ID"])
    registered_i_offsets = [0]
    registered_i_ancestry_rows: list[int] = []
    for source_rows in inner["ancestry"]:
        registered_i_ancestry_rows.extend(map(int, source_rows))
        registered_i_offsets.append(len(registered_i_ancestry_rows))
    registered_i_source = {
        "registered_I_source_vertex_rows": np.arange(ni, dtype=np.int64),
        "registered_I_source_faces": np.asarray(inner["faces"], dtype=np.int64),
        "registered_I_source_face_rows": np.asarray(inner["source_rows"], dtype=np.int64),
        "registered_I_source_exact_face_rows": np.asarray(inner["exact_source_rows"], dtype=np.int64),
        "registered_I_source_face_ids": np.asarray(inner["source_face_ids"], dtype=np.int64),
        "registered_I_source_ancestry_offsets": np.asarray(registered_i_offsets, dtype=np.int64),
        "registered_I_source_ancestry_rows": np.asarray(registered_i_ancestry_rows, dtype=np.int64),
    }
    registered_o = {
        "registered_O_source_vertex_rows": np.arange(no, dtype=np.int64),
        "registered_O_shell_node_rows": np.arange(no, dtype=np.int64),
        "registered_O_source_faces": np.asarray(outer["faces"], dtype=np.int64),
        "registered_O_shell_face_rows": np.arange(len(outer["faces"]), dtype=np.int64),
    }
    registered_d = {
        "registered_D_source_vertex_rows": np.arange(nd, dtype=np.int64),
        "registered_D_shell_node_rows": np.arange(ni, ni + nd, dtype=np.int64),
        "registered_D_source_faces": np.asarray(deep["faces"], dtype=np.int64),
        "registered_D_shell_face_rows": np.arange(len(inner_fat_faces),
                                                   len(inner_fat_faces) + len(deep_faces),
                                                   dtype=np.int64),
    }
    registered_i_skin = {
        **registered_i_source,
        "registered_I_shell_node_rows": np.arange(no, no + ni, dtype=np.int64),
        "registered_I_shell_faces": inner_skin_faces.copy(),
        "registered_I_shell_face_rows": np.arange(len(outer_faces),
            len(outer_faces) + len(inner_skin_faces), dtype=np.int64),
    }
    registered_i_fat = {
        **registered_i_source,
        "registered_I_shell_node_rows": np.arange(ni, dtype=np.int64),
        "registered_I_shell_faces": inner_fat_faces.copy(),
        "registered_I_shell_face_rows": np.arange(len(inner_fat_faces), dtype=np.int64),
    }
    skin = {"points_world_m": skin_points, "faces": skin_faces,
            "face_marker": np.asarray([r["marker"] for r in skin_records], dtype=np.int32),
            "face_ancestry": _merge_face_records([skin_records]),
            **registered_o, **registered_i_skin,
            "shell_signed_volume_m3": skin_volume}
    fat = {"points_world_m": fat_points, "faces": fat_faces,
           "face_marker": np.asarray([r["marker"] for r in fat_records], dtype=np.int32),
           "face_ancestry": _merge_face_records([fat_records]),
           **registered_i_fat, **registered_d,
           "shell_signed_volume_m3": fat_volume}
    if len(skin["faces"]) != len(skin["face_marker"]) or len(fat["faces"]) != len(fat["face_marker"]):
        raise ValueError("shell marker and ancestry face counts differ")
    report = {
        "marker_roles": MARKER,
        "wall_donor_I_ring_local_rows": inner_ring_ids.tolist(),
        "orientation": {"O_flipped_to_wall": outer_flip, "I_skin_flipped_to_wall": inner_skin_flip,
                        "I_fat_flipped_to_wall": inner_fat_flip, "D_flipped_to_wall": deep_flip,
                        "skin_global_positive_volume_flip": skin_global_flip,
                        "fat_global_positive_volume_flip": fat_global_flip},
        "skin_shell": {"vertices": len(skin_points), "faces": len(skin_faces),
                       "signed_volume_m3": skin_volume, "edge_report": edge_report(skin_faces)},
        "fat_shell": {"vertices": len(fat_points), "faces": len(fat_faces),
                      "signed_volume_m3": fat_volume, "edge_report": edge_report(fat_faces)},
    }
    return skin, fat, report


def read_wall_provenance(collar: Any, wall_csv: Path, parent_cells: Path,
                         parent_nodes_path: Path) -> dict[str, np.ndarray]:
    records = list(csv.DictReader(wall_csv.open(newline="")))
    expected = 180
    if len(records) != expected or len(collar["skin_sidewall_faces"]) != 90 or len(collar["fat_sidewall_faces"]) != 90:
        raise ValueError("proposal001 donor wall CSV/NPZ must have 90 skin and 90 fat wall rows")
    if not records or not {"wall_face_row", "role", "material_index",
            "parent_tet_row", "parent_node0", "parent_node1", "parent_node2",
            "output_node0", "output_node1", "output_node2"}.issubset(set(records[0])):
        raise ValueError("proposal001 donor wall ancestry CSV has the wrong existing columns")
    cells: dict[int, dict[str, str]] = {}
    with gzip.open(parent_cells, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            cells[int(row["parent_tet_row"])] = row
    parent_node_xyz: dict[int, np.ndarray] = {}
    with gzip.open(parent_nodes_path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            parent_node_xyz[int(row["parent_node_index"])] = np.asarray(
                [float(row[f"current_{axis}_m"]) for axis in "xyz"], dtype=np.float64)
    if not parent_node_xyz:
        raise ValueError("D37 positive parent node table is empty")
    # Keep exact donor I/D ring identity separate from the modified O envelope;
    # O ancestry is validated through proposal001's explicit vertex mapping.
    points = np.asarray(collar["points_world_m"], dtype=np.float32)
    no = len(points) - len(collar["donor_I_parent_node_rows"]) - len(collar["donor_D_parent_node_rows"])
    for key, rows, offset in (("I", collar["donor_I_parent_node_rows"], no),
                              ("D", collar["donor_D_parent_node_rows"], no + len(collar["donor_I_parent_node_rows"]))):
        for i, node in enumerate(np.asarray(rows, dtype=np.int64)):
            if int(node) not in parent_node_xyz:
                raise ValueError(f"proposal001 {key} donor ring names missing parent node {int(node)}")
            expected = np.asarray(parent_node_xyz[int(node)], dtype=np.float32)
            if not np.array_equal(points[offset + i], expected):
                raise ValueError(f"proposal001 {key} donor ring coordinate differs from D37 parent node {int(node)}")
    o_map = {int(v): int(n) for v, n in zip(collar["donor_O_to_candidate_vertex"],
                                             collar["donor_O_parent_node_rows"])}
    i_map = {no + i: int(n) for i, n in enumerate(collar["donor_I_parent_node_rows"])}
    d_map = {no + len(collar["donor_I_parent_node_rows"]) + i: int(n)
             for i, n in enumerate(collar["donor_D_parent_node_rows"])}
    node_map = {**o_map, **i_map, **d_map}
    source_arrays: dict[str, list[Any]] = {
        "skin_parent_tets": [], "skin_parent_nodes": [], "skin_materials": [],
        "fat_parent_tets": [], "fat_parent_nodes": [], "fat_materials": [],
    }
    for i, row in enumerate(records):
        local = i if i < 90 else i - 90
        role = "skin_O_to_I" if i < 90 else "fat_I_to_D"
        material = 15 if i < 90 else 16
        wall_key = "skin_sidewall_faces" if i < 90 else "fat_sidewall_faces"
        parent_key = "skin_sidewall_parent_tet_rows" if i < 90 else "fat_sidewall_parent_tet_rows"
        prefix = "skin" if i < 90 else "fat"
        parent_tet = int(row["parent_tet_row"])
        parent_nodes = [int(row[f"parent_node{k}"]) for k in range(3)]
        output_nodes = [int(row[f"output_node{k}"]) for k in range(3)]
        if int(row["wall_face_row"]) != i or row["role"] != role or int(row["material_index"]) != material:
            raise ValueError(f"proposal001 wall CSV role/material/index mismatch at row {i}")
        if output_nodes != list(map(int, collar[wall_key][local])) or int(collar[parent_key][local]) != parent_tet:
            raise ValueError(f"proposal001 wall NPZ/CSV ancestry mismatch at row {i}")
        cell = cells.get(parent_tet)
        if cell is None or int(cell["material_index"]) != material or \
           float(cell["rest_measure_volume_m3"]) <= 0.0:
            raise ValueError(f"proposal001 wall parent tet is absent, nonpositive, or wrong material at row {i}")
        cell_nodes = {int(cell[f"parent_node{k}"]) for k in range(4)}
        if len(set(parent_nodes)) != 3 or not set(parent_nodes).issubset(cell_nodes):
            raise ValueError(f"proposal001 wall parent-node triple is not a face of tet {parent_tet}")
        if any(node not in parent_node_xyz for node in parent_nodes):
            raise ValueError(f"proposal001 wall names a missing D37 parent source node at row {i}")
        if set(parent_nodes) != {node_map.get(node, INVALID) for node in output_nodes}:
            raise ValueError(f"proposal001 wall output/source node ancestry differs at row {i}")
        source_arrays[f"{prefix}_parent_tets"].append(parent_tet)
        source_arrays[f"{prefix}_parent_nodes"].append(parent_nodes)
        source_arrays[f"{prefix}_materials"].append(material)
    return {key: np.asarray(value, dtype=np.int64) for key, value in source_arrays.items()}


def write_shell_npz(path: Path, shell: dict[str, Any], label: str) -> None:
    values = {"points_world_m": np.asarray(shell["points_world_m"], dtype=np.float32),
              "faces": np.asarray(shell["faces"], dtype=np.int64),
              "face_marker": np.asarray(shell["face_marker"], dtype=np.int32),
              "face_source_role": shell["face_ancestry"]["face_source_role"],
              "face_ancestry_offsets": shell["face_ancestry"]["face_ancestry_offsets"],
              "face_ancestry_source_rows": shell["face_ancestry"]["face_ancestry_source_rows"],
              "emission_json": np.asarray([json.dumps({
                  "classification": "surface adapter output; physical admission false",
                  "layer": label,
                  "shell_signed_volume_m3": shell["shell_signed_volume_m3"],
              }, separators=(",", ":"))])}
    values.update({k: v for k, v in shell["face_ancestry"].items()
                   if k not in ("face_source_role", "face_ancestry_offsets", "face_ancestry_source_rows")})
    required_i_map_keys = (
        "registered_I_source_vertex_rows", "registered_I_shell_node_rows",
        "registered_I_source_faces", "registered_I_shell_faces",
        "registered_I_shell_face_rows", "registered_I_source_face_rows",
        "registered_I_source_exact_face_rows", "registered_I_source_face_ids",
        "registered_I_source_ancestry_offsets", "registered_I_source_ancestry_rows")
    if any(k not in shell for k in required_i_map_keys):
        raise ValueError("shell lost its exact registered-I node/face map")
    values.update({k: np.asarray(shell[k]) for k in required_i_map_keys})
    optional_layer_map_keys = (
        "registered_O_source_vertex_rows", "registered_O_shell_node_rows",
        "registered_O_source_faces", "registered_O_shell_face_rows",
        "registered_D_source_vertex_rows", "registered_D_shell_node_rows",
        "registered_D_source_faces", "registered_D_shell_face_rows")
    values.update({k: np.asarray(shell[k]) for k in optional_layer_map_keys if k in shell})
    np.savez_compressed(path, **values)


def write_face_ancestry_csv(path: Path, shell: dict[str, Any]) -> None:
    fields = ["face_row", "face_marker", "face_source_role", "face_source_row",
              "face_exact_source_row", "face_source_original_id", "face_source_layer",
              "face_source_body", "face_source_bone", "face_run_original_id",
              "face_source_face_id", "face_ancestry_kind", "face_parent_tet_row",
              "face_parent_material_index", "parent_source_node0", "parent_source_node1",
              "parent_source_node2"]
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        a = shell["face_ancestry"]
        for i in range(len(shell["faces"])):
            writer.writerow([i, int(shell["face_marker"][i]),
                str(a["face_source_role"][i]), int(a["face_source_row"][i]),
                int(a["face_exact_source_row"][i]), int(a["face_source_original_id"][i]),
                int(a["face_source_layer"][i]), int(a["face_source_body"][i]),
                int(a["face_source_bone"][i]), int(a["face_run_original_id"][i]),
                int(a["face_source_face_id"][i]), int(a["face_ancestry_kind"][i]),
                int(a["face_parent_tet_row"][i]), int(a["face_parent_material_index"][i]),
                *map(int, a["face_parent_source_node_rows"][i])])


def validate_poly_sections(path: Path, points_count: int, faces_count: int,
                           hole_count: int, region_count: int) -> None:
    rows = [line.split() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    index = 0
    def take() -> list[str]:
        nonlocal index
        if index >= len(rows):
            raise ValueError("truncated TetGen .poly section")
        out = rows[index]
        index += 1
        return out
    header = take()
    if len(header) < 4 or int(header[0]) != points_count or int(header[1]) != 3:
        raise ValueError("TetGen .poly point header mismatch")
    for _ in range(points_count): take()
    facet = take()
    if int(facet[0]) != faces_count or int(facet[1]) != 1:
        raise ValueError("TetGen .poly facet header mismatch")
    for _ in range(faces_count):
        part = take()
        if int(part[0]) != 1: raise ValueError("expected one polygon per shell facet")
        tri = take()
        if int(tri[0]) != 3: raise ValueError("expected triangular PLC facets")
        if any(int(v) < 1 or int(v) > points_count for v in tri[1:4]):
            raise ValueError("TetGen .poly face references invalid point")
    if int(take()[0]) != hole_count: raise ValueError("TetGen .poly hole count mismatch")
    for _ in range(hole_count): take()
    if int(take()[0]) != region_count: raise ValueError("TetGen .poly region count mismatch")
    for _ in range(region_count): take()
    if index != len(rows): raise ValueError("unexpected trailing TetGen .poly content")


def write_poly(path: Path, points: np.ndarray, faces: np.ndarray, markers: np.ndarray,
               holes: list[np.ndarray], regions: list[tuple[np.ndarray, int]]) -> None:
    lines = ["# Existing TetGen PLC input; physical admission false.",
             f"{len(points)} 3 0 1"]
    for i, point in enumerate(points, 1):
        lines.append(f"{i} {float(point[0]):.17g} {float(point[1]):.17g} "
                     f"{float(point[2]):.17g} 0")
    lines.append(f"{len(faces)} 1")
    for tri, marker in zip(faces, markers):
        lines.extend((f"1 0 {int(marker)}",
                      "3 " + " ".join(str(int(x) + 1) for x in tri)))
    lines.append(str(len(holes)))
    for i, point in enumerate(holes, 1):
        lines.append(f"{i} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}")
    lines.append(str(len(regions)))
    for i, (point, attribute) in enumerate(regions, 1):
        lines.append(f"{i} {point[0]:.17g} {point[1]:.17g} "
                     f"{point[2]:.17g} {int(attribute)} 0")
    path.write_text("\n".join(lines) + "\n")
    validate_poly_sections(path, len(points), len(faces), len(holes), len(regions))


def positive_f32_tetrahedra(points: np.ndarray, tetrahedra: np.ndarray,
                            expected_material: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(points, dtype=np.float32)
    if not np.isfinite(p).all():
        raise ValueError("TetGen node coordinates are nonfinite after Float32 emission")
    t = np.asarray(tetrahedra, dtype=np.int64).copy()
    if t.ndim != 2 or t.shape[1] != 4 or len(t) == 0 or len(t) > 200_000 or \
       int(t.min()) < 0 or int(t.max()) >= len(p):
        raise ValueError("TetGen cell table exceeds the bounded index/cell domain")
    pd = p.astype(np.float64)
    q = pd[t]
    signed = np.einsum("ij,ij->i", q[:, 1] - q[:, 0],
                       np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0])) / 6.0
    if np.any(~np.isfinite(signed)) or np.any(signed == 0.0):
        raise ValueError("Float32 emission collapsed a TetGen tetrahedron")
    negative = signed < 0.0
    if np.any(negative):
        t[negative, [0, 1]] = t[negative, [1, 0]]
    q = pd[t]
    volume = np.einsum("ij,ij->i", q[:, 1] - q[:, 0],
                       np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 0])) / 6.0
    if np.any(volume <= 0.0):
        raise ValueError("TetGen cells did not remain positive after Float32 reorientation")
    materials = np.full(len(t), int(expected_material), dtype=np.int32)
    return p, t, volume


def read_tetgen_mesh(node_path: Path, ele_path: Path, expected_material: int | None) -> dict[str, Any]:
    node_lines = [x.split() for x in node_path.read_text().splitlines()
                  if x.strip() and not x.lstrip().startswith("#")]
    ele_lines = [x.split() for x in ele_path.read_text().splitlines()
                 if x.strip() and not x.lstrip().startswith("#")]
    if not node_lines or not ele_lines:
        raise ValueError("TetGen node/element table is empty")
    nh = node_lines[0]
    if len(nh) != 4:
        raise ValueError("TetGen node header must declare count, 3D, zero attributes, and marker count")
    node_count, dimension, node_attributes, markers = map(int, nh)
    if node_count < 4 or node_count > 300_000 or dimension != 3 or node_attributes != 0 or markers not in (0, 1):
        raise ValueError("TetGen node table is outside the bounded native node contract")
    if len(node_lines) != node_count + 1:
        raise ValueError("TetGen node table row count differs from its header")
    node_ids = np.empty(node_count, dtype=np.int64)
    points = np.empty((node_count, 3), dtype=np.float32)
    id_to_row: dict[int, int] = {}
    for i, row in enumerate(node_lines[1:]):
        if len(row) != 4 + markers:
            raise ValueError("TetGen node row has an unexpected attribute/marker width")
        node_id = int(row[0])
        if node_id < 0 or node_id > 0xFFFFFFFF or node_id in id_to_row:
            raise ValueError("TetGen node table has an invalid or duplicate UInt32 ID")
        if markers:
            int(row[4])
        node_ids[i] = node_id
        id_to_row[node_id] = i
        points[i] = [float(row[1]), float(row[2]), float(row[3])]
    eh = ele_lines[0]
    if len(eh) != 3:
        raise ValueError("TetGen element header must declare count, four corners, and one attribute")
    cell_count, corners, attributes = map(int, eh)
    if cell_count == 0 or cell_count > 200_000 or corners != 4 or attributes != 1 or len(ele_lines) != cell_count + 1:
        raise ValueError("TetGen element table exceeds the per-domain 200000-cell/one-attribute contract")
    element_ids = np.empty(cell_count, dtype=np.int64)
    tetrahedra = np.empty((cell_count, 4), dtype=np.int64)
    source_node_ids = np.empty((cell_count, 4), dtype=np.int64)
    materials = np.empty(cell_count, dtype=np.int32)
    element_id_seen: set[int] = set()
    for i, row in enumerate(ele_lines[1:]):
        if len(row) != 6:
            raise ValueError("TetGen element row must contain one ID, four nodes, and one region attribute")
        element_id = int(row[0])
        if element_id < 0 or element_id > 0xFFFFFFFF or element_id in element_id_seen:
            raise ValueError("TetGen element table has an invalid or duplicate UInt32 ID")
        element_id_seen.add(element_id)
        attr = float(row[5])
        if not math.isfinite(attr) or not attr.is_integer() or int(attr) not in (15, 16) or \
           (expected_material is not None and int(attr) != expected_material):
            raise ValueError("TetGen element region attribute does not match its owning tissue material")
        element_ids[i] = element_id
        materials[i] = int(attr)
        for corner in range(4):
            source_id = int(row[corner + 1])
            if source_id not in id_to_row:
                raise ValueError("TetGen element references a node absent from its .node table")
            source_node_ids[i, corner] = source_id
            tetrahedra[i, corner] = id_to_row[source_id]
    original_tetrahedra = tetrahedra.copy()
    points, tetrahedra, volumes = positive_f32_tetrahedra(points, tetrahedra, expected_material if expected_material is not None else 0)
    reoriented = np.any(tetrahedra != original_tetrahedra, axis=1)
    for i, tet in enumerate(tetrahedra):
        source_node_ids[i] = node_ids[tet]
    canonical_cells = set()
    for tet in tetrahedra:
        key = tuple(sorted(map(int, tet)))
        if len(set(key)) != 4 or key in canonical_cells:
            raise ValueError("TetGen mesh contains a repeated or duplicate tetrahedron")
        canonical_cells.add(key)
    return {"node_ids": node_ids, "points": points, "id_to_row": id_to_row,
            "element_ids": element_ids, "tetrahedra": tetrahedra,
            "tetrahedron_node_ids": source_node_ids, "material_indices": materials,
            "volumes": volumes, "element_reoriented": reoriented}


def write_positive_tetgen_tables(node_path: Path, ele_path: Path,
                                  mesh: dict[str, Any]) -> None:
    with node_path.open("w") as stream:
        stream.write(f"{len(mesh['node_ids'])} 3 0 0\n")
        for node_id, point in zip(mesh["node_ids"], mesh["points"]):
            stream.write(f"{int(node_id)} {float(point[0]):.9g} {float(point[1]):.9g} {float(point[2]):.9g}\n")
    with ele_path.open("w") as stream:
        stream.write(f"{len(mesh['element_ids'])} 4 1\n")
        for element_id, nodes, material in zip(mesh["element_ids"], mesh["tetrahedron_node_ids"], mesh["material_indices"]):
            stream.write(f"{int(element_id)} " + " ".join(map(str, map(int, nodes))) + f" {int(material)}\n")


def _tetrahedron_boundary_faces(mesh: dict[str, Any]) -> dict[tuple[int, int, int], list[dict[str, Any]]]:
    points = np.asarray(mesh["points"], dtype=np.float32).astype(np.float64)
    boundary: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    local_faces = ((0, 1, 2, 3), (0, 3, 1, 2), (0, 2, 3, 1), (1, 3, 2, 0))
    for cell_row, tet in enumerate(np.asarray(mesh["tetrahedra"], dtype=np.int64)):
        for a, b, c, opposite in local_faces:
            face = [int(tet[a]), int(tet[b]), int(tet[c])]
            pa, pb, pc = points[face]
            po = points[int(tet[opposite])]
            if float(np.dot(np.cross(pb - pa, pc - pa), po - pa)) > 0.0:
                face[1], face[2] = face[2], face[1]
            key = tuple(sorted(face))
            boundary.setdefault(key, []).append({
                "element_row": cell_row,
                "element_id": int(mesh["element_ids"][cell_row]),
                "oriented_nodes": tuple(face),
                "material_index": int(mesh["material_indices"][cell_row]),
            })
    for key, rows in boundary.items():
        if len(rows) > 2:
            raise ValueError(f"TetGen mesh has a nonmanifold tetrahedron face {key}")
    return boundary


def _mesh_source_point_map(mesh: dict[str, Any], source_points: np.ndarray,
                           label: str) -> np.ndarray:
    lookup: dict[tuple[int, int, int], list[int]] = {}
    for row, point in enumerate(mesh["points"]):
        lookup.setdefault(_coordinate_key(point), []).append(row)
    mapped = np.empty(len(source_points), dtype=np.int64)
    for row, point in enumerate(np.asarray(source_points, dtype=np.float32)):
        matches = lookup.get(_coordinate_key(point), [])
        if len(matches) != 1:
            raise ValueError(f"{label} input surface node {row} has {len(matches)} exact-F32 matches in TetGen output")
        mapped[row] = matches[0]
    if len(set(map(int, mapped))) != len(mapped):
        raise ValueError(f"{label} surface source points map non-injectively to TetGen nodes")
    return mapped


def _shell_boundary_map(mesh: dict[str, Any], shell: dict[str, Any],
                        point_map: np.ndarray, label: str,
                        require_all_boundary: bool) -> list[dict[str, Any]]:
    boundary = _tetrahedron_boundary_faces(mesh)
    faces = np.asarray(shell["faces"], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    ancestry = shell["face_ancestry"]
    for face_row, face in enumerate(faces):
        child_face = tuple(int(point_map[int(v)]) for v in face)
        key = tuple(sorted(child_face))
        if key in seen:
            raise ValueError(f"{label} shell has duplicate facets after TetGen node mapping")
        seen.add(key)
        occurrences = boundary.get(key, [])
        if len(occurrences) != 1:
            raise ValueError(f"{label} source facet {face_row} maps to {len(occurrences)} TetGen boundary faces")
        source_offsets = ancestry["face_ancestry_offsets"]
        begin, end = int(source_offsets[face_row]), int(source_offsets[face_row + 1])
        rows.append({
            "shell_face_row": face_row,
            "face_marker": int(shell["face_marker"][face_row]),
            "face_source_role": str(ancestry["face_source_role"][face_row]),
            "face_source_row": int(ancestry["face_source_row"][face_row]),
            "face_exact_source_row": int(ancestry["face_exact_source_row"][face_row]),
            "face_source_original_id": int(ancestry["face_source_original_id"][face_row]),
            "face_source_layer": int(ancestry["face_source_layer"][face_row]),
            "face_source_body": int(ancestry["face_source_body"][face_row]),
            "face_source_bone": int(ancestry["face_source_bone"][face_row]),
            "face_run_original_id": int(ancestry["face_run_original_id"][face_row]),
            "face_source_face_id": int(ancestry["face_source_face_id"][face_row]),
            "face_ancestry_rows": np.asarray(ancestry["face_ancestry_source_rows"][begin:end], dtype=np.int64).tolist(),
            "face_ancestry_original_ids":(np.asarray(ancestry["face_ancestry_source_original_ids"][begin:end],dtype=np.int64).tolist() if "face_ancestry_source_original_ids" in ancestry else [int(ancestry["face_source_original_id"][face_row])]*(end-begin)),
            "face_ancestry_parent_rows":(np.asarray(ancestry["face_ancestry_parent_face_rows"][begin:end],dtype=np.int64).tolist() if "face_ancestry_parent_face_rows" in ancestry else [INVALID]*(end-begin)),
            "face_parent_tet_row": int(ancestry["face_parent_tet_row"][face_row]),
            "face_parent_material_index": int(ancestry["face_parent_material_index"][face_row]),
            "face_parent_source_node_rows": np.asarray(ancestry["face_parent_source_node_rows"][face_row], dtype=np.int64).tolist(),
            "child_node_rows": child_face,
            "boundary": occurrences[0],
        })
    if require_all_boundary and seen != {key for key, values in boundary.items() if len(values) == 1}:
        raise ValueError(f"{label} TetGen boundary facets differ from the input PLC facet set")
    return rows


def _write_joint_node_element_files(node_path: Path, ele_path: Path,
                                    points: np.ndarray, tetrahedra: np.ndarray,
                                    materials: np.ndarray) -> None:
    with node_path.open("w") as stream:
        stream.write(f"{len(points)} 3 0 0\n")
        for row, point in enumerate(points, 1):
            stream.write(f"{row} {float(point[0]):.9g} {float(point[1]):.9g} {float(point[2]):.9g}\n")
    with ele_path.open("w") as stream:
        stream.write(f"{len(tetrahedra)} 4 1\n")
        for row, (tet, material) in enumerate(zip(tetrahedra, materials), 1):
            nodes = [int(v) + 1 for v in tet]
            stream.write(f"{row} " + " ".join(map(str, nodes)) + f" {int(material)}\n")


def _opposite_triangle_orientation(a: tuple[int, int, int],
                                   b: tuple[int, int, int]) -> bool:
    return any(a == (b[shift], b[(shift - 1) % 3], b[(shift - 2) % 3])
               for shift in range(3))


def _registered_surface_tetgen_rows(
        shell: dict[str, Any], mesh: dict[str, Any], point_map: np.ndarray,
        joint_map: np.ndarray, layer: str, expected_marker: int,
        expected_role: str, label: str) -> tuple[np.ndarray, np.ndarray]:
    prefix = f"registered_{layer}"
    keys = (f"{prefix}_source_vertex_rows", f"{prefix}_shell_node_rows",
            f"{prefix}_source_faces", f"{prefix}_shell_face_rows")
    if any(key not in shell for key in keys):
        raise ValueError(f"{label} is missing its exact registered-{layer} source/node/face map")
    source_vertices = np.asarray(shell[keys[0]], dtype=np.int64)
    shell_nodes = np.asarray(shell[keys[1]], dtype=np.int64)
    source_faces = np.asarray(shell[keys[2]], dtype=np.int64)
    shell_face_rows = np.asarray(shell[keys[3]], dtype=np.int64)
    shell_points = np.asarray(shell["points_world_m"], dtype=np.float32)
    shell_faces = np.asarray(shell["faces"], dtype=np.int64)
    markers = np.asarray(shell["face_marker"], dtype=np.int64)
    ancestry = shell["face_ancestry"]
    roles = np.asarray(ancestry["face_source_role"]).astype(str)
    if source_vertices.ndim != 1 or len(source_vertices) == 0 or \
       not np.array_equal(source_vertices, np.arange(len(source_vertices), dtype=np.int64)) or \
       shell_nodes.shape != source_vertices.shape or \
       len(set(map(int, shell_nodes))) != len(shell_nodes) or \
       np.any(shell_nodes < 0) or np.any(shell_nodes >= len(shell_points)):
        raise ValueError(f"{label} has a missing, duplicate, or invalid registered-{layer} node row")
    if source_faces.ndim != 2 or source_faces.shape[1] != 3 or len(source_faces) == 0 or \
       shell_face_rows.shape != (len(source_faces),) or \
       np.any(source_faces < 0) or np.any(source_faces >= len(source_vertices)) or \
       np.any(shell_face_rows < 0) or np.any(shell_face_rows >= len(shell_faces)) or \
       len(set(map(int, shell_face_rows))) != len(shell_face_rows):
        raise ValueError(f"{label} has a missing, duplicate, or invalid registered-{layer} face row")
    if not np.array_equal(np.unique(source_faces), np.arange(len(source_vertices), dtype=np.int64)):
        raise ValueError(f"{label} registered-{layer} source faces omit or duplicate surface point coverage")
    selected = np.flatnonzero((markers == expected_marker) & (roles == expected_role))
    if not np.array_equal(np.sort(selected), np.sort(shell_face_rows)):
        raise ValueError(f"{label} registered-{layer} face markers/roles are missing, duplicated, or mismapped")
    expected_shell_faces = shell_nodes[source_faces]
    actual_shell_faces = shell_faces[shell_face_rows]
    if not np.array_equal(np.sort(expected_shell_faces, axis=1),
                          np.sort(actual_shell_faces, axis=1)):
        raise ValueError(f"{label} registered-{layer} source face rows do not match marker-selected shell facets")
    point_map = np.asarray(point_map, dtype=np.int64)
    joint_map = np.asarray(joint_map, dtype=np.int64)
    mesh_points = np.asarray(mesh["points"], dtype=np.float32)
    if point_map.shape != (len(shell_points),) or joint_map.shape != (len(mesh_points),):
        raise ValueError(f"{label} exact shell-to-TetGen node map has the wrong domain")
    tetgen_rows = point_map[shell_nodes]
    if np.any(tetgen_rows < 0) or np.any(tetgen_rows >= len(mesh_points)) or \
       not np.array_equal(mesh_points[tetgen_rows], shell_points[shell_nodes]):
        raise ValueError(f"{label} registered-{layer} source points do not exactly match mapped TetGen Float32 rows")
    joint_rows = joint_map[tetgen_rows]
    if np.any(joint_rows < 0) or len(set(map(int, joint_rows))) != len(joint_rows):
        raise ValueError(f"{label} registered-{layer} source-node map is collapsed or noninjective")
    return tetgen_rows, joint_rows


def _add_surface_owner_rows(owner: dict[int, tuple[str, int]],
                            tetgen_rows: np.ndarray, layer: str,
                            label: str) -> None:
    for source_row, raw_tetgen_row in enumerate(np.asarray(tetgen_rows, dtype=np.int64)):
        tetgen_row = int(raw_tetgen_row)
        if tetgen_row in owner:
            raise ValueError(f"{label} registered surface rows collapse across layers at TetGen node {tetgen_row}")
        owner[tetgen_row] = (layer, source_row)


def combine_tetgen_children(skin_node_path: Path, skin_ele_path: Path,
                            fat_node_path: Path, fat_ele_path: Path,
                            skin_shell: dict[str, Any], fat_shell: dict[str, Any],
                            inner: dict[str, Any], out_dir: Path,
                            require_registered_surface_maps: bool = True) -> dict[str, Any]:
    skin = read_tetgen_mesh(skin_node_path, skin_ele_path, 15)
    fat = read_tetgen_mesh(fat_node_path, fat_ele_path, 16)
    total_cells = len(skin["tetrahedra"]) + len(fat["tetrahedra"])
    if total_cells > 200_000:
        raise ValueError(f"combined skin/fat TetGen cell budget exceeded: {total_cells}>200000")
    skin_point_map = _mesh_source_point_map(skin, np.asarray(skin_shell["points_world_m"], dtype=np.float32), "skin")
    fat_point_map = _mesh_source_point_map(fat, np.asarray(fat_shell["points_world_m"], dtype=np.float32), "fat")
    skin_facet_rows = _shell_boundary_map(skin, skin_shell, skin_point_map, "skin", True)
    fat_facet_rows = _shell_boundary_map(fat, fat_shell, fat_point_map, "fat", True)

    def registered_i(mesh: dict[str, Any], shell: dict[str, Any], point_map: np.ndarray,
                     label: str) -> dict[int, int]:
        source_ids = np.asarray(shell["registered_I_source_vertex_rows"], dtype=np.int64)
        shell_rows = np.asarray(shell["registered_I_shell_node_rows"], dtype=np.int64)
        if source_ids.ndim != 1 or shell_rows.shape != source_ids.shape or len(source_ids) == 0 or \
           len(set(map(int, source_ids))) != len(source_ids) or \
           np.any(shell_rows < 0) or np.any(shell_rows >= len(point_map)):
            raise ValueError(f"{label} registered I node identity table is invalid")
        mapped = {int(source): int(point_map[int(source_row)])
                  for source, source_row in zip(source_ids, shell_rows)}
        if len(set(mapped.values())) != len(mapped):
            raise ValueError(f"{label} registered I nodes map non-injectively to TetGen output")
        return mapped

    skin_i = registered_i(skin, skin_shell, skin_point_map, "skin")
    fat_i = registered_i(fat, fat_shell, fat_point_map, "fat")
    registered_i_source_points = np.asarray(inner["points"], dtype=np.float32)
    for source_vertex in range(len(registered_i_source_points)):
        skin_shell_row = int(skin_shell["registered_I_shell_node_rows"][source_vertex])
        fat_shell_row = int(fat_shell["registered_I_shell_node_rows"][source_vertex])
        source_key = _coordinate_key(registered_i_source_points[source_vertex])
        if _coordinate_key(skin_shell["points_world_m"][skin_shell_row]) != source_key or \
           _coordinate_key(fat_shell["points_world_m"][fat_shell_row]) != source_key:
            raise ValueError(f"registered I source point {source_vertex} differs from the input I surface")
    if set(skin_i) != set(fat_i):
        raise ValueError("skin/fat do not carry the same registered I source-node identity set")
    for source_vertex in skin_i:
        if _coordinate_key(skin["points"][skin_i[source_vertex]]) != \
           _coordinate_key(fat["points"][fat_i[source_vertex]]):
            raise ValueError(f"registered I source vertex {source_vertex} differs between TetGen outputs")

    joint_points = [point.copy() for point in skin["points"]]
    skin_joint = np.arange(len(skin["points"]), dtype=np.int64)
    fat_joint = np.full(len(fat["points"]), -1, dtype=np.int64)
    source_vertex_to_joint = {source: int(skin_i[source]) for source in skin_i}
    for source_vertex, fat_row in fat_i.items():
        fat_joint[fat_row] = source_vertex_to_joint[source_vertex]
    for row in range(len(fat["points"])):
        if fat_joint[row] >= 0:
            continue
        fat_joint[row] = len(joint_points)
        joint_points.append(fat["points"][row].copy())
    joint_points_array = np.asarray(joint_points, dtype=np.float32)
    if len(joint_points_array) > 300_000:
        raise ValueError("combined TetGen node count exceeds the native loader limit of 300000")

    skin_surface_owner: dict[int, tuple[str, int]] = {}
    fat_surface_owner: dict[int, tuple[str, int]] = {}
    registered_surface_joint_rows: dict[str, np.ndarray] = {}
    if require_registered_surface_maps:
        skin_o_tet, skin_o_joint = _registered_surface_tetgen_rows(
            skin_shell, skin, skin_point_map, skin_joint, "O",
            MARKER["skin_outer_O"], "outer_O", "skin")
        skin_i_tet, skin_i_joint = _registered_surface_tetgen_rows(
            skin_shell, skin, skin_point_map, skin_joint, "I",
            MARKER["skin_inner_I"], "inner_I", "skin")
        fat_i_tet, fat_i_joint = _registered_surface_tetgen_rows(
            fat_shell, fat, fat_point_map, fat_joint, "I",
            MARKER["fat_outer_I"], "outer_I", "fat")
        fat_d_tet, fat_d_joint = _registered_surface_tetgen_rows(
            fat_shell, fat, fat_point_map, fat_joint, "D",
            MARKER["fat_inner_D"], "inner_D", "fat")
        if not np.array_equal(skin_i_joint, fat_i_joint):
            raise ValueError("skin/fat registered-I source rows do not map to the same combined nodes")
        _add_surface_owner_rows(skin_surface_owner, skin_o_tet, "O", "skin")
        _add_surface_owner_rows(skin_surface_owner, skin_i_tet, "I", "skin")
        _add_surface_owner_rows(fat_surface_owner, fat_i_tet, "I", "fat")
        _add_surface_owner_rows(fat_surface_owner, fat_d_tet, "D", "fat")
        registered_surface_joint_rows = {"O": skin_o_joint, "I": skin_i_joint,
                                         "D": fat_d_joint}
    else:
        # Kept only for the historical 7-node/2-cell synthetic interface fixture.
        # Production callers use the strict default and must supply O/I/D maps.
        skin_i_tet = np.asarray([skin_i[i] for i in range(len(inner["points"]))], dtype=np.int64)
        fat_i_tet = np.asarray([fat_i[i] for i in range(len(inner["points"]))], dtype=np.int64)
        _add_surface_owner_rows(skin_surface_owner, skin_i_tet, "I", "legacy fixture skin")
        _add_surface_owner_rows(fat_surface_owner, fat_i_tet, "I", "legacy fixture fat")
        registered_surface_joint_rows = {"O": np.empty(0, dtype=np.int64),
                                         "I": np.asarray([source_vertex_to_joint[i]
                                                          for i in range(len(inner["points"]))], dtype=np.int64),
                                         "D": np.empty(0, dtype=np.int64)}

    if require_registered_surface_maps:
        # Preserve physical coordinates and source IDs while giving the existing
        # direct accepted-FEM skin renderer a compact outer-node prefix. Do not
        # rely on TetGen file row ordering, even when input points usually lead.
        outer_rows = registered_surface_joint_rows["O"]
        retained = np.ones(len(joint_points_array), dtype=bool)
        retained[outer_rows] = False
        new_to_old = np.concatenate((outer_rows, np.flatnonzero(retained)))
        if len(new_to_old) != len(joint_points_array) or \
           len(np.unique(new_to_old)) != len(joint_points_array):
            raise ValueError("registered O prefix permutation is not bijective")
        old_to_new = np.empty(len(new_to_old), dtype=np.int64)
        old_to_new[new_to_old] = np.arange(len(new_to_old), dtype=np.int64)
        joint_points_array = joint_points_array[new_to_old]
        skin_joint = old_to_new[skin_joint]
        fat_joint = old_to_new[fat_joint]
        source_vertex_to_joint = {source: int(old_to_new[node])
                                  for source, node in source_vertex_to_joint.items()}
        registered_surface_joint_rows = {layer: old_to_new[rows]
                                         for layer, rows in registered_surface_joint_rows.items()}
        if not np.array_equal(registered_surface_joint_rows["O"],
                              np.arange(len(outer_rows), dtype=np.int64)):
            raise ValueError("registered O nodes lost their exact mechanical prefix")

    joint_tetrahedra: list[list[int]] = []
    joint_materials: list[int] = []
    cell_map_rows: list[list[Any]] = []
    joint_element_by_source: dict[tuple[str, int], int] = {}
    seen_cells: set[tuple[int, int, int, int]] = set()
    for label, mesh, row_map in (("skin", skin, skin_joint), ("fat", fat, fat_joint)):
        for tet, element_id, material, reoriented in zip(mesh["tetrahedra"], mesh["element_ids"],
                mesh["material_indices"], mesh["element_reoriented"]):
            mapped = [int(row_map[int(node)]) for node in tet]
            p = joint_points_array[mapped].astype(np.float64)
            signed = float(np.dot(p[1] - p[0], np.cross(p[2] - p[0], p[3] - p[0])) / 6.0)
            if not math.isfinite(signed) or signed <= 0.0:
                raise ValueError(f"combined Float32 cell became nonpositive at {label} source element {int(element_id)}")
            canonical = tuple(sorted(mapped))
            if len(set(canonical)) != 4 or canonical in seen_cells:
                raise ValueError("combined child mesh contains a collapsed or duplicate tetrahedron")
            seen_cells.add(canonical)
            joint_tetrahedra.append(mapped)
            joint_materials.append(int(material))
            joint_element_id = len(cell_map_rows) + 1
            cell_map_rows.append([joint_element_id, joint_element_id - 1, label,
                                  int(element_id), int(material), int(bool(reoriented))])
            joint_element_by_source[(label, int(element_id))] = joint_element_id
    joint_tetrahedra_array = np.asarray(joint_tetrahedra, dtype=np.int64)
    joint_materials_array = np.asarray(joint_materials, dtype=np.int32)
    joint_mesh = {"points": joint_points_array, "tetrahedra": joint_tetrahedra_array,
                  "element_ids": np.arange(1, total_cells + 1, dtype=np.int64),
                  "material_indices": joint_materials_array}
    joint_boundary = _tetrahedron_boundary_faces(joint_mesh)
    skin_boundary = _tetrahedron_boundary_faces(skin)
    fat_boundary = _tetrahedron_boundary_faces(fat)
    skin_i_faces = np.asarray(skin_shell["registered_I_shell_faces"], dtype=np.int64)
    fat_i_faces = np.asarray(fat_shell["registered_I_shell_faces"], dtype=np.int64)
    source_i_faces = np.asarray(inner["faces"], dtype=np.int64)
    source_i_rows = np.asarray(inner["source_rows"], dtype=np.int64)
    source_i_exact = np.asarray(inner["exact_source_rows"], dtype=np.int64)
    source_i_ids = np.asarray(inner["source_face_ids"], dtype=np.int64)
    expected_i_vertices = np.arange(len(inner["points"]), dtype=np.int64)
    if not np.array_equal(skin_shell["registered_I_source_vertex_rows"], expected_i_vertices) or \
       not np.array_equal(fat_shell["registered_I_source_vertex_rows"], expected_i_vertices) or \
       skin_i_faces.shape != source_i_faces.shape or fat_i_faces.shape != source_i_faces.shape or \
       not np.array_equal(skin_shell["registered_I_source_faces"], source_i_faces) or \
       not np.array_equal(fat_shell["registered_I_source_faces"], source_i_faces) or \
       len(source_i_rows) != len(source_i_faces) or len(source_i_exact) != len(source_i_faces) or \
       len(source_i_ids) != len(source_i_faces) or len(source_i_faces) == 0:
        raise ValueError("registered I source faces do not match both assembled shell domains")
    i_interface_rows: list[list[Any]] = []
    interface_joint_faces: set[tuple[int, int, int]] = set()
    for face_row, source_face in enumerate(source_i_faces):
        sf = tuple(int(skin_point_map[int(v)]) for v in skin_i_faces[face_row])
        ff = tuple(int(fat_point_map[int(v)]) for v in fat_i_faces[face_row])
        sk_occ = skin_boundary.get(tuple(sorted(sf)), [])
        ft_occ = fat_boundary.get(tuple(sorted(ff)), [])
        if len(sk_occ) != 1 or len(ft_occ) != 1:
            raise ValueError(f"registered I face {face_row} is not exactly one boundary facet per tissue mesh")
        sj = tuple(int(skin_joint[v]) for v in sk_occ[0]["oriented_nodes"])
        fj = tuple(int(fat_joint[v]) for v in ft_occ[0]["oriented_nodes"])
        if not _opposite_triangle_orientation(sj, fj):
            raise ValueError(f"registered I face {face_row} has non-opposite outward orientation across materials")
        joint_key = tuple(sorted(sj))
        if joint_key in interface_joint_faces:
            raise ValueError("registered I source face list contains duplicate joint facets")
        interface_joint_faces.add(joint_key)
        whole = joint_boundary.get(joint_key, [])
        if len(whole) != 2 or not _opposite_triangle_orientation(
                whole[0]["oriented_nodes"], whole[1]["oriented_nodes"]) or \
           {entry["material_index"] for entry in whole} != {15, 16}:
            raise ValueError(f"registered I face {face_row} is not an oppositely oriented skin/fat two-sided manifold interface after merge")
        i_interface_rows.append([
            face_row, int(source_i_rows[face_row]), int(source_i_exact[face_row]),
            int(source_i_ids[face_row]), *map(int, source_face),
            *map(int, skin["node_ids"][list(sf)]), *map(int, fat["node_ids"][list(ff)]),
            *[int(v) + 1 for v in sj],
            int(sk_occ[0]["element_id"]), int(ft_occ[0]["element_id"]),
            joint_element_by_source[("skin", int(sk_occ[0]["element_id"]))],
            joint_element_by_source[("fat", int(ft_occ[0]["element_id"]))],
        ])
    if len(interface_joint_faces) != len(source_i_faces):
        raise ValueError("registered I interface face mapping is incomplete")

    out_node = out_dir / "skin-fat-combined.node"
    out_ele = out_dir / "skin-fat-combined.ele"
    _write_joint_node_element_files(out_node, out_ele, joint_points_array,
                                    joint_tetrahedra_array, joint_materials_array)
    with gzip.open(out_dir / "skin-fat-combined-node-map.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["joint_node_id", "source_partition", "source_tetgen_node_id",
                         "registered_I_source_vertex_row", "merge_peer_partition",
                         "merge_peer_tetgen_node_id", "registered_surface_layer",
                         "registered_surface_vertex_row"])
        fat_source_by_i = {source: int(fat["node_ids"][row]) for source, row in fat_i.items()}
        skin_source_by_i = {source: int(skin["node_ids"][row]) for source, row in skin_i.items()}
        skin_source_i_by_row = {row: source for source, row in skin_i.items()}
        fat_source_i_by_row = {row: source for source, row in fat_i.items()}
        for row, node_id in enumerate(skin["node_ids"]):
            source_i = skin_source_i_by_row.get(row, INVALID)
            surface_layer, surface_vertex = skin_surface_owner.get(row, ("", INVALID))
            writer.writerow([int(skin_joint[row]) + 1, "skin", int(node_id), source_i,
                             "fat" if source_i != INVALID else "", fat_source_by_i.get(source_i, ""),
                             surface_layer, surface_vertex])
        for row, node_id in enumerate(fat["node_ids"]):
            source_i = fat_source_i_by_row.get(row, INVALID)
            surface_layer, surface_vertex = fat_surface_owner.get(row, ("", INVALID))
            writer.writerow([int(fat_joint[row]) + 1, "fat", int(node_id), source_i,
                             "skin" if source_i != INVALID else "", skin_source_by_i.get(source_i, ""),
                             surface_layer, surface_vertex])
    with gzip.open(out_dir / "skin-fat-combined-element-map.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["joint_element_id", "child_tet_row", "source_partition",
                         "source_tetgen_element_id", "material_index",
                         "source_orientation_reordered_to_positive"])
        writer.writerows(cell_map_rows)
    with gzip.open(out_dir / "skin-fat-combined-facet-ancestry.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_partition", "shell_face_row", "face_marker", "face_source_role",
                         "face_source_row", "face_exact_source_row", "face_source_original_id",
                         "face_source_layer", "face_source_body", "face_source_bone",
                         "face_ancestry_kind",
                         "face_run_original_id", "face_source_face_id", "face_ancestry_source_rows",
                         "face_parent_tet_row", "face_parent_material_index", "parent_source_node_rows",
                         "joint_element_id", "child_tet_row", "joint_node0", "joint_node1", "joint_node2",
                         "face_ancestry_source_original_ids","face_ancestry_parent_face_rows"])
        for label, rows, node_map, shell in (("skin", skin_facet_rows, skin_joint, skin_shell),
                                             ("fat", fat_facet_rows, fat_joint, fat_shell)):
            shell_ancestry_kind = np.asarray(shell["face_ancestry"]["face_ancestry_kind"], dtype=np.int64)
            for row in rows:
                boundary = row["boundary"]
                mapped = [int(node_map[int(v)]) + 1 for v in boundary["oriented_nodes"]]
                writer.writerow([label, row["shell_face_row"], row["face_marker"], row["face_source_role"],
                    row["face_source_row"], row["face_exact_source_row"], row["face_source_original_id"],
                    row["face_source_layer"], row["face_source_body"], row["face_source_bone"],
                    int(shell_ancestry_kind[row["shell_face_row"]]),
                    row["face_run_original_id"], row["face_source_face_id"],
                    json.dumps(row["face_ancestry_rows"], separators=(",", ":")),
                    row["face_parent_tet_row"], row["face_parent_material_index"],
                    json.dumps(row["face_parent_source_node_rows"], separators=(",", ":")),
                    joint_element_by_source.get((label, boundary["element_id"]), INVALID),
                    joint_element_by_source.get((label, boundary["element_id"]), INVALID) - 1, *mapped,
                    json.dumps(row["face_ancestry_original_ids"],separators=(",",":")),
                    json.dumps(row["face_ancestry_parent_rows"],separators=(",",":"))])
    with gzip.open(out_dir / "skin-fat-combined-I-interface.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["registered_I_source_face_row", "source_face_row", "exact_source_face_row",
                         "source_face_id", "source_vertex0", "source_vertex1", "source_vertex2",
                         "skin_child_node0", "skin_child_node1", "skin_child_node2",
                         "fat_child_node0", "fat_child_node1", "fat_child_node2",
                         "joint_node0", "joint_node1", "joint_node2",
                         "skin_tetgen_element_id", "fat_tetgen_element_id",
                         "skin_joint_element_id", "fat_joint_element_id",
                         "skin_child_tet_row", "fat_child_tet_row"])
        for row in i_interface_rows:
            writer.writerow([*row, row[-2] - 1, row[-1] - 1])
    reloaded = read_tetgen_mesh(out_node, out_ele, None)
    if not np.array_equal(reloaded["points"], joint_points_array) or \
       not np.array_equal(reloaded["tetrahedra"], joint_tetrahedra_array) or \
       not np.array_equal(reloaded["material_indices"], joint_materials_array):
        raise ValueError("combined TetGen node/element output did not round-trip its F32 mesh")
    return {
        "status": "COMBINED_POSITIVE_F32_CHILD_MESH_NOT_PHYSICAL_ADMISSION",
        "nodes": len(joint_points_array), "skin_tetrahedra": len(skin["tetrahedra"]),
        "fat_tetrahedra": len(fat["tetrahedra"]), "combined_tetrahedra": total_cells,
        "registered_I_source_vertices": len(skin_i),
        "registered_I_source_faces": len(i_interface_rows),
        "registered_surface_node_map_complete": bool(require_registered_surface_maps),
        "registered_outer_nodes_are_mechanical_prefix": bool(require_registered_surface_maps),
        "combined_node_order": ("registered_O_source_rows_then_remaining_TetGen_rows"
                                if require_registered_surface_maps else "legacy_fixture_TetGen_rows"),
        "registered_surface_node_rows": {key: int(len(value))
                                         for key, value in registered_surface_joint_rows.items()},
        "positive_float32_cells": total_cells,
        "combined_cell_budget_200000_pass": total_cells <= 200_000,
        "only_registered_I_nodes_merged": True,
        "unregistered_equal_coordinate_nodes_welded": False,
        "physical_admission": False,
        "outputs": {path.name: sha256(path) for path in out_dir.glob("skin-fat-combined*") if path.is_file()},
    }


def _test_tetgen_result(node_path: Path, ele_path: Path, material: int,
                        out_dir: Path, label: str) -> dict[str, Any]:
    mesh = read_tetgen_mesh(node_path, ele_path, material)
    points = mesh["points"]
    tets = mesh["tetrahedra"]
    volumes = mesh["volumes"]
    material_indices = mesh["material_indices"]
    positive_node_path = out_dir / f"{label}-positive.node"
    positive_ele_path = out_dir / f"{label}-positive.ele"
    write_positive_tetgen_tables(positive_node_path, positive_ele_path, mesh)
    np.savez_compressed(out_dir / f"{label}-tets.npz", points_world_m=points,
                        tetrahedra=tets, material_indices=material_indices,
                        tetrahedron_volume_m3=volumes,
                        positive_orientation_verified=np.asarray([True]))
    with gzip.open(out_dir / f"{label}-child-cells.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["child_tet_row", "material_index", "node0", "node1", "node2",
                         "node3", "centroid_x_m", "centroid_y_m", "centroid_z_m",
                         "reference_volume_m3", "source_region_label"])
        for i, (tet, volume) in enumerate(zip(tets, volumes)):
            center = points[tet].astype(np.float64).mean(axis=0)
            writer.writerow([i, material, *map(int, tet), *map(float, center),
                             float(volume), "skin" if material == 15 else "fat"])
    return {"status": "POSITIVE_F32_CELLS_NOT_PHYSICAL_ADMISSION",
            "material_index": material, "nodes": len(points),
            "tetrahedra": len(tets), "positive_f32_cells": len(tets),
            "maximum_cells_200000_pass": len(tets) <= 200_000,
            "tetgen_node_sha256": sha256(node_path),
            "tetgen_element_sha256": sha256(ele_path),
            "positive_node_file": positive_node_path.name,
            "positive_element_file": positive_ele_path.name,
            "positive_node_sha256": sha256(positive_node_path),
            "positive_element_sha256": sha256(positive_ele_path),
            "cells_reoriented_positive": int(np.count_nonzero(mesh["element_reoriented"]))}

def run_tetgen(poly_path: Path, tetgen_binary: Path, out_dir: Path,
               label: str, material: int, timeout_s: int = 60) -> dict[str, Any]:
    if not tetgen_binary.is_file():
        raise FileNotFoundError(f"TetGen binary is missing: {tetgen_binary}")
    command = [str(tetgen_binary), "-pYq1.2/10AS512", str(poly_path)]
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=out_dir, capture_output=True,
                                text=True, timeout=timeout_s)
        timed_out = False
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode("utf-8", "replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode("utf-8", "replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        (out_dir / f"{label}-tetgen.stdout.log").write_text(stdout)
        (out_dir / f"{label}-tetgen.stderr.log").write_text(stderr)
        return {"status": "TETGEN_TIMEOUT", "timeout_s": timeout_s,
                "wall_s": time.monotonic() - start, "command": command}
    (out_dir / f"{label}-tetgen.stdout.log").write_text(result.stdout)
    (out_dir / f"{label}-tetgen.stderr.log").write_text(result.stderr)
    report = {"return_code": result.returncode, "timed_out": timed_out,
              "wall_s": time.monotonic() - start, "command": command}
    node_path = out_dir / f"{poly_path.stem}.1.node"
    ele_path = out_dir / f"{poly_path.stem}.1.ele"
    if result.returncode != 0 or not node_path.exists() or not ele_path.exists():
        report["status"] = "TETGEN_FAILED"
        return report
    report.update(_test_tetgen_result(node_path, ele_path, material, out_dir, label))
    report["tetgen_wall_s"] = report["wall_s"]
    return report


def _import_manifold():
    import sys
    sys.path.insert(0, str(D / "authoring-deps"))
    import manifold3d as mf
    return mf


def _manifold(points: np.ndarray, faces: np.ndarray, markers: np.ndarray | None = None):
    mf = _import_manifold()
    kwargs: dict[str, Any] = {
        "vert_properties": np.asarray(points, dtype=np.float64, order="C", copy=True),
        "tri_verts": np.asarray(faces, dtype=np.uint64, order="C", copy=True),
    }
    if markers is not None:
        kwargs["face_id"] = np.asarray(markers, dtype=np.uint64, order="C", copy=True)
    manifold = mf.Manifold(mf.Mesh64(**kwargs))
    if manifold.status() != mf.Error.NoError:
        raise ValueError(f"Manifold3D rejected PLC shell: {manifold.status()}")
    return manifold


def _point_inside(manifold: Any, point: np.ndarray) -> bool:
    box = np.asarray(manifold.bounding_box(), dtype=np.float64)
    span = max(float(np.max(box[3:] - box[:3])), 0.01)
    votes = []
    for axis in range(3):
        start = np.asarray(point, dtype=np.float64).copy()
        start[axis] = box[axis] - span - 0.25
        other = [j for j in range(3) if j != axis]
        start[other[0]] += 0.0137 * span
        start[other[1]] -= 0.0211 * span
        votes.append(len(manifold.ray_cast(start.tolist(), np.asarray(point).tolist())) % 2 == 1)
    return sum(votes) >= 2


def _interior_point(manifold: Any) -> np.ndarray:
    mesh = manifold.to_mesh64()
    points = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    triangles = points[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    for fi in np.argsort(areas)[::-1][:min(128, len(areas))]:
        if areas[fi] <= 0.0:
            continue
        center = triangles[fi].mean(axis=0)
        normal = cross[fi] / areas[fi]
        for epsilon in (1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3):
            for sign in (-1.0, 1.0):
                candidate = center + sign * epsilon * normal
                if _point_inside(manifold, candidate):
                    return candidate
    raise ValueError("could not construct a verified interior point")


def _core_holes(paths: list[Path]) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    for path in paths:
        z = np.load(path, allow_pickle=False)
        points = np.asarray(z["points_world_m"], dtype=np.float32)
        if "source_faces" in z.files and "temporary_construction_cap_faces" in z.files:
            faces = np.vstack((z["source_faces"], z["temporary_construction_cap_faces"])).astype(np.int64)
        else:
            faces = np.asarray(z["faces"], dtype=np.int64)
        core = _manifold(points, faces)
        for component in core.decompose():
            if float(component.volume()) > 1e-15:
                holes.append(_interior_point(component))
    if not holes:
        raise ValueError("no positive closed core components supplied for TetGen holes")
    return holes


def _parent_centroids_by_material(nodes_csv_gz: Path, cells_csv_gz: Path) -> dict[int, list[np.ndarray]]:
    with gzip.open(nodes_csv_gz, "rt", newline="") as stream:
        nodes = list(csv.DictReader(stream))
    nodes.sort(key=lambda row: int(row["parent_node_index"]))
    points = np.asarray([[float(row[f"current_{axis}_m"]) for axis in "xyz"]
                         for row in nodes], dtype=np.float64)
    output: dict[int, list[np.ndarray]] = {15: [], 16: []}
    with gzip.open(cells_csv_gz, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            material = int(row["material_index"])
            if material not in output:
                continue
            tet = [int(row[f"parent_node{k}"]) for k in range(4)]
            if any(points[tet[i]].shape != (3,) for i in range(4)):
                raise ValueError("parent tetrahedron node row is invalid")
            output[material].append(points[tet].mean(axis=0))
    return output


def _region_point(shell: dict[str, Any], material: int,
                  parent_centroids: dict[int, list[np.ndarray]]) -> np.ndarray:
    manifold = _manifold(shell["points_world_m"], shell["faces"], shell["face_marker"])
    for point in parent_centroids[material]:
        if _point_inside(manifold, point):
            return point
    raise ValueError(f"no positive D37 material-{material} centroid lies inside assembled shell")


def _face_ancestry_rows(shell: dict[str, Any]) -> list[list[int]]:
    offsets = shell["face_ancestry"]["face_ancestry_offsets"]
    rows = shell["face_ancestry"]["face_ancestry_source_rows"]
    return [rows[offsets[i]:offsets[i + 1]].tolist() for i in range(len(offsets) - 1)]


def export_shells_and_plcs(skin: dict[str, Any], fat: dict[str, Any],
                           out_dir: Path, skin_holes: list[np.ndarray],
                           fat_holes: list[np.ndarray],
                           regions: dict[str, np.ndarray]) -> tuple[Path, Path]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing attempt directory: {out_dir}")
    out_dir.mkdir(parents=True)
    skin_path = out_dir / "skin-shell.npz"
    fat_path = out_dir / "fat-shell.npz"
    write_shell_npz(skin_path, skin, "skin")
    write_shell_npz(fat_path, fat, "fat")
    write_face_ancestry_csv(out_dir / "skin-face-ancestry.csv.gz", skin)
    write_face_ancestry_csv(out_dir / "fat-face-ancestry.csv.gz", fat)
    skin_poly = out_dir / "skin.poly"
    fat_poly = out_dir / "fat.poly"
    write_poly(skin_poly, skin["points_world_m"], skin["faces"], skin["face_marker"],
               skin_holes, [(regions["skin"], 15)])
    write_poly(fat_poly, fat["points_world_m"], fat["faces"], fat["face_marker"],
               fat_holes, [(regions["fat"], 16)])
    return skin_poly, fat_poly


def main() -> int:
    parser = argparse.ArgumentParser(description="one-shot O/I/D adapter using existing NPZ, TetGen PLC, and child-cell CSV conventions")
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--inner", type=Path, required=True)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--collar", type=Path, default=PROPOSAL)
    parser.add_argument("--wall-csv", type=Path, default=WALL_CSV)
    parser.add_argument("--collar-native-placement",type=Path,
                        help="existing native-placement NPZ; original wall ancestry is validated before placement")
    parser.add_argument("--parent-cells", type=Path, default=PARENT_CELLS)
    parser.add_argument("--parent-nodes", type=Path, default=PARENT_NODES)
    parser.add_argument("--inner-hole-core", type=Path, action="append", default=[])
    parser.add_argument("--deep-hole-core", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tetgen", action="store_true")
    parser.add_argument("--tetgen-binary", type=Path, default=DEFAULT_TETGEN)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing attempt directory: {args.output}")
    start = time.monotonic()
    collar = np.load(args.collar, allow_pickle=False)
    outer = load_surface(args.outer, "O", collar)
    inner = load_surface(args.inner, "I")
    deep = load_surface(args.deep, "D")
    wall_rows = read_wall_provenance(collar, args.wall_csv, args.parent_cells,
                                     args.parent_nodes)
    if "face_source_identity_schema_by_original_id_json" in outer["raw_npz_keys"]:
        # Parent wall provenance is checked against its untouched original packet
        # above. Rebinding changes only role-local indices, never donor geometry.
        if args.collar_native_placement is not None:
            collar = place_verified_donor_collar(collar,args.collar_native_placement)
        collar = rebind_typed_collar_outer(collar, outer)
    elif args.collar_native_placement is not None:
        raise ValueError("native collar placement requires the typed authoring domain")
    skin, fat, assembly = assemble_shells(outer, inner, deep, collar, wall_rows)
    skin_holes: list[np.ndarray] = []
    fat_holes: list[np.ndarray] = []
    regions: dict[str, np.ndarray] = {}
    if args.inner_hole_core or args.deep_hole_core:
        if not args.inner_hole_core or not args.deep_hole_core:
            raise ValueError("both I and D positive core surfaces are required for .poly hole points")
        skin_holes = _core_holes(args.inner_hole_core)
        fat_holes = _core_holes(args.deep_hole_core)
        centroids = _parent_centroids_by_material(args.parent_nodes, args.parent_cells)
        regions = {"skin": _region_point(skin, 15, centroids),
                   "fat": _region_point(fat, 16, centroids)}
    elif args.run_tetgen:
        raise ValueError("--run-tetgen requires positive I/D core NPZs for PLC holes")
    if args.run_tetgen and not regions:
        raise ValueError("--run-tetgen requires core surfaces so region points can be pinned")
    # Do not create the output directory until all source seam and ancestry gates pass.
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing attempt directory: {args.output}")
    args.output.mkdir(parents=True)
    write_shell_npz(args.output / "skin-shell.npz", skin, "skin")
    write_shell_npz(args.output / "fat-shell.npz", fat, "fat")
    write_face_ancestry_csv(args.output / "skin-face-ancestry.csv.gz", skin)
    write_face_ancestry_csv(args.output / "fat-face-ancestry.csv.gz", fat)
    poly_paths: dict[str, Path] = {}
    if regions:
        poly_paths = {"skin": args.output / "skin.poly", "fat": args.output / "fat.poly"}
        write_poly(poly_paths["skin"], skin["points_world_m"], skin["faces"],
                   skin["face_marker"], skin_holes, [(regions["skin"], 15)])
        write_poly(poly_paths["fat"], fat["points_world_m"], fat["faces"],
                   fat["face_marker"], fat_holes, [(regions["fat"], 16)])
    tetgen_results = None
    if args.run_tetgen:
        tetgen_results = [
            run_tetgen(poly_paths["skin"], args.tetgen_binary, args.output, "skin", 15),
            run_tetgen(poly_paths["fat"], args.tetgen_binary, args.output, "fat", 16),
        ]
    combined_result = None
    if args.run_tetgen and all(result.get("status") == "POSITIVE_F32_CELLS_NOT_PHYSICAL_ADMISSION"
                               for result in tetgen_results):
        try:
            combined_result = combine_tetgen_children(
                args.output / "skin-positive.node", args.output / "skin-positive.ele",
                args.output / "fat-positive.node", args.output / "fat-positive.ele",
                skin, fat, inner, args.output)
        except Exception as error:
            combined_result = {"status": "COMBINED_CHILD_EXPORT_REJECTED",
                               "reason": str(error), "physical_admission": False}
    report = {
        "classification": "existing-format native domain adapter; no physical admission",
        "collar_native_placement": None if args.collar_native_placement is None else {
            "path":str(args.collar_native_placement),"sha256":sha256(args.collar_native_placement),
            "original_wall_provenance_checked_before_placement":True},
        "inputs": {str(p): sha256(p) for p in (
            args.outer, args.inner, args.deep, args.collar, args.wall_csv,
            args.parent_cells, args.parent_nodes)},
        "source_npz_keys": {"O": outer["raw_npz_keys"], "I": inner["raw_npz_keys"],
                            "D": deep["raw_npz_keys"], "collar": list(collar.files)},
        "D37_positive_wall_parent_tets": {
            "skin_rows": wall_rows["skin_parent_tets"].tolist(),
            "fat_rows": wall_rows["fat_parent_tets"].tolist(),
            "all_positive_reference_volume": True},
        "assembly": assembly,
        "material_region_attributes": {"skin": 15, "fat": 16},
        "face_ancestry_files": ["skin-face-ancestry.csv.gz", "fat-face-ancestry.csv.gz"],
        "tetgen": tetgen_results,
        "combined_skin_fat_child_mesh": combined_result,
        "physical_admission": False,
        "elapsed_s": time.monotonic() - start,
        "outputs": {p.name: sha256(p) for p in args.output.iterdir() if p.is_file()},
    }
    (args.output / "adapter-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "SURFACES_AND_PLC_READY" if regions else "SURFACES_READY",
                      "output": str(args.output), "shell_faces": {
                          "skin": len(skin["faces"]), "fat": len(fat["faces"])},
                      "tetgen": tetgen_results, "combined_skin_fat_child_mesh": combined_result,
                      "physical_admission": False,
                      "elapsed_s": report["elapsed_s"]}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
