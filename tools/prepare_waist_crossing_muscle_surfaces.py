"""Build a same-ABI visual NHTISS4 candidate split at the waist owner field.

This only prepares passive source geometry. It does not alter muscle routes,
solver state, anatomy ownership, or implement physical fracture.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

EXPECTED_SELECTED = {67: "right external oblique", 68: "left external oblique"}

HEADER = struct.Struct("<8s6I32s")
RECORD = struct.Struct("<8I")
BINDING = struct.Struct("<I3f4ff")
VERTEX = struct.Struct("<6f4I4f")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decode_nhtiss4(raw: bytes):
    if len(raw) < HEADER.size:
        raise ValueError("NHTISS payload is shorter than its header")
    header = HEADER.unpack_from(raw)
    magic, abi, tissue_count, binding_count, vertex_count, index_count, registration_fp, source_sha = header
    if magic != b"NHTISS4\0" or abi != 5:
        raise ValueError("unsupported NHTISS4 ABI")
    records_offset = HEADER.size
    bindings_offset = records_offset + tissue_count * RECORD.size
    vertices_offset = bindings_offset + binding_count * BINDING.size
    indices_offset = vertices_offset + vertex_count * VERTEX.size
    end_offset = indices_offset + index_count * 4
    if end_offset != len(raw):
        raise ValueError("NHTISS payload length differs from declared extents")
    records = [RECORD.unpack_from(raw, records_offset + i * RECORD.size)
               for i in range(tissue_count)]
    bindings = raw[bindings_offset:vertices_offset]
    indices = np.frombuffer(raw, dtype="<u4", count=index_count, offset=indices_offset)
    return header, records, bindings, vertices_offset, indices


def verify_source_preservation(source_raw: bytes, candidate_raw: bytes,
                               split_stable_ids: set[int]):
    """Verify record identity and actual byte preservation outside split rows."""
    src_header, src_records, src_bindings, src_vertices_offset, src_indices = _decode_nhtiss4(source_raw)
    out_header, out_records, out_bindings, out_vertices_offset, out_indices = _decode_nhtiss4(candidate_raw)
    if src_header[:4] != out_header[:4] or src_header[6:] != out_header[6:]:
        raise ValueError("candidate changed NHTISS identity, binding count, or registration")
    if src_bindings != out_bindings:
        raise ValueError("candidate changed the source binding table")
    if len(src_records) != len(out_records):
        raise ValueError("candidate changed tissue record count")
    src_vertex_count, src_index_count = src_header[4], src_header[5]
    out_vertex_count, out_index_count = out_header[4], out_header[5]
    if len(src_indices) != src_index_count or len(out_indices) != out_index_count:
        raise ValueError("candidate index count differs from header")
    source_ids = {int(row[6]) for row in src_records}
    if not split_stable_ids or not split_stable_ids.issubset(source_ids):
        raise ValueError("split stable IDs are empty or absent from source")
    byte_exact_rows = []
    for source_record, output_record in zip(src_records, out_records):
        sf, sb, sv, sn, si, sc, sid, sl = map(int, source_record)
        of, ob, ov, on, oi, oc, oid, ol = map(int, output_record)
        if (sid, sl, sf, sb) != (oid, ol, of, ob):
            raise ValueError(f"candidate changed stable identity or binding range for row {sid}")
        if sf + sb > src_header[3] or of + ob > out_header[3]:
            raise ValueError(f"candidate tissue row {sid} has an invalid binding range")
        if sv + sn > src_vertex_count or si + sc > src_index_count:
            raise ValueError(f"source tissue row {sid} exceeds its arrays")
        if ov + on > out_vertex_count or oi + oc > out_index_count:
            raise ValueError(f"candidate tissue row {sid} exceeds its arrays")
        source_local = src_indices[si:si + sc].astype(np.int64) - sv
        output_local = out_indices[oi:oi + oc].astype(np.int64) - ov
        if np.any(source_local < 0) or np.any(source_local >= sn):
            raise ValueError(f"source tissue row {sid} has an escaping index")
        if np.any(output_local < 0) or np.any(output_local >= on):
            raise ValueError(f"candidate tissue row {sid} has an escaping index")
        if sid not in split_stable_ids:
            if sn != on or sc != oc:
                raise ValueError(f"candidate changed extent for unsplit tissue row {sid}")
            source_start = src_vertices_offset + sv * VERTEX.size
            output_start = out_vertices_offset + ov * VERTEX.size
            source_bytes = source_raw[source_start:source_start + sn * VERTEX.size]
            output_bytes = candidate_raw[output_start:output_start + on * VERTEX.size]
            if source_bytes != output_bytes:
                raise ValueError(f"candidate changed vertex bytes for unsplit tissue row {sid}")
            if not np.array_equal(source_local, output_local):
                raise ValueError(f"candidate changed local triangle indices for unsplit tissue row {sid}")
            byte_exact_rows.append(sid)
    return {
        "source_record_count": len(src_records),
        "candidate_record_count": len(out_records),
        "split_stable_ids": sorted(split_stable_ids),
        "unsplit_geometry_attribute_bytes_exact": True,
        "unsplit_local_triangle_indices_exact": True,
        "byte_checked_unsplit_stable_ids": byte_exact_rows,
        "binding_table_bytes_exact": True,
    }


def rotation_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = map(float, quaternion)
    return np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def vertex_world_and_normal(source_rows: np.ndarray, weights: np.ndarray,
                            binding_rows: list[tuple], rigid: dict):
    positions = source_rows[:, :3].astype(np.float64)
    source_normals = source_rows[:, 3:6].astype(np.float64)
    world = np.zeros_like(positions)
    world_normal = np.zeros_like(positions)
    transforms = []
    normal_transforms = []
    for binding in binding_rows:
        body = int(binding[0])
        translation = np.asarray(binding[1:4], dtype=np.float64)
        quaternion = np.asarray(binding[4:8], dtype=np.float64)
        scale = float(binding[8])
        pose = rigid["core_poses"].get(int(body))
        if pose is None:
            raise ValueError(f"NHTISS binding body {body} has no rigid source pose")
        body_rotation = rotation_xyzw(np.asarray(pose[3:7], dtype=np.float64))
        bind_rotation = rotation_xyzw(np.asarray(quaternion, dtype=np.float64))
        bind_rotation = body_rotation @ bind_rotation
        offset = np.asarray(pose[:3], dtype=np.float64) + body_rotation @ np.asarray(translation, dtype=np.float64)
        matrix = float(scale) * bind_rotation
        transforms.append((matrix, offset))
        normal_transforms.append(bind_rotation)
        world += weights[:, len(transforms)-1, None] * (positions @ matrix.T + offset)
        world_normal += weights[:, len(transforms)-1, None] * (source_normals @ bind_rotation.T)
    lengths = np.linalg.norm(world_normal, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1.0e-12):
        raise ValueError("source-weighted rest normal is degenerate")
    world_normal /= lengths[:, None]
    return world, world_normal, transforms, normal_transforms


def solve_side_binding(target: np.ndarray, target_normal: np.ndarray,
                       dense_weights: np.ndarray, transforms: list[tuple],
                       normal_transforms: list[np.ndarray]):
    weights = np.asarray(dense_weights, dtype=np.float64).copy()
    totals = weights.sum(axis=1)
    if np.any(~np.isfinite(weights)) or np.any(weights < -1.0e-10) or np.any(totals <= 1.0e-10):
        raise ValueError("split vertex has no positive side-owned binding weight")
    weights /= totals[:, None]
    count = np.count_nonzero(weights > 1.0e-9, axis=1)
    if np.max(count) > 4:
        raise ValueError("split vertex exceeds NHTISS four-influence ABI")
    matrices = np.stack([row[0] for row in transforms])
    offsets = np.stack([row[1] for row in transforms])
    normal_matrices = np.stack(normal_transforms)
    active_ids = np.argmax(weights, axis=1)
    active_count = np.all(count == 1)
    local = np.empty_like(target)
    local_normal = np.empty_like(target_normal)
    max_position_error = 0.0
    max_normal_angle_error = 0.0
    if active_count:
        for binding_index in np.unique(active_ids):
            rows = np.flatnonzero(active_ids == binding_index)
            matrix = matrices[binding_index]
            normal_matrix = normal_matrices[binding_index]
            cond = float(np.linalg.cond(matrix))
            normal_cond = float(np.linalg.cond(normal_matrix))
            if not np.isfinite(cond) or cond > 1.0e10 or not np.isfinite(normal_cond) or normal_cond > 1.0e10:
                raise ValueError("single-side source binding transform is singular")
            local[rows] = np.linalg.solve(matrix, (target[rows] - offsets[binding_index]).T).T
            local_normal[rows] = np.linalg.solve(normal_matrix, target_normal[rows].T).T
    else:
        matrices_by_vertex = np.einsum("nb,bij->nij", weights, matrices)
        offsets_by_vertex = weights @ offsets
        normals_by_vertex = np.einsum("nb,bij->nij", weights, normal_matrices)
        for index in range(len(target)):
            if np.linalg.cond(matrices_by_vertex[index]) > 1.0e10 or np.linalg.cond(normals_by_vertex[index]) > 1.0e10:
                raise ValueError("side-filtered source binding blend is singular")
            local[index] = np.linalg.solve(matrices_by_vertex[index], target[index] - offsets_by_vertex[index])
            local_normal[index] = np.linalg.solve(normals_by_vertex[index], target_normal[index])
    local_normal_lengths = np.linalg.norm(local_normal, axis=1)
    if np.any(~np.isfinite(local)) or np.any(~np.isfinite(local_normal_lengths)) or np.any(local_normal_lengths <= 1e-12):
        raise ValueError("side-filtered local source transform produced invalid rows")
    local_normal /= local_normal_lengths[:, None]
    # Round through source float32 storage, then confirm the split reconstructs
    # the exact registered rest point and target normal with its side bindings.
    local32 = local.astype("<f4").astype(np.float64)
    normal32 = local_normal.astype("<f4").astype(np.float64)
    for row in range(len(target)):
        active = np.flatnonzero(weights[row] > 1.0e-9)
        restored = np.zeros(3, dtype=np.float64)
        restored_normal = np.zeros(3, dtype=np.float64)
        for binding_index in active:
            w = weights[row, binding_index]
            restored += w * (matrices[binding_index] @ local32[row] + offsets[binding_index])
            restored_normal += w * (normal_matrices[binding_index] @ normal32[row])
        max_position_error = max(max_position_error, float(np.linalg.norm(restored - target[row])))
        restored_normal /= max(float(np.linalg.norm(restored_normal)), 1e-30)
        max_normal_angle_error = max(max_normal_angle_error,
            float(np.linalg.norm(restored_normal - target_normal[row])))
    if max_position_error > 1.0e-4 or max_normal_angle_error > 2.0e-3:
        raise ValueError(f"side reconstruction exceeds gate (position {max_position_error:g}, normal {max_normal_angle_error:g})")
    return local32, normal32, weights, max_position_error, max_normal_angle_error


def sparse_vertex_rows(local_pos: np.ndarray, local_normal: np.ndarray,
                       dense_weights: np.ndarray) -> bytes:
    out = bytearray()
    for position, normal, row in zip(local_pos, local_normal, dense_weights, strict=True):
        order = sorted((int(index) for index in np.flatnonzero(row > 1.0e-8)),
                       key=lambda index: (-float(row[index]), index))
        if not order or len(order) > 4:
            raise ValueError("NHTISS sparse row cannot encode filtered source weights")
        total = float(sum(row[index] for index in order))
        indices = order + [0] * (4 - len(order))
        lane_weights = [float(row[index] / total) for index in order] + [0.0] * (4 - len(order))
        sums = sum(lane_weights)
        if abs(sums - 1.0) > 2e-3:
            raise ValueError("NHTISS emitted sparse binding weights do not sum to one")
        values = [*map(float, position), *map(float, normal), *indices, *lane_weights]
        out.extend(VERTEX.pack(*values))
    return bytes(out)


def lineage_arrays(stable_id: int, source_first_vertex: int,
                   source_first_index: int, halves):
    blocks = {}
    for side_name, half in halves:
        n = len(half.vertex_provenance)
        source_vertex = np.full(n, -1, dtype=np.int64)
        edge_a = np.full(n, -1, dtype=np.int64)
        edge_b = np.full(n, -1, dtype=np.int64)
        edge_t = np.full(n, np.nan, dtype=np.float64)
        for i, provenance in enumerate(half.vertex_provenance):
            if provenance["kind"] == "source_vertex":
                source_vertex[i] = source_first_vertex + int(provenance["source_vertex"])
            elif provenance["kind"] == "source_edge":
                a, b = map(int, provenance["source_edge"])
                edge_a[i], edge_b[i] = source_first_vertex + a, source_first_vertex + b
                edge_t[i] = float(provenance["t_from_first"])
            else:
                raise ValueError(f"unknown split vertex ancestry: {provenance!r}")
        blocks[f"stable_{stable_id}_{side_name}_source_vertex"] = source_vertex
        blocks[f"stable_{stable_id}_{side_name}_edge_a"] = edge_a
        blocks[f"stable_{stable_id}_{side_name}_edge_b"] = edge_b
        blocks[f"stable_{stable_id}_{side_name}_edge_t"] = edge_t
        blocks[f"stable_{stable_id}_{side_name}_triangles"] = np.asarray(half.triangles, dtype=np.int64)
        faces = np.asarray(half.triangle_source_face, dtype=np.int64)
        blocks[f"stable_{stable_id}_{side_name}_source_face"] = np.where(
            faces >= 0, source_first_index // 3 + faces, -1)
        blocks[f"stable_{stable_id}_{side_name}_triangle_kind"] = np.asarray(half.triangle_kind, dtype=np.uint8)
        blocks[f"stable_{stable_id}_{side_name}_cap_loop"] = np.asarray(half.triangle_cap_loop, dtype=np.int32)
    return blocks


def edge_uses(faces: np.ndarray):
    rows: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_index, face in enumerate(faces):
        a, b, c = map(int, face)
        for x, y in ((a, b), (b, c), (c, a)):
            edge = (x, y) if x < y else (y, x)
            rows.setdefault(edge, []).append((face_index, x, y))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-nhtissue", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--nhrigid", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--output-nhtissue", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-rigid-sha256", required=True)
    args = parser.parse_args()

    INPUT, MANIFEST, RIGID_PATH = args.input_nhtissue, args.input_manifest, args.nhrigid
    INVENTORY, RECEIPT_SOURCE = args.inventory, args.input_receipt
    OUTPUT, RECEIPT_OUTPUT = args.output_nhtissue, args.output_receipt
    LINEAGE, REPORT = args.lineage, args.report
    EXPECTED_INPUT_SHA, EXPECTED_RIGID_SHA = args.expected_input_sha256.lower(), args.expected_rigid_sha256.lower()
    for path in (OUTPUT, RECEIPT_OUTPUT, LINEAGE, REPORT):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite existing output: {path}")
    if len(EXPECTED_INPUT_SHA) != 64 or len(EXPECTED_RIGID_SHA) != 64:
        raise RuntimeError("expected source identities must be SHA-256 hex strings")

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from numilab_human import waist_skin_partition
    from numilab_human.resting_scene import load_rigid

    split_surface_by_signed_field = waist_skin_partition.split_surface_by_signed_field
    CUT_HELPER = Path(waist_skin_partition.__file__)
    for path, expected in ((INPUT, EXPECTED_INPUT_SHA), (RIGID_PATH, EXPECTED_RIGID_SHA)):
        if sha(path) != expected:
            raise RuntimeError(f"pinned source identity mismatch: {path}")
    rigid = load_rigid(RIGID_PATH)
    manifest = json.loads(MANIFEST.read_text())
    inventory = json.loads(INVENTORY.read_text())
    surface_manifest = {int(row["stable_id"]): row for row in manifest["source"]["surfaces"]}
    child_bodies = set(map(int, inventory["partition"]["child_side_body_indices"]))
    parent_bodies = set(map(int, inventory["partition"]["parent_side_body_indices"]))
    if child_bodies & parent_bodies or child_bodies | parent_bodies != set(range(157)):
        raise RuntimeError("pinned body partition is incomplete or overlapping")

    raw = INPUT.read_bytes()
    magic, abi, tissue_count, binding_count, vertex_count, index_count, registration_fp, source_sha = HEADER.unpack_from(raw)
    if magic != b"NHTISS4\0" or abi != 5 or source_sha.hex() != rigid["source_sha256"]:
        raise RuntimeError("input NHTISS4 header does not match pinned rigid source")
    if tissue_count != 164 or binding_count != 595 or vertex_count != 467552 or index_count != 2536842:
        raise RuntimeError("pinned NHTISS dimensions changed")
    offset = HEADER.size
    source_records = [RECORD.unpack_from(raw, offset + RECORD.size * i) for i in range(tissue_count)]
    offset += tissue_count * RECORD.size
    source_bindings = [BINDING.unpack_from(raw, offset + BINDING.size * i) for i in range(binding_count)]
    offset += binding_count * BINDING.size
    vertex_offset = offset
    offset += vertex_count * VERTEX.size
    source_indices = np.frombuffer(raw, dtype="<u4", count=index_count, offset=offset).copy()
    if offset + index_count * 4 != len(raw):
        raise RuntimeError("pinned NHTISS4 extent differs from file length")

    out_records = []
    out_vertex_bytes = bytearray()
    out_indices: list[int] = []
    lineage = {}
    selected_rows = []
    unchanged_exact = True
    max_rest_error = 0.0
    max_normal_error = 0.0
    source_vertex_count = 0
    output_vertex_count = 0
    output_face_count = 0
    split_face_count = 0
    cap_face_count = 0

    for record_index, record in enumerate(source_records):
        first_binding, bind_count, first_vertex, v_count, first_index, i_count, stable_id, layer = record
        if first_binding + bind_count > binding_count or first_vertex + v_count > vertex_count or first_index + i_count > index_count:
            raise RuntimeError(f"source tissue record {stable_id} exceeds an array extent")
        vertex_dtype = np.dtype({
            "names": ["position", "normal", "binding_index", "weight"],
            "formats": [("<f4", (3,)), ("<f4", (3,)), ("<u4", (4,)), ("<f4", (4,))],
            "offsets": [0, 12, 24, 40],
            "itemsize": VERTEX.size,
        })
        source_vertex_rows = np.frombuffer(
            raw, dtype=vertex_dtype, count=v_count,
            offset=vertex_offset + first_vertex * VERTEX.size)
        source_rows = np.column_stack((source_vertex_rows["position"],
                                       source_vertex_rows["normal"])).astype(np.float64)
        source_sparse = source_vertex_rows["binding_index"]
        source_weights = source_vertex_rows["weight"].astype(np.float64)
        dense = np.zeros((v_count, bind_count), dtype=np.float64)
        for lane in range(4):
            active = source_weights[:, lane] > 2.0e-6
            if np.any(active & (source_sparse[:, lane] >= bind_count)):
                raise RuntimeError(f"source tissue {stable_id} has an active binding index outside its record")
            rows = np.flatnonzero(active)
            dense[rows, source_sparse[rows, lane]] += source_weights[rows, lane]
        weight_sum = dense.sum(axis=1)
        if np.any(~np.isfinite(dense)) or np.any(np.abs(weight_sum - 1.0) > 2.0e-3):
            raise RuntimeError(f"source tissue {stable_id} has malformed binding weights")
        bindings = source_bindings[first_binding:first_binding + bind_count]
        owner_bodies = [int(binding[0]) for binding in bindings]
        if any(body not in child_bodies and body not in parent_bodies for body in owner_bodies):
            if layer == 1:
                raise RuntimeError(f"muscle stableId {stable_id} has an unclassified source binding")
        child_columns = np.asarray([body in child_bodies for body in owner_bodies], dtype=bool)
        parent_columns = np.asarray([body in parent_bodies for body in owner_bodies], dtype=bool)
        child_support = dense[:, child_columns].sum(axis=1) if np.any(child_columns) else np.zeros(v_count)
        parent_support = dense[:, parent_columns].sum(axis=1) if np.any(parent_columns) else np.zeros(v_count)
        field = child_support - parent_support
        should_test = layer == 1 and np.any(field < -1.0e-8) and np.any(field > 1.0e-8)
        if should_test:
            if bind_count != 2 or set(owner_bodies) != {7, 20}:
                raise RuntimeError(f"unexpected crossing muscle ownership at stableId {stable_id}: {owner_bodies}")
            triangles_global = source_indices[first_index:first_index + i_count].astype(np.int64)
            if np.any(triangles_global < first_vertex) or np.any(triangles_global >= first_vertex + v_count):
                raise RuntimeError(f"source tissue {stable_id} has an index escaping its allocation")
            faces = (triangles_global.reshape(-1, 3) - first_vertex)
            face_edge_uses = edge_uses(faces)
            if any(len(uses) not in (1, 2) for uses in face_edge_uses.values()):
                raise RuntimeError(f"source tissue {stable_id} contains nonmanifold edges")
            existing_boundary = {edge for edge, uses in face_edge_uses.items() if len(uses) == 1}
            world, world_normals, transforms, normal_transforms = vertex_world_and_normal(
                source_rows, dense, bindings, rigid)
            split_plus, split_minus = split_surface_by_signed_field(
                world, faces, field,
                attributes={"full_binding_weights": dense,
                            "registered_rest_normals": world_normals},
                allowed_existing_boundary_edges=existing_boundary,
                tolerance=1e-11,
            )
            side_results = []
            record_out_vertices = bytearray()
            record_out_indices: list[int] = []
            record_lineage_halves = []
            record_rest_error = 0.0
            record_normal_error = 0.0
            for side_name, half, side_mask in (
                ("child", split_plus, child_columns),
                ("parent", split_minus, parent_columns),
            ):
                if not np.any(side_mask):
                    raise RuntimeError(f"crossing tissue {stable_id} has no {side_name}-side bindings")
                filtered = np.asarray(half.attributes["full_binding_weights"], dtype=np.float64).copy()
                filtered[:, ~side_mask] = 0.0
                totals = filtered.sum(axis=1)
                if np.any(~np.isfinite(totals)) or np.any(totals <= 1e-10):
                    raise RuntimeError(f"split tissue {stable_id} has unsupported {side_name}-side vertices")
                filtered /= totals[:, None]
                target_normals = np.asarray(half.attributes["registered_rest_normals"], dtype=np.float64).copy()
                # Give cap vertices the outward, geometry-derived cap normal;
                # the source boundary vertices are duplicated by the two halves.
                for loop_index in np.unique(half.triangle_cap_loop[half.triangle_cap_loop >= 0]):
                    cap_faces = half.triangles[half.triangle_cap_loop == loop_index]
                    normal_accum = np.zeros_like(target_normals)
                    for a, b, c in cap_faces:
                        cross = np.cross(half.vertices[b] - half.vertices[a], half.vertices[c] - half.vertices[a])
                        normal_accum[a] += cross
                        normal_accum[b] += cross
                        normal_accum[c] += cross
                    cap_ids = np.flatnonzero(np.linalg.norm(normal_accum, axis=1) > 1e-12)
                    target_normals[cap_ids] = normal_accum[cap_ids] / np.linalg.norm(normal_accum[cap_ids], axis=1)[:, None]
                local_pos, local_norm, filtered, pos_error, normal_error = solve_side_binding(
                    half.vertices, target_normals, filtered, transforms, normal_transforms)
                record_rest_error = max(record_rest_error, pos_error)
                record_normal_error = max(record_normal_error, normal_error)
                record_vbase = len(record_out_vertices) // VERTEX.size
                record_out_vertices.extend(sparse_vertex_rows(local_pos, local_norm, filtered))
                local_faces = np.asarray(half.triangles, dtype=np.int64) + record_vbase
                record_out_indices.extend(local_faces.reshape(-1).astype(np.int64).tolist())
                side_results.append({
                    "side": side_name,
                    "vertex_count": int(len(half.vertices)),
                    "triangle_count": int(len(half.triangles)),
                    "source_face_count": int(np.count_nonzero(half.triangle_kind == 0)),
                    "cap_triangle_count": int(np.count_nonzero(half.triangle_kind == 1)),
                    "cut_boundary_edge_count": int(sum(1 for a, b in half.boundary_edges)),
                    "cap_loop_count": int(len(half.cap_loop_projection_residuals)),
                    "cap_loop_projection_residuals_m": np.asarray(half.cap_loop_projection_residuals, dtype=np.float64).tolist(),
                    "side_weight_row_sum_max_error": float(np.max(np.abs(filtered.sum(axis=1) - 1.0))),
                    "source_bindings_retained": [owner_bodies[i] for i in np.flatnonzero(side_mask)],
                })
                record_lineage_halves.append((side_name, half))
            if record_rest_error > 1.0e-4 or record_normal_error > 2.0e-3:
                raise RuntimeError(f"surface {stable_id} exceeds split rest-state reconstruction tolerance")
            record_vertex_count = len(record_out_vertices) // VERTEX.size
            record_index_count = len(record_out_indices)
            if record_vertex_count > 1_000_000 or record_index_count > 6_000_000:
                raise RuntimeError(f"split tissue {stable_id} exceeds supported source payload sizes")
            new_first_vertex = len(out_vertex_bytes) // VERTEX.size
            new_first_index = len(out_indices)
            # The original source record remains one row with the same stable
            # identity and binding array; both disconnected halves and their
            # caps are one visual primitive under the existing material.
            out_records.append((first_binding, bind_count, new_first_vertex,
                               record_vertex_count, new_first_index,
                               record_index_count, stable_id, layer))
            out_vertex_bytes.extend(record_out_vertices)
            out_indices.extend((new_first_vertex + i) for i in record_out_indices)
            lineage.update(lineage_arrays(stable_id, first_vertex, first_index, record_lineage_halves))
            source_vertex_count += v_count
            output_vertex_count += record_vertex_count
            output_face_count += record_index_count // 3
            split_face_count += i_count // 3
            cap_face_count += sum(row["cap_triangle_count"] for row in side_results)
            source_meta = surface_manifest.get(stable_id, {})
            if source_meta.get("label") != EXPECTED_SELECTED[stable_id]:
                raise RuntimeError(f"stableId {stable_id} manifest label changed")
            matched_muscles = source_meta.get("matched_muscles", [])
            selected_rows.append({
                "stable_id": stable_id,
                "label": source_meta.get("label"),
                "member_id": source_meta.get("member_id"),
                "source_actuator_indices": [int(row["source_actuator_index"]) for row in matched_muscles],
                "source_owner_body_indices": owner_bodies,
                "source_vertex_count": int(v_count),
                "source_triangle_count": int(i_count // 3),
                "source_scalar_min_max": [float(np.min(field)), float(np.max(field))],
                "source_crossing_triangle_count": int(sum(
                    np.any(np.min(field[face]) < 0) and np.any(np.max(field[face]) > 0)
                    for face in faces)),
                "source_boundary_edge_count": int(len(existing_boundary)),
                "sides": side_results,
                "output_vertex_count": int(record_vertex_count),
                "output_triangle_count": int(record_index_count // 3),
                "output_rest_position_max_error_m": float(record_rest_error),
                "output_rest_normal_max_vector_error": float(record_normal_error),
                "source_material_layer_preserved": layer == 1,
                "source_stable_id_preserved": True,
                "source_geometry_owner_scalar": "sum(child-side authored binding weights) - sum(parent-side authored binding weights)",
            })
        else:
            new_first_vertex = len(out_vertex_bytes) // VERTEX.size
            new_first_index = len(out_indices)
            new_indices = source_indices[first_index:first_index + i_count].astype(np.uint64)
            new_indices = new_indices - int(first_vertex) + new_first_vertex
            out_records.append((first_binding, bind_count, new_first_vertex, v_count,
                               new_first_index, i_count, stable_id, layer))
            start = vertex_offset + first_vertex * VERTEX.size
            source_vertex_bytes = raw[start:start + v_count * VERTEX.size]
            appended_start = new_first_vertex * VERTEX.size
            out_vertex_bytes.extend(source_vertex_bytes)
            out_indices.extend(map(int, new_indices.tolist()))
            unchanged_exact = unchanged_exact and out_vertex_bytes[appended_start:appended_start + len(source_vertex_bytes)] == source_vertex_bytes
            source_vertex_count += v_count
            output_vertex_count += v_count
            output_face_count += i_count // 3

    selected_ids = {int(row["stable_id"]) for row in selected_rows}
    if selected_ids != set(EXPECTED_SELECTED):
        raise RuntimeError(f"unexpected crossing source surface set: {sorted(selected_ids)}")
    if len(out_records) != tissue_count or len(out_vertex_bytes) // VERTEX.size != output_vertex_count:
        raise RuntimeError("derived NHTISS record/vertex coverage differs")
    if len(out_indices) != sum(row[5] for row in out_records):
        raise RuntimeError("derived NHTISS record/index coverage differs")

    new_vertex_count = len(out_vertex_bytes) // VERTEX.size
    new_index_count = len(out_indices)
    out = bytearray(HEADER.pack(magic, abi, tissue_count, binding_count,
                                new_vertex_count, new_index_count,
                                registration_fp, source_sha))
    out.extend(b"".join(RECORD.pack(*row) for row in out_records))
    # Original binding table and transforms are retained byte-exact.
    binding_offset = HEADER.size + tissue_count * RECORD.size
    out.extend(raw[binding_offset:binding_offset + binding_count * BINDING.size])
    out.extend(out_vertex_bytes)
    out.extend(np.asarray(out_indices, dtype="<u4").tobytes())
    OUTPUT.write_bytes(out)
    np.savez_compressed(LINEAGE, **lineage)

    byte_contract = verify_source_preservation(raw, bytes(out), selected_ids)
    candidate_sha = hashlib.sha256(out).hexdigest()
    source_receipt_text = RECEIPT_SOURCE.read_text()
    source_receipt = json.loads(source_receipt_text)
    if source_receipt.get("passive_tissue_parent_bindings", {}).get("child_payload_sha256") != EXPECTED_INPUT_SHA:
        raise RuntimeError("source receipt does not identify the pinned input NHTISS payload")
    old_receipt_field = f'"child_payload_sha256": "{EXPECTED_INPUT_SHA}"'
    if source_receipt_text.count(old_receipt_field) != 1:
        raise RuntimeError("source receipt target child hash is not unique")
    derived_receipt_text = source_receipt_text.replace(
        old_receipt_field, f'"child_payload_sha256": "{candidate_sha}"', 1)
    derived_receipt = json.loads(derived_receipt_text)
    if derived_receipt.get("passive_tissue_parent_bindings", {}).get("child_payload_sha256") != candidate_sha:
        raise RuntimeError("derived receipt did not bind the candidate NHTISS payload")
    derived_receipt["passive_tissue_parent_bindings"]["child_payload_sha256"] = EXPECTED_INPUT_SHA
    if derived_receipt != source_receipt:
        raise RuntimeError("derived receipt changed fields outside the targeted child payload hash")
    RECEIPT_OUTPUT.write_text(derived_receipt_text)

    report = {
        "schema": "numi.human.waist-crossing-muscle-surface-split.v1",
        "status": "source-local_same-ABI_visual_candidate_not_native-integrated_or_physical-fracture",
        "inputs": {
            "nhtissue": str(INPUT),
            "nhtissue_sha256": EXPECTED_INPUT_SHA,
            "resting_anatomy_receipt": str(RECEIPT_SOURCE),
            "resting_anatomy_receipt_sha256": sha(RECEIPT_SOURCE),
            "nhrigid": str(RIGID_PATH),
            "nhrigid_sha256": EXPECTED_RIGID_SHA,
            "inventory": str(INVENTORY),
            "inventory_sha256": sha(INVENTORY),
            "source_manifest": str(MANIFEST),
            "source_manifest_sha256": sha(MANIFEST),
            "splitter": str(CUT_HELPER),
            "splitter_sha256": sha(CUT_HELPER),
            "source_sha256_in_header": source_sha.hex(),
        },
        "interface": {
            "source": "authored NHTISS per-vertex body-binding ownership relative to the frozen seam partition",
            "signed_field": "sum(child-side authored binding weights) - sum(parent-side authored binding weights)",
            "child_source_body_indices": sorted(child_bodies),
            "parent_source_body_indices": sorted(parent_bodies),
            "clipped_triangles_interpolated_at_source_edges": True,
            "cut_loops_capped_with_source-edge_vertices": True,
            "cap_validation": "NumiHumanPlaneCut.numilab_human.waist_skin_partition.split_surface_by_signed_field topology, winding, cap projection, and retained-boundary gates",
        },
        "selected_source_surfaces": selected_rows,
        "payload": {
            "path": str(OUTPUT),
            "sha256": candidate_sha,
            "bytes": len(out),
            "magic": magic.rstrip(b"\0").decode("ascii"),
            "payload_abi": int(abi),
            "surface_count": len(out_records),
            "vertex_count_before": int(vertex_count),
            "vertex_count_after": int(new_vertex_count),
            "index_count_before": int(index_count),
            "index_count_after": int(new_index_count),
            "binding_table_byte_exact": True,
            "record_count_and_stable_id_order_preserved": True,
        },
        "unchanged_rows": {
            "source_record_count": tissue_count,
            "split_stable_ids": sorted(selected_ids),
            "other_record_geometry_and_attributes_byte_exact": bool(unchanged_exact),
            "other_row_indices_rebased_only_for_appended_vertex_allocations": True,
        },
        "byte_contract": byte_contract,
        "derived_receipt": {
            "path": str(RECEIPT_OUTPUT),
            "sha256": sha(RECEIPT_OUTPUT),
            "bytes": RECEIPT_OUTPUT.stat().st_size,
            "only_changed_field": "passive_tissue_parent_bindings.child_payload_sha256",
            "candidate_child_payload_sha256": candidate_sha,
        },
        "ancestry": {
            "path": str(LINEAGE),
            "sha256": sha(LINEAGE),
            "bytes": LINEAGE.stat().st_size,
            "per-split-vertex_source_vertex_or_source_edge_t": True,
            "per-split-triangle_source_face_or_cap_loop": True,
            "source-triangle-face-index_is_global_nhtissue_index_order": True,
        },
        "scope": "The source geometry remains passive. This candidate only prevents two crossing muscle surfaces from visually bridging the cut; it does not split muscle physics or add contact/cohesion.",
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
