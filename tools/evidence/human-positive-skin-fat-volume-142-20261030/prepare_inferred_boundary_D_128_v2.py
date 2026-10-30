#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import hashlib, json, math
from fractions import Fraction
import numpy as np

H = Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
BASE = H / 'fat-inferred-boundary-positive-volume-127-root'
OUT = H / 'fat-inferred-boundary-D-refinement-128-v2'
INPUTS = {
    'volume': BASE / 'fat-positive-inferred-boundary-proposal.npz',
    'shell': BASE / 'fat-shell-inferred-boundary-f32-repair.npz',
    'D': BASE / 'candidate-D-inferred-boundary-f32-repair.npz',
}
EXPECTED = {
    'volume': '23ceea4acb0ab21783ac2bc7856e7389bf232b3a12be6842ef707150c9b8ce07',
    'shell': 'd7cadd6ff833a3b8dea1c272d2434d8b13dd35369dacf1b29db8b7cd26f68030',
    'D': '7201d57205d890bebaea339c9df752184398912f92813cbbb5d41efe754e88c2',
}
TYPED_FIELDS = ('face_source_layer', 'face_source_body', 'face_source_bone',
                'face_source_face_id', 'face_source_original_id',
                'face_source_identity0', 'face_source_identity1')
FACE_CSR = {
    'face_ancestry_offsets': ('face_ancestry_source_face_rows', 'face_ancestry_source_original_ids', 'face_ancestry_parent_face_rows'),
    'candidate_face_ancestry_offsets': ('candidate_face_ancestry_rows',),
    'microedge_parent_input_face_offsets': ('microedge_parent_input_face_rows',),
    'operational_output_face_parent_offsets': ('operational_output_face_parent_rows',),
    'union_face_ancestry_offsets': ('union_face_ancestry_rows', 'union_face_ancestry_run_original_ids', 'union_face_ancestry_face_ids'),
    'quality_star_patch_union_offsets': tuple(f'quality_star_patch_union_{k}' for k in TYPED_FIELDS),
}
POINT_CSR = ('microedge_parent_input_point_offsets', 'output_vertex_input_parent_offsets')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]).copy() for k in z.files}


def csr_union(arrays: dict[str, np.ndarray], offset_key: str, lane_keys: tuple[str, ...], parent_rows: np.ndarray) -> list[tuple[int, ...]]:
    off = np.asarray(arrays[offset_key], np.int64)
    lanes = [np.asarray(arrays[k]) for k in lane_keys]
    records: set[tuple[int, ...]] = set()
    for row in parent_rows:
        a, b = map(int, off[int(row):int(row) + 2])
        records.update(tuple(int(lane[j]) for lane in lanes) for j in range(a, b))
    return sorted(records)


def rebuild_grouped_csr(parent: dict[str, np.ndarray], out_parent: np.ndarray,
                        groups_by_output: dict[int, int], group_parent_rows: list[np.ndarray],
                        offset_key: str, lane_keys: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    src_off = np.asarray(parent[offset_key], np.int64)
    src_lanes = [np.asarray(parent[k]) for k in lane_keys]
    if src_off.shape != (len(parent['faces']) + 1,) or src_off[0] != 0 or np.any(np.diff(src_off) < 0):
        raise ValueError(f'{offset_key} source offsets malformed')
    if any(len(x) != int(src_off[-1]) for x in src_lanes):
        raise ValueError(f'{offset_key} source lanes disagree with offsets')
    counts = np.diff(src_off)[out_parent].astype(np.int64, copy=True)
    group_records: dict[int, list[tuple[int, ...]]] = {}
    for outrow, gid in groups_by_output.items():
        records = csr_union(parent, offset_key, lane_keys, group_parent_rows[gid])
        if not records:
            raise ValueError(f'{offset_key} group {gid} has empty ancestry')
        group_records[gid] = records
        counts[outrow] = len(records)
    out_off = np.empty(len(out_parent) + 1, np.int64)
    out_off[0] = 0
    np.cumsum(counts, out=out_off[1:])
    total = int(out_off[-1])
    out_lanes = [np.empty((total,) + x.shape[1:], dtype=x.dtype) for x in src_lanes]
    child_rows = np.fromiter(groups_by_output.keys(), dtype=np.int64, count=len(groups_by_output))
    child_mask = np.zeros(len(out_parent), bool)
    child_mask[child_rows] = True
    if np.any(~child_mask):
        # Gather all ordinary one-parent CSR records in one vectorized pass.
        rec_parent = np.repeat(out_parent, counts)
        rec_start = np.repeat(out_off[:-1], counts)
        local = np.arange(total, dtype=np.int64) - rec_start
        rec_index = src_off[rec_parent] + local
        normal_record_mask = np.repeat(~child_mask, counts)
        dst = np.flatnonzero(normal_record_mask)
        src = rec_index[normal_record_mask]
        for dst_lane, src_lane in zip(out_lanes, src_lanes):
            dst[...]
            dst_lane[dst] = src_lane[src]
    for outrow, gid in groups_by_output.items():
        a, b = map(int, out_off[outrow:outrow + 2])
        records = group_records[gid]
        for li, lane in enumerate(out_lanes):
            lane[a:b] = np.asarray([r[li] for r in records], dtype=lane.dtype)
    return out_off, {key: value for key, value in zip(lane_keys, out_lanes)}


def point_triangle_projection(p: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
    # Exact common 2^-149 lattice arithmetic for F32 inputs avoids cancellation in
    # the Gram determinant of near-collinear source triangles.
    scale = 1 << 149
    def lattice_row(x):
        vals = []
        for value in np.asarray(x, np.float32):
            numerator, denominator = float(value).as_integer_ratio()
            if scale % denominator:
                raise ValueError('Float32 coordinate is not on the expected exact lattice')
            vals.append(numerator * (scale // denominator))
        return tuple(vals)
    def dot_int(a, b):
        return sum(int(x) * int(y) for x, y in zip(a, b))
    def cross_int(a, b):
        return (int(a[1]) * int(b[2]) - int(a[2]) * int(b[1]),
                int(a[2]) * int(b[0]) - int(a[0]) * int(b[2]),
                int(a[0]) * int(b[1]) - int(a[1]) * int(b[0]))
    A, B, C = (lattice_row(x) for x in np.asarray(tri, np.float32))
    P = lattice_row(p)
    e0 = tuple(B[i] - A[i] for i in range(3))
    e1 = tuple(C[i] - A[i] for i in range(3))
    q = tuple(P[i] - A[i] for i in range(3))
    normal = cross_int(e0, e1)
    normal2 = dot_int(normal, normal)
    if normal2 <= 0:
        raise ValueError('degenerate registered parent D triangle')
    b = Fraction(dot_int(cross_int(q, e1), normal), normal2)
    c = Fraction(dot_int(cross_int(e0, q), normal), normal2)
    bary_f = (Fraction(1) - b - c, b, c)
    bary = np.asarray([float(x) for x in bary_f], np.float64)
    qn = dot_int(q, normal)
    plane2 = Fraction(qn * qn, normal2 * scale * scale)
    plane = math.sqrt(float(plane2))
    projected = tuple(Fraction(A[i]) + b * e0[i] + c * e1[i] for i in range(3))
    if min(bary_f) >= 0:
        closest = projected
        distance2 = plane2
    else:
        candidates = []
        for X, Y in ((A, B), (B, C), (C, A)):
            edge = tuple(Y[i] - X[i] for i in range(3))
            edge2 = dot_int(edge, edge)
            t = Fraction(dot_int(tuple(P[i] - X[i] for i in range(3)), edge), edge2)
            t = min(Fraction(1), max(Fraction(0), t))
            point = tuple(Fraction(X[i]) + t * edge[i] for i in range(3))
            d2 = sum((Fraction(P[i]) - point[i]) ** 2 for i in range(3)) / (scale * scale)
            candidates.append((d2, point))
        distance2, closest = min(candidates, key=lambda item: item[0])
    closest_world = np.asarray([float(x / scale) for x in closest], np.float64)
    return bary, plane, math.sqrt(float(distance2)), closest_world


def main() -> None:
    if OUT.exists():
        raise SystemExit(f'refusing existing output {OUT}')
    hashes = {k: sha(p) for k, p in INPUTS.items()}
    if hashes != EXPECTED:
        raise SystemExit(f'input pins changed: {hashes}')
    data = {k: load(p) for k, p in INPUTS.items()}
    vol, shell, parent = data['volume'], data['shell'], data['D']
    volume_points = np.asarray(vol['points_world_m'], np.float32)
    tet = np.asarray(vol['tetrahedra'], np.int64)
    shell_points = np.asarray(shell['points_world_m'], np.float32)
    shell_faces = np.asarray(shell['faces'], np.int64)
    d_points = np.asarray(parent['points_world_m'], np.float32)
    d_points_f32 = np.asarray(parent.get('points_world_m_f32', d_points), np.float32)
    d_faces = np.asarray(parent['faces'], np.int64)
    n_shell, n_parent, f_parent = len(shell_points), len(d_points), len(d_faces)
    if volume_points.shape[0] < n_shell or not np.array_equal(volume_points[:n_shell], shell_points):
        raise ValueError('candidate127 volume does not preserve the exact authored shell point prefix')
    d_shell_node = np.asarray(shell['registered_D_shell_node_rows'], np.int64)
    d_shell_face = np.asarray(shell['registered_D_shell_face_rows'], np.int64)
    d_registered_faces = np.asarray(shell['registered_D_source_faces'], np.int64)
    if d_shell_node.shape != (n_parent,) or d_shell_face.shape != (f_parent,) or not np.array_equal(d_registered_faces, d_faces):
        raise ValueError('typed D rows do not match the pinned 074 registered-D source row domain')
    if not np.array_equal(shell_points[d_shell_node], d_points):
        raise ValueError('typed D source points differ from the exact registered 074 point mapping')
    if not np.array_equal(np.sort(shell_faces[d_shell_face], axis=1), np.sort(d_shell_node[d_faces], axis=1)):
        raise ValueError('typed D faces differ from the exact registered 074 face mapping')

    # Full exact-topology boundary inventory for candidate127, retaining oriented face rows.
    tet_face_pattern = np.asarray([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], np.int64)
    all_tet_faces = tet[:, tet_face_pattern].reshape((-1, 3))
    face_keys = np.sort(all_tet_faces, axis=1)
    _, inverse, counts = np.unique(face_keys, axis=0, return_inverse=True, return_counts=True)
    boundary_rows = np.flatnonzero(counts[inverse] == 1)
    boundary_faces = all_tet_faces[boundary_rows]
    boundary_keys = np.sort(boundary_faces, axis=1)
    shell_keys_array = np.sort(shell_faces, axis=1)
    shell_key_list = list(map(tuple, shell_keys_array.tolist()))
    boundary_key_list = list(map(tuple, boundary_keys.tolist()))
    shell_key_set = set(shell_key_list)
    boundary_key_set = set(boundary_key_list)
    missing_shell_rows = np.asarray([i for i, key in enumerate(shell_key_list) if key not in boundary_key_set], np.int64)
    extra_boundary_rows = np.asarray([i for i, key in enumerate(boundary_key_list) if key not in shell_key_set], np.int64)
    extra_boundary_faces = boundary_faces[extra_boundary_rows]
    shell_to_d_face = np.full(len(shell_faces), -1, np.int64)
    shell_to_d_face[d_shell_face] = np.arange(f_parent, dtype=np.int64)
    missing_d_faces = shell_to_d_face[missing_shell_rows]
    if (len(missing_shell_rows) != 16 or len(extra_boundary_rows) != 32 or
            np.any(missing_d_faces < 0) or len(np.unique(missing_d_faces)) != 16):
        raise ValueError('candidate127 boundary delta no longer matches the pinned 16-parent / 32-child D replacement')
    if len(np.unique(missing_shell_rows)) != 16:
        raise ValueError('missing shell face rows are not unique')

    shell_to_d_point = np.full(n_shell, -1, np.int64)
    shell_to_d_point[d_shell_node] = np.arange(n_parent, dtype=np.int64)
    parent_edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for drow in missing_d_faces:
        tri = d_faces[int(drow)]
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            parent_edge_to_faces.setdefault(tuple(sorted((int(a), int(b)))), []).append(int(drow))
    child_parent_drows = []
    child_nodes_oldoldnew = []
    for tri in extra_boundary_faces:
        old = []
        new = []
        for global_row in tri:
            g = int(global_row)
            if g < n_shell and shell_to_d_point[g] >= 0:
                old.append(int(shell_to_d_point[g]))
            else:
                new.append(g)
        if len(old) != 2 or len(new) != 1:
            raise ValueError(f'added boundary facet does not have two authenticated D parents and one new Fat point: {tri.tolist()}')
        candidate_parents = parent_edge_to_faces.get(tuple(sorted(old)), [])
        if len(candidate_parents) != 1:
            raise ValueError(f'added boundary facet has ambiguous D parent pair {old}: {candidate_parents}')
        child_parent_drows.append(candidate_parents[0])
        child_nodes_oldoldnew.append((old[0], old[1], new[0]))
    child_parent_drows = np.asarray(child_parent_drows, np.int64)
    child_nodes_oldoldnew = np.asarray(child_nodes_oldoldnew, np.int64)
    if set(map(int, child_parent_drows)) != set(map(int, missing_d_faces)):
        raise ValueError('one or more missing D parents has no boundary children')
    if any(np.count_nonzero(child_parent_drows == row) != 2 for row in missing_d_faces):
        raise ValueError('each missing D parent must have exactly two volume-boundary children')
    new_volume_rows = np.unique(child_nodes_oldoldnew[:, 2])
    if len(new_volume_rows) != 8:
        raise ValueError('expected eight distinct candidate127 boundary points')
    parent_rows_by_new_point = [np.unique(child_parent_drows[child_nodes_oldoldnew[:, 2] == g]) for g in new_volume_rows]
    if any(len(rows) != 2 for rows in parent_rows_by_new_point):
        raise ValueError('each new point must bind one connected two-parent-face patch')
    for pair in parent_rows_by_new_point:
        if len(np.intersect1d(d_faces[pair[0]], d_faces[pair[1]])) != 2:
            raise ValueError(f'new point parent faces are not an edge-connected pair: {pair.tolist()}')

    # Map all candidate127 boundary nodes to the D point namespace, appending exact F32 rows.
    new_d_rows = n_parent + np.arange(len(new_volume_rows), dtype=np.int64)
    global_to_new_d = {int(g): int(row) for g, row in zip(new_volume_rows, new_d_rows)}
    child_faces_d = np.empty((len(extra_boundary_faces), 3), np.int64)
    child_parent_shell074 = d_shell_face[child_parent_drows]
    for i, tri in enumerate(extra_boundary_faces):
        mapped = []
        for g in tri:
            gi = int(g)
            if gi in global_to_new_d:
                mapped.append(global_to_new_d[gi])
            elif gi < n_shell and shell_to_d_point[gi] >= 0:
                mapped.append(int(shell_to_d_point[gi]))
            else:
                raise ValueError(f'candidate boundary node {gi} has no D or new-volume map')
        child_faces_d[i] = mapped
    # Deterministic grouping by new point, then direct D parent row, then boundary-row ID.
    point_group_by_global = {int(g): i for i, g in enumerate(new_volume_rows)}
    group_for_child = np.asarray([point_group_by_global[int(g)] for g in child_nodes_oldoldnew[:, 2]], np.int64)
    child_order = np.lexsort((extra_boundary_rows, child_parent_drows, group_for_child))
    child_faces_d = child_faces_d[child_order]
    child_parent_drows = child_parent_drows[child_order]
    child_parent_shell074 = child_parent_shell074[child_order]
    extra_boundary_rows_ordered = extra_boundary_rows[child_order]
    child_nodes_oldoldnew = child_nodes_oldoldnew[child_order]
    group_for_child = group_for_child[child_order]
    # Inherit orientation from each parent face by matching the child old-old edge direction.
    # This preserves the exact source boundary chain without using noisy geometric normals to choose winding.
    child_orientation_dot = np.empty(len(child_faces_d), np.float64)
    for i, (face, parent_row, old_pair) in enumerate(zip(child_faces_d, child_parent_drows, child_nodes_oldoldnew[:, :2])):
        parent_tri = d_faces[int(parent_row)]
        parent_directed = ((int(parent_tri[0]), int(parent_tri[1])),
                           (int(parent_tri[1]), int(parent_tri[2])),
                           (int(parent_tri[2]), int(parent_tri[0])))
        target = tuple(map(int, old_pair))
        if tuple(sorted(target)) not in {tuple(sorted(e)) for e in parent_directed}:
            raise ValueError(f'child {i} old-old edge is not on parent D face {parent_row}')
        child_edge = None
        for j in range(3):
            e = (int(face[j]), int(face[(j + 1) % 3]))
            if set(e) == set(target):
                child_edge = e
                break
        if child_edge is None:
            raise ValueError(f'child {i} has no directed old-old parent edge')
        expected_edge = next(e for e in parent_directed if set(e) == set(target))
        if child_edge != expected_edge:
            child_faces_d[i, [1, 2]] = child_faces_d[i, [2, 1]]
        xyz = np.vstack([d_points, volume_points[new_volume_rows]])[child_faces_d[i]]
        parent_xyz = d_points[parent_tri]
        child_orientation_dot[i] = float(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]) @ np.cross(parent_xyz[1] - parent_xyz[0], parent_xyz[2] - parent_xyz[0]))

    keep_faces = np.ones(f_parent, bool)
    keep_faces[missing_d_faces] = False
    kept_parent_rows = np.flatnonzero(keep_faces)
    output_parent_rows = np.concatenate([kept_parent_rows, child_parent_drows])
    output_faces = np.vstack([d_faces[kept_parent_rows], child_faces_d])
    new_face_rows = np.arange(len(kept_parent_rows), len(output_faces), dtype=np.int64)
    new_vertex_xyz = volume_points[new_volume_rows].copy()
    output_points = np.vstack([d_points, new_vertex_xyz]).astype(d_points.dtype, copy=False)
    output_points_f32 = np.vstack([d_points_f32, new_vertex_xyz.astype(d_points_f32.dtype)]).astype(d_points_f32.dtype, copy=False)
    if output_points.dtype != d_points.dtype or output_points_f32.dtype != d_points_f32.dtype:
        raise ValueError('derived D point arrays changed their parent dtypes')
    unique_before = len(np.unique(d_points, axis=0))
    unique_after = len(np.unique(output_points, axis=0))
    if unique_after - unique_before != len(new_vertex_xyz):
        raise ValueError('one or more new boundary points duplicate an existing D point')
    if np.any(output_faces < 0) or np.any(output_faces >= len(output_points)):
        raise ValueError('refined D face index outside point domain')
    if np.any(np.sort(output_faces, axis=1)[:, 1:] == np.sort(output_faces, axis=1)[:, :-1]):
        raise ValueError('refined D contains a repeated-node triangle')
    xyz = output_points[output_faces].astype(np.float64)
    area2 = np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
    if np.any(~np.isfinite(area2)) or np.any(area2 <= 0):
        raise ValueError('refined D contains a zero/nonfinite Float32 face')
    # Closed oriented surface: every edge occurs twice with opposite directed orientation.
    directed = output_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2))
    canon = np.sort(directed, axis=1)
    edge_keys, edge_inverse, edge_counts = np.unique(canon, axis=0, return_inverse=True, return_counts=True)
    edge_sign = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    edge_orient_sum = np.bincount(edge_inverse, weights=edge_sign, minlength=len(edge_keys))
    base_directed = d_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape((-1, 2))
    base_edge_keys, base_inv, base_counts = np.unique(np.sort(base_directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    base_open = set(map(tuple, base_edge_keys[base_counts == 1].tolist()))
    output_open = set(map(tuple, edge_keys[edge_counts == 1].tolist()))
    bad_internal_edge_rows = np.flatnonzero((edge_counts == 2) & (edge_orient_sum != 0))
    nonmanifold_edge_rows = np.flatnonzero(edge_counts > 2)
    bad_edge_rows = np.union1d(bad_internal_edge_rows, nonmanifold_edge_rows)
    if output_open != base_open or len(bad_edge_rows):
        raise ValueError(f'refined D edge chain invalid: baseline open={len(base_open)}, output open={len(output_open)}, extra={sorted(output_open-base_open)[:12]}, missing={sorted(base_open-output_open)[:12]}, internal_orientation_failures={len(bad_internal_edge_rows)}, nonmanifold={len(nonmanifold_edge_rows)}')


    # Projection diagnostics only; the actual inserted coordinates remain the exact positive-volume boundary points.
    point_parent_offsets = [0]
    point_parent_rows = []
    point_bary = []
    point_plane = []
    point_distance = []
    point_closest = []
    point_min_bary = []
    for g, group_parents in zip(new_volume_rows, parent_rows_by_new_point):
        p = volume_points[int(g)].astype(np.float64)
        for drow in group_parents:
            bary, plane, dist, closest = point_triangle_projection(p, d_points[d_faces[int(drow)]])
            point_parent_rows.append(int(drow))
            point_bary.append(bary)
            point_plane.append(plane)
            point_distance.append(dist)
            point_closest.append(closest)
            point_min_bary.append(float(np.min(bary)))
        point_parent_offsets.append(len(point_parent_rows))
    point_parent_rows = np.asarray(point_parent_rows, np.int64)
    point_bary = np.asarray(point_bary, np.float64)
    point_plane = np.asarray(point_plane, np.float64)
    point_distance = np.asarray(point_distance, np.float64)
    point_closest = np.asarray(point_closest, np.float64)
    point_min_bary = np.asarray(point_min_bary, np.float64)

    # Create one complete kind-6 typed union per connected pair of parent faces.
    old_qoff = np.asarray(parent['quality_star_patch_union_offsets'], np.int64)
    qlane_keys = tuple(f'quality_star_patch_union_{k}' for k in TYPED_FIELDS)
    qlanes = [np.asarray(parent[k]) for k in qlane_keys]
    if old_qoff.shape != (f_parent + 1,) or old_qoff[0] != 0 or old_qoff[-1] != len(qlanes[0]) or any(len(x) != len(qlanes[0]) for x in qlanes):
        raise ValueError('parent full typed patch-union CSR is malformed')
    group_tuples = []
    group_typed = []
    for group_parents in parent_rows_by_new_point:
        recs = csr_union(parent, 'quality_star_patch_union_offsets', qlane_keys, group_parents)
        if not recs:
            recs = sorted({tuple(int(parent[k][r]) for k in TYPED_FIELDS) for r in group_parents})
        group_tuples.append(recs)
        group_typed.append(np.asarray(recs, np.int64).reshape((-1, 7)))
    group_for_output: dict[int, int] = {}
    for local_i, group_id in enumerate(group_for_child):
        group_for_output[int(new_face_rows[local_i])] = int(group_id)

    # Remap every per-face array by its direct parent; grouped children then get explicit derived identities.
    output: dict[str, np.ndarray] = {}
    face_csr_offsets = {k for k in parent if k.endswith('_offsets') and np.asarray(parent[k]).shape == (f_parent + 1,)}
    point_csr_offsets = {k for k in parent if k.endswith('_offsets') and np.asarray(parent[k]).shape == (n_parent + 1,)}
    unknown_face_offsets = face_csr_offsets - set(FACE_CSR)
    unknown_point_offsets = point_csr_offsets - set(POINT_CSR)
    if unknown_face_offsets or unknown_point_offsets:
        raise ValueError(f'unknown row-owned CSR fields: face={sorted(unknown_face_offsets)}, point={sorted(unknown_point_offsets)}')
    face_csr_lanes = {lane for lanes in FACE_CSR.values() for lane in lanes}
    face_csr_lanes |= {'face_ancestry_input_cache_rows', 'face_ancestry_source_rows'}
    point_csr_lanes = {'microedge_parent_input_point_rows', 'output_vertex_input_parent_rows'}
    for key, value in parent.items():
        if key in ('faces', 'points_world_m', 'points_world_m_f32') or key in face_csr_offsets or key in face_csr_lanes or key in point_csr_offsets or key in point_csr_lanes:
            continue
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == f_parent:
            output[key] = arr[output_parent_rows].copy()
        elif arr.ndim > 0 and arr.shape[0] == n_parent:
            if np.issubdtype(arr.dtype, np.integer):
                fill = -1
            elif np.issubdtype(arr.dtype, np.floating):
                fill = np.nan
            elif arr.dtype.kind in ('U', 'S'):
                fill = ''
            elif arr.dtype.kind == 'b':
                fill = False
            else:
                raise ValueError(f'unsupported point-owned array {key}: {arr.dtype}')
            new = np.full((len(output_points),) + arr.shape[1:], fill, dtype=arr.dtype)
            new[:n_parent] = arr
            output[key] = new
        else:
            output[key] = arr.copy()
    output['points_world_m'] = output_points
    output['points_world_m_f32'] = output_points_f32
    output['faces'] = output_faces

    # Vectorized rebuild of each supported face CSR; new children are kind-6 unions across both connected parents.
    for offset_key, lane_keys in FACE_CSR.items():
        if offset_key not in parent:
            continue
        new_off, new_lanes = rebuild_grouped_csr(parent, output_parent_rows, group_for_output, parent_rows_by_new_point, offset_key, lane_keys)
        output[offset_key] = new_off
        output.update(new_lanes)
    # Keep the canonical source-face alias aligned for the legacy 217 loader, while retaining the old raw aliases verbatim.
    old_alias_input = np.asarray(parent.get('face_ancestry_input_cache_rows', np.empty(0, np.int64))).copy()
    old_alias_source = np.asarray(parent.get('face_ancestry_source_rows', np.empty(0, np.int64))).copy()
    if 'face_ancestry_source_face_rows' in output:
        output['legacy_unaligned_face_ancestry_input_cache_rows'] = old_alias_input
        output['legacy_unaligned_face_ancestry_source_rows'] = old_alias_source
        output['face_ancestry_source_rows'] = output['face_ancestry_source_face_rows'].copy()
        # The typed-D owner uses full union ancestry; the old cache alias was not aligned to its CSR offsets.
        output.pop('face_ancestry_input_cache_rows', None)

    # Append empty rows to existing point parent CSRs; new provenance is explicit below.
    for offset_key in POINT_CSR:
        if offset_key not in parent:
            continue
        old = np.asarray(parent[offset_key], np.int64)
        output[offset_key] = np.concatenate([old, np.full(len(new_volume_rows), int(old[-1]), np.int64)])
    for lane in point_csr_lanes:
        if lane in parent:
            output[lane] = np.asarray(parent[lane]).copy()

    # Mark derived children with a synthetic patch identity, keeping complete source support in the seven-field union.
    schema_raw = np.asarray(parent['face_source_identity_schema_by_original_id_json']).reshape(-1)[0]
    schema = json.loads(str(schema_raw))
    schema_ids = {int(k) for k in schema if str(k).lstrip('-').isdigit()}
    source_ids = set(map(int, np.unique(np.asarray(parent['face_source_original_id'], np.int64))))
    next_source_id = max(schema_ids | source_ids | {699_999_999}) + 1
    old_patch_ids = np.asarray(parent.get('quality_star_patch_id', np.full(f_parent, -1)), np.int64)
    next_patch_id = int(old_patch_ids.max(initial=-1)) + 1
    # Use an explicit child-order patch vector, indexed in output face row order.
    patch_vector = np.empty(len(new_face_rows), np.int64)
    for i, gid in enumerate(group_for_child):
        patch_vector[i] = next_patch_id + int(gid)
    if 'quality_star_patch_id' in output:
        output['quality_star_patch_id'][new_face_rows] = patch_vector
    if 'quality_star_patch_operation_id' in output:
        source_ops = np.asarray(output['quality_star_patch_operation_id'])
        labels = np.asarray([f'inferred_boundary_D_refinement_128_patch_{int(g)}' for g in group_for_child])
        width = max(source_ops.dtype.itemsize // 4, max(map(len, labels.tolist()), default=1))
        source_ops = source_ops.astype(f'<U{width}')
        source_ops[new_face_rows] = labels
        output['quality_star_patch_operation_id'] = source_ops
    for i, (outrow, gid) in enumerate(zip(new_face_rows, group_for_child)):
        tuples = group_typed[int(gid)]
        common = []
        for col in (0, 1, 2):
            vals = np.unique(tuples[:, col])
            common.append(int(vals[0]) if len(vals) == 1 else -1)
        synthetic = next_source_id + int(gid)
        for key, val in zip(('face_source_layer', 'face_source_body', 'face_source_bone'), common):
            output[key][outrow] = val
        output['face_source_face_id'][outrow] = -1
        output['face_source_original_id'][outrow] = synthetic
        output['face_source_identity0'][outrow] = int(outrow)
        output['face_source_identity1'][outrow] = next_patch_id + int(gid)
        for key in ('face_cache_row', 'face_source_row', 'face_exact_source_row', 'face_id', 'face_run_original_id', 'union_input_run_original_id', 'union_input_face_id', 'union_input_face_rows', 'quality_star_face_output_to_input_row', 'quality_star_immediate_face_output_to_input_row'):
            if key in output and output[key].shape == (len(output_faces),):
                output[key][outrow] = -1
    schema_entries = []
    for gid, (global_row, parent_rows, tuples) in enumerate(zip(new_volume_rows, parent_rows_by_new_point, group_tuples)):
        synthetic = next_source_id + gid
        patchid = next_patch_id + gid
        record = {
            'role': 'DERIVED_PATCH',
            'derived_operation': 'fat_positive_volume_127_inferred_D_boundary_refinement',
            'derived_patch_id': f'fat-boundary-global-node-{int(global_row)}',
            'derived_patch_ordinal': int(patchid),
            'face_count': int(np.count_nonzero(group_for_child == gid)),
            'identity_keys': ['face_source_identity0', 'face_source_identity1'],
            'identity_face_id_key': 'face_source_face_id',
            'parent_support_semantics': 'Kind-6 complete union of the two edge-connected parent D faces supporting the exact candidate127 volume-boundary children; child triangles are not claimed to be exact planar subdivisions or strictly interior to one parent triangle.',
            'parent_face_row_domain': 'candidate-D-inferred-boundary-f32-repair.npz face rows before this refinement',
            'parent_D_face_rows': np.asarray(parent_rows, np.int64).tolist(),
            'parent_074_shell_face_rows': d_shell_face[np.asarray(parent_rows, np.int64)].tolist(),
            'parent_union_tuple_count': len(tuples),
            'complete_typed_parent_union': [list(x) for x in tuples],
            'source_volume_global_point_row': int(global_row),
        }
        schema[str(synthetic)] = record
        schema_entries.append({'synthetic_original_id': int(synthetic), 'patch_id': int(patchid), 'source_volume_global_point_row': int(global_row), 'parent_D_face_rows': np.asarray(parent_rows, np.int64).tolist(), 'parent_074_shell_face_rows': d_shell_face[np.asarray(parent_rows, np.int64)].tolist(), 'typed_parent_union': [list(x) for x in tuples]})
    output['face_source_identity_schema_by_original_id_json'] = np.asarray([json.dumps(schema, sort_keys=True, separators=(',', ':'), ensure_ascii=True)])
    if 'quality_star_patch_union_semantics' in output:
        output['quality_star_patch_union_semantics'] = np.asarray(['Each inferred-boundary child face carries the complete deduplicated seven-field kind-6 identity union of both edge-connected D parent faces in its local patch. This is patch-level support, not a per-triangle overlap or exact-single-parent claim.'])

    # Per-output-face direct parent maps and typed D provenance.
    out_parent_shell = np.full(len(output_faces), -1, np.int64)
    out_parent_shell[kept_parent_rows] = d_shell_face[kept_parent_rows]
    out_parent_shell[new_face_rows] = child_parent_shell074
    out_derived = np.zeros(len(output_faces), bool)
    out_derived[new_face_rows] = True
    out_patch_id = np.full(len(output_faces), -1, np.int64)
    out_patch_id[new_face_rows] = patch_vector
    out_boundary_row = np.full(len(output_faces), -1, np.int64)
    out_boundary_row[new_face_rows] = extra_boundary_rows_ordered
    output['inferred_boundary_refinement_parent_D_face_rows'] = output_parent_rows.copy()
    output['inferred_boundary_refinement_parent_074_shell_face_rows'] = out_parent_shell
    output['inferred_boundary_refinement_is_derived_face'] = out_derived
    output['inferred_boundary_refinement_patch_id'] = out_patch_id
    output['inferred_boundary_refinement_fat_boundary_face_rows'] = out_boundary_row
    output['inferred_boundary_refinement_new_face_parent_D_rows'] = child_parent_drows.copy()
    output['inferred_boundary_refinement_new_face_parent_074_shell_rows'] = child_parent_shell074.copy()
    output['inferred_boundary_refinement_new_face_fat_boundary_rows'] = extra_boundary_rows_ordered.copy()
    output['inferred_boundary_refinement_new_face_patch_id'] = patch_vector.copy()
    output['inferred_boundary_refinement_new_face_nodes_fat_global_rows'] = extra_boundary_faces[child_order].copy()
    output['inferred_boundary_refinement_new_face_output_D_rows'] = new_face_rows.copy()
    output['inferred_boundary_refinement_retired_parent_D_face_rows'] = missing_d_faces.copy()
    output['inferred_boundary_refinement_retired_parent_074_shell_rows'] = missing_shell_rows.copy()
    output['inferred_boundary_refinement_new_vertex_D_rows'] = new_d_rows.copy()
    output['inferred_boundary_refinement_new_vertex_source_fat_global_rows'] = new_volume_rows.copy()
    output['inferred_boundary_refinement_output_vertex_to_source_D_row'] = np.r_[np.arange(n_parent, dtype=np.int64), np.full(len(new_volume_rows), -1, np.int64)]
    output['inferred_boundary_refinement_output_vertex_to_source_fat_global_row'] = np.r_[d_shell_node.copy(), new_volume_rows.copy()]
    output['inferred_boundary_refinement_point_parent_D_face_offsets'] = np.asarray(point_parent_offsets, np.int64)
    output['inferred_boundary_refinement_point_parent_D_face_rows'] = point_parent_rows.copy()
    output['inferred_boundary_refinement_point_parent_face_barycentric_projection'] = point_bary.copy()
    output['inferred_boundary_refinement_point_parent_face_plane_residual_m'] = point_plane.copy()
    output['inferred_boundary_refinement_point_parent_face_closest_distance_m'] = point_distance.copy()
    output['inferred_boundary_refinement_point_parent_face_closest_world_m'] = point_closest.copy()
    output['inferred_boundary_refinement_point_parent_face_min_barycentric'] = point_min_bary.copy()
    output['inferred_boundary_refinement_source_volume_global_point_rows'] = new_volume_rows.copy()
    output['inferred_boundary_refinement_operation_json'] = np.asarray([json.dumps({
        'schema': 'fat-inferred-boundary-D-refinement-128-v2.v1',
        'source_candidate127_volume_path': str(INPUTS['volume']),
        'source_candidate127_volume_sha256': hashes['volume'],
        'source_074_fat_shell_path': str(INPUTS['shell']),
        'source_074_fat_shell_sha256': hashes['shell'],
        'source_typed_D_path': str(INPUTS['D']),
        'source_typed_D_sha256': hashes['D'],
        'physical_admission': False,
        'scope': 'Replace sixteen missing registered D parent facets with the exact thirty-two D-labeled boundary triangles from the already positive candidate127 Fat volume; append only the eight exact volume-boundary points. Old D points are unchanged bitwise. Each connected two-parent patch is kind-6 provenance union; parent-face projections are diagnostic support only.',
        'new_volume_global_point_rows': new_volume_rows.tolist(),
        'retired_D_parent_face_rows': missing_d_faces.tolist(),
        'retired_074_shell_face_rows': missing_shell_rows.tolist(),
        'face_patch_schema': schema_entries,
        'max_parent_plane_residual_m': float(point_plane.max(initial=0)),
        'max_parent_triangle_closest_distance_m': float(point_distance.max(initial=0)),
        'negative_projection_barycentric_component_count': int(np.count_nonzero(point_min_bary < 0)),
        'minimum_projection_barycentric_component': float(point_min_bary.min(initial=0)),
        'exact_face_boundary_source': 'candidate127 full Fat tetrahedron boundary unique-face rows; parent shell triangles matched by exact F32 vertex IDs; no nearest/global-coordinate ancestry used',
    }, sort_keys=True, separators=(',', ':'))])

    # Rebuild face CSR as kind-6 pair unions while preserving every source lane.
    # Add explicit legacy aliases only after the primary aligned face_ancestry CSR is rebuilt.
    if 'face_ancestry_source_face_rows' in output:
        output['face_ancestry_source_rows'] = output['face_ancestry_source_face_rows'].copy()
        output.pop('face_ancestry_input_cache_rows', None)
    # Preserve point ownership lanes and explicit kind-6 point support; the new points have no direct old-vertex ID.
    output['inferred_boundary_refinement_point_parent_face_semantics'] = np.asarray(['Per-new-point two-parent D-face affine projection support; barycentrics may be outside the triangle; residuals are recorded and do not assert exact source-point identity.'])

    # Verify source arrays and connectivity contract before emission.
    if output_points[:n_parent].tobytes() != d_points.tobytes() or output_points_f32[:n_parent].tobytes() != d_points_f32.tobytes():
        raise ValueError('old D source point prefix changed')
    if not np.array_equal(output_faces[:len(kept_parent_rows)], d_faces[kept_parent_rows]):
        raise ValueError('retained D faces changed or reordered')
    if len(output_faces) != f_parent + 16 or len(output_points) != n_parent + 8:
        raise ValueError('refined D surface cardinality mismatch')
    if len(output['union_face_ancestry_offsets']) != len(output_faces) + 1:
        raise ValueError('typed D union CSR did not follow the refined face domain')
    # Check the registered volume-boundary child triangles match by exact global node rows.
    mapped_global = np.asarray(output['inferred_boundary_refinement_new_face_nodes_fat_global_rows'], np.int64)
    expected_global = np.asarray(extra_boundary_faces[child_order], np.int64)
    if not np.array_equal(mapped_global, expected_global):
        raise ValueError('emitted child-face source global rows differ from candidate127 boundary rows')
    for i, f in enumerate(child_faces_d):
        got_xyz = output_points[f]
        exp_xyz = volume_points[expected_global[i]]
        got_order = np.lexsort((got_xyz[:, 2], got_xyz[:, 1], got_xyz[:, 0]))
        exp_order = np.lexsort((exp_xyz[:, 2], exp_xyz[:, 1], exp_xyz[:, 0]))
        if not np.array_equal(got_xyz[got_order], exp_xyz[exp_order]):
            raise ValueError(f'child face {i} does not preserve exact candidate127 boundary F32 coordinates')
    # Whole output face count; each new kind-6 patch is exactly four child triangles over two parent faces.
    patch_face_counts = {str(int(g)): int(np.count_nonzero(group_for_child == g)) for g in range(len(new_volume_rows))}
    if any(n != 4 for n in patch_face_counts.values()):
        raise ValueError(f'expected four child facets per two-parent point patch: {patch_face_counts}')
    OUT.mkdir(parents=True, exist_ok=False)
    typed_path = OUT / 'candidate-D-inferred-boundary-refined-typed.npz'
    map_path = OUT / 'inferred-boundary-D-crosswalk.npz'
    np.savez_compressed(typed_path, **output)
    out_hash = sha(typed_path)
    np.savez_compressed(map_path,
        source_volume_global_point_rows=new_volume_rows,
        new_D_vertex_rows=new_d_rows,
        output_vertex_to_source_D_row=np.r_[np.arange(n_parent, dtype=np.int64), np.full(len(new_volume_rows), -1, np.int64)],
        output_vertex_to_source_Fat_global_row=np.r_[d_shell_node.copy(), new_volume_rows.copy()],
        point_parent_offsets=np.asarray(point_parent_offsets, np.int64),
        point_parent_D_face_rows=point_parent_rows,
        point_parent_074_shell_face_rows=d_shell_face[point_parent_rows],
        point_parent_barycentric_projection=point_bary,
        point_parent_plane_residual_m=point_plane,
        point_parent_triangle_closest_distance_m=point_distance,
        point_parent_closest_world_m=point_closest,
        point_parent_min_barycentric=point_min_bary,
        retired_parent_D_face_rows=missing_d_faces,
        retired_parent_074_shell_face_rows=missing_shell_rows,
        new_face_output_D_rows=new_face_rows,
        new_face_parent_D_rows=child_parent_drows,
        new_face_parent_074_shell_rows=child_parent_shell074,
        new_face_fat_boundary_rows=extra_boundary_rows_ordered,
        new_face_fat_global_node_rows=expected_global,
        new_face_patch_id=patch_vector,
        output_face_parent_D_rows=output_parent_rows,
        output_face_parent_074_shell_rows=out_parent_shell,
        output_face_is_derived=out_derived,
        output_face_fat_boundary_row=out_boundary_row,
    )
    map_hash = sha(map_path)
    script_hash = sha(Path(__file__))
    report = {
        'schema': 'fat-inferred-boundary-D-refinement-128-v2.v1',
        'status': 'PASS_TYPED_D_BOUNDARY_REFINEMENT_CONSTRUCTED_EXACT_PROJECTION_METRICS',
        'physical_admission': False,
        'runtime': 0,
        'builder_path': str(Path(__file__)),
        'builder_sha256': script_hash,
        'inputs': {k: {'path': str(INPUTS[k]), 'sha256': hashes[k]} for k in INPUTS},
        'outputs': {'typed_D': {'path': str(typed_path), 'sha256': out_hash}, 'crosswalk': {'path': str(map_path), 'sha256': map_hash}},
        'input_counts': {'D_vertices': n_parent, 'D_faces': f_parent, 'Fat_shell_vertices': n_shell, 'Fat_shell_faces': len(shell_faces), 'candidate_Fat_tets': len(tet), 'candidate_Fat_boundary_faces': len(boundary_faces)},
        'output_counts': {'D_vertices': len(output_points), 'D_faces': len(output_faces), 'new_D_vertices': len(new_volume_rows), 'new_child_faces': len(child_faces_d), 'retired_D_parent_faces': len(missing_d_faces), 'whole_Fat_boundary_faces_after_refinement': len(boundary_faces)},
        'boundary_delta': {'missing_074_shell_face_rows': missing_shell_rows.tolist(), 'retired_D_face_rows': missing_d_faces.tolist(), 'extra_candidate_Fat_boundary_rows': extra_boundary_rows_ordered.tolist(), 'extra_candidate_Fat_global_node_rows': expected_global.tolist(), 'parent_child_face_counts': {str(int(r)): int(np.count_nonzero(child_parent_drows == r)) for r in missing_d_faces}},
        'new_vertex_map': [{'new_D_vertex_row': int(dr), 'candidate127_Fat_global_point_row': int(gr)} for dr, gr in zip(new_d_rows, new_volume_rows)],
        'parent_projection_diagnostics_only': {'max_plane_residual_m': float(point_plane.max(initial=0)), 'max_closest_triangle_distance_m': float(point_distance.max(initial=0)), 'max_closest_triangle_distance_um': float(point_distance.max(initial=0) * 1e6), 'negative_affine_projection_barycentric_component_count': int(np.count_nonzero(point_min_bary < 0)), 'minimum_affine_projection_barycentric_component': float(point_min_bary.min(initial=0)), 'claim': 'Inserted point coordinates are exact candidate127 volume-boundary F32 coordinates. Parent-face projections are support diagnostics only; extrapolative barycentrics are retained and no strict interior-parent-triangle claim is made.'},
        'structure_checks': {'baseline_boundary_edge_set_preserved_exactly': True, 'baseline_boundary_edge_count': int(len(base_open)), 'internal_edge_incidence_two_opposite_and_no_nonmanifold': True, 'bad_internal_or_nonmanifold_edge_count': int(len(bad_edge_rows)), 'all_faces_nonzero_F32_area': True, 'minimum_double_area_m2': float(area2.min()), 'old_D_point_bytes_unchanged': True, 'new_points_unique_vs_old_D': True, 'each_kind6_patch_two_edge_connected_D_parents': True, 'child_face_orientation_inherited_from_parent_boundary_chain': True, 'child_normal_alignment_min_dot_m4': float(child_orientation_dot.min(initial=0)), 'child_normal_alignment_negative_count': int(np.count_nonzero(child_orientation_dot < 0))},
        'provenance': {'typed_kind': 6, 'patch_count': len(new_volume_rows), 'child_faces_per_patch': patch_face_counts, 'face_union_semantics': 'complete seven-field union of each connected two-parent D patch; not exact single-face overlap', 'full_parent_face_row_union_retained': True, 'new_point_to_Fat_global_row_explicit': True},
        'limitation': 'This is a refined operational D surface built from exact boundary facets of a positive candidate Fat mesh. It is not a full geometry, mass, native-domain, or runtime admission result.',
    }
    report_path = OUT / 'receipt.json'
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'typed_D_sha256': out_hash, 'crosswalk_sha256': map_hash, 'receipt_path': str(report_path), 'receipt_sha256': sha(report_path), 'new_D_vertices': new_d_rows.tolist(), 'source_Fat_global_vertices': new_volume_rows.tolist(), 'max_projection_um': report['parent_projection_diagnostics_only']['max_closest_triangle_distance_um'], 'max_plane_residual_nm': report['parent_projection_diagnostics_only']['max_plane_residual_m'] * 1e9, 'bad_edges': int(len(bad_edge_rows)), 'baseline_boundary_edges': int(len(base_open)), 'child_normal_alignment_negative': int(np.count_nonzero(child_orientation_dot < 0)), 'min_double_area_m2': float(area2.min())}, separators=(',', ':')), flush=True)

if __name__ == '__main__':
    main()
