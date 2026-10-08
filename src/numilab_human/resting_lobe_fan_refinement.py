"""Bounded source vertex fan refinement for adjacent registered lung lobes."""
from __future__ import annotations

from collections import defaultdict, deque
import numpy as np

from .resting_anatomy_conforming_refinement import edge_incidence, edge_key, refine_surface_edges
from .resting_anatomy_interface_patch import normals, signed_volume, topology_report


def _point_key(point):
    values = np.asarray(point, dtype="<f4")
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("invalid Float32 source point")
    return tuple(int(x) for x in values.view("<u4"))


def _face_key(vertices6, triangle):
    return tuple(sorted(_point_key(vertices6[int(v), :3]) for v in triangle))


def _oriented_key(vertices6, triangle):
    return tuple(_point_key(vertices6[int(v), :3]) for v in triangle)


def _opposite(a, b):
    return any(tuple(a[(shift-i) % 3] for i in range(3)) == tuple(b) for shift in range(3))


def _components(faces):
    parent = list(range(len(faces)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for adjacent in edge_incidence(faces).values():
        if len(adjacent) == 2:
            a, b = map(find, adjacent)
            if a != b:
                parent[b] = a
    return len({find(i) for i in range(len(faces))})


def _metrics(vertices6, faces):
    faces = np.asarray(faces, dtype=np.int64)
    xyz = np.asarray(vertices6, dtype=np.float32)[:, :3].astype(np.float64)
    topo = topology_report(faces)
    used = np.unique(faces.reshape(-1))
    euler = int(len(used) - topo["edge_count"] + len(faces))
    components = _components(faces)
    if (topo["boundary_edge_count"] or topo["nonmanifold_edge_count"]
            or topo["orientation_error_edge_count"] or topo["boundary_branch_vertex_count"]
            or components != 1 or euler != 2):
        raise ValueError(f"fan source/result is not one closed oriented genus-zero lobe: {topo}")
    tri = xyz[faces]
    twice = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)
    if not np.isfinite(twice).all() or np.any(twice <= 0):
        raise ValueError("fan source/result contains a non-finite or zero-area face")
    return {
        "vertex_count": int(len(used)), "face_count": int(len(faces)),
        "edge_count": int(topo["edge_count"]), "connected_components": components,
        "euler_characteristic": euler,
        "boundary_edge_count": int(topo["boundary_edge_count"]),
        "nonmanifold_edge_count": int(topo["nonmanifold_edge_count"]),
        "orientation_error_edge_count": int(topo["orientation_error_edge_count"]),
        "minimum_triangle_area_m2": float(twice.min() * 0.5),
        "signed_volume_m3": float(signed_volume(xyz, faces)),
        "surface_area_m2": float(twice.sum() * 0.5),
    }


def _face_mates(lobes):
    groups = defaultdict(list)
    for sid, (vertices6, faces) in sorted(lobes.items()):
        vertices6 = np.asarray(vertices6, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)
        if (vertices6.ndim != 2 or vertices6.shape[1] != 6 or not np.isfinite(vertices6).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or int(faces.min()) < 0 or int(faces.max()) >= len(vertices6)):
            raise ValueError(f"malformed lobe {sid}")
        for fi, tri in enumerate(faces):
            groups[_face_key(vertices6, tri)].append((sid, fi, _oriented_key(vertices6, tri)))
    mates = {}
    count = 0
    for group in groups.values():
        if len(group) == 1:
            continue
        if len(group) != 2 or group[0][0] == group[1][0]:
            raise ValueError("duplicate or multiply owned lobe face")
        a, b = group
        if not _opposite(a[2], b[2]):
            raise ValueError("exact reciprocal lobe faces have non-opposite winding")
        mates[(a[0], a[1])] = (b[0], b[1])
        mates[(b[0], b[1])] = (a[0], a[1])
        count += 1
    return mates, count


def refine_shared_vertex_fans(lobes, *, seed_faces, expected_shared_vertex_m):
    """Split radial edges around a pinned source vertex and propagate exact seams."""
    if len(seed_faces) != 2 or int(seed_faces[0][0]) == int(seed_faces[1][0]):
        raise ValueError("exactly two distinct-lobe seed faces are required")
    rows = {int(sid): (np.asarray(row[0], dtype=np.float32).copy(),
                       np.asarray(row[1], dtype=np.int64).copy())
            for sid, row in lobes.items()}
    expected = np.asarray(expected_shared_vertex_m, dtype="<f4")
    if expected.shape != (3,) or not np.isfinite(expected).all():
        raise ValueError("invalid pinned source vertex")
    expected_key = _point_key(expected)
    source_metrics = {sid: _metrics(*row) for sid, row in sorted(rows.items())}
    mates, source_pair_count = _face_mates(rows)

    seeds = defaultdict(set)
    for sid, fi in seed_faces:
        sid, fi = int(sid), int(fi)
        if sid not in rows or fi < 0 or fi >= len(rows[sid][1]):
            raise ValueError(f"seed face {(sid, fi)} is missing or out of bounds")
        vertices6, faces = rows[sid]
        if [_point_key(vertices6[int(v), :3]) for v in faces[fi]].count(expected_key) != 1:
            raise ValueError(f"seed face {(sid, fi)} does not contain the pinned point once")
        vids = np.flatnonzero(np.all(vertices6[:, :3] == expected.reshape(1, 3), axis=1))
        if not len(vids):
            raise ValueError(f"lobe {sid} lacks the pinned source point")
        vset = set(map(int, vids))
        incident = np.flatnonzero(np.any(np.isin(faces, vids), axis=1))
        for face_id in incident:
            tri = faces[int(face_id)]
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i+1) % 3])
                if a in vset or b in vset:
                    seeds[sid].add(edge_key(a, b))
        if not seeds[sid]:
            raise ValueError(f"lobe {sid} has an empty source vertex fan")

    split = {sid: set(edges) for sid, edges in seeds.items()}
    incidence = {sid: edge_incidence(row[1]) for sid, row in rows.items()}
    queue = deque((sid, edge) for sid, edges in split.items() for edge in edges)
    propagated = 0
    while queue:
        sid, edge = queue.popleft()
        for fi in incidence[sid].get(edge, []):
            mate = mates.get((sid, int(fi)))
            if mate is None:
                continue
            other_sid, other_fi = mate
            endpoint_keys = {_point_key(rows[sid][0][v, :3]) for v in edge}
            tri = rows[other_sid][1][other_fi]
            candidates = [edge_key(int(tri[i]), int(tri[(i+1) % 3]))
                          for i in range(3)
                          if {_point_key(rows[other_sid][0][v, :3])
                              for v in edge_key(int(tri[i]), int(tri[(i+1) % 3]))} == endpoint_keys]
            if len(candidates) != 1:
                raise ValueError("reciprocal split edge has no unique coordinate mate")
            target = candidates[0]
            if target not in split.setdefault(other_sid, set()):
                split[other_sid].add(target)
                queue.append((other_sid, target))
                propagated += 1

    refined = {sid: (v.copy(), f.copy()) for sid, (v, f) in rows.items()}
    ancestry = {sid: {i: [i] for i in range(len(row[1]))} for sid, row in rows.items()}
    reports = {}
    max_midpoint_error = 0.0
    for sid, edges in sorted(split.items()):
        before_v, before_f = rows[sid]
        after_v, after_f, old_to_new, detail = refine_surface_edges(before_v, before_f, edges)
        after_v[:, 3:6] = normals(after_v[:, :3].astype(np.float64), after_f).astype(np.float32)
        if not np.array_equal(after_v[:len(before_v), :3], before_v[:, :3]):
            raise ValueError(f"refinement moved an existing source vertex on lobe {sid}")
        for (a, b), actual in zip(sorted(edges), after_v[len(before_v):, :3], strict=True):
            midpoint = (before_v[a, :3].astype(np.float64)+before_v[b, :3].astype(np.float64))*0.5
            max_midpoint_error = max(max_midpoint_error,
                                     float(np.linalg.norm(actual.astype(np.float64)-midpoint)))
        before = source_metrics[sid]
        after = _metrics(after_v, after_f)
        volume_delta = after["signed_volume_m3"] - before["signed_volume_m3"]
        if abs(volume_delta) > 1e-12:
            raise ValueError(f"refinement changed signed source volume of lobe {sid} by {volume_delta:.9g} m3")
        refined[sid] = (after_v, after_f)
        ancestry[sid] = old_to_new
        affected_faces = set(detail["affected_face_indices"])
        reports[str(sid)] = {
            "split_edges": [list(edge) for edge in sorted(edges)],
            "affected_source_faces": sorted(affected_faces),
            "source_face_to_children": {str(fi): old_to_new[fi] for fi in sorted(affected_faces)},
            "added_vertex_count": int(detail["added_vertex_count"]),
            "added_face_count": int(detail["added_face_count"]),
            "before": before, "after": after,
            "signed_volume_delta_m3": volume_delta,
            "original_source_position_bits_preserved": True,
            "child_winding_checked": True,
        }

    reciprocal_children = 0
    for (sid, fi), (other_sid, other_fi) in mates.items():
        if (sid, fi) > (other_sid, other_fi):
            continue
        left, right = {}, {}
        for child in ancestry[sid][fi]:
            tri = refined[sid][1][child]
            left[_face_key(refined[sid][0], tri)] = _oriented_key(refined[sid][0], tri)
        for child in ancestry[other_sid][other_fi]:
            tri = refined[other_sid][1][child]
            right[_face_key(refined[other_sid][0], tri)] = _oriented_key(refined[other_sid][0], tri)
        if (len(left) != len(ancestry[sid][fi]) or len(right) != len(ancestry[other_sid][other_fi])
                or set(left) != set(right)):
            raise ValueError("refinement broke an exact reciprocal lobe child patch")
        if any(not _opposite(left[key], right[key]) for key in left):
            raise ValueError("refinement changed reciprocal child winding")
        reciprocal_children += len(left)

    return refined, ancestry, {
        "algorithm": "exact-shared-source-vertex-radial-edge-midpoint-fan-refinement-v1",
        "seed_faces": [[int(sid), int(fi)] for sid, fi in seed_faces],
        "expected_shared_source_vertex_f32_m": expected.astype(float).tolist(),
        "source_exact_reciprocal_face_group_count": int(source_pair_count),
        "candidate_exact_reciprocal_child_face_group_count": int(reciprocal_children),
        "split_edge_count_by_lobe": {str(sid): len(edges) for sid, edges in sorted(split.items())},
        "propagated_reciprocal_split_edge_count": int(propagated),
        "per_lobe": reports,
        "maximum_midpoint_rounding_deviation_m": max_midpoint_error,
        "original_source_positions_moved": False,
        "source_face_ancestry_preserved": True,
        "exact_reciprocal_child_geometry_and_winding_preserved": True,
        "native_deformed_pose_intersection_gate": "not_assessed",
    }

