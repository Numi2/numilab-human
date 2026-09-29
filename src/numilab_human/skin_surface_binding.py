"""Offline source-surface skin binding; no mechanical skin owner.

Bone samples seed the connected source skin, rather than assigning skin to an
unrelated nearby hand in Euclidean space. A positive screened graph Laplacian
smooths those inferred associations. ABI 5 receives every body weight; the four
largest weights remain diagnostics and the offline solution is independently
verified against exact source projection gaps and graph equations.
"""
from __future__ import annotations

import hashlib
import heapq
from pathlib import Path

LEGACY_METHOD = 'positive_source_surface_screened_harmonic_four_influence.v1'
METHOD = 'positive_source_surface_projection_gap_screened_harmonic_full_weight.v2'


def source_surface_binding(vertices, faces, bones, binding_count):
    import numpy as np
    import scipy
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve
    from scipy.spatial import cKDTree

    vertices, faces = np.asarray(vertices), np.asarray(faces)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    if not bool(np.isfinite(lengths).all()) or not bool((lengths > 0).all()):
        raise ValueError('source skin graph has nonfinite or zero-length edges')
    conductance = 1 / lengths
    adjacency = coo_matrix((np.r_[conductance, conductance],
                            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                           shape=(len(vertices), len(vertices))).tocsr()
    if connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError('source exterior skin graph is not connected')
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    neighbours = [[] for _ in vertices]
    for (a, b), length in zip(edges, lengths):
        neighbours[a].append((int(b), float(length)))
        neighbours[b].append((int(a), float(length)))
    tree = cKDTree(vertices)

    def closest(point):
        distance, _ = tree.query(point)
        # Resolve source-coordinate ties by source vertex order. The tiny
        # search guard only covers floating-point kd-tree arithmetic.
        ids = sorted(tree.query_ball_point(point, float(distance) + 1e-12))
        squared = np.sum((vertices[ids] - point) ** 2, axis=1)
        return ids[int(np.argmin(squared))]

    targets = []
    projection_gaps = []
    seed_owners = {}
    for anchor_index, bone in enumerate(bones):
        centroid = np.asarray(bone['centroid_world_m'])
        center = closest(centroid)
        gap = float(np.linalg.norm(vertices[center] - centroid))
        radius = bone['diameter_bound_m'] + 2 * gap
        distances = {center: 0.}
        queue = [(0., center)]
        while queue:
            distance, point = heapq.heappop(queue)
            if distance != distances[point]:
                continue
            for neighbour, length in neighbours[point]:
                candidate = distance + length
                if candidate <= radius and candidate < distances.get(neighbour, float('inf')):
                    distances[neighbour] = candidate
                    heapq.heappush(queue, (candidate, neighbour))
        samples = [(0xffffffff, centroid)] + list(zip(bone['sample_vertex_ids'], bone['sample_points_world_m']))
        for source_vertex, point in samples:
            skin_vertex = closest(point)
            admitted = skin_vertex in distances
            targets.append((anchor_index, source_vertex, skin_vertex, bone['binding_index'], int(admitted)))
            projection_gap = float(np.linalg.norm(vertices[skin_vertex] - point))
            if not np.isfinite(projection_gap) or projection_gap <= 0:
                raise ValueError('source bone-to-skin projection gap is not positive and finite')
            projection_gaps.append(projection_gap)
            if admitted:
                seed_owners.setdefault(skin_vertex, {}).setdefault(bone['binding_index'], []).append(1 / projection_gap)
    if {owner for owners in seed_owners.values() for owner in owners} != set(range(binding_count)):
        raise ValueError('source skin seeds do not cover every bound body')
    seed_ids = np.asarray(sorted(seed_owners))
    confidence = np.zeros(len(vertices))
    rhs = np.zeros((len(vertices), binding_count))
    for point in seed_ids:
        values = {owner: float(np.mean(gaps)) for owner, gaps in seed_owners[point].items()}
        confidence[point] = float(np.mean(list(values.values())))
        total = sum(values.values())
        for owner, value in values.items():
            rhs[point, owner] = confidence[point] * value / total
    # Projection confidence and source graph conductance both have units 1/m.
    # A tiny skin edge strengthens continuity, not the source seed penalty.
    operator = (diags(degree + confidence) - adjacency).tocsc()
    full = spsolve(operator, rhs, permc_spec='COLAMD')
    minimum = float(full.min())
    unity_error = float(np.max(np.abs(full.sum(axis=1) - 1)))
    residual = float(np.max(np.abs(operator @ full - rhs)) / max(1., float(np.max(np.abs(rhs)))))
    if not bool(np.isfinite(full).all()) or minimum < 0 or unity_error > 1e-10 or residual > 1e-10:
        raise ValueError('source skin harmonic solution failed positivity, partition or residual checks')
    positive = np.maximum(full, 0)
    quartet = np.argsort(-positive, axis=1, kind='stable')[:, :4]
    weights = np.take_along_axis(positive, quartet, axis=1)
    retained = weights.sum(axis=1)
    weights /= retained[:, None]
    return quartet, weights, full, np.asarray(targets, dtype='<u4'), np.asarray(projection_gaps, dtype='<f8'), {
        'method': METHOD, 'scipy_version': scipy.__version__,
        'source_bone_count': len(bones), 'seed_candidate_count': len(targets),
        'seed_vertex_count': len(seed_ids), 'rejected_seed_count': sum(row[4] == 0 for row in targets),
        'seed_rule': 'centroid_and_64_stratified_source_bone_vertices_projected_to_source_skin; sample must lie within source skin geodesic bone diameter bound plus twice centroid projection gap',
        'screening': 'inverse exact source bone-to-skin projection gap; mean confidence per distinct body then mean across bodies; target shares confidence proportionally; no mesh-edge-dependent seed penalty',
        'minimum_source_projection_gap_m': min(projection_gaps),
        'maximum_source_projection_gap_m': max(projection_gaps),
        'source_graph_edge_count': len(edges), 'relative_solve_residual': residual,
        'maximum_partition_unity_error': unity_error, 'minimum_full_solution_weight': minimum,
        'minimum_retained_four_weight_mass': float(retained.min()),
        'maximum_discarded_weight_mass': float(1 - retained.min()),
        'boundary': 'Inferred visual source-surface association; four-weight fields are diagnostics and the native ABI 5 retains the full field. Not measured skin weights, material stiffness, thickness, mass, contact, mechanical deformation or clinical registration.',
    }


def write_binding_solution(output: Path, full, targets, projection_gaps):
    import numpy as np
    path = output / 'bodyparts3d-skin-binding-solution.npz'
    np.savez_compressed(path, full_weights=np.asarray(full, dtype='<f8'), seed_targets=targets,
                        seed_projection_gaps_m=projection_gaps)
    return {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size, 'full_weight_shape': list(full.shape),
            'seed_target_columns': ['source_anchor_index', 'source_vertex_id_or_centroid_sentinel',
                                    'source_skin_vertex_id', 'binding_index', 'admitted'],
            'runtime_input': False}
