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
FULL_METHOD = 'positive_source_surface_projection_gap_screened_harmonic_full_weight.v2'
METHOD = 'positive_source_surface_exact_seam_projection_gap_full_weight.v3'

DIRICHLET_METHOD = 'positive_source_surface_dirichlet_harmonic_full_weight.v4'


def solve_source_weight_field(adjacency, confidence, rhs, *, seed_condition='screened'):
    """Solve inferred binding weights; this is an offline construction step.

    The optional Dirichlet condition preserves each admitted source projection's
    normalized owner shares exactly. It introduces no stiffness or runtime
    forces, and does not assert that the resulting geometry is collision-free.
    """
    import numpy as np
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    if seed_condition not in ('screened', 'dirichlet'):
        raise ValueError('unknown source skin seed condition')
    confidence, rhs = np.asarray(confidence), np.asarray(rhs)
    n = adjacency.shape[0]
    if (adjacency.shape != (n, n) or confidence.shape != (n,)
            or rhs.ndim != 2 or rhs.shape[0] != n
            or not np.isfinite(confidence).all() or not np.isfinite(rhs).all()
            or (confidence < 0).any() or (rhs < 0).any()
            or not np.allclose(rhs.sum(axis=1), confidence, rtol=1e-12, atol=1e-14)):
        raise ValueError('invalid source skin projection constraints')
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if seed_condition == 'screened':
        # Keep the original operator and solve arithmetic for the default path.
        operator = (diags(degree + confidence) - adjacency).tocsc()
        full = spsolve(operator, rhs, permc_spec='COLAMD')
        residual = float(np.max(np.abs(operator @ full - rhs))
                         / max(1., float(np.max(np.abs(rhs)))))
        boundary_error = None
    else:
        from scipy.sparse.csgraph import connected_components
        if connected_components(adjacency, directed=False, return_labels=False) != 1:
            raise ValueError('Dirichlet source skin graph must be connected')
        fixed = np.flatnonzero(confidence > 0)
        if not len(fixed):
            raise ValueError('Dirichlet source skin needs source projection anchors')
        free = np.flatnonzero(confidence == 0)
        full = np.zeros_like(rhs, dtype=np.float64)
        full[fixed] = rhs[fixed] / confidence[fixed, None]
        laplacian = (diags(degree) - adjacency).tocsr()
        residual = 0.
        if len(free):
            operator = laplacian[free][:, free].tocsc()
            boundary_rhs = -laplacian[free][:, fixed] @ full[fixed]
            full[free] = spsolve(operator, boundary_rhs, permc_spec='COLAMD')
            residual = float(np.max(np.abs(operator @ full[free] - boundary_rhs))
                             / max(1., float(np.max(np.abs(boundary_rhs)))))
        boundary_error = float(np.max(np.abs(
            full[fixed] - rhs[fixed] / confidence[fixed, None])))
    minimum = float(full.min())
    unity_error = float(np.max(np.abs(full.sum(axis=1) - 1)))
    if (not np.isfinite(full).all() or minimum < 0
            or unity_error > 1e-10 or residual > 1e-10):
        raise ValueError('source skin harmonic solution failed positivity, partition or residual checks')
    return full, {
        'relative_solve_residual': residual,
        'maximum_partition_unity_error': unity_error,
        'minimum_full_solution_weight': minimum,
        'seed_condition': seed_condition,
        'maximum_fixed_seed_weight_error': boundary_error,
    }


def source_surface_binding(vertices, faces, bones, binding_count, *, seed_condition='screened'):
    import numpy as np
    import scipy
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    vertices, faces = np.asarray(vertices), np.asarray(faces)
    # Exact-coordinate duplicates are one material point for visual binding.
    # Keep all source vertices/faces in the emitted payload; do not merge near
    # points, cap source boundaries, or invent mechanical skin connectivity.
    graph_vertices, inverse, multiplicity = np.unique(
        vertices, axis=0, return_inverse=True, return_counts=True)
    graph_faces = inverse[faces]
    edges = np.unique(np.sort(np.concatenate([graph_faces[:, [0, 1]], graph_faces[:, [1, 2]],
                                             graph_faces[:, [2, 0]]]), axis=1), axis=0)
    lengths = np.linalg.norm(graph_vertices[edges[:, 0]] - graph_vertices[edges[:, 1]], axis=1)
    if not bool(np.isfinite(lengths).all()) or not bool((lengths > 0).all()):
        raise ValueError('source skin graph has nonfinite or zero-length edges')
    conductance = 1 / lengths
    adjacency = coo_matrix((np.r_[conductance, conductance],
                            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                           shape=(len(graph_vertices), len(graph_vertices))).tocsr()
    if connected_components(adjacency, directed=False, return_labels=False) != 1:
        raise ValueError('source exterior skin graph is not connected')
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    neighbours = [[] for _ in graph_vertices]
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
        center = int(inverse[center])
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
            graph_vertex = int(inverse[skin_vertex])
            admitted = graph_vertex in distances
            targets.append((anchor_index, source_vertex, skin_vertex, bone['binding_index'], int(admitted)))
            projection_gap = float(np.linalg.norm(vertices[skin_vertex] - point))
            if not np.isfinite(projection_gap) or projection_gap <= 0:
                raise ValueError('source bone-to-skin projection gap is not positive and finite')
            projection_gaps.append(projection_gap)
            if admitted:
                seed_owners.setdefault(graph_vertex, {}).setdefault(bone['binding_index'], []).append(1 / projection_gap)
    if {owner for owners in seed_owners.values() for owner in owners} != set(range(binding_count)):
        raise ValueError('source skin seeds do not cover every bound body')
    seed_ids = np.asarray(sorted(seed_owners))
    confidence = np.zeros(len(graph_vertices))
    rhs = np.zeros((len(graph_vertices), binding_count))
    for point in seed_ids:
        values = {owner: float(np.mean(gaps)) for owner, gaps in seed_owners[point].items()}
        confidence[point] = float(np.mean(list(values.values())))
        total = sum(values.values())
        for owner, value in values.items():
            rhs[point, owner] = confidence[point] * value / total
    # Projection confidence and source graph conductance both have units 1/m.
    # A tiny skin edge strengthens continuity, not the source seed penalty.
    full, solve = solve_source_weight_field(
        adjacency, confidence, rhs, seed_condition=seed_condition)
    minimum = solve['minimum_full_solution_weight']
    unity_error = solve['maximum_partition_unity_error']
    residual = solve['relative_solve_residual']
    # Expand by exact source identity. Every duplicate receives identical
    # weights, including the same float32 quantization at the native boundary.
    full = full[inverse]
    positive = np.maximum(full, 0)
    quartet = np.argsort(-positive, axis=1, kind='stable')[:, :4]
    weights = np.take_along_axis(positive, quartet, axis=1)
    retained = weights.sum(axis=1)
    weights /= retained[:, None]
    return quartet, weights, full, np.asarray(targets, dtype='<u4'), np.asarray(projection_gaps, dtype='<f8'), {
        'method': METHOD if seed_condition == 'screened' else DIRICHLET_METHOD,
        'scipy_version': scipy.__version__,
        'source_bone_count': len(bones), 'seed_candidate_count': len(targets),
        **({'seed_condition': seed_condition,
            'maximum_fixed_seed_weight_error': solve['maximum_fixed_seed_weight_error']}
           if seed_condition == 'dirichlet' else {}),
        'seed_vertex_count': len(seed_ids), 'rejected_seed_count': sum(row[4] == 0 for row in targets),
        'seed_rule': 'centroid_and_64_stratified_source_bone_vertices_projected_to_source_skin; sample must lie within source skin geodesic bone diameter bound plus twice centroid projection gap',
        'screening': (
            'inverse exact source bone-to-skin projection gap; mean confidence per distinct body then mean across bodies; target shares confidence proportionally; no mesh-edge-dependent seed penalty'
            if seed_condition == 'screened' else
            'Dirichlet owner shares from the same admitted inverse-gap source projections; harmonic interpolation of unanchored graph rows; no soft seed penalty'),
        'minimum_source_projection_gap_m': min(projection_gaps),
        'maximum_source_projection_gap_m': max(projection_gaps),
        'source_graph_edge_count': len(edges), 'relative_solve_residual': residual,
        'source_graph_vertex_count': len(graph_vertices),
        'exact_coincident_vertex_group_count': int((multiplicity > 1).sum()),
        'exact_coincident_redundant_vertex_count': len(vertices) - len(graph_vertices),
        'seam_policy': 'exact source world-coordinate equality only; solve quotient graph and expand to unchanged source vertex order; no proximity tolerance or hole capping',
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
