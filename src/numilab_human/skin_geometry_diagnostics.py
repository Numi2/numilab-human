"""Source-bound skin geometry diagnostics, never a mechanical skin owner.

Exact seam correspondence, surface Jacobian singular values and source
topology are distinct from clinical shape admission and physiological strain.
No metric threshold here promotes a surface to anatomical or physical status.
"""
from functools import lru_cache
import json


@lru_cache(maxsize=2)
def _source_topology_json(world_bytes, face_bytes):
    import numpy as np
    from .cardiac_cavity_geometry import analyze_topology
    world = np.frombuffer(world_bytes, '<f8').reshape(-1, 3)
    faces = np.frombuffer(face_bytes, '<u4').reshape(-1, 3)
    unique, mapping = np.unique(world, axis=0, return_inverse=True)
    topology = analyze_topology(unique.tolist(), mapping[faces].tolist())
    keys = ['vertex_count', 'face_count', 'edge_count', 'boundary_edge_count',
            'boundary_loop_count', 'face_component_count', 'euler_characteristic',
            'closed_oriented_manifold_candidate']
    summary = {key: topology[key] for key in keys}
    for key in ['unused_vertex_ids', 'nonmanifold_edges', 'orientation_defect_edges',
                'degenerate_face_ids', 'repeated_vertex_face_ids', 'duplicate_face_ids',
                'vertex_manifold_defect_ids', 'boundary_branch_vertex_ids']:
        summary[key] = topology[key]
    return json.dumps({
        'exact_coordinate_quotient': summary,
        'source_vertex_count': len(world),
        'identified_seam_vertex_count': len(world) - len(unique),
        'identification': 'exact Float64 source world-coordinate equality; no proximity tolerance',
        'payload_geometry_modified': False,
        'closed_volume_admitted': False,
        'self_intersection_status': 'not_checked',
        'boundary': 'Source topology only. Even a closed-oriented candidate does not admit volume, material, contact or clinical anatomy.',
    }, sort_keys=True)


def source_surface_topology(world, faces):
    import numpy as np
    return json.loads(_source_topology_json(np.asarray(world, dtype='<f8').tobytes(),
                                           np.asarray(faces, dtype='<u4').tobytes()))


def source_seam_diagnostics(world, native):
    import numpy as np
    world, native = np.asarray(world, dtype=float), np.asarray(native, dtype=float)
    _, first, mapping, counts = np.unique(world, axis=0, return_index=True,
                                          return_inverse=True, return_counts=True)
    canonical_gaps = np.linalg.norm(native - native[first][mapping], axis=1)
    order = np.argsort(mapping, kind='stable')
    starts = np.r_[0, np.cumsum(counts)]
    pairs = []
    for group in np.flatnonzero(counts > 1):
        members = order[starts[group]:starts[group + 1]]
        a, b = np.triu_indices(len(members), 1)
        pairs.extend(zip(members[a].tolist(), members[b].tolist()))
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    gaps = np.linalg.norm(native[pairs[:, 0]] - native[pairs[:, 1]], axis=1)
    worst = np.argsort(-gaps, kind='stable')[:5]
    return {
        'exact_coincident_vertex_group_count': int((counts > 1).sum()),
        'exact_coincident_redundant_vertex_count': len(world) - len(first),
        'source_seam_pair_count': len(pairs),
        'maximum_native_seam_gap_m': float(gaps.max()) if len(gaps) else 0.,
        'seam_pairs_with_nonzero_native_gap': int((gaps > 0).sum()),
        'seam_vertices_separated_from_canonical': int((canonical_gaps > 0).sum()),
        'worst_source_seam_pairs': [
            {'source_skin_vertex_ids': pairs[i].tolist(),
             'native_gap_m': float(gaps[i])} for i in worst if gaps[i] > 0],
        'boundary': 'Coincident source-point continuity only; source openings, self-intersection, volume and mechanics remain separate.',
    }


def surface_face_diagnostics(world, faces, native, rendered_normals):
    import numpy as np
    world, faces, native = np.asarray(world, dtype=float), np.asarray(faces), np.asarray(native, dtype=float)
    source0 = world[faces[:, 1]] - world[faces[:, 0]]
    source1 = world[faces[:, 2]] - world[faces[:, 0]]
    source_cross = np.cross(source0, source1)
    source_area = np.linalg.norm(source_cross, axis=1) / 2
    current0 = native[faces[:, 1]] - native[faces[:, 0]]
    current1 = native[faces[:, 2]] - native[faces[:, 0]]
    current_cross = np.cross(current0, current1)
    current_area = np.linalg.norm(current_cross, axis=1) / 2
    valid = source_area > 0
    ratio = current_area[valid] / source_area[valid]
    # D maps the source tangent basis into the deformed triangle. Its two
    # singular values are invariant under a global rigid rotation/translation.
    tangent0 = source0[valid] / np.linalg.norm(source0[valid], axis=1)[:, None]
    normal = source_cross[valid] / (2 * source_area[valid, None])
    tangent1 = np.cross(normal, tangent0)
    metric = np.stack([
        np.stack([np.sum(source0[valid] * tangent0, axis=1),
                  np.sum(source1[valid] * tangent0, axis=1)], axis=1),
        np.stack([np.sum(source0[valid] * tangent1, axis=1),
                  np.sum(source1[valid] * tangent1, axis=1)], axis=1),
    ], axis=1)
    gradient = np.stack([current0[valid], current1[valid]], axis=2) @ np.linalg.inv(metric)
    singular = np.linalg.svd(gradient, compute_uv=False)
    normal_sum = np.asarray(rendered_normals, dtype=float)[faces].sum(axis=1)
    opposed = np.sum(current_cross * normal_sum, axis=1) < 0
    original_ids = np.flatnonzero(valid)
    worst = np.argsort(ratio, kind='stable')[:5]
    return {
        'face_count': len(faces),
        'source_degenerate_face_count': int((~valid).sum()),
        'native_collapsed_face_count': int((current_area == 0).sum()),
        'source_surface_area_m2': float(source_area.sum()),
        'native_surface_area_m2': float(current_area.sum()),
        'minimum_area_ratio': float(ratio.min()) if len(ratio) else None,
        'area_ratio_p01': float(np.quantile(ratio, .01)) if len(ratio) else None,
        'area_ratio_p99': float(np.quantile(ratio, .99)) if len(ratio) else None,
        'maximum_area_ratio': float(ratio.max()) if len(ratio) else None,
        'minimum_surface_stretch': float(singular[:, 1].min()) if len(singular) else None,
        'maximum_surface_stretch': float(singular[:, 0].max()) if len(singular) else None,
        'surface_stretch_p01': float(np.quantile(singular[:, 1], .01)) if len(singular) else None,
        'surface_stretch_p99': float(np.quantile(singular[:, 0], .99)) if len(singular) else None,
        'faces_area_below_0_1x': int((ratio < .1).sum()),
        'faces_area_above_10x': int((ratio > 10).sum()),
        'faces_opposed_to_rendered_vertex_normals': int(opposed.sum()),
        'lowest_area_faces': [
            {'source_face_id': int(original_ids[i]),
             'source_skin_vertex_ids': faces[original_ids[i]].tolist(),
             'source_area_m2': float(source_area[original_ids[i]]),
             'area_ratio': float(ratio[i])} for i in worst],
        'boundary': 'All-face geometric area and tangent-map diagnostics; 0.1x/10x counts are descriptive, not physiological thresholds. Normal opposition is a presentation diagnostic, not an inversion/strain certificate. No skin-quality, material or clinical admission.',
    }


def _skin_layer_reference_diagnostics(outer_m, inner_m, deep_m,
                                      skin_thickness_m, fat_thickness_m,
                                      skin_lower_m, skin_upper_m,
                                      fat_lower_m, fat_upper_m,
                                      identity_tolerance_m=2e-5):
    """Check one exact Float32 emission of prepared layer geometry in metres.

    This shares the native constructor's 20-micrometre reference-width identity
    check. Authored per-row bounds remain exact: this helper never substitutes
    global bounds, accepts unknown intervals, or promotes physical admission.
    Self/cross intersections, tetrahedra, mass and pose motion remain separate.
    """
    import numpy as np
    if not np.isfinite(identity_tolerance_m) or identity_tolerance_m < 0:
        raise ValueError('layer identity tolerance must be finite and nonnegative')
    coordinates = [np.asarray(a, dtype=float) for a in (outer_m, inner_m, deep_m)]
    n = len(coordinates[0])
    if n == 0 or any(a.shape != (n, 3) or not np.isfinite(a).all()
                     for a in coordinates):
        raise ValueError('layer coordinates must be finite aligned nonempty Nx3 metres')
    # The native payload emits metres. Casting millimetres and scaling back
    # changes the emitted geometry, especially in thin layers far from origin.
    emitted = [a.astype(np.float32).astype(float) for a in coordinates]
    if any(not np.isfinite(a).all() for a in emitted):
        raise ValueError('layer coordinates overflow Float32 metre emission')
    declarations = [np.asarray(a, dtype=float) for a in
                    (skin_thickness_m, fat_thickness_m)]
    if any(a.shape != (n,) or not np.isfinite(a).all() or np.any(a <= 0)
           for a in declarations):
        raise ValueError('declared layer widths must be finite positive aligned metres')
    intervals = [np.asarray(a, dtype=float) for a in
                 (skin_lower_m, skin_upper_m, fat_lower_m, fat_upper_m)]
    if any(a.shape != (n,) or np.isnan(a).any() for a in intervals):
        raise ValueError('layer intervals must be aligned and not NaN')
    layers = {}
    for name, a, b, declared, lo, hi in zip(
            ('skin', 'fat'), emitted[:2], emitted[1:], declarations,
            intervals[::2], intervals[1::2]):
        width = np.linalg.norm(a - b, axis=1)
        error = np.abs(width - declared)
        unknown = ~np.isfinite(lo) | ~np.isfinite(hi) | (lo <= 0) | (hi < lo)
        mismatch = np.flatnonzero(error > identity_tolerance_m)
        outside = np.flatnonzero(~unknown & ((width < lo) | (width > hi)))
        layers[name] = {
            'declared_width_mismatch_rows': mismatch.tolist(),
            'maximum_declared_width_error_m': float(error.max()),
            'outside_authored_interval_rows': outside.tolist(),
            'unknown_or_invalid_interval_rows': np.flatnonzero(unknown).tolist(),
            'minimum_emitted_width_m': float(width.min()),
            'maximum_emitted_width_m': float(width.max()),
        }
    passed = all(not v[k] for v in layers.values() for k in
                 ('declared_width_mismatch_rows', 'outside_authored_interval_rows',
                  'unknown_or_invalid_interval_rows'))
    return {'reference_width_checks_pass': passed, 'row_count': n,
            'emission': 'coordinates cast once to Float32 metres; differences measured in Float64',
            'identity_tolerance_m': float(identity_tolerance_m), 'layers': layers,
            'physical_admission': False,
            'boundary': 'Reference-width consistency only; no topology, tetrahedron, contact, mass, motion or physical admission.'}

def _skin_source_embedding_diagnostics(source_world_m, source_faces, outer_m,
                                       source_vertex_ids, embedding_faces,
                                       barycentric, residual_m,
                                       reconstruction_tolerance_m=2e-6,
                                       displacement_caps_m=None):
    """Validate prepared direct/projected ancestry against emitted source points.

    Derived points may share a one-hot source sample, but must use the projected
    identity sentinel. A direct identity remains unique. Residuals always mean
    outer minus the sampled registered source in metres; raw atlas coordinates
    and previous mechanical-delta fields are not interchangeable.
    """
    import numpy as np
    xyz = [np.asarray(a, dtype=float) for a in (source_world_m, outer_m)]
    if any(a.ndim != 2 or a.shape[1] != 3 or not len(a) or
           not np.isfinite(a).all() for a in xyz):
        raise ValueError('source and outer coordinates must be finite nonempty Nx3 metres')
    source, outer = [a.astype(np.float32).astype(float) for a in xyz]
    if any(not np.isfinite(a).all() for a in (source, outer)):
        raise ValueError('source coordinates overflow Float32 metre emission')
    n = len(outer)
    integer_inputs = [np.asarray(a) for a in
                      (source_faces, source_vertex_ids, embedding_faces)]
    if any(not np.issubdtype(a.dtype, np.integer) for a in integer_inputs):
        raise ValueError('source topology and identities must have integer types')
    faces, ids, ef = [a.astype(np.int64) for a in integer_inputs]
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or
            np.any(faces < 0) or np.any(faces >= len(source)) or
            np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0)):
        raise ValueError('source faces must be nonempty valid distinct index triples')
    if ids.shape != (n,) or ef.shape != (n,):
        raise ValueError('source identity rows must align with outer geometry')
    ids = np.where(ids == 0xffffffff, -1, ids)
    ef = np.where(ef == 0xffffffff, -1, ef)
    if np.any(ids < -1) or np.any(ids >= len(source)) or np.any(ef < -1) or np.any(ef >= len(faces)):
        raise ValueError('source identity or embedding face is outside source topology')
    beta = np.asarray(barycentric, dtype=float)
    residual = np.asarray(residual_m, dtype=float)
    if (beta.shape != (n, 3) or residual.shape != (n, 3) or
            not np.isfinite(beta).all() or not np.isfinite(residual).all()):
        raise ValueError('barycentric and residual rows must be finite aligned Nx3')
    if not np.isfinite(reconstruction_tolerance_m) or reconstruction_tolerance_m < 0:
        raise ValueError('reconstruction tolerance must be finite and nonnegative')
    direct, embedded = ids >= 0, ef >= 0
    unique, counts = np.unique(ids[direct], return_counts=True)
    duplicate_ids = unique[counts > 1]
    duplicates = np.flatnonzero(direct & np.isin(ids, duplicate_ids))
    missing = np.flatnonzero(~direct & ~embedded)
    invalid_beta = embedded & ((np.abs(beta.sum(1)-1.) > 1e-8) |
                               np.any(beta < -1e-12, axis=1) |
                               np.any(beta > 1.+1e-12, axis=1))
    sampled = np.zeros_like(outer)
    sampled[direct] = source[ids[direct]]
    valid_embedded = embedded & ~invalid_beta
    sampled[valid_embedded] = np.einsum(
        'ni,nij->nj', beta[valid_embedded], source[faces[ef[valid_embedded]]])
    both = direct & embedded & ~invalid_beta
    one_hot_identity_bad = np.zeros(n, dtype=bool)
    for row in np.flatnonzero(both):
        exact = (faces[ef[row]] == ids[row]) & (np.abs(beta[row]-1.) <= 1e-8)
        one_hot_identity_bad[row] = not np.any(exact)
    valid = (direct | embedded) & ~invalid_beta & ~one_hot_identity_bad
    error = np.linalg.norm(outer - sampled - residual, axis=1)
    mismatch = np.flatnonzero(valid & (error > reconstruction_tolerance_m))
    cap_bad = np.array([], dtype=int)
    if displacement_caps_m is not None:
        caps = np.asarray(displacement_caps_m, dtype=float)
        if caps.shape != (n,) or not np.isfinite(caps).all() or np.any(caps <= 0):
            raise ValueError('displacement caps must be finite positive aligned metres')
        cap_bad = np.flatnonzero(valid & (np.linalg.norm(outer-sampled, axis=1) > caps+1e-7))
    violations = {
        'duplicate_direct_identity_rows': duplicates.tolist(),
        'duplicate_direct_source_ids': duplicate_ids.tolist(),
        'missing_source_identity_rows': missing.tolist(),
        'invalid_barycentric_rows': np.flatnonzero(invalid_beta).tolist(),
        'direct_embedding_identity_mismatch_rows': np.flatnonzero(one_hot_identity_bad).tolist(),
        'registered_residual_mismatch_rows': mismatch.tolist(),
        'outside_declared_displacement_cap_rows': cap_bad.tolist(),
    }
    return {'source_embedding_checks_pass': not any(violations.values()),
            'row_count': n, 'direct_rows': int(direct.sum()),
            'projected_rows': int((~direct & embedded).sum()), **violations,
            'maximum_registered_residual_error_m': float(error[valid].max()) if valid.any() else None,
            'physical_admission': False,
            'boundary': 'Source ancestry and registered residual consistency only; no cells, contact, mass or motion admission.'}
