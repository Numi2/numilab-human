"""Condition narrow respiratory source/cut slivers without moving vertices.

The established respiratory-field operation is limited to one affine cell.
The separately named pre-conform CSG path is restricted to source-identified
cut fragments with matching provenance on every reciprocal surface owner.
Neither path is a time-dependent tissue solver.
"""
from __future__ import annotations

import numpy as np


def improve_sliver_faces(prepared, *, minimum_altitude_m=2.5e-7,
                         maximum_nonplanarity_m=1e-7, spacing_m=1/64):
    """Preserve the original same-field-cell respiratory conditioning path."""
    return _improve_sliver_faces(
        prepared, minimum_altitude_m=minimum_altitude_m,
        maximum_nonplanarity_m=maximum_nonplanarity_m, spacing_m=spacing_m,
        eligible_faces=None, require_same_field_cell=True,
        minimum_result_altitude_m=minimum_altitude_m,
        require_strict_improvement=False,
        method='shared_reciprocal_diagonal_flip_within_one_affine_field_cell',
    )


def improve_registered_surface_sliver_faces(prepared, *, minimum_altitude_m=1.2e-6,
                                           maximum_nonplanarity_m=1e-5,
                                           minimum_result_altitude_m=None):
    """Flip bounded near-planar slivers across numerical field-cell seams.

    Respiratory affine cells are an evaluation detail, not anatomical
    boundaries.  This explicit source-registration path keeps the existing
    reciprocal-owner, orientation, lineage, and 100 nm nonplanarity checks,
    while allowing a diagonal to cross a cell seam.  It moves no vertices and
    requires strict improvement. By default, both replacement triangles must
    meet ``minimum_altitude_m``. An explicit lower floor can be used for a
    bounded repair pass that improves a fragile face without claiming it has
    reached the target.
    """
    if minimum_altitude_m not in (2.5e-7, 1.2e-6):
        raise ValueError('unsupported registered-surface altitude target')
    if maximum_nonplanarity_m not in (1e-7, 1e-5):
        raise ValueError('unsupported registered-surface nonplanarity bound')
    if minimum_result_altitude_m is None:
        minimum_result_altitude_m = minimum_altitude_m
    if minimum_result_altitude_m not in (1.25e-7, 2.5e-7, 1.2e-6):
        raise ValueError('unsupported registered-surface result floor')
    if minimum_result_altitude_m > minimum_altitude_m:
        raise ValueError('registered-surface result floor exceeds the target')
    return _improve_sliver_faces(
        prepared, minimum_altitude_m=minimum_altitude_m,
        maximum_nonplanarity_m=maximum_nonplanarity_m, spacing_m=1/64,
        eligible_faces=None, require_same_field_cell=False,
        minimum_result_altitude_m=minimum_result_altitude_m,
        require_strict_improvement=True,
        method='shared_reciprocal_diagonal_flip_across_registered_field_cells',
    )


def collapse_registered_interior_vertex(prepared, keep_point, drop_point, *,
                                        maximum_source_displacement_m=1e-4,
                                        minimum_altitude_m=1.2e-6,
                                        maximum_volume_change_m3=1e-9,
                                        volume_bounded_surface_ids=(),
                                        volume_bounded_surface_groups=(),
                                        cumulative_volume_change_m3=None,
                                        cumulative_volume_group_change_m3=None,
                                        source_ancestry_by_point=None,
                                        complete_point_owners=None,
                                        same_winding_surface_pairs=(),
                                        allow_equal_owner_sets=False):
    """Collapse an interior point onto an existing shared seam point.

    This is the directed counterpart of a midpoint collapse: ``keep_point``
    must already exist on every surface that owns it, while ``drop_point``
    may belong to a strict subset of those owners.  Only owners of the dropped
    point are changed. With ``allow_equal_owner_sets=True``, both endpoints
    may have the same owner set; all owners are changed together and every
    reciprocal triangle is checked. The operation checks the closed-edge link condition,
    local orientation and quality, bounded signed-volume change, and exact
    reciprocal triangle sets around the seam.  An explicitly declared
    coincident representation pair (such as the derived visceral-pleura
    exterior) may require matching winding; every other changed interface
    requires opposite winding.  Thus an owner that has the
    retained seam point but not the dropped interior point keeps its exact
    boundary mesh.

    ``prepared`` uses the existing respiratory mesh owner convention:
    ``stable_id -> (vertices[N,6], faces[M,3], source_parent_to_children,
    detail)``.  It is mutated only after every participant passes.  Optional
    ancestry and cumulative-volume dictionaries are updated in place so a
    sequence of collapses can retain a bound to original coordinates.
    """
    keep_key = tuple(map(float, np.asarray(keep_point, dtype=np.float32)[:3]))
    drop_key = tuple(map(float, np.asarray(drop_point, dtype=np.float32)[:3]))
    if keep_key == drop_key:
        raise ValueError('interior collapse endpoints must be distinct')
    if (maximum_source_displacement_m <= 0 or minimum_altitude_m <= 0 or
            maximum_volume_change_m3 < 0):
        raise ValueError('collapse bounds must be positive')

    point_rows = {}
    point_owner_rows = {keep_key: {}, drop_key: {}}
    for raw_sid, row in prepared.items():
        sid = int(raw_sid)
        if len(row) != 4:
            raise ValueError(f'prepared surface {sid} must have four fields')
        vertices, faces, parent_map, detail = row
        vertices = np.asarray(vertices)
        faces = np.asarray(faces, dtype=np.int64)
        if (vertices.ndim != 2 or vertices.shape[1] not in (3, 6) or
                faces.ndim != 2 or faces.shape[1] != 3 or
                (faces.size and (faces.min() < 0 or faces.max() >= len(vertices)))):
            raise ValueError(f'prepared surface {sid} is malformed')
        coords = {}
        xyz = np.asarray(vertices[:, :3], dtype=np.float32)
        for key in (keep_key, drop_key):
            matches = np.flatnonzero(np.all(xyz == np.asarray(key, dtype=np.float32), axis=1))
            if len(matches) > 1:
                raise ValueError(f'surface {sid} contains a duplicate collapse endpoint')
            if len(matches) == 1:
                coords[key] = int(matches[0])
                point_owner_rows[key][sid] = int(matches[0])
        point_rows[sid] = (vertices, faces, parent_map, detail, coords)

    keep_owners = set(point_owner_rows.get(keep_key, {}))
    drop_owners = set(point_owner_rows.get(drop_key, {}))
    owner_sets_equal = drop_owners == keep_owners
    if (not drop_owners or not drop_owners <= keep_owners or
            (owner_sets_equal and not allow_equal_owner_sets)):
        requirement = 'a subset' if allow_equal_owner_sets else 'a strict subset'
        raise ValueError(f'drop-point owners must be {requirement} of keep-point owners')
    if complete_point_owners is not None:
        for key, actual in ((keep_key, keep_owners), (drop_key, drop_owners)):
            declared = set(map(int, complete_point_owners.get(key, ())))
            if declared != actual:
                raise ValueError('prepared surfaces do not contain every declared exact point owner')

    edge_length = float(np.linalg.norm(np.asarray(keep_key) - np.asarray(drop_key)))
    ancestry = source_ancestry_by_point
    dropped_origins = ({tuple(map(float, point)) for point in ancestry.get(drop_key, {drop_key})}
                       if ancestry is not None else {drop_key})
    source_displacement = max(float(np.linalg.norm(np.asarray(keep_key) - np.asarray(point)))
                              for point in dropped_origins)
    if source_displacement > maximum_source_displacement_m:
        raise ValueError('collapse exceeds the cumulative source displacement bound')

    cumulative = cumulative_volume_change_m3
    if cumulative is None:
        cumulative = {sid: 0.0 for sid in point_rows}
    else:
        cumulative = {int(sid): float(value) for sid, value in cumulative.items()}
        for sid in point_rows:
            cumulative.setdefault(sid, 0.0)
    bounded_ids = set(map(int, volume_bounded_surface_ids))
    bounded_groups = tuple(tuple(sorted(set(map(int, group))))
                           for group in volume_bounded_surface_groups)
    if any(not group or any(sid not in point_rows for sid in group) for group in bounded_groups):
        raise ValueError('volume-bounded surface groups must name existing nonempty surfaces')
    if len(set(bounded_groups)) != len(bounded_groups):
        raise ValueError('duplicate volume-bounded surface group')
    cumulative_groups = ({str(key): float(value) for key, value in
                          cumulative_volume_group_change_m3.items()}
                         if cumulative_volume_group_change_m3 is not None else {})
    same_winding_pairs = {tuple(sorted(map(int, pair))) for pair in same_winding_surface_pairs}
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in same_winding_pairs):
        raise ValueError('same-winding representation pairs must name two distinct surfaces')
    proposed = {}
    row_deltas = {}
    touched_keys = {keep_key, drop_key}
    score_before_total = 0.0
    score_after_total = 0.0
    minimum_before_total = float('inf')
    minimum_after_total = float('inf')

    def volume(vertices, faces):
        if not len(faces):
            return 0.0
        points = np.asarray(vertices)[np.asarray(faces, dtype=np.int64), :3].astype(np.float64)
        return float(np.einsum('ij,ij->i', points[:, 0],
                                np.cross(points[:, 1], points[:, 2])).sum() / 6.0)

    def altitudes(vertices, faces):
        points = np.asarray(vertices)[np.asarray(faces, dtype=np.int64), :3].astype(np.float64)
        normal = np.linalg.norm(np.cross(points[:, 1] - points[:, 0],
                                         points[:, 2] - points[:, 0]), axis=1)
        lengths = np.stack((np.linalg.norm(points[:, 1] - points[:, 2], axis=1),
                            np.linalg.norm(points[:, 2] - points[:, 0], axis=1),
                            np.linalg.norm(points[:, 0] - points[:, 1], axis=1)), axis=1)
        return normal / np.maximum(lengths.max(axis=1), 1e-30)

    def face_key(vertices, face):
        return tuple(sorted(tuple(map(float, np.asarray(vertices)[int(i), :3])) for i in face))

    def oriented_key(vertices, face):
        points = tuple(tuple(map(float, np.asarray(vertices)[int(i), :3])) for i in face)
        return min(points, points[1:] + points[:1], points[2:] + points[:2])

    def local_faces(vertices, faces, coords, keys):
        ids = {coords[key] for key in keys if key in coords}
        if not ids:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(np.isin(faces, np.fromiter(ids, dtype=np.int64)).any(axis=1))

    def triangle_map(vertices, faces, coords, keys):
        indices = local_faces(vertices, faces, coords, keys)
        mapped = {}
        for face_index in indices:
            key = face_key(vertices, faces[face_index])
            mapped.setdefault(key, []).append(oriented_key(vertices, faces[face_index]))
        return mapped

    def endpoint_coordinates(vertices, keys):
        xyz = np.asarray(vertices[:, :3], dtype=np.float32)
        result = {}
        for point in keys:
            matches = np.flatnonzero(np.all(xyz == np.asarray(point, dtype=np.float32), axis=1))
            if len(matches) > 1:
                raise ValueError('proposed surface contains a duplicate seam point')
            if len(matches) == 1:
                result[point] = int(matches[0])
        return result

    for sid in sorted(drop_owners):
        vertices, faces, parent_map, detail, coords = point_rows[sid]
        keep_index = coords[keep_key]
        drop_index = coords[drop_key]
        incident_keep = np.flatnonzero(np.any(faces == keep_index, axis=1))
        incident_drop = np.flatnonzero(np.any(faces == drop_index, axis=1))
        edge_faces = np.intersect1d(incident_keep, incident_drop, assume_unique=True)
        if len(edge_faces) != 2:
            raise ValueError(f'surface {sid} does not have a closed two-face collapse edge')
        affected = np.union1d(incident_keep, incident_drop)
        local = faces[affected]
        neighbor_keep, neighbor_drop, edge_opposites = set(), set(), set()
        for tri in local:
            tri_set = set(map(int, tri))
            if keep_index in tri_set:
                neighbor_keep.update(tri_set - {keep_index, drop_index})
            if drop_index in tri_set:
                neighbor_drop.update(tri_set - {keep_index, drop_index})
            if keep_index in tri_set and drop_index in tri_set:
                edge_opposites.update(tri_set - {keep_index, drop_index})
        if neighbor_keep & neighbor_drop != edge_opposites or len(edge_opposites) != 2:
            raise ValueError(f'surface {sid} fails the closed-edge link condition')

        remove_mask = np.ones(len(faces), dtype=bool)
        remove_mask[edge_faces] = False
        kept_faces = faces[remove_mask].copy()
        kept_faces[kept_faces == drop_index] = keep_index
        old_to_new_vertex = np.arange(len(vertices), dtype=np.int64)
        old_to_new_vertex = np.delete(old_to_new_vertex, drop_index)
        kept_faces[kept_faces > drop_index] -= 1
        kept_faces = kept_faces.astype(np.int64, copy=False)
        keep_new = keep_index - (1 if keep_index > drop_index else 0)
        new_vertices = np.delete(vertices, drop_index, axis=0).copy()
        new_local = np.flatnonzero(np.any(kept_faces == keep_new, axis=1))
        local_keys = [tuple(sorted(map(int, tri))) for tri in kept_faces[new_local]]
        if len(local_keys) != len(set(local_keys)):
            raise ValueError(f'surface {sid} collapse creates a duplicate face')

        old_local = affected
        old_local_kept = old_local[remove_mask[old_local]]
        old_normals = np.cross(vertices[faces[old_local_kept, 1], :3].astype(float) -
                               vertices[faces[old_local_kept, 0], :3].astype(float),
                               vertices[faces[old_local_kept, 2], :3].astype(float) -
                               vertices[faces[old_local_kept, 0], :3].astype(float))
        old_to_new_face = np.full(len(faces), -1, dtype=np.int64)
        old_to_new_face[remove_mask] = np.arange(np.count_nonzero(remove_mask), dtype=np.int64)
        new_local_for_old = old_to_new_face[old_local_kept]
        new_triangles = kept_faces[new_local_for_old]
        new_points = new_vertices[new_triangles, :3].astype(float)
        new_normals = np.cross(new_points[:, 1] - new_points[:, 0], new_points[:, 2] - new_points[:, 0])
        if np.any(np.linalg.norm(new_normals, axis=1) == 0) or np.any(
                np.einsum('ij,ij->i', old_normals, new_normals) <= 0):
            raise ValueError(f'surface {sid} collapse changes local orientation or creates zero area')

        old_score_alt = altitudes(vertices, faces[affected])
        new_score_alt = altitudes(new_vertices, kept_faces[new_local])
        def score(values):
            deficit = np.maximum(0.0, np.log(minimum_altitude_m / np.maximum(values, 1e-30)))
            return float(np.dot(deficit, deficit))
        before_score = score(old_score_alt)
        after_score = score(new_score_alt)
        before_min = float(old_score_alt.min())
        after_min = float(new_score_alt.min()) if len(new_score_alt) else float('inf')
        if after_min < before_min - 1e-12:
            raise ValueError(f'surface {sid} collapse regresses the local minimum altitude')
        score_before_total += before_score
        score_after_total += after_score
        minimum_before_total = min(minimum_before_total, before_min)
        minimum_after_total = min(minimum_after_total, after_min)

        delta = volume(new_vertices, kept_faces[new_local]) - volume(vertices, faces[affected])
        if sid in bounded_ids and abs(cumulative.get(sid, 0.0) + delta) > maximum_volume_change_m3:
            raise ValueError(f'surface {sid} collapse exceeds cumulative signed-volume bound')

        new_parent_map = {int(parent): [] for parent in parent_map}
        affected_set = set(map(int, affected))
        affected_lineage = set()
        for parent, children in parent_map.items():
            for child in children:
                child = int(child)
                if child < 0 or child >= len(faces):
                    raise ValueError(f'surface {sid} has invalid face lineage')
                if child in affected_set:
                    affected_lineage.add(child)
                if remove_mask[child]:
                    new_parent_map[int(parent)].append(int(old_to_new_face[child]))
        if affected_lineage != affected_set:
            raise ValueError(f'surface {sid} local faces have incomplete source lineage')
        proposed[sid] = (new_vertices, kept_faces, new_parent_map, detail, delta,
                         before_score, after_score, before_min, after_min,
                         int(len(edge_faces)), int(len(affected)))
        row_deltas[sid] = delta

    if score_after_total >= score_before_total - 1e-15:
        raise ValueError('collapse does not reduce the combined local sliver objective')
    if minimum_after_total < minimum_before_total - 1e-12:
        raise ValueError('collapse regresses the combined local minimum altitude')
    group_deltas = {}
    for group in bounded_groups:
        group_key = '+'.join(map(str, group))
        delta = sum(row_deltas.get(sid, 0.0) for sid in group)
        group_deltas[group_key] = delta
        if abs(cumulative_groups.get(group_key, 0.0) + delta) > maximum_volume_change_m3:
            raise ValueError(f'surface group {group_key} exceeds cumulative signed-volume bound')

    # Compare exact local triangle sets against every unchanged owner of the
    # retained seam.  Shared surfaces that also own the dropped point must
    # transform the same canonical patch, with opposite winding on each side.
    pair_reports = []
    seam_pairs = {
        tuple(sorted((participant, other)))
        for participant in drop_owners
        for other in keep_owners
        if participant != other
    }
    for first, second in sorted(seam_pairs):
        # Each pair is checked once even when both surfaces are changed.
        # Pick a changed owner as the participant and treat its counterpart
        # as either another changed owner or an unchanged seam owner.
        participant = first if first in drop_owners else second
        other = second if participant == first else first
        old_v, old_f, _, _, old_coords = point_rows[participant]
        new_v, new_f, *_ = proposed[participant]
        new_coords = endpoint_coordinates(new_v, (keep_key,))
        old_map = triangle_map(old_v, old_f, old_coords, touched_keys)
        new_map = triangle_map(new_v, new_f, new_coords, {keep_key})
        old_ov, old_of, _, _, old_oc = point_rows[other]
        old_other = triangle_map(old_ov, old_of, old_oc, touched_keys)
        before_shared = set(old_map) & set(old_other)
        if not before_shared:
            continue
        expected = set()
        if other in drop_owners:
            for tri_key in before_shared:
                transformed = tuple(sorted(keep_key if point == drop_key else point
                                           for point in tri_key))
                if len(set(transformed)) == 3:
                    expected.add(transformed)
        else:
            if any(drop_key in tri_key for tri_key in before_shared):
                raise ValueError('dropped point is unexpectedly shared with an unchanged owner')
            expected = before_shared
        if other in proposed:
            other_v, other_f, *_ = proposed[other]
            other_coords = endpoint_coordinates(other_v, (keep_key,))
            after_other = triangle_map(other_v, other_f, other_coords, {keep_key})
        else:
            after_other = old_other
        after_shared = set(new_map) & set(after_other)
        if after_shared != expected:
            raise ValueError(f'exact exposed seam triangle set changes between {participant} and {other}')
        if other in drop_owners:
            for tri_key in after_shared:
                a_orient = new_map[tri_key]
                b_orient = after_other[tri_key]
                if len(a_orient) != len(b_orient):
                    raise ValueError('reciprocal triangle multiplicity differs after collapse')
                same_winding = tuple(sorted((participant, other))) in same_winding_pairs
                for a, b in zip(sorted(a_orient), sorted(b_orient)):
                    reverse = min((b[0], b[2], b[1]), (b[2], b[1], b[0]), (b[1], b[0], b[2]))
                    if (same_winding and a != b) or (not same_winding and a != reverse):
                        expected_winding = 'matching' if same_winding else 'opposite'
                        raise ValueError(f'reciprocal triangle winding is not {expected_winding} after collapse')
        pair_reports.append({'participant': participant, 'other_owner': other,
                             'shared_triangles_before': len(before_shared),
                             'shared_triangles_after': len(after_shared),
                             'unchanged_owner_patch_preserved': other not in drop_owners})

    ready = {}
    for sid, proposal in proposed.items():
        new_vertices, new_faces, new_parents, detail, delta, *_ = proposal
        if new_vertices.shape[1] == 6:
            points = new_vertices[new_faces, :3].astype(float)
            face_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
            summed = np.zeros((len(new_vertices), 3), dtype=float)
            for axis in range(3):
                np.add.at(summed, new_faces[:, axis], face_normals)
            lengths = np.linalg.norm(summed, axis=1)
            used = np.unique(new_faces)
            if np.any(lengths[used] == 0) or not np.all(np.isfinite(lengths[used])):
                raise ValueError(f'surface {sid} has a zero or nonfinite vertex normal after collapse')
            new_vertices[used, 3:6] = (summed[used] / lengths[used, None]).astype(new_vertices.dtype)
        records = list(detail.get('registered_owner_collapses', []))
        records.append({'keep_point': keep_key, 'drop_point': drop_key,
                        'removed_edge_face_count': proposal[10]})
        new_detail = dict(detail)
        new_detail['registered_owner_collapses'] = records
        ready[sid] = (new_vertices, new_faces, new_parents, new_detail)
    for sid, row in ready.items():
        prepared[sid] = row
        cumulative[sid] = cumulative.get(sid, 0.0) + row_deltas[sid]
    for group_key, delta in group_deltas.items():
        cumulative_groups[group_key] = cumulative_groups.get(group_key, 0.0) + delta

    if source_ancestry_by_point is not None:
        kept_origins = {tuple(map(float, point)) for point in ancestry.get(keep_key, {keep_key})}
        source_ancestry_by_point[keep_key] = kept_origins | set(dropped_origins)
        source_ancestry_by_point.pop(drop_key, None)
    return {
        'method': 'directed_interior_vertex_collapse_onto_existing_shared_seam',
        'keep_point': keep_key,
        'drop_point': drop_key,
        'edge_length_m': edge_length,
        'cumulative_source_displacement_m': source_displacement,
        'keep_point_owners': sorted(keep_owners),
        'drop_point_owners': sorted(drop_owners),
        'owner_set_relation': ('equal' if owner_sets_equal else 'strict_subset'),
        'changed_surface_ids': sorted(drop_owners),
        'unchanged_seam_owners': sorted(keep_owners - drop_owners),
        'combined_local_sliver_objective_before': score_before_total,
        'combined_local_sliver_objective_after': score_after_total,
        'combined_local_minimum_altitude_before_m': minimum_before_total,
        'combined_local_minimum_altitude_after_m': minimum_after_total,
        'per_surface_signed_volume_delta_m3': {str(sid): value for sid, value in row_deltas.items()},
        'cumulative_volume_change_m3': {str(sid): cumulative[sid] for sid in sorted(row_deltas)},
        'signed_volume_delta_by_group_m3': group_deltas,
        'cumulative_volume_group_change_m3': {
            key: cumulative_groups[key] for key in sorted(group_deltas)},
        'seam_checks': pair_reports,
        'qualification': 'one bounded local source-conditioning operation; final topology, full interface, self-intersection and native accepted-cycle audits are separate',
    }


def improve_registered_interior_slivers(prepared, *, minimum_altitude_m=1.2e-6,
                                        maximum_source_displacement_m=1e-4,
                                        maximum_volume_change_m3=1e-9,
                                        volume_bounded_surface_ids=(),
                                        volume_bounded_surface_groups=(),
                                        same_winding_surface_pairs=(),
                                        maximum_operations=64,
                                        maximum_attempts_per_pass=128,
                                        cumulative_volume_change_m3=None,
                                        cumulative_volume_group_change_m3=None,
                                        source_ancestry_by_point=None,
                                        allow_equal_owner_sets=False):
    """Apply bounded directed collapses from bad-face interior points to seams.

    Candidate ownership is built from exact float32 coordinates with NumPy's
    sorted unique table; it does not build Python objects for every mesh point.
    Only edges incident to below-target faces and shorter than the source
    displacement bound are considered. Each accepted operation preserves the
    exact parent-to-child face map supplied in ``prepared``.

    This is a source-conditioning pass. It does not certify whole-surface
    embedding or the transformed native Float32 geometry.
    """
    if minimum_altitude_m <= 0 or maximum_source_displacement_m <= 0:
        raise ValueError('conditioning bounds must be positive')
    if maximum_volume_change_m3 < 0 or maximum_operations < 0 or maximum_attempts_per_pass <= 0:
        raise ValueError('conditioning operation bounds are invalid')

    ancestry = source_ancestry_by_point if source_ancestry_by_point is not None else {}
    cumulative = cumulative_volume_change_m3
    if cumulative is None:
        cumulative = {int(sid): 0.0 for sid in prepared}
    cumulative_groups = ({str(key): float(value) for key, value in
                          cumulative_volume_group_change_m3.items()}
                         if cumulative_volume_group_change_m3 is not None else {})
    failures = {}
    operations = []
    attempted = 0

    def candidates_from_current_geometry():
        ordered_sids = sorted(map(int, prepared))
        xyz_parts, offsets, cursor = [], {}, 0
        for sid in ordered_sids:
            vertices = np.asarray(prepared[sid][0])
            if vertices.ndim != 2 or vertices.shape[1] not in (3, 6):
                raise ValueError(f'prepared surface {sid} is malformed')
            xyz = np.asarray(vertices[:, :3], dtype=np.float32)
            offsets[sid] = (cursor, cursor + len(xyz))
            cursor += len(xyz)
            xyz_parts.append(xyz)
        if not cursor:
            return [], {}, {}
        unique_xyz, inverse, counts = np.unique(
            np.concatenate(xyz_parts, axis=0), axis=0, return_inverse=True, return_counts=True)

        owner_rows = {}
        for sid in ordered_sids:
            start, end = offsets[sid]
            local_unique = np.unique(inverse[start:end])
            for point_id in local_unique[counts[local_unique] > 1]:
                owner_rows.setdefault(int(point_id), []).append(sid)
        inverse_by_sid = {sid: inverse[offsets[sid][0]:offsets[sid][1]]
                          for sid in ordered_sids}

        proposals = {}
        for sid in ordered_sids:
            vertices, faces = prepared[sid][0], np.asarray(prepared[sid][1], dtype=np.int64)
            if not len(faces):
                continue
            xyz = np.asarray(vertices[:, :3], dtype=np.float64)
            triangles = xyz[faces]
            doubled_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                                   triangles[:, 2] - triangles[:, 0]), axis=1)
            edge_lengths = np.stack((np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
                                     np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
                                     np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)), axis=1)
            altitude = doubled_area / np.maximum(edge_lengths.max(axis=1), 1e-30)
            global_ids = inverse_by_sid[sid]
            for face_index in np.flatnonzero(altitude < minimum_altitude_m):
                face = faces[face_index]
                for edge_index, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
                    length = float(edge_lengths[face_index, edge_index])
                    if length <= 0 or length > maximum_source_displacement_m:
                        continue
                    aid, bid = int(global_ids[face[a]]), int(global_ids[face[b]])
                    a_owners = set(owner_rows.get(aid, (sid,)))
                    b_owners = set(owner_rows.get(bid, (sid,)))
                    if a_owners < b_owners:
                        keep_id, drop_id, keep_owners, drop_owners = bid, aid, b_owners, a_owners
                    elif b_owners < a_owners:
                        keep_id, drop_id, keep_owners, drop_owners = aid, bid, a_owners, b_owners
                    elif allow_equal_owner_sets and a_owners == b_owners and len(a_owners) > 1:
                        # Both endpoints lie on the same reciprocal seam. Try
                        # both directed choices; every owner is updated in one
                        # transaction, so no side of the fissure is left behind.
                        for keep_id, drop_id in ((aid, bid), (bid, aid)):
                            key = (keep_id, drop_id)
                            value = (float(altitude[face_index]), length,
                                     tuple(sorted(a_owners)), tuple(sorted(b_owners)))
                            if key not in proposals or value < proposals[key]:
                                proposals[key] = value
                        continue
                    else:
                        continue
                    if len(keep_owners) < 2:
                        continue
                    key = (keep_id, drop_id)
                    value = (float(altitude[face_index]), length, tuple(sorted(keep_owners)),
                             tuple(sorted(drop_owners)))
                    if key not in proposals or value < proposals[key]:
                        proposals[key] = value
        ordered = sorted((value[0], value[1], keep_id, drop_id, value[2], value[3])
                         for (keep_id, drop_id), value in proposals.items())
        owners = {point_id: tuple(rows) for point_id, rows in owner_rows.items()}
        return ordered, owners, unique_xyz

    for _ in range(int(maximum_operations)):
        candidate_rows, owner_rows, unique_xyz = candidates_from_current_geometry()
        accepted = None
        pass_attempts = 0
        for altitude, edge_length, keep_id, drop_id, keep_owners, drop_owners in candidate_rows:
            if pass_attempts >= maximum_attempts_per_pass:
                break
            pass_attempts += 1
            attempted += 1
            keep = tuple(map(float, unique_xyz[keep_id]))
            drop = tuple(map(float, unique_xyz[drop_id]))
            complete = {keep: keep_owners, drop: drop_owners}
            try:
                report = collapse_registered_interior_vertex(
                    prepared, keep, drop,
                    maximum_source_displacement_m=maximum_source_displacement_m,
                    minimum_altitude_m=minimum_altitude_m,
                    maximum_volume_change_m3=maximum_volume_change_m3,
                    volume_bounded_surface_ids=volume_bounded_surface_ids,
                    volume_bounded_surface_groups=volume_bounded_surface_groups,
                    cumulative_volume_change_m3=cumulative,
                    cumulative_volume_group_change_m3=cumulative_groups,
                    source_ancestry_by_point=ancestry,
                    complete_point_owners=complete,
                    same_winding_surface_pairs=same_winding_surface_pairs,
                    allow_equal_owner_sets=allow_equal_owner_sets)
            except (ValueError, RuntimeError) as exc:
                reason = str(exc)
                failures[reason] = failures.get(reason, 0) + 1
                continue
            report['trigger_face_altitude_m'] = altitude
            report['edge_length_m'] = edge_length
            for sid, value in report['cumulative_volume_change_m3'].items():
                cumulative[int(sid)] = float(value)
            cumulative_groups.update(report['cumulative_volume_group_change_m3'])
            accepted = report
            operations.append(report)
            break
        if accepted is None:
            break

    final_minimum = {}
    for sid in sorted(map(int, prepared)):
        vertices, faces = prepared[sid][0], np.asarray(prepared[sid][1], dtype=np.int64)
        if not len(faces):
            final_minimum[str(sid)] = None
            continue
        xyz = np.asarray(vertices[:, :3], dtype=np.float64)
        tri = xyz[faces]
        area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        lengths = np.stack((np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                             np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                             np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)), axis=1)
        final_minimum[str(sid)] = float(np.min(area2 / np.maximum(lengths.max(axis=1), 1e-30)))
    return {
        'method': 'bounded_directed_interior_to_shared_seam_edge_collapse',
        'allow_equal_owner_sets': bool(allow_equal_owner_sets),
        'target_altitude_m': minimum_altitude_m,
        'maximum_source_displacement_m': maximum_source_displacement_m,
        'maximum_volume_change_m3': maximum_volume_change_m3,
        'accepted_operation_count': len(operations),
        'attempted_candidate_count': attempted,
        'rejection_reasons': dict(sorted(failures.items())),
        'operations': operations,
        'final_minimum_altitude_m_by_surface': final_minimum,
        'cumulative_signed_volume_change_m3_by_surface': {
            str(sid): float(value) for sid, value in sorted(cumulative.items())},
        'cumulative_signed_volume_change_m3_by_group': {
            key: float(value) for key, value in sorted(cumulative_groups.items())},
        'qualification': 'source conditioning only; full reciprocal topology, self-intersection, native Float32 transform and accepted-cycle checks remain separate',
    }


def open_registered_shared_vertex_star(prepared, point, direction, displacement_m, *,
                                       maximum_source_displacement_m=1e-4,
                                       minimum_altitude_m=1.2e-6,
                                       minimum_result_altitude_m=1.25e-7,
                                       allow_local_minimum_regression=False,
                                       maximum_volume_change_m3=1e-9,
                                       volume_bounded_surface_ids=(),
                                       volume_bounded_surface_groups=(),
                                       cumulative_volume_change_m3=None,
                                       cumulative_volume_group_change_m3=None,
                                       source_ancestry_by_point=None,
                                       minimum_owner_count=2):
    """Move one exact shared vertex copy set together to open a thin star.

    The face graph and lineage do not change. Every surface carrying the exact
    Float32 coordinate receives the same quantized target, so reciprocal
    triangles remain reciprocal. This is a narrowly bounded source-geometry
    conditioning move; orientation, local sliver score, volume and original
    ancestry are checked before any row is mutated. By default, no incident
    star may lose minimum altitude. The opt-in floor mode permits a local
    minimum trade when the combined star objective improves, while keeping
    every affected face above an explicit Float32-conditioning floor.
    """
    old_key=tuple(map(float,np.asarray(point,dtype=np.float32)[:3]))
    vector=np.asarray(direction,dtype=np.float64)[:3]
    norm=float(np.linalg.norm(vector))
    if norm<=0 or displacement_m<=0 or maximum_source_displacement_m<=0:
        raise ValueError('shared-star opening bounds and direction must be positive')
    target=(np.asarray(old_key,dtype=np.float64)+(vector/norm)*float(displacement_m)).astype(np.float32)
    new_key=tuple(map(float,target))
    if new_key==old_key:
        raise ValueError('shared-star opening rounds to zero Float32 displacement')
    owners={}; indices={}
    for raw_sid,row in prepared.items():
        sid=int(raw_sid); vertices=np.asarray(row[0]); faces=np.asarray(row[1],dtype=np.int64)
        xyz=np.asarray(vertices[:,:3],dtype=np.float32)
        matches=np.flatnonzero(np.all(xyz==np.asarray(old_key,dtype=np.float32),axis=1))
        if len(matches)>1: raise ValueError(f'surface {sid} contains duplicate shared-star coordinate')
        if len(matches): owners[sid]=int(matches[0]); indices[sid]=(vertices,faces,row[2],row[3])
    if len(owners)<int(minimum_owner_count):
        raise ValueError('shared-star point has too few exact surface owners')
    ancestry=source_ancestry_by_point
    origins=({tuple(map(float,np.asarray(p,dtype=np.float32))) for p in ancestry.get(old_key,{old_key})}
             if ancestry is not None else {old_key})
    source_displacement=max(float(np.linalg.norm(np.asarray(new_key)-np.asarray(p))) for p in origins)
    if source_displacement>maximum_source_displacement_m:
        raise ValueError('shared-star opening exceeds cumulative source displacement bound')
    # Do not introduce a new exact-coordinate identification with any vertex.
    for sid,row in prepared.items():
        xyz=np.asarray(row[0][:,:3],dtype=np.float32)
        hits=np.flatnonzero(np.all(xyz==target,axis=1))
        allowed={owners[int(sid)]} if int(sid) in owners else set()
        if any(int(i) not in allowed for i in hits):
            raise ValueError(f'shared-star opening collides with another vertex on surface {sid}')

    def face_altitudes(vertices,faces):
        p=np.asarray(vertices)[np.asarray(faces,dtype=np.int64),:3].astype(np.float64)
        area2=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        lengths=np.stack((np.linalg.norm(p[:,1]-p[:,0],axis=1),
                          np.linalg.norm(p[:,2]-p[:,1],axis=1),
                          np.linalg.norm(p[:,0]-p[:,2],axis=1)),axis=1)
        return area2/np.maximum(lengths.max(axis=1),1e-30)
    def sliver_score(values):
        deficit=np.maximum(0.0,np.log(minimum_altitude_m/np.maximum(values,1e-30)))
        return float(np.dot(deficit,deficit))
    proposals={}; deltas={}; score_before=score_after=0.0
    min_before=float('inf'); min_after=float('inf')
    bounded_ids=set(map(int,volume_bounded_surface_ids))
    groups=tuple(tuple(sorted(set(map(int,g)))) for g in volume_bounded_surface_groups)
    if any(not g or any(sid not in prepared for sid in g) for g in groups):
        raise ValueError('volume-bounded star groups must name existing surfaces')
    cumulative=({int(sid):float(v) for sid,v in cumulative_volume_change_m3.items()}
                if cumulative_volume_change_m3 is not None else {})
    cumulative_groups=({str(k):float(v) for k,v in cumulative_volume_group_change_m3.items()}
                       if cumulative_volume_group_change_m3 is not None else {})
    for sid,index in owners.items():
        vertices,faces,parent_map,detail=indices[sid]
        incident=np.flatnonzero(np.any(faces==index,axis=1))
        if not len(incident): raise ValueError(f'surface {sid} shared-star vertex is isolated')
        new_vertices=vertices.copy();new_vertices[index,:3]=target
        old_tri=vertices[faces[incident],:3].astype(np.float64)
        new_tri=new_vertices[faces[incident],:3].astype(np.float64)
        old_normals=np.cross(old_tri[:,1]-old_tri[:,0],old_tri[:,2]-old_tri[:,0])
        new_normals=np.cross(new_tri[:,1]-new_tri[:,0],new_tri[:,2]-new_tri[:,0])
        if np.any(np.linalg.norm(new_normals,axis=1)==0) or np.any(np.einsum('ij,ij->i',old_normals,new_normals)<=0):
            raise ValueError(f'surface {sid} shared-star move changes local orientation or creates zero area')
        before=face_altitudes(vertices,faces[incident]);after=face_altitudes(new_vertices,faces[incident])
        before_min=float(before.min());after_min=float(after.min())
        if allow_local_minimum_regression:
            if minimum_result_altitude_m <= 0 or after_min < minimum_result_altitude_m:
                raise ValueError(f'surface {sid} shared-star move crosses the minimum-result altitude floor')
        elif after_min<before_min-1e-12:
            raise ValueError(f'surface {sid} shared-star move regresses local minimum altitude')
        sb=sliver_score(before);sa=sliver_score(after)
        score_before+=sb;score_after+=sa;min_before=min(min_before,before_min);min_after=min(min_after,after_min)
        old_xyz=old_tri;new_xyz=new_tri
        old_v=float(np.einsum('ij,ij->i',old_xyz[:,0],np.cross(old_xyz[:,1],old_xyz[:,2])).sum()/6)
        new_v=float(np.einsum('ij,ij->i',new_xyz[:,0],np.cross(new_xyz[:,1],new_xyz[:,2])).sum()/6)
        delta=new_v-old_v;deltas[sid]=delta
        if sid in bounded_ids and abs(cumulative.get(sid,0.0)+delta)>maximum_volume_change_m3:
            raise ValueError(f'surface {sid} shared-star move exceeds cumulative signed-volume bound')
        proposals[sid]=(new_vertices,faces.copy(),parent_map,detail,incident,before_min,after_min,delta)
    if score_after>=score_before-1e-15:
        raise ValueError('shared-star move does not reduce combined local sliver objective')
    group_deltas={}
    for group in groups:
        name='+'.join(map(str,group));delta=sum(deltas.get(sid,0.0) for sid in group)
        group_deltas[name]=delta
        if abs(cumulative_groups.get(name,0.0)+delta)>maximum_volume_change_m3:
            raise ValueError(f'surface group {name} exceeds cumulative signed-volume bound')
    # Every exact shared face touching the point must remain an exact reciprocal
    # triangle, with its previous orientation relation.
    pair_checks=[]
    same_winding={(min(int(a),int(b)),max(int(a),int(b))) for a,b in ()}
    for i,a in enumerate(sorted(owners)):
        va,fa=indices[a][0],indices[a][1]
        for b in sorted(owners)[i+1:]:
            vb,fb=indices[b][0],indices[b][1]
            def local_map(v,f,idx):
                rows={}
                for tri in f[np.any(f==idx,axis=1)]:
                    p=[tuple(map(float,np.asarray(v[k,:3],dtype=np.float32))) for k in tri]
                    keytri=tuple(sorted(p));orient=min(tuple(p),tuple(p[1:]+p[:1]),tuple(p[2:]+p[:2]))
                    rows.setdefault(keytri,[]).append(orient)
                return rows
            old_a=local_map(va,fa,owners[a]);old_b=local_map(vb,fb,owners[b])
            common=set(old_a)&set(old_b)
            if not common: continue
            new_a=local_map(proposals[a][0],proposals[a][1],owners[a]);new_b=local_map(proposals[b][0],proposals[b][1],owners[b])
            new_common=set(new_a)&set(new_b)
            expected_common={tuple(sorted(new_key if point==old_key else point for point in tri))
                             for tri in common}
            if expected_common!=new_common:
                raise ValueError(f'shared-star move changes reciprocal seam triangles between {a} and {b}')
            same_count=opposite_count=0
            for face_key in common:
                old_relations={('same' if x==y else 'opposite' if
                    x==min((y[0],y[2],y[1]),(y[2],y[1],y[0]),(y[1],y[0],y[2])) else 'other')
                    for x in old_a[face_key] for y in old_b[face_key]}
                moved_face_key=tuple(sorted(new_key if point==old_key else point for point in face_key))
                new_relations={('same' if x==y else 'opposite' if
                    x==min((y[0],y[2],y[1]),(y[2],y[1],y[0]),(y[1],y[0],y[2])) else 'other')
                    for x in new_a[moved_face_key] for y in new_b[moved_face_key]}
                if old_relations!=new_relations:
                    raise ValueError(f'shared-star move changes reciprocal winding between {a} and {b}')
                if 'other' in old_relations:
                    raise ValueError(f'shared-star move finds incompatible reciprocal winding between {a} and {b}')
                if 'same' in old_relations: same_count+=1
                if 'opposite' in old_relations: opposite_count+=1
            pair_checks.append({'surface_a':a,'surface_b':b,'shared_local_triangle_count':len(common),
                                'same_winding_triangle_count':same_count,
                                'opposite_winding_triangle_count':opposite_count,'preserved':True})
    # Validate every owner's rebuilt normals before publishing any owner. A
    # later rejection must leave geometry, detail and ancestry unchanged.
    ready = {}
    for sid,(new_vertices,faces,parent_map,detail,incident,*_) in proposals.items():
        points=new_vertices[faces,:3].astype(np.float64)
        normals=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
        sums=np.zeros((len(new_vertices),3),dtype=np.float64)
        for axis in range(3): np.add.at(sums,faces[:,axis],normals)
        lengths=np.linalg.norm(sums,axis=1)
        used = np.unique(faces)
        if np.any(lengths[used] == 0) or not np.all(np.isfinite(lengths[used])):
            raise ValueError(f'surface {sid} develops a zero or nonfinite vertex normal')
        # Source crops may retain unused vertices. They carry no surface
        # normal contribution; preserve their stored attributes verbatim.
        new_vertices[used,3:6]=(sums[used]/lengths[used,None]).astype(new_vertices.dtype)
        new_detail = dict(detail)
        records=list(detail.get('registered_shared_star_openings',[]))
        records.append({'old_point':old_key,'new_point':new_key,'displacement_m':float(np.linalg.norm(np.asarray(new_key)-np.asarray(old_key)))})
        new_detail['registered_shared_star_openings']=records
        ready[sid]=(new_vertices,faces,parent_map,new_detail)
    for sid, row in ready.items():
        prepared[sid]=row
        cumulative[sid]=cumulative.get(sid,0.0)+deltas[sid]
    for name,delta in group_deltas.items(): cumulative_groups[name]=cumulative_groups.get(name,0.0)+delta
    if ancestry is not None:
        ancestry.pop(old_key,None);ancestry[new_key]=origins
    return {'method':'bounded_shared_coordinate_vertex_star_opening','old_point':old_key,'new_point':new_key,
        'displacement_m':float(np.linalg.norm(np.asarray(new_key)-np.asarray(old_key))),
        'allow_local_minimum_regression':bool(allow_local_minimum_regression),
        'minimum_result_altitude_m':float(minimum_result_altitude_m),
        'cumulative_source_displacement_m':source_displacement,'source_ancestry_count':len(origins),
        'owner_surface_ids':sorted(owners),'combined_local_sliver_objective_before':score_before,
        'combined_local_sliver_objective_after':score_after,'combined_local_minimum_altitude_before_m':min_before,
        'combined_local_minimum_altitude_after_m':min_after,'per_surface_signed_volume_delta_m3':{str(s):v for s,v in deltas.items()},
        'cumulative_volume_group_change_m3':{k:cumulative_groups[k] for k in group_deltas},'reciprocal_seam_checks':pair_checks,
        'qualification':'bounded source-space star opening; final topology, embedding, native transformed geometry and accepted-cycle checks remain separate'}


def improve_registered_shared_vertex_stars(prepared, *, minimum_altitude_m=1.2e-6,
                                           target_altitude_m=2.4e-6,
                                           minimum_result_altitude_m=1.25e-7,
                                           allow_local_minimum_regression=False,
                                           search_all_face_vertices=False,
                                           maximum_source_displacement_m=1e-4,
                                           maximum_volume_change_m3=1e-9,
                                           volume_bounded_surface_ids=(),
                                           volume_bounded_surface_groups=(),
                                           cumulative_volume_change_m3=None,
                                           cumulative_volume_group_change_m3=None,
                                           source_ancestry_by_point=None,
                                           maximum_operations=64,
                                           maximum_attempts_per_pass=256):
    """Apply small shared-coordinate openings to residual thin triangle stars."""
    if not 0<minimum_altitude_m<target_altitude_m or maximum_operations<0 or maximum_attempts_per_pass<=0:
        raise ValueError('shared-star altitude or operation bounds are invalid')
    ancestry=source_ancestry_by_point if source_ancestry_by_point is not None else {}
    cumulative=(cumulative_volume_change_m3 if cumulative_volume_change_m3 is not None
                else {int(s):0.0 for s in prepared})
    cumulative_groups=(cumulative_volume_group_change_m3 if cumulative_volume_group_change_m3 is not None else {})
    operations=[];failures={};attempted=0
    def _single_triangle_altitude(points):
        points=np.asarray(points,dtype=np.float64)
        area2=float(np.linalg.norm(np.cross(points[1]-points[0],points[2]-points[0])))
        longest=max(float(np.linalg.norm(points[1]-points[0])),
                    float(np.linalg.norm(points[2]-points[1])),
                    float(np.linalg.norm(points[0]-points[2])))
        return area2/max(longest,1e-30)
    for _ in range(int(maximum_operations)):
        xyzparts=[];offsets={};cursor=0
        for sid in sorted(map(int,prepared)):
            v=np.asarray(prepared[sid][0]);xyz=np.asarray(v[:,:3],dtype=np.float32)
            offsets[sid]=(cursor,cursor+len(xyz));cursor+=len(xyz);xyzparts.append(xyz)
        unique,inverse,counts=np.unique(np.concatenate(xyzparts),axis=0,return_inverse=True,return_counts=True)
        owner_rows={};inverse_by_sid={}
        for sid in sorted(map(int,prepared)):
            start,end=offsets[sid];local=inverse[start:end];inverse_by_sid[sid]=local
            for pid in np.unique(local[counts[local]>1]):owner_rows.setdefault(int(pid),[]).append(sid)
        candidates=[]
        for sid in sorted(map(int,prepared)):
            v,f=prepared[sid][0],np.asarray(prepared[sid][1],dtype=np.int64)
            if not len(f):continue
            p=np.asarray(v[:,:3],dtype=np.float64)[f]
            area2=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
            lens=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],axis=1)
            alt=area2/lens.max(axis=1)
            for fi in np.flatnonzero(alt<minimum_altitude_m):
                face=f[fi]; edges=((0,1,2),(1,2,0),(2,0,1))
                longest=max(edges,key=lambda e:float(lens[fi,((0,1),(1,2),(2,0)).index((e[0],e[1]))]))
                step=max(target_altitude_m-float(alt[fi]),target_altitude_m*0.05)
                a,b,apex=longest
                local_vertices=(apex,) if not search_all_face_vertices else (0,1,2)
                for local_vertex in local_vertices:
                    point_id=int(inverse_by_sid[sid][face[local_vertex]])
                    owners=owner_rows.get(point_id,())
                    if len(owners)<2:continue
                    if local_vertex==apex:
                        p0,p1,pc=p[fi,a],p[fi,b],p[fi,apex]
                        edge=p1-p0;edge2=float(edge@edge)
                        if edge2<=0:continue
                        projection=p0+edge*float((pc-p0)@edge)/edge2
                        gradient=pc-projection
                    else:
                        # Include long-edge endpoints as candidate owners. Their
                        # altitude gradients can improve a thin face while a
                        # shared apex would invert another incident face.
                        trial=p[fi].copy();eps=max(1e-9,target_altitude_m*0.05)
                        gradient=np.zeros(3,dtype=np.float64)
                        for axis in range(3):
                            trial[local_vertex,axis]+=eps
                            plus=_single_triangle_altitude(trial)
                            trial[local_vertex,axis]-=2*eps
                            minus=_single_triangle_altitude(trial)
                            trial[local_vertex,axis]+=eps
                            gradient[axis]=(plus-minus)/(2*eps)
                    dn=float(np.linalg.norm(gradient))
                    if dn<=0:continue
                    direction=gradient/dn
                    factors=(1.0,2.0) if search_all_face_vertices else (1.0,)
                    for factor in factors:
                        candidates.append((float(alt[fi]),step*factor,point_id,direction,tuple(sorted(owners))))
        candidates.sort(key=lambda x:(x[0],x[1],x[2]))
        accepted=None;pass_attempts=0;seen=set()
        for altitude,step,pid,direction,owners in candidates:
            if pass_attempts>=maximum_attempts_per_pass:break
            point=tuple(map(float,unique[pid]));key=(point,tuple(owners),tuple(np.round(direction,6)),round(step,9))
            if key in seen:continue
            seen.add(key);pass_attempts+=1;attempted+=1
            try:
                op=open_registered_shared_vertex_star(prepared,point,direction,step,
                    maximum_source_displacement_m=maximum_source_displacement_m,
                    minimum_altitude_m=minimum_altitude_m,
                    minimum_result_altitude_m=minimum_result_altitude_m,
                    allow_local_minimum_regression=allow_local_minimum_regression,
                    maximum_volume_change_m3=maximum_volume_change_m3,
                    volume_bounded_surface_ids=volume_bounded_surface_ids,
                    volume_bounded_surface_groups=volume_bounded_surface_groups,
                    cumulative_volume_change_m3=cumulative,
                    cumulative_volume_group_change_m3=cumulative_groups,
                    source_ancestry_by_point=ancestry)
            except (ValueError,RuntimeError) as exc:
                reason=str(exc);failures[reason]=failures.get(reason,0)+1;continue
            op['trigger_face_altitude_m']=altitude;op['proposed_displacement_m']=step
            for sid,value in op['per_surface_signed_volume_delta_m3'].items():cumulative[int(sid)]=cumulative.get(int(sid),0.0)+float(value)
            cumulative_groups.update(op['cumulative_volume_group_change_m3'])
            operations.append(op);accepted=op;break
        if accepted is None:break
    final={}
    for sid in sorted(map(int,prepared)):
        v,f=prepared[sid][0],np.asarray(prepared[sid][1],dtype=np.int64)
        p=np.asarray(v[:,:3],dtype=np.float64)[f]
        area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        lens=np.stack([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)],axis=1)
        h=area/np.maximum(lens.max(axis=1),1e-30)
        final[str(sid)]={'minimum_altitude_m':float(h.min()) if len(h) else None,'below_target_count':int(np.count_nonzero(h<minimum_altitude_m))}
    max_disp=0.0;max_w=None
    for point,origins in ancestry.items():
        for origin in origins:
            d=float(np.linalg.norm(np.asarray(point)-np.asarray(origin)))
            if d>max_disp:max_disp=d;max_w={'current_point':point,'source_point':origin,'distance_m':d}
    return {'method':'bounded_shared_coordinate_vertex_star_opening','minimum_altitude_m':minimum_altitude_m,
        'minimum_result_altitude_m':minimum_result_altitude_m,
        'allow_local_minimum_regression':bool(allow_local_minimum_regression),
        'search_all_face_vertices':bool(search_all_face_vertices),
        'target_altitude_m':target_altitude_m,'maximum_source_displacement_m':maximum_source_displacement_m,
        'accepted_operation_count':len(operations),'attempted_candidate_count':attempted,'rejection_reasons':dict(sorted(failures.items())),
        'operations':operations,'final_by_surface':final,'maximum_source_ancestry_displacement_m':max_disp,
        'maximum_source_ancestry_witness':max_w,'cumulative_signed_volume_change_m3_by_surface':{str(k):float(v) for k,v in sorted(cumulative.items())},
        'cumulative_signed_volume_change_m3_by_group':{str(k):float(v) for k,v in sorted(cumulative_groups.items())},
        'qualification':'source-space shared-star conditioning only; exact final surface audit and native accepted-state checks remain separate'}


def improve_csg_cut_fragment_faces(prepared, source_owner_by_face, original_source_meshes):
    """Condition only CSG fragments with reciprocal owner/provenance support.

    The source owner array is the existing clipped-mesh lineage field.  A face
    is eligible only if its coordinate triangle is not an exact original source
    triangle and its owner is an explicit retained/cutter value (0 or 1).
    Every reciprocal copy must have both faces eligible with matching owner
    provenance.  This pre-conform operation intentionally has no Kuhn-cell
    restriction; the existing post-conform API retains that restriction.
    """
    stable_ids = set(map(int, prepared))
    if set(map(int, source_owner_by_face)) != stable_ids or set(map(int, original_source_meshes)) != stable_ids:
        raise ValueError('CSG fragment lineage must cover every prepared reciprocal surface')
    eligible_faces = {}
    exact_source_face_count = {}
    original_parent_ids = {sid: set(map(int, prepared[sid][2])) for sid in stable_ids}
    for sid in sorted(stable_ids):
        vertices, faces, _, _ = prepared[sid]
        vertices = np.asarray(vertices)
        faces = np.asarray(faces, dtype=np.int64)
        owners = np.asarray(source_owner_by_face[sid])
        if owners.shape != (len(faces),) or not np.issubdtype(owners.dtype, np.integer):
            raise ValueError(f'CSG source-owner lineage is malformed for surface {sid}')
        if not set(map(int, np.unique(owners))).issubset({0, 1}):
            raise ValueError(f'CSG source-owner lineage has an unsupported value for surface {sid}')
        source_vertices, source_faces = original_source_meshes[sid]
        source_vertices = np.asarray(source_vertices)
        source_faces = np.asarray(source_faces, dtype=np.int64)
        if (source_vertices.ndim != 2 or source_vertices.shape[1] not in (3, 6)
                or source_faces.ndim != 2 or source_faces.shape[1] != 3
                or (source_faces.size and (source_faces.min() < 0 or source_faces.max() >= len(source_vertices)))):
            raise ValueError(f'original source mesh is malformed for surface {sid}')

        def key(v, face):
            return tuple(sorted(tuple(map(float, v[int(i), :3])) for i in face))

        source_keys = {key(source_vertices, tri) for tri in source_faces}
        eligible = {}
        original_count = 0
        for face_id, face in enumerate(faces):
            if key(vertices, face) in source_keys:
                original_count += 1
                continue
            eligible[face_id] = int(owners[face_id])
        eligible_faces[sid] = eligible
        exact_source_face_count[sid] = original_count

    report = _improve_sliver_faces(
        prepared, minimum_altitude_m=2.5e-7,
        maximum_nonplanarity_m=1e-7, spacing_m=1/64,
        eligible_faces=eligible_faces, require_same_field_cell=False,
        minimum_result_altitude_m=1.25e-7,
        require_strict_improvement=True,
        method='shared_diagonal_flip_of_inferred_csg_cut_fragments_before_field_conforming',
    )
    multi_parent_counts = {}
    lineage_complete = True
    for sid, (_, faces, parents, _) in prepared.items():
        per_face = [set() for _ in faces]
        for parent, children in parents.items():
            for child in children:
                per_face[int(child)].add(int(parent))
        multi_parent_counts[str(sid)] = sum(len(owners) > 1 for owners in per_face)
        if (not all(per_face) or set().union(*per_face) != original_parent_ids[sid]):
            lineage_complete = False
    if not lineage_complete:
        raise ValueError('CSG conditioning lost source face ancestry')
    report.update({
        'eligible_cut_fragment_face_count_by_surface': {str(sid): len(rows) for sid, rows in eligible_faces.items()},
        'exact_original_source_face_count_by_surface': {str(sid): exact_source_face_count[sid] for sid in sorted(stable_ids)},
        'all_exact_original_source_faces_excluded': True,
        'reciprocal_provenance_gate': 'every owner copy must be eligible and the two adjacent fragments must have equal source_owner values',
        'vertices_moved': False,
        'multi_parent_lineage_preserved': lineage_complete,
        'multi_parent_face_count_by_surface': multi_parent_counts,
        'qualification': 'offline CSG fragment conditioning only; source topology, exact self/pleura, interfaces, and native cycle remain separate checks',
    })
    return report


def _improve_sliver_faces(prepared, *, minimum_altitude_m, maximum_nonplanarity_m,
                          spacing_m, eligible_faces, require_same_field_cell,
                          minimum_result_altitude_m, require_strict_improvement,
                          method):
    if (minimum_altitude_m not in (2.5e-7, 1.2e-6) or
            maximum_nonplanarity_m not in (1e-7, 1e-5) or
            spacing_m != 1/64):
        raise ValueError('unsupported respiratory mesh quality bounds')
    if minimum_result_altitude_m not in (2.5e-7, 1.25e-7, 1.2e-6):
        raise ValueError('unsupported resulting altitude bound')
    points, lookup, local_to_global, global_to_local = [], {}, {}, {}
    faces, lineage = {}, {}
    for sid in sorted(prepared):
        vertices, source_faces, parents, _ = prepared[sid]
        local = []
        for p in vertices[:, :3]:
            key = tuple(map(float, p))
            if key not in lookup:
                lookup[key] = len(points)
                points.append(key)
            local.append(lookup[key])
        local_to_global[sid] = np.asarray(local, dtype=np.int64)
        if len(set(local)) != len(local):
            raise ValueError('quality improvement requires welded source coordinates')
        global_to_local[sid] = {v: i for i, v in enumerate(local)}
        faces[sid] = local_to_global[sid][source_faces].copy()
        lineage[sid] = [set() for _ in source_faces]
        for parent, children in parents.items():
            for child in children:
                lineage[sid][child].add(parent)
    xyz = np.asarray(points, dtype=float)
    owners, edges = {}, {}

    def key(tri):
        return tuple(sorted(map(int, tri)))

    def tri_edges(tri):
        return [tuple(sorted((int(tri[a]), int(tri[b]))))
                for a, b in ((0, 1), (1, 2), (2, 0))]

    def add(sid, index):
        tri = faces[sid][index]
        owners.setdefault(key(tri), set()).add((sid, index))
        for edge in tri_edges(tri):
            edges.setdefault((sid, edge), set()).add(index)

    def remove(sid, index):
        tri = faces[sid][index]
        owner_key = key(tri)
        owners[owner_key].remove((sid, index))
        if not owners[owner_key]:
            del owners[owner_key]
        for edge in tri_edges(tri):
            edge_key = (sid, edge)
            edges[edge_key].remove(index)
            if not edges[edge_key]:
                del edges[edge_key]

    for sid, f in faces.items():
        for index in range(len(f)):
            add(sid, index)

    def geometry(f):
        p = xyz[f]
        normal = np.cross(p[..., 1, :] - p[..., 0, :], p[..., 2, :] - p[..., 0, :])
        lengths = np.stack([np.linalg.norm(p[..., 1, :] - p[..., 2, :], axis=-1),
                            np.linalg.norm(p[..., 2, :] - p[..., 0, :], axis=-1),
                            np.linalg.norm(p[..., 0, :] - p[..., 1, :], axis=-1)], axis=-1)
        return normal, lengths, np.linalg.norm(normal, axis=-1) / lengths.max(axis=-1)

    planes = np.asarray(((1, 0, 0), (0, 1, 0), (0, 0, 1),
                         (1, -1, 0), (1, 0, -1), (0, 1, -1)))
    initial = []
    for sid, f in faces.items():
        _, _, alt = geometry(f)
        initial.extend((float(alt[i]), sid, int(i)) for i in np.flatnonzero(alt < minimum_altitude_m))
    changes, skips = [], {}

    def skip(reason):
        skips[reason] = skips.get(reason, 0) + 1

    for _, sid, index in sorted(initial):
        tri = faces[sid][index]
        normal, lengths, altitude = geometry(tri)
        if altitude >= minimum_altitude_m:
            continue
        opposite = int(np.argmax(lengths))
        c, a, b = (int(tri[opposite]), int(tri[(opposite+1) % 3]), int(tri[(opposite+2) % 3]))
        edge = tuple(sorted((a, b)))
        adjacent = edges[(sid, edge)]
        if len(adjacent) != 2:
            skip('non-two-face edge'); continue
        neighbor = next(i for i in adjacent if i != index)
        other = faces[sid][neighbor]
        d = int(next(i for i in other if i not in edge))
        copies1, copies2 = owners[key(tri)].copy(), owners[key(other)].copy()
        owner_ids = {s for s, _ in copies1}
        if (owner_ids != {s for s, _ in copies2} or
                len(copies1) != len(owner_ids) or len(copies2) != len(owner_ids)):
            skip('anatomical interface ownership boundary'); continue
        if len({a, b, c, d}) != 4 or any((s, tuple(sorted((c, d)))) in edges for s in owner_ids):
            skip('existing diagonal'); continue
        if eligible_faces is not None:
            eligible = True
            for owner in sorted(owner_ids):
                i = next(i for s, i in copies1 if s == owner)
                j = next(i for s, i in copies2 if s == owner)
                allowed = eligible_faces.get(owner, {})
                if i not in allowed or j not in allowed or allowed[i] != allowed[j]:
                    eligible = False
                    break
            if not eligible:
                skip('outside inferred CSG cut patch or reciprocal provenance differs'); continue
        quad = xyz[[a, b, c, d]]
        if require_same_field_cell:
            coordinates = quad @ planes.T / spacing_m
            if not np.all(np.ceil(coordinates.max(axis=0)-1) <= np.floor(coordinates.min(axis=0))):
                skip('different affine field cells'); continue
        neighbor_normal = np.cross(xyz[b]-xyz[a], xyz[d]-xyz[a])
        neighbor_length = np.linalg.norm(neighbor_normal)
        if not neighbor_length:
            skip('degenerate neighbor'); continue
        nonplanarity = abs(float(np.dot(xyz[c]-xyz[a], neighbor_normal))) / neighbor_length
        if nonplanarity > maximum_nonplanarity_m:
            skip('nonplanar source patch'); continue
        proposed = np.asarray(((c, a, d), (c, d, b)), dtype=np.int64)
        new_normal, _, new_alt = geometry(proposed)
        old_normal, _, _ = geometry(np.asarray((tri, other)))
        mean = old_normal.sum(axis=0)
        if np.dot(new_normal[0], mean) < 0:
            proposed = proposed[:, (0, 2, 1)]
            new_normal = -new_normal
        if (not np.all(new_normal @ mean > 0)
                or new_alt.min() < minimum_result_altitude_m
                or (require_strict_improvement and new_alt.min() <= altitude)):
            skip('orientation or quality not improved'); continue
        updates = []
        for owner in sorted(owner_ids):
            i = next(i for s, i in copies1 if s == owner)
            j = next(i for s, i in copies2 if s == owner)
            old_normals, _, _ = geometry(faces[owner][[i, j]])
            oriented = proposed.copy()
            if np.dot(new_normal[0], old_normals.sum(axis=0)) < 0:
                oriented = oriented[:, (0, 2, 1)]
            updates.append((owner, i, j, oriented))
        for owner, i, j, oriented in updates:
            remove(owner, i); remove(owner, j)
            faces[owner][[i, j]] = oriented
            combined_lineage = lineage[owner][i] | lineage[owner][j]
            lineage[owner][i] = combined_lineage.copy()
            lineage[owner][j] = combined_lineage.copy()
            add(owner, i); add(owner, j)
        changes.append({'source_surface': sid, 'source_face': index, 'neighbor_face': neighbor,
                        'shared_surface_owners': sorted(owner_ids),
                        'old_altitude_m': float(altitude), 'new_minimum_altitude_m': float(new_alt.min()),
                        'maximum_patch_nonplanarity_m': nonplanarity,
                        'modified_faces': {str(s): [i, j] for s, i, j, _ in updates}})
    remaining = {}
    ready = {}
    for sid in sorted(prepared):
        vertices, old_faces, parents, detail = prepared[sid]
        f = np.asarray([[global_to_local[sid][int(i)] for i in tri] for tri in faces[sid]], dtype=np.int64)
        if len(f) != len(old_faces):
            raise AssertionError('diagonal flips changed face count')
        remapped = {parent: [] for parent in parents}
        for i, original_parents in enumerate(lineage[sid]):
            for parent in original_parents:
                remapped[parent].append(i)
        out = vertices.copy()
        p = out[f, :3].astype(float)
        normals = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        summed = np.zeros((len(out), 3), dtype=float)
        for k in range(3):
            np.add.at(summed, f[:, k], normals)
        lengths = np.linalg.norm(summed, axis=1)
        used = np.unique(f)
        if np.any(lengths[used] == 0) or not np.all(np.isfinite(lengths[used])):
            raise ValueError('quality improvement left a zero or nonfinite vertex normal')
        out[used, 3:6] = summed[used] / lengths[used, None]
        if not np.array_equal(out[:, :3], vertices[:, :3]):
            raise AssertionError('diagonal flips moved source vertices')
        _, _, alt = geometry(faces[sid])
        remaining[str(sid)] = {'minimum_altitude_m': float(alt.min()),
                               'below_bound_face_count': int(np.count_nonzero(alt < minimum_altitude_m))}
        new_detail = dict(detail)
        new_detail['diagonal_flip_count'] = sum(sid in x['shared_surface_owners'] for x in changes)
        ready[sid] = out, f, remapped, new_detail
    prepared.update(ready)
    return {'method': method,
            'minimum_altitude_target_m': minimum_altitude_m,
            'minimum_result_altitude_m': minimum_result_altitude_m,
            'maximum_nonplanarity_m': maximum_nonplanarity_m,
            'same_affine_field_cell_required': require_same_field_cell,
            'strict_altitude_improvement_required': require_strict_improvement,
            'source_vertex_displacement_m': 0, 'flip_count': len(changes),
            'changes': changes, 'skipped_reasons': skips, 'remaining': remaining,
            'qualification': 'local mesh conditioning only; remaining thin faces and native cycle validation must be checked'}
