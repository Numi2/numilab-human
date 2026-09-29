"""Bounded visual-only untangle of two pinned biceps femoris source tips.

The BodyParts3D FJ1444/FJ1444M members each have three real source-surface
self-intersections around two exact coincident-point groups. This derived
candidate keeps every source vertex and triangle identity, and keeps the raw
OBJ immutable. It does not certify clinical muscle shape or force transfer.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from .compiled_quotient_embeddedness import classify_quotient


SOURCE_SHA256 = {
    'FJ1444': 'c071097501796cee5ff2a011b3d84c7bea4cb153241fce0f8b0a73d070bd990f',
    'FJ1444M': '2794a3899545d9b66f61fb9de3459132b577939c63b527fae101d497e28b154e',
}
EXPECTED_PAIRS = [[1088, 1116], [1088, 1117], [1117, 1119]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError('biceps femoris visual tip untangle: ' + message)


def _plane(points: np.ndarray, faces: np.ndarray, face_id: int):
    triangle = points[faces[face_id]]
    normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
    length = float(np.linalg.norm(normal))
    _require(length > 0 and math.isfinite(length), 'source plane degeneracy')
    return triangle[0], normal/length


def _compiled(points_mm: np.ndarray) -> list[list[float]]:
    return [[struct.unpack('<f', struct.pack('<f', float(coordinate*.001)))[0]
             for coordinate in point] for point in points_mm]


def untangle(vertices_mm: list[list[float]], triangles: list[list[int]],
             member_id: str, source_sha256: str) -> tuple[list[list[float]], dict]:
    """Return a separately recorded visual candidate; refuse source drift."""
    _require(member_id in SOURCE_SHA256
             and source_sha256 == SOURCE_SHA256[member_id], 'pinned member identity')
    points = np.asarray(vertices_mm, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    _require(points.shape == ((1211 if member_id == 'FJ1444' else 1215), 3)
             and faces.shape == (1728, 3) and np.isfinite(points).all()
             and (faces >= 0).all() and (faces < len(points)).all(),
             'source mesh shape')
    raw_source = classify_quotient((points*.001).tolist(), faces.tolist())
    raw_compiled = classify_quotient(_compiled(points), faces.tolist())
    for result in (raw_source, raw_compiled):
        _require(result['topology']['closed_oriented_manifold_candidate']
                 and result['exact_intersection_pairs'] == 3
                 and result['first_intersecting_face_pairs'] == EXPECTED_PAIRS,
                 'source crossing signature changed')

    main_origin, main_normal = _plane(points, faces, 1088)
    cap_origin, cap_normal = _plane(points, faces, 1119)
    witness = 1057 if member_id == 'FJ1444' else 1061
    main_sign = float(np.sign(main_normal @ (points[witness]-main_origin)))
    cap_sign = float(np.sign(cap_normal @ (points[witness]-cap_origin)))
    _require(main_sign == -1.0 and cap_sign == 1.0,
             'source tip orientation changed')
    centre_group = np.flatnonzero(np.all(points == points[553], axis=1))
    rim_group = np.flatnonzero(np.all(points == points[552], axis=1))
    expected_centre = ([553, 1062, 1063, 1064] if member_id == 'FJ1444'
                       else [553, 1066, 1067, 1068])
    expected_rim = [552, 1061] if member_id == 'FJ1444' else [552, 1065]
    _require(centre_group.tolist() == expected_centre
             and rim_group.tolist() == expected_rim,
             'coincident source-point groups changed')

    # Move only the two problematic exact point groups across the local source
    # face planes by a small positive margin. The mirrored source uses the same
    # signed construction. Exact duplicate vertices receive identical outputs.
    candidate = points.copy()
    candidate[centre_group] = points[553] + main_sign*.015*main_normal
    candidate[rim_group] = points[552] + cap_sign*.065*cap_normal
    old_normals = np.cross(points[faces[:, 1]]-points[faces[:, 0]],
                           points[faces[:, 2]]-points[faces[:, 0]])
    new_normals = np.cross(candidate[faces[:, 1]]-candidate[faces[:, 0]],
                           candidate[faces[:, 2]]-candidate[faces[:, 0]])
    old_length = np.linalg.norm(old_normals, axis=1)
    new_length = np.linalg.norm(new_normals, axis=1)
    _require(bool((old_length > 0).all() and (new_length > 0).all()),
             'triangle collapse')
    area_ratio = new_length/old_length
    normal_cosine = np.sum(old_normals*new_normals, axis=1)/(old_length*new_length)
    _require(float(area_ratio.min()) > .85 and float(area_ratio.max()) < 1.03
             and float(normal_cosine.min()) > .9,
             'local source support or orientation changed too much')
    exact_candidate = classify_quotient((candidate*.001).tolist(), faces.tolist())
    compiled_candidate = classify_quotient(_compiled(candidate), faces.tolist())
    for result in (exact_candidate, compiled_candidate):
        _require(result['topology']['closed_oriented_manifold_candidate']
                 and result['topology']['face_component_count'] == 1
                 and result['self_intersection'] == 'exact_checked'
                 and result['exact_intersection_pairs'] == 0,
                 'derived candidate not embedded')
    changed = sorted(set(centre_group.tolist()+rim_group.tolist()))
    return candidate.tolist(), {
        'method': 'pinned_source_tip_two_point_group_signed_plane_untangle_v1',
        'source_member_sha256': source_sha256,
        'source_visual_geometry_modified': True,
        'source_obj_retained_unchanged': True,
        'moved_source_vertex_ids': changed,
        'retained_source_vertex_count': len(points),
        'retained_source_triangle_count': len(faces),
        'vertices_added_or_removed': False,
        'triangles_added_removed_or_reindexed': False,
        'centre_group_shift_mm': .015,
        'rim_group_shift_mm': .065,
        'maximum_vertex_displacement_mm': float(np.linalg.norm(candidate-points, axis=1).max()),
        'minimum_face_area_ratio': float(area_ratio.min()),
        'maximum_face_area_ratio': float(area_ratio.max()),
        'minimum_face_normal_cosine': float(normal_cosine.min()),
        'source_exact_self_intersection_pairs_before': 3,
        'compiled_fp32_self_intersection_pairs_before': 3,
        'source_exact_self_intersection_pairs_after': 0,
        'compiled_fp32_self_intersection_pairs_after': 0,
        'clinical_anatomy': False,
        'physical_volume_owner': False,
        'mechanics_changed': False,
        'boundary': ('A local derived visual surface closes three source-tip mesh '
                     'crossings per side. This is not independent anatomical '
                     'registration, material calibration, or force qualification.'),
    }
