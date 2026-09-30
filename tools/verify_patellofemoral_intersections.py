"""Exact static intersection audit for the pinned Open Knee patellofemoral pair.

The source and both compiled sides are inspected separately. A positive result
is a geometric intersection, not a contact force, penetration depth, or a
clinical verdict about the source pose.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from numilab_human.abdominal_organ_separation import exact_integer_meshes
from numilab_human.cardiac_cavity_intersections import triangle_intersection_points
from numilab_human.open_knee import EXPECTED_HASHES, parse_source


REGIONS = (
    ('PTC', 'PTC_@_FMC_ContactFaces'),
    ('FMC', 'FMC_@_PTC_ContactFaces'),
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_meshes(source) -> dict:
    meshes = {}
    for region_name, surface_name in REGIONS:
        region = source.regions[region_name]
        local = {identifier: index for index, identifier in enumerate(region.node_ids)}
        meshes[region_name] = (
            np.asarray(region.nodes_mm, dtype=np.float64) * .001,
            np.asarray([[local[i] for i in face]
                        for face in source.surfaces[surface_name].faces],
                       dtype=np.int64),
        )
    return meshes


def _compiled_meshes(decoded: dict) -> dict:
    meshes = {}
    for region_name, surface_name in REGIONS:
        region = decoded['regions'][region_name]
        surface = decoded['surfaces'][surface_name]
        first_node, count = region['first_node'], region['node_count']
        first_face, face_count = surface['first_face'], surface['face_count']
        meshes[region_name] = (
            decoded['positions'][first_node:first_node + count],
            decoded['faces'][first_face:first_face + face_count].astype(np.int64)
            - first_node,
        )
    return meshes


def _intersections(meshes: dict) -> dict:
    denominator, exact = exact_integer_meshes(meshes)
    patella, femur = (exact[name] for name in ('PTC', 'FMC'))
    femur_min, femur_max = femur['triangle_min'], femur['triangle_max']
    candidate_count = point_count = crossing_count = 0
    crossing_pairs = []
    intersection_points = []
    maximum_segment = 0.0
    for patellar_index, (low, high) in enumerate(zip(
            patella['triangle_min'], patella['triangle_max'], strict=True)):
        candidate = np.flatnonzero(np.all(
            (femur_min <= high) & (femur_max >= low), axis=1))
        candidate_count += len(candidate)
        for femoral_index in candidate:
            points = set(triangle_intersection_points(
                patella['records'][patellar_index][0],
                femur['records'][int(femoral_index)][0]))
            if not points:
                continue
            if len(points) == 1:
                point_count += 1
                continue
            crossing_count += 1
            crossing_pairs.append([patellar_index, int(femoral_index)])
            metre_points = [tuple(float(value) / denominator for value in point)
                            for point in points]
            intersection_points.extend(metre_points)
            maximum_segment = max(maximum_segment, max(
                math.dist(a, b) for a, b in itertools.combinations(metre_points, 2)))
    bounds = None
    if intersection_points:
        coordinates = np.asarray(intersection_points)
        bounds = {
            'minimum_m': coordinates.min(axis=0).tolist(),
            'maximum_m': coordinates.max(axis=0).tolist(),
        }
    return {
        'aabb_candidate_triangle_pairs': candidate_count,
        'exact_segment_or_polygon_crossing_pairs': crossing_count,
        'exact_point_contact_pairs': point_count,
        'crossing_triangle_pairs': crossing_pairs,
        'maximum_intersection_segment_m': maximum_segment,
        'intersection_bounds_m': bounds,
        'static_noninterpenetration_qualified': crossing_count == point_count == 0,
    }


def audit(root: Path) -> dict:
    # Reuse only the format decoder from the orientation audit. The triangle
    # intersection calculation is a separate exact-predicate implementation.
    from tools.verify_patellofemoral_surface import payload

    root = Path(root).resolve()
    source_dir = root / 'Sources/open-knee-oks003'
    source = parse_source(source_dir)
    orientation_path = root / 'Docs/media/patellofemoral-surface-20260930/receipt.json'
    orientation = json.loads(orientation_path.read_text())
    if (orientation['status'] !=
            'passed_source_and_bilateral_compiled_static_face_winding' or
            orientation['source_file_sha256'] != {
                name: _sha(source_dir / name) for name in EXPECTED_HASHES}):
        raise RuntimeError('patellofemoral contact: orientation source identity differs')
    result = {
        'schema': 'numi.human.patellofemoral-contact-intersections.v1',
        'source_file_sha256': {name: _sha(source_dir / name)
                               for name in EXPECTED_HASHES},
        'source': _intersections(_source_meshes(source)),
        'compiled': {},
        'orientation_receipt_sha256': _sha(orientation_path),
        'contact_pressure_qualified': False,
        'loaded_knee_qualified': False,
        'clinical_anatomy_qualified': False,
    }
    for side, stem in (
            ('left', 'open-knee-oks003-left'),
            ('right', 'open-knee-oks003-right-mirrored')):
        directory = root / 'Build/patellofemoral-surface-20260930' / side
        manifest_path = directory / f'{stem}.manifest.json'
        payload_path = directory / f'{stem}.nhknee'
        manifest = json.loads(manifest_path.read_text())
        decoded = payload(payload_path, manifest, source)
        if decoded['side'] != side:
            raise RuntimeError(f'patellofemoral contact: {side} payload side mismatch')
        if (orientation['compiled'][side]['payload_sha256'] != _sha(payload_path) or
                orientation['compiled'][side]['manifest_sha256'] !=
                _sha(manifest_path)):
            raise RuntimeError(f'patellofemoral contact: {side} orientation payload differs')
        result['compiled'][side] = {
            'manifest_sha256': _sha(manifest_path),
            'payload_sha256': _sha(payload_path),
            **_intersections(_compiled_meshes(decoded)),
        }
    pair_sets = [result['source']['crossing_triangle_pairs']] + [
        result['compiled'][side]['crossing_triangle_pairs']
        for side in ('left', 'right')]
    if pair_sets[0] != pair_sets[1] or pair_sets[0] != pair_sets[2]:
        raise RuntimeError('patellofemoral contact: source/compiled crossing identity differs')
    scopes = [result['source'], result['compiled']['left'],
              result['compiled']['right']]
    if len({row['exact_point_contact_pairs'] for row in scopes}) != 1:
        raise RuntimeError('patellofemoral contact: source/compiled point-contact count differs')
    result['source_and_bilateral_crossing_pair_identity'] = True
    result['status'] = ('passed_static_noninterpenetration'
                        if all(row['static_noninterpenetration_qualified']
                               for row in scopes)
                        else 'failed_static_noninterpenetration')
    result['boundary'] = (
        'Exact binary source/compiled triangle intersections are a static mesh '
        'result. They are not penetration depth, contact pressure, load transfer, '
        'or clinical evidence. The pinned source has the same crossing pairs '
        'as both compiled sides; no geometry was moved or clipped.')
    result['auditor_sha256'] = _sha(Path(__file__))
    result['predicate_and_decoder_sha256'] = {
        'abdominal_organ_separation.py': _sha(
            root / 'src/numilab_human/abdominal_organ_separation.py'),
        'cardiac_cavity_intersections.py': _sha(
            root / 'src/numilab_human/cardiac_cavity_intersections.py'),
        'verify_patellofemoral_surface.py': _sha(
            root / 'tools/verify_patellofemoral_surface.py'),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': result['status'],
                      'source_crossings': result['source'][
                          'exact_segment_or_polygon_crossing_pairs'],
                      'left_crossings': result['compiled']['left'][
                          'exact_segment_or_polygon_crossing_pairs'],
                      'right_crossings': result['compiled']['right'][
                          'exact_segment_or_polygon_crossing_pairs']}))
    return 0 if result['status'] == 'passed_static_noninterpenetration' else 2


if __name__ == '__main__':
    raise SystemExit(main())
