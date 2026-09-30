"""Build exact barycentric witnesses for the source patellofemoral crossing loop.

This is a geometric contact-initiation candidate, not a pressure or force law.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from numilab_human.abdominal_organ_separation import exact_integer_meshes
from numilab_human.cardiac_cavity_intersections import triangle_intersection_points
from numilab_human.open_knee import EXPECTED_HASHES, parse_source
from tools.verify_patellofemoral_surface import payload


ROOT = Path(__file__).resolve().parents[1]
REGIONS = (
    ('PTC', 'PTC_@_FMC_ContactFaces'),
    ('FMC', 'FMC_@_PTC_ContactFaces'),
)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError('patellofemoral intersection loop: ' + message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def barycentric(point: tuple, triangle: tuple) -> tuple[Fraction, ...]:
    a, b, c = triangle
    ab = tuple(b[i] - a[i] for i in range(3))
    ac = tuple(c[i] - a[i] for i in range(3))
    normal = (ab[1]*ac[2] - ab[2]*ac[1],
              ab[2]*ac[0] - ab[0]*ac[2],
              ab[0]*ac[1] - ab[1]*ac[0])
    axis = max(range(3), key=lambda index: abs(normal[index]))
    require(normal[axis] != 0, 'degenerate intersection triangle')
    u, v = (index for index in range(3) if index != axis)
    ap_u, ap_v = point[u] - a[u], point[v] - a[v]
    denominator = ab[u]*ac[v] - ab[v]*ac[u]
    weight_b = Fraction(ap_u*ac[v] - ap_v*ac[u], denominator)
    weight_c = Fraction(ab[u]*ap_v - ab[v]*ap_u, denominator)
    weights = (1 - weight_b - weight_c, weight_b, weight_c)
    require(all(0 <= weight <= 1 for weight in weights) and
            all(sum(weights[i] * triangle[i][axis]
                    for i in range(3)) == point[axis]
                for axis in range(3)),
            'intersection point lacks exact triangle barycentric witness')
    return weights


def meshes_from_source(source) -> dict:
    meshes = {}
    for region_name, surface_name in REGIONS:
        region = source.regions[region_name]
        local = {identifier: index for index, identifier in enumerate(region.node_ids)}
        meshes[region_name] = (
            np.asarray(region.nodes_mm, dtype=np.float64) * .001,
            np.asarray([[local[identifier] for identifier in face]
                        for face in source.surfaces[surface_name].faces],
                       dtype=np.int64),
        )
    return meshes


def meshes_from_payload(decoded: dict) -> dict:
    meshes = {}
    for region_name, surface_name in REGIONS:
        region, surface = (decoded[table][name] for table, name in (
            ('regions', region_name), ('surfaces', surface_name)))
        start, count = region['first_node'], region['node_count']
        first, face_count = surface['first_face'], surface['face_count']
        meshes[region_name] = (
            decoded['positions'][start:start + count],
            decoded['faces'][first:first + face_count].astype(np.int64) - start,
        )
    return meshes


def build_loop(meshes: dict, *, node_offsets: dict[str, int]) -> dict:
    denominator, exact = exact_integer_meshes(meshes)
    patella, femur = exact['PTC'], exact['FMC']
    segments = []
    points = set()
    for patellar_index, (low, high) in enumerate(zip(
            patella['triangle_min'], patella['triangle_max'], strict=True)):
        candidates = np.flatnonzero(np.all(
            (femur['triangle_min'] <= high) &
            (femur['triangle_max'] >= low), axis=1))
        for femoral_index in candidates:
            first = patella['records'][patellar_index][0]
            second = femur['records'][int(femoral_index)][0]
            pair_points = sorted(set(triangle_intersection_points(first, second)))
            if not pair_points:
                continue
            require(len(pair_points) == 2, 'non-segment intersection in source pair')
            points.update(pair_points)
            segments.append((patellar_index, int(femoral_index), pair_points,
                             first, second))
    point_order = sorted(points)
    point_id = {point: index for index, point in enumerate(point_order)}
    adjacency: dict[int, set[int]] = defaultdict(set)
    rows = []
    normal_cosines = []
    for patellar_index, femoral_index, endpoints, first, second in segments:
        a, b = (point_id[point] for point in endpoints)
        require(a != b and b not in adjacency[a], 'duplicate or zero crossing edge')
        adjacency[a].add(b)
        adjacency[b].add(a)
        first_face = patella['faces'][patellar_index]
        second_face = femur['faces'][femoral_index]
        first_triangle = meshes['PTC'][0][first_face]
        second_triangle = meshes['FMC'][0][second_face]
        first_normal = np.cross(first_triangle[1] - first_triangle[0],
                                first_triangle[2] - first_triangle[0])
        second_normal = np.cross(second_triangle[1] - second_triangle[0],
                                 second_triangle[2] - second_triangle[0])
        normal_cosine = float(np.dot(first_normal, second_normal) /
                              np.linalg.norm(first_normal) /
                              np.linalg.norm(second_normal))
        require(math.isfinite(normal_cosine) and normal_cosine < -.99,
                'intersection faces do not oppose across the joint')
        normal_cosines.append(normal_cosine)
        witnesses = []
        for point in endpoints:
            weights_a = barycentric(point, first)
            weights_b = barycentric(point, second)
            position = [float(value) / denominator for value in point]
            for name, face, weights in (
                    ('PTC', first_face, weights_a),
                    ('FMC', second_face, weights_b)):
                vertices = meshes[name][0][face]
                reconstructed = sum(float(weights[i]) * vertices[i]
                                    for i in range(3))
                require(float(np.linalg.norm(reconstructed - position)) <= 1e-15,
                        'Float64 barycentric reconstruction drift')
            witnesses.append({
                'point_id': point_id[point],
                'position_m': position,
                'patellar_barycentric': [float(value) for value in weights_a],
                'femoral_barycentric': [float(value) for value in weights_b],
            })
        rows.append({
            'patellar_surface_face_index': patellar_index,
            'femoral_surface_face_index': femoral_index,
            'patellar_node_indices': [int(index) + node_offsets['PTC']
                                      for index in first_face],
            'femoral_node_indices': [int(index) + node_offsets['FMC']
                                    for index in second_face],
            'point_ids': [a, b],
            'opposing_face_normal_cosine': normal_cosine,
            'witnesses': witnesses,
            'segment_length_m': math.dist(*[w['position_m'] for w in witnesses]),
        })
    require(len(rows) == len(point_order) == 18 and
            len(adjacency) == 18 and
            all(len(neighbors) == 2 for neighbors in adjacency.values()),
            'crossings do not form the pinned degree-two loop')
    # Pick the lexicographically first point and first neighbor to obtain a
    # deterministic traversal independent of triangle-pair iteration order.
    start = 0
    cycle = [start, min(adjacency[start])]
    while len(cycle) < len(point_order):
        previous, current = cycle[-2:]
        candidate = adjacency[current] - {previous}
        require(len(candidate) == 1, 'intersection loop branches')
        following = next(iter(candidate))
        require(following not in cycle, 'intersection loop closes early')
        cycle.append(following)
    require(start in adjacency[cycle[-1]], 'intersection loop remains open')
    pair_indices = [[row['patellar_surface_face_index'],
                     row['femoral_surface_face_index']] for row in rows]
    require(pair_indices == sorted(pair_indices), 'noncanonical face-pair order')
    return {
        'segment_count': len(rows),
        'unique_intersection_point_count': len(point_order),
        'connected_component_count': 1,
        'closed_degree_two_loop': True,
        'total_intersection_curve_length_m': sum(
            row['segment_length_m'] for row in rows),
        'opposing_face_normal_cosine_min': min(normal_cosines),
        'opposing_face_normal_cosine_max': max(normal_cosines),
        'cycle_point_ids': cycle,
        'triangle_pair_indices': pair_indices,
        'segments': rows,
    }


def run() -> dict:
    source_dir = ROOT / 'Sources/open-knee-oks003'
    source = parse_source(source_dir)
    intersection_path = ROOT / 'Docs/media/patellofemoral-intersections-20260930/receipt.json'
    intersection = json.loads(intersection_path.read_text())
    result = {
        'schema': 'numi.human.patellofemoral-intersection-loop.v1',
        'status': 'compiled_source_bound_closed_intersection_loop',
        'source_file_sha256': {name: sha(source_dir / name)
                               for name in EXPECTED_HASHES},
        'intersection_receipt_sha256': sha(intersection_path),
        'source': build_loop(meshes_from_source(source),
                             node_offsets={'PTC': 0, 'FMC': 0}),
        'compiled': {},
        'pressure_qualified': False,
        'loaded_contact_qualified': False,
        'clinical_anatomy_qualified': False,
    }
    for side, stem in (
            ('left', 'open-knee-oks003-left'),
            ('right', 'open-knee-oks003-right-mirrored')):
        directory = ROOT / 'Build/patellofemoral-surface-20260930' / side
        manifest_path = directory / f'{stem}.manifest.json'
        payload_path = directory / f'{stem}.nhknee'
        decoded = payload(payload_path, json.loads(manifest_path.read_text()), source)
        require(decoded['side'] == side and
                sha(payload_path) == intersection['compiled'][side]['payload_sha256'],
                f'{side} compiled source identity')
        row = build_loop(meshes_from_payload(decoded), node_offsets={
            name: decoded['regions'][name]['first_node'] for name in ('PTC', 'FMC')})
        require(row['triangle_pair_indices'] ==
                intersection['compiled'][side]['crossing_triangle_pairs'] ==
                result['source']['triangle_pair_indices'],
                f'{side} exact crossing pairs changed')
        result['compiled'][side] = {
            'payload_sha256': sha(payload_path),
            'manifest_sha256': sha(manifest_path),
            **row,
        }
    require(abs(result['compiled']['left']['total_intersection_curve_length_m'] -
                result['compiled']['right']['total_intersection_curve_length_m'])
            <= 1e-10, 'bilateral loop length differs')
    result['boundary'] = (
        'Exact source and compiled triangle crossings form one closed 18-edge '
        'intersection curve. Every segment endpoint has barycentric witnesses '
        'on both authored surfaces, suitable for a native initial-contact '
        'candidate. This does not set a separation direction, penetration '
        'depth, contact pressure, material law, or an accepted loaded state.')
    result['producer_sha256'] = sha(Path(__file__))
    result['predicate_sha256'] = {
        name: sha(ROOT / 'src/numilab_human' / name)
        for name in ('abdominal_organ_separation.py',
                     'cardiac_cavity_intersections.py')}
    return result


if __name__ == '__main__':
    report = run()
    output = ROOT / 'Docs/media/patellofemoral-loop-20260930/receipt.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    print(json.dumps({
        'status': report['status'],
        'source_length_mm': report['source']['total_intersection_curve_length_m'] * 1000,
        'left_length_mm': report['compiled']['left']['total_intersection_curve_length_m'] * 1000,
        'right_length_mm': report['compiled']['right']['total_intersection_curve_length_m'] * 1000,
    }))
