"""Evaluate a source-bound rigid patellar pose candidate, without adopting it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from numilab_human.abdominal_organ_separation import exact_integer_meshes
from numilab_human.cardiac_cavity_geometry import analyze_topology
from numilab_human.cardiac_cavity_intersections import triangle_intersection_points
from numilab_human.open_knee import parse_source
from tools.verify_patellofemoral_surface import payload


ROOT = Path(__file__).resolve().parents[1]
SHIFT_M = 20.0e-6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_boundary(decoded: dict, region: str) -> tuple[np.ndarray, np.ndarray]:
    row = decoded['regions'][region]
    surface = decoded['surfaces'][f'{region}_All_Faces']
    start, count = row['first_node'], row['node_count']
    first, size = surface['first_face'], surface['face_count']
    return (decoded['positions'][start:start + count],
            decoded['faces'][first:first + size].astype(np.int64) - start)


def count_intersections(exact: dict) -> dict:
    first, second = exact['PTC'], exact['FMC']
    candidate_pairs = segment_pairs = point_pairs = 0
    for index, (low, high) in enumerate(zip(first['triangle_min'],
                                             first['triangle_max'], strict=True)):
        candidates = np.flatnonzero(np.all(
            (second['triangle_min'] <= high) &
            (second['triangle_max'] >= low), axis=1))
        candidate_pairs += len(candidates)
        for other in candidates:
            points = set(triangle_intersection_points(
                first['records'][index][0],
                second['records'][int(other)][0]))
            segment_pairs += len(points) >= 2
            point_pairs += len(points) == 1
    return {'aabb_candidate_pairs': candidate_pairs,
            'segment_or_polygon_crossing_pairs': segment_pairs,
            'point_contact_pairs': point_pairs}


def boundary_topology(vertices: np.ndarray, faces: np.ndarray,
                      global_node_offset: int,
                      source_node_ids: list[int]) -> dict:
    used = np.unique(faces)
    compact = np.searchsorted(used, faces)
    report = analyze_topology(vertices[used].tolist(), compact.tolist())
    fan_counts = []
    for compact_index in report['vertex_manifold_defect_ids']:
        local_node = int(used[compact_index])
        incident = faces[np.any(faces == local_node, axis=1)]
        adjacency: dict[int, set[int]] = {}
        for face in incident:
            first, second = (int(value) for value in face if value != local_node)
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)
        unseen = set(adjacency)
        components = 0
        while unseen:
            components += 1
            stack = [next(iter(unseen))]
            while stack:
                node = stack.pop()
                if node not in unseen:
                    continue
                unseen.remove(node)
                stack.extend(adjacency[node] & unseen)
        fan_counts.append(components)
    return {
        'used_surface_vertices': len(used),
        'closed_oriented_manifold_candidate':
            report['closed_oriented_manifold_candidate'],
        'face_component_count': report['face_component_count'],
        'boundary_edge_count': report['boundary_edge_count'],
        'nonmanifold_edge_count': len(report['nonmanifold_edges']),
        'orientation_defect_edge_count': len(report['orientation_defect_edges']),
        'vertex_manifold_defect_global_node_indices': [
            global_node_offset + int(used[index])
            for index in report['vertex_manifold_defect_ids']],
        'vertex_manifold_defect_source_node_ids': [
            int(source_node_ids[int(used[index])])
            for index in report['vertex_manifold_defect_ids']],
        'vertex_manifold_defect_link_fan_counts': fan_counts,
        'euler_characteristic': report['euler_characteristic'],
    }


def side_result(side: str, source, loop: dict) -> dict:
    stem = ('open-knee-oks003-left' if side == 'left'
            else 'open-knee-oks003-right-mirrored')
    folder = ROOT / 'Build/patellofemoral-surface-20260930' / side
    payload_path = folder / f'{stem}.nhknee'
    manifest_path = folder / f'{stem}.manifest.json'
    decoded = payload(payload_path, json.loads(manifest_path.read_text()), source)
    assert sha(payload_path) == loop['compiled'][side]['payload_sha256']
    meshes = {region: full_boundary(decoded, region)
              for region in ('PTC', 'FMC')}
    topology = {region: boundary_topology(*meshes[region],
                decoded['regions'][region]['first_node'],
                source.regions[region].node_ids)
                for region in meshes}
    normals = []
    for segment in loop['compiled'][side]['segments']:
        triangle = decoded['positions'][segment['patellar_node_indices']]
        normal = np.cross(triangle[1] - triangle[0],
                          triangle[2] - triangle[0])
        normals.append(normal / np.linalg.norm(normal))
    direction = np.mean(normals, axis=0)
    direction /= np.linalg.norm(direction)
    initial_denominator, initial = exact_integer_meshes(meshes)
    baseline = count_intersections(initial)
    assert baseline['segment_or_polygon_crossing_pairs'] == 18
    shifted = (meshes['PTC'][0] - SHIFT_M * direction).astype('<f4').astype(np.float64)
    changed = {**meshes, 'PTC': (shifted, meshes['PTC'][1])}
    candidate_denominator, candidate = exact_integer_meshes(changed)
    clearance = count_intersections(candidate)
    assert clearance['segment_or_polygon_crossing_pairs'] == 0
    assert clearance['point_contact_pairs'] == 0
    assert topology['PTC']['closed_oriented_manifold_candidate']
    assert not topology['FMC']['closed_oriented_manifold_candidate']
    assert topology['FMC']['vertex_manifold_defect_source_node_ids'] == [233523]
    assert topology['FMC']['vertex_manifold_defect_link_fan_counts'] == [2]
    actual = np.linalg.norm(shifted - meshes['PTC'][0], axis=1)
    return {
        'payload_sha256': sha(payload_path),
        'manifest_sha256': sha(manifest_path),
        'baseline': baseline,
        'candidate': clearance,
        'outward_mean_normal': direction.tolist(),
        'prescribed_pose_translation_m': (-SHIFT_M * direction).tolist(),
        'compiled_float32_displacement_m': {
            'minimum': float(actual.min()), 'maximum': float(actual.max())},
        'source_boundary_topology': topology,
        'candidate_volume_disjointness_qualified': False,
        'source_full_boundary_faces': {
            region: len(meshes[region][1]) for region in meshes},
        'exact_integer_denominators': {
            'baseline': str(initial_denominator),
            'candidate': str(candidate_denominator)},
    }


def run() -> dict:
    source = parse_source(ROOT / 'Sources/open-knee-oks003')
    loop_path = ROOT / 'Docs/media/patellofemoral-loop-20260930/receipt.json'
    loop = json.loads(loop_path.read_text())
    result = {
        'schema': 'numi.human.patellofemoral-pose-clearance-candidate.v1',
        'status': 'unadopted_bilateral_geometric_pose_candidate',
        'intersection_loop_receipt_sha256': sha(loop_path),
        'source_files_sha256': loop['source_file_sha256'],
        'prescribed_translation_magnitude_m': SHIFT_M,
        'sides': {side: side_result(side, source, loop)
                  for side in ('left', 'right')},
        'boundary': 'Only PTC current positions were translated for this exact face-crossing experiment. The FMC full boundary has a source vertex-link manifold defect, so zero pairwise face intersections does not qualify disjoint volumes. Adopting a patellar pose requires coherent PTB, QAT and PTL attachments, source rest-state and force/energy validation; no runtime state or source payload was changed.',
        'loaded_contact_qualified': False,
        'clinical_anatomy_qualified': False,
    }
    out = ROOT / 'Docs/media/patellofemoral-pose-clearance-20260930'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'receipt.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True))
