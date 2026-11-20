"""Relate deterministic Metal contact-limiter witnesses to the emitted tetrahedra."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CONTACTS = (1208222, 1208247, 1208564)
NODES_PER_FEATURE = 4
SCENE_NODE_OFFSET = 1665  # 1-based TetGen node to 0-based native index


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def read_nodes(path, wanted):
    points = {}
    with path.open() as stream:
        next(stream)
        for line in stream:
            node, *xyz = line.split()
            key = int(node)
            if key in wanted:
                points[key] = np.array(xyz[:3], dtype=np.float32).astype(np.float64)
    if wanted != points.keys():
        raise ValueError('Missing emitted point')
    return points


def main():
    mesh = ROOT / 'absolute-volume-repaired-tetgen'
    node_path = mesh / 'mesh.node'
    tet_path = mesh / 'mesh.ele'
    logfile = ROOT / 'floor-native-live-parent-007/stderr.log'
    records = {}
    with logfile.open() as source:
        for line in source:
            if not line.startswith('fem_deformable_limit_trace='):
                continue
            trace = json.loads(line.split('=', 1)[1])
            if trace['iteration'] != 0:
                continue
            for row in trace['contacts']:
                if row['contact_slot'] in CONTACTS:
                    records[row['contact_slot']] = row
    if records.keys() != set(CONTACTS):
        raise ValueError('Not all expected contact witnesses were retained')
    cells = {}
    memberships = {}
    needed_points = set()
    for slot, record in records.items():
        feature = record['nodes'][:NODES_PER_FEATURE]
        expected = [node - SCENE_NODE_OFFSET for node in feature]
        if record['geometry_coordinate_frame'] != 'reference_local':
            raise ValueError(f'Unsupported witness coordinate frame for slot {slot}')
        memberships[slot] = expected
        needed_points.update(expected)
    incident_points = set(needed_points)
    with tet_path.open() as stream:
        next(stream)
        for line in stream:
            i, a, b, c, d, material = map(int, line.split())
            cell = (a, b, c, d)
            if incident_points.isdisjoint(cell):
                continue
            cells[i] = (cell, material)
            needed_points.update(cell)
    points = read_nodes(node_path, needed_points)
    cell_metrics = {}
    for cell_id, (cell, material) in cells.items():
        xyz = np.array([points[i] for i in cell])
        jacobian = (xyz[1:] - xyz[0]).T
        edges = xyz[:, None, :] - xyz[None, :, :]
        max_edge = np.max(np.linalg.norm(edges, axis=-1))
        signed_volume = float(np.linalg.det(jacobian) / 6)
        cell_metrics[cell_id] = {
            'material': material, 'tetgen_nodes': cell,
            'signed_volume_m3': signed_volume,
            'normalized_volume': signed_volume / max_edge**3,
            'basis_condition_number': float(np.linalg.cond(jacobian)),
        }
    result = []
    for slot, expected in memberships.items():
        record = records[slot]
        rest = np.array([points[i] for i in expected])
        origin = np.array(record['reference_origin_m'])
        start = np.array(record['feature_start'])
        end = np.array(record['feature_end'])
        matching_error = float(np.max(np.linalg.norm(rest - origin - start, axis=1)))
        if matching_error > 1e-9:
            raise ValueError(f'Native-to-TetGen node identity failed at contact slot {slot}')
        incident = {i: cell_metrics[i] for i, (cell, _) in cells.items()
                    if any(n in cell for n in expected)}
        point_to_corners = []
        for corner in expected[1:]:
            common_cells = [i for i, (cell, _) in cells.items()
                            if expected[0] in cell and corner in cell]
            point_to_corners.append({
                'tetgen_node': corner,
                'distance_m': float(np.linalg.norm(points[expected[0]] - points[corner])),
                'shared_tetgen_cells': common_cells,
            })
        def by_condition(item):
            return item[1]['basis_condition_number']
        worst_cell, worst_metrics = max(incident.items(), key=by_condition)
        result.append({
            'contact_slot': slot, 'source': record['source'], 'contact_identity': record['identity'],
            'runtime_nodes': record['nodes'][:4], 'tetgen_nodes': expected,
            'maximum_rest_coordinate_matching_error_m': matching_error,
            'weights': record['weights'][:4],
            'point_triangle_corner_adjacency': point_to_corners,
            'all_four_in_one_tetrahedron': any(set(expected) <= set(cell) for cell, _ in cells.values()),
            'incident_tet_count': len(incident), 'minimum_signed_volume_m3':
                min(c['signed_volume_m3'] for c in incident.values()),
            'minimum_normalized_volume':
                min(c['normalized_volume'] for c in incident.values()),
            'worst_basis_cell': {'tetgen_cell_id': worst_cell, **worst_metrics},
            'ccd_certified': record['ccd_certified'], 'ccd_hit': record['ccd_hit'],
            'collision_thickness_m': record['collision_thickness_m'],
            'initial_separation_m': record['contact_point_and_separation'][3],
            'incoming_fraction': record['requested_alpha'],
            'post_contact_fraction': record['lane_bound_after'],
            'trial_motion_m': (end-start).tolist(),
            'raw_newton_direction_m': ((end-start)/record['requested_alpha']).tolist(),
        })
    print(json.dumps({
        'mesh_node_sha256': sha256(node_path), 'mesh_ele_sha256': sha256(tet_path),
        'witness_log_sha256': sha256(logfile), 'witnesses': result,
    }, indent=2))


if __name__ == '__main__':
    main()
