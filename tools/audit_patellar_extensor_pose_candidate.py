"""Check a source-bound patellar clearance pose across the extensor stack.

This is an anatomical geometry candidate, not a native mechanics step. PTB,
PTC and QAT translate with the patella; PTL tapers from its patellar tie to
its unchanged tibial tie. The executable verifies source/payload/input hashes,
tie gaps, exact tetrahedral orientation and the previously audited PTC pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from numilab_human.open_knee import parse_source
from numilab_human.surface_topology_audit import exact_embedding
from tools.verify_patellofemoral_surface import payload
from tools.export_patellofemoral_matter_input import HEADER


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'Sources/open-knee-oks003'
POSE = ROOT / 'Docs/media/patellofemoral-pose-clearance-20260930/receipt.json'
NATIVE = ROOT / 'Docs/media/patellofemoral-full-native-20260930/receipt.json'
SOLID = ROOT / 'Docs/media/patellofemoral-tetrahedral-separation-20260930/receipt.json'
INPUT = ROOT / 'Build/full-patellofemoral-matter-20260930'
COMPILED = ROOT / 'Build/patellofemoral-surface-20260930'
OUTPUT = ROOT / 'Docs/media/patellar-extensor-pose-candidate-20260930/receipt.json'
LATTICE = 1 << 40
CHANGED = ('PTB', 'PTC', 'QAT', 'PTL')
TIE_PAIRS = (
    ('PTC_@_PTB_TiesNodes', 'PTB_@_PTC_TiesNodes'),
    ('QAT_@_PTB_TiesNodes', 'PTB_@_QAT_TiesNodes'),
    ('PTL_@_PTB_TiesNodes', 'PTB_@_PTL_TiesNodes'),
    ('PTL_@_TBB_TiesNodes', 'TBB_@_PTL_TiesNodes'),
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError('patellar extensor pose: ' + message)


def region_nodes(decoded: dict, name: str) -> np.ndarray:
    row = decoded['regions'][name]
    first, count = row['first_node'], row['node_count']
    return decoded['positions'][first:first + count]


def set_indices(source, name: str) -> np.ndarray:
    owner = name.split('_@_', 1)[0]
    index = {identifier: local for local, identifier in
             enumerate(source.regions[owner].node_ids)}
    return np.asarray([index[identifier] for identifier in source.node_sets[name]],
                      dtype=np.int64)


def signed_six(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    a, b, c, d = (points[cells[:, index]] for index in range(4))
    return np.einsum('ij,ij->i', np.cross(b - a, c - a), d - a)


def exact_orientation(points: np.ndarray, cells: np.ndarray) -> dict:
    scaled = np.asarray(points, dtype=np.float64) * LATTICE
    require(np.array_equal(scaled, np.rint(scaled)),
            'a candidate node is not exactly representable on the Float32 lattice')
    integer = scaled.astype(np.int64)
    nonpositive = 0
    minimum = None
    for cell in cells:
        a, b, c, d = (tuple(int(x) for x in integer[int(i)]) for i in cell)
        u = tuple(b[k] - a[k] for k in range(3))
        v = tuple(c[k] - a[k] for k in range(3))
        w = tuple(d[k] - a[k] for k in range(3))
        cross = (u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2],
                 u[0]*v[1] - u[1]*v[0])
        six = sum(cross[k]*w[k] for k in range(3))
        nonpositive += six <= 0
        minimum = six if minimum is None else min(minimum, six)
    require(nonpositive == 0, 'candidate inverted an authored tetrahedron')
    return {'positive_tetrahedra': len(cells), 'zero_or_negative_tetrahedra': 0,
            'minimum_exact_six_volume_lattice': minimum}


def bone_surface_orientation(source, before: np.ndarray,
                             after: np.ndarray) -> dict:
    region = source.regions['PTB']
    index = {identifier: local for local, identifier in enumerate(region.node_ids)}
    faces = np.asarray([[index[identifier] for identifier in face]
                        for face in region.elements], dtype=np.int64)
    def normals(points: np.ndarray) -> np.ndarray:
        a, b, c = (points[faces[:, i]] for i in range(3))
        return np.cross(b - a, c - a)
    old = normals(before)
    new = normals(after)
    old_length = np.linalg.norm(old, axis=1)
    new_length = np.linalg.norm(new, axis=1)
    require(np.all(old_length > 0) and np.all(new_length > 0) and
            np.all(np.einsum('ij,ij->i', old, new) > 0),
            'PTB bone surface degenerated or reversed')
    scaled = after * LATTICE
    require(np.array_equal(scaled, np.rint(scaled)),
            'PTB candidate left Float32 exact lattice')
    integer = scaled.astype(np.int64)
    exact_degenerate = 0
    for face in faces:
        a, b, c = (tuple(int(x) for x in integer[int(i)]) for i in face)
        u = tuple(b[k] - a[k] for k in range(3))
        v = tuple(c[k] - a[k] for k in range(3))
        cross = (u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2],
                 u[0]*v[1] - u[1]*v[0])
        exact_degenerate += cross == (0, 0, 0)
    require(exact_degenerate == 0, 'PTB has an exactly degenerate candidate face')
    ratio = new_length / old_length
    return {'triangles': len(faces), 'exact_degenerate_triangles': 0,
            'reversed_triangles': 0, 'minimum_area_ratio': float(ratio.min()),
            'maximum_area_ratio': float(ratio.max())}


def surface_embedding(source, name: str, points: np.ndarray) -> dict:
    index = {identifier: local for local, identifier in
             enumerate(source.regions[name].node_ids)}
    faces = np.asarray([[index[identifier] for identifier in face]
                        for face in source.surfaces[name + '_All_Faces'].faces],
                       dtype=np.int64)
    used = np.unique(faces)
    compact = np.searchsorted(used, faces)
    result = exact_embedding(points[used].tolist(), compact.tolist())
    require(result['count'] == 0 and
            result['topology']['closed_oriented_manifold_candidate'],
            f'{name} surface is not a closed exact embedding')
    return {'faces': len(faces), 'exact_self_intersection_pairs': 0,
            'closed_oriented_manifold': True,
            'aabb_candidate_pairs': result['aabb_candidate_pairs']}


def tie_gap(points: dict[str, np.ndarray], source,
            first: str, second: str) -> dict:
    a_name, b_name = first.split('_@_', 1)[0], second.split('_@_', 1)[0]
    a = points[a_name][set_indices(source, first)]
    b = points[b_name][set_indices(source, second)]
    ab = cKDTree(b).query(a)[0]
    ba = cKDTree(a).query(b)[0]
    return {'first_to_second': ab, 'second_to_first': ba}


def native_ptc_current(side: str, native: dict) -> np.ndarray:
    data = (INPUT / f'open-knee-{side}-ptc-fmc.nhcar').read_bytes()
    require(sha(data) == native['source_inputs']['20um'][side]['input_sha256'],
            'pinned native cartilage input changed')
    offset = HEADER.size + 12 * 26121
    return np.frombuffer(data, dtype='<f4', count=3 * 26121,
                         offset=offset).reshape(-1, 3)


def run(output: Path) -> dict:
    source = parse_source(SOURCE)
    pose = json.loads(POSE.read_text())
    native = json.loads(NATIVE.read_text())
    solid = json.loads(SOLID.read_text())
    require(pose['status'] == 'unadopted_bilateral_geometric_pose_candidate'
            and solid['all_retained_native_states_volume_disjoint'] is True
            and solid['full_native_receipt_sha256'] == sha(NATIVE.read_bytes()),
            'source clearance/native solid evidence is not bound')
    sides = {}
    for side in ('left', 'right'):
        stem = ('open-knee-oks003-left' if side == 'left'
                else 'open-knee-oks003-right-mirrored')
        path = COMPILED / side / f'{stem}.nhknee'
        manifest_path = path.with_suffix('.manifest.json')
        require(sha(path.read_bytes()) == pose['sides'][side]['payload_sha256'],
                'compiled knee payload changed')
        decoded = payload(path, json.loads(manifest_path.read_text()), source)
        delta = np.asarray(pose['sides'][side]['prescribed_pose_translation_m'],
                           dtype=np.float64)
        require(abs(float(np.linalg.norm(delta)) - 20e-6) < 1e-12,
                'pose displacement is not the audited 20 micrometres')
        before = {name: region_nodes(decoded, name) for name in
                  ('PTB', 'PTC', 'QAT', 'PTL', 'TBB')}
        after = {name: values.copy() for name, values in before.items()}
        for name in ('PTB', 'PTC', 'QAT'):
            after[name] = (before[name] + delta).astype('<f4').astype(np.float64)
        ptb_tie = set_indices(source, 'PTL_@_PTB_TiesNodes')
        tbb_tie = set_indices(source, 'PTL_@_TBB_TiesNodes')
        patella_center = before['PTL'][ptb_tie].mean(axis=0)
        tibia_center = before['PTL'][tbb_tie].mean(axis=0)
        axis = patella_center - tibia_center
        require(float(axis @ axis) > 0.002**2,
                'PTL patella-to-tibia tie axis is degenerate')
        weight = np.clip((before['PTL'] - tibia_center) @ axis / (axis @ axis),
                         0.0, 1.0)
        require(not np.intersect1d(ptb_tie, tbb_tie).size,
                'PTL node has conflicting patella and tibia ties')
        weight[ptb_tie] = 1.0
        weight[tbb_tie] = 0.0
        after['PTL'] = (before['PTL'] + weight[:, None] * delta).astype('<f4').astype(np.float64)
        require(np.array_equal(after['PTC'].astype('<f4'), native_ptc_current(side, native)),
                'coherent PTC coordinates differ from the native separated pose')
        require(np.array_equal(after['PTL'][tbb_tie], before['PTL'][tbb_tie]),
                'PTL tibial tie moved')

        regions = {}
        for name in CHANGED:
            row = decoded['regions'][name]
            shift = np.linalg.norm(after[name] - before[name], axis=1)
            require(float(shift.max()) < 20.1e-6,
                    f'{side} {name} exceeded clearance displacement')
            metrics = {
                'nodes': len(before[name]),
                'candidate_positions_sha256': sha(after[name].astype('<f4').tobytes()),
                'maximum_node_displacement_m': float(shift.max()),
            }
            if name == 'PTB':
                metrics.update(bone_surface_orientation(source, before[name],
                                                        after[name]))
            else:
                cells = decoded['tetrahedra'][row['first_tet']:
                                              row['first_tet'] + row['tet_count']]
                cells = cells.astype(np.int64) - row['first_node']
                previous = signed_six(before[name], cells)
                current = signed_six(after[name], cells)
                require(np.all(previous > 0) and np.all(current > 0),
                        f'{side} {name} volume sign changed')
                ratio = current / previous
                metrics.update({
                    'tetrahedra': len(cells),
                    'minimum_deformation_jacobian': float(ratio.min()),
                    'maximum_deformation_jacobian': float(ratio.max()),
                    **exact_orientation(after[name], cells),
                })
            metrics['source_surface'] = surface_embedding(
                source, name, before[name])
            metrics['candidate_surface'] = surface_embedding(
                source, name, after[name])
            regions[name] = metrics
        ties = {}
        for first, second in TIE_PAIRS:
            prior = tie_gap(before, source, first, second)
            current = tie_gap(after, source, first, second)
            ties[first + '<->' + second] = {
                'first_nodes': len(prior['first_to_second']),
                'second_nodes': len(prior['second_to_first']),
                'maximum_nearest_gap_change_m': max(
                    float(np.max(np.abs(current[key] - prior[key])))
                    for key in ('first_to_second', 'second_to_first')),
            }
        sides[side] = {
            'compiled_payload_sha256': sha(path.read_bytes()),
            'compiled_manifest_sha256': sha(manifest_path.read_bytes()),
            'native_input_sha256': native['source_inputs']['20um'][side]['input_sha256'],
            'pose_translation_m': delta.tolist(),
            'ptl_tie_axis_length_m': float(np.linalg.norm(axis)),
            'ptl_taper_minimum': float(weight.min()),
            'ptl_taper_maximum': float(weight.max()),
            'ptl_tibial_tie_unchanged': True,
            'ptc_native_clearance_positions_bitwise': True,
            'regions': regions, 'ties': ties,
        }
    result = {
        'schema': 'numi.human.patellar-extensor-pose-candidate.v1',
        'status': 'bounded_coherent_extensor_geometry_candidate',
        'producer_sha256': sha(Path(__file__).read_bytes()),
        'source_file_sha256': solid['source_file_sha256'],
        'pose_receipt_sha256': sha(POSE.read_bytes()),
        'native_receipt_sha256': sha(NATIVE.read_bytes()),
        'solid_separation_receipt_sha256': sha(SOLID.read_bytes()),
        'sides': sides,
        'patella_rigid_joint_pose_updated': False,
        'quadriceps_proximal_muscle_attachment_qualified': False,
        'native_coupled_knee_step_qualified': False,
        'clinical_anatomy_qualified': False,
        'boundary': ('The derived PTB/PTC/QAT translation and PTL axial taper '
                     'are an unadopted geometry field. Source meshes, joint '
                     'equalities, rigid patella pose and native force owners '
                     'remain unchanged; no physiological loading is tested.'),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    row = run(parser.parse_args().output)
    print(json.dumps({'status': row['status'], 'sides': {
        side: {'ptl_min_j': values['regions']['PTL']['minimum_deformation_jacobian'],
               'maximum_tie_gap_change_m': max(tie['maximum_nearest_gap_change_m']
                                                for tie in values['ties'].values())}
        for side, values in row['sides'].items()}}, sort_keys=True))
