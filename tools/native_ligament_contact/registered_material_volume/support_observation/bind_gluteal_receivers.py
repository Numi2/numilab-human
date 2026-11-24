"""Bind posterior sites to the existing gluteal P1 model in its registered reference pose."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from inspect_shoulder_source import nearest
from numilab_human.capsular_material_registration import (
    emitted_body_vertices, material_bank, read_material_banks,
)
from pose_glutes import parts


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_transforms(poses):
    matrices = {}
    for pose in poses:
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(pose['quaternion_xyzw']).as_matrix()
        matrix[:3, 3] = pose['position_m']
        matrices[pose['body_index']] = matrix
    return {i: np.linalg.solve(matrices[128], m) for i, m in matrices.items()}


def confirm_sources(old, current, manifest):
    sources = [Path(a[a.index('--soft-tissue-payload') + 1]) for a in (old, current)]
    require(digest(sources[0]) == manifest['source'], 'Wrong source for the material atlas')
    for sid in (29, 30):
        a, b = [parts(path, sid) for path in sources]
        require(all(np.array_equal(x, y) for x, y in zip(a[:4], b[:4])),
                f'Muscle {sid} geometry or donor mismatch')
        require(len(a[4]) == len(b[4]) and all(np.array_equal(x, y) for p, q in
                zip(a[4], b[4]) for x, y in zip(p, q)), f'Muscle {sid} frame mismatch')
    current_bones = Path(current[3])
    banks = read_material_banks(current_bones)
    for sid, expected in manifest['matched_cortical_banks'].items():
        _, rotation, vertices, faces = material_bank(banks, int(sid), expected['owner'])
        points = emitted_body_vertices(vertices, rotation).astype('<f4')
        for key, array in [('points_sha256', points), ('faces_sha256', faces.astype('<u4'))]:
            require(hashlib.sha256(array.tobytes()).hexdigest() == expected[key],
                    f'Cortical bank {sid} differs in {key}')
    return {'original_muscles': digest(sources[0]), 'current_muscles': digest(sources[1]),
            'current_bones': digest(current_bones), 'matched_muscles': [29, 30],
            'matched_cortical_banks': sorted(map(int, manifest['matched_cortical_banks']))}


def load_compliance(assets, manifest, bindings):
    """Project the existing material flexibility onto the bound sites on both ends."""
    for name in ('cortical-flexibility.bin', 'cortical-free-index.bin'):
        require(digest(assets / name) == manifest['files'][name], f'Invalid material atlas {name}')
    vectors = np.memmap(assets / 'cortical-flexibility.bin', '<f4', mode='r').reshape(-1, 4)
    index = np.fromfile(assets / 'cortical-free-index.bin', '<u4')
    diagnostics = []
    for spec in manifest['contact_flexibility']:
        bindings_here = [b for b in bindings if b['muscle'] == spec['muscle']]
        if not bindings_here:
            continue
        n = spec['nodes']
        k = spec['input_nodes']
        offset = spec['first_vector']
        atlas = vectors[offset:offset + n*k*3].reshape(n, k, 3, 4)
        blocks = np.zeros((len(bindings_here), 3, len(bindings_here), 3))
        for j, target in enumerate(bindings_here):
            parents = np.array(target['material_parent_nodes'], dtype=int)
            live = np.array(target['weights']) != 0
            cols = index[parents[live]]
            require(np.all(cols < k), 'A loaded parent has no material response column')
            for i, observed in enumerate(bindings_here):
                rows = np.array(observed['material_parent_nodes']) - spec['first_node']
                block = atlas[rows[:, None], cols[None, :], :, :3]
                blocks[i, :, j, :] = np.einsum('i,ijba,j->ab', observed['weights'], block,
                                                np.asarray(target['weights'])[live])
        matrix = blocks.reshape(3 * len(bindings_here), -1)
        symmetric = (matrix + matrix.T) * 0.5
        spectrum = np.linalg.eigvalsh(symmetric)
        scale = np.linalg.norm(matrix, 2)
        require(scale > 0 and np.isfinite(matrix).all(), 'Invalid reference compliance')
        reciprocity = np.linalg.norm(matrix - matrix.T, 2) / scale
        require(reciprocity < 1e-6 and spectrum[0] >= -1e-6 * scale,
                'Material projection fails reciprocity or signed energy')
        diagnostics.append({'muscle': spec['muscle'], 'sites': len(bindings_here),
                            'reference_compliance_eigenvalue_range_m_per_N':
                                [float(spectrum[0]), float(spectrum[-1])],
                            'relative_reciprocity_error': float(reciprocity)})
    return diagnostics


def bind(case, prior_config, prior_pose, assets, columns, library):
    live = read(case / 'launch.json')
    cap_file = case / 'native-traces/accepted-geometry/step-1.receipt.json'
    cap = read(cap_file)
    sites = read(columns)
    manifest = read(assets / 'manifest.json')
    require(sites['geometry_library'] == digest(library) and
            sites['pack'] == cap['pack_file_sha256'], 'Mismatched site inputs')
    require(digest(prior_pose) == manifest['common_material_reference']['pose'] ==
            manifest['surface_material_reference']['pose'], 'Wrong registration pose for the material atlas')
    for f in (live['argv'][3], live['argv'][live['argv'].index('--soft-tissue-payload') + 1]):
        require(digest(Path(f)) == live['pins'][f], 'Current anatomical payload changed')
    source = confirm_sources(read(prior_config)['argv'], live['argv'], manifest)
    prior = relative_transforms(read(prior_pose)['initial_anatomical_registration']['body_poses'])
    now = relative_transforms(cap['initial_anatomical_registration']['body_poses'])
    err = max(abs(now[i] - prior[i]).max() for i in (1, 128, 131, 145))
    require(err <= 2e-7, 'The material atlas and current starting skeleton disagree')

    def table(name, dtype, width):
        p = assets / (name + '.bin')
        require(digest(p) == manifest['files'][p.name], f'Invalid atlas payload {name}')
        return np.fromfile(p, dtype).reshape(-1, width, 4)

    nodes = table('pelvic-reference-nodes', '<f4', 3).astype(float)
    nodes = nodes[:, 0, :3] + nodes[:, 1, :3]
    boundary = table('constitutive-boundary-reference', '<f4', 3).astype(float)
    xyz = boundary[:, 0, :3] + boundary[:, 1, :3]
    vertex_data = table('closed-material-vertices', '<f4', 2)
    parents = vertex_data[:, 1].copy().view('<u4')
    shape = vertex_data[:, 0].astype(float) + boundary[:, 2]
    require(np.max(abs(shape.sum(1) - 1)) < 1e-10 and
            np.max(abs(np.einsum('ij,ijk->ik', shape, nodes[parents]) - xyz)) < 1e-12,
            'Boundary coordinates disagree with their P1 representation')
    faces = table('closed-material-triangles', '<u4', 3)[:, 0, :3]
    pose = next(row for row in cap['accepted_body_poses'] if row['body_index'] == 128)
    rot = Rotation.from_quat(pose['quaternion_xyzw'])
    origin = np.asarray(pose['position_m'])
    result = []
    for spec, span in zip(manifest['contact_flexibility'], manifest['closed_reverse']['sides']):
        sid = spec['muscle']
        local_sites = [s for s in sites['columns'] if s['muscle_id'] == sid]
        if not local_sites:
            continue
        first, count = span['first_face'], span['faces']
        tri = np.ascontiguousarray(faces[first:first + count], 'i4')
        deep = np.array([s['deep_fat_point_m'] for s in local_sites])
        closest, distance, which = nearest((np.ascontiguousarray(xyz), tri),
                                           rot.inv().apply(deep - origin), library)
        for s, point, gap, face in zip(local_sites, closest, distance, which):
            corners = tri[face]
            a, b, c = xyz[corners]
            uv = np.linalg.lstsq(np.column_stack((b-a, c-a)), point-a, rcond=None)[0]
            bary = np.r_[1 - uv.sum(), uv]
            require(bary.min() >= -1e-8 and np.linalg.norm(bary @ xyz[corners] - point) < 2e-9,
                    'Material hit outside its registered facet')
            ids, inverse = np.unique(parents[corners].ravel(), return_inverse=True)
            w = np.bincount(inverse, weights=(bary[:, None] * shape[corners]).ravel())
            require(ids.min() >= spec['first_node'] and ids.max() < spec['first_node'] + spec['nodes'],
                    'Boundary transfer crossed anatomical owners')
            moment_error = np.linalg.norm(w @ nodes[ids] - point)
            require(abs(w.sum() - 1) < 1e-10 and moment_error < 2e-9 and w.min() >= -1e-8,
                    'Receiver fails force or moment reproduction')
            result.append({'skin_node': s['node'], 'muscle': sid, 'reference_face': int(first + face),
                           'receiver_distance_m': float(gap), 'display_distance_m': s['unsigned_distance_m'],
                           'material_parent_nodes': ids.tolist(), 'weights': w.tolist(),
                           'first_moment_error_m': float(moment_error)})
    require(len(result) == sum(s['muscle_id'] in (29, 30) for s in sites['columns']),
            'Unbound pelvic sites')
    return {'source': source, 'registered_skeleton_max_matrix_error': float(err),
            'atlas_sha256': digest(assets / 'manifest.json'), 'pose': digest(cap_file),
            'original_pose': digest(prior_pose), 'sites_sha256': digest(columns),
            'unbound_sites': [s['node'] for s in sites['columns'] if s['muscle_id'] not in (29, 30)],
            'frame': 'Reference muscle volume carried in the accepted pelvis frame. Unloaded receiver mapping only.',
            'reference_response': load_compliance(assets, manifest, result), 'receivers': result}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('case', 'prior_config', 'prior_pose', 'assets', 'columns', 'library', 'output'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    result = bind(a.case, a.prior_config, a.prior_pose, a.assets, a.columns, a.library)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
