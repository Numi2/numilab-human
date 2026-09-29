"""Source-rest, full-support laterality audit of explicitly bilateral anatomy.

This checks the actual selected native anatomy representation. A centroid can
be on the expected side while a small disconnected source component is not.
Only named bilateral organs and eye/lung families are assessed; left/right
heart or liver labels are not assumed to lie wholly across the body midline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from . import model as human
from . import lung_envelope as lung
from . import surface_topology_repair as repair
from . import whole_body_surface_gate as geometry_gate

SCHEMA = 'numi.human.source-rest-bilateral-support-laterality.v1'
COUNTS = {
    'left kidney': 1, 'right kidney': 1,
    'left eye': 22, 'right eye': 21,
    'left lacrimal gland': 1, 'right lacrimal gland': 1,
    'left adrenal gland': 1, 'right adrenal gland': 1,
    'left testis': 1, 'right testis': 1,
    'left ureter': 1, 'right ureter': 1,
    'left lung lobe': 2, 'right lung lobe': 3,
}


def require(ok, message):
    if not ok:
        raise human.ImportError('anatomical laterality: ' + message)


def quaternion_rotation(q):
    import numpy as np
    q = np.asarray(q, dtype=np.float64)
    require(q.shape == (4,) and np.isfinite(q).all(), 'invalid native quaternion')
    norm = float(np.linalg.norm(q))
    require(norm > 0 and abs(norm-1) < 1e-3, 'native quaternion norm')
    x, y, z, w = q/norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def source_metadata(base):
    specs, manifest_hashes = repair.source_specs(base)
    paths = [human.REPOSITORY_ROOT/'Build/lung-source-coverage-20260929/payload/bodyparts3d-myosim-torso-anatomy.manifest.json',
             human.REPOSITORY_ROOT/'Build/organ-family-coverage-20260929/payload.final/source-organ-family-anatomy.manifest.json',
             base.with_name('source-organ-family-anatomy.manifest.json')]
    docs = [human.read_json(p) for p in paths]
    rows = [*docs[0]['source']['surfaces'], *lung.configuration()['objects'],
            *docs[1]['surfaces'], *docs[2]['surfaces']]
    require(len(rows) == 579 and [r['stable_id'] for r in rows] == list(range(1, 580)),
            'incomplete ordered semantic source metadata')
    for row, spec in zip(rows, specs, strict=True):
        require((row.get('member_id') or row.get('object_name')) ==
                (spec.get('member_id') or spec.get('object_name')),
                'semantic member differs from pinned mesh source')
    return rows, manifest_hashes


def cohort(metadata):
    selected = {}
    actual = Counter()
    for row in metadata:
        name = row.get('label') or row.get('source_name') or row.get('object_name')
        if row['stable_id'] in range(305, 310):
            name = 'left lung lobe' if 'left lung' in name.lower() else 'right lung lobe'
        if name in COUNTS:
            selected[row['stable_id']] = {'family': name, 'side': name.split()[0]}
            actual[name] += 1
    require(dict(actual) == COUNTS, f'bilateral semantic coverage changed: {dict(actual)}')
    require(len(selected) == 58, 'bilateral surface cohort count')
    return selected


def support_metrics(world, faces, torso_origin, left_axis, side_sign, source_faces):
    import numpy as np
    world = np.asarray(world, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    require(world.ndim == 2 and world.shape[1] == 3 and len(world) > 0
            and faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
            and np.isfinite(world).all() and (faces >= 0).all() and (faces < len(world)).all(),
            'invalid selected triangle support')
    require(len(source_faces) == len(faces), 'selected source-face ancestry count')
    signed = ((world-torso_origin) @ left_axis)*side_sign
    tri = world[faces]
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1)*.5
    require(np.isfinite(area).all() and float(area.sum()) > 0, 'selected triangle area')
    centroid = (tri.mean(axis=1)*area[:, None]).sum(axis=0)/area.sum()
    centroid_signed = float(((centroid-torso_origin) @ left_axis)*side_sign)
    wrong_vertices = np.flatnonzero(signed <= 0)
    wrong_faces = np.flatnonzero((signed[faces] <= 0).any(axis=1))
    return {
        'all_selected_vertices_on_expected_side': len(wrong_vertices) == 0,
        'centroid_on_expected_side': centroid_signed > 0,
        'centroid_signed_midline_distance_m': centroid_signed,
        'minimum_signed_midline_distance_m': float(signed.min()),
        'wrong_side_vertex_count': len(wrong_vertices),
        'wrong_side_triangle_count': len(wrong_faces),
        'wrong_side_triangle_area_fraction': float(area[wrong_faces].sum()/area.sum()),
        'wrong_side_selected_face_ids': wrong_faces.tolist(),
        'wrong_side_source_face_ids': sorted({source_faces[i] for i in wrong_faces}),
    }


def inspect_selected(payload, selected_rows, metadata, pose):
    import numpy as np
    _, header, records, vertices, indices = lung.decode(payload)
    require(header[0] == 5 and header[1] in (598, 599), 'candidate ABI/source count')
    require(len(selected_rows) == 579 and
            [x['source_stable_id'] for x in selected_rows] == list(range(1, 580)),
            'selected view incomplete')
    require(pose['surface_count'] == header[1] and pose['schema'] == 'numi.human.native-torso-anatomy-pose-snapshot.v1',
            'native pose does not own all anatomy surfaces')
    bodies = {b['body_index']: b for b in pose['bodies']}
    require(len(bodies) == len(pose['bodies']) and 20 in bodies, 'native torso pose missing')
    torso = bodies[20]
    torso_origin = np.asarray(torso['position_world_m'], dtype=np.float64)
    torso_rotation = quaternion_rotation(torso['orientation_world_xyzw'])
    left_axis = torso_rotation @ np.array([0., 0., -1.])
    # This published source-rest profile maps anatomical left to world +X.
    require(float(left_axis[0]) > .999 and
            all(abs(b['position_world_m'][0]-torso_origin[0]) < .002 for b in bodies.values()),
            'source-rest midline/orientation identity changed')
    members = cohort(metadata)
    candidate_manifest_path = payload.with_name(repair.MANIFEST)
    if header[1] == 598:
        require(candidate_manifest_path.is_file(), 'missing source-preserving candidate manifest')
        candidate_manifest = human.read_json(candidate_manifest_path)
        ancestry = {r['candidate_stable_id']: r['source_derivation']['source_face_ids']
                    for r in candidate_manifest['candidates']}
    else:
        ancestry = {}
    output = []
    for selected in selected_rows:
        sid = selected['source_stable_id']
        if sid not in members:
            continue
        visible_id = selected['visible_stable_id']
        owner, fv, nv, fi, ni, stable, layer, reserved = map(int, records[visible_id-1])
        require(stable == visible_id and layer == selected['source_layer_code']
                and owner in bodies and ni % 3 == 0 and reserved == 0,
                'selected native record/pose identity')
        body = bodies[owner]
        local = vertices[fv:fv+nv, :3].astype(np.float64)
        world = local @ quaternion_rotation(body['orientation_world_xyzw']).T + body['position_world_m']
        side_sign = 1 if members[sid]['side'] == 'left' else -1
        faces = (indices[fi:fi+ni].reshape(-1, 3)-fv).astype(np.int64)
        source_faces = ancestry.get(visible_id)
        if source_faces is None:
            source_faces = list(range(len(faces)))
        measured = support_metrics(world, faces, torso_origin, left_axis, side_sign, source_faces)
        output.append({
            'source_stable_id': sid, 'visible_stable_id': visible_id,
            'source_member_id': metadata[sid-1].get('member_id'),
            'source_object_name': metadata[sid-1].get('object_name'),
            'family': members[sid]['family'], 'expected_side': members[sid]['side'],
            'geometry_embedded_candidate': selected['selected_closed_embedded_surface_candidate'],
            **measured,
        })
    require(len(output) == 58 and [x['source_stable_id'] for x in output] == sorted(members),
            'missing bilateral selected geometry')
    return output


def audit(base, indexed, quotient, candidate, proof, selection, native_dir, pose_path):
    base, candidate, pose_path = (Path(x).resolve() for x in (base, candidate, pose_path))
    gate = geometry_gate.audit(base, indexed, quotient, candidate, proof, selection, native_dir)
    gate_path = Path(quotient).parent/'selected-surface-gate.json'
    if gate_path.is_file():
        require(human.read_json(gate_path) == gate, 'published selected surface gate differs from replay')
    source_rest = next(x for x in gate['native_profiles'] if x['profile'] == 'source-rest')
    require(human.sha256(pose_path) == source_rest['pose_snapshot_sha256'],
            'native source-rest snapshot differs from audited packet')
    metadata, provenance = source_metadata(base)
    pose = human.read_json(pose_path)
    rows = inspect_selected(candidate, gate['rows'], metadata, pose)
    defects = [r for r in rows if not r['all_selected_vertices_on_expected_side']]
    return {
        'schema': SCHEMA,
        'baseline_payload_sha256': human.sha256(base),
        'selected_payload_sha256': human.sha256(candidate),
        'selected_geometry_gate_sha256': human.sha256(gate_path) if gate_path.is_file() else None,
        'source_rest_pose_sha256': human.sha256(pose_path),
        'native_pack_sha256': source_rest['visual_packet_sha256'],
        'semantic_manifest_sha256': provenance,
        'source_surface_count': 579,
        'audited_bilateral_source_surface_count': 58,
        'centroid_side_pass_count': sum(r['centroid_on_expected_side'] for r in rows),
        'full_support_side_pass_count': len(rows)-len(defects),
        'full_support_side_failure_source_ids': [r['source_stable_id'] for r in defects],
        'geometry_embedded_but_laterality_failed_source_ids': [r['source_stable_id'] for r in defects
            if r['geometry_embedded_candidate']],
        'not_assessed_source_surface_count': 579-len(rows),
        'rows': rows,
        'clinical_anatomy': False, 'cross_surface_placement': 'not_checked',
        'boundary': 'Only full selected triangle support against the source-rest sagittal midline for 58 explicitly bilateral surfaces. Midline strays are source-bound defects; a passing side check does not prove clinical anatomy, component containment, cross-surface placement or mechanics.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'indexed', 'quotient', 'candidate', 'proof', 'selection',
                 'native-dir', 'pose', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.base, args.indexed, args.quotient, args.candidate,
                   args.proof, args.selection, args.native_dir, args.pose)
    human.write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('audited_bilateral_source_surface_count',
        'centroid_side_pass_count', 'full_support_side_pass_count',
        'full_support_side_failure_source_ids',
        'geometry_embedded_but_laterality_failed_source_ids')}), flush=True)


if __name__ == '__main__':
    main()
