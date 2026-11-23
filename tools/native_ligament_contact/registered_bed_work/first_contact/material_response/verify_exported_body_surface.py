"""Audit the entire exported native material surface against the pinned release anatomy."""
import json
from pathlib import Path
import runpy
import sys

import numpy as np

from recount_registered_crossings import crossings, digest, read


def verify(parent, initial, run, change, export, output):
    native = read(export / 'native-import.json')
    pack = run / 'native-traces/resting-human.mrvpack'
    pose = run / 'native-initial-bodies.json'
    sealed = read(initial / 'release-envelope.json')
    src = parent / 'native-input.npz'
    patch = change / 'registered-positions.npz'
    if digest(pack) != sealed['pack_sha256'] or digest(pose) != sealed['pose_sha256']:
        raise ValueError('Mismatched native release pose or asset pack')
    if digest(src) != read(parent / 'harmonic/result.json')['source'] or digest(patch) != read(change / 'native-bank-comparison.json')['patch']:
        raise ValueError('Mismatched material source or candidate')
    node = export / 'mesh.node'
    if digest(node) != native['node_sha256']:
        raise ValueError('Native material input mismatch')
    nodes = np.loadtxt(node, skiprows=1, dtype='f8')
    if not np.array_equal(nodes[:,0], np.arange(1, native['nodes']+1)):
        raise ValueError('Unexpected native node IDs')
    with np.load(src) as mesh, np.load(patch) as changed:
        outer, faces = mesh['node_O'], mesh['face_O']
        skin = nodes[outer,1:].astype(np.float32)
        source = mesh['x']
        expected = source.copy()
        expected[changed['nodes']] = changed['x']
        if not np.array_equal(skin, expected[outer]):
            raise ValueError('Emitted outer nodes differ from registered repair')
        changed_faces = np.flatnonzero(np.any(source[outer] != skin,axis=1)[faces].any(axis=1))
    poses = {r['body_index']: dict(position_m=np.asarray(r['current']['position_world_m'],dtype='f4').tolist(),
                                    quaternion_xyzw=np.asarray(r['current']['orientation_world_xyzw'],dtype='f4').tolist())
             for r in read(pose)['all_bodies']}
    reader = next(d/'resting-surface-clearance-001/audit.py' for d in parent.parents if (d/'resting-surface-clearance-001/audit.py').is_file())
    bank = next(d/'audit_registered_initial_envelope.py' for d in parent.parents if (d/'audit_registered_initial_envelope.py').is_file())
    sys.path.insert(0, str(reader.parent/'src'))
    api = runpy.run_path(str(reader))
    payload = pack.read_bytes()
    (_, _, reference_points, reference_faces), = api['primitives'](payload, poses, skin_only=True)
    if not np.array_equal(reference_points, source[outer]) or not np.array_equal(reference_faces, faces):
        raise ValueError('Skin triangle ordering or source pose differs')
    output.mkdir(exist_ok=True)
    runpy.run_path(str(bank))['scan_banks'](api, payload, poses, skin, faces, {}, output, 'all-banks.json')
    measured = crossings(output/'target-witnesses.jsonl')
    affected = set(changed_faces)
    expected_pairs = {p for p in crossings(initial/'target-witnesses.jsonl') if p[3] not in affected}
    expected_pairs.update((s,b,k,int(changed_faces[i]),j) for s,b,k,i,j in crossings(change/'target-witnesses.jsonl'))
    if measured != expected_pairs:
        raise ValueError(('Independent triangle-pair discrepancy',len(measured-expected_pairs),len(expected_pairs-measured)))
    result = dict(native_pack=digest(pack), pose=digest(pose), material_nodes=digest(node), reference_input=digest(src),
                  audited_outer_triangles=len(faces), audited_banks=sum(read(output/'all-banks.json')['target_banks_by_semantic'].values()),
                  audit_program=digest(bank), intersection_program=digest(reader), witnesses=digest(output/'target-witnesses.jsonl'),
                  physical_skin_to_internal_anatomy_pairs=len(measured), differing_pair_identities=0, skin_to_bone_pairs=sum(q[0]==51004 for q in measured))
    (output/'exported-clearance.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    verify(*(Path(s).resolve() for s in sys.argv[1:]))
