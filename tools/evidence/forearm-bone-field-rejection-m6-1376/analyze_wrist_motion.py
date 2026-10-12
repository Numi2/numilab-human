from __future__ import annotations
import gc, hashlib, json, math, struct, sys
from pathlib import Path
import numpy as np

A = Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
OUT = A / 'wrist-motion-census-001'
RIGID = Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid')
CA_SOURCE = Path('/Users/n/numi-human-anatomy-publication-1308/src')
sys.path.insert(0, str(CA_SOURCE))
from numilab_human import common_atlas_skin_clearance as ca

RUNS = {
    'N16_back412_right177_16s': {
        'root': A / 'back-forearm-native-014-016-back412-right177',
        'steps': [0, 3807, 4767, 5023, 5599, 6207, 7423, 8000],
    },
    'N14_right177_60s': {
        'root': A / 'back-forearm-native-014-014-right177-60s',
        'steps': [0, 7999, 10015, 15007, 19999, 24991, 27999, 30000],
    },
}
SURFACE_KEYS = {'PQ': (51005, 130), 'FDS': (51005, 132), 'FDP': (51005, 134)}
EXPECTED_LABELS = {130: 'left pronator quadratus', 132: 'left flexor digitorum superficialis', 134: 'left flexor digitorum profundus'}
WRIST_JOINTS = {41: {'joint': 43, 'parent_body': 43, 'child_body': 44, 'axis_label': 'authored local axis (0,0,1)'},
                42: {'joint': 44, 'parent_body': 44, 'child_body': 45, 'axis_label': 'authored local axis (1,0,0)'},
                79: {'joint': 93, 'parent_body': 93, 'child_body': 94, 'axis_label': 'authored local axis (0,0,1)'},
                80: {'joint': 94, 'parent_body': 94, 'child_body': 95, 'axis_label': 'authored local axis (-1,0,0)'}}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw,
                     aw*bw-ax*bx-ay*by-az*bz], dtype=np.float64)

def qconj(a):
    return np.array([-a[0], -a[1], -a[2], a[3]], dtype=np.float64)

def extract_model():
    raw = RIGID.read_bytes()
    magic, pabi, eabi, source_count, body_count, joint_count, nq, nv, root, virtual, reserved, embedded = struct.unpack_from('<8s10I32s', raw, 0)
    assert magic == b'NHRIGID2' and pabi == 1 and eabi == 5 and body_count == 157 and joint_count == 156 and nq == 129 and nv == 128
    joint_off = 224 + 160 * body_count
    default_q_off = joint_off + 144 * joint_count + 64 * nv
    default_q = np.frombuffer(raw, dtype='<f4', count=nq, offset=default_q_off).astype(np.float64)
    joints = {}
    for ji in range(joint_count):
        x = struct.unpack_from('<8I28f', raw, joint_off + 144 * ji)
        parent, child, jt, flags, qo, nqj, vo, nvj = x[:8]
        if qo in WRIST_JOINTS:
            axis = np.asarray(x[8:11], dtype=np.float64)
            axis /= np.linalg.norm(axis)
            joints[qo] = {'joint_index': ji, 'parent_body': parent, 'child_body': child,
                          'joint_type': jt, 'q_offset': qo, 'nq': nqj, 'v_offset': vo, 'nv': nvj,
                          'axis': axis, 'parent_rotation': np.asarray(x[28:32], dtype=np.float64),
                          'child_rotation': np.asarray(x[32:36], dtype=np.float64)}
    assert set(joints) == set(WRIST_JOINTS)
    for qo, expected in WRIST_JOINTS.items():
        j = joints[qo]
        assert j['joint_index'] == expected['joint'] and j['parent_body'] == expected['parent_body'] and j['child_body'] == expected['child_body']
        assert j['joint_type'] == 0 and j['nq'] == j['nv'] == 1 and abs(default_q[qo]) < 1e-7
    return joints, default_q, {'sha256': sha(RIGID), 'nhrigid_source_archive_sha256': embedded.hex(),
                               'source_body_count': source_count, 'engine_body_count': body_count,
                               'joint_count': joint_count, 'nq': nq, 'nv': nv}

def q_at(receipt, j):
    poses = {int(x['body_index']): np.asarray(x['quaternion_xyzw'], dtype=np.float64) for x in receipt['accepted_body_poses']}
    parent_q = poses[j['parent_body']]
    child_q = poses[j['child_body']]
    parent_joint_world = qmul(parent_q, j['parent_rotation'])
    child_joint_world = qmul(child_q, j['child_rotation'])
    motion = qmul(qconj(parent_joint_world), child_joint_world)
    motion /= np.linalg.norm(motion)
    if motion[3] < 0: motion = -motion
    proj = float(np.dot(motion[:3], j['axis']))
    orthogonal = float(np.linalg.norm(motion[:3] - proj * j['axis']))
    return 2.0 * math.atan2(proj, float(motion[3])), orthogonal

def manifest_labels(path):
    x = json.loads(path.read_text())
    out = {}
    for row in x['source']['surfaces']:
        sid = int(row['stable_id'])
        if sid in EXPECTED_LABELS:
            out[sid] = row['label']
    assert out == EXPECTED_LABELS, out
    return out

def surface_centroids(pack):
    pos, surfaces, _ = ca._pack_surfaces(pack, set(SURFACE_KEYS.values()))
    out = {}
    for name, key in SURFACE_KEYS.items():
        row = surfaces[key]
        faces = np.asarray(row['faces'], dtype=np.int64)
        ids = np.unique(faces.reshape(-1))
        xyz = np.asarray(pos[ids], dtype=np.float64)
        assert xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) > 0
        out[name] = {'centroid_m': xyz.mean(axis=0), 'unique_vertex_count': int(len(ids)), 'triangle_count': int(len(faces))}
    del pos, surfaces
    gc.collect()
    return out

joints, default_q, model_info = extract_model()
pins = {str(Path(__file__)): sha(Path(__file__)), str(RIGID): model_info['sha256'], str(Path(ca.__file__)): sha(Path(ca.__file__))}
report = {'schema': 'numi.human.wrist-motion-census.v1',
          'scope': 'Read-only reconstruction of four wrist revolute coordinates from accepted rigid body poses, plus centroid-motion comparison for the captured left PQ/FDS/FDP passive rows. No simulation, runtime changes, or admission claim.',
          'method': {'joint_pose_formula': 'motion = inverse(parent_body_quaternion * parent_joint_rotation) * (child_body_quaternion * child_joint_rotation); q = 2 atan2(dot(motion.xyz, authored_axis), motion.w).',
                     'validation': 'NHRIGID2 joint type/indices/axes are pinned; all four coordinates reconstruct approximately zero at accepted step 0 and retain quaternion-vector orthogonal residual below the threshold recorded per sample.',
                     'geometry': 'accepted MRVPACK stable rows 130/132/134; centroid is arithmetic mean of unique captured surface vertex positions; relative vector is centroid(other) - centroid(PQ), with changes relative to same-run step 0.',
                     'coordinate_units': 'accepted geometry and body positions are metres; report geometry centroids/displacements in millimetres; wrist angles in degrees.',
                     'interpretation_limit': 'Body-pose-derived q is reconstructed from accepted body orientations, not a directly serialized generalized-coordinate trace. Surface centroid motion is descriptive whole-surface kinematics, not local contact/separation or anatomical efficacy.'},
          'model': model_info,
          'surface_labels': EXPECTED_LABELS,
          'wrist_coordinates': {str(qo): {**WRIST_JOINTS[qo], **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in joints[qo].items() if k not in ('parent_rotation','child_rotation')}} for qo in WRIST_JOINTS},
          'runs': {}, 'pins': pins, 'complete': False}

for run_name, spec in RUNS.items():
    root = spec['root']; native = root / 'baseline/native-run'
    inv = native / 'invocation.json'; execution = root / 'baseline/execution.json'
    tiss = root / 'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
    manifest = root / 'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json'
    assert all(p.is_file() for p in [inv, execution, tiss, manifest])
    inv_data = json.loads(inv.read_text()); exec_data = json.loads(execution.read_text())
    assert exec_data.get('returncode') == 0 and exec_data.get('timed_out') is False
    assert exec_data.get('native_argv_matches_prepared_cli_preview') is True and exec_data.get('native_environment_matches_prepared_cli_preview') is True
    canonical_assets = {str(Path(k).resolve()): v for k, v in inv_data['asset_sha256'].items()}
    assert canonical_assets.get(str(RIGID.resolve())) == model_info['sha256']
    assert canonical_assets.get(str(tiss.resolve())) == sha(tiss)
    labels = manifest_labels(manifest)
    assert json.loads(manifest.read_text())['payload']['sha256'] == sha(tiss)
    for p in [inv, execution, tiss, manifest]: pins[str(p)] = sha(p)
    entry = {'run_root': str(root), 'accepted_steps': [], 'invocation_sha256': pins[str(inv)],
             'execution_sha256': pins[str(execution)], 'tissue_sha256': pins[str(tiss)],
             'manifest_sha256': pins[str(manifest)], 'surface_labels': labels,
             'native_execution': {k: exec_data.get(k) for k in ['schema','returncode','timed_out','wall_seconds','utc_finished','native_argv_matches_prepared_cli_preview','native_environment_matches_prepared_cli_preview','qualification']},
             'captured_geometry': [], 'wrist_summary': {}, 'run_variant_note': 'N16 and N14 are distinct native composition/run variants; within-run deltas use each run’s own step 0.'}
    baseline_pair_vectors = None
    for step in spec['steps']:
        capdir = native / 'accepted-geometry'
        receipt_path = capdir / f'step-{step}.receipt.json'
        pack_path = capdir / f'step-{step}.mrvpack'
        assert receipt_path.is_file() and pack_path.is_file()
        receipt = json.loads(receipt_path.read_text())
        assert int(receipt['accepted_step']) == step and int(receipt['accepted_body_pose_count']) == 157
        pack_hash = sha(pack_path)
        assert receipt['pack_file_sha256'] == pack_hash
        assert Path(receipt['accepted_pack_path']).name == pack_path.name
        pins[str(receipt_path)] = sha(receipt_path); pins[str(pack_path)] = pack_hash
        qvals = {}
        for qo, j in joints.items():
            qrad, orth = q_at(receipt, j)
            assert orth < 1e-5, (run_name, step, qo, orth)
            qzero, _ = q_at(json.loads((capdir / 'step-0.receipt.json').read_text()), j)
            qvals[str(qo)] = {'degrees': math.degrees(qrad - qzero), 'absolute_degrees_from_nhrigid_default': math.degrees(qrad - float(default_q[qo])), 'off_axis_quaternion_vector_norm': orth}
        capture = {'step': step, 'time_s': float(receipt['accepted_time_s']), 'receipt_sha256': pins[str(receipt_path)],
                   'pack_sha256': pack_hash, 'q_degrees_relative_step0': qvals}
        geo = surface_centroids(pack_path)
        centers = {name: item['centroid_m'] for name, item in geo.items()}
        rel = {}
        for a, b in [('PQ','FDS'), ('PQ','FDP'), ('FDS','FDP')]:
            vec = centers[b] - centers[a]
            if step == 0: baseline_pair_vectors = baseline_pair_vectors or {}; baseline_pair_vectors[f'{a}_to_{b}'] = vec.copy()
            delta = vec - baseline_pair_vectors[f'{a}_to_{b}']
            rel[f'{a}_to_{b}'] = {'centroid_vector_mm': (vec * 1e3).tolist(),
                                  'centroid_distance_mm': float(np.linalg.norm(vec) * 1e3),
                                  'relative_vector_change_from_step0_mm': (delta * 1e3).tolist(),
                                  'relative_displacement_from_step0_mm': float(np.linalg.norm(delta) * 1e3)}
        capture['surface_centroids'] = {name: {'centroid_mm': (item['centroid_m'] * 1e3).tolist(),
                                               'unique_vertex_count': item['unique_vertex_count'],
                                               'triangle_count': item['triangle_count']} for name, item in geo.items()}
        capture['pairwise_centroid_motion'] = rel
        entry['accepted_steps'].append(step); entry['captured_geometry'].append(capture)
        del geo, centers
        gc.collect()
        print(json.dumps({'run': run_name, 'step': step, 'q': {k: round(v['degrees'], 6) for k,v in qvals.items()},
                          'PQ_FDS_relative_delta_mm': round(rel['PQ_to_FDS']['relative_displacement_from_step0_mm'], 6),
                          'PQ_FDP_relative_delta_mm': round(rel['PQ_to_FDP']['relative_displacement_from_step0_mm'], 6)}), flush=True)
    summary = {}
    for qo in WRIST_JOINTS:
        vals = [(x['step'], x['q_degrees_relative_step0'][str(qo)]['degrees']) for x in entry['captured_geometry']]
        lo = min(vals, key=lambda t:t[1]); hi = max(vals, key=lambda t:t[1]); peak = max(vals, key=lambda t:abs(t[1]))
        summary[str(qo)] = {'min_deg': lo[1], 'min_step': lo[0], 'max_deg': hi[1], 'max_step': hi[0],
                            'peak_abs_deg': abs(peak[1]), 'peak_abs_step': peak[0], 'range_deg': hi[1]-lo[1]}
    entry['wrist_summary'] = summary
    report['runs'][run_name] = entry

report['pins'] = pins
report['inputs_unchanged_at_end'] = all(sha(Path(p)) == v for p,v in pins.items())
report['all_captures_zero_reference_within_1e-4_deg'] = all(abs(x['q_degrees_relative_step0'][str(qo)]['degrees']) < 1e-4
   for run in report['runs'].values() for x in run['captured_geometry'] if x['step'] == 0 for qo in WRIST_JOINTS)
report['all_step0_wrist_q_matches_nhrigid_default_within_1e-4_deg'] = all(abs(x['q_degrees_relative_step0'][str(qo)]['absolute_degrees_from_nhrigid_default']) < 1e-4
   for run in report['runs'].values() for x in run['captured_geometry'] if x['step'] == 0 for qo in WRIST_JOINTS)
report['all_wrist_relative_orientation_residuals_below_1e-5'] = all(x['q_degrees_relative_step0'][str(qo)]['off_axis_quaternion_vector_norm'] < 1e-5
   for run in report['runs'].values() for x in run['captured_geometry'] for qo in WRIST_JOINTS)
report['complete'] = (report['inputs_unchanged_at_end'] and report['all_captures_zero_reference_within_1e-4_deg'] and report['all_step0_wrist_q_matches_nhrigid_default_within_1e-4_deg'] and report['all_wrist_relative_orientation_residuals_below_1e-5'])
(OUT / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
print('COMPLETE', report['complete'], 'pins', len(pins))
