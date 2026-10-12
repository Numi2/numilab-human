#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, os, pickle, shutil, struct, sys, time, traceback
from collections import defaultdict
from fractions import Fraction as F
from pathlib import Path
import numpy as np

A = Path('/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276')
ROOT = Path('/Users/numi/numi-human-fds132-m6-1350')
PUB = Path('/Users/numi/numi-human-left-pq-publication-1344')
PARENT = A / 'forearm054-native-diagnostic-007-candidate060'
OUT = A / 'forearm054-native-diagnostic-009-refined085'
PARENT_TISS = PARENT / 'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
PARENT_MANIFEST = PARENT_TISS.with_suffix('.manifest.json')
PARENT_RECEIPT = PARENT / 'resting-anatomy-receipt.json'
PARENT_DECL = PARENT / 'baseline/run-declaration.json'
PARENT_PREP = PARENT / 'preparation.json'
PARENT_LAUNCH = PARENT / 'launch_arm.py'

CAND = ROOT / 'attempt-085-interface-refinement-2mm-001/result-001/candidate.pkl'
CAND_REPORT = ROOT / 'attempt-085-interface-refinement-2mm-001/result-001/report.json'
CAND_SOURCE_REPORT = ROOT / 'attempt-085-interface-refinement-2mm-001/result-001/source-checks.json'
BASE_MAP = ROOT / 'attempt-061-step0-capture-remesh-diagnostic/candidate060-adapted-source-parent-map-v2.pkl'
BASE_EXACT_REPORT = ROOT / 'attempt-060-paired-cut-sliver-contraction-001/exact-check-001/report.json'
FIELD = ROOT / 'attempt-088-registered-route-field-refined-2mm-001/result-001/route-anchor-reference-field-001.npz'
FRAME_PROOF = ROOT / 'attempt-088-registered-route-field-refined-2mm-001/result-001/registration-frame-proof.json'
REGISTRATION = ROOT / 'registration-source-b1b410ad.json'
RECOVERY = ROOT / 'registration-source-b1b410ad.recovery-receipt.json'

CARRIERS_087 = ROOT / 'attempt-087-registered-shared-field-carriers-load-001'
CARRIER_087 = CARRIERS_087 / 'shared-field-carriers.nhtissue'
EXT_087 = CARRIERS_087 / 'passive-tissue-parent-bindings.json'
CARRIER_PREP_087 = CARRIERS_087 / 'preparation.json'
ENCODER = PUB / 'src/numilab_human/passive_tissue_parent_carriers.py'
COMPOSER = PUB / 'src/numilab_human/passive_attachment_composition.py'
FORWARD = A / 'neck-back-retained-pose-audit-002/forward_all_positive_weights.py'

RAW_SOURCE_TISS = A / 'left-fds-native-m6-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
RIGID = Path('/Users/numi/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid')
ARCHIVE = Path('/Users/numi/numi-human-retained-delivery-20261009/skin-source-binding-1202/source-inputs/myosim/myo_sim-33c89c2b.tar.gz')
W = Path('/Users/numi/numi-lab-parent-bindings-1353')
B = Path('/Users/numi/numi-lab-parent-bindings-build-1353-003')
BIN = B / 'bin/numi-human-native'
EXPECTED = {
    'candidate': '4792b40d2828a019cee8fdd5160f29f4667a0596bfe423960abd8f90e7bc1e2a',
    'candidate_report': '7a787d8b12d18dca3873797fac94b0882558b6cb0e3233990e273b95d8adceee',
    'candidate_source_report': '01b76b2575e62f6f04f4e1dc3f91d091f8d6949516f9e1550bc1e99860e06e38',
    'base_map': 'bd3753a02890269914e0caa181bf2da0e978124bf8f736fc017bf5263b715d78',
    'field': '71497028ba99db22e048a020a4e58799ccadd5dae0f5f9bac006ac270ab5b02b',
    'registration': 'b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e',
    'frame_proof': 'd9d2256d1c50d3ffbbcde6718c081bf4bd8e858973628ab81fae42debbeca0fb',
    'recovery': 'f2b04ad9898f2d8e45148f0f744492c04237b159b24beb0f337e0cac6ec7dcdc',
    'carrier087': '7f99dc0c1e4a390543b3c880a898330741f615d14fa30bb21d8e2fb1532d4a9b',
    'extension087': '0c5411ab97cce59e8fc6da2ef7bb7a8c80981619a3c75732e0acd078d42c8262',
    'native': '147c413bdfa84687f4afa3ce4e0d37576a853a3b4d5f421866e4faefeab4f174',
    'parent_tiss': '746c5f27f2d19929259fc14df124c0c12a0d796470a3d068af5e42d8476e9057',
}
STEPS = [0, 3807, 4767, 5023, 5599, 6207, 7423, 8000]
START = time.monotonic()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def log(path, event, **fields):
    row = {'event': event, 'elapsed_seconds': time.monotonic() - START, **fields}
    with Path(path).open('a') as f:
        f.write(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n')
    print(json.dumps(row, sort_keys=True), flush=True)


def link_exact(source: Path, dest: Path, expected_sha: str | None = None):
    source, dest = Path(source), Path(dest)
    require(source.is_file(), f'missing immutable source: {source}')
    observed = sha(source)
    if expected_sha is not None:
        require(observed == expected_sha, f'source hash mismatch: {source}')
    require(not dest.exists(), f'refusing to replace staged asset: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, dest)
    require(sha(dest) == observed, f'hardlink content changed: {dest}')


def load_file_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def helper_module(path: Path):
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    names = {'require', 'vertex_normals', 'load_tissue', 'row_data', 'rotate32', 'forward'}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in body} == names, 'registered-rest helper API changed')
    module = ast.Module(body=body, type_ignores=[])
    ns = {'np': np, 'struct': struct, 'Path': Path}
    exec(compile(module, str(path), 'exec'), ns)
    return ns


def stats(parent: np.ndarray, child: np.ndarray):
    delta = parent.astype(np.float64) - child.astype(np.float64)
    d = np.linalg.norm(delta, axis=1) * 1e6
    sq = np.zeros(len(delta), dtype='<f4')
    for k in range(3):
        sq = np.add(sq, np.multiply(delta[:, k].astype('<f4'), delta[:, k].astype('<f4'), dtype='<f4'), dtype='<f4')
    over_native = int(np.count_nonzero(sq > np.float32(4.0e-12)))
    return {
        'count': int(len(d)), 'max_um': float(d.max(initial=0.0)),
        'rms_um': float(np.sqrt(np.mean(d*d))), 'median_um': float(np.median(d)),
        'p95_um': float(np.quantile(d, 0.95)), 'over_2um': int(np.count_nonzero(d > 2.0)),
        'native_float32_gate_failures': over_native,
        'max_index': int(np.argmax(d)) if len(d) else None,
        'max_delta_um': (delta[int(np.argmax(d))] * 1e6).tolist() if len(d) else None,
    }


def exact_fraction(v):
    return v if isinstance(v, F) else F(v)


def decode_default_poses(rigid_path: Path, archive_path: Path):
    raw = rigid_path.read_bytes()
    (magic, pabi, eabi, source_count, body_count, joint_count, nq, nv,
     root, virtual, reserved, embedded) = struct.unpack_from('<8s10I32s', raw, 0)
    require(magic == b'NHRIGID2' and pabi == 1 and nq == nv + 1,
            'NHRIGID header changed')
    require(embedded.hex() == sha(archive_path), 'NHRIGID source archive identity mismatch')
    body_off = 224
    joint_off = body_off + 160 * body_count
    dof_off = joint_off + 144 * joint_count
    q_off = dof_off + 64 * nv
    q = np.frombuffer(raw, dtype='<f4', count=nq, offset=q_off).copy()
    v_off = q_off + 4 * nq
    map_off = v_off + 4 * nv
    source_to_core = np.frombuffer(raw, dtype='<u4', count=source_count, offset=map_off).copy()
    pose_off = map_off + 4 * source_count
    poses = {}
    for si, core in enumerate(source_to_core):
        x, y, z, qx, qy, qz, qw = struct.unpack_from('<7f', raw, pose_off + 28 * si)
        poses[int(core)] = (np.asarray([x, y, z], dtype='<f4'),
                            np.asarray([qx, qy, qz, qw], dtype='<f4'))
    require(pose_off + 28 * source_count == len(raw), 'NHRIGID pose table extent changed')
    return poses, {'nq': int(nq), 'nv': int(nv), 'source_body_count': int(source_count),
                   'engine_body_count': int(body_count), 'source_archive_sha256': embedded.hex(),
                   'default_q_sha256': hashlib.sha256(q.tobytes()).hexdigest()}


def main():
    require(not OUT.exists(), f'preserve prior attempts; refusing to overwrite {OUT}')
    for p in (PARENT_TISS, PARENT_MANIFEST, PARENT_RECEIPT, PARENT_DECL, PARENT_PREP,
              PARENT_LAUNCH, CAND, CAND_REPORT, CAND_SOURCE_REPORT, BASE_MAP,
              BASE_EXACT_REPORT, FIELD, FRAME_PROOF, REGISTRATION, RECOVERY,
              CARRIER_087, EXT_087, CARRIER_PREP_087, ENCODER, COMPOSER, FORWARD,
              RAW_SOURCE_TISS, RIGID, ARCHIVE, BIN):
        require(p.is_file(), f'required pinned input absent: {p}')
    for key, path in [('candidate', CAND), ('candidate_report', CAND_REPORT),
                      ('candidate_source_report', CAND_SOURCE_REPORT), ('base_map', BASE_MAP), ('field', FIELD),
                      ('registration', REGISTRATION), ('frame_proof', FRAME_PROOF),
                      ('recovery', RECOVERY), ('carrier087', CARRIER_087),
                      ('extension087', EXT_087), ('native', BIN), ('parent_tiss', PARENT_TISS)]:
        require(sha(path) == EXPECTED[key], f'expected pinned {key} input hash changed')

    parent_decl = json.loads(PARENT_DECL.read_text())
    parent_prep = json.loads(PARENT_PREP.read_text())
    parent_receipt = json.loads(PARENT_RECEIPT.read_text())
    carrier_prep_087 = json.loads(CARRIER_PREP_087.read_text())
    reg = json.loads(REGISTRATION.read_text())
    frame_proof = json.loads(FRAME_PROOF.read_text())
    recovery = json.loads(RECOVERY.read_text())
    require(parent_prep.get('status') == 'prepared_experimental_not_admitted',
            '007 parent is no longer the retained experimental source scene')
    require(parent_decl.get('capture_steps') == STEPS and parent_decl.get('accepted_steps') == 8000
            and parent_decl.get('execution_host') == 'numi', '007 native grid or host changed')
    require(sha(BIN) == EXPECTED['native'], 'tested native executable differs from 1353 build')
    require(reg.get('coordinate_system', {}).get('global_source_mm_to_myosim_world_m'),
            'recovered registration has no global transform')
    require(recovery.get('source_sha256') == EXPECTED['registration'] and recovery.get('verified') is True,
            'recovered authoring registration receipt no longer binds source')
    require(frame_proof.get('sha256') == EXPECTED['registration']
            and Path(frame_proof.get('authoring_registration_path', '')).resolve() == REGISTRATION.resolve(),
            '088 frame proof does not bind recovered registration')
    global_mm = reg.get('coordinate_system', {}).get('global_source_mm_to_myosim_world_m')
    require(isinstance(global_mm, list) and len(global_mm) == 4, 'recovered registration lacks its global affine matrix')
    global_m = [list(row) for row in global_mm]
    for i in range(3):
        for j in range(3):
            global_m[i][j] *= 1000.0
    proof_m = frame_proof.get('source_m_to_registered_world_m')
    require(isinstance(proof_m, list) and len(proof_m) == 3
            and max(abs(global_m[i][j] - proof_m[i][j]) for i in range(3) for j in range(4)) <= 1e-15,
            '088 registered-frame matrix differs from recovered authoring transform')
    require(carrier_prep_087.get('status') == 'prepared_load_only_not_anatomically_admitted',
            '087 source carrier is not a load-only preparation')
    def matching_pin(mapping, target):
        return next((digest for path, digest in mapping.items()
                     if Path(path).resolve() == Path(target).resolve()), None)
    require(matching_pin(carrier_prep_087.get('input_sha256', {}), ENCODER) == sha(ENCODER),
            'carrier encoder differs from the tested 087 serializer')
    require(matching_pin(carrier_prep_087.get('input_sha256', {}), COMPOSER) == sha(COMPOSER),
            'passive attachment composer differs from tested 087 serializer')
    require(carrier_prep_087.get('carrier_payload', {}).get('sha256') == sha(CARRIER_087)
            and carrier_prep_087.get('extension', {}).get('sha256') == sha(EXT_087),
            '087 carrier assets do not match preparation pins')

    OUT.mkdir()
    (OUT / 'baseline').mkdir()
    (OUT / 'row-patches-001').mkdir()
    progress = OUT / 'progress.jsonl'
    log(progress, 'preflight_started', candidate_sha256=sha(CAND), field_sha256=sha(FIELD))

    # Load the exact refined mesh, its 060 ancestry, and the registered route field.
    with CAND.open('rb') as f:
        candidate = pickle.load(f)
    with BASE_MAP.open('rb') as f:
        base_map = pickle.load(f)
    require(candidate.get('schema') == 'numi.human.left-fds132-fdp134-conforming-source-interface-geometry-083.v1',
            '085 geometry candidate schema changed')
    require(Path(candidate.get('input_source_parent_map_path', '')).resolve() == BASE_MAP.resolve()
            and candidate.get('input_source_parent_map_sha256') == sha(BASE_MAP),
            '085 geometry candidate has a different source-parent map')
    base_surfaces = base_map.get('surfaces', {})
    require(set(base_surfaces) == {'132', '134'}, '060 source-parent map rows changed')
    raw_source = base_map.get('parent_source_payloads', [{}])[0]
    require(Path(raw_source.get('path', '')).resolve() == RAW_SOURCE_TISS.resolve()
            and raw_source.get('sha256') == sha(RAW_SOURCE_TISS),
            '060 exact parent map no longer points to the pinned original 21c3 source payload')
    with np.load(FIELD, allow_pickle=False) as field:
        core_ids = np.asarray(field['core_body_indices'], dtype='<u4').copy()
        field_arrays = {}
        for sid in (132, 134):
            field_arrays[sid] = {
                'global_ids': np.asarray(field[f'stable{sid}_global_vertex_ids'], dtype='<u8').copy(),
                'positions': np.asarray(field[f'stable{sid}_rest_positions_m'], dtype='<f4').copy(),
                'faces': np.asarray(field[f'stable{sid}_faces'], dtype='<u4').copy(),
                'weights': np.asarray(field[f'stable{sid}_core_weights'], dtype='<f8').copy(),
            }
    require(len(core_ids) == 21 and len(set(map(int, core_ids))) == 21,
            '088 canonical core order changed')

    pc = load_file_module('passive_attachment_composition_009', COMPOSER)
    base_tissue = pc._read_nhtiss4(PARENT_TISS)
    h = helper_module(FORWARD)
    base_native = h['load_tissue'](PARENT_TISS)
    base_rows = {sid: h['row_data'](base_native, sid) for sid in (132, 134)}
    base_local_by_global = {}
    for sid in (132, 134):
        src = base_surfaces[str(sid)]
        gids = np.asarray(src['global_vertex_ids'], dtype='<u8')
        row = base_rows[sid]
        require(len(gids) == row['vc'] and np.array_equal(gids, field_arrays[sid]['global_ids'][:len(gids)]),
                f'007 base vertex/global-ID ordering changed row {sid}')
        require(np.array_equal(row['positions'], field_arrays[sid]['positions'][:len(gids)]),
                f'007 base source positions differ from 088 refined field row {sid}')
        base_local_by_global[sid] = {int(g): i for i, g in enumerate(gids)}

    refinement = candidate.get('refinement', {})
    midpoint_parent = {int(k): tuple(map(int, v)) for k, v in refinement.get('new_vertex_parent_global_ids', {}).items()}
    exact_global = candidate.get('global_vertices_exact_m', {})
    ancestry = {'schema': 'numi.human.forearm-shared-field-refined085-source-ancestry.v1',
                'status': 'verified_pinned_source_ancestry', 'candidate_path': str(CAND),
                'candidate_sha256': sha(CAND), 'candidate_report_path': str(CAND_REPORT),
                'candidate_report_sha256': sha(CAND_REPORT), 'source_parent_map_path': str(BASE_MAP),
                'source_parent_map_sha256': sha(BASE_MAP), 'source_parent_map_exact_check_path': str(BASE_EXACT_REPORT),
                'source_parent_map_exact_check_sha256': sha(BASE_EXACT_REPORT),
                'exact_midpoint_definition': 'Every new vertex is the exact rational midpoint of its two retained 060 global endpoint IDs; original global IDs retain the 060 exact 054 parent terms.',
                'rows': {}}
    row_data = {}
    for sid in (132, 134):
        surf = candidate['surfaces'][sid]
        ids = np.asarray(surf['global_vertex_ids'], dtype='<u8')
        points = np.asarray(surf['vertices_float32_m'], dtype='<f4')
        faces = np.asarray(surf['faces'], dtype='<u4')
        require(np.array_equal(ids, field_arrays[sid]['global_ids']), f'085/088 global-ID arrays differ row {sid}')
        require(np.array_equal(points, field_arrays[sid]['positions']), f'085/088 rest positions differ row {sid}')
        require(np.array_equal(faces, field_arrays[sid]['faces']), f'085/088 face arrays differ row {sid}')
        candidate_core_weights = surf.get('core_weights_float64')
        if candidate_core_weights is not None:
            require(candidate_core_weights.shape == field_arrays[sid]['weights'].shape
                    and np.array_equal(np.asarray(candidate_core_weights, dtype='<f8'), field_arrays[sid]['weights']),
                    f'085/088 binary64 core field differs row {sid}')
        require(np.isfinite(field_arrays[sid]['weights']).all()
                and np.all(field_arrays[sid]['weights'] >= 0)
                and np.max(np.abs(field_arrays[sid]['weights'].sum(axis=1) - 1.0)) <= 1e-12,
                f'088 field is not a nonnegative partition of unity row {sid}')
        require(len(surf['parent_060_face_indices']) == len(faces)
                and len(surf['parent_054_face_refs']) == len(faces),
                f'085 face ancestry coverage differs row {sid}')
        orig_ids = set(base_local_by_global[sid])
        new_ids = [int(g) for g in ids if int(g) not in orig_ids]
        require(all(g in midpoint_parent for g in new_ids), f'085 new vertex lacks exact midpoint ancestry row {sid}')
        for gid in new_ids:
            p0, p1 = midpoint_parent[gid]
            require(p0 in exact_global and p1 in exact_global and gid in exact_global,
                    f'085 exact midpoint coordinate record missing for global ID {gid}')
            expected_xyz = tuple((exact_fraction(a) + exact_fraction(b)) / 2
                                 for a, b in zip(exact_global[p0], exact_global[p1]))
            require(tuple(exact_fraction(x) for x in exact_global[gid]) == expected_xyz,
                    f'085 exact coordinate is not the declared midpoint at global ID {gid}')
        row_data[sid] = {'ids': ids, 'points': points, 'faces': faces,
                         'field_weights': field_arrays[sid]['weights'], 'surface': surf,
                         'base_row': base_rows[sid]}
        ancestry['rows'][str(sid)] = {
            'vertex_count': int(len(ids)), 'face_count': int(len(faces)),
            'retained_060_global_vertices': int(len(ids) - len(new_ids)),
            'new_exact_midpoint_vertices': int(len(new_ids)),
            'all_faces_have_parent_060_and_054_face_references': True,
            'parent_060_face_indices_count': int(len(surf['parent_060_face_indices'])),
            'parent_054_face_refs_count': int(len(surf['parent_054_face_refs'])),
        }
    ancestry_path = OUT / 'source-ancestry-proof.json'
    ancestry_path.write_text(json.dumps(ancestry, indent=2, sort_keys=True) + '\n')
    log(progress, 'candidate_and_field_identity_verified', row_counts={str(s): {'vertices': len(row_data[s]['ids']), 'faces': len(row_data[s]['faces'])} for s in (132, 134)})

    # Reconstruct each child NHTISS row by interpolating the existing source-bound four-lane load-time map.
    # The new shared route field, not these provisional lanes, owns native motion via the parent extension.
    patches = {}
    child_fields = {}
    for sid in (132, 134):
        d = row_data[sid]
        old = d['base_row']
        old_local = old['local']
        old_weights = old['weights']
        ids = d['ids']
        old_map = base_local_by_global[sid]
        loc = np.zeros((len(ids), 4), dtype='<u4')
        weights = np.zeros((len(ids), 4), dtype='<f4')
        other_sid = 134 if sid == 132 else 132
        current_keys = [old['binding_bytes'][i:i+36] for i in range(0, len(old['binding_bytes']), 36)]
        current_key_to_index = {key: i for i, key in enumerate(current_keys)}
        other = base_rows[other_sid]
        other_keys = [other['binding_bytes'][i:i+36] for i in range(0, len(other['binding_bytes']), 36)]
        other_key_to_current = {i: current_key_to_index[key] for i, key in enumerate(other_keys)
                                if key in current_key_to_index}
        other_global = base_local_by_global[other_sid]
        resolved = {}
        resolving = set()
        def resolve_global(gid):
            gid = int(gid)
            if gid in resolved:
                return resolved[gid]
            if gid in resolving:
                raise RuntimeError(f'cyclic 085 midpoint ancestry row {sid}/{gid}')
            if gid in old_map:
                j = old_map[gid]
                result = (old_local[j].copy(), old_weights[j].copy())
                resolved[gid] = result
                return result
            if gid not in midpoint_parent and gid in other_global:
                j = other_global[gid]
                loc_other = other['local'][j]
                w_other = other['weights'][j]
                mapped = np.zeros(4, dtype='<u4')
                for lane in range(4):
                    if w_other[lane] > 0:
                        source_index = int(loc_other[lane])
                        require(source_index in other_key_to_current,
                                f'cross-row midpoint support is absent from row {sid} binding table for global ID {gid}')
                        mapped[lane] = other_key_to_current[source_index]
                result = (mapped, w_other.copy())
                resolved[gid] = result
                return result
            require(gid in midpoint_parent, f'085 midpoint ancestry missing from source row {sid}: {gid}')
            resolving.add(gid)
            left, right = midpoint_parent[gid]
            left_local, left_weights = resolve_global(left)
            right_local, right_weights = resolve_global(right)
            accum = defaultdict(float)
            for lane in range(4):
                if left_weights[lane] > 0:
                    accum[int(left_local[lane])] += 0.5 * float(left_weights[lane])
                if right_weights[lane] > 0:
                    accum[int(right_local[lane])] += 0.5 * float(right_weights[lane])
            ranked = sorted(accum, key=lambda binding: (-accum[binding], binding))
            selected = ranked[:4]
            require(len(selected) == 4, f'four-lane midpoint interpolation lost support row {sid}/{gid}')
            total = sum(accum[b] for b in selected)
            out_local = np.zeros(4, dtype='<u4')
            out_weights = np.zeros(4, dtype='<f4')
            selected.sort()
            for lane, binding in enumerate(selected):
                out_local[lane] = binding
                out_weights[lane] = np.float32(accum[binding] / total)
            resolving.remove(gid)
            result = (out_local, out_weights)
            resolved[gid] = result
            return result
        for i, raw_gid in enumerate(ids):
            src_local, src_weights = resolve_global(int(raw_gid))
            loc[i] = src_local
            weights[i] = src_weights
        require(np.isfinite(weights).all() and np.all(weights >= 0)
                and np.max(np.abs(weights.sum(axis=1, dtype=np.float64) - 1.0)) < 1e-5,
                f'interpolated child weights are not a four-lane partition row {sid}')
        normals = h['vertex_normals'](d['points'], d['faces'])
        vertices6 = np.column_stack((d['points'], normals)).astype('<f4')
        patch = OUT / 'row-patches-001' / f'row-{sid}.npz'
        np.savez_compressed(patch, vertices6=vertices6, binding_indices=loc,
                            weights=weights, faces=d['faces'])
        patch_report = OUT / 'row-patches-001' / f'row-{sid}-report.json'
        patch_report.write_text(json.dumps({
            'schema': 'numi.human.forearm-refined085-row-patch.v1',
            'status': 'prepared_experimental_not_admitted', 'stable_id': sid,
            'candidate_path': str(CAND), 'candidate_sha256': sha(CAND),
            'route_field_path': str(FIELD), 'route_field_sha256': sha(FIELD),
            'row_vertex_count': int(len(ids)), 'face_count': int(len(d['faces'])),
            'normal_source': 'Area-weighted unit normals recomputed from this closed refined rest mesh; no posed capture normals.',
            'child_four_lane_rule': 'For original global vertices, retain the 007 source-bound four lanes byte-for-byte. For each exact midpoint, average endpoint binding contributions by the existing row-local binding identity, choose the four largest positive contributions with deterministic binding-index tie-break, renormalize, and sort selected lane indices. These lanes exist only for the native registered-rest child gate; the full 21-influence shared field owns runtime motion.',
            'geometry_ancestry': 'Candidate085 retains exact 060/054 face ancestry; new vertices are exact midpoints of candidate085 new_vertex_parent_global_ids endpoints.',
            'gpu_run_performed': False, 'anatomical_admission_claimed': False,
        }, indent=2, sort_keys=True) + '\n')
        patches[sid] = (patch, patch_report)
        child_fields[sid] = (loc, weights)

    # Compose the two child rows into the existing 007 NHTISS parent; the composer preserves the other 156 rows and binding tables.
    compose_proof = pc.compose(PARENT_TISS, OUT / 'assets',
        [(sid, patches[sid][0], patches[sid][1]) for sid in (132, 134)],
        reference_surface_rows=(132, 134))
    child_payload = OUT / 'assets' / PARENT_TISS.name
    child_manifest = child_payload.with_suffix('.manifest.json')
    composed_report = OUT / 'assets' / 'report.json'
    require(child_payload.is_file() and child_manifest.is_file() and composed_report.is_file(),
            'passive composer did not emit its payload, manifest, and proof')
    require(compose_proof.get('binding_table_byte_exact') is True
            and len(compose_proof.get('unchanged_row_vertex_bytes_and_local_faces', [])) == 156,
            'passive composer did not retain all 156 non-target source rows')
    child_native = h['load_tissue'](child_payload)
    child_rows = {sid: h['row_data'](child_native, sid) for sid in (132, 134)}
    for sid in (132, 134):
        row = child_rows[sid]
        d = row_data[sid]
        require(np.array_equal(row['positions'], d['points']) and np.array_equal(row['faces'], d['faces']),
                f'composed NHTISS roundtrip changed candidate geometry row {sid}')
        require(row['binding_bytes'] == base_rows[sid]['binding_bytes'],
                f'composed NHTISS roundtrip changed source binding table row {sid}')
        require(np.array_equal(row['local'], child_fields[sid][0])
                and np.array_equal(row['weights'], child_fields[sid][1]),
                f'composed NHTISS roundtrip changed provisional child binding fields row {sid}')
    log(progress, 'nhtiss_composition_roundtrip_verified', child_payload_sha256=sha(child_payload), unchanged_rows=156)

    # Encode candidate085 geometry and the binary64 field088 in the existing tested NHTISS carrier ABI.
    carrier_tissue_087 = pc._read_nhtiss4(CARRIER_087)
    carrier_native_087 = h['load_tissue'](CARRIER_087)
    carrier_base_row = h['row_data'](carrier_native_087, 132)
    binding_bytes = carrier_base_row['binding_bytes']
    core_from_bytes = [struct.unpack_from('<I8f', binding_bytes, offset)[0]
                       for offset in range(0, len(binding_bytes), 36)]
    require(core_from_bytes == list(map(int, core_ids)),
            '088 route core order differs from tested 087 canonical NHTISS carrier bindings')
    row134_carrier = h['row_data'](carrier_native_087, 134)
    require(row134_carrier['binding_bytes'] == binding_bytes,
            '087 paired carrier binding tables are not byte-identical')
    carrier_encoder = load_file_module('passive_tissue_parent_carriers_009', ENCODER)
    CarrierRow = carrier_encoder.CarrierRow
    encode_carriers = carrier_encoder.encode_carriers
    # The NHTISS header keeps the existing physical-source digest and registration fingerprint.
    (magic, abi, nr, nb, nv, ni, fingerprint, digest) = struct.unpack_from('<8s6I32s', carrier_tissue_087['raw'], 0)
    require(magic == b'NHTISS4\x00' and abi == 5 and fingerprint == base_tissue['fingerprint']
            and digest == base_tissue['source_digest'], '087 carrier and 007 child NHTISS registration/source identity differ')
    carrier_rows = []
    for sid in (132, 134):
        d = row_data[sid]
        vertices6 = np.column_stack((d['points'], h['vertex_normals'](d['points'], d['faces']))).astype('<f4')
        carrier_rows.append(CarrierRow(sid, vertices6, d['faces'], binding_bytes,
                                       d['field_weights'], layer=1))
    carrier_payload_bytes, extension_surfaces, encoder_summary = encode_carriers(
        carrier_rows, registration_fingerprint=int(fingerprint), source_digest=digest)
    carrier_payload = OUT / 'assets' / 'shared-field-carriers.nhtissue'
    carrier_payload.write_bytes(carrier_payload_bytes)
    ext = {
        'child_payload_sha256': sha(child_payload),
        'source_payloads': [{'path': 'assets/shared-field-carriers.nhtissue', 'sha256': sha(carrier_payload)}],
        'surfaces': extension_surfaces,
        'provenance': {
            'schema': 'numi.human.inferred-shared-route-anchor-field-extension-provenance.v1',
            'method': 'Inferred passive shared spatial reference field from the deduplicated union of authored MyoSim route controls; coefficients are the binary64 values pinned in route-field 088 and encoded through the existing native source-parent loader.',
            'registration_frame_proof_path': str(FRAME_PROOF), 'registration_frame_proof_sha256': sha(FRAME_PROOF),
            'recovered_authoring_registration_path': str(REGISTRATION), 'recovered_authoring_registration_sha256': sha(REGISTRATION),
            'child_geometry_candidate_path': str(CAND), 'child_geometry_candidate_sha256': sha(CAND),
            'source_parent_map_path': str(BASE_MAP), 'source_parent_map_sha256': sha(BASE_MAP),
            'not_measured_enthesis_or_active_muscle_mechanics': True,
            'source_carrier_vertices_are_runtime_render_geometry': False,
        },
    }
    extension_path = OUT / 'passive-tissue-parent-bindings.json'
    extension_path.write_text(json.dumps(ext, indent=2, sort_keys=True) + '\n')

    # Full carrier ABI roundtrip: reconstruct each field row from expanded positive-influence carrier vertices and terms.
    carrier_loaded = h['load_tissue'](carrier_payload)
    carrier_rows_loaded = {sid: h['row_data'](carrier_loaded, sid) for sid in (132, 134)}
    roundtrip_rows = {}
    for sid in (132, 134):
        d = row_data[sid]
        cr = carrier_rows_loaded[sid]
        counts = np.count_nonzero(d['field_weights'] > 0, axis=1)
        first = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)[:-1]))
        require(cr['bc'] == 21 and cr['vc'] == int(counts.sum()), f'carrier row ABI counts changed row {sid}')
        require(cr['binding_bytes'] == binding_bytes, f'carrier row binding bytes changed row {sid}')
        require(np.array_equal(cr['faces'], first[d['faces']]), f'carrier triangulation roundtrip changed row {sid}')
        # Verify the expanded carrier's per-vertex coordinates/normals/body slot and field terms exactly.
        source_vertex = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        source_body = np.concatenate([np.flatnonzero(row > 0) for row in d['field_weights']]).astype(np.int64)
        require(len(source_vertex) == cr['vc'] and len(source_body) == cr['vc'],
                f'carrier positive-influence expansion inventory changed row {sid}')
        require(np.array_equal(cr['positions'], d['points'][source_vertex]),
                f'carrier expanded source positions changed row {sid}')
        require(np.array_equal(cr['normals'], h['vertex_normals'](d['points'], d['faces'])[source_vertex]),
                f'carrier expanded source normals changed row {sid}')
        require(np.array_equal(cr['local'][:, 0], source_body.astype('<u4'))
                and np.all(cr['weights'][:, 0] == np.float32(1.0))
                and np.all(cr['weights'][:, 1:] == 0), f'carrier binding slot expansion changed row {sid}')
        terms = extension_surfaces[0 if sid == 132 else 1]['vertices']
        require(len(terms) == len(d['ids']), f'carrier parent map vertex order changed row {sid}')
        for vi, vertex_terms in enumerate(terms):
            expected_cols = np.flatnonzero(d['field_weights'][vi] > 0)
            require(len(vertex_terms) == len(expected_cols), f'carrier field support changed row {sid}/{vi}')
            for term, col in zip(vertex_terms, expected_cols):
                require(term[0] == 0 and term[1] == sid
                        and term[2] == int(first[vi] + np.searchsorted(np.flatnonzero(d['field_weights'][vi] > 0), col))
                        and float(term[3]) == float(d['field_weights'][vi, col]),
                        f'carrier parent term differs from pinned binary64 field row {sid}/{vi}')
        roundtrip_rows[str(sid)] = {'source_vertices': int(len(d['ids'])),
                                    'expanded_positive_influences': int(cr['vc']),
                                    'face_count': int(len(cr['faces'])),
                                    'all_positions_normals_faces_bindings_and_binary64_parent_terms_roundtrip': True}
    carrier_roundtrip_tissue = pc._read_nhtiss4(carrier_payload)
    require(carrier_roundtrip_tissue['fingerprint'] == base_tissue['fingerprint']
            and carrier_roundtrip_tissue['source_digest'] == base_tissue['source_digest'],
            'serialized carrier roundtrip changed physical source identity')
    encoder_report = {
        'schema': 'numi.human.forearm-shared-field-carrier-009-roundtrip.v1',
        'status': 'roundtrip_passed', 'encoder_summary': encoder_summary,
        'carrier_payload_path': str(carrier_payload), 'carrier_payload_sha256': sha(carrier_payload),
        'extension_path': str(extension_path), 'extension_sha256': sha(extension_path),
        'canonical_core_ids': list(map(int, core_ids)), 'canonical_binding_bytes_sha256': hashlib.sha256(binding_bytes).hexdigest(),
        'rows': roundtrip_rows, 'gpu_used': False, 'anatomical_admission_claimed': False,
    }
    encoder_report_path = OUT / 'carrier-roundtrip-report.json'
    encoder_report_path.write_text(json.dumps(encoder_report, indent=2, sort_keys=True) + '\n')
    log(progress, 'carrier_roundtrip_verified', carrier_sha256=sha(carrier_payload), extension_sha256=sha(extension_path))

    # The mandatory registered-rest source gate: child four-lane NHTISS versus the full 21-core native parent field.
    poses, pose_info = decode_default_poses(RIGID, ARCHIVE)
    parent_world = {}
    child_world = {}
    rest_rows = {}
    for sid in (132, 134):
        d = row_data[sid]
        n = len(d['ids'])
        parent_rest = np.zeros((n, 3), dtype='<f4')
        local = np.zeros((n, 4), dtype='<u4')
        one = np.zeros((n, 4), dtype='<f4')
        one[:, 0] = np.float32(1.0)
        carrier_row = {'fb': 0, 'bc': 21}
        for col in range(21):
            local[:, 0] = col
            world = h['forward'](d['points'], local, one, carrier_row, carrier_loaded, poses)
            coeff = np.asarray(d['field_weights'][:, col], dtype='<f4')
            for k in range(3):
                product = np.multiply(coeff, world[:, k], dtype='<f4')
                parent_rest[:, k] = np.add(parent_rest[:, k], product, dtype='<f4')
        child = child_rows[sid]
        child_rest = h['forward'](child['positions'], child['local'], child['weights'], child, child_native, poses)
        parent_world[sid] = parent_rest
        child_world[sid] = child_rest
        rest_rows[str(sid)] = stats(parent_rest, child_rest)
    rest_pass = all(v['max_um'] <= 2.0 and v['native_float32_gate_failures'] == 0
                    for v in rest_rows.values())
    rest_report = {
        'schema': 'numi.human.forearm-shared-route-field-009-registered-rest-preflight.v1',
        'status': 'passed' if rest_pass else 'failed',
        'candidate_geometry_path': str(CAND), 'candidate_geometry_sha256': sha(CAND),
        'route_field_path': str(FIELD), 'route_field_sha256': sha(FIELD),
        'parent_carrier_payload_sha256': sha(carrier_payload), 'extension_sha256': sha(extension_path),
        'child_payload_path': str(child_payload), 'child_payload_sha256': sha(child_payload),
        'registration_pose_source': {'path': str(RIGID), 'sha256': sha(RIGID), **pose_info},
        'child_binding_rule': 'Existing 007 source-bound child four lanes retained at original global vertices; at exact refinement midpoints, endpoint binding contributions are averaged by original row-local binding identity, then top-four positive contributions are normalized for the childRest load-time registration field. These lanes do not own runtime motion.',
        'parent_binding_rule': 'The 21 positive route-field coefficients from field088 are encoded binary64 in the existing passive_tissue_parent_bindings extension; source carrier terms use the native carrier rows with canonical shared body bindings and are accumulated in ascending core-column order with the native Float32 coefficient product.',
        'native_gate': 'Every vertex must satisfy parentRest-vs-childRest squared Float32 residual <= 4e-12 m^2, equivalent to the fixed 2 micrometre source-registration bound; this is a loader prerequisite, not an intersection test.',
        'rows': rest_rows, 'passes_fixed_2um_gate': rest_pass,
        'gpu_used': False, 'anatomical_admission_claimed': False,
    }
    rest_report_path = OUT / 'rest-registration-preflight.json'
    rest_report_path.write_text(json.dumps(rest_report, indent=2, sort_keys=True) + '\n')
    log(progress, 'registered_rest_gate', status=rest_report['status'], rows=rest_rows)
    require(rest_pass, '009 child four-lane interpolation fails the native 2 micrometre registered-rest gate; no launch declaration was created')

    # Save provenance and replace only the two parent-extension surfaces in the cloned receipt.
    link_exact(CARRIER_087, OUT / 'assets' / 'source-carrier-087.nhtissue', EXPECTED['carrier087'])
    common = parent_receipt.get('provenance', {}).get('cardiac_geometry_binding', {}).get('common_field')
    require(isinstance(common, dict), '007 receipt has no common cardiac field declaration')
    for owner in ('map', 'polynomials', 'domain_boxes'):
        record = common.get(owner)
        require(isinstance(record, dict) and isinstance(record.get('path'), str)
                and isinstance(record.get('sha256'), str), f'007 common cardiac field lacks {owner}')
        relative = Path(record['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,
                f'common cardiac path must remain receipt-relative: {owner}')
        link_exact(PARENT / relative, OUT / relative, record['sha256'])

    receipt = json.loads(PARENT_RECEIPT.read_text())
    receipt['provenance']['native_muscle_surfaces'].update(
        payload_path=str(child_payload), sha256=sha(child_payload),
        manifest_path=str(child_manifest), manifest_sha256=sha(child_manifest))
    receipt['passive_tissue_parent_bindings'] = ext
    receipt['experimental_refined085_geometry_ancestry_009'] = {
        'status': 'prepared_experimental_not_admitted',
        'candidate_path': str(CAND), 'candidate_sha256': sha(CAND),
        'candidate_report_path': str(CAND_REPORT), 'candidate_report_sha256': sha(CAND_REPORT),
        'source_parent_map_path': str(BASE_MAP), 'source_parent_map_sha256': sha(BASE_MAP),
        'ancestry_proof_path': str(ancestry_path), 'ancestry_proof_sha256': sha(ancestry_path),
        'rest_registration_preflight_path': str(rest_report_path), 'rest_registration_preflight_sha256': sha(rest_report_path),
        'child_manifest_composition_report_path': str(composed_report), 'child_manifest_composition_report_sha256': sha(composed_report),
        'refinement_threshold_mm': float(candidate['threshold_mm']),
        'source_faces_and_global_vertex_ids_preserved_with_exact_midpoint_refinement': True,
        'all_active_physical_routes_and_non_target_tissue_rows_unchanged': True,
        'qualification_boundary': '085 refined geometry and 088 shared passive route-anchor field are experimental. Native GPU geometry and whole-pose child/union audits remain pending; no anatomical admission is claimed.',
    }
    historical = receipt.get('experimental_left_fds_fdp_054_source_parent_native_diagnostic')
    if isinstance(historical, dict):
        historical['historical_only'] = True
        historical['status'] = 'historical_experiment_not_admitted_superseded_by_refined085_shared_field_009'
        historical['superseded_by'] = str(OUT)
    receipt['inferred_shared_route_anchor_field_native_diagnostic_008'] = {
        'status': 'historical_experimental_not_admitted_superseded_by_refined085_009',
        'native008_scene_path': str(A / 'forearm-shared-field-native-diagnostic-008'),
        'native008_audit_path': str(A / 'forearm-shared-field-native-diagnostic-008/native-audit-001/report.json'),
        'native008_union_audit_path': str(A / 'forearm-shared-field-native-diagnostic-008/native-union-audit-001/report.json'),
        'reason_for_successor': 'Native008 used the unrefined060 forearm mesh and showed child/union failures after Float32 deformation; the 009 preparation uses refined085 geometry with the corrected registered route field 088.',
        'superseded_by': str(OUT),
    }
    receipt['inferred_shared_route_anchor_field_native_diagnostic_009'] = {
        'status': 'prepared_experimental_not_admitted',
        'carrier_payload_path': str(carrier_payload), 'carrier_payload_sha256': sha(carrier_payload),
        'parent_extension_path': str(extension_path), 'parent_extension_sha256': sha(extension_path),
        'field_path': str(FIELD), 'field_sha256': sha(FIELD),
        'registration_frame_proof_path': str(FRAME_PROOF), 'registration_frame_proof_sha256': sha(FRAME_PROOF),
        'ancestry_proof_path': str(ancestry_path), 'ancestry_proof_sha256': sha(ancestry_path),
        'carrier_roundtrip_report_path': str(encoder_report_path), 'carrier_roundtrip_report_sha256': sha(encoder_report_path),
        'registered_rest_preflight_path': str(rest_report_path), 'registered_rest_preflight_sha256': sha(rest_report_path),
        'method': 'Inferred passive shared spatial reference field from the deduplicated union of authored MyoSim route controls. The field preserves physical route/control identities; no measured enthesis or active muscle mechanics are asserted.',
        'child_geometry': 'Conforming refined085 mesh with exact 060/054 face ancestry and exact midpoint vertex ancestry. Rest normals recomputed from closed refined rest faces.',
        'child_four_lane_binding_semantics': 'Source-bound NHTISS four-lane values are used only by the native registered-rest gate. The 21-influence passive parent extension owns the experimental runtime reference deformation.',
        'runtime_carrier_geometry_is_not_rendered': True,
        'capture_steps': STEPS, 'gpu_run_performed': False, 'anatomical_admission_claimed': False,
    }
    receipt['qualification'] = (
        'Prepared experimental native diagnostic for refined085 forearm geometry under an inferred passive shared route-anchor field. '
        'Source ancestry, carrier ABI roundtrip, and the native registered-rest load gate are checked; actual GPU geometry and union checks remain pending. '
        'Physical routes, activation, force, physiology, and all non-target rows retain their existing owners. No anatomical admission is claimed.'
    )
    receipt_path = OUT / 'resting-anatomy-receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')

    # Rebuild the accepted 007 invocation with only output/receipt paths changed; this agent does not launch it.
    source_argv = list(parent_decl['argv'])
    module_index = source_argv.index('numilab_human.resting_run')
    argv_tail = source_argv[module_index + 1:]
    for flag, value in (
        ('--output', str(OUT / 'baseline/native-run')),
        ('--anatomy-receipt', str(receipt_path)),
        ('--build', str(B)), ('--lab', str(W)),
    ):
        argv_tail[argv_tail.index(flag) + 1] = value
    pythonpath_value = next((x.split('=', 1)[1] for x in source_argv[2:module_index-2]
                             if x.startswith('PYTHONPATH=')), None)
    require(pythonpath_value is not None, '007 run declaration has no runtime PYTHONPATH')
    sys.path.insert(0, pythonpath_value)
    from numilab_human import resting_run as rr
    parser = argparse.ArgumentParser()
    rr.add_arguments(parser)
    args = parser.parse_args(argv_tail)
    native_argv, assets = rr.command(args)
    env = dict(x.split('=', 1) for x in source_argv[2:module_index-2])
    env.update(NUMI_BUILD_DIR=str(B), NUMI_LAB_ROOT=str(W),
               NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT=str(OUT / 'baseline/native-run/common-field-failure.json'),
               NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=','.join(map(str, STEPS)))
    env['DYLD_LIBRARY_PATH'] = ':'.join(str(Path(p).resolve()) for p in env['DYLD_LIBRARY_PATH'].split(':'))
    runtime_rr = Path(pythonpath_value) / 'numilab_human/resting_run.py'
    require(sha(runtime_rr) == sha(Path(rr.__file__)), 'runtime resting_run differs from accepted 007 owner CLI')
    command_argv = ['/usr/bin/env', '-i', *[k + '=' + v for k, v in sorted(env.items())],
                    '/Users/n/numi-human-prep-venv-20261005/bin/python3.13',
                    '-m', 'numilab_human.resting_run', *argv_tail]
    require(native_argv and Path(native_argv[0]).is_file() and sha(Path(native_argv[0])) == EXPECTED['native'],
            'resting_run CLI selected an unexpected native executable')

    pins = {}
    def pin(path: Path, expected_sha: str | None = None):
        path = Path(path).resolve()
        require(path.is_file(), f'immutable input is absent: {path}')
        observed = sha(path)
        if expected_sha is not None:
            require(observed == expected_sha, f'immutable input hash mismatch: {path}')
        if str(path) in pins:
            require(pins[str(path)] == observed, f'conflicting duplicate input pin: {path}')
        pins[str(path)] = observed
    pinned_sources = [Path(__file__), CAND, CAND_REPORT, CAND_SOURCE_REPORT, BASE_MAP, BASE_EXACT_REPORT,
                      FIELD, FRAME_PROOF, REGISTRATION, RECOVERY, PARENT_TISS, PARENT_MANIFEST,
                      PARENT_RECEIPT, PARENT_DECL, PARENT_PREP, PARENT_LAUNCH,
                      CARRIER_087, EXT_087, CARRIER_PREP_087, ENCODER, COMPOSER, FORWARD,
                      RAW_SOURCE_TISS, RIGID, ARCHIVE, BIN, receipt_path, child_payload, child_manifest,
                      composed_report, ancestry_path, carrier_payload, extension_path,
                      encoder_report_path, rest_report_path]
    for path in pinned_sources:
        pin(path)
    for path, digest in parent_decl.get('immutable_assets', {}).items():
        pin(Path(path), digest)
    for path, digest in assets.items():
        pin(Path(path), digest)
    require(any(Path(path).name == 'numi-human-native' and digest == EXPECTED['native']
                for path, digest in assets.items()), 'CLI assets do not include pinned native binary')

    launcher = OUT / 'launch_arm.py'
    shutil.copyfile(PARENT_LAUNCH, launcher)
    launcher_hash = sha(launcher)
    declaration = {
        'schema': 'numi.human.native-smoke.run-declaration.v1', 'seconds': 16.0,
        'accepted_steps': 8000, 'capture_steps': STEPS, 'immutable_assets': pins,
        'argv': command_argv, 'launch_adapter': {'path': str(launcher), 'sha256': launcher_hash},
        'owner_cli_preview': {'native_argv': native_argv, 'asset_sha256': assets,
                              'recorded_invocation_environment': rr.invocation_environment(env)},
        'execution_host': 'numi', 'hardware': 'Apple M6,12 GPU cores,16 GiB,macOS 27.0',
        'qualification': 'Experimental refined085 shared-route-anchor field native diagnostic only. Native run and actual-GPU geometry checks remain pending; do not infer anatomical admission or physical qualification.',
    }
    declaration_path = OUT / 'baseline/run-declaration.json'
    declaration_path.write_text(json.dumps(declaration, indent=2, sort_keys=True) + '\n')
    guard = {'launcher_sha256': launcher_hash,
             'arms': {'baseline': {'run_declaration_sha256': sha(declaration_path),
                                   'native_output': str(OUT / 'baseline/native-run')}}}
    guard_path = OUT / 'launch-guard.json'
    guard_path.write_text(json.dumps(guard, indent=2, sort_keys=True) + '\n')
    command_path = OUT / 'launch-command.txt'
    command_path.write_text(f'/Users/n/numi-human-prep-venv-20261005/bin/python3.13 {launcher} baseline\n')
    prep = {
        'schema': 'numi.human.forearm-shared-route-field-native-diagnostic-009-preparation.v1',
        'status': 'prepared_experimental_not_admitted', 'parent_scene_path': str(PARENT),
        'parent_scene_preparation_sha256': sha(PARENT_PREP),
        'child_payload_path': str(child_payload), 'child_payload_sha256': sha(child_payload),
        'carrier_payload_path': str(carrier_payload), 'carrier_payload_sha256': sha(carrier_payload),
        'parent_extension_path': str(extension_path), 'parent_extension_sha256': sha(extension_path),
        'candidate_path': str(CAND), 'candidate_sha256': sha(CAND),
        'route_field_path': str(FIELD), 'route_field_sha256': sha(FIELD),
        'source_ancestry_path': str(ancestry_path), 'source_ancestry_sha256': sha(ancestry_path),
        'carrier_roundtrip_report_path': str(encoder_report_path), 'carrier_roundtrip_report_sha256': sha(encoder_report_path),
        'registered_rest_preflight_path': str(rest_report_path), 'registered_rest_preflight_sha256': sha(rest_report_path),
        'registered_rest_preflight_status': rest_report['status'],
        'capture_steps': STEPS, 'gpu_run_performed': False, 'anatomical_admission_claimed': False,
        'run_declaration_path': str(declaration_path), 'run_declaration_sha256': sha(declaration_path),
        'launch_guard_path': str(guard_path), 'launch_guard_sha256': sha(guard_path),
        'launch_command': command_path.read_text().strip(), 'launcher_sha256': launcher_hash,
        'immutable_input_count': len(pins), 'input_sha256': pins,
    }
    prep_path = OUT / 'preparation.json'
    prep_path.write_text(json.dumps(prep, indent=2, sort_keys=True) + '\n')
    log(progress, 'preparation_complete', status=prep['status'], run_declaration_sha256=sha(declaration_path),
        immutable_input_count=len(pins), gpu_run_performed=False)
    print(json.dumps({'status': prep['status'], 'output': str(OUT),
                      'run_declaration': str(declaration_path), 'run_declaration_sha256': sha(declaration_path),
                      'child_payload_sha256': sha(child_payload), 'carrier_payload_sha256': sha(carrier_payload),
                      'extension_sha256': sha(extension_path), 'rest_preflight': rest_rows,
                      'launch_command': prep['launch_command'], 'gpu_run_performed': False}, indent=2, sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            failure = {'schema': 'numi.human.forearm-shared-route-field-native-diagnostic-009-preparation-failure.v1',
                       'status': 'retained_failure', 'error_type': type(exc).__name__, 'error': str(exc),
                       'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - START}
            (OUT / 'preparation-failure.json').write_text(json.dumps(failure, indent=2, sort_keys=True) + '\n')
        raise
