"""Independent exact replay of the D309 quotient composed onto the I399 Fat cells."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, math
import numpy as np

H = Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
BASE = H/'fat614-yy-volume-replay-004/run-014-current-target-rebound-v3/fat-D307-I377-YY-components-current-target-rebound.npz'
CURRENT = H/'paired-I-interface-399-root/iteration002/batch/fat-coupled-I-quotient-volume-proposal.npz'
DSEED = H/'d-shortedge-quotient-385-run006/identity-seed-D.npz'
DTYPED = H/'d-shortedge-quotient-385-run006/typed-rebind-003/candidate-D-typed-current.npz'
D697 = H/'d-shortedge-quotient-385-run006/actual-697-D-load-003.json'
DRAW = H/'d-shortedge-quotient-385-run006/surface-run001/candidate-D-short-edge-collapse-raw.npz'
LEDGER = H/'d-shortedge-quotient-385-run006/surface-run001/accepted-operations.jsonl'
PREFLIGHT = H/'d-shortedge-quotient-385-run006/preflight.json'
SIDECAR = H/'d-shortedge-quotient-385-run006/accepted-volume-star-map-001.npz'
CANDIDATE = H/'combined-D309-I399-volume-404-root/fat-D309-I399-full-domain-volume-proposal.npz'
PRODUCER_REPORT = H/'combined-D309-I399-volume-404-root/report.json'
EXPECTED = {
    str(BASE): '8915c3c7a58f8622d0568da5006bf4b7b0f2832f802b65e29bd144e95f11001d',
    str(CURRENT): '40ecc1079ead7b78e86fb5eac24c5cd6de80bb35e41004fae5130995fc1157b1',
    str(DSEED): 'cf27a4d3485724b33f884061fcc94529d8d0f8f980aa46ae496d713dcd59c94c',
    str(DTYPED): '9a5ca486d7b2025a7a2140798155df79c208871703f1eaf75171761b388ba7eb',
    str(D697): '0b3202abee7afa7044cad9588b8ffe03a12a08fe5954246eabbb97468ee241ba',
    str(DRAW): 'f8220c4a021acca0d15eea29c02ce7b51f7102954e242e8403ebf8adf9f971b7',
    str(LEDGER): 'f4ece3d374e877dd26d20ec8df9313af160e4c718a59bb2da446871ef5a4c958',
    str(PREFLIGHT): 'ce18c8b27ab4fbea4f956f13910cb8d2bc7fe5be08f2a7ae85f970b296d8d0a5',
    str(SIDECAR): 'e4ebb7f243f1969312ab826082c7d1ebc68681bd6607ba5a1369cbde8cbe844e',
}
PAT = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int64)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def parity(face):
    a, b, c = map(int, face)
    return 1 if ((a > b) + (a > c) + (b > c)) % 2 == 0 else -1

def face_identity(faces):
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    s = np.sort(faces, axis=1)
    sign = np.fromiter((parity(f) for f in faces), dtype=np.int8, count=len(faces))
    order = np.lexsort((sign, s[:, 2], s[:, 1], s[:, 0]))
    return np.column_stack((s[order], sign[order]))

def volume_boundary(tets):
    f = np.asarray(tets, dtype=np.int64)[:, PAT].reshape(-1, 3)
    sf = np.sort(f, axis=1)
    order = np.lexsort((sf[:, 2], sf[:, 1], sf[:, 0]))
    ss = sf[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(ss[1:] != ss[:-1], axis=1))]
    counts = np.diff(np.r_[starts, len(ss)])
    if np.any(counts > 2):
        raise AssertionError('nonmanifold tet face')
    signs = np.fromiter((parity(f[i]) for i in order), dtype=np.int8, count=len(order))
    sums = np.add.reduceat(signs, starts)
    if np.any(sums[counts == 2] != 0):
        raise AssertionError('inconsistent internal face winding')
    rows = order[starts[counts == 1]]
    return f[rows]

def f32_lattice_int(value):
    n, d = float(np.float32(value)).as_integer_ratio()
    exponent = d.bit_length() - 1
    assert d == (1 << exponent) and exponent <= 149
    return n << (149 - exponent)

def integer_point_cache(points, node_rows):
    rows = np.unique(np.asarray(node_rows, dtype=np.int64))
    return {int(i): tuple(f32_lattice_int(x) for x in points[i]) for i in rows}

def det6(ip, tet):
    a, b, c, d = (ip[int(i)] for i in tet)
    u = tuple(b[i] - a[i] for i in range(3))
    v = tuple(c[i] - a[i] for i in range(3))
    w = tuple(d[i] - a[i] for i in range(3))
    return (u[0] * (v[1]*w[2] - v[2]*w[1])
          - u[1] * (v[0]*w[2] - v[2]*w[0])
          + u[2] * (v[0]*w[1] - v[1]*w[0]))

def double_metrics(points, tets):
    q = points[tets].astype(np.float64)
    u = q[:, 1:] - q[:, :1]
    det = np.einsum('ij,ij->i', u[:, 0], np.cross(u[:, 1], u[:, 2]))
    edges = [np.linalg.norm(q[:, i] - q[:, j], axis=1) for i in range(4) for j in range(i+1, 4)]
    maxedge = np.max(np.stack(edges), axis=0)
    normalized = det / np.maximum(maxedge**3, np.finfo(float).tiny)
    return det, normalized

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate', type=Path, default=CANDIDATE)
    ap.add_argument('--producer-report', type=Path, default=PRODUCER_REPORT)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    start = __import__('time').monotonic()
    inputs = {}
    for path, expected in EXPECTED.items():
        actual = sha(path)
        assert actual == expected, (path, actual, expected)
        inputs[path] = actual
    report = json.loads(args.producer_report.read_text())
    assert report['candidate_sha256'] == sha(args.candidate)
    assert report['producer_sha256'] == sha(H/'compose_D309_I399_volume_404_root.py')
    preflight = json.loads(PREFLIGHT.read_text())
    base, current, seed, raw, side = (load(x) for x in (BASE, CURRENT, DSEED, DRAW, SIDECAR))
    input_d_path = Path(preflight['source_pins']['current_D']['path'])
    assert sha(input_d_path) == preflight['source_pins']['current_D']['sha256'] == report['source_D_sha256']
    source_d = load(input_d_path)
    typed_d = load(DTYPED)
    d697 = json.loads(D697.read_text())
    assert d697['status'] == 'PASS_ACTUAL_697_217_TYPED_D_LOAD'
    assert d697['typed_source_sha256'] == EXPECTED[str(DTYPED)]
    assert d697['adapter_sha256'] == '74b3cc83bcc428873afb1c75b679f43d9562469f2a1706cf4d2f79fbebe2fe77'
    assert typed_d['source_current_candidate_sha256'].item() == EXPECTED[str(DRAW)]
    assert np.array_equal(typed_d['points_world_m_f32'].view(np.uint8), seed['points_world_m_f32'].view(np.uint8))
    assert np.array_equal(typed_d['faces'], raw['faces'])
    out = load(args.candidate)
    with LEDGER.open() as f:
        ops = [json.loads(line) for line in f if line.strip()]
    assert len(ops) == 309
    p = current['points_world_m']
    assert p.dtype == np.float32
    assert np.array_equal(p.view(np.uint8), base['points_world_m'].view(np.uint8))
    assert np.array_equal(out['points_world_m'].view(np.uint8), p.view(np.uint8))
    assert np.array_equal(out['points_world_m_f32'].view(np.uint8), p.view(np.uint8))
    assert len(np.unique(p, axis=0)) == len(p)
    assert np.all(current['material_indices'] == 16)

    # Locate only the preflighted 1,415 old cells by unordered exact vertex key and material.
    old_star_rows = np.asarray(side['D_operation_volume_star_current_cell_rows'], dtype=np.int64)
    offsets = np.asarray(side['D_operation_volume_star_offsets'], dtype=np.int64)
    assert len(old_star_rows) == 1415 and len(np.unique(old_star_rows)) == 1415
    old_t = base['tetrahedra']
    old_mat = base['material_indices']
    wanted = {}
    for r in old_star_rows:
        key = (int(old_mat[r]), *sorted(map(int, old_t[r])))
        assert key not in wanted
        wanted[key] = int(r)
    base_to_current = np.full(len(old_t), -1, dtype=np.int64)
    for r, (tet, material) in enumerate(zip(current['tetrahedra'], current['material_indices'])):
        key = (int(material), *sorted(map(int, tet)))
        source = wanted.get(key)
        if source is not None:
            assert base_to_current[source] < 0
            base_to_current[source] = r
    assert np.all(base_to_current[old_star_rows] >= 0)
    joined = base_to_current[old_star_rows]
    assert np.array_equal(current['tetrahedra'][joined], old_t[old_star_rows])
    assert np.array_equal(current['material_indices'][joined], old_mat[old_star_rows])

    # Independently reconstruct node and cell quotient without calling producer try_op.
    seed_points = seed.get('points_world_m_f32', seed['points_world_m']).astype(np.float32)
    assert np.array_equal(seed_points.view(np.uint8), seed['points_world_m_f32'].astype(np.float32).view(np.uint8))
    assert np.array_equal(seed['faces'], source_d['faces'])
    assert np.array_equal(seed_points.view(np.uint8), source_d['points_world_m_f32'].view(np.uint8))
    d_point_to_volume = np.asarray(base['fat_current_target_rebind_D_point_to_volume_rows'], dtype=np.int64)
    assert len(d_point_to_volume) == len(seed_points) == len(side['D_point_to_current_rebound_volume_point_row'])
    assert np.array_equal(d_point_to_volume, side['D_point_to_current_rebound_volume_point_row'])
    assert np.array_equal(p[d_point_to_volume].view(np.uint8), seed_points.view(np.uint8))
    node_q = np.arange(len(p), dtype=np.int64)
    surface_q = np.arange(len(seed_points), dtype=np.int64)
    active = np.ones(len(current['tetrahedra']), dtype=bool)
    expected_t = current['tetrahedra'].copy()
    touched = np.zeros(len(active), dtype=bool)
    exact_delta = 0
    per_op = []
    drops = set()
    keeps = set()
    for oi, op in enumerate(ops):
        drop_d = int(op['collapsed_input_point_row'])
        keep_d = int(op['retained_input_point_row'])
        drop_v = int(side['D_operation_volume_drop_point_rows'][oi])
        keep_v = int(side['D_operation_volume_keep_point_rows'][oi])
        assert drop_d == int(side['D_operation_collapsed_point_rows'][oi])
        assert keep_d == int(side['D_operation_retained_point_rows'][oi])
        assert drop_v == int(d_point_to_volume[drop_d]) and keep_v == int(d_point_to_volume[keep_d])
        assert node_q[drop_v] == drop_v and node_q[keep_v] == keep_v
        assert surface_q[drop_d] == drop_d and surface_q[keep_d] == keep_d
        assert drop_v not in drops and keep_v not in keeps and drop_v not in keeps and keep_v not in drops
        drops.add(drop_v); keeps.add(keep_v)
        node_q[drop_v] = keep_v
        surface_q[drop_d] = keep_d
        a, b = map(int, offsets[oi:oi+2])
        rows8915 = old_star_rows[a:b]
        rows399 = base_to_current[rows8915]
        assert len(rows399) > 0 and not np.any(touched[rows399])
        touched[rows399] = True
        old_star = expected_t[rows399].copy()
        new_star = node_q[old_star]
        sorted_new = np.sort(new_star, axis=1)
        deleted = np.any(sorted_new[:, 1:] == sorted_new[:, :-1], axis=1)
        nodes = np.unique(np.r_[old_star.ravel(), new_star.ravel()])
        ip = integer_point_cache(p, nodes)
        old_det = [det6(ip, q) for q in old_star]
        new_det = [0 if dead else det6(ip, q) for q, dead in zip(new_star, deleted)]
        assert min(old_det) > 0
        surviving_det = [d for d, dead in zip(new_det, deleted) if not dead]
        if surviving_det:
            assert min(surviving_det) > 0
        delta = sum(new_det) - sum(old_det)
        exact_delta += delta
        expected_t[rows399] = new_star
        active[rows399[deleted]] = False
        per_op.append({'operation': oi, 'I399_cell_rows': rows399.tolist(), 'source_8915_cell_rows': rows8915.tolist(), 'deleted_cell_count': int(deleted.sum()), 'exact_det6_delta': str(delta)})
    expected_rows = np.flatnonzero(active)
    expected_t = expected_t[active]
    expected_mat = current['material_indices'][active]
    assert np.array_equal(out['retained_input_cell_rows'], expected_rows)
    assert np.array_equal(out['material_indices'], expected_mat)
    assert np.array_equal(out['tetrahedra'], expected_t)
    assert np.array_equal(out['D309_previous_retained_input_cell_rows'], current['retained_input_cell_rows'])
    assert np.array_equal(out['D309_input_node_quotient'], node_q)
    assert np.array_equal(out['source_D_vertex_quotient'], surface_q)
    assert np.array_equal(out['D309_source_8915_affected_cell_rows'], old_star_rows)
    assert np.array_equal(out['D309_source_I399_affected_cell_rows'], joined)
    assert np.all(expected_mat == 16)
    assert len(np.unique(np.sort(expected_t, axis=1), axis=0)) == len(expected_t)

    # Exact signed boundary replay: candidate tet boundary equals the mapped I399 boundary.
    old_boundary = volume_boundary(current['tetrahedra'])
    new_boundary = volume_boundary(out['tetrahedra'])
    mapped = node_q[old_boundary]
    mapped = mapped[~np.any(np.diff(np.sort(mapped, axis=1), axis=1) == 0, axis=1)]
    assert np.array_equal(face_identity(new_boundary), face_identity(mapped))
    assert np.array_equal(face_identity(out['boundary_faces']), face_identity(new_boundary))
    d_faces = surface_q[seed['faces']]
    d_faces = d_faces[~np.any(np.diff(np.sort(d_faces, axis=1), axis=1) == 0, axis=1)]
    assert np.array_equal(face_identity(d_faces), face_identity(raw['faces']))

    before_det, before_q = double_metrics(p, current['tetrahedra'])
    after_det, after_q = double_metrics(p, out['tetrahedra'])
    assert np.all(before_det > 0) and np.all(after_det > 0)
    before_bad = int(np.count_nonzero((before_det / 6 <= 1e-18) | (before_q < 1e-5)))
    after_bad = int(np.count_nonzero((after_det / 6 <= 1e-18) | (after_q < 1e-5)))
    assert before_bad == report['bad_before'] and after_bad == report['bad_after']
    assert exact_delta == int(report['exact_F32_det6_volume_delta'])
    assert report['physical_admission'] is False and report['native_steps'] == 0 and report['mass_mapping'] == 'NOT_RUN'
    assert 'complete typed current target' in report['limitation']

    result = {
        'schema': 'independent.numi.D309-I399-volume-transfer-replay.v1',
        'status': 'PASS_D309_CELL_QUOTIENT_AND_SIGNED_BOUNDARY_REPLAY',
        'inputs_sha256': inputs,
        'candidate': str(args.candidate),
        'candidate_sha256': sha(args.candidate),
        'producer_report_sha256': sha(args.producer_report),
        'producer_source_sha256': sha(H/'compose_D309_I399_volume_404_root.py'),
        'independent_checks': {
            'all_309_operation_endpoint_maps_match_sidecar': True,
            'all_1415_affected_8915_cell_keys_join_uniquely_to_I399': True,
            'all_unaffected_I399_cells_and_materials_bitwise_preserved': True,
            'output_tets_exactly_reconstructed_from_I399': True,
            'all_output_material_labels_preserved': True,
            'all_F32_double_determinants_positive': True,
            'all_changed_cell_integer_lattice_determinants_positive': True,
            'output_signed_boundary_equals_mapped_I399_boundary': True,
            'output_boundary_array_matches_tet_boundary': True,
            'D_raw_faces_equal_source_surface_quotient': True,
            'D_typed_face_and_point_arrays_equal_raw_candidate': True,
            'prequotient_D_input_equals_identity_seed_geometry': True,
            'actual_697_loader_receipt_pinned_and_passed': True,
            'full_point_prefix_bitwise_unchanged': True,
            'native_quality_failures_before': before_bad,
            'native_quality_failures_after': after_bad,
            'exact_F32_det6_volume_delta': str(exact_delta),
            'physical_admission': False,
            'native_steps': 0,
            'mass_mapping': 'NOT_RUN',
            'current_typed_target_rebind': 'PENDING; producer retains old I377/D185 target metadata',
        },
        'operation_replay': per_op,
        'elapsed_s': __import__('time').monotonic() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k!='operation_replay'}, indent=2), flush=True)

if __name__ == '__main__':
    main()
