#!/usr/bin/env python3
"""Bounded read-only exact audit of native-022 accepted native skin mesh."""
from __future__ import annotations
import gc, hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path('/Users/n/numi-human-retained-delivery-20261009')
A = ROOT / 'anatomy-completion-1276'
N = A / 'back412-skin-native-022-001-60s'
PUB = Path('/Users/n/numi-human-left-pq-publication-1344/src')
INV = Path('/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv')
OUT = N / 'skin-native-audit-001'
STEPS = [0, 3807, 4767, 7999, 14975, 20991, 26975, 30000]
CURRENT_SKIN_SHA = 'b01028f163bde0d46c9605d730f2e104713a1c490be3c718e77ff9cebb865710'
CURRENT_NHA_SHA = '1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241'
CURRENT_TISSUE_SHA = '3aa1dcf1458f6c7cda617a63ae4b4ed37a3280128da0b3dd3f1af571a0ce56d4'
EXPECTED_CA_SHA = 'fccc7f29d421c9ad8e7da40d61d16cc8633244a16829dc7a7cd9d83ff109cfae'
EXPECTED_CI_SHA = '05ba7856fc373f55a1270eb5f3ef887a4eac493ad9fb4590a618caee952d2a32'

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load(path: Path):
    return json.loads(path.read_text())

def main():
    sys.path.insert(0, str(PUB))
    from numilab_human import common_atlas_skin_clearance as ca
    from numilab_human import cardiac_cavity_intersections as ci
    assert sha(Path(ca.__file__)) == EXPECTED_CA_SHA
    assert sha(Path(ci.__file__)) == EXPECTED_CI_SHA

    run = load(N / 'baseline/native-run/run-metadata.json')
    decl = load(N / 'baseline/run-declaration.json')
    exec_receipt = load(N / 'baseline/execution.json')
    require_inputs = decl['immutable_assets']
    assert len(require_inputs) == 182
    assert not exec_receipt['changed_inputs'] and exec_receipt['returncode'] == 0
    assert run['host'] == 'numi.local' and run['exit_code'] == 0
    assets_by_name = {Path(p).name: digest for p, digest in run['asset_sha256'].items()}
    assert assets_by_name['bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'] == CURRENT_TISSUE_SHA
    assert assets_by_name['resting-thorax.nhanatomy'] == CURRENT_NHA_SHA
    assert assets_by_name['bodyparts3d-myosim-skinned-shell.nhskin'] == CURRENT_SKIN_SHA
    assert all(Path(p).is_file() and sha(Path(p)) == digest for p, digest in run['asset_sha256'].items())
    target_keys, _, _, _ = ca._load_target_inventory(INV)
    target_keys = set(target_keys) | {(51005, sid) for sid in range(151, 165)}
    selected = {
        key for key in target_keys
        if (key[0] == 51005 and 151 <= key[1] <= 164)
        or (key[0] == 51010 and 381 <= key[1] <= 397)
        or (key[0] == 51011 and 464 <= key[1] <= 524)
    }
    assert len(selected) == 92
    assert sum(k[0] == 51005 for k in selected) == 14
    assert sum(k[0] == 51010 for k in selected) == 17
    assert sum(k[0] == 51011 for k in selected) == 61

    source_receipt = load(Path(decl['argv'][decl['argv'].index('--anatomy-receipt') + 1]))
    mass = source_receipt['mass_geometry_accounting']
    assert mass['skin_boundary_edge_count'] == 115 and mass['skin_boundary_loop_count'] == 2
    assert sorted(len(x['source_vertex_ids']) for x in mass['skin_boundary_repair']['retained_anatomical_interfaces']) == [55, 60]
    assert all('ocular' in x['interface'] for x in mass['skin_boundary_repair']['retained_anatomical_interfaces'])
    assert source_receipt['payload']['sha256'] == CURRENT_NHA_SHA

    results = []
    start_all = time.monotonic()
    for step in STEPS:
        t0 = time.monotonic()
        pack = N / f'baseline/native-run/accepted-geometry/step-{step}.mrvpack'
        receipt_path = pack.with_suffix('.receipt.json')
        rc = load(receipt_path)
        assert rc['accepted_step'] == step and rc['physical_endpoint'] == 'accepted'
        assert rc['surface_audit_endpoint'] == 'passed'
        assert rc['common_field_source_anatomy_payload_sha256'] == CURRENT_NHA_SHA
        assert len(rc['accepted_body_poses']) == 157
        assert sha(pack) == rc['pack_file_sha256']
        positions, surfaces, pack_meta = ca._pack_surfaces(pack, target_keys)
        assert set(target_keys).issubset(surfaces)
        skin = surfaces[(51007, 1)]
        skin_faces = np.asarray(skin['faces'], dtype=np.int64)
        skin_used = np.unique(skin_faces)
        skin_local_faces = np.searchsorted(skin_used, skin_faces)
        skin_xyz = np.asarray(positions[skin_used], dtype='<f4')
        assert len(skin_faces) == 109211 and len(skin_xyz) == 54663
        skin_records = ca._exact_surface_records(skin_xyz, skin_local_faces)
        self_audit = ci._audit_pair(skin_records, skin_records, same_surface=True)
        target_faces = {k: np.asarray(surfaces[k]['faces'], dtype=np.int64) for k in sorted(selected)}
        target_audits = ca._target_intersection_audit(skin_records, target_faces, positions)
        nonzero = {k: v for k, v in target_audits.items() if v['count']}
        result = {
            'step': step,
            'time_s': rc['accepted_time_s'],
            'accepted_pose_count': len(rc['accepted_body_poses']),
            'pack_sha256': rc['pack_file_sha256'],
            'skin_surface_key': '51007:1',
            'skin_face_count': int(len(skin_faces)),
            'skin_referenced_vertex_count': int(len(skin_xyz)),
            'skin_exact_self_pairs': int(self_audit['count']),
            'skin_degenerate_face_rows': int(len(self_audit.get('degenerate_face_rows', []))),
            'source_skin_topology': {'boundary_edges': mass['skin_boundary_edge_count'], 'boundary_loops': mass['skin_boundary_loop_count'], 'nonmanifold_edges': mass['skin_nonmanifold_edge_count'], 'orientation_errors': mass['skin_orientation_error_count']},
            'target_count': len(target_audits),
            'target_category_counts': {'new_muscle_ids_151_164': 14, 'ocular_ids_381_397': 17, 'major_vessel_ids_464_524': 61},
            'target_exact_pair_total': int(sum(v['count'] for v in target_audits.values())),
            'target_nonzero_pairs_by_surface': nonzero,
            'elapsed_seconds': time.monotonic() - t0,
            'pack_counts': pack_meta,
        }
        results.append(result)
        OUT.mkdir(parents=True, exist_ok=True)
        progress = {
            'schema': 'numi.human.native-022-skin-repair-interface-progress.v1',
            'scope': 'read-only exact CPU audit of skin self intersections plus all 14 newly appended muscle rows, 17 ocular surfaces, and 61 major-vessel surfaces at the eight current native-022 accepted captures; no mesh/runtime modifications',
            'source_modules': {'common_atlas_skin_clearance.py': sha(Path(ca.__file__)), 'cardiac_cavity_intersections.py': sha(Path(ci.__file__))},
            'current_assets': {'nhskin_sha256': CURRENT_SKIN_SHA, 'nhanatomy_sha256': CURRENT_NHA_SHA, 'nhtissue_sha256': CURRENT_TISSUE_SHA},
            'completed_capture_count': len(results), 'captures': results,
        }
        (OUT / 'native-skin-repair-022-progress.json').write_text(json.dumps(progress, indent=2, sort_keys=True) + '\n')
        print(json.dumps({'step': step, 'self_pairs': result['skin_exact_self_pairs'], 'degenerate_faces': result['skin_degenerate_face_rows'], 'target_pairs': result['target_exact_pair_total'], 'nonzero_target_surfaces': len(nonzero), 'elapsed_s': result['elapsed_seconds']}, sort_keys=True), flush=True)
        del positions, surfaces, skin_records, target_audits, skin_xyz
        gc.collect()

    final = load(OUT / 'native-skin-repair-022-progress.json')
    final['complete'] = True
    final['all_current_native_captures_exact_skin_self_zero'] = all(r['skin_exact_self_pairs'] == 0 and r['skin_degenerate_face_rows'] == 0 for r in results)
    final['all_target_pair_counts_zero'] = all(r['target_exact_pair_total'] == 0 for r in results)
    final['candidate_offline_pair_table_reused'] = False
    final['candidate_table_reuse_reason'] = 'native-022 skin world geometry is audited from each current native MRVPACK; the native-022 accepted pose records differ from the 30-surface candidate cohort after step 0.'
    final['elapsed_seconds_total'] = time.monotonic() - start_all
    # Recheck all native-022 assets after this read-only audit.
    assert all(Path(p).is_file() and sha(Path(p)) == digest for p, digest in run['asset_sha256'].items())
    final['inputs_unchanged'] = True
    (OUT / 'native-skin-repair-022-audit.json').write_text(json.dumps(final, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: final[k] for k in ('complete','all_current_native_captures_exact_skin_self_zero','all_target_pair_counts_zero','elapsed_seconds_total','inputs_unchanged')}, sort_keys=True), flush=True)

if __name__ == '__main__':
    main()
