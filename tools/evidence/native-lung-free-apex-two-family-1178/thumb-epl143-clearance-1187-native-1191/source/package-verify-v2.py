#!/usr/bin/env python3
"""Fail-closed verifier for the EPL143 candidate and CPU-only owner preview."""
from __future__ import annotations
import copy, hashlib, json, subprocess
from pathlib import Path

E = Path('/Users/n/numi-human-resting-evidence-20261005')
ROOT = E / 'native-skin-epl143-clearance-1187'
PKG = ROOT / 'package-003'
OUT = PKG / 'attempt-010/admission-output'
BASE = E / 'final-native-scene-preflight-936/skin-927-lung-1178-viewer-018-v015-attempt3'
OWNER = Path('/Users/n/numi-human-free-apex-two-family-1178')
EXPECTED_REV = 'f13b57547f25d601431e65a4e090b61f2b6672c7'
EXPECTED_CANDIDATE = 'b2d235e32c1c7d7f753eb83d1e8e9d045a1fd62be9c5c6da65dfde8844e2622b'
EXPECTED_CORRECTION = 'd919c72c24e8016bd166a41b1e9e67fdec51479ee12b4e7abcd17a191e954930'
EXPECTED_BASELINE_REPORT = '9df248cf4cdcad92dc28a7ad5b0336c365e93ecf712e5811b8d392a11b708e69'
EXPECTED_TREATMENT_REPORT = '8fe877107f54ba38e35311edc4ac3d5d45d8908347d6f85183405cbee82b95df'
EXPECTED_REVIEW = 'c030ff4f9421cfef096b6fba6662d3dde28f355af8516935e4262c7b95999868'
EXPECTED_PREVIEW = '65fe92af6f652935812a3416175ea2dfacbd0fc989cec1ca82a1815795aba335'
BASE_STEPS = ['47519', '49151', '51903', '54047', '55647', '152191', '154143', '155000']
TREATMENT_STEPS = ['47519', '49151', '51903', '54047', '55647', '152447', '154367', '155000']
SIDECARS = {'common-cardiac-map-f32.bin', 'common-cardiac-volumes-f32.bin', 'common-cardiac-domains-f32.bin'}

def need(ok, msg):
    if not ok: raise RuntimeError(msg)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()

def bound(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}

def check_pose_report(report, expected_steps):
    need(report['status'] == 'candidate_geometry_only_pass', 'pose report is not a pass')
    need(report['qualification'] == 'inferred NHSKIN reference geometry only; no native viewer or contact/physics qualification', 'pose scope changed')
    need(set(report['pose_results']) == set(expected_steps) and len(report['pose_results']) == len(expected_steps), 'pose set changed')
    need(set(report['step_inputs']) == set(expected_steps), 'pose input set changed')
    need(report['gates']['EPL143_clear_at_all_8_poses'] is True, 'EPL143 gate failed')
    need(report['gates']['skin_self_clear_at_all_8_poses'] is True, 'skin self gate failed')
    need(report['gates']['no_new_pairs_against_all_859_targets_at_all_8_poses'] is True, 'new target pairs present')
    need(report['gates']['all_859_target_surfaces_scanned_against_changed_faces'] is True, 'target coverage incomplete')
    need(report['gates']['native_run_or_physics_admission'] == 'not performed', 'scope accidentally widened')
    for step in expected_steps:
        pose = report['pose_results'][step]
        need(pose['changed_vertex_count'] == 33 and pose['changed_face_count'] == 90, f'changed support differs at {step}')
        need(pose['skin_self']['unallowed_pair_count'] == 0, f'skin self contacts at {step}')
        need(pose['changed_face_geometry']['degenerate_count'] == 0 and pose['changed_face_geometry']['reversed_count'] == 0, f'bad changed-face geometry at {step}')
        scan = pose['all_859_target_scan']
        need(scan['target_surface_count'] == 859 and scan['candidate_changed_face_pairs'] == 0, f'changed skin face intersections at {step}')
        need(scan['introduced_pair_count'] == 0, f'new skin-target pairs at {step}')
        need(scan['untouched_face_pair_results_carried_only_from_complete_pinned_1172_witnesses'] if 'untouched_face_pair_results_carried_only_from_complete_pinned_1172_witnesses' in scan else '1172' in scan['scope'], f'untouched pair lineage absent at {step}')
    return {step: report['step_inputs'][step] for step in expected_steps}

def main():
    candidate = ROOT/'attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin'
    correction = ROOT/'attempt-006/epl143-position-correction.nhskin'
    geom_report = ROOT/'attempt-006/report.json'
    treatment_report_path = ROOT/'attempt-007-treatment/report.json'
    review = ROOT/'package-002/result/review.json'
    recipe = PKG/'recipe/prepare_epl143_candidate.py'
    admission_script = PKG/'attempt-010/prepare_admission.py'
    preview_path = OUT/'admission-preview.json'
    command_path = OUT/'native-owner-command.json'
    candidate_manifest = OUT/'candidate-skin-geometry-registration.manifest.json'
    composed_receipt = OUT/'composed-anatomy/resting-anatomy-receipt.json'
    composed_manifest = OUT/'composed-anatomy/resting-anatomy-manifest.json'
    derived_scene = OUT/'resting-supine-scene.manifest.json'
    for p in (candidate, correction, geom_report, treatment_report_path, review, recipe, admission_script, preview_path, command_path, candidate_manifest, composed_receipt, composed_manifest, derived_scene):
        need(p.is_file(), f'missing package input {p}')
    need(sha(candidate) == EXPECTED_CANDIDATE, 'candidate NHSKIN hash mismatch')
    need(sha(correction) == EXPECTED_CORRECTION, 'correction NHSKIN hash mismatch')
    need(sha(geom_report) == EXPECTED_BASELINE_REPORT, 'baseline geometry report changed')
    need(sha(treatment_report_path) == EXPECTED_TREATMENT_REPORT, 'treatment geometry report changed')
    need(sha(review) == EXPECTED_REVIEW, 'package-002 independent review changed')
    need(sha(preview_path) == EXPECTED_PREVIEW, 'admission preview changed')
    need(sha(recipe) == '93f3b3465323871149ba12524d789e56794b263ef54923eb1db4e52310269f4f', 'copied candidate recipe differs from executed driver')
    base_report = json.loads(geom_report.read_text())
    treatment_report = json.loads(treatment_report_path.read_text())
    baseline_steps = check_pose_report(base_report, BASE_STEPS)
    treatment_steps = check_pose_report(treatment_report, TREATMENT_STEPS)
    need(base_report['candidate_assets']['composed_candidate_sha256'] == EXPECTED_CANDIDATE, 'baseline candidate identity mismatch')
    need(treatment_report['candidate_assets']['composed_candidate_sha256'] == EXPECTED_CANDIDATE, 'treatment candidate identity mismatch')
    need(treatment_report['fixed_treatment_candidate']['source_candidate_sha256'] == EXPECTED_CANDIDATE, 'treatment candidate bytes differ')
    field = base_report['source_frame_and_field']
    need(field['selected_engineering_trial_amplitude_mm'] == 0.5 and field['maximum_rest_world_displacement_cap_mm'] == 2.0, 'final correction parameter changed')
    need(field['active_taper_vertices_before_float32_rounding'] == 33 and field['changed_source_positions'] == 33, 'correction support changed')
    need(field['changed_source_vertex_ids'] == json.loads(review.read_text())['candidate']['changed_positions'], 'changed vertex list mismatch')
    need(base_report['candidate_assets']['changed_normal_vertex_count'] == 207, 'normal support count mismatch')
    need(base_report['candidate_assets']['full_86_weight_matrix_identical'] and base_report['candidate_assets']['all_86_bindings_identical'] and base_report['candidate_assets']['triangle_indices_and_order_identical'], 'geometry invariants failed')
    need(base_report['candidate_assets']['support_payload_edited'] is False, 'support payload was edited')
    review_doc = json.loads(review.read_text())
    support = review_doc['support_seed_check']
    need(support['all_changed_vertices_stay_in_seed_region'] is True and support['changed_seed_vertex_overlap'] is False, 'static support seed check changed')
    need({r['nearest_seed_contact_index'] for r in support['changed_vertices_static_nearest_seed_assignment']} == {15}, 'static nearest-seed region changed')
    preview = json.loads(preview_path.read_text())
    need(preview['status'] == 'owner_composition_and_native_command_preparation_passed' and preview['native_launch_executed'] is False, 'admission preview invalid')
    need(preview['owner']['revision'] == EXPECTED_REV and preview['owner']['functional_bindings_unchanged'] and preview['owner']['NHA_unchanged'], 'current owner composition check failed')
    need(preview['native_owner_preview']['argv_matches_parent_after_declared_path_substitutions'] is True, 'argv differs beyond approved paths')
    changed = preview['native_owner_preview']['candidate_vs_parent_input_changes']
    expected_changed = {str(BASE/'resting-scene/resting-supine-scene.manifest.json'), str(BASE/'anatomy/resting-anatomy-receipt.json'), str(Path(json.loads((BASE/'anatomy/resting-anatomy-receipt.json').read_text())['mass_geometry_accounting']['skin_payload_path']).resolve())}
    need(set(changed) == expected_changed, 'owner input changes exceed candidate scene/receipt/skin')
    base_receipt = json.loads((BASE/'anatomy/resting-anatomy-receipt.json').read_text())
    candidate_receipt = json.loads(composed_receipt.read_text())
    need(candidate_receipt['payload'] == base_receipt['payload'] and candidate_receipt['functional_bindings'] == base_receipt['functional_bindings'], 'NHA or functional bindings changed')
    need(candidate_receipt['mass_geometry_accounting']['skin_payload_sha256'] == EXPECTED_CANDIDATE, 'receipt does not bind candidate')
    candidate_manifest_doc = json.loads(candidate_manifest.read_text())
    need(candidate_manifest_doc['output_payload']['sha256'] == EXPECTED_CANDIDATE, 'candidate manifest payload mismatch')
    need(candidate_manifest_doc['qualification']['physical_or_collision_use'] == 'not admitted', 'candidate manifest admits physics/contact')
    base_scene = json.loads((BASE/'resting-scene/resting-supine-scene.manifest.json').read_text())
    scene = json.loads(derived_scene.read_text())
    scene_without_skin_delta = copy.deepcopy(scene)
    scene_without_skin_delta['source']['skin'] = copy.deepcopy(base_scene['source']['skin'])
    need(scene_without_skin_delta == base_scene, 'derived scene changed outside skin path/hash')
    need(scene['source']['skin']['sha256'] == EXPECTED_CANDIDATE, 'derived scene skin hash mismatch')
    base_assets = {str(Path(k).resolve()): v for k,v in json.loads((BASE/'native-run/invocation.json').read_text())['asset_sha256'].items()}
    candidate_assets = {str(Path(k).resolve()): v for k,v in preview['native_owner_preview']['asset_sha256'].items()}
    candidate_dir = composed_receipt.parent
    relocated = {}
    for name in sorted(SIDECARS):
        old = next((k for k in base_assets if Path(k).name == name and Path(k).parent == (BASE/'anatomy').resolve()), None)
        new = next((k for k in candidate_assets if Path(k).name == name and Path(k).parent == candidate_dir.resolve()), None)
        need(old is not None and new is not None, f'missing sidecar mapping {name}')
        need(base_assets[old] == candidate_assets[new], f'relocated sidecar bytes changed {name}')
        need(sha(old) == base_assets[old] and sha(new) == candidate_assets[new], f'relocated sidecar file hash mismatch {name}')
        relocated[name] = {'parent': {'path': old, 'sha256': base_assets[old]}, 'candidate': {'path': new, 'sha256': candidate_assets[new]}, 'bytes_identical_by_sha256': True}
    need(not (OUT/'native-run-not-executed').exists(), 'native output path was created')
    owner_files = preview['owner']['source_files']
    source_root = Path(preview['owner']['source_snapshot']).resolve()
    source_manifest = Path(preview['owner']['source_snapshot_manifest']['path']).resolve()
    need(sha(source_manifest) == preview['owner']['source_snapshot_manifest']['sha256'], 'pinned owner source snapshot manifest changed')
    snapshot_doc = json.loads(source_manifest.read_text())
    need(snapshot_doc['revision'] == EXPECTED_REV and snapshot_doc['file_count'] == len(snapshot_doc['files']), 'owner source snapshot metadata changed')
    for rel,digest in snapshot_doc['files'].items():
        need(sha(source_root.parent/rel) == digest, f'owner source snapshot changed: {rel}')
    for file in owner_files.values():
        need(sha(file['path']) == file['sha256'], f'preview owner module snapshot changed: {file["path"]}')
    worktree_status = preview['owner']['post_snapshot_worktree_status']
    need(subprocess.run(['git','-C',str(OWNER),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip() == EXPECTED_REV, 'owner repository revision drifted')
    need((OWNER/'src/numilab_human/common_atlas_skin_geometry_registration.py').is_file(), 'geometry owner helper missing')
    need(sha(OWNER/'src/numilab_human/common_atlas_skin_geometry_registration.py') == '61857a0edfc098a941f2923113ebbc2ecafbb3c3aafee3a7ff4a83ba28ffd1c0', 'geometry composition owner differs from candidate recipe')
    pose_evidence = {'baseline_64_iteration': baseline_steps, 'treatment_64_iteration': treatment_steps}
    manifest = {
        'schema': 'numi.human.epl143-clearance-package.v2',
        'supersedes': {'path': str(PKG/'package-manifest.json'), 'sha256': sha(PKG/'package-manifest.json'), 'status': 'superseded because v1 mislabeled the source snapshot as a clean-worktree precondition'},
        'status': 'owner_composition_and_1178_native_command_preview_passed_pending_fresh_native_pose_validation',
        'qualification': 'inferred registered reference-skin geometry candidate; offline saved-pose differential and CPU-only 1178 owner assembly preview only; not native or physics admitted',
        'candidate': bound(candidate), 'position_correction': bound(correction),
        'generation_recipe': {'original_driver': bound(recipe), 'original_executed_driver_path': base_report['driver']['path'], 'sha256_matches_execution': True, 'bounded_screen_report': bound(geom_report), 'treatment_screen_report': bound(treatment_report_path), 'selected_amplitude_mm': 0.5, 'maximum_world_displacement_cap_mm': 2.0, 'source_field': 'quintic smoothstep complement over geodesic distance, outward per-vertex source NHSKIN normal; radius 2 x median seed incident edge; exact 86-weight LBS directional gain used for amplitude conversion'},
        'candidate_geometry': {'target': base_report['target_identity'], 'changed_vertex_ids': field['changed_source_vertex_ids'], 'changed_incident_face_rows': field['changed_incident_face_rows'], 'changed_vertices': len(field['changed_source_vertex_ids']), 'changed_faces': len(field['changed_incident_face_rows']), 'recomputed_normals': base_report['candidate_assets']['changed_normal_vertex_ids'], 'recomputed_normal_count': base_report['candidate_assets']['changed_normal_vertex_count'], 'weights_bindings_topology_source_archive_preserved': True},
        'support_seed_check': {'report_sha256': sha(review), 'all_changed_vertices_stay_in_seed_region': support['all_changed_vertices_stay_in_seed_region'], 'nearest_seed_contact_indices': [15], 'changed_seed_vertex_overlap': support['changed_seed_vertex_overlap'], 'active_contact_caveat': support['limitation']},
        'offline_pose_evidence': {'scope':'prior 1159/1170 baseline and treatment saved poses; these are not captures from the 1178 parent scene and do not qualify 1178 dynamic clearance', 'screen_NHA_sha256':base_report['source_identity']['NHA_sha256'], 'screen_baseline_declaration':base_report['source_identity']['baseline_declaration_path'], 'screen_baseline_declaration_sha256':base_report['source_identity']['baseline_declaration_sha256'], **pose_evidence},
        'attempt_history': {
            'candidate_generation_failed_attempts': review_doc['failed_attempts'],
            'history': bound(PKG/'attempt-history.json'),
            'admission_attempt_001': bound(PKG/'attempt-001/failure.json'),
            'admission_attempt_003': bound(PKG/'attempt-003/failure.json'),
            'admission_attempt_004': bound(PKG/'attempt-004/failure.json'),
            'admission_attempt_005': bound(PKG/'attempt-005/failure.json'),
            'admission_attempt_005_diagnosis': bound(PKG/'attempt-005/diagnose_argv.py'),
            'admission_attempt_006': bound(PKG/'attempt-006/failure.json'),
            'admission_attempt_007': bound(PKG/'attempt-007/failure.json'),
            'admission_attempt_009': bound(PKG/'attempt-009/failure.json')
        },
        'admission': {'parent_scene_label':'skin-927-lung-1178-viewer-018-v015-attempt3', 'preview': bound(preview_path), 'verifier': bound(PKG/'verify_package_v2.py'), 'native_owner_command': bound(command_path), 'candidate_manifest': bound(candidate_manifest), 'composed_receipt': bound(composed_receipt), 'composed_manifest': bound(composed_manifest), 'derived_scene_manifest': bound(derived_scene), 'base_scene_manifest': bound(BASE/'resting-scene/resting-supine-scene.manifest.json'), 'base_receipt': bound(BASE/'anatomy/resting-anatomy-receipt.json'), 'sidecar_relocations': relocated, 'only_declared_owner_input_changes': sorted(changed), 'argv_sha256': hashlib.sha256(json.dumps(preview['native_owner_preview']['argv'],separators=(',',':')).encode()).hexdigest(), 'native_launch_executed': False},
        'owner': {'repository': str(OWNER), 'revision': EXPECTED_REV, 'source_snapshot_manifest': bound(source_manifest), 'source_files': owner_files, 'composition_owner_sha256': sha(OWNER/'src/numilab_human/common_atlas_skin_geometry_registration.py'), 'source_snapshot_from_committed_revision': True, 'current_checkout_was_clean_after_snapshot': not bool(worktree_status), 'preview_worktree_status_after_source_snapshot': worktree_status, 'note': 'The preview imports byte-verified f13b575 source-snapshot files from the archived source tree. The later checkout status recorded by the preview was dirty in resting_run.py and tests/test_resting_run.py; no clean-worktree precondition was required or passed. A future native launch must bind the exact source bytes it actually uses.'},
        'source_pins': {'927_skin_manifest': bound(E/'native-common-skin-multipose-clearance-candidate-927/asset-candidate-001/common-atlas-skin-geometry-registration.manifest.json'), '927_skin_payload': bound(E/'native-common-skin-multipose-clearance-candidate-927/asset-candidate-001/bodyparts3d-myosim-skinned-shell.nhskin'), 'parent_1178_NHA_sha256': json.loads((BASE/'anatomy/resting-anatomy-receipt.json').read_text())['payload']['sha256'], 'parent_1178_scene': bound(BASE/'resting-scene/resting-supine-scene.manifest.json'), 'parent_1178_anatomy_receipt': bound(BASE/'anatomy/resting-anatomy-receipt.json'), 'NHCNT_support_payload': bound(BASE/'resting-scene/myosim-fullbody-resting-bed-support.nhcnt')},
        'limits': ['The 0.5 mm field is an inferred engineering reference-geometry correction, not measured anatomy or measured skin thickness.', 'All 16 accepted saved poses screened are baseline/treatment pose snapshots; there is no continuous-time guarantee.', 'Dynamic support minimum, selected point, selector transitions and force Jacobian for the candidate were not recomputed.', 'No native viewer process, physical contact run, or admission to the collision/contact owner has occurred.', 'Untouched face-pair results are inherited only from the pinned complete 1172 audit; candidate changed faces were rechecked against all 859 targets and skin self geometry.'],
        'recomputation_note': 'This verifier rehashes the 46 MB candidate/correction, owner source modules and small manifests. The 16 MRVPACK inputs total about 4.57 GB; their exact hashes remain embedded in the two source-pinned scan reports and are not reread here.'
    }
    out = PKG/'package-manifest-v2.json'
    need(not out.exists(), f'refuse overwrite {out}')
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    result = {'schema':'numi.human.epl143-clearance-package-verification.v2','status':'pass','package_manifest':bound(out),'candidate_sha256':EXPECTED_CANDIDATE,'baseline_pose_count':len(baseline_steps),'treatment_pose_count':len(treatment_steps),'changed_owner_input_count':len(changed),'relocated_sidecar_count':len(relocated),'native_launch_executed':False}
    verification = PKG/'verification-v2.json'
    need(not verification.exists(), f'refuse overwrite {verification}')
    verification.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
