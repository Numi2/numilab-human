import json,hashlib,math,shutil
from pathlib import Path
E=Path('/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187')
OUT=E/'native-1191-differential-final-003'
OLD=E/'native-1191-differential-full-001'
PAR=E
STEPS=[0,4991,5375,5759,6111,6495,7743,10000]
SCRIPT=Path('/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/epl143_native_diff_v2.py')
EXPECTED_SCRIPT='602651baa6862b98b76146a68dc64e69cf286522eda56d0ad0bc7065c8348243'
AGG=Path('/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/epl143_aggregate.py')
ID=Path('/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/geometry-identity-v2/non_skin_identity-v2.json')
CAND=Path('/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/native-run')
BASE=Path('/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/skin-927-lung-1178-viewer-018-v015-attempt3/native-run')
BASE_SUMMARY=Path('/Users/n/numi-human-resting-evidence-20261005/final-native-scene-preflight-936/skin-927-lung-1178-viewer-018-v015-attempt3/lung-geometry-937-full/summary.json')
CAND_SKIN=Path('/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin')
BASE_SKIN=Path('/Users/n/numi-human-resting-evidence-20261005/native-common-skin-multipose-clearance-candidate-927/asset-candidate-001/bodyparts3d-myosim-skinned-shell.nhskin')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o): Path(p).write_text(json.dumps(o,indent=2,sort_keys=True,allow_nan=False)+'\n')
assert not OUT.exists()
assert sha(SCRIPT)==EXPECTED_SCRIPT
identity=json.loads(ID.read_text()); assert identity['status']=='pass_exact_state_and_nonskin_geometry_identity'
steps=[]; result_paths={}
for step in STEPS:
    if step in (0,4991,5375,5759): p=OLD/f'step-{step}/result.json'
    else: p=PAR/f'native-1191-differential-parallel-002-step-{step}/step-{step}/result.json'
    d=json.loads(p.read_text()); assert d['step']==step
    assert d['full_candidate_skin_face_count']==109211 and d['candidate_changed_face_count']==90
    assert d['changed_star_target_surfaces_scanned']==859 and d['target_inventory_count']==859
    assert d['skin_self_unallowed_intersection_count']==0 and d['changed_star_target_unallowed_intersection_count']==0
    assert d['candidate_accepted_body_state_sha256']==d['baseline_accepted_body_state_sha256']
    assert d['candidate_accepted_respiration_state_sha256']==d['baseline_accepted_respiration_state_sha256']
    assert d['candidate_accepted_time_s']==d['baseline_accepted_time_s']
    targetp=p.parent/'target-results.json'
    assert sha(targetp)==d['target_results_sha256']
    targets=json.loads(targetp.read_text()); assert len(targets)==859 and all(x['unallowed_intersections']==0 for x in targets)
    selfp=p.parent/'skin-self-unallowed.jsonl.gz'; crossp=p.parent/'changed-skin-target-intersections.jsonl.gz'
    assert sha(selfp)==d['self_witnesses_sha256'] and sha(crossp)==d['cross_witnesses_sha256']
    steps.append({'step':step,'accepted_time_s':d['candidate_accepted_time_s'],'accepted_body_state_sha256':d['candidate_accepted_body_state_sha256'],
        'accepted_respiration_state_sha256':d['candidate_accepted_respiration_state_sha256'],'candidate_pack_sha256':d['candidate_pack_sha256'],
        'candidate_receipt_sha256':d['candidate_receipt_sha256'],'baseline_pack_sha256':d['baseline_pack_sha256'],'baseline_receipt_sha256':d['baseline_receipt_sha256'],
        'full_candidate_skin_face_count':d['full_candidate_skin_face_count'],'candidate_changed_face_count':d['candidate_changed_face_count'],
        'skin_self_unallowed_intersection_count':d['skin_self_unallowed_intersection_count'],
        'skin_self_allowed_shared_vertex_or_edge_pair_count':d['skin_self_allowed_shared_vertex_or_edge_pair_count'],
        'changed_star_target_surface_count':d['changed_star_target_surfaces_scanned'],'changed_star_target_unallowed_intersection_count':d['changed_star_target_unallowed_intersection_count'],
        'target_results_sha256':d['target_results_sha256'],'self_witnesses_sha256':d['self_witnesses_sha256'],'cross_witnesses_sha256':d['cross_witnesses_sha256'],
        'elapsed_wall_seconds':d['elapsed_wall_seconds'],'rss_peak_bytes':d['rss_peak_bytes']})
    result_paths[str(step)]={'path':str(p),'sha256':sha(p),'target_results_path':str(targetp),'self_witnesses_path':str(selfp),'cross_witnesses_path':str(crossp)}
# All per-pose preflight declarations share identical input bindings and full capture table.
decls=[]
for step in STEPS:
    if step in (0,4991,5375,5759): dpath=OLD/'declaration.json'
    else: dpath=PAR/f'native-1191-differential-parallel-002-step-{step}/declaration.json'
    decls.append(json.loads(dpath.read_text()))
first=decls[0]
assert all(d['source_hashes']==first['source_hashes'] for d in decls)
assert all(d['capture_file_sha256']==first['capture_file_sha256'] for d in decls)
# Re-hash source/capture inputs at finalization.
source_stable=all(sha(p)==h for p,h in first['source_hashes'].items())
capture_stable=all(sha(p)==h for runm in first['capture_file_sha256'].values() for p,h in runm.items())
assert source_stable and capture_stable
assert sha(ID)== '6478a22fe17f7ecb67e5e3042dcf0d46a6403ada7d18f236f7ff9c58a22ce4e9'
OUT.mkdir()
report={'schema':'numi.human.native-skin-epl143-changed-star-differential.summary.v1',
 'status':'complete_differential_coverage_intersection_free',
 'qualification':'Eight discrete actual native accepted poses only. All candidate skin self pairs were freshly audited at each pose. Every candidate-modified skin face was freshly tested against all 859 target surfaces. For unchanged skin-target faces, zero intersections transfer from the pinned complete 937 baseline audit only after exact source-index/weight, captured target-geometry, unchanged-skin-vertex, accepted body/respiration/time identity checks. This is not continuous-time, treatment-pose, support-selector/Jacobian, or full-body all-pair qualification.',
 'candidate_native_run_path':str(CAND),'candidate_run_metadata_sha256':sha(CAND/'run-metadata.json'),'candidate_invocation_sha256':sha(CAND/'invocation.json'),
 'baseline_native_run_path':str(BASE),'baseline_run_metadata_sha256':sha(BASE/'run-metadata.json'),'baseline_invocation_sha256':sha(BASE/'invocation.json'),
 'candidate_skin_path':str(CAND_SKIN),'candidate_skin_sha256':sha(CAND_SKIN),'baseline_skin_path':str(BASE_SKIN),'baseline_skin_sha256':sha(BASE_SKIN),
 'candidate_source_recipe_report_path':str(E/'attempt-006/report.json'),'candidate_source_recipe_report_sha256':sha(E/'attempt-006/report.json'),
 'candidate_skin_source_changed_position_vertices':33,'candidate_skin_source_changed_normal_vertices':207,'candidate_skin_changed_face_rows':90,
 'full_skin_source_face_count':109211,'target_surface_count':859,
 'baseline_complete_audit_summary_path':str(BASE_SUMMARY),'baseline_complete_audit_summary_sha256':sha(BASE_SUMMARY),
 'corrected_non_skin_identity_report_path':str(ID),'corrected_non_skin_identity_report_sha256':sha(ID),
 'identity_serialization_note':'The earlier non_skin_identity.json was preserved unchanged with a literal backslash-n at EOF. This successor replaces only that two-byte terminator with a single newline; parsed JSON content is unchanged.',
 'original_identity_report_sha256':'c6e3d348e4fb6fa498676dbfe7425b67f05d2303debc851bde50bab09355c2d3',
 'audit_script_path':str(SCRIPT),'audit_script_sha256':sha(SCRIPT),'aggregation_script_path':str(AGG),'aggregation_script_sha256':sha(AGG),'audit_workers':'Two independent Python processes for final four poses. The earlier two-thread attempt completed four poses and was terminated after measured GIL serialization; all its completed pose results are reused exactly. Predicate and geometry checks are unchanged.',
 'source_hashes_unchanged_after_all_scans':source_stable,'capture_hashes_unchanged_after_all_scans':capture_stable,
 'source_hashes':first['source_hashes'],'capture_file_sha256':first['capture_file_sha256'],
 'pose_result_paths':result_paths,'steps':steps,'all_eight_steps_complete':len(steps)==8,
 'all_steps_full_skin_self_scanned':True,'all_steps_all_859_targets_scanned_for_all_90_changed_faces':True,
 'all_steps_candidate_skin_self_unallowed_pair_count_zero':all(x['skin_self_unallowed_intersection_count']==0 for x in steps),
 'all_steps_candidate_changed_face_target_intersection_count_zero':all(x['changed_star_target_unallowed_intersection_count']==0 for x in steps),
 'elapsed_wall_scope':'Per-step values are audit-process wall seconds. Four step scans in the first two-thread run were GIL-contended; final four were independent processes. Native 1191 timing is separate.',
 'per_step_results_source':'Exact per-pose JSON and witness files remain in the referenced output directories; this summary records their full paths and SHA-256.'}
write(OUT/'summary.json',report)
shutil.copyfile(SCRIPT,OUT/'epl143_native_diff_v2.py')
shutil.copyfile(AGG,OUT/'epl143_aggregate.py')
# Preserve the interrupted optimization attempt without altering its original declaration or four results.
partial={'schema':'numi.human.native-skin-epl143-differential-partial-attempt.v1','status':'partial_interrupted_after_four_complete_poses',
 'path':str(OLD),'audit_script_path':str(SCRIPT),'audit_script_sha256':sha(SCRIPT),'command':'python3.13 epl143_native_diff_v2.py --out native-1191-differential-full-001 --workers 2',
 'completed_steps':[0,4991,5375,5759],'remaining_steps':[6111,6495,7743,10000],
 'reason':'Stopped after four completed poses when wall time showed that two Python threads serialized under the GIL. All completed results were retained and used; remaining poses were completed in separate processes with unchanged exact predicates.',
 'retained_result_sha256':{str(s):result_paths[str(s)]['sha256'] for s in [0,4991,5375,5759]},
 'partial_output_hashes':{str(p.relative_to(OLD)):sha(p) for p in OLD.rglob('*') if p.is_file()}}
write(OUT/'partial-attempt-001.json',partial)
readme='''# EPL143 candidate native captured-geometry differential\n\nThis report covers eight accepted captures from native run 1191. It freshly checks all 109,211 candidate skin faces for self-intersection and scans the 90 source faces incident to the 33 changed skin vertices against each of 859 target surfaces. Unchanged skin-target pairs transfer only from the complete, zero-intersection 937 baseline after exact native state, target, topology, weights, and unchanged skin-vertex checks.\n\nThe result is discrete captured-geometry evidence. It does not establish continuous-time, treatment-pose, support-selector/Jacobian, full-body all-pair, or runtime contact qualification.\n\nReproduction uses the pinned Python 3.13 environment and runner in the summary. Output directories are versioned and immutable; the runner refuses existing output paths.\n'''
(OUT/'README.md').write_text(readme)
manifest={'files':{str(p):sha(p) for p in [OUT/'summary.json',OUT/'partial-attempt-001.json',OUT/'README.md',OUT/'epl143_native_diff_v2.py',OUT/'epl143_aggregate.py']},'schema':'numi.human.native-skin-epl143-differential-file-manifest.v1'}
write(OUT/'manifest.json',manifest)
print(json.dumps({'output':str(OUT),'summary_sha256':sha(OUT/'summary.json'),'partial_sha256':sha(OUT/'partial-attempt-001.json'),'manifest_sha256':sha(OUT/'manifest.json'),'status':report['status'],'steps':[(x['step'],x['skin_self_unallowed_intersection_count'],x['changed_star_target_unallowed_intersection_count'],x['elapsed_wall_seconds']) for x in steps]},indent=2))
