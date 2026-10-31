from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,os,shutil,subprocess,sys
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
D=H/'quality-volume-archive-chunk-003-plan'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def load(p): return json.loads(Path(p).read_text())
planp=D/'selection-plan.json'; proofp=D/'current-input-proof-pre-retirement.json'; intentp=D/'retirement-intent-refreshed-003.json'; oldintentp=D/'retirement-intent.json'; remoteP=D/'archive-remote-verification.json'; buildP=D/'archive-build-receipt.json'; indexP=D/'archive-index.json'; erratump=D/'revalidation-intent-collision-erratum.json'
plan=load(planp); proof=load(proofp); intent=load(intentp); remote=load(remoteP); build=load(buildP); oldintent=load(oldintentp); erratum=load(erratump)
assert sha(planp)=='2ecbbc9c24655ee242e83b837b851422b620936f8d0420ec174d15e6772c4128'
assert sha(proofp)==intent['current_input_proof']['sha256'] and sha(proofp)==intent['current_input_proof']['sha256']
assert sha(intentp)==intent.get('intent_sha256',sha(intentp))
assert intent['status']=='PRECHECK_PASS_EXACT_16_ONLY_REFRESHED_PROOF' and intent['target_count']==16 and intent['retired_original_bytes']==338546535
assert intent['supersedes_historical_intent']['sha256']==sha(oldintentp)==erratum['old_intent_sha256']
assert erratum['no_source_files_retired'] is True
assert proof['status']=='PASS_CURRENT_859_861_AND_CANONICAL824_REHASHED; PASS_SELECTED_16_MATCH_RECEIPTS_ARCHIVE; PASS_NO_DIRECT_OVERLAP'
assert build['status']=='PASS_LOCAL_STREAMED_ARCHIVE_BUILD' and remote['status']=='PASS_LOCAL_BUILD_AND_GITHUB_ASSET_RESTORE_VERIFICATION'
assert build['archive_sha256']==remote['github_asset']['digest'].removeprefix('sha256:')
archive=Path(plan['release_target']['local_archive_path']); download=Path(plan['release_target']['independent_download_directory'])/plan['release_target']['asset_name']
assert sha(archive)==sha(download)==build['archive_sha256']
assert sha(indexP)==build['index_sha256'] and remote['github_asset']['name']==plan['release_target']['asset_name']
rb=load(D/'npz-readback-report.json'); assert rb['status']=='PASS_ALL_NPZ_ZIP_CRC_AND_NPY_HEADERS' and rb['npz_count']==16 and rb['archive_sha256']==build['archive_sha256']
restored=Path(remote['restore_tool']['restore_root'])/plan['files'][0]['source_relative_path']
assert restored.is_file() and sha(restored)==plan['files'][0]['expected_sha256_from_source_receipt']
# Recheck state nodes and current direct assets against the just-created proof.
current={r['path']:r for r in proof['current_states_rehashed']}; direct_expected={r['path']:r for r in proof['current_direct_npz_assets_rehashed']}; direct_now={}
for statepath,row in current.items():
 p=Path(statepath); assert p.is_file() and not p.is_symlink() and sha(p)==row['sha256']
 s=load(p)
 for role in ('Fat_current','Skin_current'):
  v=s.get(role)
  if isinstance(v,dict) and isinstance(v.get('path'),str):
   q=Path(v['path']); assert q.is_file() and not q.is_symlink()
   h=sha(q); assert h==v.get('sha256')
   direct_now[str(q)]={'path':str(q),'sha256':h,'size_bytes':q.stat().st_size,'state':str(p),'role':role}
assert set(direct_now)==set(direct_expected) and all(direct_now[k]['sha256']==direct_expected[k]['sha256'] for k in direct_now)
# Verify 861 remains sourced from the currently pinned qualified 859 state.
s861=load(H/'all-quality-861-root/state.json'); assert s861['source_state']['path']==str(H/'all-quality-859-qualified-root/quality-working-state.json') and s861['source_state']['sha256']==current[str(H/'all-quality-859-qualified-root/quality-working-state.json')]['sha256']
# Recheck the previously protected post-824 direct-input proof.
priorp=H/'quality-volume-archive-chunk-002-plan/current-input-proof-post-retirement.json'; prior=load(priorp)
assert prior['status']=='PASS_POST_RETIREMENT_12_PATHS_ABSENT; PASS_PROTECTED_CURRENT_INPUTS_UNCHANGED; PASS_RESTORE_REHYDRATION'
assert sha(priorp)==proof['prior_protected_current_input_proof']['sha256']
# Exact 16-path authorization must equal both plan and refreshed proof; no other paths are touched.
rows=plan['files']; targets=proof['selected_paths_pre_retirement']; assert len(rows)==len(targets)==16
by_path={r['path']:r for r in targets}; assert set(by_path)=={r['path'] for r in rows}
selected_bytes=0
for row in rows:
 p=Path(row['path']); pr=by_path[str(p)]
 assert not p.is_symlink() and p.is_file() and p.stat().st_size==row['size_bytes_from_stat']==pr['size_bytes']
 assert row['archive_member']==pr['archive_member'] and str(p)==str(H/row['source_relative_path']) and 700<=int(row['source_relative_path'].split('/')[0].removeprefix('all-quality-').removesuffix('-root'))<=790
 assert row['source_relative_path'].startswith('all-quality-703-root/')
 assert sha(p)==row['expected_sha256_from_source_receipt']==pr['sha256']
 assert str(p) not in direct_now
 selected_bytes+=p.stat().st_size
assert selected_bytes==338546535
# Check no unrelated process has these exact old volume files open.
lsof=shutil.which('lsof')
if not lsof: raise RuntimeError('lsof unavailable; refusing retirement without exact open-handle check')
opened=subprocess.run([lsof,'-t',*[r['path'] for r in rows]],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if opened.returncode not in (0,1) or opened.stdout.strip(): raise RuntimeError('selected source file has an open process: '+opened.stdout.strip())
progressp=D/'retirement-progress-003.json'
progress={'schema':'numi-human.quality-volume-archive-retirement-progress.v1','status':'PREFLIGHT_PASS; RETIRING_EXACT_16','created_utc':datetime.now(timezone.utc).isoformat(),'selection_plan_sha256':sha(planp),'fresh_input_proof_sha256':sha(proofp),'refreshed_intent_sha256':sha(intentp),'archive_sha256':build['archive_sha256'],'selected_count':16,'selected_bytes':selected_bytes,'unlinked':[],'current_direct_assets_rehashed':len(direct_now),'lsof_exact_targets_open':False}
def save_progress():
 t=progressp.with_suffix('.json.tmp'); t.write_text(json.dumps(progress,indent=2,sort_keys=True)+'\n'); os.replace(t,progressp)
save_progress()
for row in rows:
 p=Path(row['path']); os.unlink(p)
 progress['unlinked'].append({'path':str(p),'sha256':row['expected_sha256_from_source_receipt'],'size_bytes':row['size_bytes_from_stat']})
 save_progress()
progress['status']='PASS_EXACT_16_PRE824_VOLUME_NPZ_RETIRED'
progress['completed_utc']=datetime.now(timezone.utc).isoformat(); progress['disk_free_after_unlink_bytes']=shutil.disk_usage(H).free
save_progress()
receipt={**progress,'receipt_schema':'numi-human.quality-volume-archive-retirement-receipt.v1','no_json_or_nonselected_file_removed':True,'github_recovery_tag':plan['release_target']['tag'],'github_asset_id':remote['github_asset']['id'],'github_asset_digest':remote['github_asset']['digest'],'independent_download_sha256':sha(download),'archive_still_present':archive.is_file(),'download_still_present':download.is_file(),'restore_smoke_still_present':restored.is_file()}
receiptp=D/'retirement-receipt-003-pre-postproof.json'; receiptp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':progress['status'],'retired_count':len(progress['unlinked']),'retired_bytes':selected_bytes,'receipt':str(receiptp),'receipt_sha256':sha(receiptp),'progress':str(progressp),'progress_sha256':sha(progressp),'disk_free_after_unlink_bytes':progress['disk_free_after_unlink_bytes'],'archive_sha256':build['archive_sha256']},sort_keys=True))
