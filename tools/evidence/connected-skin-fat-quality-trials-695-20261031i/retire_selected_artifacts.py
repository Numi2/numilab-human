#!/usr/bin/env python3
import argparse,hashlib,json,shutil,time
from pathlib import Path
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
PK=Path('/Users/numi/Work/human-quality-trials-695-publish-20261031i/tools/evidence/connected-skin-fat-quality-trials-695-20261031i')
OUT=H/'quality-695-failed-trials-archive-001'
PLAN=PK/'selection-and-retirement.json'
RETIRE=PK/'retirement-plan.json'
VERIFY=PK/'archive-verification.json'
REMOTE=PK/'archive-remote-verification.json'
INDEX=PK/'archive-index.json'
ARCHIVE=OUT/'quality695-trial01-failed-proposals-001-012.tar.zst'
DOWNLOAD=Path('/Users/numi/Work/.quality695-independent-download/quality695-trial01-failed-proposals-001-012.tar.zst')
RESTORE_ROOT=Path('/Users/numi/Work/.quality695-restored-verify')
TOMBSTONE=OUT/'retirement-tombstone-001.json'
RECEIPT=PK/'retirement-receipt.json'
BEST=H/'all-quality-738-root/round003/points/verified-state.json'
BEST_SHA='40776e9704c3fce24699537dd0df6d40df2b29685b1429e7ce43963fae5c3f1c'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def load(p): return json.loads(Path(p).read_text())
def checks():
 plan=load(PLAN); idx=load(INDEX); ver=load(VERIFY); rem=load(REMOTE)
 if ver.get('status')!='PASS_STREAMED_ARCHIVE_AND_SELECTED_RESTORE' or rem.get('status')!='PASS_INDEPENDENT_RELEASE_ASSET_DOWNLOAD_AND_HASH': raise RuntimeError('independent archive verification not passed')
 assetsha=rem['github_asset_digest'].removeprefix('sha256:')
 if assetsha!=rem['independent_download_sha256'] or assetsha!=rem['local_archive_sha256']: raise RuntimeError('remote/local archive digest mismatch')
 if ver['archive_sha256']!=assetsha or ver['verified_blob_count']!=40 or ver['verified_blob_bytes']!=637509229: raise RuntimeError('full stream proof does not match selection')
 if sha(BEST)!=BEST_SHA: raise RuntimeError('current 738 best-state pin changed')
 if sha(PK/'checkpoint-snapshot.json')!=plan['current_checkpoint']['sha256']: raise RuntimeError('frozen checkpoint snapshot changed')
 protected={x['path'] for x in plan['protected_current_state_references']}
 rows={x['path']:x for x in idx['files']}
 if len(plan['files'])!=40 or len(rows)!=40: raise RuntimeError('selection/index count mismatch')
 out=[]; root=(H/'all-quality-695-root').resolve()
 for c in plan['files']:
  p=Path(c['path']); rp=p.resolve()
  if rp!=p or root not in rp.parents or not p.is_file(): raise RuntimeError('source path missing/unexpected: '+str(p))
  if str(rp) in protected: raise RuntimeError('source is in current protected path set: '+str(p))
  row=rows.get(str(p))
  if not row or row['sha256']!=c['sha256'] or row['size_bytes']!=c['size_bytes']: raise RuntimeError('index mismatch: '+str(p))
  if sha(p)!=c['sha256']: raise RuntimeError('source bytes no longer match their state receipt: '+str(p))
  if p.parent.name!='trial01' or p.name=='candidate-I-typed-current.npz': raise RuntimeError('source is outside authorized trial01 selection: '+str(p))
  out.append({'path':str(p),'sha256':c['sha256'],'size_bytes':c['size_bytes'],'kind':c['kind']})
 if sum(x['size_bytes'] for x in out)!=637509229: raise RuntimeError('retirement byte count mismatch')
 for p in (ARCHIVE,DOWNLOAD):
  if not p.is_file() or sha(p)!=assetsha: raise RuntimeError('temporary archive copy missing or unverified: '+str(p))
 restore=Path(ver['restore_root'])/Path(ver['restore_path']).resolve().relative_to(H.resolve())
 if not restore.is_file() or sha(restore)!=ver['restore_sha256']: raise RuntimeError('selected restore verification lost')
 return plan,idx,ver,rem,out,assetsha,restore
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--prepare',action='store_true'); ap.add_argument('--retire',action='store_true'); ap.add_argument('--plan-sha'); a=ap.parse_args()
 if not (a.prepare^a.retire): ap.error('choose --prepare or --retire')
 plan,idx,ver,rem,files,assetsha,restore=checks()
 if a.prepare:
  free=shutil.disk_usage(H).free
  d={'schema':'numi-human.quality695-retirement-plan.v1','status':'READY_FOR_EXACT_FILE_RETIREMENT','selected_count':len(files),'selected_logical_bytes':sum(x['size_bytes'] for x in files),'archive_asset_sha256':assetsha,'archive_tag':rem['release_tag'],'independent_stream_restore_status':ver['status'],'protected_current_best_state':{'path':str(BEST),'sha256':BEST_SHA},'files':files,'temporary_files_to_remove':[str(ARCHIVE),str(DOWNLOAD),str(restore)],'free_bytes_before_retirement':free,'scope':'exact listed NPZ files only; preserve all JSON, source, unselected NPZ, checkpoint/state and current DAG paths'}
  RETIRE.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
  print(json.dumps({'status':'PASS_RETIREMENT_PREFLIGHT','count':len(files),'bytes':d['selected_logical_bytes'],'plan':str(RETIRE),'plan_sha256':sha(RETIRE),'free_bytes':free})); return
 if not a.plan_sha or sha(RETIRE)!=a.plan_sha: raise RuntimeError('retirement plan SHA missing/mismatch')
 frozen=load(RETIRE)
 if frozen.get('status')!='READY_FOR_EXACT_FILE_RETIREMENT' or len(frozen.get('files',[]))!=40: raise RuntimeError('retirement plan state changed')
 free0=shutil.disk_usage(H).free; removed=[]; temp_removed=[]
 # Remove only the verified temporary archive copies and the exact restored sample.
 for p in (ARCHIVE,DOWNLOAD,restore):
  p=Path(p)
  if p.exists():
   if p==restore:
    if sha(p)!=ver['restore_sha256']: raise RuntimeError('restore temp changed')
   elif sha(p)!=assetsha: raise RuntimeError('archive temp changed')
   size=p.stat().st_size; p.unlink(); temp_removed.append({'path':str(p),'size_bytes':size})
 for d in (RESTORE_ROOT/'all-quality-695-root/round001/trial01',RESTORE_ROOT/'all-quality-695-root/round001',RESTORE_ROOT/'all-quality-695-root',RESTORE_ROOT,DOWNLOAD.parent):
  if d.exists():
   try: d.rmdir()
   except OSError: pass
 # Recheck source hashes immediately before unlink, then remove exact selected paths.
 for row in files:
  p=Path(row['path'])
  if not p.is_file() or sha(p)!=row['sha256']: raise RuntimeError('retirement source changed during cleanup: '+str(p))
  p.unlink(); removed.append(row)
 missing=[x['path'] for x in removed if Path(x['path']).exists()]
 if missing: raise RuntimeError('selected source remains after unlink: '+repr(missing))
 free1=shutil.disk_usage(H).free
 receipt={'schema':'numi-human.quality695-retirement-receipt.v1','status':'PASS_SELECTED_UNREFERENCED_NPZ_RETIREMENT','retired_count':len(removed),'retired_logical_bytes':sum(x['size_bytes'] for x in removed),'selected_paths_and_hashes':removed,'temporary_verified_copies_removed':temp_removed,'archive':{'tag':rem['release_tag'],'url':rem['download_url'],'sha256':assetsha,'size_bytes':rem['independent_download_size_bytes']},'verification':{'archive_stream_restore':'PASS','independent_release_download':'PASS','selected_restore_path':ver['restore_path'],'selected_restore_sha256':ver['restore_sha256']},'free_bytes_before':free0,'free_bytes_after':free1,'measured_free_byte_delta':free1-free0,'logical_source_bytes_retired':sum(x['size_bytes'] for x in removed),'restore_command':'python3 restore_archive.py --archive quality695-trial01-failed-proposals-001-012.tar.zst --restore-root <root> --restore-path <original absolute source path>','preserved':['trial JSON state/config/report/telemetry','all unselected NPZ inputs','checkpoint snapshot and current 738 ancestry','all source and application assets'],'physical_admission':False,'native_steps':0}
 RECEIPT.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 TOMBSTONE.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':receipt['status'],'retired_count':receipt['retired_count'],'logical_bytes':receipt['retired_logical_bytes'],'free_before':free0,'free_after':free1,'measured_free_delta':free1-free0,'receipt':str(RECEIPT),'tombstone':str(TOMBSTONE)}))
if __name__=='__main__': main()

