#!/usr/bin/env python3
"""Remove only the already GH-verified local staging archive, preserving sources/evidence."""
import datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path
OUT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/quality-958-source-snapshot-20261101-plan')
PLAN=OUT/'selection-plan.json'; BUILD=OUT/'archive-build-receipt.json'; VERIFIED=OUT/'independent-verification.json'; META=OUT/'github-release-metadata.json'; ARCHIVE=OUT/'quality-958-source-snapshot-20261101-001.tar.zst'; RELEASE='human-quality-source-snapshot-20261101-958-001'; ASSET='quality-958-source-snapshot-20261101-001.tar.zst'; REPO='Numi2/numilab-human'
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def noopen_handles(p):
 lsof=shutil.which('lsof') or '/usr/sbin/lsof'
 if not Path(lsof).is_file(): raise RuntimeError('lsof unavailable; cannot prove no open local archive handle')
 r=subprocess.run([lsof,'-t','--',str(p)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=20)
 if r.returncode==0 and r.stdout.strip(): raise RuntimeError('archive is open by process(es): '+r.stdout.strip())
 if r.returncode not in (0,1): raise RuntimeError('lsof failed: '+r.stderr[-1000:])
 return {'status':'NO_OPEN_HANDLES','lsof_returncode':r.returncode,'pids':r.stdout.split()}
def main():
 plan=json.loads(PLAN.read_text()); build=json.loads(BUILD.read_text()); verified=json.loads(VERIFIED.read_text()); meta=json.loads(META.read_text())
 if not ARCHIVE.is_file() or ARCHIVE.is_symlink(): raise RuntimeError('expected local archive missing, nonregular, or symlinked')
 ap=ARCHIVE.resolve(); expected_path=Path(build['archive_path']).resolve()
 if ap!=expected_path: raise RuntimeError('local archive path differs from build receipt')
 if build.get('status')!='PASS_LOCAL_STREAMED_SOURCE_ONLY_SNAPSHOT_ARCHIVE' or verified.get('status')!='PASS_INDEPENDENT_GH_ARCHIVE_FULL_BLOB_AND_EXACT_NPZ_RAM_RESTORE': raise RuntimeError('archive build/independent verification status is not passing')
 if verified.get('no_current_files_retired') is not True or verified.get('physical_admission') is not False or verified.get('native_steps')!=0: raise RuntimeError('source-only/no-retirement receipt labels mismatch')
 if build.get('archive_sha256')!=verified.get('github_asset',{}).get('archive_sha256_from_remote_stream') or build.get('archive_size_bytes')!=verified.get('github_asset',{}).get('archive_size_from_remote_stream'): raise RuntimeError('build and independent remote archive identities differ')
 if plan.get('snapshot',{}).get('physical_admission') is not False or plan['snapshot'].get('native_steps')!=0 or plan['snapshot'].get('mass_reconciliation_status')!='PENDING_VOLUME_CHANGE_RECONCILIATION': raise RuntimeError('source-only status changed')
 if sha_file(PLAN)!=build.get('selection_plan_sha256') or verified.get('selection_plan_sha256')!=build.get('selection_plan_sha256'): raise RuntimeError('selection plan hash differs across receipts')
 gh=subprocess.run(['/opt/homebrew/bin/gh','release','view',RELEASE,'--repo',REPO,'--json','id,tagName,name,url,assets'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30)
 if gh.returncode:raise RuntimeError('fresh GitHub release check failed: '+gh.stderr.decode('utf8','replace')[-1000:])
 live=json.loads(gh.stdout); saved=json.loads(META.read_text())
 if live.get('tagName')!=RELEASE or saved.get('tagName')!=RELEASE:raise RuntimeError('live/saved release tag mismatch')
 matches=[a for a in live.get('assets',[]) if a.get('name')==ASSET]
 if len(matches)!=1:raise RuntimeError('expected exactly one live archive asset')
 asset=matches[0]; expected_sha=build['archive_sha256']; expected_size=build['archive_size_bytes']
 if asset.get('digest')!='sha256:'+expected_sha or asset.get('size')!=expected_size:raise RuntimeError('live GH asset hash/size differs from verified local archive')
 saved_matches=[a for a in saved.get('assets',[]) if a.get('name')==ASSET]
 if len(saved_matches)!=1 or any(saved_matches[0].get(k)!=asset.get(k) for k in ('id','node_id','name','digest','size')):raise RuntimeError('live/saved GH asset identity mismatch')
 if verified.get('github_asset',{}).get('id')!=asset.get('id') or verified.get('github_asset',{}).get('node_id')!=asset.get('node_id'):raise RuntimeError('live GH asset identity differs from independent verification receipt')
 if sha_file(ARCHIVE)!=expected_sha or ARCHIVE.stat().st_size!=expected_size:raise RuntimeError('fresh local archive hash/size mismatch')
 if ARCHIVE.stat().st_nlink!=1:raise RuntimeError('refuse local archive retirement because link count is not one')
 handles=noopen_handles(ARCHIVE)
 # Freshly prove every selected original input remains in place and byte-identical.
 source_checks=[]
 for row in plan['files']:
  p=Path(row['path'])
  if not p.is_file() or p.is_symlink() or p.stat().st_size!=row['size_bytes'] or sha_file(p)!=row['sha256']:
   raise RuntimeError('original source/evidence input changed: '+row['role'])
  source_checks.append({'role':row['role'],'path':str(p),'sha256':row['sha256'],'size_bytes':row['size_bytes'],'unchanged':True})
 free_before=shutil.disk_usage(ARCHIVE.parent).free; archive_size=ARCHIVE.stat().st_size; archive_sha=sha_file(ARCHIVE)
 # This is the sole unlink in this script; it names only the verified local archive.
 os.unlink(ARCHIVE)
 if ARCHIVE.exists():raise RuntimeError('verified staging archive remains after unlink')
 free_after=shutil.disk_usage(OUT).free
 receipt={'schema':'numi-human.verified-local-snapshot-staging-retirement.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'),'status':'PASS_REMOVED_ONLY_INDEPENDENTLY_VERIFIED_LOCAL_STAGING_ARCHIVE','removed_archive_path':str(ap),'removed_archive_sha256':archive_sha,'removed_archive_size_bytes':archive_size,'independent_github_release_tag':RELEASE,'independent_github_asset_name':ASSET,'independent_github_asset_id':asset.get('id'),'independent_github_asset_node_id':asset.get('node_id'),'independent_github_asset_digest':asset.get('digest'),'independent_archive_stream_sha256':verified['github_asset']['archive_sha256_from_remote_stream'],'github_release_url':live.get('url'),'build_receipt_sha256':sha_file(BUILD),'independent_verification_receipt_sha256':sha_file(VERIFIED),'selection_plan_sha256':sha_file(PLAN),'live_github_release_identity_matches_saved':live.get('id')==saved.get('id') and live.get('tagName')==saved.get('tagName') and live.get('url')==saved.get('url'), 'live_github_asset_identity_matches_saved':len([a for a in saved.get('assets',[]) if a.get('name')==ASSET])==1 and all(next(a for a in saved.get('assets',[]) if a.get('name')==ASSET).get(k)==asset.get(k) for k in ('id','node_id','name','digest','size')),'local_archive_no_open_handles':handles,'local_archive_link_count_before':1,'selected_original_inputs_rehashed_unchanged':len(source_checks),'selected_original_inputs':source_checks,'no_original_source_or_evidence_file_removed':True,'free_bytes_before':free_before,'free_bytes_after':free_after,'measured_free_bytes_recovered':free_after-free_before,'physical_admission':False,'native_steps':0,'mass_reconciliation_status':'PENDING_VOLUME_CHANGE_RECONCILIATION'}
 p=OUT/'local-archive-retirement-receipt.json';p.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':receipt['status'],'removed_archive_path':str(ap),'sha256':archive_sha,'size_bytes':archive_size,'free_bytes_before':free_before,'free_bytes_after':free_after,'recovered_bytes':free_after-free_before,'receipt':str(p),'source_files_unchanged':len(source_checks)},sort_keys=True))
if __name__=='__main__':main()
