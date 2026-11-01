#!/usr/bin/env python3
"""Stream a current source-only Human 943 snapshot into content-addressed tar.zst."""
import argparse, datetime, hashlib, io, json, os, shutil, subprocess, tarfile
from pathlib import Path

def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def has_pin(node,target,digest):
 if isinstance(node,dict):
  p=node.get('path'); s=node.get('sha256')
  if isinstance(p,str) and Path(p).is_absolute() and str(Path(p).resolve())==target and s==digest: return True
  return any(has_pin(v,target,digest) for v in node.values())
 if isinstance(node,list): return any(has_pin(v,target,digest) for v in node)
 return False
def add_bytes(tar,name,data):
 info=tarfile.TarInfo(name); info.size=len(data); info.mtime=0; info.uid=info.gid=0; info.uname=info.gname=''; info.mode=0o644
 tar.addfile(info,io.BytesIO(data))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--plan',required=True); ap.add_argument('--archive',required=True); ap.add_argument('--build',action='store_true'); args=ap.parse_args()
 plan_path=Path(args.plan).resolve(); plan=json.loads(plan_path.read_text()); out=plan_path.parent
 if plan.get('schema')!='numi-human.source-only-geometry-snapshot-plan.v1': raise RuntimeError('unexpected plan schema')
 if plan.get('preparation_status')!='READY_FOR_REVIEW_LIVE_PAYLOAD_HASH_PENDING': raise RuntimeError('plan is not prepared')
 if plan['chunk'].get('retirement_enabled') is not False or plan['selection'].get('no_current_files_retired') is not True: raise RuntimeError('plan must be no-retirement')
 source_root=Path(plan['source_root']).resolve(); rows=plan['files']
 if len(rows)!=plan['chunk']['file_count'] or sum(r['size_bytes'] for r in rows)!=plan['chunk']['original_bytes']: raise RuntimeError('plan count/size mismatch')
 if len({r['path'] for r in rows})!=len(rows): raise RuntimeError('duplicate input path')
 # Verify source-only status and all source pins before emitting any bytes.
 if plan['snapshot']['physical_admission'] is not False or plan['snapshot']['native_steps']!=0 or plan['snapshot']['fat_native_quality_failures']!=6 or plan['snapshot']['skin_native_quality_failures']!=6 or plan['snapshot']['mass_reconciliation_status']!='PENDING_VOLUME_CHANGE_RECONCILIATION': raise RuntimeError('snapshot qualification labels changed')
 docs={r['role']:json.loads(Path(r['path']).read_text()) for r in rows if r['path'].endswith('.json')}
 pin_files={r['role']:r for r in rows}
 checked=[]
 for row in rows:
  p=Path(row['path'])
  if not p.is_absolute() or p.resolve()!=p or p.is_symlink() or not p.is_file(): raise RuntimeError('source not a regular canonical file: '+str(p))
  if p.stat().st_size!=row['size_bytes']: raise RuntimeError('source size changed: '+str(p))
  digest=sha_file(p)
  if digest!=row['sha256']: raise RuntimeError('source SHA changed: '+str(p))
  if p.suffix.lower()=='.npz':
   actual_pinners=[]
   for role in row['pinning_selected_json_roles']:
    if role not in docs or not has_pin(docs[role],str(p),digest): raise RuntimeError(f'cross-pin mismatch for {row["role"]} in {role}')
    actual_pinners.append(role)
   if row['pin_status']=='CROSS_PINNED_BY_SELECTED_JSON' and not actual_pinners: raise RuntimeError('expected selected JSON cross-pin missing: '+row['role'])
  checked.append((row,p,digest))
 if not args.build:
  print(json.dumps({'status':'PREFLIGHT_PASS_NO_ARCHIVE_WRITTEN','snapshot':plan['snapshot']['id'],'file_count':len(rows),'logical_bytes':plan['chunk']['original_bytes'],'live_files_rehashed':len(checked)},sort_keys=True)); return
 archive=Path(args.archive).resolve(); partial=archive.with_name(archive.name+'.partial')
 if archive.exists() or partial.exists(): raise RuntimeError('refuse existing archive/partial')
 # Bound the local write; keep a 150 MiB reserve with additional 20 MiB margin.
 free_before=shutil.disk_usage(archive.parent).free
 reserve=150*1024*1024
 safety=20*1024*1024
 max_archive=min(300*1024*1024,free_before-reserve-safety)
 if max_archive<=0: raise RuntimeError('insufficient free space for bounded archive')
 index_files=[]; unique={}
 for row,p,digest in checked:
  index_files.append({'role':row['role'],'kind':row['kind'],'path':row['path'],'source_relative_path':row['source_relative_path'],'sha256':digest,'size_bytes':row['size_bytes'],'pin_status':row['pin_status'],'pinning_selected_json_roles':row['pinning_selected_json_roles'],'archive_member':'blobs/'+digest})
  unique.setdefault(digest,{'sha256':digest,'size_bytes':row['size_bytes']})
 method_files=[]
 for name in ['prepare_snapshot.py','build_943_snapshot.py']:
  p=out/name; method_files.append({'path':str(p),'sha256':sha_file(p),'size_bytes':p.stat().st_size,'archive_member':'snapshot-tooling/'+name})
 provenance={'schema':'numi-human.source-only-snapshot-provenance.v1','snapshot':plan['snapshot'],'qualification_receipt_roles':[r['role'] for r in rows if 'proof' in r['kind'] or 'lineage_proof'==r['kind']],'source_roles':[r['role'] for r in rows if r['kind'] in ('source_geometry','volume_proposal')],'files':index_files,'limitations':['This archive does not close the complete historical source/volume ancestry.','This is source-only evidence: physical admission is false, native steps are zero, six Fat and six Skin native quality failures remain, and mass reconciliation is pending.','The archived proposals are retained as current 943 inputs; no local source, volume, state, receipt, or evidence file is retired.'],'method_files':method_files}
 prov_bytes=(json.dumps(provenance,indent=2,sort_keys=True)+'\n').encode()
 index={'schema':'numi-human.quality-source-only-snapshot-archive.v1','snapshot':{'id':plan['snapshot']['id'],'kind':'current_source_only_qualified_geometry_snapshot','source_state':plan['snapshot']['status'],'physical_admission':False,'native_steps':0,'fat_native_quality_failures':6,'skin_native_quality_failures':6,'mass_reconciliation_status':'PENDING_VOLUME_CHANGE_RECONCILIATION','full_original_ancestry_included':False,'scope_limitation':plan['snapshot']['original_ancestry_closure']},'workflow':plan['source_workflow'],'selection_plan_sha256':sha_file(plan_path),'provenance_sha256':sha_bytes(prov_bytes),'method_files':method_files,'files':index_files,'unique_blobs':sorted(unique.values(),key=lambda x:x['sha256']),'unique_blob_count':len(unique),'unique_blob_bytes':sum(x['size_bytes'] for x in unique.values()),'file_reference_count':len(index_files),'no_input_retirement':True}
 index_bytes=(json.dumps(index,indent=2,sort_keys=True)+'\n').encode(); selection_bytes=plan_path.read_bytes(); builder_path=Path(__file__).resolve(); builder_bytes=builder_path.read_bytes(); prep_path=out/'prepare_snapshot.py'; prep_bytes=prep_path.read_bytes()
 zstd=Path('/opt/homebrew/bin/zstd')
 if not zstd.is_file(): raise RuntimeError('pinned zstd executable missing')
 with partial.open('xb') as archive_stream:
  proc=subprocess.Popen([str(zstd),'-T1','--fast=1','-c'],stdin=subprocess.PIPE,stdout=archive_stream)
  try:
   with tarfile.open(fileobj=proc.stdin,mode='w|') as tar:
    add_bytes(tar,'archive-index.json',index_bytes)
    add_bytes(tar,'snapshot/selection-plan.json',selection_bytes)
    add_bytes(tar,'snapshot/snapshot-provenance.json',prov_bytes)
    add_bytes(tar,'snapshot-tooling/prepare_snapshot.py',prep_bytes)
    add_bytes(tar,'snapshot-tooling/build_943_snapshot.py',builder_bytes)
    emitted=set()
    for row,p,digest in checked:
     if digest in emitted: continue
     emitted.add(digest); info=tarfile.TarInfo('blobs/'+digest); info.size=p.stat().st_size; info.mtime=0; info.uid=info.gid=0; info.uname=info.gname=''; info.mode=0o644
     with p.open('rb') as f:
      hasher=hashlib.sha256(); count=0
      class Reader:
       def read(self,n=-1):
        nonlocal hasher,count
        b=f.read(n)
        if b: hasher.update(b); count+=len(b)
        return b
      tar.addfile(info,Reader())
     if hasher.hexdigest()!=digest or count!=info.size: raise RuntimeError('source changed during archive stream: '+str(p))
   proc.stdin.close(); rc=proc.wait()
   if rc: raise RuntimeError('zstd exited '+str(rc))
  except BaseException:
   if proc.poll() is None: proc.kill(); proc.wait()
   raise
 size=partial.stat().st_size
 if size>max_archive: raise RuntimeError(f'archive size {size} exceeds bounded max {max_archive}')
 os.replace(partial,archive)
 free_after=shutil.disk_usage(archive.parent).free
 if free_after<reserve: raise RuntimeError(f'archive created but free-space reserve is below 150 MiB: {free_after}')
 receipt={'schema':'numi-human.source-only-snapshot-archive-build-receipt.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'),'status':'PASS_LOCAL_STREAMED_SOURCE_ONLY_SNAPSHOT_ARCHIVE','snapshot_id':plan['snapshot']['id'],'archive_path':str(archive),'archive_sha256':sha_file(archive),'archive_size_bytes':size,'selection_plan_sha256':sha_file(plan_path),'archive_index_sha256':sha_bytes(index_bytes),'provenance_sha256':sha_bytes(prov_bytes),'selected_file_count':len(rows),'unique_blob_count':len(unique),'original_bytes':plan['chunk']['original_bytes'],'free_bytes_before':free_before,'free_bytes_after':free_after,'minimum_free_reserve_bytes':reserve,'no_source_retirement_performed':True,'physical_admission':False,'native_steps':0,'mass_reconciliation_status':'PENDING_VOLUME_CHANGE_RECONCILIATION'}
 (out/'archive-index.json').write_bytes(index_bytes)
 (out/'snapshot-provenance.json').write_bytes(prov_bytes)
 (out/'archive-build-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps(receipt,sort_keys=True))
if __name__=='__main__': main()
