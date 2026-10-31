#!/usr/bin/env python3
import argparse, hashlib, json, os, re, subprocess, tarfile, io
from pathlib import Path

H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
PUB=Path('/Users/numi/Work/human-quality-trials-695-publish-20261031i')
PK=PUB/'tools/evidence/connected-skin-fat-quality-trials-695-20261031i'
OUT=H/'quality-695-failed-trials-archive-001'
PLAN=PK/'selection-and-retirement.json'
INDEX=PK/'archive-index.json'
EXPECTED_BYTES=637_509_229
EXPECTED_COUNT=40
STATUS='FAIL_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
D_ROUNDS={1,2,3,4,5,6,8,9,11}
O_ROUNDS={1,2,3,4,5,6,7}
ACTIVE={
'all-quality-721w-root/round-01/proposal-exact-630/report.json':'35beac5c5c18fe572a5a23dec4a5a6fe49cd6fb125a6b8384f1a78d0a7fb7f9c',
'all-quality-721w-root/round-02/proposal-exact-630/report.json':'be1552381df3efab18829eca7a0c3f389b1c5f3487f40ecd5b8a6c7723f12c5b',
'all-quality-721y-root/round-01/proposal-exact-630/report.json':'1dcdbf04efac9ed5286b6d18d16b3826b532b28a43da6ccd88dd82063d9acef1',
'all-quality-721y-root/round-02/proposal-exact-630/report.json':'b182379ac1d74683916251e0fee17d9907bcbfc35d144c24aa19a2a82a43cb5a',
'all-quality-730-root/round-01/proposal-exact-630/report.json':'77279f036d868f3b7c63d29ab720a010008243da9ca4b452e1d623baf6df49b3',
'all-quality-730-root/round-02/proposal-exact-630/report.json':'5311043a88ead9b2f3da5dae42b9f479893247fe17bd83b8dc519b5b8878c831',
}
STATES={
'all-quality-721y-root/round-02/verified-state.json':'d64ce92370062ae1402657e043cd02f62ed8adf26b5f19324e14b90a41676b52',
'all-quality-732-root/verified-state.json':'e68c1a0e3958ca365327b35fd1476505468ebaaa5969770138e16647434a0c43',
'all-quality-738-root/round003/points/verified-state.json':'40776e9704c3fce24699537dd0df6d40df2b29685b1429e7ce43963fae5c3f1c',
'all-quality-730-root/round-02/verified-state.json':'587b91036cab62e53447125b2c109a5d82727a879089611cc509af3f6290fa01',
}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def load(path): return json.loads(Path(path).read_text())
def walk(obj,refs):
 if isinstance(obj,dict):
  p=obj.get('path'); s=obj.get('sha256')
  if isinstance(p,str): refs.append((p,s if isinstance(s,str) else None))
  for v in obj.values(): walk(v,refs)
 elif isinstance(obj,list):
  for v in obj: walk(v,refs)
def resolved(p): return str(Path(p).resolve())
def preflight():
 cp=H/'checkpoint.json'
 snapshot=PK/'checkpoint-snapshot.json'
 if not snapshot.exists(): snapshot.write_bytes(cp.read_bytes())
 cp_sha=sha(snapshot)
 checkpoint_data=load(snapshot)
 current=checkpoint_data.get('quality_repair_progress_20261031',{}).get('current_state',{})
 if current.get('path')!=str(H/'all-quality-730-root/round-02/verified-state.json') or current.get('sha256')!='587b91036cab62e53447125b2c109a5d82727a879089611cc509af3f6290fa01': raise RuntimeError('checkpoint snapshot current-state pin changed')
 newest=H/'all-quality-738-root/round003/points/verified-state.json'
 if sha(newest)!='40776e9704c3fce24699537dd0df6d40df2b29685b1429e7ce43963fae5c3f1c': raise RuntimeError('contemporaneous best state pin changed')
 active=[]; state_tips=[]; protected={}
 def add_refs(path):
  d=load(path); refs=[]; walk(d,refs)
  protected.setdefault(resolved(path),set()).add(sha(path))
  for p,s in refs:
   if s: protected.setdefault(resolved(p),set()).add(s)
  return d
 add_refs(snapshot)
 for rel,expected in STATES.items():
  p=H/rel; got=sha(p)
  if got!=expected: raise RuntimeError('current state pin changed: '+rel)
  state_tips.append({'path':str(p),'sha256':got})
 queue=[H/rel for rel in STATES]; visited=set()
 while queue:
  p=queue.pop(); rp=resolved(p)
  if rp in visited: continue
  visited.add(rp); d=add_refs(p)
  ss=d.get('source_state')
  if isinstance(ss,dict) and isinstance(ss.get('path'),str): queue.append(Path(ss['path']))
 for rel,expected in ACTIVE.items():
  p=H/rel; got=sha(p)
  if got!=expected: raise RuntimeError('active report pin changed: '+rel)
  active.append({'path':str(p),'sha256':got}); add_refs(p)
 selected=[]; internal=set()
 for r in range(1,13):
  d=H/f'all-quality-695-root/round{r:03d}/trial01'; internal.add(resolved(d))
  st=load(d/'state.json'); report=load(d/'context001/report.json')
  if report.get('status')!=STATUS or report.get('physical_admission') is not False or report.get('native_steps')!=0:
   raise RuntimeError(f'not an unadmitted failed exact trial: {d}')
  for role,name in [('Skin','skin-connected-quality-volume-proposal.npz'),('Fat','fat-connected-quality-volume-proposal.npz')]:
   rec=st.get(role+'_current'); p=d/name
   if not isinstance(rec,dict) or resolved(rec.get('path',''))!=resolved(p): raise RuntimeError(f'{role} lineage mismatch {p}')
   selected.append({'path':str(p),'sha256':rec['sha256'],'size_bytes':p.stat().st_size,'kind':'failed_'+role.lower()+'_proposal','round':r,'status':STATUS,'physical_admission':False,'native_steps':0})
  for role,rounds in [('D',D_ROUNDS),('O',O_ROUNDS)]:
   if r not in rounds: continue
   rec=st.get('source_surfaces',{}).get(role); p=d/f'candidate-{role}-typed-current.npz'
   if not isinstance(rec,dict) or resolved(rec.get('path',''))!=resolved(p): raise RuntimeError(f'{role} exact source input mismatch {p}')
   selected.append({'path':str(p),'sha256':rec['sha256'],'size_bytes':p.stat().st_size,'kind':f'unreferenced_{role}_input_snapshot','round':r,'status':STATUS,'physical_admission':False,'native_steps':0})
 # Every file must be present, state-receipt sized, and absent from protected live references.
 for c in selected:
  p=Path(c['path'])
  if not p.is_file() or p.stat().st_size!=c['size_bytes']: raise RuntimeError('missing/changed selected file '+str(p))
  if resolved(p) in protected: raise RuntimeError('selected file is current-DAG referenced '+str(p))
  if len(c['sha256'])!=64: raise RuntimeError('invalid state SHA '+str(p))
 if len(selected)!=EXPECTED_COUNT or sum(x['size_bytes'] for x in selected)!=EXPECTED_BYTES:
  raise RuntimeError(f'selection mismatch {len(selected)} files / {sum(x["size_bytes"] for x in selected)} bytes')
 return {'schema':'numi-human.quality695-selection-retirement.v1','purpose':'archive selected historical unadmitted trial artifacts; not the live checkpoint','current_checkpoint':{'live_path':str(cp),'snapshot_path':str(snapshot),'sha256':cp_sha,'checkpoint_current_state_path':current['path'],'checkpoint_current_state_sha256':current['sha256']},'contemporaneous_best_state':{'path':str(newest),'sha256':'40776e9704c3fce24699537dd0df6d40df2b29685b1429e7ce43963fae5c3f1c'},'protected_state_tips':state_tips,'protected_active_reports':active,'protected_current_state_chain_file_count':len(visited),'protected_current_state_references':[{'path':k,'sha256':h} for k,hs in sorted(protected.items()) for h in sorted(hs)],'consumer_audit':{'scope':'bounded all-quality 624-726 JSON consumer scan completed in preceding read-only selection audit','external_selected_file_consumers':0,'selection_is_limited_to_previously-cleared-40-file-set':True},'selection_rule':{'failed_proposals':'Skin and Fat from trial01 rounds001-012','additional_inputs':'D rounds001-006,008-009,011; O rounds001-007; I inputs excluded','active_roots_excluded':['721w','721y','730']},'selected_file_count':len(selected),'selected_original_bytes':sum(x['size_bytes'] for x in selected),'files':sorted(selected,key=lambda x:x['path']),'retirement_rule':'Do not remove originals until remote asset digest, independent download, full streamed blob verification, and selected-member restore all pass. Preserve every JSON report/state/config and every unselected NPZ.'}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--prepare-only',action='store_true'); ap.add_argument('--build',action='store_true'); a=ap.parse_args()
 PK.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
 plan=preflight()
 if a.prepare_only:
  PLAN.write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
  print(json.dumps({'status':'PASS_PREPARE_ONLY','count':plan['selected_file_count'],'bytes':plan['selected_original_bytes'],'plan':str(PLAN)})); return
 if not a.build: ap.error('select --prepare-only or --build')
 frozen=load(PLAN)
 if frozen!=plan: raise RuntimeError('frozen selection plan changed')
 files=[]; blobs={}
 for c in plan['files']:
  got=sha(c['path'])
  if got!=c['sha256']: raise RuntimeError('exact state SHA mismatch '+c['path'])
  row=dict(c); row['source_relative_path']=str(Path(c['path']).resolve().relative_to(H.resolve())); row['archive_member']='blobs/'+got
  files.append(row); blobs[got]=c['size_bytes']
 index={'schema':'numi-human.quality695-failed-trials-archive.v1','snapshot':{'name':'all-quality-695 trial01 failed proposals and cleared input snapshots','not_live_checkpoint':True,'physical_admission':False,'native_steps':0,'status':STATUS},'current_checkpoint':plan['current_checkpoint'],'contemporaneous_best_state':plan['contemporaneous_best_state'],'protected_state_tips':plan['protected_state_tips'],'protected_active_reports':plan['protected_active_reports'],'protected_current_state_references':plan['protected_current_state_references'],'consumer_audit':plan['consumer_audit'],'external_dependencies':[{'path':plan['current_checkpoint']['snapshot_path'],'sha256':plan['current_checkpoint']['sha256'],'reason':'checkpoint snapshot is included as archive metadata'}, {'path':plan['contemporaneous_best_state']['path'],'sha256':plan['contemporaneous_best_state']['sha256'],'reason':'current quality state is retained in its active lineage; not duplicated'}],'files':sorted(files,key=lambda x:(x['archive_member'],x['path'])),'unique_blobs':[{'archive_member':'blobs/'+h,'sha256':h,'size_bytes':n} for h,n in sorted(blobs.items())],'unique_blob_count':len(blobs),'unique_blob_bytes':sum(blobs.values()),'file_reference_count':len(files),'selection_and_retirement':'selection-and-retirement.json'}
 INDEX.write_text(json.dumps(index,indent=2,sort_keys=True)+'\n')
 archive=OUT/'quality695-trial01-failed-proposals-001-012.tar.zst'; partial=archive.with_suffix('.tar.zst.partial')
 zstd='/opt/homebrew/bin/zstd'
 if not Path(zstd).is_file(): raise RuntimeError('zstd missing')
 out=partial.open('wb'); proc=subprocess.Popen([zstd,'-T1','-6','-c'],stdin=subprocess.PIPE,stdout=out)
 try:
  with tarfile.open(fileobj=proc.stdin,mode='w|') as tf:
   def add_bytes(name,data):
    ti=tarfile.TarInfo(name); ti.size=len(data); ti.mtime=0; ti.uid=ti.gid=0; ti.uname=ti.gname=''; ti.mode=0o644; tf.addfile(ti,io.BytesIO(data))
   add_bytes('archive-index.json',INDEX.read_bytes()); add_bytes('selection-and-retirement.json',PLAN.read_bytes()); add_bytes('checkpoint-snapshot.json',(PK/'checkpoint-snapshot.json').read_bytes())
   done=set()
   for row in index['files']:
    h=row['sha256']
    if h in done: continue
    done.add(h); ti=tarfile.TarInfo('blobs/'+h); ti.size=row['size_bytes']; ti.mtime=0; ti.uid=ti.gid=0; ti.uname=ti.gname=''; ti.mode=0o644
    with Path(row['path']).open('rb') as f: tf.addfile(ti,f)
  proc.stdin.close(); rc=proc.wait(); out.close()
  if rc: raise RuntimeError(f'zstd error {rc}')
 except BaseException:
  if proc.poll() is None: proc.kill(); proc.wait()
  out.close(); partial.unlink(missing_ok=True); raise
 os.replace(partial,archive)
 receipt={'status':'PASS_LOCAL_STREAMED_ARCHIVE_BUILD','archive_path':str(archive),'archive_sha256':sha(archive),'archive_size_bytes':archive.stat().st_size,'archive_index_sha256':sha(INDEX),'selection_plan_sha256':sha(PLAN),'selected_count':len(files),'unique_blob_count':len(blobs),'original_bytes':sum(x['size_bytes'] for x in plan['files']),'compression':'zstd -T1 -6; streamed tar, no uncompressed staging copy','physical_admission':False,'native_steps':0}
 (OUT/'archive-build-receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n'); print(json.dumps(receipt))
if __name__=='__main__': main()

