from pathlib import Path
import json,subprocess,tarfile,hashlib,shutil,os,time
D=Path(__file__).resolve().parent;S=D.parent;idx=json.loads((D/'archive-index.json').read_text());H=Path(idx['source_root']);start=time.monotonic();free0=shutil.disk_usage(D).free;state={'status':'BUILDING_ARCHIVES_NO_ORIGINAL_DELETED','free_before':free0,'stages':[]};repo=idx['repository'];tag=idx['tag']
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save():
 state.update(elapsed_s=time.monotonic()-start,free_now=shutil.disk_usage(D).free);(D/'execution.json').write_text(json.dumps(state,indent=2)+'\n')
def run(name,args):
 print(name,'begin',flush=True);t=time.monotonic()
 with (D/(name+'.stdout.log')).open('w') as o,(D/(name+'.stderr.log')).open('w') as e:r=subprocess.run(args,stdout=o,stderr=e,timeout=900)
 state['stages'].append({'name':name,'argv':list(map(str,args)),'exit_code':r.returncode,'elapsed_s':time.monotonic()-t});save();print(name,r.returncode,flush=True);assert r.returncode==0
save();groups=[idx['files'][:2],idx['files'][2:3],idx['files'][3:4],idx['files'][4:]];assets=[]
for i,rs in enumerate(groups,1):
 name=f'part-{i:03d}.tar.gz';out=D/name;assert not out.exists()
 with tarfile.open(out,'w:gz',compresslevel=1) as archive:
  for r in rs:
   p=Path(r['path']);assert sha(p)==r['sha256'];archive.add(p,arcname=r['member'],recursive=False)
 assert out.stat().st_size<1_900_000_000;assets.append({'name':name,'sha256':sha(out),'bytes':out.stat().st_size,'members':[r['member'] for r in rs]});print('built',name,out.stat().st_size,flush=True)
idx.update(status='FROZEN_ARCHIVES_ORIGINALS_RETAINED',assets=assets);(D/'archive-index.json').write_text(json.dumps(idx,indent=2)+'\n');index_sha=sha(D/'archive-index.json');(D/'SHA256SUMS').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in assets)+f'{index_sha}  archive-index.json\n');(D/'release-notes.md').write_text('Retention archive for five superseded pre-compensation witness outputs. This is failed/search evidence, not qualified anatomy or physical motion. Current zero-failure Skin/Fat source, current native inputs and accepted evidence remain on SSH Numi. archive-index.json records exact original paths, file hashes and archive member mapping. Restore with authenticated gh release download and standard tar. Originals are retired only after independent download, complete member hash verification, an actual restore test and fresh current-input/process checks.\n')
state['status']='ARCHIVES_BUILT_UPLOAD_STARTED_NO_ORIGINAL_DELETED';save();run('release-create',['gh','release','create',tag,'--repo',repo,'--draft','--target','main','--title','Retained superseded pre-compensation witnesses 2026-11-02','--notes-file',str(D/'release-notes.md')])
for a in assets+[{'name':'archive-index.json'},{'name':'SHA256SUMS'}]:run('upload-'+a['name'],['gh','release','upload',tag,str(D/a['name']),'--repo',repo])
state['status']='UPLOAD_COMPLETE_AWAITING_INDEPENDENT_DOWNLOAD';save();fresh=D/'independent-download';fresh.mkdir(exist_ok=False);run('independent-download',['gh','release','download',tag,'--repo',repo,'--dir',str(fresh)]);assert sha(fresh/'archive-index.json')==index_sha
# Keep archive verification CPU work out of any native GPU stage.
N=S/'native-import-115'
while N.exists() and (N/'install-motion.active.json').exists() and not (N/'install-motion.execution.json').exists() and (N/'install-motion.stderr.log').exists() and 'newton_tail_dispatch_root=' in (N/'install-motion.stderr.log').read_text():
 state['status']='WAITING_FOR_NATIVE_GPU_BEFORE_ARCHIVE_VERIFY';save();time.sleep(5)
expected={r['member']:r for r in idx['files']};verified=set();restore=D/'restore-test';restore.mkdir(exist_ok=False);restore_pin=None
for a in assets:
 p=fresh/a['name'];assert sha(p)==a['sha256']
 with tarfile.open(p,'r:gz') as archive:
  for member in archive:
   assert member.isfile() and member.name in expected and member.name not in verified;r=expected[member.name];assert member.size==r['bytes'];stream=archive.extractfile(member);digest=hashlib.sha256()
   target=restore/'restored-first-witness.csv' if restore_pin is None else None
   out=target.open('xb') if target else None
   try:
    for block in iter(lambda:stream.read(4*1024*1024),b''):
     digest.update(block)
     if out:out.write(block)
   finally:
    if out:out.close()
   assert digest.hexdigest()==r['sha256'];verified.add(member.name)
   if target:assert sha(target)==r['sha256'];restore_pin={'source':r['path'],'sha256':sha(target),'bytes':target.stat().st_size};target.unlink()
assert verified==set(expected);restore.rmdir();(D/'restore-verification.json').write_text(json.dumps({'status':'PASS','independently_downloaded_assets':assets,'downloaded_index_sha256':index_sha,'all_member_hashes_verified':len(verified),'actual_restore_test':restore_pin},indent=2)+'\n');state['status']='INDEPENDENT_DOWNLOAD_AND_RESTORE_PASS_NO_ORIGINAL_DELETED';save()
plan=json.loads((S/'native-import-106/install-motion.plan.json').read_text())
for kind in ['input_pins','role_input_pins']:
 for raw,h in plan[kind].items():assert sha(raw)==h,raw
retired=[]
for r in idx['files']:
 p=Path(r['path']);s=p.stat();assert (s.st_ino,s.st_size,s.st_mtime_ns)==(r['inode'],r['bytes'],r['mtime_ns']) and sha(p)==r['sha256'] and s.st_nlink==1
 assert not subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True).stdout.strip()
 retired.append(r);(D/'retirement-intent.json').write_text(json.dumps({'verified_remote_index_sha256':index_sha,'files':retired},indent=2)+'\n');p.unlink()
state.update(status='PASS_REMOTE_RESTORE_VERIFIED_SUPERSEDED_ORIGINALS_RETIRED',retired_files=len(retired),retired_original_allocated_bytes=sum(r['allocated_bytes'] for r in retired),recovery_tag=tag,repository=repo)
for a in assets:(D/a['name']).unlink()
for p in fresh.iterdir():p.unlink()
fresh.rmdir();state['local_archive_and_download_scratch_removed']=True;save();print(json.dumps(state),flush=True)
