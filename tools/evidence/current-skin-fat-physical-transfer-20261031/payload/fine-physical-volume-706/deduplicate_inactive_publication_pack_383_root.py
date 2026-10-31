"""Retire one inactive publication pack only after all objects are independently recoverable.
The repository already borrows the Human object store. Preserve its unique objects
in a verified self-contained replacement pack; retain every ref and alternate.
"""
from pathlib import Path
import os,json,subprocess,hashlib,time
H=Path(__file__).resolve().parent;P=H.parent
OLD=P/'exact-lattice-repair-publication-605/repo.git'
ALT=Path('/Users/numi/numilab-human/.git/objects')
STEM='pack-b40dc87bcf16ccfab09ba894967746b17c06740b'
PACK=OLD/'objects/pack'/(STEM+'.pack');IDX=PACK.with_suffix('.idx')
OUT=H/'inactive-publication-pack-dedup-383-root';OUT.mkdir(exist_ok=False)
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def free():
 z=os.statvfs(H);return z.f_bavail*z.f_frsize
def run(args,**kw):return subprocess.run(args,check=True,capture_output=True,**kw)
env=os.environ.copy();env.update(GIT_NO_LAZY_FETCH='1',GIT_TERMINAL_PROMPT='0')
start=time.monotonic();before=free()
assert OLD.is_dir() and ALT.is_dir()
alts=(OLD/'objects/info/alternates').read_text();assert alts.strip()==str(ALT)
refs=run(['git','--git-dir='+str(OLD),'show-ref'],text=True).stdout
wt=run(['git','--git-dir='+str(OLD),'worktree','list','--porcelain'],text=True).stdout
assert wt.strip()=='worktree '+str(OLD)+'\nbare'
proc=subprocess.run(['/usr/sbin/lsof','-t',str(PACK),str(IDX)],capture_output=True,text=True)
assert proc.returncode==1 and not proc.stdout.strip(),'pack is in use'
rows=run(['git','show-index'],input=IDX.read_bytes()).stdout.decode().splitlines()
ids=[x.split()[1] for x in rows];assert len(ids)==11204 and len(set(ids))==len(ids)
alt_env=dict(env,GIT_OBJECT_DIRECTORY=str(ALT))
checks=run(['git','--git-dir='+str(OLD),'cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input='\n'.join(ids)+'\n',text=True,env=alt_env).stdout.splitlines()
missing=[x.split()[0] for x in checks if x.endswith(' missing')]
shared=[x.split()[0] for x in checks if not x.endswith(' missing')]
assert len(missing)==1017 and len(shared)==10187
# Read every shared object's actual bytes from the existing alternate and verify
# its content hash; a header-only presence check is insufficient for retirement.
# Feed a frozen ID list to avoid pipe input/output deadlock.
input_ids=OUT/'shared-object-ids.txt';input_ids.write_text('\n'.join(shared)+'\n')
total=0
with input_ids.open('rb') as feed:
 cp=subprocess.Popen(['git','--git-dir='+str(OLD),'cat-file','--batch'],stdin=feed,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=alt_env)
 for oid in shared:
  header=cp.stdout.readline().decode().strip().split();assert header[0]==oid and len(header)==3 and header[2].isdigit()
  kind=header[1];size=int(header[2]);hh=hashlib.sha1((kind+' '+str(size)+'\0').encode())
  remain=size
  while remain:
   data=cp.stdout.read(min(remain,4*1024**2));assert data;hh.update(data);remain-=len(data)
  assert cp.stdout.read(1)==b'\n' and hh.hexdigest()==oid
  total+=size
 assert cp.stdout.read()==b''
 assert cp.wait()==0
# New pack contains all unique objects, no thin external bases.
r=run(['git','--git-dir='+str(OLD),'pack-objects','--delta-base-offset','--threads=1','--window-memory=64m',str(OLD/'objects/pack/pack')],input='\n'.join(missing)+'\n',text=True,env=env)
newstem='pack-'+r.stdout.strip();newpack=OLD/'objects/pack'/(newstem+'.pack');newidx=newpack.with_suffix('.idx')
assert newpack.exists() and newidx.exists() and newpack!=PACK
vp=run(['git','verify-pack','-v',str(newidx)],text=True,env=env).stdout
(OUT/'replacement-verify-pack.txt').write_text(vp)
newids={x.split()[0] for x in vp.splitlines() if len(x.split())>=5 and x.split()[1] in ('blob','tree','commit','tag')}
assert set(missing)==newids
# A temporary isolated object view proves the old pack is unnecessary before deletion.
view=OUT/'verification-objects';(view/'pack').mkdir(parents=True)
os.symlink(newpack,view/'pack'/newpack.name);os.symlink(newidx,view/'pack'/newidx.name)
viewenv=dict(env,GIT_OBJECT_DIRECTORY=str(view),GIT_ALTERNATE_OBJECT_DIRECTORIES=str(ALT))
after=run(['git','--git-dir='+str(OLD),'cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input='\n'.join(ids)+'\n',text=True,env=viewenv).stdout
assert len(after.splitlines())==len(ids) and not any(x.endswith(' missing') for x in after.splitlines())
assert run(['git','--git-dir='+str(OLD),'show-ref'],text=True).stdout==refs
assert (OLD/'objects/info/alternates').read_text()==alts
oldpins=[{'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in [PACK,IDX]]
(OUT/'before-retirement.json').write_text(json.dumps({'refs':refs,'alternates':alts,'old_pack':oldpins,'verified_shared_objects':len(shared),'verified_shared_content_bytes':total,'unique_objects':len(missing),'replacement_pack':str(newpack),'replacement_sha256':sha(newpack),'all_old_pack_objects_available_without_old_pack':True},indent=2)+'\n')
# Preserve any promisor marker on the replacement; no object content is removed.
promisor=PACK.with_suffix('.promisor')
if promisor.exists():
 newpack.with_suffix('.promisor').write_bytes(promisor.read_bytes())
for p in [PACK,IDX,promisor]:
 if p.exists():p.unlink()
assert run(['git','--git-dir='+str(OLD),'show-ref'],text=True).stdout==refs
fresh=run(['git','--git-dir='+str(OLD),'cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input='\n'.join(ids)+'\n',text=True,env=env).stdout
assert len(fresh.splitlines())==len(ids) and not any(x.endswith(' missing') for x in fresh.splitlines())
receipt={'status':'PASS_VERIFIED_INACTIVE_PACK_DEDUPLICATION','host':'ssh numi','old_pack':oldpins,'existing_alternate_unchanged':str(ALT),'refs_bit_identical':True,'all_old_objects_retained':len(ids),'shared_objects_actual_content_hash_verified':len(shared),'shared_object_content_bytes':total,'unique_objects_preserved_in_new_pack':len(missing),'replacement_pack':str(newpack),'replacement_sha256':sha(newpack),'replacement_bytes':newpack.stat().st_size,'available_before_bytes':before,'available_after_bytes':free(),'available_delta_bytes':free()-before,'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start,'limitations':'No source/evidence payload or ref deleted. Existing alternate dependency is retained; no new dependency added.'}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
