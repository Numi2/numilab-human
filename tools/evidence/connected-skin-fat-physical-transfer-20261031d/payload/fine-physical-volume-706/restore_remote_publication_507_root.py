
"""Independently restore remote publication blobs before reversible sparse retirement."""
from pathlib import Path
import json,subprocess,hashlib,time,io,zipfile
H=Path(__file__).resolve().parent;PUB=Path('/Users/numi/Work/human-volume-proof-publication-20261031')
REV='aed0a821ab90c7fc24679d2fdf8157c289b627d8'
OUT=H/'remote-publication-restore-507-root';OUT.mkdir(exist_ok=False)
G=OUT/'independent.git';started=time.monotonic()
def cmd(a,**kw):return subprocess.run(a,check=True,capture_output=True,**kw)
def git(*a,**kw):return cmd(['git','--git-dir='+str(G),*a],**kw)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert cmd(['git','-C',str(PUB),'rev-parse','HEAD'],text=True).stdout.strip()==REV
assert not cmd(['git','-C',str(PUB),'status','--porcelain'],text=True).stdout.strip()
raw=cmd(['git','-C',str(PUB),'ls-tree','-r','-z',REV,'tools/evidence']).stdout
items=[]
for rec in raw.split(b'\0'):
 if not rec:continue
 meta,pb=rec.split(b'\t',1);mode,typ,oid=meta.decode().split();rel=pb.decode();p=PUB/rel
 if not p.exists():continue
 assert p.is_file() and not p.is_symlink() and typ=='blob' and mode=='100644',rel
 items.append({'path':rel,'oid':oid,'sha256':sha(p),'bytes':p.stat().st_size})
assert len(items)>10000
(OUT/'local-source-manifest.json').write_text(json.dumps({'revision':REV,'files':items},indent=2)+'\n')
git_init=cmd(['git','init','--bare',str(G)],text=True)
git('remote','add','origin','https://github.com/Numi2/numilab-human.git')
git('config','remote.origin.promisor','true')
git('config','remote.origin.partialclonefilter','blob:none')
git('fetch','--no-tags','--depth=1','--filter=blob:none','origin',REV)
assert git('rev-parse','FETCH_HEAD',text=True).stdout.strip()==REV
assert not (G/'objects/info/alternates').exists()
oids=sorted({e['oid'] for e in items})
for i in range(0,len(oids),512):
 batch=oids[i:i+512]
 git('-c','fetch.negotiationAlgorithm=noop','fetch','--no-tags','--no-write-fetch-head','--stdin','origin',input=('\n'.join(batch)+'\n').encode())
 (OUT/'progress.json').write_text(json.dumps({'phase':'remote-fetch','blobs_requested':min(i+512,len(oids)),'unique_blobs':len(oids),'files':len(items),'physical_admission':False})+'\n')
 print(json.dumps({'phase':'remote-fetch','blobs_requested':min(i+512,len(oids)),'unique_blobs':len(oids)}),flush=True)
p=subprocess.Popen(['git','--git-dir='+str(G),'cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
checked=[];zipchecks=0;total=0
for e in items:
 p.stdin.write((e['oid']+'\n').encode());p.stdin.flush()
 line=p.stdout.readline().decode().strip().split()
 assert line==[e['oid'],'blob',str(e['bytes'])],(e['path'],line)
 remaining=e['bytes'];digest=hashlib.sha256();restore=io.BytesIO() if e['path'].endswith('.npz') else None
 while remaining:
  b=p.stdout.read(min(1024*1024,remaining));assert b
  remaining-=len(b);digest.update(b)
  if restore is not None:restore.write(b)
 assert p.stdout.read(1)==b'\n'
 assert digest.hexdigest()==e['sha256'],e['path']
 assert sha(PUB/e['path'])==e['sha256'],('source drift',e['path'])
 if restore is not None:
  restore.seek(0)
  with zipfile.ZipFile(restore) as z:
   assert z.testzip() is None and all(n.endswith('.npy') for n in z.namelist()),e['path']
  zipchecks+=1
 checked.append(e);total+=e['bytes']
p.stdin.close();assert p.wait()==0,p.stderr.read()
receipt={'status':'PASS_INDEPENDENT_GITHUB_FETCH_FULL_BYTE_RESTORE_AND_NPZ_CRC','repository':'Numi2/numilab-human','revision':REV,'files':len(checked),'unique_blobs':len(oids),'bytes':total,'npz_crc_restores':zipchecks,'manifest':str(OUT/'local-source-manifest.json'),'manifest_sha256':sha(OUT/'local-source-manifest.json'),'remote_objects':str(G),'remote_objects_have_no_local_alternates':True,'publication_worktree_unchanged':True,'local_retirement_performed':False,'physical_admission':False,'native_steps':0,'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-started}
assert not cmd(['git','-C',str(PUB),'status','--porcelain'],text=True).stdout.strip()
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
