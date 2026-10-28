from pathlib import Path
import hashlib,json,subprocess,tarfile,io,os,shutil,time
R=Path(__file__).resolve().parent;GH='/opt/homebrew/bin/gh';repo='Numi2/numilab-human';tag='human-hand-207-rejected-trials-20261028';commit='9ccf48b96c7d9a9f6e87a9581cf1a5435b2969df';receipt=R/'recovery-flow-030.json'
def gh(*a,check=True):return subprocess.run([GH,*a],check=check,text=True,capture_output=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
v=gh('release','view',tag,'--repo',repo,check=False)
if v.returncode:
 notes=R/'archive-release-notes.txt';notes.write_text('Losslessly retained rejected offline Human hand-skin authoring trials. These trials are not admitted anatomy or native physics. Source and zero-crossing diagnostic are retained at commit '+commit+'. Each archive contains a SHA-256 per-file manifest. The session retirement receipt records an independent GitHub download, full per-file byte checks, restoration of a proposal NPZ, and a fresh source recheck before deleting only failed query subdirectories.\n')
 gh('release','create',tag,'--repo',repo,'--target',commit,'--title','Rejected Human hand-skin trials: 207','--notes-file',str(notes),'--prerelease','--latest=false')
records=[]
for name in ['R/ambient-flow-030/trial-001','R/ambient-flow-030/trial-001-continuation']:
 root=R/name;files=[p for p in sorted(root.rglob('*'))if p.is_file() and not p.is_symlink()]
 queries=[p for p in sorted(root.rglob('query'))if p.is_dir()]
 assert files and queries and root.is_relative_to(R)
 manifest={str(p.relative_to(R)):{'sha256':sha(p),'bytes':p.stat().st_size}for p in files};payload=json.dumps(manifest,sort_keys=True,indent=2).encode();archive=R/('human-hand-207-'+name.replace('/','--')+'.tar.gz')
 with tarfile.open(archive,'w:gz',compresslevel=1)as tf:
  ti=tarfile.TarInfo('MANIFEST.json');ti.size=len(payload);ti.mtime=0;tf.addfile(ti,io.BytesIO(payload))
  for p in files:tf.add(p,arcname=str(p.relative_to(R)),recursive=False)
 ah=sha(archive);print(json.dumps({'stage':'archive','run':name,'bytes':archive.stat().st_size,'sha256':ah}),flush=True)
 gh('release','upload',tag,str(archive),'--repo',repo)
 release=json.loads(gh('api','repos/'+repo+'/releases/tags/'+tag).stdout);asset=next(a for a in release['assets']if a['name']==archive.name)
 proc=subprocess.Popen([GH,'api','repos/'+repo+'/releases/assets/'+str(asset['id']),'-H','Accept: application/octet-stream'],stdout=subprocess.PIPE,stderr=subprocess.PIPE);digest=hashlib.sha256()
 class Reader:
  def read(self,n=-1):
   b=proc.stdout.read(n);digest.update(b);return b
 reader=Reader();seen=set();restore=None
 with tarfile.open(fileobj=reader,mode='r|gz')as tf:
  for member in tf:
   assert member.isfile();stream=tf.extractfile(member);h=hashlib.sha256();capture=member.name=='MANIFEST.json' or (restore is None and member.name.endswith('.npz'));buf=[]
   for b in iter(lambda:stream.read(1<<20),b''):
    h.update(b)
    if capture:buf.append(b)
   if member.name=='MANIFEST.json':assert json.loads(b''.join(buf))==manifest
   else:
    assert member.name in manifest and h.hexdigest()==manifest[member.name]['sha256'] and member.size==manifest[member.name]['bytes'];seen.add(member.name)
    if capture:restore=(member.name,b''.join(buf))
 while reader.read(1<<20):pass
 assert proc.wait()==0 and digest.hexdigest()==ah and seen==set(manifest)
 assert restore is not None
 restored=R/'restoration-test.npz';restored.write_bytes(restore[1]);assert sha(restored)==manifest[restore[0]]['sha256']
 import numpy as np
 with np.load(restored,allow_pickle=False)as z:assert len(z.files)>0
 restored.unlink()
 # Independent GitHub bytes and restored proposal are verified. Recheck local inputs and live consumers.
 assert all(p.is_file()and sha(p)==manifest[str(p.relative_to(R))]['sha256']for p in files)
 retired=[];freed=0
 for q in queries:
  use=subprocess.run(['/usr/sbin/lsof','+D',str(q)],capture_output=True)
  if use.returncode!=1 or use.stdout:continue
  assert q.name=='query' and q.is_relative_to(root)
  count=sum(p.stat().st_size for p in q.rglob('*')if p.is_file());shutil.rmtree(q);freed+=count;retired.append(str(q.relative_to(R)))
 archive.unlink();row={'run':name,'asset_id':asset['id'],'url':asset['browser_download_url'],'archive_sha256':ah,'archive_bytes':asset['size'],'file_manifest':manifest,'independent_download_verified':True,'restored_npz_verified':restore[0],'retired_query_directories':retired,'retired_bytes':freed};records.append(row);receipt.write_text(json.dumps({'release':release['html_url'],'commit':commit,'records':records},indent=2)+'\n');print(json.dumps({'stage':'verified_and_retired','run':name,'freed_bytes':freed,'retired_directories':len(retired)}),flush=True)
gh('release','upload',tag,str(receipt),'--repo',repo);print(json.dumps({'stage':'complete','release':release['html_url'],'freed_bytes':sum(a['retired_bytes']for a in records)}),flush=True)
