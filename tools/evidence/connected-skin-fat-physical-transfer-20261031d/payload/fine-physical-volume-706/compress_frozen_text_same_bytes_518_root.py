"""Lossless transparent compression of frozen task-owned text, reusing identical bytes."""
from pathlib import Path
import hashlib,subprocess,json,os,time,collections,stat
H=Path(__file__).resolve().parent
out=H/'lossless-frozen-text-storage-518-root';out.mkdir(exist_ok=False)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def avail():
 s=os.statvfs(H);return s.f_bavail*s.f_frsize
def opened(p):return bool(subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True).stdout.strip())
paths=[Path(e['path']) for e in json.loads((H/'compressible-frozen-text-census-517-root.json').read_text())['files']]
start=time.monotonic();before=avail();groups=collections.defaultdict(list);rows=[]
for p in sorted(paths):
 if opened(p):rows.append({'path':str(p),'status':'SKIP_OPEN'});continue
 groups[(p.stat().st_size,sha(p))].append(p)
r={'status':'RUNNING_LOSSLESS_FROZEN_TEXT_COMPRESSION','source_or_evidence_contents_deleted':False,'available_bytes_before':before,'files':rows,'producer_sha256':sha(Path(__file__)),'original_paths_retained':True}
def save():
 r['available_bytes_current']=avail();(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
for (size,pin),ps in sorted(groups.items(),key=lambda x:-x[0][0]*len(x[1])):
 if time.monotonic()-start>100 or avail()>=14*1024**3:break
 compressed=None
 for p in ps:
  s=p.stat();entry={'path':str(p),'sha256':pin,'size_bytes':size,'allocated_before':s.st_blocks*512};rows.append(entry)
  if opened(p) or time.time()-s.st_mtime<600:entry['status']='SKIP_OPEN_OR_CHANGED';continue
  tmp=p.with_name(p.name+'.samebytes518.tmp');assert not tmp.exists()
  if compressed is None:
   cmd=['/usr/bin/ditto','--hfsCompression','--noclone',str(p),str(tmp)]
  else:cmd=['/bin/cp','-c','-p',str(compressed),str(tmp)]
  rr=subprocess.run(cmd,capture_output=True,text=True)
  if rr.returncode:entry['status']='COPY_FAILED_ORIGINAL_RETAINED';entry['error']=rr.stderr[:400];save();continue
  if compressed is not None:
   os.chmod(tmp,s.st_mode);os.utime(tmp,ns=(s.st_atime_ns,s.st_mtime_ns))
  ns=tmp.stat()
  assert (ns.st_size,ns.st_uid,ns.st_gid,ns.st_mode,ns.st_mtime_ns)==(s.st_size,s.st_uid,s.st_gid,s.st_mode,s.st_mtime_ns)
  assert sha(tmp)==pin
  now=p.stat()
  if (now.st_ino,now.st_size,now.st_mtime_ns)!=(s.st_ino,s.st_size,s.st_mtime_ns) or opened(p):
   entry['status']='CHANGED_DURING_COPY_ORIGINAL_RETAINED';tmp.unlink();save();continue
  if not getattr(ns,'st_flags',0)&32 or ns.st_blocks>=s.st_blocks:
   entry['status']='NO_STORAGE_GAIN_ORIGINAL_RETAINED';tmp.unlink();save();continue
  os.replace(tmp,p);assert sha(p)==pin
  entry.update(status='PASS_SAME_BYTES_SAME_METADATA_TRANSPARENT_COMPRESSION',allocated_after=p.stat().st_blocks*512,clone_source=str(compressed) if compressed else None)
  compressed=p;save()
r.update(status='PASS_LOSSLESS_FROZEN_TEXT_STORAGE_BATCH',available_bytes_after=avail(),observed_available_delta_bytes=avail()-before,elapsed_s=time.monotonic()-start)
save();print(json.dumps({k:v for k,v in r.items() if k!='files'}),flush=True)

