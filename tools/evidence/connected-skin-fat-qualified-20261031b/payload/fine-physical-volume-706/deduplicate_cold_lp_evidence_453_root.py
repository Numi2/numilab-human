"""Cold exact-byte evidence deduplication through APFS COW; no content/path deletion."""
from pathlib import Path
import os,stat,subprocess,json,hashlib,time,collections
H=Path(__file__).resolve().parent
ROOT=H.parent/'fractional-semantic-support-090-sparse-graph/fat-residual-source-site-lp-001'
OUT=H/'cold-lp-verified-COW-453-root'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def attrs(p):
 return {name:subprocess.check_output(['/usr/bin/xattr','-px',name,str(p)],text=True) for name in subprocess.check_output(['/usr/bin/xattr',str(p)],text=True).splitlines()}
def opened(p):
 return bool(subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True).stdout.strip())
def free():
 s=os.statvfs(H);return s.f_bavail*s.f_frsize
OUT.mkdir(exist_ok=False);start=time.monotonic();before=free();sizes=collections.defaultdict(list)
for b,ds,fs in os.walk(ROOT):
 for name in fs:
  p=Path(b)/name
  if p.suffix!='.npz':continue
  s=p.lstat()
  if stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size>=8_000_000 and time.time()-s.st_mtime>=3600:sizes[s.st_size].append(p)
groups=collections.defaultdict(list);records=[]
for size,paths in sorted(sizes.items(),reverse=True):
 if len(paths)<2:continue
 for p in paths:
  if opened(p):continue
  groups[(size,sha(p))].append(p)
for (size,pin),paths in sorted(groups.items(),key=lambda x:-x[0][0]*len(x[1])):
 if len(paths)<2:continue
 src=paths[0];ss=src.stat();sx=attrs(src)
 for dst in paths[1:]:
  old=dst.stat();entry={'source':str(src),'copy':str(dst),'sha256':pin,'bytes':size};records.append(entry)
  if opened(src) or opened(dst):entry['status']='SKIP_OPEN';continue
  if attrs(dst)!=sx or ss.st_flags!=old.st_flags:entry['status']='SKIP_DISTINCT_XATTRS';continue
  assert sha(src)==pin and sha(dst)==pin
  tmp=dst.with_name(dst.name+'.cow453-temporary');assert not tmp.exists()
  subprocess.run(['/bin/cp','-c','-p',str(src),str(tmp)],check=True)
  os.chmod(tmp,old.st_mode);os.utime(tmp,ns=(old.st_atime_ns,old.st_mtime_ns))
  ns=tmp.stat()
  assert (ns.st_uid,ns.st_gid,ns.st_mode,ns.st_size,ns.st_mtime_ns,ns.st_flags)==(old.st_uid,old.st_gid,old.st_mode,old.st_size,old.st_mtime_ns,old.st_flags)
  assert attrs(tmp)==sx and sha(tmp)==pin and sha(src)==pin
  fresh=dst.stat()
  if (fresh.st_ino,fresh.st_size,fresh.st_mtime_ns)!=(old.st_ino,old.st_size,old.st_mtime_ns) or opened(dst):
   tmp.unlink();entry['status']='SKIP_CHANGED_BEFORE_REPLACE';continue
  os.replace(tmp,dst);assert sha(dst)==pin
  entry['status']='PASS_SAME_BYTES_AND_METADATA_INDEPENDENT_INODE_COW'
r={'status':'PASS_COLD_LP_EXACT_BYTE_COW_DEDUP','physical_admission':False,'native_steps':0,'root':str(ROOT),'available_before_bytes':before,'available_after_bytes':free(),'observed_available_delta_bytes':free()-before,'records':records,'content_and_paths_preserved':True,'producer_sha256':sha(__file__),'elapsed_s':time.monotonic()-start}
(OUT/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='records'}),flush=True);print('records',len(records))
