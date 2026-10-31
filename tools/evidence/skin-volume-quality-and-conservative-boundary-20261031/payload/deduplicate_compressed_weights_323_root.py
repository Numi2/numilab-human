from pathlib import Path
from collections import defaultdict
import hashlib,subprocess,json,os,stat
H=Path(__file__).resolve().parent;out=H/'compressed-identical-weight-cow-323-root';out.mkdir(exist_ok=False)
r=json.loads((H/'byte-preserving-weight-compression-321-root/receipt.json').read_text());groups=defaultdict(list)
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def avail():
 s=os.statvfs(H);return s.f_bavail*s.f_frsize
for x in r['files']:
 if x['status']=='PASS_BYTE_IDENTICAL_TRANSPARENT_COMPRESSION':groups[(x['sha256'],x['logical_bytes'])].append(Path(x['path']))
report={'status':'PASS_COMPRESSED_BYTE_IDENTICAL_COW_DEDUPLICATION','available_bytes_before':avail(),'source_or_evidence_contents_deleted':False,'groups':[]}
for (pin,n),paths in groups.items():
 canonical=paths[0];assert sha(canonical)==pin
 row={'canonical':str(canonical),'sha256':pin,'logical_bytes':n,'replaced':[]};report['groups'].append(row)
 for p in paths[1:]:
  s=p.stat()
  if p.is_symlink() or s.st_nlink!=1 or not getattr(s,'st_flags',0)&32:raise ValueError('target metadata')
  if subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True).stdout.strip():row['replaced'].append({'path':str(p),'status':'SKIP_OPEN'});continue
  assert sha(p)==pin
  tmp=p.with_name(p.name+'.compressed-cow323.tmp');assert not tmp.exists()
  subprocess.run(['/bin/cp','-c','-p',str(canonical),str(tmp)],check=True)
  os.chmod(tmp,stat.S_IMODE(s.st_mode));os.utime(tmp,ns=(s.st_atime_ns,s.st_mtime_ns))
  ns=tmp.stat();assert (ns.st_size,ns.st_uid,ns.st_gid,ns.st_mode,ns.st_mtime_ns)==(s.st_size,s.st_uid,s.st_gid,s.st_mode,s.st_mtime_ns)
  assert getattr(ns,'st_flags',0)&32 and sha(tmp)==pin
  now=p.stat();assert (now.st_ino,now.st_size,now.st_mtime_ns)==(s.st_ino,s.st_size,s.st_mtime_ns)
  if subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True).stdout.strip():tmp.unlink();row['replaced'].append({'path':str(p),'status':'OPEN_DURING_STAGING_ORIGINAL_RETAINED'});continue
  os.replace(tmp,p);assert sha(p)==pin and p.stat().st_nlink==1
  row['replaced'].append({'path':str(p),'status':'PASS_IDENTICAL_COMPRESSED_COW','original_inode':s.st_ino,'new_inode':p.stat().st_ino})
  (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
report['available_bytes_after']=avail();report['actual_available_change_bytes']=report['available_bytes_after']-report['available_bytes_before'];report['script_sha256']=sha(Path(__file__))
(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='groups'}))
