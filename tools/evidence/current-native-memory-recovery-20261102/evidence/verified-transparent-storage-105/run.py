from pathlib import Path
import os,json,hashlib,subprocess,time,shutil
D=Path(__file__).resolve().parent;H=D.parent.parent
rows=json.loads((D/'selection.json').read_text());start=time.monotonic();free0=shutil.disk_usage(D).free;result={'status':'RUNNING','original_paths_and_contents_retained':True,'rows':[],'free_before':free0}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save():
 result['free_after']=shutil.disk_usage(D).free;(D/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
for raw in rows:
 if time.monotonic()-start>240 or shutil.disk_usage(D).free>=6*1024**3:break
 p=Path(raw);s=p.stat()
 if s.st_nlink!=1 or s.st_flags&32 or subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True).stdout.strip():continue
 digest=sha(p);tmp=p.with_name(p.name+'.compression105-tmp');assert not tmp.exists();entry={'path':str(p),'bytes':s.st_size,'sha256':digest,'allocated_before':s.st_blocks*512};result['rows'].append(entry);save()
 cmd=['/usr/bin/ditto','--hfsCompression','--noclone',str(p),str(tmp)];t=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True);entry.update(command=cmd,exit_code=r.returncode,elapsed_s=time.monotonic()-t,stderr=r.stderr)
 if r.returncode:tmp.unlink(missing_ok=True);entry['status']='COPY_FAILED_ORIGINAL_RETAINED';save();break
 ns=tmp.stat();assert sha(tmp)==digest
 assert (ns.st_size,ns.st_uid,ns.st_gid,ns.st_mode,ns.st_mtime_ns)==(s.st_size,s.st_uid,s.st_gid,s.st_mode,s.st_mtime_ns)
 if not ns.st_flags&32 or ns.st_blocks>=s.st_blocks:
  entry.update(status='NO_ALLOCATION_REDUCTION_ORIGINAL_RETAINED',flags=ns.st_flags,allocated_copy=ns.st_blocks*512);tmp.unlink();save();break
 now=p.stat();assert (now.st_ino,now.st_size,now.st_mtime_ns,now.st_nlink)==(s.st_ino,s.st_size,s.st_mtime_ns,1)
 assert not subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True).stdout.strip()
 os.replace(tmp,p);assert sha(p)==digest;entry.update(status='PASS_BYTE_IDENTICAL_TRANSPARENT_COMPRESSION',allocated_after=p.stat().st_blocks*512,flags=p.stat().st_flags);save();print(entry['status'],s.st_size,entry['allocated_after'],p,flush=True)
result.update(status='COMPLETE_SAME_PATHS_SAME_BYTES',elapsed_s=time.monotonic()-start);save();print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
