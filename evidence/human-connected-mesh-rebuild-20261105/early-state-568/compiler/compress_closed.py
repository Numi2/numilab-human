from pathlib import Path
import os,json,hashlib,subprocess,shutil,stat
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=W/'early-state-screen-568/proposal/compiler';O=W/'early570-compiler-compression';O.mkdir(exist_ok=False)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
paths=[p for p in D.rglob('*') if p.is_file() and not p.is_symlink() and p.stat().st_size>2**20 and p.suffix not in ['.gz']]
receipt={'physical_inputs_changed':False,'files':[],'skipped':[],'free_before_bytes':shutil.disk_usage(W).free}
for p in dict.fromkeys(paths):
 if not p.is_file():receipt['skipped'].append({'path':str(p),'reason':'not emitted'});continue
 st=p.stat()
 if st.st_nlink!=1 or not stat.S_ISREG(st.st_mode) or st.st_blocks*512<st.st_size*.75:
  receipt['skipped'].append({'path':str(p),'reason':'already shared or compressed'});continue
 assert p.is_relative_to(D)
 r=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True)
 
 if r.stdout.strip():receipt['skipped'].append({'path':str(p),'reason':'open file; preserved'});continue
 before=sha(p);tmp=p.with_name(p.name+'.hfs-compress-tmp');assert not tmp.exists()
 subprocess.run(['/usr/bin/ditto','--hfsCompression','--noclone',str(p),str(tmp)],check=True,capture_output=True)
 assert sha(tmp)==before and tmp.stat().st_size==st.st_size
 assert p.stat().st_ino==st.st_ino and p.stat().st_mtime_ns==st.st_mtime_ns and sha(p)==before
 ns=tmp.stat()
 if ns.st_blocks>=st.st_blocks:
  tmp.unlink();receipt['skipped'].append({'path':str(p),'reason':'no allocated savings'});continue
 os.chmod(tmp,stat.S_IMODE(st.st_mode));os.utime(tmp,ns=(st.st_atime_ns,st.st_mtime_ns));os.replace(tmp,p)
 assert sha(p)==before and stat.S_IMODE(p.stat().st_mode)==stat.S_IMODE(st.st_mode)
 receipt['files'].append({'path':str(p),'sha256':before,'bytes':st.st_size,'old_allocated_bytes':st.st_blocks*512,'new_allocated_bytes':p.stat().st_blocks*512})
 (O/'progress.json').write_text(json.dumps(receipt,indent=2)+'\n')
receipt['free_after_bytes']=shutil.disk_usage(W).free;receipt['status']='PASS_EXACT_BYTES_PATHS_MODES_PRESERVED';(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'files':len(receipt['files']),'saved_allocated_bytes':sum(x['old_allocated_bytes']-x['new_allocated_bytes'] for x in receipt['files']),'free_bytes':receipt['free_after_bytes']}),flush=True)
