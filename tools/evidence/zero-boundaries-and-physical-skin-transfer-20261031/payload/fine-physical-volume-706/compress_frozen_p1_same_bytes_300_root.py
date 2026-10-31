from pathlib import Path
import os,subprocess,json,hashlib,time
H=Path(__file__).resolve().parent
out=H/'byte-preserving-p1-compression-300-root';out.mkdir(exist_ok=False)
p=H/'combined-child-handoff-142/semantic-labels-171-vectorized-full-export/p1-anchor-projection.csv'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
s=p.stat();before=os.statvfs(p.parent)
if p.is_symlink() or s.st_nlink!=1 or getattr(s,'st_flags',0)&32:raise RuntimeError('target is linked or already compressed')
opened=subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True)
if opened.stdout.strip():raise RuntimeError('target currently open; skipped')
h=sha(p);tmp=p.with_name(p.name+'.hfscompress-300.tmp')
if tmp.exists():raise RuntimeError('staging path exists')
tic=time.monotonic();r=subprocess.run(['/usr/bin/ditto','--hfsCompression',str(p),str(tmp)],capture_output=True,text=True)
(out/'ditto.stdout.txt').write_text(r.stdout);(out/'ditto.stderr.txt').write_text(r.stderr)
if r.returncode:raise RuntimeError('ditto failed')
n=tmp.stat()
if n.st_size!=s.st_size or sha(tmp)!=h:raise RuntimeError('byte identity failed; original preserved')
now=p.stat()
if (now.st_ino,now.st_size,now.st_mtime_ns)!=(s.st_ino,s.st_size,s.st_mtime_ns):raise RuntimeError('source changed; original preserved')
apply=getattr(n,'st_flags',0)&32 and n.st_blocks<s.st_blocks
if apply:os.replace(tmp,p)
else:tmp.unlink()
after=os.statvfs(p.parent);final=p.stat()
if sha(p)!=h:raise RuntimeError('postreplace identity failed')
receipt={'status':'PASS_BYTE_IDENTICAL_TRANSPARENT_COMPRESSION' if apply else 'NO_COMPRESSION_GAIN_ORIGINAL_RETAINED',
 'path':str(p),'sha256':h,'logical_bytes':s.st_size,'source_inode_links':s.st_nlink,
 'blocks_bytes_before':s.st_blocks*512,'blocks_bytes_after':final.st_blocks*512,'compressed_flag_after':bool(getattr(final,'st_flags',0)&32),
 'available_bytes_before':before.f_bavail*before.f_frsize,'available_bytes_after':after.f_bavail*after.f_frsize,
 'observed_available_change_bytes':after.f_bavail*after.f_frsize-before.f_bavail*before.f_frsize,
 'elapsed_s':time.monotonic()-tic,'sources_or_evidence_deleted':False,'content_identity_before_staged_and_after':True}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
