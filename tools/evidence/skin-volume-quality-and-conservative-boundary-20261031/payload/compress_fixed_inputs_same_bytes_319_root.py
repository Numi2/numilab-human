from pathlib import Path
import hashlib,subprocess,json,os,time
H=Path(__file__).resolve().parent;D=H.parent/'fractional-semantic-support-090-sparse-graph';out=H/'byte-preserving-fixed-input-compression-319-root';out.mkdir(exist_ok=False)
paths=[]
for d in [D/'fat-853406-carrier-run-009-partial-flow-export/partial-flow-008',D/'skin-743497-carrier-run-014-partial-flow-root/partial-flow-skin-014']:
 paths.extend(sorted(d.glob('*.csv')))
paths.extend(H/'combined-child-handoff-142'/n for n in ['skin.node','skin.ele','fat.node','fat.ele'])
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def avail():
 s=os.statvfs(H);return s.f_bavail*s.f_frsize
def opened(p):return bool(subprocess.run(['/usr/sbin/lsof','-t','--',str(p)],capture_output=True,text=True).stdout.strip())
report={'status':'BYTE_IDENTICAL_LOSSLESS_STORAGE_BATCH','source_or_evidence_contents_deleted':False,'available_bytes_before':avail(),'files':[]}
for i,p in enumerate(paths):
 s=p.stat()
 if p.is_symlink() or s.st_size<2_000_000 or s.st_size>536870912 or s.st_nlink!=1 or getattr(s,'st_flags',0)&32:continue
 row={'path':str(p),'logical_bytes':s.st_size,'allocated_bytes_before':s.st_blocks*512};report['files'].append(row)
 if opened(p):row['status']='SKIP_CURRENTLY_OPEN';continue
 if time.time()-s.st_mtime<300:row['status']='SKIP_RECENTLY_CHANGED';continue
 pin=sha(p);row['sha256']=pin;tmp=p.with_name(p.name+'.samebytes319.tmp')
 if tmp.exists():raise RuntimeError('stagingpathalreadyexists')
 r=subprocess.run(['/usr/bin/ditto','--hfsCompression','--noclone',str(p),str(tmp)],capture_output=True,text=True)
 row['ditto_returncode']=r.returncode
 if r.returncode:raise RuntimeError('copy failed original retained')
 ns=tmp.stat()
 if (ns.st_size,ns.st_uid,ns.st_gid,ns.st_mode,ns.st_mtime_ns)!=(s.st_size,s.st_uid,s.st_gid,s.st_mode,s.st_mtime_ns) or sha(tmp)!=pin:raise RuntimeError('stagedbyteormetadatamismatch originalretained')
 now=p.stat()
 if (now.st_ino,now.st_size,now.st_mtime_ns)!=(s.st_ino,s.st_size,s.st_mtime_ns) or opened(p):row['status']='SOURCE_CHANGED_OR_OPEN_ORIGINAL_RETAINED';tmp.unlink();continue
 if not getattr(ns,'st_flags',0)&32 or ns.st_blocks>=s.st_blocks:row['status']='NO_GAIN_ORIGINAL_RETAINED';tmp.unlink();continue
 os.replace(tmp,p)
 if sha(p)!=pin:raise RuntimeError('postreplacebytes')
 row.update(status='PASS_BYTE_IDENTICAL_TRANSPARENT_COMPRESSION',allocated_bytes_after=p.stat().st_blocks*512)
 report['available_bytes_current']=avail();(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
report['available_bytes_after']=avail();report['observed_available_delta_bytes']=report['available_bytes_after']-report['available_bytes_before']
report['measured_allocated_bytes_saved']=sum(x['allocated_bytes_before']-x['allocated_bytes_after'] for x in report['files'] if 'allocated_bytes_after' in x)
report['script_sha256']=sha(Path(__file__));(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='files'}))
print(json.dumps([{'path':x['path'],'status':x['status']} for x in report['files']]))
