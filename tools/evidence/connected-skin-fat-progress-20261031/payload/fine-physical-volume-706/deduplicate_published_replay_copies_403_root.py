"""Deduplicate only byte-identical completed replay/seed copies by APFS COW cloning."""
from pathlib import Path
import json,hashlib,subprocess,os,time
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def attrs(path):
 names=subprocess.check_output(['/usr/bin/xattr',str(path)],text=True).splitlines()
 return {name:bytes.fromhex(subprocess.check_output(['/usr/bin/xattr','-px',name,str(path)],text=True)) for name in names}
out=H/'verified-duplicate-clones-403-root';out.mkdir(exist_ok=False)
r=json.loads((H/'matched-indexed-interface-386-root/receipt.json').read_text())
old=json.loads(Path(r['reference_report']).read_text());new=json.loads(Path(r['indexed_report']).read_text())
pairs=[(Path(old['candidate_I']),Path(new['candidate_I']))]
pairs.extend((Path(a['output']),Path(b['output'])) for a,b in zip(old['roles'],new['roles']))
pkg=Path('/Users/numi/Work/human-volume-proof-publication-20261031/tools/evidence/current-skin-fat-physical-transfer-20261031')
manifest=json.loads((pkg/'manifest.json').read_text())
by_source={x['source_path']:x for x in manifest['files']}
copies=[]
for src,oldcopy in pairs:
 item=by_source[str(oldcopy)]
 dst=pkg/item['payload']
 assert sha(src)==sha(dst)==item['sha256']
 copies.append((src,dst))
pairs=copies
def free():
 s=os.statvfs(H);return s.f_bavail*s.f_frsize
before=free();records=[];start=time.monotonic()
for src,dst in pairs:
 assert src!=dst and H in src.parents and pkg in dst.parents
 a=src.stat();b=dst.stat();pin=sha(src);got=sha(dst)
 if a.st_size!=b.st_size or pin!=got:
  records.append({'source':str(src),'copy':str(dst),'status':'SKIP_NONIDENTICAL'});continue
 sx=attrs(src)
 dx=attrs(dst)
 if sx!=dx or a.st_flags!=b.st_flags:
  records.append({'source':str(src),'copy':str(dst),'status':'SKIP_DISTINCT_EXTENDED_METADATA'});continue
 tmp=dst.with_name(dst.name+'.cow403-temporary');assert not tmp.exists()
 subprocess.run(['/bin/cp','-c','-p',str(src),str(tmp)],check=True)
 assert sha(tmp)==pin
 os.chmod(tmp,b.st_mode);os.utime(tmp,ns=(b.st_atime_ns,b.st_mtime_ns))
 if tmp.stat().st_uid!=b.st_uid or tmp.stat().st_gid!=b.st_gid:os.chown(tmp,b.st_uid,b.st_gid)
 assert sha(dst)==pin and sha(src)==pin
 os.replace(tmp,dst)
 assert sha(dst)==pin and dst.stat().st_mtime_ns==b.st_mtime_ns and dst.stat().st_size==b.st_size
 records.append({'source':str(src),'copy':str(dst),'status':'PASS_INDEPENDENT_INODE_COW_SAME_BYTES_AND_METADATA','sha256':pin,'size_bytes':b.st_size})
receipt={'status':'PASS_VERIFIED_DUPLICATE_REPLAY_COW_STORAGE_DEDUP','physical_admission':False,'native_steps':0,'records':records,'available_before_bytes':before,'available_after_bytes':free(),'available_delta_bytes':free()-before,'elapsed_s':time.monotonic()-start,'producer_sha256':sha(__file__)}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='records'}),flush=True)
print('cloned',sum(x['status'].startswith('PASS') for x in records),'skipped',sum(x['status'].startswith('SKIP') for x in records),flush=True)
