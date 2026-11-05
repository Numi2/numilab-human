"""Deduplicate only closed, byte-identical preparation packs; preserve every path."""
from pathlib import Path
import collections,hashlib,json,os,shutil,subprocess,stat,time
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
O=W/'preparation-pack-dedup-366';O.mkdir(exist_ok=False)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
processes=subprocess.check_output(['ps','-axo','command='],text=True)
assert not any('numi-human-native ' in x and ' -c ' not in x for x in processes.splitlines())
files=sorted(p for p in W.glob('*/install-motion/resting-human.mrvpack') if p.is_file() and not p.is_symlink())
groups=collections.defaultdict(list);cache={};inventory=[]
for p in files:
 s=p.stat();key=(s.st_dev,s.st_ino)
 if key not in cache:cache[key]=sha(p)
 h=cache[key];groups[(s.st_dev,s.st_size,h)].append(p)
 inventory.append({'path':str(p),'inode':s.st_ino,'bytes':s.st_size,'sha256':h})
(O/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
before=shutil.disk_usage(W).free;changed=[];skipped=[]
for key,paths in groups.items():
 source=paths[0];ss=source.stat()
 for target in paths[1:]:
  ts=target.stat()
  if ts.st_ino==ss.st_ino:continue
  if (stat.S_IMODE(ts.st_mode),ts.st_uid,ts.st_gid)!=(stat.S_IMODE(ss.st_mode),ss.st_uid,ss.st_gid):
   skipped.append({'path':str(target),'reason':'ownership_or_mode_differs'});continue
  assert subprocess.run(['/usr/sbin/lsof','-t','--',str(source),str(target)],capture_output=True).returncode==1
  assert sha(target)==key[2] and sha(source)==key[2]
  temp=target.with_name(target.name+'.dedup366')
  assert not temp.exists()
  os.link(source,temp);os.replace(temp,target)
  assert target.stat().st_ino==ss.st_ino and sha(target)==key[2]
  changed.append({'path':str(target),'source':str(source),'old_inode':ts.st_ino,'new_inode':ss.st_ino,'sha256':key[2],'bytes':key[1]})
os.sync()
r={'status':'PASS_CLOSED_EXACT_PREPARATION_PACK_PATHS_PRESERVED','changed':changed,'skipped':skipped,'file_count':len(files),'unique_inodes_hashed':len(cache),'free_before_bytes':before,'free_after_bytes':shutil.disk_usage(W).free,'note':'Actual df recovery only; do not claim logical sum as physical recovery.'}
(O/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print('PACK_DEDUP_PASS',len(changed),'paths','actual_recovered_MiB',(r['free_after_bytes']-before)/2**20)
