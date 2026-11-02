from pathlib import Path
import os,json,ctypes,hashlib,time,shutil,subprocess,collections,errno,stat
from contextlib import contextmanager
D=Path(__file__).resolve().parent;inv={};lib=ctypes.CDLL(None,use_errno=True);fn=lib.clonefile;fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint32];fn.restype=ctypes.c_int
def clone(a,b):
 if fn(os.fsencode(a),os.fsencode(b),0):raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()),str(b))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def equal(a,b):
 with a.open('rb') as x,b.open('rb') as y:
  while True:
   u=x.read(1024*1024);v=y.read(1024*1024)
   if u!=v:return False
   if not u:return True
def meta(s):return (s.st_dev,s.st_ino,s.st_size,s.st_uid,s.st_gid,s.st_mode,s.st_mtime_ns,s.st_nlink,s.st_flags)
lx=lib.listxattr;lx.argtypes=[ctypes.c_char_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int];lx.restype=ctypes.c_ssize_t
gx=lib.getxattr;gx.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint32,ctypes.c_int];gx.restype=ctypes.c_ssize_t
def attrs(p):
 raw=os.fsencode(p);size=lx(raw,None,0,0)
 if size<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()),str(p))
 if size==0:return {}
 buf=ctypes.create_string_buffer(size);got=lx(raw,buf,size,0);assert got>=0;out={}
 for name in buf.raw[:got].split(b'\0'):
  if not name:continue
  n=gx(raw,name,None,0,0,0)
  if n<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()),str(p))
  value=ctypes.create_string_buffer(n);read=gx(raw,name,value,n,0,0);assert read==n;out[name]=value.raw[:n]
 return out
@contextmanager
def writable_parent(path):
 parent=path.parent;before=parent.stat();oldmode=stat.S_IMODE(before.st_mode);changed=False
 if not os.access(parent,os.W_OK):
  if before.st_uid!=os.getuid() or before.st_flags!=0:
   yield False;return
  try:os.chmod(parent,oldmode|stat.S_IWUSR|stat.S_IXUSR);changed=True
  except PermissionError:
   yield False;return
 try:yield os.access(parent,os.W_OK)
 finally:
  if changed:
   os.chmod(parent,oldmode);after=parent.stat();assert after.st_mode==before.st_mode and after.st_uid==before.st_uid and after.st_gid==before.st_gid

start=time.monotonic();free0=shutil.disk_usage(D).free;plan=json.loads((D.parent/'native-import-106/install-motion.plan.json').read_text());protected={*plan['input_pins'],*plan['role_input_pins'],plan['reused_immutable_preparation_dump']['source']['path']};byhash={};rows=[];journal=(D/'replacements.jsonl').open('x')
for name,sub in [('078','source-preflight'),('085','install-motion'),('092','install-motion'),('096','install-motion'),('099','install-motion'),('106','install-motion')]:
 root=D.parent/f'native-import-{name}'/sub
 for p in root.rglob('*'):
  if not p.is_file() or p.is_symlink() or p.stat().st_size<1024**2 or p.suffix not in ['.csv','.mrvpack','.bin','.txt']:continue
  s=p.stat();h=sha(p)
  if h not in byhash:byhash[h]=p;continue
  src=byhash[h]
  if str(p) in protected or p.stat().st_ino==src.stat().st_ino or s.st_nlink!=1 or attrs(src)!=attrs(p):continue
  assert equal(src,p);temp=p.with_name(p.name+'.cow116-tmp');assert not temp.exists()
  with writable_parent(p) as allowed:
   assert allowed;clone(src,temp);os.chmod(temp,stat.S_IMODE(s.st_mode));os.utime(temp,ns=(s.st_atime_ns,s.st_mtime_ns));assert sha(temp)==h and meta(p.stat())==meta(s);proof={'phase':'VERIFIED_BEFORE_REPLACEMENT','source':str(src),'target':str(p),'sha256':h,'bytes':s.st_size,'allocated_before':s.st_blocks*512,'original_inode':s.st_ino,'target_mtime_ns':s.st_mtime_ns,'target_mode':s.st_mode};journal.write(json.dumps(proof)+'\n');journal.flush();os.replace(temp,p);assert sha(p)==h;proof.update(phase='REPLACED_INDEPENDENT_COW',new_inode=p.stat().st_ino);journal.write(json.dumps(proof)+'\n');journal.flush();rows.append(proof)
journal.close();result={'status':'PASS','replacements':len(rows),'candidate_identical_allocated_bytes':sum(r['allocated_before'] for r in rows),'free_before':free0,'free_after':shutil.disk_usage(D).free,'elapsed_s':time.monotonic()-start,'all_paths_and_bytes_preserved':True};(D/'execution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
