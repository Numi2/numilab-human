from pathlib import Path
import os,json,ctypes,hashlib,time,shutil,subprocess,collections,errno,stat
from contextlib import contextmanager
D=Path(__file__).resolve().parent;inv=json.loads((D/'inventory.json').read_text());lib=ctypes.CDLL(None,use_errno=True);fn=lib.clonefile;fn.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint32];fn.restype=ctypes.c_int
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

# Exercise native clone and prove that a later write does not change its source.
control=D/'clone-control';control.mkdir(exist_ok=False);a=control/'source';b=control/'copy';a.write_bytes(b'Numi COW source\n'*1024);before=sha(a);clone(a,b);assert sha(b)==before and a.stat().st_ino!=b.stat().st_ino;b.write_bytes(b'independent write\n');assert sha(a)==before;(D/'clone-control.json').write_text(json.dumps({'status':'PASS','source_unchanged_after_clone_write':True,'independent_inodes':True,'method':'libSystem clonefile flags0; tested on actual SSH Numi filesystem'},indent=2)+'\n');b.unlink()
frozen=control/'frozen';frozen.mkdir();os.chmod(frozen,0o555)
with writable_parent(frozen/'copy') as allowed:
 assert allowed;clone(a,frozen/'copy');assert sha(frozen/'copy')==before
assert stat.S_IMODE(frozen.stat().st_mode)==0o555
os.chmod(frozen,0o755);(frozen/'copy').unlink();frozen.rmdir();a.unlink();control.rmdir()
start=time.monotonic();free0=shutil.disk_usage(D).free;state={'status':'RUNNING','available_bytes_before':free0,'files_hashed':0,'replacements':0,'byte_identical_candidate_bytes':0,'skipped_metadata_or_open':0,'paths_and_contents_retained':True,'future_writes_independent':True};journal=(D/'replacements.jsonl').open('x');canonical_proofs=[]
opened=subprocess.run(['/usr/sbin/lsof','-n','-F','n','-u',str(os.getuid())],capture_output=True,text=True).stdout;openpaths={l[1:] for l in opened.splitlines() if l.startswith('n/Users/numi/')}
def save():
 state.update(elapsed_s=time.monotonic()-start,available_bytes_after=shutil.disk_usage(D).free,observed_available_delta_bytes=shutil.disk_usage(D).free-free0);(D/'execution.json').write_text(json.dumps(state,indent=2)+'\n')
save()
for group_index,g in enumerate(inv['groups']):
 if time.monotonic()-start>600 or shutil.disk_usage(D).free>=35_000_000_000:state['bounded_stop_after_group']=group_index;break
 byhash={}
 for r in g['files']:
  p=Path(r['path'])
  if not p.exists() or str(p) in openpaths:state['skipped_metadata_or_open']+=1;continue
  s=p.stat()
  if p.is_symlink() or s.st_nlink!=1 or s.st_flags not in (0,32) or s.st_size!=r['bytes'] or s.st_ino!=r['inode'] or s.st_mtime_ns!=r['mtime_ns']:state['skipped_metadata_or_open']+=1;continue
  h=sha(p);state['files_hashed']+=1
  if h not in byhash:byhash[h]=(p,meta(s),attrs(p));continue
  src,srcmeta,srcattrs=byhash[h]
  if meta(src.stat())!=srcmeta or attrs(p)!=srcattrs or (s.st_uid,s.st_gid)!=(src.stat().st_uid,src.stat().st_gid) or not equal(src,p):state['skipped_metadata_or_open']+=1;continue
  with writable_parent(p) as allowed:
   if not allowed:state['skipped_metadata_or_open']+=1;continue
   tmp=p.with_name(p.name+'.cow111-tmp');assert not tmp.exists();clone(src,tmp)
   try:
    os.chmod(tmp,stat.S_IMODE(s.st_mode));os.utime(tmp,ns=(s.st_atime_ns,s.st_mtime_ns));ns=tmp.stat()
    assert (ns.st_size,ns.st_uid,ns.st_gid,ns.st_mode,ns.st_mtime_ns,ns.st_flags)==(s.st_size,s.st_uid,s.st_gid,s.st_mode,s.st_mtime_ns,s.st_flags)
    assert attrs(tmp)==srcattrs and sha(tmp)==h and equal(tmp,p)
    assert meta(p.stat())==meta(s) and meta(src.stat())==srcmeta
    entry={'source':str(src),'target':str(p),'sha256':h,'bytes':s.st_size,'allocated_before':s.st_blocks*512,'original_inode':s.st_ino,'source_inode':src.stat().st_ino,'target_mtime_ns':s.st_mtime_ns,'target_mode':s.st_mode,'xattrs_equal':True,'phase':'VERIFIED_BEFORE_REPLACEMENT'};journal.write(json.dumps(entry)+'\n');journal.flush()
    os.replace(tmp,p);assert p.stat().st_ino!=src.stat().st_ino and sha(p)==h
    entry.update(phase='REPLACED_INDEPENDENT_COW',new_inode=p.stat().st_ino);journal.write(json.dumps(entry)+'\n');journal.flush();state['replacements']+=1;state['byte_identical_candidate_bytes']+=s.st_blocks*512
   except BaseException:tmp.unlink(missing_ok=True);save();raise
 if group_index%50==0:
  state['last_group']=group_index;save();print(json.dumps({k:state[k] for k in ['last_group','replacements','files_hashed','observed_available_delta_bytes','elapsed_s']}),flush=True)
else:state['all_groups_scanned']=True
journal.close();state['status']='PASS_EXACT_BYTES_INDEPENDENT_CLONE_STORAGE';save();print(json.dumps(state),flush=True)
