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

def patch_clone(base,target,temp):
 clone(base,temp);length=target.stat().st_size;fd=os.open(temp,os.O_WRONLY);changed=0
 try:
  os.ftruncate(fd,length)
  with base.open('rb') as a,target.open('rb') as b:
   offset=0
   while True:
    expected=b.read(1024*1024)
    if not expected:break
    original=a.read(len(expected));original+=b'\0'*(len(expected)-len(original))
    if original!=expected:
     begin=None
     for at in range(0,len(expected),4096):
      end=min(len(expected),at+4096);different=original[at:end]!=expected[at:end]
      if different and begin is None:begin=at
      if begin is not None and (not different or end==len(expected)):
       stop=at if not different else end;data=expected[begin:stop];written=os.pwrite(fd,data,offset+begin);assert written==len(data);changed+=len(data);begin=None
    offset+=len(expected)
 finally:os.close(fd)
 return changed
# Primitive tests include independent writes, growth, shrinkage and unchanged bytes.
C=D/'block-control';C.mkdir(exist_ok=False);base=C/'base';payload=os.urandom(1024*1024);base.write_bytes(payload);control=[]
for i,data in enumerate([payload,payload[:65536],payload+b'new tail'*1000,payload[:4096]+b'x'*4096+payload[8192:]]):
 target=C/f'target-{i}';temp=C/f'clone-{i}';target.write_bytes(data);changed=patch_clone(base,target,temp);assert temp.read_bytes()==data and sha(base)==hashlib.sha256(payload).hexdigest();control.append({'case':i,'bytes':len(data),'changed_bytes':changed,'exact':True});temp.write_bytes(b'independent');assert base.read_bytes()==payload;target.unlink();temp.unlink()
base.unlink();C.rmdir();(D/'block-control.json').write_text(json.dumps({'status':'PASS','cases':control,'source_unchanged_after_clone_write':True},indent=2)+'\n')
plan=json.loads((D.parent/'native-import-106/install-motion.plan.json').read_text());protected={*plan['input_pins'],*plan['role_input_pins']};pins={(Path(p).stat().st_dev,Path(p).stat().st_ino) for p in protected};pairs=json.loads((D/'page-similarity-survey.json').read_text())['rows'];cache=D/'raw-template-cache';cache.mkdir(exist_ok=False);templates={};start=time.monotonic();free0=shutil.disk_usage(D).free;rows=[];state={'status':'RUNNING','free_before':free0,'paths_and_exact_bytes_retained':True,'rows':rows};journal=(D/'replacements.jsonl').open('x')
def save():
 state.update(free_after=shutil.disk_usage(D).free,observed_available_delta_bytes=shutil.disk_usage(D).free-free0,elapsed_s=time.monotonic()-start);(D/'execution.json').write_text(json.dumps(state,indent=2)+'\n')
for proposal in pairs:
 if time.monotonic()-start>300 or shutil.disk_usage(D).free>=35_000_000_000:break
 p=Path(proposal['target']);source=Path(proposal['template']);s=p.stat();entry={'path':str(p),'source':str(source)};rows.append(entry)
 if s.st_flags!=0 or s.st_nlink!=1 or (s.st_dev,s.st_ino) in pins or attrs(p) or s.st_uid!=os.getuid():entry['status']='RETAINED_STORAGE_OR_CURRENT_INPUT';continue
 original_sha=sha(p)
 if str(source) not in templates:
  base=cache/f'{len(templates):03d}.raw';ss=source.stat()
  with source.open('rb') as src,base.open('xb') as out:shutil.copyfileobj(src,out,1024*1024)
  assert base.stat().st_flags==0 and sha(base)==sha(source) and meta(source.stat())==meta(ss);templates[str(source)]=base
 base=templates[str(source)];temp=p.with_name(p.name+'.cow112-tmp');assert not temp.exists()
 with writable_parent(p) as allowed:
  if not allowed:entry['status']='RETAINED_UNWRITABLE_PARENT';continue
  changed=patch_clone(base,p,temp)
  try:
   entry.update(bytes=s.st_size,changed_bytes=changed,matched_bytes=s.st_size-changed)
   if changed>s.st_size*.35:entry['status']='RETAINED_LOW_SHARING';temp.unlink();continue
   os.chmod(temp,stat.S_IMODE(s.st_mode));os.utime(temp,ns=(s.st_atime_ns,s.st_mtime_ns));ns=temp.stat();assert sha(temp)==original_sha and equal(temp,p) and meta(p.stat())==meta(s)
   assert (ns.st_size,ns.st_uid,ns.st_gid,ns.st_mode,ns.st_mtime_ns,ns.st_flags)==(s.st_size,s.st_uid,s.st_gid,s.st_mode,s.st_mtime_ns,s.st_flags) and attrs(temp)==attrs(p)
   proof={'phase':'VERIFIED_BEFORE_REPLACEMENT','source':str(source),'target':str(p),'sha256':original_sha,'bytes':s.st_size,'allocated_before':s.st_blocks*512,'original_inode':s.st_ino,'target_mtime_ns':s.st_mtime_ns,'target_mode':s.st_mode,'changed_bytes':changed,'matched_bytes':s.st_size-changed};journal.write(json.dumps(proof)+'\n');journal.flush();os.replace(temp,p);assert sha(p)==original_sha;proof.update(phase='REPLACED_INDEPENDENT_COW',new_inode=p.stat().st_ino);journal.write(json.dumps(proof)+'\n');journal.flush();entry['status']='PASS_EXACT_BYTES_SHARED_UNCHANGED_PAGES';save()
  except BaseException:temp.unlink(missing_ok=True);save();raise
for p in templates.values():p.unlink()
cache.rmdir();journal.close();state.update(status='PASS_EXACT_BYTES_COW_PAGE_SHARING',raw_template_scratch_removed=True);save();print(json.dumps({k:v for k,v in state.items() if k!='rows'}),flush=True)
