from pathlib import Path
import ctypes,hashlib,json,os,stat,struct,subprocess,sys,time
BASE=Path("/Users/numi/numi-human-retained-delivery-20261009")
OUT=BASE/"storage-clone-duplicates-20261014"
OUT.mkdir(exist_ok=True)
lib=ctypes.CDLL("/usr/lib/libSystem.B.dylib",use_errno=True)
class Attr(ctypes.Structure):
 _fields_=[("bitmapcount",ctypes.c_uint16),("reserved",ctypes.c_uint16),("commonattr",ctypes.c_uint32),("volattr",ctypes.c_uint32),("dirattr",ctypes.c_uint32),("fileattr",ctypes.c_uint32),("forkattr",ctypes.c_uint32)]
class TS(ctypes.Structure):
 _fields_=[("sec",ctypes.c_long),("nsec",ctypes.c_long)]
def ck(v):
 if v:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
def birth(p):
 a=Attr(5,0,0x200,0,0,0,0);b=ctypes.create_string_buffer(20)
 ck(lib.getattrlist(os.fsencode(p),ctypes.byref(a),b,len(b),0))
 assert struct.unpack_from("<I",b.raw)[0]==20
 return struct.unpack_from("<qq",b.raw,4)
def birthset(p,value):
 a=Attr(5,0,0x200,0,0,0,0);b=TS(*value)
 ck(lib.setattrlist(os.fsencode(p),ctypes.byref(a),ctypes.byref(b),ctypes.sizeof(b),0))
def digest(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
 return h.hexdigest()
def meta(p):
 s=Path(p).stat()
 names=subprocess.check_output(["/usr/bin/xattr",str(p)],text=True).splitlines()
 attrs={k:hashlib.sha256(subprocess.check_output(["/usr/bin/xattr","-px",k,str(p)])).hexdigest() for k in names}
 acl=subprocess.check_output(["/bin/ls","-lde",str(p)],text=True).splitlines()[1:]
 return {"bytes":s.st_size,"mode":s.st_mode,"uid":s.st_uid,"gid":s.st_gid,"flags":s.st_flags,"birth":list(birth(p)),"atime_ns":s.st_atime_ns,"mtime_ns":s.st_mtime_ns,"xattrs":attrs,"acl":acl}
def opened(p):
 r=subprocess.run(["/usr/sbin/lsof","-F","pfn","--",str(p)],capture_output=True,text=True)
 if r.returncode not in (0,1):raise RuntimeError("lsof failed")
 return bool(r.stdout.strip())
def free():
 return os.statvfs(BASE).f_bavail*os.statvfs(BASE).f_frsize
groups=json.loads((BASE/"duplicate-map-full-sha-groups-20261014.json").read_text())
pilot="--pilot" in sys.argv
job="pilot002" if pilot else "remaining002"
journal=OUT/(job+"-replacements.jsonl")
if journal.exists():raise RuntimeError("refuse repeat job")
manifest=[]
for g in groups:
 source=Path(g["files"][0]["path"])
 single=[r for r in g["files"] if Path(r["path"]).stat().st_nlink==1 and Path(r["path"])!=source]
 for row in single[:1] if pilot else single[1:]:
  target=Path(row["path"]);manifest.append({"source":str(source),"target":str(target),"sha256":g["sha256"],"bytes":g["bytes"]})
(OUT/(job+"-manifest.json")).write_text(json.dumps(manifest,indent=2)+"\n")
before=free();start=time.monotonic();done=0
with journal.open("x") as j:
 for row in manifest:
  source=Path(row["source"]);target=Path(row["target"])
  assert source!=target and not target.is_symlink() and not source.is_symlink()
  assert target.stat().st_nlink==1
  assert digest(source)==digest(target)==row["sha256"]
  m=meta(target);assert m["flags"]==0
  if opened(source) or opened(target):raise RuntimeError("active handle, target preserved")
  stage=target.with_name(target.name+".clone-stage-20261014")
  assert not stage.exists()
  subprocess.run(["/bin/cp","-c","-p",str(source),str(stage)],check=True)
  ck(lib.copyfile(os.fsencode(target),os.fsencode(stage),None,7))
  birthset(stage,tuple(m["birth"]));os.utime(stage,ns=(m["atime_ns"],m["mtime_ns"]))
  assert digest(stage)==row["sha256"]
  os.utime(stage,ns=(m["atime_ns"],m["mtime_ns"]))
  assert meta(stage)==m
  assert not opened(target) and digest(target)==row["sha256"]
  os.replace(stage,target)
  assert digest(target)==row["sha256"]
  os.utime(target,ns=(m["atime_ns"],m["mtime_ns"]))
  assert meta(target)==m
  row.update({"metadata":m,"verified":True,"inode_changed":True,"ctime_changed":True})
  j.write(json.dumps(row,sort_keys=True)+"\n");j.flush();done+=1
after=free()
report={"count":done,"logical_duplicate_bytes":sum(x["bytes"] for x in manifest),"free_before_bytes":before,"free_after_bytes":after,"measured_reclaimed_bytes":after-before,"wall_seconds":time.monotonic()-start,"sources_and_evidence_bytes_preserved":True,"copy_on_write":True,"scope":"exact SHA256 duplicate common mapping binaries only; independent edits preserved; no source or evidence deletion"}
(OUT/(job+"-report.json")).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
