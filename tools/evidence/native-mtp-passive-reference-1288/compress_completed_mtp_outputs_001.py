from pathlib import Path
import argparse,json,hashlib,os,stat,subprocess,shutil,time
p=argparse.ArgumentParser();p.add_argument("run",type=Path);args=p.parse_args()
r=args.run.resolve()
assert r.parent.name=="anatomy-completion-1276" and r.name.startswith("mtp-passive-native-sensitivity-")
execution=json.loads((r/"baseline/execution.json").read_text())
assert execution["returncode"]==0 and not execution["changed_inputs"]
out=r/"completed-output-lossless-compression.json";assert not out.exists()
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
report={"scope":"Lossless filesystem compression of completed bounded native outputs; all logical bytes and paths retained, no input assets touched.","free_before":shutil.disk_usage(r).free,"files":[],"complete":False}
def save():out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
save()
for f in sorted((r/"baseline/native-run").rglob("*")):
 if not f.is_file() or f.is_symlink():continue
 s=f.stat()
 if s.st_size<1000000 or s.st_blocks*512<s.st_size*.9 or s.st_nlink!=1 or s.st_flags&(stat.UF_IMMUTABLE|stat.SF_IMMUTABLE):continue
 digest=sha(f);temp=f.with_name(f.name+".lossless-compression.tmp");assert not temp.exists()
 subprocess.run(["/usr/bin/ditto","--hfsCompression",str(f),str(temp)],check=True)
 assert temp.stat().st_size==s.st_size and sha(temp)==digest and stat.S_IMODE(temp.stat().st_mode)==stat.S_IMODE(s.st_mode)
 assert f.stat().st_ino==s.st_ino and f.stat().st_mtime_ns==s.st_mtime_ns and sha(f)==digest
 os.replace(temp,f);assert sha(f)==digest
 report["files"].append({"path":str(f),"sha256":digest,"logical_bytes":s.st_size,"allocated_before":s.st_blocks*512,"allocated_after":f.stat().st_blocks*512});save()
report.update(complete=True,free_after=shutil.disk_usage(r).free);save()
print(json.dumps({"complete":True,"files":len(report["files"]),"saved":sum(x["allocated_before"]-x["allocated_after"] for x in report["files"]),"free":report["free_after"]}))

