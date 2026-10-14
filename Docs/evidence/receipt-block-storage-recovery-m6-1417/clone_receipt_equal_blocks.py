from pathlib import Path
import hashlib,json,os,subprocess,sys,time
BASE=Path("/Users/numi/numi-human-retained-delivery-20261009")
old=BASE/"clone_exact_duplicate_maps_20261014_002.py"
exec(old.read_text().split("groups=json.loads",1)[0])
pilot="--pilot" in sys.argv
job="receipt-block-pilot001" if pilot else "receipt-block-remaining001"
manifest_path=OUT/"receipt-block-preflight-manifest-001.json"
manifest=json.loads(manifest_path.read_text())
rows=manifest[:1] if pilot else manifest[1:]
journal=OUT/(job+"-replacements.jsonl")
if journal.exists():raise RuntimeError("refuse repeated block-clone job")
before=free();started=time.monotonic();done=[];skipped=[]
with journal.open("x") as log:
 for row in rows:
  source=Path(row["source"]);target=Path(row["target"])
  assert source!=target and not source.is_symlink() and not target.is_symlink()
  assert source.is_file() and target.is_file()
  if target.stat().st_nlink!=1 or opened(source) or opened(target):
   skipped.append({"target":str(target),"reason":"hard-linked or open; original preserved"})
   continue
  assert digest(source)==row["source_sha256"]
  assert digest(target)==row["target_sha256"]
  m=meta(target);assert m["flags"]==0
  stage=target.with_name(target.name+".clone-equal-block-stage-20261014")
  assert not stage.exists()
  subprocess.run(["/bin/cp","-c","-p",str(source),str(stage)],check=True)
  equal_bytes=rewritten=offset=0
  with source.open("rb") as src,target.open("rb") as wanted,stage.open("r+b") as dst:
   while True:
    actual=wanted.read(row["block_bytes"])
    if not actual:break
    base=src.read(row["block_bytes"])
    if actual==base:
     equal_bytes+=len(actual)
    else:
     dst.seek(offset);assert dst.write(actual)==len(actual);rewritten+=len(actual)
    offset+=len(actual)
   dst.truncate(m["bytes"]);dst.flush();os.fsync(dst.fileno())
  assert equal_bytes==row["same_offset_verified_bytes"]
  ck(lib.copyfile(os.fsencode(target),os.fsencode(stage),None,7))
  birthset(stage,tuple(m["birth"]));os.utime(stage,ns=(m["atime_ns"],m["mtime_ns"]))
  assert digest(stage)==row["target_sha256"]
  os.utime(stage,ns=(m["atime_ns"],m["mtime_ns"]))
  assert meta(stage)==m
  assert not opened(target) and not opened(source)
  assert digest(source)==row["source_sha256"] and digest(target)==row["target_sha256"]
  os.utime(target,ns=(m["atime_ns"],m["mtime_ns"]))
  assert meta(target)==m
  os.replace(stage,target)
  assert digest(target)==row["target_sha256"]
  os.utime(target,ns=(m["atime_ns"],m["mtime_ns"]))
  assert meta(target)==m
  result={**row,"rewritten_bytes":rewritten,"metadata":m,"verified":True,"inode_changed":True,"ctime_changed":True}
  log.write(json.dumps(result,sort_keys=True)+"\n");log.flush();done.append(result)
after=free()
report={"count":len(done),"skipped":skipped,"free_before_bytes":before,"free_after_bytes":after,"measured_reclaimed_bytes":after-before,"wall_seconds":time.monotonic()-started,"sources_and_evidence_bytes_preserved":True,"copy_on_write":True,"scope":"same-offset SHA256-matched receipt blocks; full before/stage/after target SHA and metadata preserved"}
(OUT/(job+"-report.json")).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
