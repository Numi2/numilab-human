#!/usr/bin/env python3
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def main():
 p=Path(sys.argv[1]);m=json.loads(p.read_text());argv=m["argv"]
 if not argv or argv[0]!="/usr/bin/env":raise SystemExit("run manifest argv is not env form")
 env=os.environ.copy();cmd=[]
 for token in argv[1:]:
  if "=" in token and not cmd:
   k,v=token.split("=",1);env[k]=v
  else:cmd.append(token)
 if not cmd:raise SystemExit("no CLI command")
 t=time.monotonic();rc=subprocess.run(cmd,cwd=m["cwd"],env=env).returncode;elapsed=time.monotonic()-t
 outdir=Path(env["NUMI_H706_FRACTIONAL_OUTPUT_DIR"])
 files=[]
 if outdir.exists():
  for q in sorted(outdir.rglob("*")):
   if q.is_file():files.append({"path":str(q),"sha256":sha(q),"bytes":q.stat().st_size})
 d={"schema":"numi.h706.current-fractional-mass-cli-execution.033.v1","exit_code":rc,"elapsed_wall_seconds":elapsed,
    "run_manifest":{"path":str(p),"sha256":sha(p)},"binary":m["binary"],"native_steps":0,"physical_admission":False,
    "allocator_run":True,"output_dir":str(outdir),"output_files":files}
 ep=Path(m["execution_json"]);ep.write_text(json.dumps(d,indent=2,sort_keys=True)+"\n")
 return rc
if __name__=="__main__":raise SystemExit(main())

