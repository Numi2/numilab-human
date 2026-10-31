from pathlib import Path
import hashlib,json,subprocess,time,datetime
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
ROOT=H/"paired-D307-I364-volume-composition-007"
SOURCE=ROOT/"independent-check-source-001/independent_check_composition_007.corrected-003.py"
SOURCE_SHA="76765980c74b41394e942245cf5f7619b245d118eac020bfe732c2c9fa4056ac"
PY=Path("/Users/numi/numi-human-prep-venv-20261005/bin/python")
WRAPPER=H.parent/"run_geometry_serial.py"
OUT=ROOT/"independent-execution-003"
OUT.mkdir(parents=True,exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
if sha(SOURCE)!=SOURCE_SHA:raise SystemExit("independent source SHA mismatch")
cmd=[str(PY),str(WRAPPER),str(PY),str(SOURCE)]
t0=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
proc=subprocess.run(cmd,cwd=str(ROOT),text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
ended=datetime.datetime.now(datetime.timezone.utc).isoformat()
stdout=OUT/"stdout.log";stderr=OUT/"stderr.log"
stdout.write_text(proc.stdout);stderr.write_text(proc.stderr)
report=ROOT/"independent-replay-003/report.json"
receipt={"schema":"h706.independent-composition-007-execution-receipt.v1","status":"PASS_INDEPENDENT_REPLAY" if proc.returncode==0 and report.exists() else "FAIL_OR_INCOMPLETE_INDEPENDENT_REPLAY","exit_code":int(proc.returncode),"started_utc":started,"ended_utc":ended,"elapsed_seconds":time.monotonic()-t0,"command":cmd,"producer":{"path":str(SOURCE),"sha256":sha(SOURCE)},"logs":{"stdout":{"path":str(stdout),"sha256":sha(stdout)},"stderr":{"path":str(stderr),"sha256":sha(stderr)}},"independent_report":{"path":str(report),"sha256":sha(report) if report.exists() else None}}
if report.exists():
 j=json.loads(report.read_text());receipt["independent_report_status"]=j.get("status")
receipt_path=OUT/"execution-receipt.json";receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps({"exit_code":proc.returncode,"status":receipt["status"],"report":receipt["independent_report"],"receipt":str(receipt_path),"receipt_sha256":sha(receipt_path)},indent=2))
raise SystemExit(proc.returncode)
