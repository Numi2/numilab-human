import json,pathlib,subprocess,time,hashlib,os,signal
p=pathlib.Path(__file__).with_name("run-manifest.json");d=json.loads(p.read_text())
pins=d["file_pins"]+[d["binary"],d["source"],d["source_control_manifest"],d["graph_qualification"],d["root_run_spec"],d["base_manifest"]]
for ent in pins:
 with open(ent["path"],"rb") as f:assert hashlib.file_digest(f,"sha256").hexdigest()==ent["sha256"],ent["path"]
t=time.monotonic()
with open(d["stdout_log"],"xb") as log:
 proc=subprocess.Popen([d["python"],d["wrapper"]]+d["argv"],cwd=d["cwd"],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=proc.wait(timeout=600)
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  code=124
result={"exit_code":code,"elapsed_wall_seconds":time.monotonic()-t,"manifest_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"native_steps":0,"physical_admission":False}
p.with_name("execution.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True);raise SystemExit(code)
