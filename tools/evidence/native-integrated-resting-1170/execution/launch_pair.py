"""Launch the bounded, two-arm owner workflow detached from the SSH transport."""
import argparse,hashlib,json,pathlib,subprocess,sys,time
O=pathlib.Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plan",type=pathlib.Path,required=True)
    ap.add_argument("--plan-sha256",required=True)
    a=ap.parse_args()
    script=O/"execute_pair.py"
    if (O/"launch.json").exists() or (O/"execution.log").exists():raise RuntimeError("launch evidence already exists")
    if hashlib.sha256(a.plan.read_bytes()).hexdigest()!=a.plan_sha256:raise RuntimeError("plan hash mismatch")
    argv=[sys.executable,str(script),"--plan",str(a.plan),"--plan-sha256",a.plan_sha256]
    with (O/"execution.log").open("x") as log:
        p=subprocess.Popen(argv,cwd=O,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record={"argv":argv,"cwd":str(O),"pid":p.pid,"started_unix":time.time(),
      "script_sha256":hashlib.sha256(script.read_bytes()).hexdigest(),"plan_sha256":a.plan_sha256,
      "scope":"Bounded registration, 2x310s native runs, analysis and verification; no recurring automation."}
    (O/"launch.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record))
if __name__=="__main__":main()
