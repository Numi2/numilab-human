from pathlib import Path
import json,subprocess,sys,hashlib,time
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");out=H/"all-quality-887-endpoint-D-root";out.mkdir(exist_ok=False)
def pin(p):return {"path":str(p.resolve()),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
parent=H/"all-quality-884-qualified-root/verified-state.json";receipt=json.loads((parent.parent/"receipt.json").read_text());assert receipt["status"].startswith("PASS_") and receipt["state"]==pin(parent)
owner=H/"quality-endpoint-collapse-001";batch=owner/"batch_inner_endpoint_quotient_005.py";census=owner/"census_bad_short_endpoint_edges_004.py"
assert pin(batch)["sha256"]=="eb57b4ad8a91feec2ed84ca7cf483093ec2b49f367bb1ff55977f927a4997619"
assert pin(census)["sha256"]=="33bd68879cb2db5fc235220d3eee54d0b568843a4bef7c3d827fc976c86d487c"
def run(name,cmd):
 t=time.monotonic()
 with (out/(name+".stdout.log")).open("w") as f,(out/(name+".stderr.log")).open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
 (out/(name+".command.json")).write_text(json.dumps({"command":cmd,"returncode":r.returncode,"seconds":time.monotonic()-t},indent=2)+"\n")
 if r.returncode:print((out/(name+".stderr.log")).read_text());raise SystemExit(r.returncode)
roster=out/"roster.json";run("census",[sys.executable,str(census),"--state",str(parent),"--role","D","--output",str(roster)])
run("batch",[sys.executable,str(batch),"--state",str(parent),"--roster",str(roster),"--output",str(out/"proposal"),"--role","D","--max-ops","64","--max-trials","180","--seconds","90"])
print((out/"proposal/report.json").read_text()[-6000:],flush=True)
