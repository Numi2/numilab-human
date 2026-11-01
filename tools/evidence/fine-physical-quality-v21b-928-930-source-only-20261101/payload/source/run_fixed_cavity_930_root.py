from pathlib import Path
import sys,json,hashlib,subprocess,time
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");out=H/"all-quality-930-fixed-cavity-batch-root";out.mkdir()
owner=H/"quality-endpoint-collapse-001/fixed_boundary_cavity_884c5_count_first_root.py"
assert hashlib.sha256(owner.read_bytes()).hexdigest()=="c21559d5871e0c3c8414bcde592550be8f5a44ad88afdada3d9bb02ff1d79467"
cmd=[sys.executable,str(owner),"--state",str(H/"all-quality-929-fixed-boundary-complete-cow-root/flips/verified-state.json"),"--output",str(out/"batch"),"--roles","Skin,Fat","--seconds-per-role","60","--solve-seconds","3","--max-patches","32"]
t=time.monotonic()
with (out/"batch.stdout.log").open("w") as f,(out/"batch.stderr.log").open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
(out/"batch.command.json").write_text(json.dumps({"command":cmd,"returncode":r.returncode,"seconds":time.monotonic()-t},indent=2))
print("RETURN",r.returncode,flush=True);print((out/"batch.stderr.log").read_text()[-3000:]);print((out/"batch.stdout.log").read_text()[-5000:])
