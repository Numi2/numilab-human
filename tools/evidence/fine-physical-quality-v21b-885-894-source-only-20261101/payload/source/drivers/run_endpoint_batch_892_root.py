from pathlib import Path
import subprocess,sys,hashlib,json,time
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");p=H/"quality-endpoint-collapse-001/batch_inner_endpoint_quotient_006.py";assert hashlib.sha256(p.read_bytes()).hexdigest()=="595ee0252bf80451f9024ea7ab54e758e14182c02a67beb6626400b60f183c2d"
o=H/"all-quality-892-endpoint-I-root";o.mkdir(exist_ok=False)
cmd=[sys.executable,str(p),"--state",str(H/"all-quality-890-qualified-root/verified-state.json"),"--roster",str(H/"all-quality-891-I-roster-root.json"),"--output",str(o/"proposal"),"--role","I","--max-ops","32","--max-trials","180","--seconds","90"]
t=time.monotonic()
with (o/"batch.stdout.log").open("w") as f,(o/"batch.stderr.log").open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
(o/"batch.command.json").write_text(json.dumps({"command":cmd,"returncode":r.returncode,"seconds":time.monotonic()-t},indent=2)+"\n");print("RETURN",r.returncode,flush=True)
print((o/("batch.stdout.log" if r.returncode==0 else "batch.stderr.log")).read_text()[-5000:],flush=True)
raise SystemExit(r.returncode)
