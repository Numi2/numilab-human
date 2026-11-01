from pathlib import Path
import subprocess,sys,json,hashlib,time
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");p=H/"quality-partial-cavity-684-root/paired_connected_boundary_patch_batch_879_multiface_fixturefix_root.py";assert hashlib.sha256(p.read_bytes()).hexdigest()=="cf2953c56f5850f125cdfd30888d6d21b2d84ddb40e155c7d987fdc8cb2a03b3"
o=H/"all-quality-893-multiface-I-root";o.mkdir(exist_ok=False)
cmd=[sys.executable,str(p),"--state",str(H/"all-quality-892-qualified-root/verified-state.json"),"--role","I","--output",str(o/"proposal"),"--maximum-edge-m",".1","--batch-seconds","120","--solve-seconds","3","--max-patches","32","--max-pairs-per-patch","1","--max-multiface-pockets","64","--maximum-old-cells","64","--maximum-points","24","--maximum-local-volume-fraction",".01","--maximum-global-volume-fraction","1e-4","--inner-back-topology-only","--fixed-coordinate-topology-only"]
t=time.monotonic()
with (o/"batch.stdout.log").open("w") as f,(o/"batch.stderr.log").open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
(o/"batch.command.json").write_text(json.dumps({"command":cmd,"returncode":r.returncode,"seconds":time.monotonic()-t},indent=2)+"\n");print("RETURN",r.returncode,flush=True)
print((o/("batch.stdout.log" if r.returncode==0 else "batch.stderr.log")).read_text()[-5000:],flush=True)
raise SystemExit(r.returncode)
