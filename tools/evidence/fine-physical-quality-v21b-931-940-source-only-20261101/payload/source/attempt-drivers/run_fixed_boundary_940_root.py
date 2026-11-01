from pathlib import Path
import subprocess,sys,json,time,hashlib
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");root=H/"all-quality-940-fixed-boundary-complete-cow-root";root.mkdir()
owner=H/"incremental_fixed_boundary_polygon_search_660_cow_complete_root.py"
assert hashlib.sha256(owner.read_bytes()).hexdigest()=="caf8b7280ece02a7e6ac1929fe68b0446c72eae875b60d32b8530e92026856f0"
cmd=[sys.executable,str(owner),"--state",str(H/"all-quality-939-b5-backtracking-root/qualified/verified-state.json"),"--output",str(root/"flips"),"--seconds","30","--maximum-operations","64"]
t=time.monotonic()
with (root/"flips.stdout.log").open("w") as f,(root/"flips.stderr.log").open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
(root/"flips.command.json").write_text(json.dumps({"command":cmd,"returncode":r.returncode,"seconds":time.monotonic()-t},indent=2))
print("RETURN",r.returncode,flush=True)
print((root/"flips.stderr.log").read_text()[-3000:])
if r.returncode:raise SystemExit(r.returncode)
s=json.loads((root/"flips/state.json").read_text());print({r:s[r+"_current"]["native_quality_failures"] for r in ("Skin","Fat")},flush=True)
