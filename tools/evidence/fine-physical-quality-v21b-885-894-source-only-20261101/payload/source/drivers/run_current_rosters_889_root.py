from pathlib import Path
import subprocess,sys,time,json
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706");out=H/"all-quality-889-rosters-root";out.mkdir(exist_ok=False)
for role in ("D","I"):
 cmd=[sys.executable,str(H/"quality-endpoint-collapse-001/census_bad_short_endpoint_edges_004.py"),"--state",str(H/"all-quality-888-qualified-root/verified-state.json"),"--role",role,"--output",str(out/(role+"-roster.json"))]
 with (out/(role+".stdout.log")).open("w") as f,(out/(role+".stderr.log")).open("w") as e:r=subprocess.run(cmd,stdout=f,stderr=e)
 (out/(role+".command.json")).write_text(json.dumps({"command":cmd,"returncode":r.returncode},indent=2)+"\n")
 assert r.returncode==0
 d=json.loads((out/(role+"-roster.json")).read_text());print(role,d.get("eligible_count"),flush=True)
