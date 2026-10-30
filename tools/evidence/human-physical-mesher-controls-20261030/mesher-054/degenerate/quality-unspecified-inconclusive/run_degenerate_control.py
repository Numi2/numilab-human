from pathlib import Path
import hashlib,json,subprocess,time
root=Path(__file__).parent
binary=root.parent/"tetgen"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cmd=[str(binary),"-rO0/0/0",str(root/"degenerate")]
start=time.monotonic()
r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
(root/"stdout.log").write_text(r.stdout);(root/"stderr.log").write_text(r.stderr)
receipt={"argv":cmd,"exit_code":r.returncode,"wall_s":time.monotonic()-start,
 "input_node_sha256":sha(root/"degenerate.node"),"input_element_sha256":sha(root/"degenerate.ele"),
 "stdout_tail":r.stdout[-3000:],"stderr_tail":r.stderr[-3000:],
 "physical_admission":False,"runtime":0,"accepted_steps":0}
if r.returncode!=0:
 receipt["status"]="DEGENERATE_INPUT_REJECTED"
 receipt["validator_attribution"]="UNPROVEN; TetGen may reject during reconstruction before the explicit no-improvement scan"
else:
 receipt["status"]="INCONCLUSIVE_EXIT_ZERO"
 receipt["validator_attribution"]="Output requires manual quality interpretation; this control does not establish the scan path"
for ext in ("node","ele"):
 p=root/("degenerate.1."+ext)
 if p.exists():
  with p.open() as f: receipt[ext+"_header"]=f.readline().strip()
(root/"result.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(receipt,sort_keys=True))
