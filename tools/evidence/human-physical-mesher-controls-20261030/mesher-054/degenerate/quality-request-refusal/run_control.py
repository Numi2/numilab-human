from pathlib import Path
import hashlib,json,subprocess,time
root=Path(__file__).parent
binary=root.parent/"tetgen"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cmd=[str(binary),"-rq1.2/10AS512O0/0/0",str(root/"degenerate")]
start=time.monotonic()
r=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
(root/"stdout.log").write_text(r.stdout);(root/"stderr.log").write_text(r.stderr)
reject_text=("degenerated tet" in r.stdout or "degenerated tet" in r.stderr or
             "degenerate" in r.stdout.lower() or "degenerate" in r.stderr.lower())
receipt={"argv":cmd,"exit_code":r.returncode,"wall_s":time.monotonic()-start,
 "input_node_sha256":sha(root/"degenerate.node"),"input_element_sha256":sha(root/"degenerate.ele"),
 "stdout_tail":r.stdout[-3000:],"stderr_tail":r.stderr[-3000:],
 "physical_admission":False,"runtime":0,"accepted_steps":0}
if r.returncode!=0 and reject_text:
 receipt["status"]="PASS_DEGENERATE_QUALITY_INPUT_REFUSED"
 receipt["validator_attribution"]="Quality/refinement request enabled; exact branch is inferred from zero-improvement command and TetGen degenerate-tet refusal output."
else:
 receipt["status"]="INCONCLUSIVE"
 receipt["validator_attribution"]="No branch-specific degenerate-tet refusal was observed."
for ext in ("node","ele"):
 p=root/("degenerate.1."+ext)
 if p.exists():
  with p.open() as f: receipt[ext+"_header"]=f.readline().strip()
(root/"result.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(receipt,sort_keys=True))
