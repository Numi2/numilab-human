from pathlib import Path
import hashlib,importlib.util,json,sys,time
import numpy as np
P=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent/"changed-D010-vs-current-layers"
OUT.mkdir(exist_ok=False)
D=P/"canonical-D-boundary-binding-696/run-010-combined-operational-reference-candidate"
loader=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location("D010_cross_owner",loader)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
pred=m.load_exact_owner()
assert sha(pred.__file__)=="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"
paths={
"D010":D/"candidate-D-canonical14-reference-remap.npz",
"O081":P/"reduced-physical-boundary-400/attempt-046-fine-role-simplification/run-081-OI-local-topology-repair/candidate-O-self-and-OI-repaired.npz",
"I328":P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz",
}
expected={
"D010":"7d8419bdc4ad1d3892a946c93ad7f0f9b2db2e4302f8f666480612166e25de2a",
"O081":"c9074b9f54bfad16193cfd14dc52fbdcda0fcca86f01a34f9552529fc6bb6b72",
"I328":"5d46ee99df2613e9f30a43075af95a6402beffe26392cd9f017d8c1a154bf410"}
for k,path in paths.items():assert sha(path)==expected[k]
arrays={}
for k,path in paths.items():
 with np.load(path,allow_pickle=False) as z:
  arrays[k]={a:np.asarray(z[a]) for a in ("points_world_m","faces")}
targetmap=D/"candidate-operational-target-map.json"
changed=json.loads(targetmap.read_text())["modified_output_face_rows"]
a=arrays["D010"]
records=pred.float32_surface_records(np.asarray(a["points_world_m"],np.float32).astype(float),np.asarray(a["faces"],np.int64),source_seam_groups=())[0]
left=[records[i] for i in changed];gates=[];start=time.monotonic()
for k in ("O081","I328"):
 a=arrays[k];right=pred.float32_surface_records(np.asarray(a["points_world_m"],np.float32).astype(float),np.asarray(a["faces"],np.int64),source_seam_groups=())[0]
 res=pred._audit_pair(left,right,same_surface=False)
 pairs=np.asarray(res.pop("triangle_pairs",[]),np.int64).reshape(-1,2)
 f=OUT/(k+"-pairs.npz");np.savez_compressed(f,triangle_pairs=pairs)
 gates.append({"name":"changed-D010-vs-"+k,**res,"pair_count":len(pairs),"pairs_path":str(f),"pairs_sha256":sha(f)})
 print(json.dumps({"gate":k,"count":len(pairs)}),flush=True)
 if len(pairs):break
result={"status":"PASS_CHANGED_INTERFACES_ONLY" if len(gates)==2 and all(g["pair_count"]==0 for g in gates) else "FAIL",
 "inputs":{k:{"path":str(v),"sha256":expected[k]} for k,v in paths.items()},
 "changed_face_rows":changed,"construction_map_sha256":sha(targetmap),
 "gates":gates,"exact_owner_path":str(pred.__file__),"exact_owner_sha256":sha(pred.__file__),
 "driver_sha256":sha(__file__),"elapsed_s":time.monotonic()-start,
 "physical_admission":False,"runtime":0,"accepted_steps":0}
(OUT/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"status":result["status"],"receipt":str(OUT/"receipt.json")}))
if result["status"]!="PASS_CHANGED_INTERFACES_ONLY":raise SystemExit(2)
