from pathlib import Path
import sys,hashlib,json,importlib.util,math,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276";BASE=A/"stable-141-positive-union-lift-001"
OUT=BASE/"pose-pair-diagnostic-001";OUT.mkdir(exist_ok=False)
TISS=A/"forty-two-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";MAN=TISS.with_suffix(".manifest.json")
CAND=BASE/"stable-141-reference-union-row-patch.npz";H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";CAP=A/"forty-two-surface-native-composition-001/baseline/native-run/accepted-geometry"
CI=Path("/Users/n/numi-human-touching-loop-arrangement-1290/src/numilab_human/cardiac_cavity_intersections.py")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(x,m):
 if not x:raise RuntimeError(m)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_intersections as ci
req(sha(CI)=="90c7a7fb0ce4383397f97519dd5fe163201d86a0c97e18e3a9eb3fa9858cd561","predicate pin")
sp=importlib.util.spec_from_file_location("fwd",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h);h.TISS=TISS;h.MAN=MAN
t=h.load_tissue(TISS);row=h.row_data(t,141);z=np.load(CAND,allow_pickle=False)
p=np.asarray(z["vertices6"][:,:3],dtype="<f4");loc=np.asarray(z["binding_indices"],dtype="<u4");w=np.asarray(z["weights"],dtype="<f4");f=np.asarray(z["faces"],dtype=np.int64);orig=np.asarray(z["face_origins"],dtype=np.int64)
pin={str(x):sha(x) for x in [Path(__file__),TISS,MAN,CAND,H,CI]}
def stats(x,tri):
 q=x[tri].astype(np.float64);v=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);return {"area_m2":float(np.linalg.norm(v)/2),"edge_lengths_m":[float(np.linalg.norm(q[(i+1)%3]-q[i])) for i in range(3)],"centroid_m":q.mean(axis=0).tolist()}
poses=[]
for step in [0,4767,5023,5599,6207,6815,7423,8000]:
 pack=CAP/f"step-{step}.mrvpack";receipt=CAP/f"step-{step}.receipt.json";r=json.loads(receipt.read_text());req((r.get("pack_file_sha256") or r.get("accepted_pack_file_sha256"))==sha(pack),"capture pin")
 pr=r.get("accepted_body_poses") or r.get("accepted_registered_body_poses");pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in pr}
 world=np.asarray(h.forward(p,loc,w,row,t,pm),dtype="<f4");wr,wd=h.exact_rows(world,f,ci)
 baseworld=np.asarray(h.forward(row["positions"],row["local"],row["weights"],row,t,pm),dtype="<f4");br,bd=h.exact_rows(baseworld,row["faces"],ci)
 ca=[] if wd else ci._audit_pair(wr,wr,same_surface=True)["triangle_pairs"]
 ba=[] if bd else ci._audit_pair(br,br,same_surface=True)["triangle_pairs"]
 rec={"step":step,"candidate_pairs":[],"baseline_pairs":[],"candidate_degenerate":wd,"baseline_degenerate":bd}
 for pair in ca:
  a,b=map(int,pair);rec["candidate_pairs"].append({"faces":[a,b],"parent_faces":[int(orig[a]),int(orig[b])],"shared_vertex_ids":sorted(set(map(int,f[a]))&set(map(int,f[b]))),"triangle_a":stats(world,f[a]),"triangle_b":stats(world,f[b]),"candidate_vertices":[f[a].astype(int).tolist(),f[b].astype(int).tolist()],"source_vertices":[p[f[a]].tolist(),p[f[b]].tolist()]})
 for pair in ba:
  a,b=map(int,pair);rec["baseline_pairs"].append({"faces":[a,b],"triangle_a":stats(baseworld,row["faces"][a]),"triangle_b":stats(baseworld,row["faces"][b]),"source_vertices":[row["positions"][row["faces"][a]].tolist(),row["positions"][row["faces"][b]].tolist()]})
 poses.append(rec)
 pin[str(pack)]=sha(pack);pin[str(receipt)]=sha(receipt)
doc={"schema":"stable141-exact-self-witnesses.v1","stable_id":141,"predicate_sha256":sha(CI),"candidate_path":str(CAND),"candidate_sha256":sha(CAND),"poses":poses,"complete":True,"inputs_unchanged":all(sha(x)==v for x,v in pin.items()),"pins":pin}
(OUT/"report.json").write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(OUT/"report.json"),"sha256":sha(OUT/"report.json"),"poses":[{"step":x["step"],"baseline":len(x["baseline_pairs"]),"candidate":len(x["candidate_pairs"])} for x in poses],"inputs_unchanged":doc["inputs_unchanged"]},sort_keys=True))
