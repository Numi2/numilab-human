from pathlib import Path
import sys, json, hashlib, importlib.util, time
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
R=Path("/Users/n/numi-human-retained-delivery-20261009")
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
CAND=A/"remaining-limb-positive-union-lift-001/stable-134-reference-union-row-patch.npz"
T=R/"source-seam-connectivity-1247/fhl-current-7b23-count-reconciliation-1258/attempt-002/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
CIROOT=Path("/Users/n/numi-human-self-separation-partial-resume-1267/src")
sys.path.insert(0,str(CIROOT))
from numilab_human import cardiac_cavity_intersections as ci
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location("fw",H);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
h.TISS=T; h.MAN=T.with_suffix(".manifest.json")
t=h.load_tissue(T);row=h.row_data(t,134)
z=np.load(CAND);p=z["vertices6"][:,:3].astype(np.float32);local=z["binding_indices"];weights=z["weights"];faces=z["faces"].astype(np.int64)
# face components via shared indexed edges
from collections import defaultdict,deque
edge_faces=defaultdict(list)
for fi,f in enumerate(faces):
 for a,b in ((f[0],f[1]),(f[1],f[2]),(f[2],f[0])):edge_faces[tuple(sorted((int(a),int(b))))].append(fi)
adj=[set() for _ in faces]
for fs in edge_faces.values():
 for a in fs:
  adj[a].update(x for x in fs if x!=a)
comp=np.full(len(faces),-1,dtype=int);ciid=0
for i in range(len(faces)):
 if comp[i]>=0:continue
 q=[i];comp[i]=ciid
 while q:
  x=q.pop()
  for y in adj[x]:
   if comp[y]<0:comp[y]=ciid;q.append(y)
 ciid+=1
# pose inventory from same 24 retained receipts as prior audit
pose_specs=[]; early=A/"twenty-eight-surface-native-composition-001/baseline/native-run/accepted-geometry"
for st in [0,4767,5023,5599,6207,6815,7423,8000]:pose_specs.append(("early",st,early))
for arm in ("baseline","intervention"):
 for st in h.STEPS:pose_specs.append((arm,st,h.PAIR/arm/"native-run/accepted-geometry"))
rows=[];pins={str(x):sha(x) for x in [Path(__file__),H,CAND,T,T.with_suffix(".manifest.json"),Path(ci.__file__)]}
for group,step,directory in pose_specs:
 rp=directory/f"step-{step}.receipt.json";pack=directory/f"step-{step}.mrvpack";pins[str(rp)]=sha(rp);pins[str(pack)]=sha(pack)
 rc=json.loads(rp.read_text());pose_rows=rc.get("accepted_body_poses") or rc.get("accepted_registered_body_poses")
 poses={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in pose_rows}
 world=h.forward(p,local,weights,row,t,poses)
 rec,deg=h.exact_rows(world,faces,ci)
 aud=ci._audit_pair(rec,rec,same_surface=True) if not deg else {"count":None,"triangle_pairs":[]}
 pairs=[list(map(int,x)) for x in aud["triangle_pairs"]]
 rows.append({"group":group,"step":step,"count":aud["count"],"pairs":pairs,"degenerate":deg,"world_sha256":hashlib.sha256(np.ascontiguousarray(world,dtype="<f4").tobytes()).hexdigest()})
report={"schema":"bounded-stable-134-residual-pair-diagnostic.v1","candidate":str(CAND),"candidate_sha256":sha(CAND),"tissue_sha256":sha(T),"predicate":str(Path(ci.__file__)),"predicate_sha256":sha(Path(ci.__file__)),"face_components":int(ciid),"face_component_by_row":comp.tolist(),"pose_audits":rows,"input_sha256":pins,"inputs_unchanged":all(sha(k)==v for k,v in pins.items()),"diagnostic_only":True}
out=Path(__file__).with_name("residual-pairs-24pose.json");out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"out":str(out),"sha256":sha(out),"self_counts":[x["count"] for x in rows],"pairsets":sorted({tuple(tuple(y) for y in x["pairs"]) for x in rows}),"components":int(ciid),"inputs_unchanged":report["inputs_unchanged"]},indent=2))
