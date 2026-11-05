from pathlib import Path
import numpy as np,json,hashlib,importlib.util,itertools,time
from scipy.spatial import Delaunay
W=Path("/Users/numi/Work/human-integrated-gaps-20261104")
OUT=W/"witness-cavity-352";OUT.mkdir(exist_ok=False)
source=W/"reference-screened-interior-341/compiler/rich433/candidate-current-typed-volume.npz"
owner=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/fixed_boundary_volume_flip_198_root.py")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
spec=importlib.util.spec_from_file_location("exact",owner);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
plan=dict(source=str(source),source_sha256=sha(source),exact_owner=str(owner),exact_owner_sha256=sha(owner),witness_global_tets=[1025686,1025688],object_element_offset=3315,seconds=120,max_parent_cells=32,method="existing exact fixed-boundary cavity predicates; local flips then bounded Delaunay proposals; unchanged coordinates, volume, material, donor and region",physical_admission=False)
(OUT/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
start=time.monotonic()
z=np.load(source);p=z["points_reference_world_m_f64"];t=z["volume_tetrahedra"]
labels=np.column_stack([z[k] for k in ("material_index","geometric_donor_body","anatomical_region_id")])
flat=t.ravel();order=np.argsort(flat,kind="stable");off=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
def incident(v):return order[off[v]:off[v+1]]//4
def face_neighbors(row):
 out=set()
 for f in t[row][m.PAT]:
  r=incident(int(f[0]))
  for v in f[1:]:r=np.intersect1d(r,incident(int(v)),assume_unique=True)
  out.update(map(int,r))
 out.discard(int(row));return out
def valid(rr,nt):
 old=t[rr];nd,nq=m.measurements(p,nt);od,oq=m.measurements(p,old)
 if not np.all(m.passes(nd,nq)) or min(nq)<min(oq)*1.02:return None
 ok,why=m.certify(p,old,nt)
 if not ok:return None
 # Extra nonoverlap/ball proof is independently required from native cavity owner.
 return dict(old_rows=list(map(int,rr)),new_cells=nt.tolist(),old_min_q=float(min(oq)),new_min_q=float(min(nq)),labels=labels[rr[0]].tolist())
trials=[];passing=[]
for seed in [1022371,1022372,1022373]:
 for other in sorted(face_neighbors(seed)):
  if not np.array_equal(labels[seed],labels[other]):continue
  rr=[seed,other];proposal=m.propose_pair(p,*t[rr])
  if proposal is None:continue
  nt,edge=proposal
  if len(np.intersect1d(incident(edge[0]),incident(edge[1]),assume_unique=True)):continue
  d=valid(rr,nt)
  if d:passing.append(d)
 # Expand by weakest neighbor first, testing every prefix. Connected same-label cavity only.
 rr={seed};front=face_neighbors(seed);seen=set()
 while front and len(rr)<32 and time.monotonic()-start<120:
  choices=[r for r in front if np.array_equal(labels[r],labels[seed])]
  if not choices:break
  # Prioritize neighboring slivers and then keep geometrically local expansion.
  scores=[]
  for r in choices:
   _,q=m.measurements(p,t[[r]])
   center=np.mean(p[t[r]],axis=0);base=np.mean(p[t[seed]],axis=0)
   scores.append((float(np.linalg.norm(center-base)),float(q[0]),r))
  r=min(scores)[2];front.remove(r);rr.add(r);front.update(face_neighbors(r)-rr)
  rows=sorted(rr);old=t[rows];nodes=np.unique(old);x=p[nodes]
  if len(nodes)<5:continue
  scale=max(np.ptp(x,axis=0));local=(x-x.mean(0))/scale
  for option in ["Qbb Qc Qz Q12","Qbb Qc Qz Q12 Qt"]:
   try:tri=Delaunay(local,qhull_options=option);nt=m.orient(p,nodes[tri.simplices])
   except Exception:continue
   # Keep simplices whose centroids are inside the original cavity. Exact chain proves final union.
   centers=p[nt].mean(axis=1);inside=np.zeros(len(nt),bool)
   for oldtet in old:
    xp=p[oldtet];b=np.linalg.solve((xp[1:]-xp[0]).T,(centers-xp[0]).T).T
    inside|=(b.min(axis=1)>=-1e-9)&(b.sum(axis=1)<=1+1e-9)
   nt=nt[inside]
   if not 2<=len(nt)<=64:continue
   d=valid(rows,nt)
   if d:passing.append(d)
  trials.append(dict(seed=seed,parents=len(rr),nodes=len(nodes),passing=len(passing)))
# Prefer patch that repairs all witness slivers, then highest worst quality.
passing.sort(key=lambda d:(-len(set(d["old_rows"])&{1022371,1022372,1022373}),-d["new_min_q"],len(d["old_rows"])))
report=dict(status="PASS_PROPOSAL_ONLY" if passing else "NO_VALID_FIXED_BOUNDARY_PROGRESS",elapsed_s=time.monotonic()-start,passing_candidates=passing,trials=trials,source=plan,physical_admission=False)
if passing:
 d=passing[0];rr=np.array(d["old_rows"],np.int64);nt=np.array(d["new_cells"],np.int64);nr=len(nt);keep=np.flatnonzero(~np.isin(np.arange(len(t)),rr))
 np.savez_compressed(OUT/"fixed-boundary-volume-proposal.npz",retained_input_cell_rows=keep,replacement_tetrahedra=nt,replacement_tet_offsets=np.array([0,nr],np.int64),replacement_parent_offsets=np.arange(nr+1,dtype=np.int64)*len(rr),replacement_parent_cell_rows=np.tile(rr,nr),source_volume_sha256=np.array(sha(source)))
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ["trials","passing_candidates"]},indent=2))
if passing:print("best",json.dumps(passing[0]))
