from pathlib import Path
import numpy as np,json,hashlib,importlib.util,itertools,time
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
W=Path("/Users/numi/Work/human-integrated-gaps-20261104");O=W/"witness-cavity-milp-353";O.mkdir(exist_ok=False)
source=W/"reference-screened-interior-341/compiler/rich433/candidate-current-typed-volume.npz"
owner=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/fixed_boundary_volume_flip_198_root.py")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
spec=importlib.util.spec_from_file_location("m",owner);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
z=np.load(source);p=z["points_reference_world_m_f64"];t=z["volume_tetrahedra"];labels=np.column_stack([z[k] for k in ["material_index","geometric_donor_body","anatomical_region_id"]]);seed=1022373
plan=dict(source=str(source),source_sha256=sha(source),exact_owner=str(owner),exact_owner_sha256=sha(owner),method="Existing oriented face-balance MILP; fixed coordinates and exact boundary; same material/donor/region; mandatory independent native cavity certification and mass routing",max_cells=32,seconds_per_solve=5,physical_admission=False)
(O/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
flat=t.ravel();order=np.argsort(flat,kind="stable");off=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
def incident(v):return order[off[v]:off[v+1]]//4
def near(r):
 out=set()
 for f in t[r][m.PAT]:
  ids=incident(f[0])
  for v in f[1:]:ids=np.intersect1d(ids,incident(v),assume_unique=True)
  out.update(map(int,ids))
 return out-{r}
rr={seed};front=near(seed);patches=[]
while front and len(rr)<32:
 choices=[r for r in front if np.array_equal(labels[r],labels[seed])]
 if not choices:break
 r=min(choices,key=lambda r:np.linalg.norm(p[t[r]].mean(0)-p[t[seed]].mean(0)));rr.add(r);front.remove(r);front.update(near(r)-rr)
 if len(rr) in [3,5,8,11,14,20,32]:patches.append(sorted(rr))
if sorted(rr) not in patches:patches.append(sorted(rr))
report=[];best=None
for rows in patches:
 old=t[rows];nodes=np.unique(old);boundary=m.chain(old)
 cells=m.orient(p,np.array(list(itertools.combinations(nodes,4)),np.int64))
 det,q=m.measurements(p,cells);_,oq=m.measurements(p,old)
 for goal in [.001,float(oq.min()*1.021)]:
  mask=m.passes(det,q)&(q>=goal)
  # Exclude all exact current witness slivers; no old-cell substitution.
  forbidden={tuple(sorted(x)) for x in t[[1022371,1022372,1022373]]}
  mask&=np.array([tuple(sorted(x)) not in forbidden for x in cells])
  cand=cells[mask]
  faces=sorted(set(boundary)|set(map(tuple,np.sort(cand[:,m.PAT].reshape(-1,3),axis=1))))
  lookup={f:i for i,f in enumerate(faces)};ri=[];ci=[];va=[]
  for j,ct in enumerate(cand):
   for f in ct[m.PAT]:
    a,b,c=map(int,f);ri.append(lookup[tuple(sorted(f))]);ci.append(j);va.append(1 if ((a>b)+(a>c)+(b>c))%2==0 else -1)
  A=coo_matrix((np.array(va,float),(ri,ci)),shape=(len(faces),len(cand))).tocsc();rhs=np.zeros(len(faces))
  for f,v in boundary.items():rhs[lookup[f]]=v
  result=milp(np.ones(len(cand)),integrality=np.ones(len(cand),np.uint8),bounds=Bounds(0,1),constraints=LinearConstraint(A,rhs,rhs),options={"time_limit":5,"mip_rel_gap":0.01})
  entry=dict(parents=len(rows),vertices=len(nodes),candidates=len(cand),goal=goal,status=int(result.status),message=str(result.message));report.append(entry)
  if result.x is None:continue
  sel=np.rint(result.x).astype(np.int8)
  if np.any(np.abs(result.x-sel)>1e-6) or not np.array_equal(A@sel,rhs):continue
  nt=cand[sel!=0];ok,why=m.certify(p,old,nt);entry["exact_certificate"]=why
  if ok:
   best=dict(old_rows=rows,new_cells=nt.tolist(),old_min_q=float(oq.min()),new_min_q=float(m.measurements(p,nt)[1].min()));break
 if best:break
out=dict(status="PROPOSAL_NATIVE_CERTIFICATE_PENDING" if best else "NO_FEASIBLE_FIXED_BOUNDARY_REPAIR",trials=report,best=best,physical_admission=False)
if best:
 rows=np.array(best["old_rows"],np.int64);nt=np.array(best["new_cells"],np.int64);nr=len(nt)
 np.savez_compressed(O/"fixed-boundary-volume-proposal.npz",retained_input_cell_rows=np.flatnonzero(~np.isin(np.arange(len(t)),rows)),replacement_tetrahedra=nt,replacement_tet_offsets=np.array([0,nr],np.int64),replacement_parent_offsets=np.arange(nr+1,dtype=np.int64)*len(rows),replacement_parent_cell_rows=np.tile(rows,nr),source_volume_sha256=np.array(sha(source)))
(O/"report.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
