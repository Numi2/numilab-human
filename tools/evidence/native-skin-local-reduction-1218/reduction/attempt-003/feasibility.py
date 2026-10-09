#!/usr/bin/env python3
"""Bounded two-variable source-correction feasibility; no native or admission."""
from pathlib import Path
import os
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"): os.environ[k]="1"
import sys,json,hashlib,importlib.util,time
import numpy as np
from scipy.optimize import linprog
R=Path("/Users/n/numi-human-retained-delivery-20261009")
B=R/"skin-resting-multipose-clearance-1218"
OUT=B/"local-self-reduction-003"
ROOT=Path("/Users/n/numi-human-conforming-composition-source-1216")
sys.path.insert(0,str(ROOT/"src"))
from numilab_human import common_atlas_skin_clearance as c
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def j(p): return json.loads(Path(p).read_text())
def imp(p,n):
 s=importlib.util.spec_from_file_location(n,p); m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def write(p,d): p.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+"\n")
def pin(p): return {"path":str(p),"sha256":sha(p),"bytes":p.stat().st_size}
def axes(a,b):
 ea=np.roll(a,-1,axis=0)-a; eb=np.roll(b,-1,axis=0)-b
 na=np.cross(ea[0],ea[1]);nb=np.cross(eb[0],eb[1])
 aa=np.asarray([na,nb,*[np.cross(x,y) for x in ea for y in eb],*[np.cross(x,na) for x in ea],*[np.cross(x,nb) for x in eb]])
 lens=np.linalg.norm(aa,axis=1); return aa[lens>1e-14]/lens[lens>1e-14,None]
def bestplane(a,b):
 aa=axes(a,b); pa=a@aa.T;pb=b@aa.T
 gap1=pb.min(axis=0)-pa.max(axis=0);gap2=pa.min(axis=0)-pb.max(axis=0)
 k1=int(np.argmax(gap1));k2=int(np.argmax(gap2))
 return (aa[k1],float(gap1[k1])) if gap1[k1]>=gap2[k2] else (-aa[k2],float(gap2[k2]))
audit=imp(B/"heldout-001/audit_six_holdouts_1218.py","audit")
fm=imp(audit.FORWARD,"forward")
skin=audit.load_skin(audit.SKIN); ref=np.unique(skin["faces"]); faces=np.searchsorted(ref,skin["faces"])
candidate=np.load(B/"fit-attempt-001/candidate-source-positions-f32.npy",allow_pickle=False)
pre=j(B/"fit-attempt-001/preflight.json")
protected=np.asarray(pre["fixed_support_source_vertex_ids"]+pre["preserved_source_anchor_vertex_ids"],dtype=np.int64)
right=[49688,49720,49721,49723,49759,49760,49761,49796]
left=[60436,60437,60478,60479,60515,60516]
# Three edge rings taper a reduction of the already inferred field. No expansion.
masks=[]; regions=[]
for rows in (right,left):
 seeds=np.unique(skin["faces"][rows]); dist=np.full(skin["nv"],4,dtype=int);dist[seeds]=0
 for ring in range(1,4):
  touching=np.any(dist[skin["faces"]] < ring,axis=1)
  ids=np.unique(skin["faces"][touching]);dist[ids]=np.minimum(dist[ids],ring)
 t=np.clip(1-dist/3.,0.,1.);mask=t*t*(3-2*t)
 assert not np.any(mask[protected])
 masks.append(mask);regions.append(np.flatnonzero(mask>0))
assert not np.intersect1d(regions[0],regions[1]).size
D=np.stack([-(candidate.astype(float)-skin["pos"])*m[:,None] for m in masks],axis=-1)
affected=np.flatnonzero(np.any(D!=0,axis=(1,2)))
affected_faces=np.flatnonzero(np.any(np.isin(skin["faces"],affected),axis=1))
native=R/"muscle-conforming-refinement-1216/native-refinement-1217-attempt2/native-run"
held=R/"native-lung1178-thumb1187-smoke-1191/native-run"
poses=[(native,0),(held,4991),(held,5375),(held,5759),(held,6111),(held,6495),(held,7743),(native,9983),(native,20000)]
keys,_,_,_=c._load_target_inventory(audit.load_wrapper().load_runner().INV)
capture=[]; receipts=[]; inputs=[Path(__file__),audit.SKIN,B/"fit-attempt-001/candidate-source-positions-f32.npy",B/"fit-attempt-001/preflight.json",audit.FORWARD,ROOT/"src/numilab_human/common_atlas_skin_clearance.py"]
for run,step in poses:
 p=run/"accepted-geometry"/f"step-{step}.mrvpack"; rp=p.with_suffix(".receipt.json");rd=j(rp)
 assert rd["accepted_step"]==step and rd["pack_file_sha256"]==sha(p)
 inputs.extend([p,rp])
 pos,surfs,counts=c._pack_surfaces(p,set(keys)); sf=surfs[(51007,1)]["faces"];base=int(sf.min())
 assert np.array_equal(sf-base,skin["faces"])
 capture.append(pos[base:base+skin["nv"]][ref].copy())
 receipts.append({"step":step,"body_poses":rd["accepted_registered_body_poses"],"respiratory_motion":rd["accepted_respiratory_motion"]})
 if step==0: initial=rd
 if step==20000:
  targets={k:pos[surfs[k]["faces"]].astype(float) for k in ((51005,35),(51005,36))}
 del pos,surfs
 print(json.dumps({"phase":"capture_loaded","step":step}),flush=True)
before={str(p):sha(p) for p in inputs}
forward,mr,selectors=fm.build_forward(skin=skin,source_positions=skin["pos"],referenced_ids=ref,captured_by_pose=np.asarray(capture),state_receipts=receipts,initial_body_poses=initial["initial_anatomical_registration"]["body_poses"],map_receipt=initial,anatomy_parameters_path=initial["skin_source_mapping"]["anatomy_parameters"]["path"],respiration_source=Path("/Users/n/numi-human-resting-resp-source-20261006"),clearance_module=c)
f=forward(candidate); assert f["diagnostics"]["admissible"]
world=np.asarray(f["world_positions_by_pose"]); J=np.asarray(f["jacobians_by_pose"])
effect=np.einsum("pnij,njk->pnik",J,D[ref])
assert float(np.max(np.abs(effect)))>0
from scipy.sparse import coo_matrix
Arows=[];Acols=[];Avalues=[];rhs=[];labels=[]
local_variables={int(np.searchsorted(ref,v)):i for i,v in enumerate(affected)}
nv=len(affected)

margin=.00025
def add_constraint(a,b,da,db,axis,label,requested_margin,ia,ib=None):
 # Sparse per-vertex retention factors on the same predeclared tapered field.
 for i in range(3):
  for k in range(3):
   row=len(rhs)
   vi=local_variables.get(int(ia[i]))
   vk=None if ib is None else local_variables.get(int(ib[k]))
   if vi is not None:
    Arows.append(row);Acols.append(vi);Avalues.append(float(axis@da[i].sum(axis=-1)))
   if vk is not None:
    Arows.append(row);Acols.append(vk);Avalues.append(float(-axis@db[k].sum(axis=-1)))
   rhs.append(float((b[k]-a[i])@axis-requested_margin));labels.append(label)
for pi,(_,step) in enumerate(poses):
 witness=B/"heldout-002/scan-output"/f"step-{step}.self-witnesses.jsonl"
 if not witness.exists(): continue
 inputs.append(witness);before[str(witness)]=sha(witness)
 for line in witness.read_text().splitlines():
  r=json.loads(line);sa=int(r["skin_source_face_row"]);sb=int(r["target_surface_face_row"])
  ia=faces[sa];ib=faces[sb]
  axis,gap=bestplane(np.asarray(capture)[pi,ia],np.asarray(capture)[pi,ib])
  assert gap>0
  add_constraint(world[pi,ia],world[pi,ib],effect[pi,ia],effect[pi,ib],axis,{"kind":"known_self","step":step,"faces":[sa,sb],"baseline_separating_gap_mm":gap*1000},margin,ia,ib)
# Conservative fixed separating planes for each local swept AABB neighbor.
pi=len(poses)-1
diagnostics=[]; paircount=0
for key,tt in targets.items():
 tlo=tt.min(axis=1);thi=tt.max(axis=1)
 for fi in affected_faces:
  ids=faces[fi];a=world[pi,ids];da=effect[pi,ids]
  lo=np.min(a+np.minimum(da,0).sum(axis=2),axis=0)-margin
  hi=np.max(a+np.maximum(da,0).sum(axis=2),axis=0)+margin
  neigh=np.flatnonzero(np.all(thi>=lo,axis=1)&np.all(tlo<=hi,axis=1))
  for ti in neigh:
   axis,gap=bestplane(a,tt[ti])
   if gap < -1e-9: raise RuntimeError("claimed clear candidate lacks separating plane")
   # Preserve existing clearance if below requested engineering margin.
   keep=min(margin,max(0.,gap))
   add_constraint(a,tt[ti],da,np.zeros((3,3,2)),axis,{"kind":"settled_target","step":20000,"skin_face":int(fi),"target":list(key),"target_face":int(ti),"initial_gap_mm":gap*1000},keep,ids)
   diagnostics.append(gap);paircount+=1
A=coo_matrix((Avalues,(Arows,Acols)),shape=(len(rhs),nv)).tocsr()
rhs=np.asarray(rhs)
# First separate the original uniform-patch conflict from per-vertex feasibility.
side=np.stack([m[affected]>0 for m in masks],axis=1).astype(float)
two=np.asarray(A@side); selfrows=np.asarray([x["kind"]=="known_self" for x in labels])
from scipy.optimize import minimize,LinearConstraint,Bounds
selfonly=linprog(np.ones(2),A_ub=two[selfrows],b_ub=rhs[selfrows],bounds=[(0,1),(0,1)],method="highs")
targetcaps=[]
for col in range(2):
 cost=np.zeros(2);cost[col]=-1
 z=linprog(cost,A_ub=two[~selfrows],b_ub=rhs[~selfrows],bounds=[(0,1),(0,1)],method="highs")
 targetcaps.append({"side":col,"success":z.success,"max_fraction":float(z.x[col]) if z.success else None})
conflict={"self_only_uniform_fractions":selfonly.x.tolist() if selfonly.success else None,"target_only_uniform_caps":targetcaps}
if selfonly.success:
 violation=two@selfonly.x-rhs
 worst=np.argsort(violation)[-8:][::-1]
 conflict["worst_target_constraints_at_self_only_solution"]=[{"label":labels[int(i)],"coefficients":two[int(i)].tolist(),"rhs_mm":float(rhs[i]*1000),"violation_mm":float(violation[i]*1000)} for i in worst]
delta=D[affected].sum(axis=-1)*1000.
# Geometry-energy objective: source change plus differences along original mesh edges.
Q=np.diag(np.sum(delta*delta,axis=1))
edges=np.unique(np.sort(np.concatenate([skin["faces"][:,[0,1]],skin["faces"][:,[1,2]],skin["faces"][:,[2,0]]]),axis=1),axis=0)
vm={int(v):i for i,v in enumerate(affected)}
for a,b in edges:
 i=vm.get(int(a));k=vm.get(int(b))
 if i is None and k is None: continue
 if i is not None: Q[i,i]+=float(delta[i]@delta[i])
 if k is not None: Q[k,k]+=float(delta[k]@delta[k])
 if i is not None and k is not None:
  v=float(delta[i]@delta[k]);Q[i,k]-=v;Q[k,i]-=v
lp=linprog(np.linalg.norm(delta,axis=1),A_ub=A*1000,b_ub=rhs*1000,bounds=[(0,1)]*nv,method="highs")
report={"status":"infeasible" if not lp.success else "feasible_pending_exact_full_checks","method":"per-vertex bounded reductions of existing correction within the same cubic three-edge-ring support; convex source-change plus edge-difference energy; no anatomy or native admission","affected_vertices":affected.tolist(),"affected_face_count":len(affected_faces),"region_vertex_counts":[len(r) for r in regions],"known_self_pairs":30,"settled_target_pairs_constrained":paircount,"constraint_count":len(rhs),"target_min_initial_separating_gap_mm":min(diagnostics)*1000,"uniform_mode_conflict":conflict,"lp_status":lp.message,"inputs_before":before,"map_report":mr}
if lp.success:
 # Preserve every unreduced inequality for postvalidation; reduce redundant
 # singleton inequalities to scalar bounds for the QP's numerical conditioning.
 lower=np.zeros(nv);upper=np.ones(nv); coupled=[]; zero_count=0
 for ri in range(A.shape[0]):
  row=A.getrow(ri); nz=np.flatnonzero(row.data!=0.)
  if len(nz)==0:
   if rhs[ri] < -1e-12: raise RuntimeError("infeasible constant constraint")
   zero_count+=1
  elif len(nz)==1:
   k=nz[0];col=row.indices[k];value=row.data[k];bound=rhs[ri]/value
   if value>0: upper[col]=min(upper[col],bound)
   else: lower[col]=max(lower[col],bound)
  else: coupled.append(ri)
 if np.any(lower-upper>1e-9): raise RuntimeError("incompatible singleton bounds")
 # Sub-picometre projection roundoff may produce a negative upper bound at zero.
 narrow=lower>upper
 upper[narrow]=lower[narrow]
 reduced=A[coupled].toarray()*1000;rrhs=rhs[coupled]*1000
 seed=np.clip(lp.x,lower,upper)
 opt=minimize(lambda x:float(x@Q@x),seed,jac=lambda x:2*Q@x,method="SLSQP",bounds=Bounds(lower,upper),constraints=[LinearConstraint(reduced,np.full(len(rrhs),-np.inf),rrhs)],options={"ftol":1e-10,"maxiter":200})
 x=opt.x
 report["qp_constraint_reduction"]={"coupled_rows":len(coupled),"constant_rows":zero_count,"singleton_rows":len(rhs)-len(coupled)-zero_count,"bounds_roundoff_count":int(narrow.sum()),"unreduced_postcheck_required":True}
 if not opt.success or float((A@x-rhs).max())>1e-10:
  report.update({"status":"qp_failed_no_proposal","solver_success":bool(opt.success),"solver_status":opt.message})
  write(OUT/"feasibility.json",report)
  raise RuntimeError("QP did not complete; no fallback proposal")
 proposal=candidate.copy()
 proposal[affected]=np.asarray(candidate[affected].astype(float)+D[affected].sum(axis=-1)*x[:,None],dtype="<f4")
 out=forward(proposal);pw=np.asarray(out["world_positions_by_pose"])
 predicted=world.copy()
 ids=np.searchsorted(ref,affected)
 predicted[:,ids]+=effect[:,ids].sum(axis=-1)*x[None,:,None]
 error=float(np.linalg.norm(pw-predicted,axis=2).max())
 assert np.array_equal(proposal[protected],candidate[protected])
 assert np.array_equal(proposal[~np.isin(np.arange(len(proposal)),affected)],candidate[~np.isin(np.arange(len(proposal)),affected)])
 np.save(OUT/"unadmitted-source-proposal.npy",proposal,allow_pickle=False)
 report.update({"reduction_fractions":x.tolist(),"solver_success":bool(opt.success),"solver_status":opt.message,"maximum_linear_constraint_violation_m":float((A@x-rhs).max()),"maximum_F32_forward_linearization_error_um":error*1e6,"maximum_source_change_mm":float(np.linalg.norm(proposal-candidate,axis=1).max()*1000),"proposal":pin(OUT/"unadmitted-source-proposal.npy")})
from scipy.sparse import save_npz
save_npz(OUT/"constraint-matrix.npz",A)
np.save(OUT/"constraint-rhs.npy",rhs,allow_pickle=False)
write(OUT/"constraint-labels.json",labels)
after={p:sha(p) for p in before};assert before==after;report["inputs_after"]=after;report["inputs_unchanged"]=True
write(OUT/"feasibility.json",report)
print(json.dumps({k:v for k,v in report.items() if k not in ("inputs_before","inputs_after","map_report","affected_vertices")}),flush=True)
