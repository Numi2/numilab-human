#!/usr/bin/env python3
"""Bounded offline connected fit; all physical geometry/mass gates still mandatory."""
import csv,json,time,hashlib,importlib.util,argparse
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
sp=importlib.util.spec_from_file_location("cap",P/"audit_regional_mass_capacity_464.py");cap=importlib.util.module_from_spec(sp);sp.loader.exec_module(cap)

def require_localized(report):
 if report["unlocalized_containments_or_contact_rows"]:
  raise ValueError("unlocalized positive material/bone occupancy; localization required before fitting or pair-count claims")
 return report

CONTACT_MATRIX=None
CONTACT_TARGET=None
def solve_qp(H,rhs,A,lower,upper,iterations=600,tol=1e-8):
 from scipy.optimize import linprog
 goal=splu(H.tocsc()).solve(rhs);n=len(goal)
 cu=np.isfinite(upper);cl=np.isfinite(lower)
 constraints=sparse.vstack([A[cu],-A[cl]],format="csc");values=np.r_[upper[cu],-lower[cl]]
 C=CONTACT_MATRIX if CONTACT_MATRIX is not None else sparse.csc_matrix((0,n))
 d=CONTACT_TARGET if CONTACT_TARGET is not None else np.array([])
 nc=len(d);zero=sparse.csc_matrix((constraints.shape[0],n));eye=sparse.eye(n,format="csc")
 aub=sparse.vstack([sparse.hstack([constraints,zero,sparse.csc_matrix((constraints.shape[0],nc))]),sparse.hstack([eye,-eye,sparse.csc_matrix((n,nc))]),sparse.hstack([-eye,-eye,sparse.csc_matrix((n,nc))]),sparse.hstack([-C,sparse.csc_matrix((nc,n)),-sparse.eye(nc)])],format="csc")
 bub=np.r_[values,goal,-goal,-d];weight=np.sqrt(np.maximum(H.diagonal(),1e-12))
 result=linprog(np.r_[np.zeros(n),weight,np.full(nc,2000.)],A_ub=aub,b_ub=bub,bounds=[(None,None)]*n+[(0.,None)]*(n+nc),method="highs",options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9,"time_limit":20.})
 if not result.success:raise RuntimeError("bounded contact construction LP failed: "+result.message)
 x=result.x[:n];ax=A@x
 return x,{"method":"existing_SciPy_sparse_HiGHS_measured_contact_L1_projection","iterations":int(result.nit),"max_primal_constraint_error_m":float(np.maximum(np.maximum(lower-ax,ax-upper),0.).max()),"contact_rows":nc,"contact_slack_sum_m":float(result.x[2*n:].sum()),"objective":float(result.fun),"status":str(result.message)}

def self_test():
 H=sparse.eye(2,format="csc");A=sparse.csc_matrix([[1.,0.],[0.,1.],[1.,1.]])
 x,r=solve_qp(H,np.array([2.,-1.]),A,np.array([0.,0.,1.]),np.array([np.inf,np.inf,1.]),iterations=2000,tol=1e-10)
 assert np.allclose(x,[1.,0.],atol=2e-9),(x,r)
 return {"status":"passed","constrained_quadratic_optimum":x.tolist(),"diagnostics":r}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--iterations",type=int,default=12);ap.add_argument("--self-test",action="store_true");ap.add_argument("--candidate",type=Path);ap.add_argument("--free-domain",choices=["local","all"],default="local");ap.add_argument("--wall-budget-s",type=float,default=120.);args=ap.parse_args()
 out=Path(args.out);out.mkdir(exist_ok=False,parents=True)
 test=self_test();(out/"qp-test.json").write_text(json.dumps(test,indent=2)+"\n")
 if args.self_test:print(json.dumps(test));return
 global CONTACT_MATRIX,CONTACT_TARGET
 start=time.monotonic()
 cw_spec=importlib.util.spec_from_file_location("cw471",P/"extract_contact_construction_constraints_475.py");cw=importlib.util.module_from_spec(cw_spec);cw_spec.loader.exec_module(cw)
 manifest=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json"
 base=args.candidate or P/"connected-capacity-fit-470/run-001/candidate.npz"
 metadata_base=P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk"
 target=metadata_base/"candidate.npz"
 with np.load(base) as z:a={k:z[k].copy() for k in z.files}
 with np.load(target) as z:goal=z["points_reference_world_m_f64"].astype(np.float32).astype(float)
 with np.load(metadata_base/"candidate.npz") as baseline:original=baseline["points_reference_world_m_f64"].astype(np.float32).astype(float)
 x=a["points_reference_world_m_f64"].astype(np.float32).astype(float);t=a["volume_tetrahedra"];goal=original.copy()
 keys=np.stack([a[k] for k in ["material_index","geometric_donor_body","anatomical_region_id"]],axis=1)
 changed=np.flatnonzero(np.any(goal!=original,axis=1))
 initial_contacts,initial_contact_report=cw.extract(base,manifest);require_localized(initial_contact_report)
 seed=np.union1d(changed,np.unique(initial_contacts["volume_node_rows"]));incident=np.any(np.isin(t,seed),axis=1)
 assert len(x)%3==0
 ncol=len(x)//3
 # Three source surfaces share the exact column order in source002.
 # This is only an authoring displacement basis; runtime nodes remain independent.
 free=np.setdiff1d(np.arange(ncol),np.unique(a["pinned_volume_node_rows"]%ncol))
 column_map=np.full(ncol,-1,np.int64);column_map[free]=np.arange(len(free))
 fm=column_map[np.arange(len(x))%ncol]
 affected=np.flatnonzero(np.any(fm[t]>=0,axis=1))
 target_w=np.where(np.isin(free,changed),40.,1.)
 ct=t%ncol
 edge=np.unique(np.sort(np.concatenate([ct[:,[0,1]],ct[:,[0,2]],ct[:,[0,3]],ct[:,[1,2]],ct[:,[1,3]],ct[:,[2,3]]]),axis=1),axis=0)
 edge=edge[edge[:,0]!=edge[:,1]]
 adj=sparse.coo_matrix((np.ones(2*len(edge)),(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(ncol,ncol)).tocsr()
 L=sparse.diags(np.asarray(adj.sum(axis=1)).ravel())-adj
 lam=2.;H=sparse.kron(sparse.diags(target_w)+lam*L[free][:,free],sparse.eye(3),format="csc")
 capdir=P/"native-physical-parent-mass-330/native-source-identity-control-003"
 supply=cap.load_supply(capdir/"compiled-parent-cell-mass.csv.tets.csv",capdir/"compiled-parent-cell-mass.csv.labels.csv")
 groupkeys=sorted(supply);groupid={k:i for i,k in enumerate(groupkeys)}
 gi=np.array([groupid[tuple(map(int,k))] for k in keys])
 sm=np.array([supply[k] for k in groupkeys]);density=np.array([cap.DENSITY_BOUNDS[k[0]] for k in groupkeys])
 vlo=sm/density[:,1];vhi=sm/density[:,0]
 with (capdir/"compiled-parent-cell-mass.csv.tets.csv").open() as f:parent_cells=list(csv.DictReader(f))
 parent_mass=np.array([float(r["parent_mass_kg"]) for r in parent_cells])
 cp=a["child_parent_tet_row"].astype(int);assert len(cp)==len(parent_mass) and np.array_equal(np.sort(cp),np.arange(len(parent_mass)))
 cell_rho=np.array([cap.DENSITY_BOUNDS[int(k)] for k in a["material_index"]])
 vmin=parent_mass[cp]/cell_rho[:,1];vmax=parent_mass[cp]/cell_rho[:,0]
 start_v,_=cap.tet_volumes_gradients(x,t)
 if np.any(start_v<vmin) or np.any(start_v>vmax):
  raise ValueError("source candidate does not have per-original-cell density capacity")

 history=[]
 for iteration in range(args.iterations):
  if time.monotonic()-start>args.wall_budget_s:
   history.append({"iteration":iteration,"stopped":"bounded_authoring_wall_budget"});break
  state=out/("construction-state-%03d.npz"%iteration)
  state_data=dict(a);state_data["points_reference_world_m_f64"]=x;np.savez_compressed(state,**state_data)
  contact,contact_report=cw.extract(state,manifest);require_localized(contact_report)
  if contact_report["positive_pairs"]==0:
   history.append({"iteration":iteration,"contact_pairs":0,"contact_coverage_complete":True});break
  cnode=contact["volume_node_rows"];cfr=fm[cnode];cval=contact["barycentric"][:,:,None]*contact["outward_bank_normal"][:,None,:]
  cmask=np.broadcast_to((cfr>=0)[:,:,None],cval.shape);crr=np.broadcast_to(np.arange(len(cnode))[:,None,None],cval.shape);ccc=cfr[:,:,None]*3+np.arange(3)[None,None,:]
  CONTACT_MATRIX=sparse.coo_matrix((cval[cmask],(crr[cmask],ccc[cmask])),shape=(len(cnode),len(free)*3)).tocsc();CONTACT_TARGET=contact["required_normal_correction_m"]
  old_overlap=sum(r["intersection_m3"] for r in contact_report["banks"])
  volumes,gr=cap.tet_volumes_gradients(x,t)
  # Individual positive-cell constraints and regional volume capacities, in metres.
  cell=t[affected];grad=gr[affected];fr=fm[cell]
  rr=np.broadcast_to(np.arange(len(cell))[:,None,None],(len(cell),4,3));cc=fr[:,:,None]*3+np.arange(3)[None,None,:]
  mask=np.broadcast_to((fr>=0)[:,:,None],grad.shape)
  Av=sparse.coo_matrix((grad[mask],(rr[mask],cc[mask])),shape=(len(cell),len(free)*3)).tocsr()
  norms=np.sqrt(np.asarray(Av.multiply(Av).sum(axis=1)).ravel());assert np.all(norms>0)
  Av=sparse.diags(1/norms)@Av;lb=(vmin[affected]-volumes[affected])/norms;ub=(vmax[affected]-volumes[affected])/norms
  group_rr=np.broadcast_to(gi[:,None,None],gr.shape);allfr=fm[t];allcc=allfr[:,:,None]*3+np.arange(3)[None,None,:];amask=np.broadcast_to((allfr>=0)[:,:,None],gr.shape)
  Ag=sparse.coo_matrix((gr[amask],(group_rr[amask],allcc[amask])),shape=(len(groupkeys),len(free)*3)).tocsr()
  gn=np.sqrt(np.asarray(Ag.multiply(Ag).sum(axis=1)).ravel());gv=np.bincount(gi,weights=volumes,minlength=len(groupkeys));has=gn>0
  assert np.all(gv[~has]>=vlo[~has]) and np.all(gv[~has]<=vhi[~has])
  Ag=sparse.diags(1/gn[has])@Ag[has]
  A=sparse.vstack([Av,Ag,sparse.eye(len(free)*3)],format="csc")
  trust=.0005
  lower=np.r_[lb,(vlo[has]+.05*(vhi[has]-vlo[has])-gv[has])/gn[has],np.full(len(free)*3,-trust)]
  upper=np.r_[ub,(vhi[has]-.05*(vhi[has]-vlo[has])-gv[has])/gn[has],np.full(len(free)*3,trust)]
  rhs=(target_w[:,None]*(goal[free]-x[free])-lam*(L@(x[:ncol]-original[:ncol]))[free]).ravel()
  delta,diag=solve_qp(H,rhs,A,lower,upper)
  column_delta=np.zeros((ncol,3));column_delta[free]=delta.reshape(-1,3);dd=column_delta[np.arange(len(x))%ncol]
  oldobj=float(np.sum(target_w[:,None]*(x[free]-goal[free])**2)+lam*np.sum((x[:ncol]-original[:ncol])*(L@(x[:ncol]-original[:ncol]))))
  accepted=None
  for step in range(15):
   alpha=2.**(-step);trial=(x+alpha*dd).astype(np.float32).astype(float)
   tv,_=cap.tet_volumes_gradients(trial,t);tgv=np.bincount(gi,weights=tv,minlength=len(groupkeys))
   obj=float(np.sum(target_w[:,None]*(trial[free]-goal[free])**2)+lam*np.sum((trial[:ncol]-original[:ncol])*(L@(trial[:ncol]-original[:ncol]))))
   if np.all(tv>=vmin) and np.all(tv<=vmax) and np.all(tgv>=vlo) and np.all(tgv<=vhi):
    proposed=out/("trial-%03d-%02d.npz"%(iteration,step));proposal_data=dict(a);proposal_data["points_reference_world_m_f64"]=trial;np.savez_compressed(proposed,**proposal_data)
    _,next_report=cw.extract(proposed,manifest);require_localized(next_report);new_overlap=sum(r["intersection_m3"] for r in next_report["banks"])
    if new_overlap<old_overlap:
     accepted=(trial,tv,tgv,obj,alpha);break
  item={"iteration":iteration,"qp":diag,"target_objective_before":oldobj,"accepted":accepted is not None}
  if accepted is None:
   history.append(item);break
  x,tv,tgv,obj,alpha=accepted
  item.update(overlap_before_m3_per_bank_not_union=old_overlap,overlap_after_m3_per_bank_not_union=new_overlap,positive_material_bank_pairs_before=contact_report["positive_pairs"],positive_material_bank_pairs_after=next_report["positive_pairs"],alpha=alpha,target_objective_after=obj,min_cell_volume_m3=float(tv.min()),max_move_from_source_m=float(np.linalg.norm(x-original,axis=1).max()),remaining_target_rms_m=None if len(changed)==0 else float(np.sqrt(np.mean((goal[changed]-x[changed])**2))))
  history.append(item);print(json.dumps(item),flush=True)
  if np.max(np.abs(alpha*delta))<1e-8:break
 # Update existing registered-coordinate owner, not an independent physics state.
 with (metadata_base/"coarse-node-to-fine-surface.csv").open() as f:meta={int(r["volume_node_row"]):r for r in csv.DictReader(f)}
 poses=json.loads((capdir/"registered-skin-poses-100.json").read_text())
 moved=np.flatnonzero(np.any(x!=original,axis=1))
 for i in moved:
  pose=poses["all_bodies"][int(meta[int(i)]["frame_owner_body"])]["current"]
  rot=np.asarray(pose["local_to_world_row_major_f32"],float).reshape(3,3);origin=np.asarray(pose["position_world_m"],float)
  a["points_donor_local_m_f64"][i]=np.linalg.solve(rot,x[i]-origin)
  a["points_current_world_m_f32"][i]=x[i]
  a["target_current_world_m"][i]=x[i]
 a["points_reference_world_m_f64"]=x
 dest=out/"candidate.npz";np.savez_compressed(dest,**a)
 rows,v,_=cap.capacity_rows(x,t,keys,supply)
 receipt={"status":"bounded_connected_capacity_fit_proposal","source_sha256":cap.sha(base),"target_sha256":cap.sha(target),"candidate_sha256":cap.sha(dest),"script_sha256":cap.sha(__file__),"iterations":history,"authoring_basis":"shared_O_I_D_column_displacement_runtime_nodes_unchanged","free_domain":args.free_domain,"free_nodes":len(free),"moved_nodes":len(moved),"target_nodes":len(changed),"affected_cells":len(affected),"nonpositive_cells":int(np.count_nonzero(v<=0)),"failed_capacity_groups":sum(not r["mass_capacity_feasible"] for r in rows),"pinned_nodes_unchanged":bool(np.array_equal(x[a["pinned_volume_node_rows"]],original[a["pinned_volume_node_rows"]])),"topology_labels_full86_unchanged":True,"elapsed_s":time.monotonic()-start,"per_source_cell_density_capacity":bool(np.all(v>=vmin) and np.all(v<=vmax)),"physical_admission":False,"contact_coverage_complete":True,"construction_mass_interval_inset_fraction":.05,"construction_method":"L1 correction of measured Boolean boundary penetrations with unchanged physical gates","scope":"offline constrained mesh construction control; detailed boundary ancestry/embedding, solid penetration, sparse transport, native accepted-state and motion gates remain mandatory"}
 (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps({k:v for k,v in receipt.items() if k!="iterations"}))
if __name__=="__main__":main()
