#!/usr/bin/env python3
"""Bounded offline connected fit; all physical geometry/mass gates still mandatory."""
import csv,json,time,hashlib,importlib.util,argparse
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
sp=importlib.util.spec_from_file_location("cap",P/"audit_regional_mass_capacity_464.py");cap=importlib.util.module_from_spec(sp);sp.loader.exec_module(cap)

def solve_qp(H,rhs,A,lower,upper,iterations=600,tol=2e-8):
 rho=20.;factor=splu((H+rho*(A.T@A)).tocsc())
 x=np.zeros(H.shape[0]);z=np.minimum(np.maximum(A@x,lower),upper);u=np.zeros(A.shape[0])
 history=[]
 for k in range(iterations):
  x=factor.solve(rhs+rho*(A.T@(z-u)))
  ax=A@x;old=z;z=np.minimum(np.maximum(ax+u,lower),upper);u+=ax-z
  primal=float(np.max(np.abs(ax-z))) if len(z) else 0.
  dual=float(np.max(np.abs(rho*(A.T@(z-old))))) if len(x) else 0.
  if k%100==0:history.append([k,primal,dual])
  if primal<tol and dual<tol:break
 return x,{"iterations":k+1,"max_primal_constraint_error_m":primal,"max_dual_error":dual,"sampled":history}

def self_test():
 H=sparse.eye(2,format="csc");A=sparse.csc_matrix([[1.,0.],[0.,1.],[1.,1.]])
 x,r=solve_qp(H,np.array([2.,-1.]),A,np.array([0.,0.,1.]),np.array([np.inf,np.inf,1.]),iterations=2000,tol=1e-10)
 assert np.allclose(x,[1.,0.],atol=2e-9),(x,r)
 return {"status":"passed","constrained_quadratic_optimum":x.tolist(),"diagnostics":r}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",required=True);ap.add_argument("--iterations",type=int,default=12);ap.add_argument("--self-test",action="store_true");args=ap.parse_args()
 out=Path(args.out);out.mkdir(exist_ok=False,parents=True)
 test=self_test();(out/"qp-test.json").write_text(json.dumps(test,indent=2)+"\n")
 if args.self_test:print(json.dumps(test));return
 start=time.monotonic()
 base=P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz"
 target=P/"reduced-physical-boundary-400/attempt-012-combined-safe-targets/candidate.npz"
 with np.load(base) as z:a={k:z[k].copy() for k in z.files}
 with np.load(target) as z:goal=z["points_reference_world_m_f64"].astype(np.float32).astype(float)
 original=a["points_reference_world_m_f64"].astype(np.float32).astype(float);x=original.copy();t=a["volume_tetrahedra"]
 keys=np.stack([a[k] for k in ["material_index","geometric_donor_body","anatomical_region_id"]],axis=1)
 changed=np.flatnonzero(np.any(goal!=original,axis=1));incident=np.any(np.isin(t,changed),axis=1)
 free=np.setdiff1d(np.unique(t[incident]),a["pinned_volume_node_rows"])
 # All influence remains local to incident cell nodes; every affected cell is checked.
 fm=np.full(len(x),-1,np.int64);fm[free]=np.arange(len(free))
 affected=np.flatnonzero(np.any(fm[t]>=0,axis=1))
 target_w=np.where(np.isin(free,changed),40.,1.)
 edge=np.unique(np.sort(np.concatenate([t[:,[0,1]],t[:,[0,2]],t[:,[0,3]],t[:,[1,2]],t[:,[1,3]],t[:,[2,3]]]),axis=1),axis=0)
 adj=sparse.coo_matrix((np.ones(2*len(edge)),(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(len(x),len(x))).tocsr()
 L=sparse.diags(np.asarray(adj.sum(axis=1)).ravel())-adj
 lam=2.;H=sparse.kron(sparse.diags(target_w)+lam*L[free][:,free],sparse.eye(3),format="csc")
 capdir=P/"native-physical-parent-mass-330/native-source-identity-control-003"
 supply=cap.load_supply(capdir/"compiled-parent-cell-mass.csv.tets.csv",capdir/"compiled-parent-cell-mass.csv.labels.csv")
 groupkeys=sorted(supply);groupid={k:i for i,k in enumerate(groupkeys)}
 gi=np.array([groupid[tuple(map(int,k))] for k in keys])
 sm=np.array([supply[k] for k in groupkeys]);density=np.array([cap.DENSITY_BOUNDS[k[0]] for k in groupkeys])
 vlo=sm/density[:,1];vhi=sm/density[:,0]
 v0,_=cap.tet_volumes_gradients(original,t);vmin=v0*.02
 history=[]
 for iteration in range(args.iterations):
  volumes,gr=cap.tet_volumes_gradients(x,t)
  # Individual positive-cell constraints and regional volume capacities, in metres.
  cell=t[affected];grad=gr[affected];fr=fm[cell]
  rr=np.broadcast_to(np.arange(len(cell))[:,None,None],(len(cell),4,3));cc=fr[:,:,None]*3+np.arange(3)[None,None,:]
  mask=np.broadcast_to((fr>=0)[:,:,None],grad.shape)
  Av=sparse.coo_matrix((grad[mask],(rr[mask],cc[mask])),shape=(len(cell),len(free)*3)).tocsr()
  norms=np.sqrt(np.asarray(Av.multiply(Av).sum(axis=1)).ravel());assert np.all(norms>0)
  Av=sparse.diags(1/norms)@Av;lb=(vmin[affected]-volumes[affected])/norms;ub=np.full(len(cell),np.inf)
  group_rr=np.broadcast_to(gi[:,None,None],gr.shape);allfr=fm[t];allcc=allfr[:,:,None]*3+np.arange(3)[None,None,:];amask=np.broadcast_to((allfr>=0)[:,:,None],gr.shape)
  Ag=sparse.coo_matrix((gr[amask],(group_rr[amask],allcc[amask])),shape=(len(groupkeys),len(free)*3)).tocsr()
  gn=np.sqrt(np.asarray(Ag.multiply(Ag).sum(axis=1)).ravel());gv=np.bincount(gi,weights=volumes,minlength=len(groupkeys));has=gn>0
  assert np.all(gv[~has]>=vlo[~has]) and np.all(gv[~has]<=vhi[~has])
  Ag=sparse.diags(1/gn[has])@Ag[has]
  A=sparse.vstack([Av,Ag,sparse.eye(len(free)*3)],format="csc")
  trust=.0005
  lower=np.r_[lb,(vlo[has]-gv[has])/gn[has],np.full(len(free)*3,-trust)]
  upper=np.r_[ub,(vhi[has]-gv[has])/gn[has],np.full(len(free)*3,trust)]
  rhs=(target_w[:,None]*(goal[free]-x[free])-lam*(L@(x-original))[free]).ravel()
  delta,diag=solve_qp(H,rhs,A,lower,upper)
  dd=np.zeros_like(x);dd[free]=delta.reshape(-1,3)
  oldobj=float(np.sum(target_w[:,None]*(x[free]-goal[free])**2)+lam*np.sum((x-original)*(L@(x-original))))
  accepted=None
  for step in range(15):
   alpha=2.**(-step);trial=(x+alpha*dd).astype(np.float32).astype(float)
   tv,_=cap.tet_volumes_gradients(trial,t);tgv=np.bincount(gi,weights=tv,minlength=len(groupkeys))
   obj=float(np.sum(target_w[:,None]*(trial[free]-goal[free])**2)+lam*np.sum((trial-original)*(L@(trial-original))))
   if np.all(tv>0) and np.all(tgv>=vlo) and np.all(tgv<=vhi) and obj<oldobj:
    accepted=(trial,tv,tgv,obj,alpha);break
  item={"iteration":iteration,"qp":diag,"target_objective_before":oldobj,"accepted":accepted is not None}
  if accepted is None:
   history.append(item);break
  x,tv,tgv,obj,alpha=accepted
  item.update(alpha=alpha,target_objective_after=obj,min_cell_volume_m3=float(tv.min()),max_move_from_source_m=float(np.linalg.norm(x-original,axis=1).max()),remaining_target_rms_m=float(np.sqrt(np.mean((goal[changed]-x[changed])**2))))
  history.append(item);print(json.dumps(item),flush=True)
  if np.max(np.abs(alpha*delta))<1e-8:break
 # Update existing registered-coordinate owner, not an independent physics state.
 with (base.parent/"coarse-node-to-fine-surface.csv").open() as f:meta={int(r["volume_node_row"]):r for r in csv.DictReader(f)}
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
 receipt={"status":"bounded_connected_capacity_fit_proposal","source_sha256":cap.sha(base),"target_sha256":cap.sha(target),"candidate_sha256":cap.sha(dest),"script_sha256":cap.sha(__file__),"iterations":history,"free_nodes":len(free),"moved_nodes":len(moved),"target_nodes":len(changed),"affected_cells":len(affected),"nonpositive_cells":int(np.count_nonzero(v<=0)),"failed_capacity_groups":sum(not r["mass_capacity_feasible"] for r in rows),"pinned_nodes_unchanged":bool(np.array_equal(x[a["pinned_volume_node_rows"]],original[a["pinned_volume_node_rows"]])),"topology_labels_full86_unchanged":True,"elapsed_s":time.monotonic()-start,"physical_admission":False,"scope":"offline constrained mesh construction control; detailed boundary ancestry/embedding, solid penetration, sparse transport, native accepted-state and motion gates remain mandatory"}
 (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n");print(json.dumps({k:v for k,v in receipt.items() if k!="iterations"}))
if __name__=="__main__":main()
