#!/usr/bin/env python3
"""Offline nonlinear feasibility restoration of an existing connected proposal."""
import argparse,csv,json,time,hashlib,importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
def load(name,file):
 s=importlib.util.spec_from_file_location(name,P/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
cap=load("cap503","audit_regional_mass_capacity_464.py")
fit=load("fit503","fit_connected_measured_contact_501.py")
cw=load("cw503","extract_contact_construction_constraints_475.py")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path,required=True);ap.add_argument("--alpha",type=float,default=.5);ap.add_argument("--iterations",type=int,default=8);args=ap.parse_args()
 args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 src=P/"connected-step-blockers-502/run-001/linear-proposal-000.npz"
 with np.load(src) as z:x0=z["current_reference"];dd=z["proposal_delta"]
 base=P/"connected-measured-contact-501/all-free-001/candidate.npz"
 with np.load(base) as z:a={k:z[k].copy() for k in z.files}
 t=a["volume_tetrahedra"];keys=np.stack([a[k] for k in ["material_index","geometric_donor_body","anatomical_region_id"]],axis=1)
 meta=P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk"
 with np.load(meta/"candidate.npz") as z:original=z["points_reference_world_m_f64"].astype(np.float32).astype(float)
 parent=P/"native-physical-parent-mass-330/native-source-identity-control-003"
 supply=cap.load_supply(parent/"compiled-parent-cell-mass.csv.tets.csv",parent/"compiled-parent-cell-mass.csv.labels.csv")
 groupkeys=sorted(supply);gmap={k:i for i,k in enumerate(groupkeys)};gi=np.array([gmap[tuple(map(int,k))] for k in keys])
 mass=np.array([supply[k] for k in groupkeys]);rho=np.array([cap.DENSITY_BOUNDS[k[0]] for k in groupkeys])
 vlo=mass/rho[:,1];vhi=mass/rho[:,0];vmin=cap.tet_volumes_gradients(original,t)[0]*.02
 pins=a["pinned_volume_node_rows"]
 x=(x0+args.alpha*dd).astype(np.float32).astype(float);target=x.copy()
 hist=[]
 def stats(x):
  v,gr=cap.tet_volumes_gradients(x,t);gv=np.bincount(gi,weights=v,minlength=len(groupkeys))
  cv=np.maximum(vmin-v,0);gc=np.maximum(np.maximum(vlo-gv,gv-vhi),0)
  merit=float(max(np.max(cv/np.maximum(vmin,1e-20)),np.max(gc/np.maximum(vlo,1e-20))))
  return v,gr,gv,merit
 for it in range(args.iterations):
  v,gr,gv,merit=stats(x);bc=np.flatnonzero(v<vmin);bg=np.flatnonzero((gv<vlo)|(gv>vhi))
  item={"iteration":it,"nonpositive_cells":int(np.count_nonzero(v<=0)),"below_construction_volume_floor":len(bc),"failed_capacity_groups":len(bg),"scaled_exact_violation":merit}
  if not len(bc) and not len(bg):hist.append(item);break
  seed=np.unique(t[np.union1d(bc,np.flatnonzero(np.isin(gi,bg)))])
  incident=np.any(np.isin(t,seed),axis=1);free=np.setdiff1d(np.unique(t[incident]),pins);fm=np.full(len(x),-1,np.int64);fm[free]=np.arange(len(free))
  affected=np.flatnonzero(np.any(fm[t]>=0,axis=1));cell=t[affected];g=gr[affected];fr=fm[cell]
  rr=np.broadcast_to(np.arange(len(cell))[:,None,None],g.shape);cc=fr[:,:,None]*3+np.arange(3)[None,None,:];mask=np.broadcast_to((fr>=0)[:,:,None],g.shape)
  Av=sparse.coo_matrix((g[mask],(rr[mask],cc[mask])),shape=(len(cell),len(free)*3)).tocsr()
  vn=np.sqrt(np.asarray(Av.multiply(Av).sum(axis=1)).ravel());assert np.all(vn>0)
  Av=sparse.diags(1/vn)@Av
  ar=np.broadcast_to(gi[:,None,None],gr.shape);fr=fm[t];ac=fr[:,:,None]*3+np.arange(3)[None,None,:];mask=np.broadcast_to((fr>=0)[:,:,None],gr.shape)
  Ag=sparse.coo_matrix((gr[mask],(ar[mask],ac[mask])),shape=(len(groupkeys),len(free)*3)).tocsr()
  gn=np.sqrt(np.asarray(Ag.multiply(Ag).sum(axis=1)).ravel());has=gn>0
  assert np.all(gv[~has]>=vlo[~has]) and np.all(gv[~has]<=vhi[~has])
  Ag=sparse.diags(1/gn[has])@Ag[has]
  A=sparse.vstack([Av,Ag,sparse.eye(len(free)*3)],format="csc")
  # Construction margin protects exact Float32 emission; physical capacities unchanged.
  margin=.01*(vhi-vlo)
  lower=np.r_[(1.25*vmin[affected]-v[affected])/vn,(vlo[has]+margin[has]-gv[has])/gn[has],np.full(len(free)*3,-.00025)]
  upper=np.r_[np.full(len(affected),np.inf),(vhi[has]-margin[has]-gv[has])/gn[has],np.full(len(free)*3,.00025)]
  fit.CONTACT_MATRIX=None;fit.CONTACT_TARGET=None
  H=sparse.eye(len(free)*3,format="csc")
  delta,diag=fit.solve_qp(H,(target[free]-x[free]).ravel(),A,lower,upper)
  dx=np.zeros_like(x);dx[free]=delta.reshape(-1,3)
  retained=None
  for bt in range(12):
   alpha=2.**(-bt);trial=(x+dx*alpha).astype(np.float32).astype(float);sv=stats(trial);m=sv[3]
   if m<merit or m==0:retained=(trial,alpha,m);break
  item.update(free_nodes=len(free),affected_cells=len(affected),qp=diag,accepted=retained is not None)
  if retained is None:hist.append(item);break
  x,alpha,after=retained;item.update(alpha=alpha,violation_after=after,max_restoration_m=float(np.linalg.norm(x-target,axis=1).max()));hist.append(item);print(json.dumps(item),flush=True)
 moved=np.flatnonzero(np.any(x!=original,axis=1))
 with (meta/"coarse-node-to-fine-surface.csv").open() as f:rows={int(r["volume_node_row"]):r for r in csv.DictReader(f)}
 poses=json.loads((parent/"registered-skin-poses-100.json").read_text())
 for n in moved:
  pose=poses["all_bodies"][int(rows[int(n)]["frame_owner_body"])]["current"]
  rot=np.asarray(pose["local_to_world_row_major_f32"]).reshape(3,3);origin=np.asarray(pose["position_world_m"])
  a["points_donor_local_m_f64"][n]=np.linalg.solve(rot,x[n]-origin)
  a["points_current_world_m_f32"][n]=x[n];a["target_current_world_m"][n]=x[n]
 a["points_reference_world_m_f64"]=x;dest=args.out/"candidate.npz";np.savez_compressed(dest,**a)
 v,gr,gv,merit=stats(x)
 rep={"script_sha256":cap.sha(__file__),"source_sha256":cap.sha(base),"linear_proposal_sha256":cap.sha(src),"candidate_sha256":cap.sha(dest),"proposal_alpha":args.alpha,"iterations":hist,"nonpositive_cells":int(np.count_nonzero(v<=0)),"failed_capacity_groups":int(np.count_nonzero((gv<vlo)|(gv>vhi))),"scaled_exact_violation":merit,"pinned_nodes_unchanged":bool(np.array_equal(x[pins],original[pins])),"physical_admission":False,"elapsed_s":time.monotonic()-start}
 if merit==0:
  manifest=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json"
  _,r=cw.extract(dest,manifest);fit.require_localized(r);rep["positive_material_bank_pairs"]=r["positive_pairs"];rep["overlap_m3_summed_per_bank_not_union"]=sum(z["intersection_m3"] for z in r["banks"])
 (args.out/"receipt.json").write_text(json.dumps(rep,indent=2)+"\n");print(json.dumps(rep),flush=True)
if __name__=="__main__":main()
