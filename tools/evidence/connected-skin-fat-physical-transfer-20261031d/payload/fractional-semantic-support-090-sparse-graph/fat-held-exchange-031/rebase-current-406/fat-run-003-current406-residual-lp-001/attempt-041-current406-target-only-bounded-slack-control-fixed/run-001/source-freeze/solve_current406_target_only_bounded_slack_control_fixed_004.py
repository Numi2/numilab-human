#!/usr/bin/env python3
import csv,hashlib,json,time,resource,os
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack, csr_matrix, eye
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
H=ROOT/"fine-physical-volume-706";S=ROOT/"fractional-semantic-support-090-sparse-graph"
R=S/"fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001"
G=R/"attempt-005/lp-input-current406.npz"
GPREP=R/"attempt-005/graph-preparation-report.json"; GREP=R/"attempt-005/lp-report.json"
MERGE=R/"attempt-022-source-exhaustion-reserved-batch/run-001/merged-held-centroid-noncentroid-source-exhaustion.npz"
V022=R/"attempt-022-source-exhaustion-reserved-batch/independent-replay-013.json"
P022=R/"attempt-022-source-exhaustion-reserved-batch/run-001/proposal.npz"
TARGET=H/"current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
SUP=H/"connected-region-mass-support-143-current406-root/fat-connected-region-support.npz"
SUPIV=SUP.parent/"independent-verification.json"
PARENT=S/"fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv"
SRC=ROOT/"native-physical-parent-mass-330/native-source-identity-control-003"
OUT=R/"attempt-041-current406-target-only-bounded-slack-control-fixed/run-001";OUT.mkdir(parents=True,exist_ok=True)
PREFLIGHT_ONLY=os.environ.get("CURRENT406_TARGET_ONLY_PREFLIGHT")=="1"
PHASE1BASE=R/"attempt-026-held-parent-exchange-top100/run-007/phase1-diagnostic.json"
RADIUS=.019353; LOW=850.; HIGH=950.
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(1<<20),b""):h.update(x)
 return h.hexdigest()
def csvrows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
def peak():
 v=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 import sys
 return v if sys.platform=="darwin" else v*1024
def assemble_lp(selected,parent_mass,var_parent,var_child,bary,dist,fixed,child_volume,forced_zero,time_limit=60.,remainder_corner_cap=None,receiver_minimum=None,solve=True,target_slack_children=None,target_slack_caps_mass=None):
 selected=np.asarray(selected,dtype=np.int32);var_parent=np.asarray(var_parent,dtype=np.int32);var_child=np.asarray(var_child,dtype=np.int32);bary=np.asarray(bary,dtype=np.float64);dist=np.asarray(dist,dtype=np.float64)
 if bary.shape!=(len(var_parent),4) or len(var_child)!=len(var_parent) or len(dist)!=len(var_parent):raise RuntimeError("variable-array shape mismatch")
 pmap={int(p):i for i,p in enumerate(selected)}
 try:lp=np.asarray([pmap[int(p)] for p in var_parent],np.int32)
 except KeyError as e:raise RuntimeError(f"variable parent not selected: {e}")
 child_ids=np.unique(var_child)
 target_global=np.empty(0,np.int32);target_slack_caps=np.empty(0,np.float64);target_local=np.empty(0,np.int32);nslack=0
 if target_slack_children is not None:
  target_global=np.unique(np.asarray(target_slack_children,dtype=np.int32))
  if target_slack_caps_mass is None:raise RuntimeError("target slack requires an explicit mass cap")
  target_slack_caps=np.asarray(target_slack_caps_mass,dtype=np.float64)
  if target_slack_caps.shape!=(len(target_global),) or not np.isfinite(target_slack_caps).all() or np.any(target_slack_caps<0):raise RuntimeError("target slack mass caps invalid")
  child_ids=np.union1d(child_ids,target_global).astype(np.int32)
  target_local=np.searchsorted(child_ids,target_global).astype(np.int32)
  if not np.array_equal(child_ids[target_local],target_global):raise RuntimeError("target child local/global map mismatch")
  nslack=len(target_global)
 lc=np.searchsorted(child_ids,var_child).astype(np.int32)
 pm=parent_mass[var_parent];nsite=len(var_parent);nsel=len(selected)
 if np.any(pm<=0):raise RuntimeError("nonpositive source parent mass")
 if remainder_corner_cap is None:
  nrem=0;remcap=np.empty(0,np.float64)
 else:
  remcap=np.asarray(remainder_corner_cap,dtype=np.float64)
  if remcap.shape!=(nsel,4) or not np.isfinite(remcap).all() or np.any(remcap<0):raise RuntimeError("unallocated remainder corner caps invalid")
  nrem=4*nsel
 # Variables are parent-mass fractions. Four source rows retain the exact M/4 target;
 # explicit remainder fractions satisfy those rows without increasing the 022 unallocated corners.
 eq_rows=np.concatenate([lp+k*nsel for k in range(4)])
 eq_cols=np.tile(np.arange(nsite,dtype=np.int32),4)
 eq_vals=np.concatenate([bary[:,k] for k in range(4)])
 Aeq0=coo_matrix((eq_vals,(eq_rows,eq_cols)),shape=(4*nsel,nsite)).tocsr()
 beq=np.full(4*nsel,.25,dtype=np.float64)
 if nrem:
  rp=np.repeat(np.arange(nsel,dtype=np.int32),4);rk=np.tile(np.arange(4,dtype=np.int32),nsel)
  rr=rp+rk*nsel;rc=np.arange(nrem,dtype=np.int32)
  Ar=coo_matrix((np.ones(nrem),(rr,rc)),shape=(4*nsel,nrem)).tocsr()
  Aeq=hstack([Aeq0,Ar],format="csr")
  ub=np.concatenate([np.minimum(np.minimum(np.full(nsite,np.inf),1.),np.minimum.reduce([np.divide(.25,bary[:,k],out=np.full(nsite,np.inf),where=bary[:,k]>0) for k in range(4)])),remcap.reshape(-1)])
 else:
  Aeq=Aeq0
  ub=np.minimum(1.,np.minimum.reduce([np.divide(.25,bary[:,k],out=np.full(nsite,np.inf),where=bary[:,k]>0) for k in range(4)]))
 max_parent_mass_by_child=np.zeros(len(child_ids),dtype=np.float64);np.maximum.at(max_parent_mass_by_child,lc,pm)
 child_scale=np.maximum.reduce([HIGH*child_volume[child_ids],max_parent_mass_by_child/1e10,np.full(len(child_ids),1e-300)])
 if nslack:
  cap_fraction=target_slack_caps/child_scale[target_local]
  if np.any(~np.isfinite(cap_fraction)):raise RuntimeError("target slack normalized caps nonfinite")
  ub=np.concatenate([ub,cap_fraction])
  Aeq=hstack([Aeq,csr_matrix((4*nsel,nslack))],format="csr")
 coeff=pm/child_scale[lc]
 high=HIGH*child_volume[child_ids]
 low=LOW*child_volume[child_ids] if receiver_minimum is None else np.asarray(receiver_minimum,dtype=np.float64)[child_ids]
 local_fixed=fixed[child_ids]
 local_over=local_fixed>high
 live=~local_over[lc]&~np.asarray(forced_zero,dtype=bool)
 live_local=np.flatnonzero(~local_over)
 rmap=np.full(len(child_ids),-1,np.int32);rmap[live_local]=np.arange(len(live_local),dtype=np.int32)
 ar=np.repeat(2*rmap[lc[live]],2)+np.tile(np.array([0,1],np.int32),int(live.sum()))
 ac=np.repeat(np.flatnonzero(live),2)
 av=np.empty(2*int(live.sum()),dtype=np.float64);av[0::2]=coeff[live];av[1::2]=-coeff[live]
 Aub0=coo_matrix((av,(ar,ac)),shape=(2*len(live_local),nsite)).tocsr()
 Aub=hstack([Aub0,csr_matrix((Aub0.shape[0],nrem+nslack))],format="csr") if (nrem+nslack) else Aub0
 bub=np.empty(2*len(live_local),dtype=np.float64)
 bub[0::2]=(high[live_local]-local_fixed[live_local])/child_scale[live_local]
 bub[1::2]=(local_fixed[live_local]-low[live_local])/child_scale[live_local]
 if nslack:
  target_site=np.flatnonzero(np.isin(lc,target_local))
  target_rows=np.searchsorted(target_local,lc[target_site])
  ats=coo_matrix((-coeff[target_site],(target_rows,target_site)),shape=(nslack,nsite)).tocsr()
  Atarget=hstack([ats,csr_matrix((nslack,nrem)),-eye(nslack,format="csr")],format="csr")
  target_bub=(local_fixed[target_local]-LOW*child_volume[child_ids[target_local]])/child_scale[target_local]
  Aub=vstack([Aub,Atarget],format="csr");bub=np.concatenate([bub,target_bub])
 ub[:nsite][np.asarray(forced_zero,dtype=bool)|local_over[lc]]=0.
 sitecost=pm*(dist/RADIUS+1e-10)
 if nslack:
  cost=np.zeros(len(ub),dtype=np.float64)
  cost[nsite+nrem:]=child_scale[target_local]
 else:
  cost=np.concatenate([sitecost,2.*np.repeat(parent_mass[selected],4)]) if nrem else sitecost
 checks={"cost_finite":bool(np.isfinite(cost).all()),"Aeq_finite":bool(np.isfinite(Aeq.data).all()),"beq_finite":bool(np.isfinite(beq).all()),"Aub_finite":bool(np.isfinite(Aub.data).all()),"bub_finite":bool(np.isfinite(bub).all()),"bounds_finite":bool(np.isfinite(ub).all()),"bounds_nonnegative":bool(np.all(ub>=0)),"Aeq_indices_in_range":bool(Aeq.indices.size==0 or (Aeq.indices.min()>=0 and Aeq.indices.max()<len(ub))),"Aeq_indptr_ok":bool(Aeq.indptr[-1]==Aeq.nnz),"Aub_indices_in_range":bool(Aub.indices.size==0 or (Aub.indices.min()>=0 and Aub.indices.max()<len(ub))),"Aub_indptr_ok":bool(Aub.indptr[-1]==Aub.nnz),"Aeq_shape":list(Aeq.shape),"Aub_shape":list(Aub.shape),"nsite":int(nsite),"n_remainder":int(nrem),"nvar":int(len(ub)),"Aeq_nnz":int(Aeq.nnz),"Aub_nnz":int(Aub.nnz),"receiver_row_coefficient_cap":1e10,"cost_minmax":[float(cost.min(initial=0)),float(cost.max(initial=0))],"Aeq_minmax":[float(Aeq.data.min(initial=0)),float(Aeq.data.max(initial=0))],"beq_minmax":[float(beq.min(initial=0)),float(beq.max(initial=0))],"Aub_minmax":[float(Aub.data.min(initial=0)),float(Aub.data.max(initial=0))],"bub_minmax":[float(bub.min(initial=0)),float(bub.max(initial=0))],"upper_minmax":[float(ub.min(initial=0)),float(ub.max(initial=0))],"zero_Aeq_rows":int(np.count_nonzero(np.diff(Aeq.indptr)==0)),"zero_Aub_rows":int(np.count_nonzero(np.diff(Aub.indptr)==0))}
 if not all(checks[k] for k in ["cost_finite","Aeq_finite","beq_finite","Aub_finite","bub_finite","bounds_finite","bounds_nonnegative","Aeq_indices_in_range","Aeq_indptr_ok","Aub_indices_in_range","Aub_indptr_ok"]):raise RuntimeError("sparse LP model preflight invalid: "+str(checks))
 (OUT/"matrix-preflight.json").write_text(json.dumps(checks,indent=2,sort_keys=True)+"\n")
 res=linprog(cost,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=list(zip(np.zeros(len(ub)),ub)),method="highs",options={"time_limit":float(time_limit),"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10,"presolve":True}) if solve else None
 return res,{"selected_parent_rows":selected,"variable_parent_rows":var_parent,"variable_child_rows":var_child,"barycentric":bary,"distance_m":dist,"child_ids":child_ids,"child_local_rows":lc,"parent_local_rows":lp,"receiver_row_scale":child_scale,"fixed_child_mass_kg":fixed[child_ids],"receiver_lower_mass_kg":low,"Aeq":Aeq,"beq":beq,"Aub":Aub,"bub":bub,"upper_fraction":ub,"forced_zero":np.asarray(forced_zero,dtype=bool),"preexisting_over_cap_local_rows":child_ids[local_over],"cost":cost,"n_site_vars":int(nsite),"n_remainder_vars":int(nrem),"n_target_slack_vars":int(nslack),"target_slack_global_rows":target_global,"target_slack_local_rows":target_local,"target_slack_caps_mass_kg":target_slack_caps,"target_slack_caps_fraction":(target_slack_caps/child_scale[target_local] if nslack else np.empty(0,np.float64)),"n_lp_variables":int(len(ub))}
# End-to-end local-index control on actual noncontiguous parent IDs and child IDs.
def control():
 pm=np.zeros(6007);pm[4274]=2.;pm[6006]=3.
 sel=np.array([4274,6006],np.int32)
 vp=np.repeat(sel,4);vc=np.array([9,9,71,71,71,71,404,404],np.int32)
 b=np.tile(np.eye(4),(2,1));d=np.zeros(8);fixed=np.zeros(405);fixed[[9,71,404]]=[1.55,.05,1.05];vol=np.full(405,.003)
 r,a=assemble_lp(sel,pm,vp,vc,b,d,fixed,vol,np.zeros(8,bool),5.)
 if not r.success:raise RuntimeError("noncontiguous parent/receiver control LP failed")
 x=r.x*pm[vp];corn=np.column_stack([np.bincount(vp,weights=x*b[:,k],minlength=len(pm)) for k in range(4)])
 if np.max(np.abs(corn[sel]-pm[sel,None]*.25))>1e-12:raise RuntimeError("noncontiguous M/4 control failed")
 if not np.array_equal(a["child_ids"],np.array([9,71,404])):raise RuntimeError("noncontiguous child map control failed")
 return {"status":"PASS_ACTUAL_ASSEMBLER_NONCONTIGUOUS_PARENT_CHILD_M4_CONTROL","selected_parent_rows":sel.tolist(),"child_rows":a["child_ids"].tolist(),"variable_count":len(vp),"max_M4_error_kg":float(np.max(np.abs(corn[sel]-pm[sel,None]*.25)))}
def bounded_slack_control():
 pm=np.zeros(6007);pm[4274]=2.
 sel=np.array([4274],np.int32);vp=np.repeat(sel,4)
 vc=np.full(4,9,np.int32);b=np.eye(4);d=np.zeros(4)
 fixed=np.zeros(10);vol=np.zeros(10);vol[9]=.002
 rem=np.full((1,4),.25);forced=np.zeros(4,bool);cap=np.array([850.*vol[9]])
 base_slack=cap.copy()
 r,a=assemble_lp(sel,pm,vp,vc,b,d,fixed,vol,forced,remainder_corner_cap=rem,receiver_minimum=np.minimum(fixed,850.*vol),solve=True,target_slack_children=np.array([9]),target_slack_caps_mass=cap)
 if not r.success:raise RuntimeError("bounded target-slack assembler control did not solve")
 baseline=np.r_[np.zeros(4),rem.reshape(-1),base_slack/a["receiver_row_scale"][a["target_slack_local_rows"]]]
 eq=np.asarray(a["Aeq"]@baseline-a["beq"]).reshape(-1)
 ubres=np.asarray(a["Aub"]@baseline-a["bub"]).reshape(-1)
 if np.max(np.abs(eq),initial=0)>1e-14 or np.max(ubres,initial=0)>1e-14:raise RuntimeError("022-style baseline vector fails bounded target-slack matrix control")
 x=np.asarray(r.x);flow=x[:4]*pm[vp];slack=x[-1]*a["receiver_row_scale"][a["target_slack_local_rows"]][0]
 m4=np.asarray(a["Aeq"]@x-a["beq"]).reshape(-1)
 recv=float(flow.sum())
 if np.max(np.abs(m4),initial=0)>1e-9 or recv+slack<cap[0]-1e-10 or recv>950.*vol[9]+1e-10:raise RuntimeError("bounded target-slack optimized control replay failed")
 if slack>1e-8:raise RuntimeError("bounded target-slack control failed to remove feasible target deficit")
 return {"status":"PASS_ACTUAL_ASSEMBLER_BOUNDED_TARGET_SLACK_BASELINE_AND_OPTIMIZATION_CONTROL","parent_role_rows":sel.tolist(),"child_rows":[9],"baseline_target_slack_kg":float(base_slack[0]),"optimized_target_slack_kg":float(slack),"optimized_receiver_mass_kg":recv,"receiver_upper_kg":float(950.*vol[9]),"baseline_matrix_max_equality_residual":float(np.max(np.abs(eq),initial=0)),"baseline_matrix_max_inequality_residual":float(np.max(ubres,initial=0)),"optimized_source_corner_max_residual_fraction":float(np.max(np.abs(m4),initial=0))}
control_receipt={"noncontiguous_parent_child":control(),"bounded_target_only_slack":bounded_slack_control()}
if os.environ.get("CURRENT406_TARGET_ONLY_CONTROL_ONLY")=="1":
 print(json.dumps({"stage":"target-only-assembler-control","controls":control_receipt},sort_keys=True),flush=True)
 raise SystemExit(0)
t0=time.monotonic()
v22=json.loads(V022.read_text())
if v22.get("status")!="PASS_INDEPENDENT_PARTIAL_CURRENT406_RESERVED_SOURCE_EXHAUSTION_BATCH_REPLAY":raise RuntimeError("022 independent receipt is not passing")
if v22.get("physical_admission") is not False or v22.get("native_steps")!=0:raise RuntimeError("022 receipt scope incorrect")
with np.load(G,allow_pickle=False) as z:
 P=z["parent_role_rows"].astype(np.int32);C=z["child_rows"].astype(np.int32);B=z["barycentric"].astype(np.float64);D=z["distance_m"].astype(np.float64);vol=z["child_volume_hybrid_m3"].astype(np.float64);PG=z["parent_global_source330_rows"].astype(np.int64)
with np.load(MERGE,allow_pickle=False) as z:
 fp=z["flow_parent_role_rows"].astype(np.int32);fg=z["flow_global_source330_rows"].astype(np.int64);fc=z["flow_current_child_rows"].astype(np.int32);fm=z["flow_mass_kg"].astype(np.float64);fb=z["flow_source_barycentric"].astype(np.float64);fpt=z["flow_source_point_world_m"].astype(np.float64);fd=z["flow_current_centroid_distance_m"].astype(np.float64);fo=z["flow_origin_code"].astype(np.uint8);fbase=z["base_ledger_flow_rows"].astype(np.int64);pmass=z["parent_source_mass_kg"].astype(np.float64);qold=z["parent_corner_residual_after_kg"].astype(np.float64);childmass=z["child_current_mass_kg"].astype(np.float64)
parents=csvrows(PARENT)
prole=np.asarray([int(r["parent_row"]) for r in parents],np.int32);pglobal=np.asarray([int(r["global_source330_row"]) for r in parents],np.int64);pm_csv=np.asarray([float(r["mass_kg"]) for r in parents]);pregion=np.asarray([int(r["region"]) for r in parents],np.int32);pmat=np.asarray([int(r["material"]) for r in parents],np.int32)
if not np.array_equal(prole,np.arange(len(pmass),dtype=np.int32)) or not np.array_equal(pmass,pm_csv):raise RuntimeError("022/source-parent role-mass map mismatch")
if not np.array_equal(pglobal,np.arange(14316,14316+len(pmass),dtype=np.int64)):raise RuntimeError("Fat parent global row map mismatch")
if not np.array_equal(fg,pglobal[fp]):raise RuntimeError("022 flow global-source mapping mismatch")
with np.load(TARGET,allow_pickle=False) as z:
 points=z["points_world_m"].astype(np.float32,copy=False);tets=z["tetrahedra"].astype(np.int64,copy=False);mat=z["material_indices"].astype(np.int32,copy=False)
cent=np.mean(points[tets].astype(np.float64),axis=1)
if len(tets)!=len(childmass) or len(tets)!=len(vol):raise RuntimeError("current 406 child-domain mismatch")
with np.load(SUP,allow_pickle=False) as z:
 off=z["cell_admissible_region_offsets"].astype(np.int64);regs=z["cell_admissible_region_ids"].astype(np.int32)
if len(off)!=len(tets)+1:raise RuntimeError("143 support child CSR mismatch")
# Authenticate 143 one-hop support proof and current 406 geometry pins.
supportiv=json.loads(SUPIV.read_text())
if supportiv.get("status")!="PASS_INDEPENDENT_AUTHENTICATED_CONNECTED_SUPPORT" or supportiv.get("physical_admission") is not False:raise RuntimeError("143 support independent proof missing")
fatres=[x for x in supportiv.get("results",[]) if x.get("role")=="Fat"]
if len(fatres)!=1 or fatres[0].get("cells")!=len(tets):raise RuntimeError("143 Fat child row domain mismatch")
# Source330 exact rest-coordinate map.
nodes=csvrows(SRC/"compiled-parent-cell-mass.csv.nodes.csv");xyz=np.asarray([[float(r["rest_"+a+"_m"]) for a in "xyz"] for r in nodes],np.float64)
trows=csvrows(SRC/"compiled-parent-cell-mass.csv.tets.csv");gtet=np.asarray([int(r["parent_tet_row"]) for r in trows],np.int64);tnodes=np.asarray([[int(r[f"parent_node{k}"]) for k in range(4)] for r in trows],np.int64)
gmax=int(max(gtet.max(initial=0),pglobal.max(initial=0)));tetnodes=np.full((gmax+1,4),-1,np.int64);tetnodes[gtet]=tnodes
if np.any(tetnodes[pglobal]<0):raise RuntimeError("missing role parent in source330 tet map")
parentverts=xyz[tetnodes[pglobal]]
if np.any(pmat!=16) or np.any(mat!=16):raise RuntimeError("Fat material namespace mismatch")
# Prioritize parents by their independently bounded all-held-release potential into underfloor receivers.
priorcapacity=json.loads((R/"attempt-024-held-release-capacity-analysis/report.json").read_text())
top=priorcapacity["metrics"]["top_held_parents_with_deficit_capacity"][:100]
selected0=np.sort(np.asarray([x["parent_role_row"] for x in top],np.int32))
if len(selected0)!=100 or len(np.unique(selected0))!=100:raise RuntimeError("top-100 parent selection malformed")
phase1base=json.loads(PHASE1BASE.read_text())
if phase1base.get("status")!="PASS_PHASE1_RECEIVER_SLACK_DIAGNOSTIC":raise RuntimeError("pinned current406 phase-I donor target receipt missing")
target_children=np.sort(np.asarray([int(r["child_row"]) for r in phase1base["largest_receiver_slacks"][:40]],np.int32))
if len(target_children)!=40 or len(np.unique(target_children))!=40:raise RuntimeError("phase-I target receiver domain malformed")
cut_donor_mask=np.isin(C,target_children)&~np.isin(P,selected0)
cut_donors=np.unique(P[cut_donor_mask]).astype(np.int32)
if len(cut_donors)!=182:raise RuntimeError(f"current406 cut donor roster changed: {len(cut_donors)}")
selected=np.sort(np.concatenate([selected0,cut_donors]))
if len(selected)!=282 or len(np.unique(selected))!=282:raise RuntimeError("expanded selected parent domain malformed")
selectedmask=np.isin(fp,selected)
held_parent_rows=int(np.count_nonzero(selectedmask));held_mass=float(fm[selectedmask].sum())
# Keep all graph rows for selected donors; generation cap is not a hard mass limit once source corner and receiver bounds are enforced.
gmask=np.isin(P,selected)&np.isin(C,target_children);gp=P[gmask];gc=C[gmask];gb=B[gmask];gd=D[gmask]
if not len(gp):raise RuntimeError("selected parents have no legal graph sites")
if not np.array_equal(PG[gmask],pglobal[gp]):raise RuntimeError("graph global source row map mismatch")
# Recompute all graph site coordinates and radius using source330 rest coordinates and widened current Float32 child centroids.
gxyz=np.einsum("ij,ijk->ik",gb,parentverts[gp])
grecomp=np.linalg.norm(gxyz-cent[gc],axis=1)
max_graph_radius_error=float(np.max(np.abs(grecomp-gd),initial=0.))
if max_graph_radius_error>5e-12:raise RuntimeError(f"cached graph distance mismatch {max_graph_radius_error}")
if np.any(grecomp>RADIUS):raise RuntimeError("cached graph contains out-of-radius site")
if np.any(~np.isfinite(gb)) or np.any(gb<0) or np.any(np.abs(gb.sum(axis=1)-1)>2e-14):raise RuntimeError("cached graph barycentric invalid")
# Authenticate every selected graph edge's region permission using 143 cell-admissible CSR.
regionkeys=np.repeat(np.arange(len(tets),dtype=np.int64),np.diff(off))*65536+np.concatenate([regs[off[i]:off[i+1]] for i in range(len(tets))]).astype(np.int64)
regionkeys=np.unique(regionkeys)
edgekey=gc.astype(np.int64)*65536+pregion[gp].astype(np.int64)
pos=np.searchsorted(regionkeys,edgekey)
if np.any(pos>=len(regionkeys)) or not np.all(regionkeys[np.minimum(pos,len(regionkeys)-1)]==edgekey):raise RuntimeError("selected graph site fails authenticated one-hop region permission")
# Revalidate held selected sites; they are additional legal points where the candidate-site preparation has no exact column.
hp=fp[selectedmask];hc=fc[selectedmask];hb=fb[selectedmask];hd=fd[selectedmask];hpt=fpt[selectedmask];hmass=fm[selectedmask]
hxyz=np.einsum("ij,ijk->ik",hb,parentverts[hp])
hrec=np.linalg.norm(hxyz-cent[hc],axis=1)
if np.max(np.abs(hxyz-hpt),initial=0.)>2e-12:raise RuntimeError("held source point differs from exact source330 barycentric reconstruction")
if np.max(np.abs(hrec-hd),initial=0.)>5e-12 or np.any(hrec>RADIUS):raise RuntimeError("held source radius replay failed")
hkey=hc.astype(np.int64)*65536+pregion[hp].astype(np.int64);hpos=np.searchsorted(regionkeys,hkey)
if np.any(hpos>=len(regionkeys)) or not np.all(regionkeys[np.minimum(hpos,len(regionkeys)-1)]==hkey):raise RuntimeError("held flow fails current 143 one-hop region")
# Merge candidate and held point-child columns, deduplicating only exact (parent,child,bary bits) duplicates.
vp=np.concatenate([gp,hp]);vc=np.concatenate([gc,hc]);vb=np.concatenate([gb,hb]);vd=np.concatenate([grecomp,hrec]);origin=np.concatenate([np.zeros(len(gp),np.uint8),np.ones(len(hp),np.uint8)])
bits=vb.view(np.uint64).reshape(-1,4);keys=np.column_stack([vp.astype(np.uint64),vc.astype(np.uint64),bits])
_,first=np.unique(keys,axis=0,return_index=True);first=np.sort(first)
vp=vp[first];vc=vc[first];vb=vb[first];vd=vd[first];origin=origin[first]
if len(vp)!=len(first):raise RuntimeError("site dedup row-domain error")
# Reopen selected donors by removing their entire held ledger. Other parent flows remain fixed.
selected_held_by_child=np.bincount(fc[selectedmask],weights=fm[selectedmask],minlength=len(tets))
fixed=childmass-selected_held_by_child
if np.min(fixed,initial=0)<-1e-12:raise RuntimeError("selected held child subtraction became negative")
# A child over its 950 upper bound from unchanged nonselected flows cannot receive selected variables; retain and report it unchanged.
high=HIGH*vol
overfixed=fixed>high
bad_over=overfixed & (fixed-high>5e-10*vol)
if np.any(bad_over):raise RuntimeError("selected closure leaves nonselected parent mass materially above local 950 bound")
forced_zero=overfixed[vc]
# Catch row/value alignment before large solve.
if len(vp)!=len(vc) or len(vp)!=len(vb) or len(vp)!=len(vd) or len(vp)!=len(origin):raise RuntimeError("final LP column arrays misaligned")
if not np.array_equal(np.sort(np.unique(vp)),selected):raise RuntimeError("selected parent local index domain mismatch")
# Pre-solve input/capacity receipt.
pre={"schema":"numi.human.current406-held-parent-exchange-preflight.v1","status":"PASS_BOUNDED_LOCAL_PARENT_EXCHANGE_PREFLIGHT","physical_admission":False,"native_steps":0,"inputs":{str(x):sha(x) for x in [G,GPREP,GREP,MERGE,V022,P022,PHASE1BASE,TARGET,SUP,SUPIV,PARENT,SRC/"compiled-parent-cell-mass.csv.nodes.csv",SRC/"compiled-parent-cell-mass.csv.tets.csv"]},"control":control_receipt,"selected_parent_rows":selected.tolist(),"counts":{"candidate_graph_columns":int(len(gp)),"prior_selected_held_rows":held_parent_rows,"prior_selected_held_children":int(len(np.unique(hc))),"final_unique_parent_child_site_columns":int(len(vp)),"candidate_children":int(len(np.unique(vc))),"fixed_over_upper_receivers":int(np.count_nonzero(overfixed))},"mass_kg":{"selected_parent_total":float(pmass[selected].sum()),"selected_held_mass_released":held_mass,"selected_source_corner_residual":float(qold[selected].sum()),"underfloor_before_global":float(np.maximum(LOW*vol-childmass,0).sum()),"max_fixed_over_upper":float(np.max(fixed-high,initial=0))},"metrics":{"graph_recomputed_max_distance_error_m":max_graph_radius_error,"max_held_barycentric_reconstruction_error_m":float(np.max(np.abs(hxyz-hpt),initial=0)),"selected_parents":int(len(selected)),"new_cut_donor_parents":int(len(cut_donors)),"phase1_target_child_rows":target_children.tolist(),"reopened_parent_scope":"original top-100 optimistic parents plus 182 distinct held donors on the top-40 pinned phase-I receiver witness; all source mass per selected parent, exact four M/4 rows, all prior held flows released once"},"allocation":{"generation_column_mass_cap_used":False,"hard_source_M4":True,"hard_radius_m":RADIUS,"material_match":True,"143_one_hop_region_support":True,"density_kg_m3":[LOW,HIGH],"unselected_held_flows_fixed":True},"physical_admission":False,"complete_allocation":False}
(OUT/"preflight.json").write_text(json.dumps(pre,indent=2,sort_keys=True)+"\n")
print(json.dumps({"stage":"preflight","status":pre["status"],"counts":pre["counts"],"mass_kg":pre["mass_kg"],"sha256":sha(OUT/"preflight.json")},sort_keys=True),flush=True)
# First prove the exact 022 local baseline satisfies the same sparse matrix with target floors disabled.
with np.load(G,allow_pickle=False) as z:
 volume=z["child_volume_hybrid_m3"].astype(np.float64)
lowmass=LOW*volume
old_minimum=np.minimum(childmass,lowmass)
raw_remcap=qold[selected]/pmass[selected,None]
corner_tolerance=np.broadcast_to(5e-13*np.maximum(pmass[selected,None]*.25,1e-300),raw_remcap.shape)
negative_rem=raw_remcap<0
if np.any((-raw_remcap[negative_rem]*np.broadcast_to(pmass[selected,None],raw_remcap.shape)[negative_rem])>corner_tolerance[negative_rem]):raise RuntimeError("022 source remainder negative beyond existing M/4 tolerance")
# Negative residuals are signed summation roundoff in the immutable 022 ledger; do not alter that ledger. They provide zero nonnegative remainder capacity.
remcap=np.maximum(raw_remcap,0.)
baseline_site=np.zeros(len(vp),dtype=np.float64)
site_map={(int(vp[j]),int(vc[j]),tuple(vb[j].view(np.uint64))):j for j in range(len(vp))}
for fi in np.flatnonzero(selectedmask):
 key=(int(fp[fi]),int(fc[fi]),tuple(fb[fi].view(np.uint64)))
 j=site_map.get(key)
 if j is None:raise RuntimeError("022 selected held row has no exact LP site column")
 baseline_site[j]+=fm[fi]/pmass[fp[fi]]
target_deficit=np.maximum(lowmass[target_children]-childmass[target_children],0.)
_,abase=assemble_lp(selected,pmass,vp,vc,vb,vd,fixed,volume,forced_zero,remainder_corner_cap=remcap,receiver_minimum=old_minimum,solve=False,target_slack_children=target_children,target_slack_caps_mass=target_deficit)
base_receivers=fixed+np.bincount(vc,weights=baseline_site*pmass[vp],minlength=len(tets))
baseline_target_slack=np.maximum(lowmass[target_children]-base_receivers[target_children],0.)/abase["receiver_row_scale"][abase["target_slack_local_rows"]]
baseline_vector=np.concatenate([baseline_site,remcap.reshape(-1),baseline_target_slack])
eqres=np.asarray(abase["Aeq"]@baseline_vector-abase["beq"]).reshape(-1)
ubres=np.asarray(abase["Aub"]@baseline_vector-abase["bub"]).reshape(-1)
bound_low=float(max(0.,-baseline_vector.min(initial=0.)))
bound_high=float(max(0.,np.max(baseline_vector-abase["upper_fraction"],initial=0.)))
receiver_tol=64*np.finfo(float).eps*np.maximum(high,1e-300)
eqkg=eqres.reshape(4,len(selected)).T*pmass[selected,None]
eqtol=5e-13*np.maximum(pmass[selected,None]*.25,1e-300)
eqratio=float(np.max(np.abs(eqkg)/eqtol,initial=0.))
baseline_control={"schema":"numi.human.current406-022-local-exchange-baseline-matrix-control.v1","status":"PASS_CURRENT406_022_BASELINE_RESTORES_LOCAL_EXCHANGE_MATRIX" if (eqratio<=1. and bound_low==0. and bound_high<=1e-14 and np.all(base_receivers[abase["child_ids"]]>=old_minimum[abase["child_ids"]]-receiver_tol[abase["child_ids"]] ) and np.all(base_receivers[abase["child_ids"]]<=high[abase["child_ids"]]+receiver_tol[abase["child_ids"]])) else "FAIL_CURRENT406_022_BASELINE_LOCAL_MATRIX_CONTROL","complete_allocation":False,"physical_admission":False,"native_steps":0,"inputs":pre["inputs"],"parent_rows":selected.tolist(),"target_children_with_bounded_slack":target_children.tolist(),"counts":{"site_variables":int(len(vp)),"remainder_corner_variables":int(4*len(selected)),"local_receiver_rows":int(len(abase["child_ids"])),"selected_held_flow_rows":int(np.count_nonzero(selectedmask)),"negative_022_corner_residuals":int(np.count_nonzero(negative_rem))},"mass_kg":{"selected_held_restored":float(fm[selectedmask].sum()),"selected_signed_remainder_restored":float(qold[selected].sum()),"selected_nonnegative_remainder_capacity":float((remcap*pmass[selected,None]).sum()),"selected_parent_total":float(pmass[selected].sum()),"reconstructed_baseline_flow_delta":float(base_receivers.sum()-childmass.sum()),"max_negative_022_corner_roundoff_kg":float(np.max((-qold[selected][negative_rem])*1.,initial=0.))},"residuals":{"max_parent_corner_equation_kg":float(np.max(np.abs(eqkg),initial=0.)),"max_parent_corner_error_over_existing_022_tolerance":eqratio,"max_receiver_inequality_normalized":float(np.max(ubres,initial=0.)),"max_lower_bound_violation":bound_low,"max_upper_bound_violation":bound_high,"max_receiver_mass_error_kg":float(np.max(np.abs(base_receivers[abase["child_ids"]]-childmass[abase["child_ids"]]),initial=0.)),"max_remainder_cap_excess_fraction":float(np.max(baseline_vector[len(vp):len(vp)+4*len(selected)]-remcap.reshape(-1),initial=0.))},"matrix":{"Aeq_shape":list(abase["Aeq"].shape),"Aub_shape":list(abase["Aub"].shape),"variables":int(abase["n_lp_variables"]),"target_floor_rows":int(abase["n_target_slack_vars"])},"roundoff_policy":"The frozen 022 signed q array is preserved. Negative entries (if any) within its existing M/4 replay tolerance give zero available nonnegative remainder capacity; old held flows are not changed."}
baseline_control["producer_source_path"]=str(Path(__file__))
baseline_control["producer_source_sha256"]=sha(Path(__file__))
(OUT/"baseline-control.json").write_text(json.dumps(baseline_control,indent=2,sort_keys=True)+"\n")
pre["baseline_control_path"]=str(OUT/"baseline-control.json")
pre["baseline_control_sha256"]=sha(OUT/"baseline-control.json")
(OUT/"preflight.json").write_text(json.dumps(pre,indent=2,sort_keys=True)+"\n")
print(json.dumps(baseline_control,indent=2,sort_keys=True),flush=True)
if baseline_control["status"]!="PASS_CURRENT406_022_BASELINE_RESTORES_LOCAL_EXCHANGE_MATRIX":raise RuntimeError("022 baseline does not satisfy local exchange matrix with no-worsening floors")
if PREFLIGHT_ONLY:raise SystemExit(0)
target_minimum=old_minimum.copy()
res,a=assemble_lp(selected,pmass,vp,vc,vb,vd,fixed,volume,forced_zero,time_limit=60.,remainder_corner_cap=remcap,receiver_minimum=target_minimum,target_slack_children=target_children,target_slack_caps_mass=target_deficit)
rep={"schema":"numi.human.current406-bounded-held-parent-exchange.v1","status":"LP_"+str(res.status),"physical_admission":False,"native_steps":0,"complete_allocation":False,"allocation_policy":"Target-only bounded-slack local exchange. Selected parents retain exact source-corner M/4 with explicit nonnegative remainder capped by the signed 022 residual (negative roundoff entries provide zero extra remainder capacity). Every receiver upper 950 bound and no-worsening lower bound against 022 are hard. Only the pinned target rows receive slack, bounded from zero through their exact 022 floor deficit; each target still must satisfy mass+slack>=850V. All held rows outside the selected source-parent closure are immutable. Same-material, authenticated 143 one-hop regions, current Fat rest coordinates and the 19.353mm source-point radius remain hard. A nonzero target slack is recorded only as unmet deficit; it is never accepted mass. Physical admission remains false.","optimizer":"SciPy HiGHS continuous LP","cost_optimality_certified":False,"solver_status":int(res.status),"solver_message":str(res.message),"time_limit_seconds":60,"inputs":pre["inputs"],"preflight_sha256":sha(OUT/"preflight.json"),"selected_parent_rows":selected.tolist(),"selected_parent_global_source330_rows":pglobal[selected].tolist(),"candidate_generation_column_mass_cap_used":False,"density_bounds_kg_m3":[LOW,HIGH],"hard_radius_m":RADIUS,"counts":{"variables":int(len(vp)),"selected_parent_rows":int(len(selected)),"local_receiver_rows":int(len(a["child_ids"])),"local_over_cap_inherited_receivers":int(len(a["preexisting_over_cap_local_rows"])),"positive_solution_variables":0},"mass_kg":{"selected_parent_total":float(pmass[selected].sum()),"selected_held_mass_released":held_mass,"source_corner_residual_before":float(qold[selected].sum()),"global_underfloor_deficit_before":float(np.maximum(LOW*vol-childmass,0).sum()),"selected_local_residual_after":None,"global_underfloor_deficit_after":None},"metrics":{"graph_distance_replay_max_error_m":max_graph_radius_error,"peak_rss_bytes":peak(),"elapsed_seconds_before_solve":time.monotonic()-t0},"result":{"proposal_path":None,"merged_ledger_path":None}}
if res.x is None or not res.success:
 rep.update({"status":"FAIL_TARGETED_BOUNDED_SLACK_LOCAL_EXCHANGE_"+str(res.status),"solver_message":str(res.message),"metrics":{"graph_distance_replay_max_error_m":max_graph_radius_error,"elapsed_seconds":time.monotonic()-t0,"peak_rss_bytes":peak()},"counts":{"variables":int(len(a["upper_fraction"])),"selected_parent_rows":int(len(selected)),"local_receiver_rows":int(len(a["child_ids"])),"target_floor_rows":int(a["n_target_slack_vars"])},"result":{"proposal_path":None,"merged_ledger_path":None},"target_slack_kg":{"baseline_bound":float(target_deficit.sum()),"solution":None},"note":"No secondary phase-I solver is run. The target-only bounded-slack formulation has a separately passing exact 022 baseline control; this attempt is source-only and no LP allocation is accepted."})
 (OUT/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\\n")
 print(json.dumps(rep,indent=2,sort_keys=True),flush=True)
 raise SystemExit(0)
solution=np.asarray(res.x,np.float64)
y=solution[:len(vp)];x=y*pmass[vp]
remainder_fraction=solution[len(vp):len(vp)+4*len(selected)].reshape(len(selected),4)
remainder_kg=remainder_fraction*pmass[selected,None]
target_slack_fraction=solution[len(vp)+4*len(selected):]
target_slack_kg=target_slack_fraction*a["receiver_row_scale"][a["target_slack_local_rows"]]
# Directly replay source M/4 and local receiver constraints from the unsimplified rows.
corner=np.column_stack([np.bincount(vp,weights=x*vb[:,k],minlength=len(pmass)) for k in range(4)])
targetcorner=pmass[:,None]*.25
localcorner_err=corner[selected]+remainder_kg-targetcorner[selected]
if np.any(np.abs(localcorner_err)>5e-13*np.maximum(targetcorner[selected],1e-300)):raise RuntimeError("LP selected-parent M/4 plus explicit remainder replay failed")
if np.any(remainder_fraction<0) or np.any(remainder_fraction>remcap+1e-12):raise RuntimeError("LP remainder exceeded nonnegative 022 cap")
delivered=np.bincount(vc,weights=x,minlength=len(tets))
newchild=fixed+delivered
localids=a["child_ids"];lowmass=target_minimum;highmass=HIGH*volume
local_overage=overfixed
# Strict local selected-closure checks; inherited overages are unchanged because selected variables are forced to zero there.
loc=localids[~overfixed[localids]]
tolmass=64*np.finfo(float).eps*np.maximum(highmass[loc],1e-300)
if np.any(newchild[loc]<old_minimum[loc]-tolmass) or np.any(newchild[loc]>highmass[loc]+tolmass):raise RuntimeError("local no-worsening/upper receiver replay failed")
target_rows=a["target_slack_global_rows"]
target_caps=a["target_slack_caps_mass_kg"]
target_floor_error=newchild[target_rows]+target_slack_kg-lowmass[target_rows]
target_slack_tol=64*np.finfo(float).eps*np.maximum(highmass[target_rows],1e-300)
if np.any(target_slack_fraction<0) or np.any(target_slack_fraction>a["target_slack_caps_fraction"]+1e-14):raise RuntimeError("target slack exceeds its bounded 022 deficit")
if np.any(target_floor_error < -target_slack_tol):raise RuntimeError("target floor plus bounded slack replay failed")
if np.any(newchild[target_rows]<old_minimum[target_rows]-target_slack_tol):raise RuntimeError("target exchange worsened its 022 baseline lower bound")
if np.any(target_slack_kg>target_caps+target_slack_tol):raise RuntimeError("target slack mass exceeds 022 original floor deficit")
target_slack_baseline=float(target_caps.sum());target_slack_after=float(target_slack_kg.sum())
if target_slack_after>target_slack_baseline+float(target_slack_tol.sum()):raise RuntimeError("target floor slack worsened baseline")
if np.any(newchild[overfixed]>fixed[overfixed]+tolmass[:0].sum()):raise RuntimeError("selected exchange worsened an inherited over-cap receiver")
# Recompute all parent corners and source residuals from the merged flow ledger.
keep=~selectedmask
keep_p=fp[keep];keep_c=fc[keep];keep_m=fm[keep];keep_b=fb[keep];keep_pt=fpt[keep];keep_d=fd[keep];keep_o=fo[keep];keep_base=fbase[keep]
positive=x>0
np_parent=vp[positive];np_child=vc[positive];np_mass=x[positive];np_b=vb[positive];np_dist=vd[positive]
np_point=np.einsum("ij,ijk->ik",np_b,parentverts[np_parent])
merged_parent=np.concatenate([keep_p,np_parent]);merged_child=np.concatenate([keep_c,np_child]);merged_mass=np.concatenate([keep_m,np_mass]);merged_b=np.concatenate([keep_b,np_b]);merged_point=np.concatenate([keep_pt,np_point]);merged_dist=np.concatenate([keep_d,np_dist]);merged_origin=np.concatenate([keep_o,np.full(len(np_mass),4,np.uint8)]);merged_base=np.concatenate([keep_base,np.full(len(np_mass),-1,np.int64)])
merged_global=pglobal[merged_parent]
cornall=np.column_stack([np.bincount(merged_parent,weights=merged_mass*merged_b[:,k],minlength=len(pmass)) for k in range(4)])
qnew=targetcorner-cornall
childnew=np.bincount(merged_child,weights=merged_mass,minlength=len(tets))
if not np.allclose(childnew,newchild,rtol=0,atol=128*np.finfo(float).eps*np.maximum(highmass,1e-300)):raise RuntimeError("merged child state differs from LP replay")
# Check every selected parent complete and unselected parent q exactly retained within original 022 tolerance.
m4tol=5e-13*np.maximum(targetcorner,1e-300)
if np.any(np.abs(cornall[selected]-targetcorner[selected])>m4tol[selected]):raise RuntimeError("merged selected parent M/4 failed")
if np.any(np.abs(qnew[~np.isin(np.arange(len(pmass)),selected)]-qold[~np.isin(np.arange(len(pmass)),selected)])>m4tol[~np.isin(np.arange(len(pmass)),selected)]):raise RuntimeError("unselected parent residual changed")
if np.any(qnew < -m4tol):raise RuntimeError("merged ledger overdraws source M/4")
if np.any(qnew[selected]>np.maximum(qold[selected],0.)+m4tol[selected]):raise RuntimeError("local exchange increased unallocated source corner mass beyond existing replay tolerance")
if np.any(np.abs(qnew[selected]-remainder_kg)>m4tol[selected]):raise RuntimeError("explicit local remainder differs from merged source corner residual")
# Direct Float32 determinant density gate: only previously-overfull receivers may remain over, and no overage may worsen.
Tdirect=np.zeros(len(tets),np.float64)
for lo in range(0,len(tets),100000):
 hi=min(len(tets),lo+100000);tt=tets[lo:hi];xx=points[tt].astype(np.float64)
 de=np.linalg.det(np.stack([xx[:,1]-xx[:,0],xx[:,2]-xx[:,0],xx[:,3]-xx[:,0]],axis=1));Tdirect[lo:hi]=np.abs(de)/6.
base_direct=childmass/Tdirect;new_direct=childnew/Tdirect
tolrho=5e-10
base_over=base_direct>HIGH+tolrho;new_over=new_direct>HIGH+tolrho
if np.any(new_over&~base_over) or np.any(new_direct[base_over]>base_direct[base_over]+tolrho):raise RuntimeError("direct-volume upper density overage introduced/worsened")
# Save proposal and merged candidate only after all direct checks pass.
proposal=OUT/"local-exchange-proposal.npz"
np.savez_compressed(proposal,selected_parent_rows=selected,source_parent_rows=np_parent,source_global_source330_rows=pglobal[np_parent],child_rows=np_child,source_barycentric=np_b,source_point_world_m=np_point,flow_mass_kg=np_mass,flow_distance_m=np_dist,flow_origin_kind=np.full(len(np_mass),4,np.uint8),selected_parent_corner_mass_kg=cornall[selected],selected_parent_remainder_corner_mass_kg=remainder_kg,target_children=target_rows,target_floor_slack_kg=target_slack_kg,selected_parent_target_corner_mass_kg=targetcorner[selected],local_receiver_rows=localids,local_fixed_mass_kg=fixed[localids],local_delivered_mass_kg=delivered[localids],local_final_mass_kg=newchild[localids],local_child_volume_hybrid_m3=volume[localids])
merged=OUT/"merged-held-ledger.npz"
np.savez_compressed(merged,flow_parent_role_rows=merged_parent,flow_global_source330_rows=merged_global,flow_current_child_rows=merged_child,flow_mass_kg=merged_mass,flow_source_barycentric=merged_b,flow_source_point_world_m=merged_point,flow_current_centroid_distance_m=merged_dist,flow_origin_code=merged_origin,base_ledger_flow_rows=merged_base,parent_source_mass_kg=pmass,parent_corner_residual_after_kg=qnew,child_current_mass_kg=childnew)
rep.update({"status":"PASS_LOCAL_CURRENT406_TARGETED_BOUNDED_SLACK_PARTIAL_EXCHANGE" if target_slack_after < target_slack_baseline-float(target_slack_tol.sum()) else "PASS_LOCAL_CURRENT406_TARGETED_BOUNDED_SLACK_NO_IMPROVEMENT", "target_slack_kg":{"baseline":target_slack_baseline,"after":target_slack_after,"reduction":target_slack_baseline-target_slack_after,"target_rows":target_rows.tolist()}, "objective":"minimize bounded target-floor slack in kilograms; no slack-filled mass is accepted","counts":{"variables":int(len(a["upper_fraction"])),"site_variables":int(len(vp)),"remainder_variables":int(4*len(selected)),"target_slack_variables":int(len(target_rows)),"positive_variables":int(np.count_nonzero(positive)),"new_flow_rows":int(len(np_mass)),"merged_flow_rows":int(len(merged_mass)),"selected_parent_rows":int(len(selected)),"local_receiver_rows":int(len(localids)),"local_receiver_rows_satisfied":int(np.count_nonzero((newchild[localids]>=target_minimum[localids]-64*np.finfo(float).eps*np.maximum(highmass[localids],1e-300))&(newchild[localids]<=highmass[localids]+64*np.finfo(float).eps*np.maximum(highmass[localids],1e-300)))),"inherited_overcap_receivers":int(np.count_nonzero(base_over)),"new_or_worsened_direct_overcap_receivers":0},"mass_kg":{"selected_parent_total":float(pmass[selected].sum()),"selected_held_mass_released":held_mass,"selected_parent_source_residual_before":float(qold[selected].sum()),"selected_parent_mass_allocated":float(np_mass.sum()),"global_flow_total_before":float(fm.sum()),"global_flow_total_after":float(merged_mass.sum()),"global_source_residual_before":float(qold.sum()),"global_source_residual_after":float(qnew.sum()),"global_underfloor_deficit_before":float(np.maximum(LOW*volume-childmass,0).sum()),"global_underfloor_deficit_after":float(np.maximum(LOW*volume-childnew,0).sum()),"local_underfloor_deficit_before":float(np.maximum(lowmass[localids]-childmass[localids],0).sum()),"local_underfloor_deficit_after":float(np.maximum(lowmass[localids]-childnew[localids],0).sum()),"target_slack_baseline":target_slack_baseline,"target_slack_after":target_slack_after,"target_slack_reduction":target_slack_baseline-target_slack_after,"selected_parent_released_old_held_mass":held_mass,"new_mass_from_selected_parent_residual":float(qold[selected].sum()),"net_new_flow_mass":float(merged_mass.sum()-fm.sum())},"metrics":{"solver_objective_target_slack_kg":float(res.fun),"solver_message":str(res.message),"max_selected_source_corner_abs_error_kg":float(np.max(np.abs(localcorner_err),initial=0)),"max_selected_source_corner_relative_error":float(np.max(np.abs(localcorner_err)/np.maximum(targetcorner[selected],1e-300),initial=0)),"max_radius_m":float(np.max(np_dist,initial=0)),"max_radius_graph_recompute_error_m":max_graph_radius_error,"max_local_density_min_kg_m3":float(np.min(newchild[loc]/volume[loc],initial=np.inf)),"max_local_density_max_kg_m3":float(np.max(newchild[loc]/volume[loc],initial=0)),"max_direct_density_kg_m3":float(new_direct.max(initial=0)),"direct_overcap_base_count":int(np.count_nonzero(base_over)),"direct_overcap_after_count":int(np.count_nonzero(new_over)),"elapsed_seconds":time.monotonic()-t0,"peak_rss_bytes":peak()},"result":{"proposal_path":str(proposal),"proposal_sha256":sha(proposal),"merged_ledger_path":str(merged),"merged_ledger_sha256":sha(merged)},"checks":{"all_selected_parent_four_corner_M4_plus_explicit_remainder":True,"nonnegative_remainder_capped_by_022":True,"non_target_lower_bounds_no_worse_than_022":True,"all_local_receiver_upper_and_no_worsening_lower_bounds":True,"target_floor_slack_bounded_by_022_deficit":True,"all_new_source_points_replay_from_source330_rest_coordinates":True,"all_new_source_points_within_unchanged_19_353mm":True,"same_material":True,"143_one_hop_region_permission":True,"all_unselected_held_rows_preserved":True,"no_direct_volume_upper_overage_introduced_or_worsened":True,"physical_admission":False,"complete_allocation":False}})
(OUT/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
print(json.dumps(rep,indent=2,sort_keys=True),flush=True)

