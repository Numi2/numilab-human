#!/usr/bin/env python3
import csv,hashlib,json,time,resource
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
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
OUT=R/"attempt-026-held-parent-exchange-top100/run-004";OUT.mkdir(parents=True,exist_ok=True)
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
def assemble_lp(selected,parent_mass,var_parent,var_child,bary,dist,fixed,child_volume,forced_zero, time_limit=60.):
 selected=np.asarray(selected,dtype=np.int32);var_parent=np.asarray(var_parent,dtype=np.int32);var_child=np.asarray(var_child,dtype=np.int32);bary=np.asarray(bary,dtype=np.float64);dist=np.asarray(dist,dtype=np.float64)
 if bary.shape!=(len(var_parent),4) or len(var_child)!=len(var_parent) or len(dist)!=len(var_parent):raise RuntimeError("variable-array shape mismatch")
 pmap={int(p):i for i,p in enumerate(selected)}
 try:lp=np.asarray([pmap[int(p)] for p in var_parent],np.int32)
 except KeyError as e:raise RuntimeError(f"variable parent not selected: {e}")
 child_ids=np.unique(var_child);lc=np.searchsorted(child_ids,var_child).astype(np.int32)
 pm=parent_mass[var_parent];nvar=len(var_parent);nsel=len(selected)
 if np.any(pm<=0):raise RuntimeError("nonpositive source parent mass")
 # Variables are parent-mass fractions. Four source rows are exactly sum(y*b_k)=1/4.
 eq_rows=np.concatenate([lp+k*nsel for k in range(4)])
 eq_cols=np.tile(np.arange(nvar,dtype=np.int32),4)
 eq_vals=np.concatenate([bary[:,k] for k in range(4)])
 Aeq=coo_matrix((eq_vals,(eq_rows,eq_cols)),shape=(4*nsel,nvar)).tocsr()
 beq=np.full(4*nsel,.25,dtype=np.float64)
 max_parent_mass_by_child=np.zeros(len(child_ids),dtype=np.float64);np.maximum.at(max_parent_mass_by_child,lc,pm)
 child_scale=np.maximum.reduce([HIGH*child_volume[child_ids],max_parent_mass_by_child/1e10,np.full(len(child_ids),1e-300)])
 coeff=pm/child_scale[lc]
 high=HIGH*child_volume[child_ids]; low=LOW*child_volume[child_ids]
 local_fixed=fixed[child_ids]
 local_over=local_fixed>high
 # Preserve pre-existing over-cap rows without worsening them; selected variables into them are fixed to zero.
 live=~local_over[lc]&~np.asarray(forced_zero,dtype=bool)
 live_local=np.flatnonzero(~local_over)
 rmap=np.full(len(child_ids),-1,np.int32);rmap[live_local]=np.arange(len(live_local),dtype=np.int32)
 ar=np.repeat(2*rmap[lc[live]],2)+np.tile(np.array([0,1],np.int32),int(live.sum()))
 ac=np.repeat(np.flatnonzero(live),2)
 av=np.empty(2*int(live.sum()),dtype=np.float64);av[0::2]=coeff[live];av[1::2]=-coeff[live]
 Aub=coo_matrix((av,(ar,ac)),shape=(2*len(live_local),nvar)).tocsr()
 bub=np.empty(2*len(live_local),dtype=np.float64)
 bub[0::2]=(high[live_local]-local_fixed[live_local])/child_scale[live_local]
 bub[1::2]=(local_fixed[live_local]-low[live_local])/child_scale[live_local]
 capfrac=np.full(nvar,np.inf,dtype=np.float64)
 qcorner=parent_mass[var_parent,None]*.25
 np.divide(qcorner,bary,out=np.broadcast_to(capfrac[:,None],bary.shape).copy(),where=bary>0)
 # Stable per-column source M/4 bound; sum of four corner equations also bounds total parent share.
 for k in range(4):
  capfrac=np.minimum(capfrac,np.divide(.25,bary[:,k],out=np.full(nvar,np.inf),where=bary[:,k]>0))
 ub=np.minimum(capfrac,1.)
 ub[np.asarray(forced_zero,dtype=bool)|local_over[lc]]=0.
 cost=pm*(dist/RADIUS + 1e-10)
 
 checks={"cost_finite":bool(np.isfinite(cost).all()),"Aeq_finite":bool(np.isfinite(Aeq.data).all()),"beq_finite":bool(np.isfinite(beq).all()),"Aub_finite":bool(np.isfinite(Aub.data).all()),"bub_finite":bool(np.isfinite(bub).all()),"bounds_finite":bool(np.isfinite(ub).all()),"bounds_nonnegative":bool(np.all(ub>=0)),"Aeq_indices_in_range":bool(Aeq.indices.size==0 or (Aeq.indices.min()>=0 and Aeq.indices.max()<nvar)),"Aeq_indptr_ok":bool(Aeq.indptr[-1]==Aeq.nnz),"Aub_indices_in_range":bool(Aub.indices.size==0 or (Aub.indices.min()>=0 and Aub.indices.max()<nvar)),"Aub_indptr_ok":bool(Aub.indptr[-1]==Aub.nnz),"Aeq_shape":list(Aeq.shape),"Aub_shape":list(Aub.shape),"nvar":int(nvar),"Aeq_nnz":int(Aeq.nnz),"Aub_nnz":int(Aub.nnz),"receiver_row_coefficient_cap":1e10,"cost_minmax":[float(cost.min(initial=0)),float(cost.max(initial=0))],"Aeq_minmax":[float(Aeq.data.min(initial=0)),float(Aeq.data.max(initial=0))],"beq_minmax":[float(beq.min(initial=0)),float(beq.max(initial=0))],"Aub_minmax":[float(Aub.data.min(initial=0)),float(Aub.data.max(initial=0))],"bub_minmax":[float(bub.min(initial=0)),float(bub.max(initial=0))],"upper_minmax":[float(ub.min(initial=0)),float(ub.max(initial=0))],"zero_Aeq_rows":int(np.count_nonzero(np.diff(Aeq.indptr)==0)),"zero_Aub_rows":int(np.count_nonzero(np.diff(Aub.indptr)==0))}
 if not all(checks[k] for k in ["cost_finite","Aeq_finite","beq_finite","Aub_finite","bub_finite","bounds_finite","bounds_nonnegative","Aeq_indices_in_range","Aeq_indptr_ok","Aub_indices_in_range","Aub_indptr_ok"]):raise RuntimeError("sparse LP model preflight invalid: "+str(checks))
 (OUT/"matrix-preflight.json").write_text(json.dumps(checks,indent=2,sort_keys=True)+"\n")
 res=linprog(cost,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=list(zip(np.zeros(nvar),ub)),method="highs",options={"time_limit":float(time_limit),"primal_feasibility_tolerance":1e-10,"dual_feasibility_tolerance":1e-10,"presolve":True})
 return res,{"selected_parent_rows":selected,"variable_parent_rows":var_parent,"variable_child_rows":var_child,"barycentric":bary,"distance_m":dist,"child_ids":child_ids,"child_local_rows":lc,"parent_local_rows":lp,"fixed_child_mass_kg":fixed[child_ids],"Aeq":Aeq,"beq":beq,"Aub":Aub,"bub":bub,"upper_fraction":ub,"forced_zero":np.asarray(forced_zero,dtype=bool),"preexisting_over_cap_local_rows":child_ids[local_over],"cost":cost}
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
control_receipt=control()
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
selected=np.sort(np.asarray([x["parent_role_row"] for x in top],np.int32))
if len(selected)!=100 or len(np.unique(selected))!=100:raise RuntimeError("top-100 parent selection malformed")
selectedmask=np.isin(fp,selected)
held_parent_rows=int(np.count_nonzero(selectedmask));held_mass=float(fm[selectedmask].sum())
# Keep all graph rows for selected donors; generation cap is not a hard mass limit once source corner and receiver bounds are enforced.
gmask=np.isin(P,selected);gp=P[gmask];gc=C[gmask];gb=B[gmask];gd=D[gmask]
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
pre={"schema":"numi.human.current406-held-parent-exchange-preflight.v1","status":"PASS_BOUNDED_LOCAL_PARENT_EXCHANGE_PREFLIGHT","physical_admission":False,"native_steps":0,"inputs":{str(x):sha(x) for x in [G,GPREP,GREP,MERGE,V022,P022,TARGET,SUP,SUPIV,PARENT,SRC/"compiled-parent-cell-mass.csv.nodes.csv",SRC/"compiled-parent-cell-mass.csv.tets.csv"]},"control":control_receipt,"selected_parent_rows":selected.tolist(),"counts":{"candidate_graph_columns":int(len(gp)),"prior_selected_held_rows":held_parent_rows,"prior_selected_held_children":int(len(np.unique(hc))),"final_unique_parent_child_site_columns":int(len(vp)),"candidate_children":int(len(np.unique(vc))),"fixed_over_upper_receivers":int(np.count_nonzero(overfixed))},"mass_kg":{"selected_parent_total":float(pmass[selected].sum()),"selected_held_mass_released":held_mass,"selected_source_corner_residual":float(qold[selected].sum()),"underfloor_before_global":float(np.maximum(LOW*vol-childmass,0).sum()),"max_fixed_over_upper":float(np.max(fixed-high,initial=0))},"metrics":{"graph_recomputed_max_distance_error_m":max_graph_radius_error,"max_held_barycentric_reconstruction_error_m":float(np.max(np.abs(hxyz-hpt),initial=0)),"selected_parents":100,"reopened_parent_scope":"all source mass per selected parent; exact four M/4 rows, all prior held flows released once"},"allocation":{"generation_column_mass_cap_used":False,"hard_source_M4":True,"hard_radius_m":RADIUS,"material_match":True,"143_one_hop_region_support":True,"density_kg_m3":[LOW,HIGH],"unselected_held_flows_fixed":True},"physical_admission":False,"complete_allocation":False}
(OUT/"preflight.json").write_text(json.dumps(pre,indent=2,sort_keys=True)+"\n")
print(json.dumps({"stage":"preflight","status":pre["status"],"counts":pre["counts"],"mass_kg":pre["mass_kg"],"sha256":sha(OUT/"preflight.json")},sort_keys=True),flush=True)
# Solve full selected-parent source moments against all recipients reachable by these legal source sites.
with np.load(G,allow_pickle=False) as z:
 volume=z["child_volume_hybrid_m3"].astype(np.float64)
res,a=assemble_lp(selected,pmass,vp,vc,vb,vd,fixed,volume,forced_zero,time_limit=60.)
rep={"schema":"numi.human.current406-bounded-held-parent-exchange.v1","status":"LP_"+str(res.status),"physical_admission":False,"native_steps":0,"complete_allocation":False,"allocation_policy":"The top 100 source parents by optimistic current lower-deficit support are fully reopened. Every original selected held flow is removed once; every exact old held point plus every pinned 143/radius/material-permitted candidate point is a variable. Per-site candidate generation caps are omitted; source corner M/4 equations, actual source-tet rest coordinates, same-material, authenticated one-hop region CSR, 19.353mm radius, and 850-950 density bounds remain hard. All other parent flows are immutable.","optimizer":"SciPy HiGHS continuous LP","cost_optimality_certified":False,"solver_status":int(res.status),"solver_message":str(res.message),"time_limit_seconds":60,"inputs":pre["inputs"],"preflight_sha256":sha(OUT/"preflight.json"),"selected_parent_rows":selected.tolist(),"selected_parent_global_source330_rows":pglobal[selected].tolist(),"candidate_generation_column_mass_cap_used":False,"density_bounds_kg_m3":[LOW,HIGH],"hard_radius_m":RADIUS,"counts":{"variables":int(len(vp)),"selected_parent_rows":int(len(selected)),"local_receiver_rows":int(len(a["child_ids"])),"local_over_cap_inherited_receivers":int(len(a["preexisting_over_cap_local_rows"])),"positive_solution_variables":0},"mass_kg":{"selected_parent_total":float(pmass[selected].sum()),"selected_held_mass_released":held_mass,"source_corner_residual_before":float(qold[selected].sum()),"global_underfloor_deficit_before":float(np.maximum(LOW*vol-childmass,0).sum()),"selected_local_residual_after":None,"global_underfloor_deficit_after":None},"metrics":{"graph_distance_replay_max_error_m":max_graph_radius_error,"peak_rss_bytes":peak(),"elapsed_seconds_before_solve":time.monotonic()-t0},"result":{"proposal_path":None,"merged_ledger_path":None}}
if res.x is None or not res.success:
 rep["status"]="FAIL_LOCAL_HELD_PARENT_EXCHANGE_"+str(res.status)
 rep["metrics"]["elapsed_seconds"]=time.monotonic()-t0;rep["metrics"]["peak_rss_bytes"]=peak()
 (OUT/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
 print(json.dumps(rep,indent=2,sort_keys=True),flush=True);raise SystemExit(0)
y=np.asarray(res.x,np.float64);x=y*pmass[vp]
# Directly replay source M/4 and local receiver constraints from the unsimplified rows.
corner=np.column_stack([np.bincount(vp,weights=x*vb[:,k],minlength=len(pmass)) for k in range(4)])
targetcorner=pmass[:,None]*.25
corner_err=corner-targetcorner
localcorner_err=corner_err[selected]
if np.any(np.abs(localcorner_err)>5e-13*np.maximum(targetcorner[selected],1e-300)):raise RuntimeError("LP selected-parent M/4 replay failed")
delivered=np.bincount(vc,weights=x,minlength=len(tets))
newchild=fixed+delivered
localids=a["child_ids"];lowmass=LOW*volume;highmass=HIGH*volume
local_overage=overfixed
# Strict local selected-closure checks; inherited overages are unchanged because selected variables are forced to zero there.
loc=localids[~overfixed[localids]]
tolmass=64*np.finfo(float).eps*np.maximum(highmass[loc],1e-300)
if np.any(newchild[loc]<lowmass[loc]-tolmass) or np.any(newchild[loc]>highmass[loc]+tolmass):raise RuntimeError("local receiver density replay failed")
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
np.savez_compressed(proposal,selected_parent_rows=selected,source_parent_rows=np_parent,source_global_source330_rows=pglobal[np_parent],child_rows=np_child,source_barycentric=np_b,source_point_world_m=np_point,flow_mass_kg=np_mass,flow_distance_m=np_dist,flow_origin_kind=np.full(len(np_mass),4,np.uint8),selected_parent_corner_mass_kg=cornall[selected],selected_parent_target_corner_mass_kg=targetcorner[selected],local_receiver_rows=localids,local_fixed_mass_kg=fixed[localids],local_delivered_mass_kg=delivered[localids],local_final_mass_kg=newchild[localids],local_child_volume_hybrid_m3=volume[localids])
merged=OUT/"merged-held-ledger.npz"
np.savez_compressed(merged,flow_parent_role_rows=merged_parent,flow_global_source330_rows=merged_global,flow_current_child_rows=merged_child,flow_mass_kg=merged_mass,flow_source_barycentric=merged_b,flow_source_point_world_m=merged_point,flow_current_centroid_distance_m=merged_dist,flow_origin_code=merged_origin,base_ledger_flow_rows=merged_base,parent_source_mass_kg=pmass,parent_corner_residual_after_kg=qnew,child_current_mass_kg=childnew)
rep.update({"status":"PASS_LOCAL_CURRENT406_HELD_PARENT_EXCHANGE_CANDIDATE","counts":{"variables":int(len(vp)),"positive_variables":int(np.count_nonzero(positive)),"new_flow_rows":int(len(np_mass)),"merged_flow_rows":int(len(merged_mass)),"selected_parent_rows":int(len(selected)),"local_receiver_rows":int(len(localids)),"local_receiver_rows_satisfied":int(np.count_nonzero((newchild[localids]>=lowmass[localids]-64*np.finfo(float).eps*np.maximum(highmass[localids],1e-300))&(newchild[localids]<=highmass[localids]+64*np.finfo(float).eps*np.maximum(highmass[localids],1e-300)))),"inherited_overcap_receivers":int(np.count_nonzero(base_over)),"new_or_worsened_direct_overcap_receivers":0},"mass_kg":{"selected_parent_total":float(pmass[selected].sum()),"selected_held_mass_released":held_mass,"selected_parent_source_residual_before":float(qold[selected].sum()),"selected_parent_mass_allocated":float(np_mass.sum()),"global_flow_total_before":float(fm.sum()),"global_flow_total_after":float(merged_mass.sum()),"global_source_residual_before":float(qold.sum()),"global_source_residual_after":float(qnew.sum()),"global_underfloor_deficit_before":float(np.maximum(LOW*volume-childmass,0).sum()),"global_underfloor_deficit_after":float(np.maximum(LOW*volume-childnew,0).sum()),"local_underfloor_deficit_before":float(np.maximum(lowmass[localids]-childmass[localids],0).sum()),"local_underfloor_deficit_after":float(np.maximum(lowmass[localids]-childnew[localids],0).sum()),"selected_parent_released_old_held_mass":held_mass,"new_mass_from_selected_parent_residual":float(qold[selected].sum()),"net_new_flow_mass":float(merged_mass.sum()-fm.sum())},"metrics":{"solver_objective":float(res.fun),"solver_message":str(res.message),"max_selected_source_corner_abs_error_kg":float(np.max(np.abs(localcorner_err),initial=0)),"max_selected_source_corner_relative_error":float(np.max(np.abs(localcorner_err)/np.maximum(targetcorner[selected],1e-300),initial=0)),"max_radius_m":float(np.max(np_dist,initial=0)),"max_radius_graph_recompute_error_m":max_graph_radius_error,"max_local_density_min_kg_m3":float(np.min(newchild[loc]/volume[loc],initial=np.inf)),"max_local_density_max_kg_m3":float(np.max(newchild[loc]/volume[loc],initial=0)),"max_direct_density_kg_m3":float(new_direct.max(initial=0)),"direct_overcap_base_count":int(np.count_nonzero(base_over)),"direct_overcap_after_count":int(np.count_nonzero(new_over)),"elapsed_seconds":time.monotonic()-t0,"peak_rss_bytes":peak()},"result":{"proposal_path":str(proposal),"proposal_sha256":sha(proposal),"merged_ledger_path":str(merged),"merged_ledger_sha256":sha(merged)},"checks":{"all_selected_parent_four_corner_M4":True,"all_local_receiver_bounds":True,"all_new_source_points_replay_from_source330_rest_coordinates":True,"all_new_source_points_within_unchanged_19_353mm":True,"same_material":True,"143_one_hop_region_permission":True,"all_unselected_held_rows_preserved":True,"no_direct_volume_upper_overage_introduced_or_worsened":True,"physical_admission":False,"complete_allocation":False}})
(OUT/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
print(json.dumps(rep,indent=2,sort_keys=True),flush=True)

