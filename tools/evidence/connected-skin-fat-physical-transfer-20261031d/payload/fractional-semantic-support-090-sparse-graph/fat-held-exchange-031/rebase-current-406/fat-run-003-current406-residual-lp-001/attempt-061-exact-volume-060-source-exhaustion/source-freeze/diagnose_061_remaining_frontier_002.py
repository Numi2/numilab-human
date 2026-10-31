#!/usr/bin/env python3
"""Read-only source/site/receiver frontier census after independently verified 061."""
import csv,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
H=ROOT/"fine-physical-volume-706"
S=ROOT/"fractional-semantic-support-090-sparse-graph"
R=S/"fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001"
RUN=R/"attempt-061-exact-volume-060-source-exhaustion/run-001"
LEDGER=RUN/"merged-held-ledger.npz"; IV=RUN/"independent-replay/independent-replay.json"
TARGET=H/"current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
GRAPH=R/"attempt-005/lp-input-current406.npz"
SUP=H/"connected-region-mass-support-143-current406-root/fat-connected-region-support.npz"
PARENT=S/"fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv"
SRC=ROOT/"native-physical-parent-mass-330/native-source-identity-control-003"
NODES=SRC/"compiled-parent-cell-mass.csv.nodes.csv";TETS=SRC/"compiled-parent-cell-mass.csv.tets.csv"
EXACT=H/"solve_oriented_cavity_milp_081.py"
OWNER=S/"fat-held-exchange-031/rebase_held_flows_current_volume_406_regionaware.frozen.py"
ALLOC=R/"attempt-061-exact-volume-060-source-exhaustion/source-freeze/extend_060_with_source_exhaustion_061.py"
OUT=R/"attempt-061-exact-volume-060-source-exhaustion/frontier-diagnostic-002"
LOW=850.;HIGH=950.;RADIUS=.019353;TOL=5e-10;CHUNK=100000
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(1<<20),b""):h.update(x)
 return h.hexdigest()
def rows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
t0=time.monotonic()
iv=json.loads(IV.read_text())
assert iv["status"]=="PASS_INDEPENDENT_FULL_LEDGER_REPLAY_PARTIAL_SOURCE_EXHAUSTION"
if OUT.exists():raise RuntimeError("fresh output dir required")
OUT.mkdir(parents=True)
m=load(LEDGER)
base_rows=rows(PARENT);pglobal=np.asarray([int(x["global_source330_row"]) for x in base_rows],np.int64)
pmass=np.asarray([float(x["mass_kg"]) for x in base_rows],np.float64);preg=np.asarray([int(x["region"]) for x in base_rows],np.int32)
nparent=len(pmass);fp=m["flow_parent_role_rows"].astype(np.int32);fc=m["flow_current_child_rows"].astype(np.int32);fm=m["flow_mass_kg"].astype(np.float64);fb=m["flow_source_barycentric"].astype(np.float64)
q=m["parent_corner_residual_after_kg"].astype(np.float64)
with np.load(TARGET,allow_pickle=False) as z:points=z["points_world_m"].astype(np.float32,copy=True);tets=z["tetrahedra"].astype(np.int64,copy=True);mat=z["material_indices"].astype(np.int32,copy=True)
with np.load(GRAPH,allow_pickle=False) as z:
 P=z["parent_role_rows"].astype(np.int32);PG=z["parent_global_source330_rows"].astype(np.int64);C=z["child_rows"].astype(np.int32);B=z["barycentric"].astype(np.float64);D=z["distance_m"].astype(np.float64);sitecap=z["column_mass_scale_kg"].astype(np.float64)
allocspec=importlib.util.spec_from_file_location("allocation_owner",ALLOC);alloc=importlib.util.module_from_spec(allocspec);allocspec.loader.exec_module(alloc)
remaining,site_map,site_stats,reps=alloc.map_prior_site_usage(P,C,B,sitecap,fp,fc,fb,fm)
assert np.array_equal(PG,pglobal[P])
with np.load(SUP,allow_pickle=False) as z:off=z["cell_admissible_region_offsets"].astype(np.int64);regs=z["cell_admissible_region_ids"].astype(np.int32)
cellid=np.repeat(np.arange(len(tets),dtype=np.int64),np.diff(off));regionkeys=np.unique(cellid*65536+regs.astype(np.int64))
edge=PREG=None
edgekey=C.astype(np.int64)*65536+preg[P].astype(np.int64);ri=np.searchsorted(regionkeys,edgekey);region_ok=(ri<len(regionkeys));region_ok &= regionkeys[np.minimum(ri,len(regionkeys)-1)]==edgekey
mat_ok=mat[C]==np.asarray([int(x["material"]) for x in base_rows],np.int32)[P]
bary_ok=np.all(np.isfinite(B),axis=1)&np.all(B>=0,axis=1)&(np.abs(B.sum(axis=1)-1)<=2e-14)
e_spec=importlib.util.spec_from_file_location("exact081_frontier",EXACT);ex=importlib.util.module_from_spec(e_spec);e_spec.loader.exec_module(ex)
used=np.unique(tets.ravel()); ip={int(i):ex.ip(points[int(i)]) for i in used}
det6=np.fromiter((int(ex.det(*(ip[int(v)] for v in tet))) for tet in tets),dtype=object,count=len(tets))
assert all(int(d)>0 for d in det6)
vol=np.fromiter((float(int(d))/(6.*2.**447) for d in det6),dtype=np.float64,count=len(det6))
childmass=np.bincount(fc,weights=fm,minlength=len(tets))
need=np.maximum(LOW*vol-childmass,0.);head=HIGH*vol-childmass
owner_spec=importlib.util.spec_from_file_location("owner_centroid_frontier",OWNER);owner=importlib.util.module_from_spec(owner_spec);owner_spec.loader.exec_module(owner)
cent=owner.centroids(points,tets)
srcnodes=rows(NODES);nodeids=np.asarray([int(x["parent_node_index"]) for x in srcnodes],np.int64);xyz=np.asarray([[float(x["rest_"+a+"_m"]) for a in "xyz"] for x in srcnodes],np.float64)
srctets=rows(TETS);tids=np.asarray([int(x["parent_tet_row"]) for x in srctets],np.int64);conn=np.asarray([[int(x[f"parent_node{k}"]) for k in range(4)] for x in srctets],np.int64)
no=np.argsort(nodeids,kind="stable");ns=nodeids[no];ni=np.searchsorted(ns,conn);assert np.array_equal(ns[ni],conn);tv=xyz[no[ni]]
to=np.argsort(tids,kind="stable");ts=tids[to];gi=np.searchsorted(ts,pglobal);assert np.array_equal(ts[gi],pglobal);pverts=tv[to[gi]]
raw_legal=0;radius_legal=0;source_positive=0;floor_positive=0;source_exhaustion=0
sum_site_cap=0.;sum_floor_cap=0.;sum_exhaust_cap=0.
perchild_count=np.zeros(len(tets),np.int64);perchild_cap=np.zeros(len(tets),np.float64)
perchild_floor_count=np.zeros(len(tets),np.int64);perchild_floor_cap=np.zeros(len(tets),np.float64)
perchild_exhaust_count=np.zeros(len(tets),np.int64);perchild_exhaust_cap=np.zeros(len(tets),np.float64)
maxradius=0.
for s in range(0,len(P),CHUNK):
 e=min(s+CHUNK,len(P));ix=np.arange(s,e)
 base=region_ok[ix]&mat_ok[ix]&bary_ok[ix]&(remaining[ix]>0)&(head[C[ix]]>0)
 raw_legal+=int(base.sum())
 rr=ix[base]
 if len(rr)==0:continue
 world=np.einsum("ij,ijk->ik",B[rr],pverts[P[rr]])
 radius=np.linalg.norm(world-cent[C[rr]],axis=1)
 maxradius=max(maxradius,float(radius.max(initial=0)))
 rr=rr[radius<=RADIUS]
 radius_legal+=len(rr)
 if len(rr)==0:continue
 bb=B[rr];qq=q[P[rr]]
 ratio=np.full_like(bb,np.inf)
 np.divide(qq,bb,out=ratio,where=bb>0)
 sc=ratio.min(axis=1)
 rowcap=np.minimum(np.minimum(remaining[rr],head[C[rr]]),sc)
 pos=rowcap>0
 rr=rr[pos];rowcap=rowcap[pos]
 if len(rr)==0:continue
 source_positive+=len(rr);sum_site_cap+=float(rowcap.sum())
 cc=C[rr]
 np.add.at(perchild_count,cc,1);np.add.at(perchild_cap,cc,rowcap)
 fmask=need[cc]>0
 rf=rr[fmask];cf=C[rf];capf=np.minimum(rowcap[fmask],need[cf])
 good=capf>0;rf=rf[good];cf=cf[good];capf=capf[good]
 floor_positive+=len(rf);sum_floor_cap+=float(capf.sum())
 np.add.at(perchild_floor_count,cf,1);np.add.at(perchild_floor_cap,cf,capf)
 smask=need[cc]<=0
 rs=rr[smask];cs=C[rs];caps=rowcap[smask]
 good=caps>0;rs=rs[good];cs=cs[good];caps=caps[good]
 source_exhaustion+=len(rs);sum_exhaust_cap+=float(caps.sum())
 np.add.at(perchild_exhaust_count,cs,1);np.add.at(perchild_exhaust_cap,cs,caps)
# corner and receiver frontier
m4=np.repeat(pmass[:,None]*.25,4,axis=1)
qsummary=[]
for k in range(4):
 vals=q[:,k]
 order=np.argsort(vals)[::-1][:8]
 qsummary.append({"corner":k,"positive_parent_count":int(np.count_nonzero(vals>0)),"residual_mass_kg":float(vals.sum()),
  "max_residual_kg":float(vals.max(initial=0)),"top_parent_role_rows":[{"row":int(i),"global_source330_row":int(pglobal[i]),"q_kg":float(vals[i]),"M4_target_kg":float(m4[i,k])} for i in order if vals[i]>0]})
floorrows=np.flatnonzero(need>0)
ordfloor=floorrows[np.argsort(need[floorrows])[::-1][:20]]
topfloor=[]
for c in ordfloor:
 rk=int(np.searchsorted(off,c));rids=regs[off[c]:off[c+1]]
 topfloor.append({"child_row":int(c),"floor_deficit_kg":float(need[c]),"upper_headroom_kg":float(head[c]),
  "admissible_regions":[int(x) for x in rids],"radius_region_material_site_rows":int(perchild_count[c]),
  "one_site_cap_sum_upper_bound_kg":float(perchild_cap[c]),"floor_phase_site_rows":int(perchild_floor_count[c]),
  "floor_phase_one_site_cap_sum_upper_bound_kg":float(perchild_floor_cap[c])})
topheadrows=np.flatnonzero(head>0);ordhead=topheadrows[np.argsort(head[topheadrows])[::-1][:10]]
report={"schema":"numi.human.current406-source-exhaustion-061-frontier-diagnostic.v1","status":"PASS_SOURCE_ONLY_FRONTIER_DIAGNOSTIC_NOT_INFEASIBILITY_PROOF","physical_admission":False,"native_steps":0,"complete_allocation":False,
 "inputs":{str(x):sha(x) for x in [LEDGER,IV,TARGET,GRAPH,SUP,PARENT,NODES,TETS,EXACT,OWNER,ALLOC]},
 "source_state":{"parent_count":nparent,"source_mass_total_kg":float(pmass.sum()),"unallocated_residual_total_kg":float(q.sum()),
  "corner_residuals":qsummary,"parents_with_any_positive_corner_residual":int(np.count_nonzero(np.any(q>0,axis=1))),
  "parents_with_all_four_positive_corner_residuals":int(np.count_nonzero(np.all(q>0,axis=1)))},
 "receiver_state":{"child_count":len(tets),"floor_deficit_kg":float(need.sum()),"underfloor_receivers":int(len(floorrows)),
  "max_floor_deficit_kg":float(need.max(initial=0)),"top_floor_deficits":topfloor,
  "nominal950_headroom_kg":float(np.maximum(head,0).sum()),"above_nominal_950_receivers":int(np.count_nonzero(head<0)),
  "top_nominal_headroom":[{"child_row":int(c),"headroom_kg":float(head[c]),"floor_deficit_kg":float(need[c]),"admissible_regions":[int(x) for x in regs[off[c]:off[c+1]]]} for c in ordhead]},
 "candidate_frontier":{"candidate_graph_site_rows":len(P),"exact_unique_rows":int(site_stats["unique_exact_site_rows"]),
  "prior_flow_rows":int(len(fm)),"prior_flow_rows_matched_to_005_sites":int(site_stats["prior_flows_matched_to_graph_sites"]),
  "prior_flow_rows_outside_005_site_domain":int(site_stats["prior_flow_rows_not_in_graph_site_domain"]),
  "prior_site_cap_overrun_rows":int(site_stats["prior_site_cap_overrun_rows"]),
  "site_rows_with_remaining_cap_before_geometry":int(site_stats["available_unique_site_rows"]),
  "site_rows_same_material_region_bary_positive_head_before_radius":raw_legal,"site_rows_after_exact_radius":radius_legal,"site_rows_with_positive_current_source_corner_cap":source_positive,
  "source_positive_site_row_cap_sum_kg_nonadditive_upper_bound":sum_site_cap,
  "floor_phase_site_rows":floor_positive,"floor_phase_site_cap_sum_kg_nonadditive_upper_bound":sum_floor_cap,
  "source_exhaustion_phase_site_rows":source_exhaustion,"source_exhaustion_site_cap_sum_kg_nonadditive_upper_bound":sum_exhaust_cap,
  "max_candidate_radius_recomputed_m":maxradius},
 "scope_note":"Per-site capacity sums are deliberately labeled non-additive upper bounds because rows compete for the same parent-corner residual and receiver capacity. This census is not a continuous/global feasibility proof. It identifies remaining exact graph support under fixed 143 region, material, radius and exact M/4 residual state.",
 "elapsed_seconds":time.monotonic()-t0}
out=OUT/"frontier.json";out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":report["status"],"source_state":report["source_state"],"receiver_summary":{k:v for k,v in report["receiver_state"].items() if k!="top_floor_deficits" and k!="top_nominal_headroom"},"candidate_frontier":report["candidate_frontier"],"top_floor_deficits":topfloor,"elapsed_seconds":report["elapsed_seconds"],"receipt_sha256":sha(out)},sort_keys=True))
