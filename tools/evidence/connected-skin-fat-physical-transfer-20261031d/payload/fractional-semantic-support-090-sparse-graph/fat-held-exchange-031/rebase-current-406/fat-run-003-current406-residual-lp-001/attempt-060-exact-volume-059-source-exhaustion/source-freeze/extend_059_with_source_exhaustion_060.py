#!/usr/bin/env python3
"""Fresh 100k source-exhaustion batch from the independently replayed 058 ledger.

Uses the existing 009 deterministic two-phase allocator. All 058 held flows
enter the state exactly once; exact F32 dyadic 081 volumes drive nominal 950
caps and the unchanged 950+5e-10 acceptance replay.
"""
import csv,hashlib,importlib.util,json,resource,time
from pathlib import Path
import numpy as np
from fractions import Fraction
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
H=ROOT/"fine-physical-volume-706"
S=ROOT/"fractional-semantic-support-090-sparse-graph"
R=S/"fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001"
RUN=R/"attempt-059-exact-volume-058-fresh-source-exhaustion/run-001"
OUT=R/"attempt-060-exact-volume-059-source-exhaustion/run-001"
GRAPH=R/"attempt-005/lp-input-current406.npz"
GRAPH_REPORT=R/"attempt-005/lp-report.json"
LEDGER=RUN/"merged-held-ledger.npz"
LEDGER_REPORT=RUN/"report.json"
LEDGER_INDEPENDENT=RUN/"independent-replay/independent-replay.json"
TARGET=H/"current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
SUP=H/"connected-region-mass-support-143-current406-root/fat-connected-region-support.npz"
SUPIV=SUP.parent/"independent-verification.json"
PARENT=S/"fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv"
SRC=ROOT/"native-physical-parent-mass-330/native-source-identity-control-003"
NODES=SRC/"compiled-parent-cell-mass.csv.nodes.csv"
TETS=SRC/"compiled-parent-cell-mass.csv.tets.csv"
EXACT=H/"solve_oriented_cavity_milp_081.py"
REBASE=S/"fat-held-exchange-031/rebase_held_flows_current_volume_406_regionaware.frozen.py"
PHASE3=R/"attempt-022-source-exhaustion-reserved-batch/source-freeze/phase3_source_exhaustion_009.py"
RADIUS=.019353
LOW=850.
HIGH=950.
TOL_DENSITY=5e-10
MAX_ROWS=100000
LOWER_FIRST_ROWS=50000
CHUNK=100000
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def read_csv(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
def peak():
 import sys
 n=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 return n if sys.platform=="darwin" else n*1024
def make_keys(p,c,b):
 dt=np.dtype([("p","<i4"),("c","<i4"),("b0","<u8"),("b1","<u8"),("b2","<u8"),("b3","<u8")])
 x=np.empty(len(p),dtype=dt)
 x["p"]=p;x["c"]=c
 x["b0"]=b[:,0].view(np.uint64);x["b1"]=b[:,1].view(np.uint64);x["b2"]=b[:,2].view(np.uint64);x["b3"]=b[:,3].view(np.uint64)
 return x
def map_prior_site_usage(site_p,site_c,site_b,site_cap,flow_p,flow_c,flow_b,flow_mass):
 """Exact (parent, child, bary bits) join; duplicate graph rows collapse to first row."""
 n=len(site_p)
 if len(site_c)!=n or len(site_b)!=n or len(site_cap)!=n:raise RuntimeError("candidate site arrays misaligned")
 keys=make_keys(site_p,site_c,site_b)
 order=np.argsort(keys,kind="stable",order=keys.dtype.names)
 sorted_keys=keys[order]
 starts=np.r_[0,np.flatnonzero(sorted_keys[1:]!=sorted_keys[:-1])+1]
 reps=order[starts]
 group_cap=np.minimum.reduceat(site_cap[order],starts)
 dup_count=n-len(reps)
 del sorted_keys
 fq=make_keys(flow_p,flow_c,flow_b)
 pos=np.searchsorted(keys,fq,sorter=order)
 inrange=pos<len(order)
 hit=np.zeros(len(fq),dtype=bool)
 if np.any(inrange):
  test=order[pos[inrange]]
  hit[inrange]=keys[test]==fq[inrange]
 mapped=np.full(len(fq),-1,dtype=np.int64)
 mapped[hit]=order[pos[hit]]
 used=np.bincount(mapped[hit],weights=flow_mass[hit],minlength=n)
 remaining=np.zeros(n,dtype=np.float64)
 remaining[reps]=group_cap-used[reps]
 # Candidate-generation caps are never borrowed from an already represented
 # held site. A saturated prior site is simply unavailable for this batch.
 negative=remaining[reps]<0
 over=float(-remaining[reps][negative].sum())
 remaining[reps]=np.maximum(remaining[reps],0.)
 return remaining,mapped,{"candidate_site_rows":n,"unique_exact_site_rows":len(reps),"duplicate_graph_rows":dup_count,
  "prior_flow_rows":len(flow_mass),"prior_flows_matched_to_graph_sites":int(hit.sum()),
  "prior_flow_mass_on_graph_sites_kg":float(flow_mass[hit].sum()),"prior_flow_rows_not_in_graph_site_domain":int((~hit).sum()),
  "sites_exhausted_by_prior_flow":int(np.count_nonzero(remaining[reps]<=0)),
  "prior_site_cap_overrun_rows":int(negative.sum()),"prior_site_cap_overrun_mass_kg":over,
  "available_unique_site_rows":int(np.count_nonzero(remaining[reps]>0))},reps
def controls():
 gp=np.asarray([4274,4274,6006],np.int32)
 gc=np.asarray([17,17,9],np.int32)
 gb=np.asarray([[.25,.25,.25,.25],[.25,.25,.25,.25],[1.,0,0,0]],np.float64)
 cap=np.asarray([.30,.25,.40])
 fp=np.asarray([4274,6006],np.int32);fc=np.asarray([17,9],np.int32)
 fb=np.asarray([[.25,.25,.25,.25],[1.,0,0,0]],np.float64);fm=np.asarray([.10,.05])
 rem,mapped,stats,reps=map_prior_site_usage(gp,gc,gb,cap,fp,fc,fb,fm)
 if stats["unique_exact_site_rows"]!=2 or stats["prior_flows_matched_to_graph_sites"]!=2:raise RuntimeError("exact graph-site join control failed")
 if mapped[0]!=0 or mapped[1]!=2 or not np.allclose(rem[[0,1,2]],[.15,0.,.35],rtol=0,atol=1e-16):raise RuntimeError("duplicate-site capacity/first-row control failed")
 if len(reps)!=2:raise RuntimeError("deduplicated graph-row domain control failed")
 phase=importlib.util.spec_from_file_location("phase3_source_exhaustion_009",PHASE3)
 alloc=importlib.util.module_from_spec(phase);phase.loader.exec_module(alloc)
 parent_global=np.zeros(6007,dtype=np.int64);parent_global[4274]=14316+4274;parent_global[6006]=14316+6006
 p=np.asarray([4274,6006,4274],np.int32);c=np.asarray([17,9,17],np.int32)
 b=np.asarray([[.25]*4,[1.,0,0,0],[0.,.5,.5,0]],np.float64);d=np.asarray([.001,.002,.003])
 q=np.zeros((6007,4));q[4274]=[.2]*4;q[6006]=[.2]*4
 site=np.asarray([.1,.1,.1]);need=np.zeros(20);head=np.zeros(20);head[[9,17]]=.3;need[17]=.1
 out=alloc.select_batch(np.arange(3,dtype=np.int64),p,parent_global,c,b,d,parent_global,site,q,need,head,4,2)
 if len(out["rows"])==0 or np.any(out["global"]!=parent_global[out["parents"]]):raise RuntimeError("actual 009 allocator noncontiguous source/global control failed")
 # Verify the actual two-phase rule: phase two cannot consume a receiver still below floor.
 blocked=alloc.select_batch(np.arange(3,dtype=np.int64),p,parent_global,c,b,d,parent_global,site,q,need,head,3,0)
 if np.any(need[blocked["children"]]>0) or 17 in blocked["children"]:raise RuntimeError("actual 009 phase-two floor control failed")
 return {"status":"PASS_EXACT_SITE_JOIN_AND_009_ALLOCATOR_CONTROLS","site_join":stats,"duplicate_site_representatives":reps.tolist(),
  "remaining_site_caps_kg":rem.tolist(),"allocator_rows":int(len(out["rows"])),"allocator_source_map":"PASS"}
def exact_vol(points,tets,mod):
 used=np.unique(tets.ravel());pd={int(i):mod.ip(points[int(i)]) for i in used}
 det6=np.fromiter((mod.det(*(pd[int(v)] for v in tet)) for tet in tets),dtype=object,count=len(tets))
 if any(int(x)<=0 for x in det6):raise RuntimeError("exact 081 volume contains nonpositive current cell")
 v=np.fromiter((float(int(x))/(6.*2.**447) for x in det6),dtype=np.float64,count=len(tets))
 if np.any(~np.isfinite(v)) or np.any(v<=0):raise RuntimeError("exact current F32 volumes invalid")
 return v,det6
def main():
 import argparse,sys
 ap=argparse.ArgumentParser()
 ap.add_argument("--control-only",action="store_true")
 a=ap.parse_args()
 c=controls()
 if a.control_only:
  print(json.dumps({"stage":"control-only","controls":c},sort_keys=True));return
 if OUT.exists():raise RuntimeError("output directory must be new")
 OUT.mkdir(parents=True)
 t0=time.monotonic()
 inputs=[GRAPH,GRAPH_REPORT,LEDGER,LEDGER_REPORT,LEDGER_INDEPENDENT,TARGET,SUP,SUPIV,PARENT,NODES,TETS,EXACT,REBASE,PHASE3,Path(__file__).resolve()]
 pins={str(p):sha(p) for p in inputs}
 expected={str(GRAPH):"4798d69dad18ffca8acedd65ec1b761135763d5121c0de67042c5cc083271328",
  str(GRAPH_REPORT):"d5abd6f7d3c9b44d7a474127436284d2213549156afa4e92a462088abd299767",
  str(TARGET):"5dcebfd751dedcb1891559352e9d0e673ea3a8e15a8edeae8179c6d04e847211",
  str(SUP):"e24f0d343af79153e5030bee5fe85f8a12676a17daa56adcfa99e15956da4526",
  str(SUPIV):"b7a78c081cf2768fc66ffd0403e5abf581ed703f39858ad1aab14cea84ed7482",
  str(LEDGER):"8940a3e080c5544a1307d3175dc069ca51f01fd9b2097a235a131bc06915ebd8",
  str(LEDGER_REPORT):"0b1242b824a1ae873179ca00c8aaf70497c04bea7c6e8be359c2ddf0399a6480",
  str(LEDGER_INDEPENDENT):"f1cd4f88924a814f33553b112c3c64c79e3f86de3a805391e6c1b4fd826f661b",
  str(EXACT):"3f7c276bef3bb4427f6acb41f6eeedffbeae273c9493a3b7854b244a5a2fa024",
  str(PHASE3):"206ce9a64d907b7bfb8d8cdb511bb3946264e7f69555f53cade6af77ba9461a5"}
 for p,h in expected.items():
  if pins[p]!=h:raise RuntimeError("pinned input hash mismatch "+p)
 lr=json.loads(LEDGER_REPORT.read_text());iv=json.loads(LEDGER_INDEPENDENT.read_text())
 if lr.get("physical_admission") is not False or lr.get("native_steps")!=0 or lr.get("complete_allocation") is not False:raise RuntimeError("058 producer scope changed")
 if iv.get("status")!="PASS_INDEPENDENT_FULL_LEDGER_REPLAY_PARTIAL_SOURCE_EXHAUSTION":raise RuntimeError("058 independent replay is not passing")
 if json.loads(SUPIV.read_text()).get("status")!="PASS_INDEPENDENT_AUTHENTICATED_CONNECTED_SUPPORT":raise RuntimeError("143 support receipt not passing")
 # Import only the frozen batch-selection function from 009; do not call its old 022-based runner.
 ps=importlib.util.spec_from_file_location("phase3_source_exhaustion_009",PHASE3);alloc=importlib.util.module_from_spec(ps);ps.loader.exec_module(alloc)
 from importlib.util import spec_from_file_location,module_from_spec
 owner_spec=spec_from_file_location("rebase_owner_406",REBASE);owner=module_from_spec(owner_spec);owner_spec.loader.exec_module(owner)
 ex_spec=spec_from_file_location("exact081",EXACT);ex=module_from_spec(ex_spec);ex_spec.loader.exec_module(ex)
 with np.load(TARGET,allow_pickle=False) as z:
  points=np.asarray(z["points_world_m"],dtype=np.float32).copy();tets=np.asarray(z["tetrahedra"],dtype=np.int64).copy();childmat=np.asarray(z["material_indices"],dtype=np.int32).copy()
 with np.load(GRAPH,allow_pickle=False) as z:
  P=np.asarray(z["parent_role_rows"],dtype=np.int32).copy();PG=np.asarray(z["parent_global_source330_rows"],dtype=np.int64).copy()
  C=np.asarray(z["child_rows"],dtype=np.int32).copy();B=np.asarray(z["barycentric"],dtype=np.float64).copy()
  D=np.asarray(z["distance_m"],dtype=np.float64).copy();cap=np.asarray(z["column_mass_scale_kg"],dtype=np.float64).copy()
 if not(len(P)==len(PG)==len(C)==len(B)==len(D)==len(cap)):raise RuntimeError("attempt005 site columns have inconsistent row domains")
 if B.shape!=(len(P),4) or not np.array_equal(PG,np.arange(14316,28632,dtype=np.int64)[P]):raise RuntimeError("attempt005 parent/global crosswalk mismatch")
 if np.any(P<0)|np.any(P>=14316)|np.any(C<0)|np.any(C>=len(tets)):raise RuntimeError("attempt005 row ids outside current406 domains")
 if np.any(~np.isfinite(B))|np.any(B<0)|np.any(np.abs(B.sum(axis=1)-1)>2e-14)|np.any(~np.isfinite(cap))|np.any(cap<0):raise RuntimeError("candidate site barycentric/capacity invalid")
 if np.any(~np.isfinite(D)) or np.any(D<0):raise RuntimeError("candidate site distances invalid")
 with np.load(LEDGER,allow_pickle=False) as z: L={k:z[k].copy() for k in z.files}
 fp=L["flow_parent_role_rows"].astype(np.int32);fc=L["flow_current_child_rows"].astype(np.int32)
 fm=L["flow_mass_kg"].astype(np.float64);fb=L["flow_source_barycentric"].astype(np.float64)
 fx=L["flow_source_point_world_m"].astype(np.float64);fo=L["flow_origin_code"].astype(np.uint8);fbase=L["base_ledger_flow_rows"].astype(np.int64);fsite=L.get("flow_source_candidate_site_rows",np.full(len(fp),-1,dtype=np.int64)).astype(np.int64)
 parent_rows=read_csv(PARENT);pglobal=np.asarray([int(x["global_source330_row"]) for x in parent_rows],np.int64)
 pmass=np.asarray([float(x["mass_kg"]) for x in parent_rows],np.float64);preg=np.asarray([int(x["region"]) for x in parent_rows],np.int32);pmat=np.asarray([int(x["material"]) for x in parent_rows],np.int32)
 if not np.array_equal(pglobal,np.arange(14316,28632,dtype=np.int64)) or not np.array_equal(mpmass:=L["parent_source_mass_kg"],pmass):raise RuntimeError("058 source330 parent map/mass mismatch")
 # Materialize source330 coordinates in the exact role-parent order.
 src_nodes=read_csv(NODES);src_tets=read_csv(TETS)
 node_ids=np.asarray([int(x["parent_node_index"]) for x in src_nodes],np.int64);xyz=np.asarray([[float(x["rest_"+a+"_m"]) for a in "xyz"] for x in src_nodes],np.float64)
 tet_ids=np.asarray([int(x["parent_tet_row"]) for x in src_tets],np.int64);conn=np.asarray([[int(x[f"parent_node{k}"]) for k in range(4)] for x in src_tets],np.int64)
 no=np.argsort(node_ids,kind="stable");ns=node_ids[no];ni=np.searchsorted(ns,conn)
 if np.any(ni>=len(ns)) or not np.array_equal(ns[ni],conn):raise RuntimeError("source330 tet/node ID crosswalk failed")
 tetverts=xyz[no[ni]];to=np.argsort(tet_ids,kind="stable");ts=tet_ids[to];gi=np.searchsorted(ts,pglobal)
 if np.any(gi>=len(ts)) or not np.array_equal(ts[gi],pglobal):raise RuntimeError("source330 role/global tet crosswalk failed")
 pverts=tetverts[to[gi]]
 if not np.all(pmat==16) or not np.all(childmat==16):raise RuntimeError("Fat material label changed")
 if len(fc)!=len(fm) or len(fsite)!=len(fp) or np.any(fp<0)|np.any(fp>=len(pmass))|np.any(fc<0)|np.any(fc>=len(tets)) or np.any(~np.isfinite(fm))|np.any(fm<=0):raise RuntimeError("prior flow row domain/mass invalid")
 if not np.array_equal(L["flow_global_source330_rows"],pglobal[fp]):raise RuntimeError("058 retained source global IDs mismatched")
 source_rebuild=np.einsum("ij,ijk->ik",fb,pverts[fp])
 if np.max(np.abs(source_rebuild-fx),initial=0)>2e-12:raise RuntimeError("058 source point reconstruction changed")
 # Current base geometry convention uses the frozen native four-corner centroid owner.
 centers=owner.centroids(points,tets)
 direct_old=np.linalg.norm(fx-centers[fc],axis=1)
 invalid_radius=direct_old>RADIUS
 # Keep/restore only rows that satisfy unchanged material, source-region, and radius ownership.
 regiondata=np.load(SUP,allow_pickle=False)
 off=regiondata["cell_admissible_region_offsets"].astype(np.int64);regions=regiondata["cell_admissible_region_ids"].astype(np.int32);regiondata.close()
 if len(off)!=len(tets)+1 or off[0]!=0 or off[-1]!=len(regions):raise RuntimeError("143 support row domain mismatch")
 regionkeys=np.repeat(np.arange(len(tets),dtype=np.int64),np.diff(off))*65536+regions.astype(np.int64)
 regionkeys=np.unique(regionkeys)
 old_region_key=fc.astype(np.int64)*65536+preg[fp]
 ri=np.searchsorted(regionkeys,old_region_key); old_region_ok=(ri<len(regionkeys))
 old_region_ok &= regionkeys[np.minimum(ri,len(regionkeys)-1)]==old_region_key
 old_material_ok=childmat[fc]==pmat[fp]
 valid_old=(~invalid_radius)&old_region_ok&old_material_ok
 release_old=~valid_old
 release_reason=np.zeros(len(fp),np.uint8)
 release_reason[~old_material_ok]=2;release_reason[invalid_radius]=3;release_reason[~old_region_ok]=5
 vp=fp[valid_old];vc=fc[valid_old];vm=fm[valid_old];vb=fb[valid_old];vx=fx[valid_old];vo=fo[valid_old];vbase=fbase[valid_old]
 child_before=np.bincount(vc,weights=vm,minlength=len(tets))
 corner_before=np.column_stack([np.bincount(vp,weights=vm*vb[:,k],minlength=len(pmass)) for k in range(4)])
 target_corner=pmass[:,None]*.25
 q0=target_corner-corner_before
 m4tol=5e-13*np.maximum(target_corner,1e-300)
 if np.any(q0 < -m4tol):raise RuntimeError("058 retained source flows overdraw source M/4")
 if np.any(np.abs(corner_before+q0-target_corner)>m4tol):raise RuntimeError("058 source-corner M/4 state invalid")
 old_store=L["parent_corner_residual_after_kg"]
 if np.any(np.abs(old_store-q0)>m4tol):raise RuntimeError("058 independent source residual does not replay")
 # Exact F32 volume rows, not the candidate graph's hybrid volume lane.
 vol,det6=exact_vol(points,tets,ex)
 if len(vol)!=len(child_before):raise RuntimeError("current406 child volume domain mismatch")
 threshold=Fraction.from_float(HIGH+TOL_DENSITY)
 old_exact_over=[int(i) for i,(d,mass) in enumerate(zip(det6,child_before)) if Fraction.from_float(float(mass))*6*(1<<447)>threshold*int(d)]
 if old_exact_over:raise RuntimeError("058 base ledger fails exact density acceptance under independent 081 volume replay")
 need=np.maximum(LOW*vol-child_before,0.)
 head=HIGH*vol-child_before
 # Every already-over-nominal cell is immutable for this addition pass.
 overnom=np.flatnonzero(head<0)
 # Current source and recipient roles/same-material/143 permission are checked on graph site IDs.
 edgekey=C.astype(np.int64)*65536+preg[P].astype(np.int64);at=np.searchsorted(regionkeys,edgekey)
 region_ok=at<len(regionkeys)
 region_ok &= regionkeys[np.minimum(at,len(regionkeys)-1)]==edgekey
 mat_ok=childmat[C]==pmat[P]
 bary_ok=np.all(np.isfinite(B),axis=1)&np.all(B>=0,axis=1)&(np.abs(B.sum(axis=1)-1)<=2e-14)
 # Directly recompute all potential graph radii in bounded tiles with canonical current centroids.
 directD=np.full(len(P),np.inf,dtype=np.float64)
 base_eligible=region_ok&mat_ok&bary_ok
 for start in range(0,len(P),CHUNK):
  stop=min(start+CHUNK,len(P));ix=np.arange(start,stop)
  use=base_eligible[ix]
  if not np.any(use):continue
  rr=ix[use];world=np.einsum("ij,ijk->ik",B[rr],pverts[P[rr]])
  directD[rr]=np.linalg.norm(world-centers[C[rr]],axis=1)
 D_error=np.abs(directD-D)
 geom_ok=np.isfinite(directD)&(directD<=RADIUS)
 # Exact site identity join against every held row, counting held mass once.
 site_remaining,prior_site_map,site_stats,reps=map_prior_site_usage(P,C,B,cap,vp,vc,vb,vm)
 # Original graph rows are the source-site identity domain; duplicate aliases get no second capacity.
 candidates=np.flatnonzero((site_remaining>0)&base_eligible&geom_ok&(head[C]>0))
 log={"status":"PREFLIGHT_PASS","candidate_count":int(len(candidates)),"graph_columns":len(P),"unique_graph_sites":len(reps),
  "prior_flows":len(fp),"retained_prior_flows":int(len(vm)),"released_prior_flows":int(release_old.sum()),"released_prior_mass_kg":float(fm[release_old].sum()),
  "source_residual_kg":float(q0.sum()),"global_underfloor_deficit_kg":float(need.sum()),"nominal_upper_headroom_kg":float(np.maximum(head,0).sum()),
  "nominal_over_rows_immutable":int(len(overnom)),"site_usage":site_stats,"max_graph_distance_recompute_error_m":float(np.max(D_error[np.isfinite(D_error)],initial=0)),
  "exact081_volume_method":"positive dyadic det6/(6*2^447), all receivers","max_exact_density_before":float(np.max(child_before/vol,initial=0)),
  "control":c,"inputs":pins}
 (OUT/"preflight.json").write_text(json.dumps(log,indent=2,sort_keys=True)+"\n")
 # Two-phase original 009 allocator, now seeded by the entire independently verified 058 ledger exactly once.
 result=alloc.select_batch(candidates,P,pglobal,C,B,directD,pglobal,site_remaining,q0,need,np.maximum(head,0),MAX_ROWS,LOWER_FIRST_ROWS)
 rows=result["rows"]
 if len(rows)==0:raise RuntimeError("no positive legal source-site additions from 058 state")
 np_parent=result["parents"];np_global=result["global"];np_child=result["children"];np_bary=result["bary"];np_mass=result["mass"]
 # Independent direct geometry/material/region replay for selected rows.
 point=np.einsum("ij,ijk->ik",np_bary,pverts[np_parent])
 direct=np.linalg.norm(point-centers[np_child],axis=1)
 rk=np.searchsorted(regionkeys,np_child.astype(np.int64)*65536+preg[np_parent].astype(np.int64))
 selected_region=(rk<len(regionkeys))
 selected_region &= regionkeys[np.minimum(rk,len(regionkeys)-1)]==np_child.astype(np.int64)*65536+preg[np_parent].astype(np.int64)
 if np.any(direct>RADIUS)|np.any(childmat[np_child]!=pmat[np_parent])|np.any(~selected_region):raise RuntimeError("selected 009 site failed direct 19.353mm/material/region replay")
 if not np.array_equal(np_global,pglobal[np_parent]):raise RuntimeError("selected site parent/global map mismatch")
 # Source M/4 and receiver state replay.
 delta_corner=np.column_stack([np.bincount(np_parent,weights=np_mass*np_bary[:,k],minlength=len(pmass)) for k in range(4)])
 delta_child=np.bincount(np_child,weights=np_mass,minlength=len(tets))
 qafter=q0-delta_corner
 if np.any(delta_corner>q0+m4tol) or np.any(qafter < -m4tol):raise RuntimeError("fresh 009 additions exceed original parent-corner M/4")
 expected_child=child_before+delta_child
 if np.any(delta_child>np.maximum(head,0)+64*np.finfo(float).eps*np.maximum(HIGH*vol,1e-300)):raise RuntimeError("fresh additions exceed nominal receiver headroom")
 # Compose all 058 retained rows and new source-exhaustion rows. Origin 5 is unique to this batch.
 merged_p=np.concatenate([vp,np_parent]);merged_c=np.concatenate([vc,np_child]);merged_m=np.concatenate([vm,np_mass]);merged_b=np.concatenate([vb,np_bary])
 merged_x=np.concatenate([vx,point]);merged_d=np.concatenate([direct_old[valid_old],direct])
 merged_o=np.concatenate([vo,np.full(len(rows),5,np.uint8)]);merged_base=np.concatenate([vbase,np.full(len(rows),-1,np.int64)]);merged_site=np.concatenate([fsite[valid_old],rows.astype(np.int64)])
 merged_global=pglobal[merged_p]
 replay_child=np.bincount(merged_c,weights=merged_m,minlength=len(tets))
 child_replay_tol=5e-10*vol+128*np.finfo(float).eps*np.maximum(np.abs(expected_child),HIGH*vol)
 if np.any(np.abs(replay_child-expected_child)>child_replay_tol):raise RuntimeError("merged full child mass replay failed")
 replay_corner=np.column_stack([np.bincount(merged_p,weights=merged_m*merged_b[:,k],minlength=len(pmass)) for k in range(4)])
 if np.any(np.abs(replay_corner+qafter-target_corner)>m4tol):raise RuntimeError("merged full source M/4 replay failed")
 if abs((merged_m.sum()+qafter.sum())-pmass.sum())>5e-13*pmass.sum():raise RuntimeError("source mass total plus unallocated corner residual failed")
 # Entire current domain: exact dyadic density upper bound with unchanged tolerance.
 exact_over=[];maxrho=0.;maxrow=-1
 for i,(d,mass) in enumerate(zip(det6,replay_child)):
  if Fraction.from_float(float(mass))*6*(1<<447)>threshold*int(d):exact_over.append(int(i))
  rho=float(mass)/vol[i]
  if rho>maxrho:maxrho,maxrow=rho,i
 if exact_over:
  np.savez_compressed(OUT/"unaccepted-density-proposal.npz",source_candidate_site_rows=rows,parent_role_rows=np_parent,global_source330_rows=np_global,child_rows=np_child,barycentric=np_bary,flow_mass_kg=np_mass,base_child_mass_kg=child_before,merged_child_mass_kg=replay_child,exact_overage_rows=np.asarray(exact_over,dtype=np.int32))
  raise RuntimeError(f"full exact 081 density replay failed on {len(exact_over)} cells; diagnostic retained")
 # Report exact lower/upper accounting and preserve all baseline rows that passed actual owner radius/region.
 low_before=float(np.maximum(LOW*vol-child_before,0).sum());low_after=float(np.maximum(LOW*vol-replay_child,0).sum())
 maxm4=float(np.max(np.abs(replay_corner+qafter-target_corner),initial=0))
 proposal=OUT/"proposal.npz"
 np.savez_compressed(proposal,source_candidate_site_rows=rows,parent_role_rows=np_parent,global_source330_rows=np_global,child_current_rows=np_child,
  source_barycentric=np_bary,source_point_world_m=point,flow_mass_kg=np_mass,flow_distance_m=direct,
  parent_corner_residual_after_kg=qafter,child_lower_residual_after_kg=np.maximum(LOW*vol-replay_child,0),
  child_upper_headroom_after_kg=np.maximum(HIGH*vol-replay_child,0))
 merged=OUT/"merged-held-ledger.npz"
 np.savez_compressed(merged,flow_parent_role_rows=merged_p,flow_global_source330_rows=merged_global,flow_current_child_rows=merged_c,
  flow_mass_kg=merged_m,flow_source_barycentric=merged_b,flow_source_point_world_m=merged_x,flow_current_centroid_distance_m=merged_d,
  flow_origin_code=merged_o,base_ledger_flow_rows=merged_base,parent_source_mass_kg=pmass,parent_corner_residual_after_kg=qafter,
  child_current_mass_kg=replay_child,flow_source_candidate_site_rows=merged_site)
 rep={"schema":"numi.human.current406-source-exhaustion-from-059.v1","status":"PASS_SOURCE_ONLY_PARTIAL_SOURCE_EXHAUSTION_BATCH","physical_admission":False,"native_steps":0,"complete_allocation":False,
  "allocation_method":"Existing deterministic 009 two-phase bounded greedy batch; source-only partial allocation, no optimizer optimality claim.",
  "baseline_prior_ledger_sha256":sha(LEDGER),"baseline_prior_independent_replay_sha256":sha(LEDGER_INDEPENDENT),"inputs":pins,
  "limits":{"total_new_rows":MAX_ROWS,"lower_first_rows":LOWER_FIRST_ROWS,"source_exhaustion_rows":MAX_ROWS-LOWER_FIRST_ROWS,
  "density_nominal_upper_kg_m3":HIGH,"density_acceptance_tolerance_kg_m3":TOL_DENSITY,"source_moment_M_over_4":True,"radius_m":RADIUS},
  "counts":{"parent_rows":len(pmass),"child_rows":len(tets),"prior_flow_rows":len(fm),"prior_site_metadata_rows_retained":int(np.count_nonzero(fsite[valid_old]>=0)),"owner_radius_region_retained_prior_flows":len(vm),
  "prior_rows_released_for_owner_radius_or_region":int(release_old.sum()),"candidate_graph_rows":len(P),"unique_candidate_graph_sites":len(reps),
  "available_after_exact_prior_site_usage":len(candidates),"prior_graph_site_usage":site_stats,
  "added_rows":len(rows),"added_phase1_lower_priority_rows":int(result["lower_rows"]),"added_phase2_source_exhaustion_rows":int(result["source_rows"]),
  "added_distinct_parents":int(len(np.unique(np_parent))),"added_distinct_receivers":int(len(np.unique(np_child))),
  "nominal_over_receivers_immutable":int(len(overnom)),"baseline_exact_over_tolerance_rows":0,"candidate_exact_over_tolerance_rows":0},
  "mass_kg":{"source_parent_total":float(pmass.sum()),"baseline_058_flow_total":float(fm.sum()),"owner-valid-held_flow_total":float(vm.sum()),
  "prior_owner_release_mass":float(fm[release_old].sum()),"new_flow_total":float(np_mass.sum()),"merged_flow_total":float(merged_m.sum()),
  "unallocated_source_residual_before":float((pmass[:,None]*.25-corner_before).sum()),"unallocated_source_residual_after":float(qafter.sum()),
  "global_floor_deficit_before":low_before,"global_floor_deficit_after":low_after,"lower_priority_added_kg":float(result["lower_mass"]),
  "source_exhaustion_added_kg":float(result["source_mass"]),"nominal_upper_headroom_before":float(np.maximum(head,0).sum()),
  "nominal_upper_headroom_after":float(np.maximum(HIGH*vol-replay_child,0).sum())},
  "metrics":{"max_selected_distance_m":float(np.max(direct,initial=0)),"max_graph_distance_replay_error_m":float(np.max(D_error[np.isfinite(D_error)],initial=0)),
  "max_parent_M4_abs_residual_kg":maxm4,"max_parent_M4_relative_residual":float(np.max(np.abs(replay_corner+qafter-target_corner)/m4tol,initial=0)),
  "max_density_kg_m3":maxrho,"max_density_child_row":maxrow,"maximum_volume_relative_delta_vs_graph_hybrid":float(np.max(np.abs(vol-np.load(GRAPH,allow_pickle=False)["child_volume_hybrid_m3"])/vol,initial=0)),
  "peak_rss_bytes":peak(),"elapsed_seconds":time.monotonic()-t0},
  "checks":{"all_prior_flows_counted_once":"PASS","prior_site_metadata_preserved":"PASS","source_moments_original_M_over_4":"PASS","exact081_volume_receiver_caps":"PASS","nominal950_optimizer_caps":"PASS","unchanged950_plus_5e-10_exact_replay":"PASS","same_material":"PASS","143_current406_one_hop_region":"PASS","19_353mm_canonical_centroid_radius":"PASS","all_baseline_flows_preserved_except_explicit_invalid_release":"PASS","no_new_or_worsened_density_overages":"PASS","lower_density_complete":False,"complete_allocation":False},
  "release_reason_counts":{"radius":int(np.count_nonzero(invalid_radius)),"current_region":int(np.count_nonzero(~old_region_ok)),"material":int(np.count_nonzero(~old_material_ok))},
  "outputs":{}}
 for pth in [proposal,merged]:
  rep["outputs"][pth.name]={"sha256":sha(pth),"size_bytes":pth.stat().st_size}
 rp=OUT/"report.json";rp.write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
 # Read-back output pins, maps, and scope before declaring source-only partial progress.
 reread=json.loads(rp.read_text())
 if reread.get("physical_admission") is not False or reread.get("native_steps")!=0 or reread.get("complete_allocation") is not False:raise RuntimeError("source-only scope changed")
 for name,meta in reread["outputs"].items():
  if sha(OUT/name)!=meta["sha256"] or (OUT/name).stat().st_size!=meta["size_bytes"]:raise RuntimeError("serialized output pin mismatch")
 print(json.dumps({"status":rep["status"],"counts":rep["counts"],"mass_kg":rep["mass_kg"],"metrics":rep["metrics"],"outputs":rep["outputs"],"report_sha256":sha(rp)},sort_keys=True))
if __name__=="__main__":main()
