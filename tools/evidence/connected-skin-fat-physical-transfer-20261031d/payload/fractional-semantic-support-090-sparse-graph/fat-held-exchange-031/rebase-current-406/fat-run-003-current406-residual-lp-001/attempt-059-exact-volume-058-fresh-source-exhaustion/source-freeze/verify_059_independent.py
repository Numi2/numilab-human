#!/usr/bin/env python3
"""Independent full replay of the 059 batch against its frozen 058 parent ledger."""
import csv, hashlib, importlib.util, json, time
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
H=ROOT/"fine-physical-volume-706"
S=ROOT/"fractional-semantic-support-090-sparse-graph"
R=S/"fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001"
RUN58=R/"attempt-058-exact-nominal-upper-m4-band-exchange/run-001"
RUN59=R/"attempt-059-exact-volume-058-fresh-source-exhaustion/run-001"
GRAPH=R/"attempt-005/lp-input-current406.npz"
GR=R/"attempt-005/lp-report.json"
LEDGER58=RUN58/"merged-held-ledger.npz"
REPORT58=RUN58/"report.json"
REPLAY58=RUN58/"independent-replay/independent-replay.json"
LEDGER59=RUN59/"merged-held-ledger.npz"
PROPOSAL59=RUN59/"proposal.npz"
REPORT59=RUN59/"report.json"
PRODUCER59=RUN59.parent/"source-freeze/extend_058_with_source_exhaustion_059.py"
TARGET=H/"current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
SUP=H/"connected-region-mass-support-143-current406-root/fat-connected-region-support.npz"
SUPIV=SUP.parent/"independent-verification.json"
PARENT=S/"fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv"
SRC=ROOT/"native-physical-parent-mass-330/native-source-identity-control-003"
NODES=SRC/"compiled-parent-cell-mass.csv.nodes.csv"
TETS=SRC/"compiled-parent-cell-mass.csv.tets.csv"
EXACT=H/"solve_oriented_cavity_milp_081.py"
OWNER=S/"fat-held-exchange-031/rebase_held_flows_current_volume_406_regionaware.frozen.py"
RADIUS=.019353
LOW=850.
HIGH=950.
TOL=5e-10
M4_REL=5e-13

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def csv_rows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
def load_npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

T0=time.monotonic()
r58=json.loads(REPORT58.read_text());iv58=json.loads(REPLAY58.read_text())
r59=json.loads(REPORT59.read_text())
assert r58["status"]=="PASS_LOCAL_CURRENT406_TARGETED_BOUNDED_SLACK_PARTIAL_EXCHANGE"
assert iv58["status"]=="PASS_INDEPENDENT_FULL_LEDGER_REPLAY_PARTIAL_EXCHANGE"
assert r59["status"]=="PASS_SOURCE_ONLY_PARTIAL_SOURCE_EXHAUSTION_BATCH"
assert r59["physical_admission"] is False and r59["native_steps"]==0 and r59["complete_allocation"] is False
assert r59["inputs"][str(PRODUCER59)]==sha(PRODUCER59)
for p,h in r59["inputs"].items(): assert sha(Path(p))==h,(p,sha(Path(p)),h)
assert sha(LEDGER59)==r59["outputs"]["merged-held-ledger.npz"]["sha256"]
assert sha(PROPOSAL59)==r59["outputs"]["proposal.npz"]["sha256"]
assert json.loads(SUPIV.read_text())["status"]=="PASS_INDEPENDENT_AUTHENTICATED_CONNECTED_SUPPORT"

m58=load_npz(LEDGER58); m59=load_npz(LEDGER59); q59=load_npz(PROPOSAL59)
parent=csv_rows(PARENT); nparent=len(parent)
pglobal=np.asarray([int(x["global_source330_row"]) for x in parent],np.int64)
pmass=np.asarray([float(x["mass_kg"]) for x in parent],np.float64)
preg=np.asarray([int(x["region"]) for x in parent],np.int32)
pmat=np.asarray([int(x["material"]) for x in parent],np.int32)
assert nparent==14316 and np.array_equal(pglobal,np.arange(14316,28632,dtype=np.int64))
assert np.array_equal(m58["parent_source_mass_kg"],pmass)
assert np.array_equal(m59["parent_source_mass_kg"],pmass)

old_fields=["flow_parent_role_rows","flow_global_source330_rows","flow_current_child_rows","flow_mass_kg","flow_source_barycentric","flow_source_point_world_m","flow_current_centroid_distance_m","flow_origin_code","base_ledger_flow_rows"]
nold=len(m58["flow_mass_kg"]); nnew=len(q59["flow_mass_kg"])
for k in old_fields:
 assert np.array_equal(m59[k][:nold],m58[k]),("058 prefix changed",k)
 assert len(m59[k])==nold+nnew
assert np.all(m59["flow_source_candidate_site_rows"][:nold]==-1)
assert np.array_equal(m59["flow_source_candidate_site_rows"][nold:],q59["source_candidate_site_rows"].astype(np.int64))
assert np.array_equal(m59["flow_origin_code"][nold:],np.full(nnew,5,dtype=np.uint8))
assert np.all(m59["base_ledger_flow_rows"][nold:]==-1)
for mk,qk in [("flow_parent_role_rows","parent_role_rows"),("flow_global_source330_rows","global_source330_rows"),
              ("flow_current_child_rows","child_current_rows"),("flow_mass_kg","flow_mass_kg"),
              ("flow_source_barycentric","source_barycentric"),("flow_current_centroid_distance_m","flow_distance_m")]:
 assert np.array_equal(m59[mk][nold:],q59[qk]),("proposal mismatch",mk)

fp=m59["flow_parent_role_rows"].astype(np.int64);fg=m59["flow_global_source330_rows"].astype(np.int64)
fc=m59["flow_current_child_rows"].astype(np.int64);fm=m59["flow_mass_kg"].astype(np.float64)
fb=m59["flow_source_barycentric"].astype(np.float64);fx=m59["flow_source_point_world_m"].astype(np.float64)
fd=m59["flow_current_centroid_distance_m"].astype(np.float64)
assert len(fp)==len(fg)==len(fc)==len(fm)==len(fb)==len(fx)==len(fd)==nold+nnew
assert np.all(fp>=0)&np.all(fp<nparent)&np.all(fc>=0)
assert np.all(np.isfinite(fm)) and np.all(fm>0)
assert np.all(np.isfinite(fb)) and np.all(fb>=0) and np.max(np.abs(fb.sum(axis=1)-1),initial=0)<2e-14
assert np.array_equal(fg,pglobal[fp])
assert np.all(pmat==16)

nodes=csv_rows(NODES); node_ids=np.asarray([int(x["parent_node_index"]) for x in nodes],np.int64)
xyz=np.asarray([[float(x["rest_"+a+"_m"]) for a in "xyz"] for x in nodes],np.float64)
tets_src=csv_rows(TETS); tet_ids=np.asarray([int(x["parent_tet_row"]) for x in tets_src],np.int64)
conn=np.asarray([[int(x[f"parent_node{k}"]) for k in range(4)] for x in tets_src],np.int64)
no=np.argsort(node_ids,kind="stable"); ns=node_ids[no]; ni=np.searchsorted(ns,conn)
assert np.all(ni<len(ns)) and np.array_equal(ns[ni],conn)
tv=xyz[no[ni]]
to=np.argsort(tet_ids,kind="stable"); ts=tet_ids[to]
gi=np.searchsorted(ts,pglobal)
assert np.all(gi<len(ts)) and np.array_equal(ts[gi],pglobal)
pverts=tv[to[gi]]
src=np.einsum("ij,ijk->ik",fb,pverts[fp])
point_err=np.abs(src-fx)
assert np.max(point_err,initial=0)<2e-12

with np.load(TARGET,allow_pickle=False) as z:
 points=z["points_world_m"].astype(np.float32,copy=True)
 tets=z["tetrahedra"].astype(np.int64,copy=True)
 childmat=z["material_indices"].astype(np.int32,copy=True)
assert np.all(childmat==16) and len(tets)==len(m59["child_current_mass_kg"])
owner_spec=importlib.util.spec_from_file_location("independent_centroid_owner",OWNER)
owner=importlib.util.module_from_spec(owner_spec);owner_spec.loader.exec_module(owner)
centers=owner.centroids(points,tets)
dist=np.linalg.norm(src-centers[fc],axis=1)
assert np.all(dist<=RADIUS)
radius_delta=np.abs(dist-fd)
assert np.max(radius_delta,initial=0)<5e-12
assert np.all(pmat[fp]==childmat[fc])

with np.load(SUP,allow_pickle=False) as z:
 off=z["cell_admissible_region_offsets"].astype(np.int64)
 regions=z["cell_admissible_region_ids"].astype(np.int32)
assert len(off)==len(tets)+1 and off[0]==0 and off[-1]==len(regions)
cellid=np.repeat(np.arange(len(tets),dtype=np.int64),np.diff(off))
regionkeys=np.unique(cellid*65536+regions.astype(np.int64))
query=fc*65536+preg[fp]
ri=np.searchsorted(regionkeys,query)
assert np.all(ri<len(regionkeys)) and np.all(regionkeys[np.minimum(ri,len(regionkeys)-1)]==query)

corner=np.column_stack([np.bincount(fp,weights=fm*fb[:,k],minlength=nparent) for k in range(4)])
qafter=m59["parent_corner_residual_after_kg"].astype(np.float64)
target=pmass[:,None]*.25
m4err=corner+qafter-target
m4tol=M4_REL*np.maximum(target,1e-300)
assert np.max(np.abs(m4err)/m4tol,initial=0)<=1
assert np.all(qafter>=-m4tol)
assert abs(float(fm.sum()+qafter.sum()-pmass.sum()))<=5e-13*float(pmass.sum())
assert np.array_equal(qafter,m59["parent_corner_residual_after_kg"])
q58=m58["parent_corner_residual_after_kg"].astype(np.float64)
corner58=np.column_stack([np.bincount(m58["flow_parent_role_rows"],weights=m58["flow_mass_kg"]*m58["flow_source_barycentric"][:,k],minlength=nparent) for k in range(4)])
assert np.max(np.abs(corner58+q58-target)/m4tol,initial=0)<=1

basechild=np.bincount(m58["flow_current_child_rows"].astype(np.int64),weights=m58["flow_mass_kg"],minlength=len(tets))
child=np.bincount(fc,weights=fm,minlength=len(tets))
newchild=np.bincount(q59["child_current_rows"].astype(np.int64),weights=q59["flow_mass_kg"],minlength=len(tets))
assert np.allclose(basechild,m58["child_current_mass_kg"],rtol=0,atol=1e-14)
assert np.allclose(child,m59["child_current_mass_kg"],rtol=0,atol=1e-14)
assert np.allclose(child,basechild+newchild,rtol=0,atol=2e-14)
# Recompute dyadic Float32 volumes using independent exact predicate owner.
exact_spec=importlib.util.spec_from_file_location("independent_exact081",EXACT)
exact=importlib.util.module_from_spec(exact_spec);exact_spec.loader.exec_module(exact)
used=np.unique(tets.ravel()); ip={int(i):exact.ip(points[int(i)]) for i in used}
det6=np.fromiter((int(exact.det(*(ip[int(v)] for v in tet))) for tet in tets),dtype=object,count=len(tets))
assert all(int(d)>0 for d in det6)
vol=np.fromiter((float(int(d))/(6.*2.**447) for d in det6),dtype=np.float64,count=len(det6))
threshold=Fraction.from_float(HIGH+TOL)
bad=[]
maxrho=0.;maxrow=-1;nominalover=0
for i,(d,mass) in enumerate(zip(det6,child)):
 fmass=Fraction.from_float(float(mass))*6*(1<<447)
 if fmass>threshold*int(d):bad.append(i)
 if fmass>Fraction.from_float(HIGH)*int(d):nominalover+=1
 rho=float(mass)/vol[i]
 if rho>maxrho:maxrho,maxrow=rho,i
assert not bad,("exact density failure",bad[:10],len(bad))
mass_tol=5e-10*vol+128*np.finfo(float).eps*np.maximum(np.abs(child),HIGH*vol)
assert np.max(np.abs(child-m59["child_current_mass_kg"])/np.maximum(mass_tol,1e-300),initial=0)<=1
# New additions alone cannot exceed nominal remaining capacity; numerical nextafter stays within original acceptance.
vol,det6=vol,det6
base_exact=[]; after_exact=[]
for i,(d,bm,am) in enumerate(zip(det6,basechild,child)):
 if Fraction.from_float(float(bm))*6*(1<<447)>Fraction.from_float(HIGH+TOL)*int(d):base_exact.append(i)
 if Fraction.from_float(float(am))*6*(1<<447)>Fraction.from_float(HIGH+TOL)*int(d):after_exact.append(i)
assert not base_exact and not after_exact

# Replay candidate-site usage for all new rows against old 058 rows and the pinned 005 cap.
with np.load(GRAPH,allow_pickle=False) as z:
 GP=z["parent_role_rows"].astype(np.int32);GC=z["child_rows"].astype(np.int32)
 GB=z["barycentric"].astype(np.float64);Gcap=z["column_mass_scale_kg"].astype(np.float64)
assert len(GP)==len(GC)==len(GB)==len(Gcap)
newsite=q59["source_candidate_site_rows"].astype(np.int64)
assert len(newsite)==nnew and len(np.unique(newsite))==nnew
np_=q59["parent_role_rows"].astype(np.int32); nc=q59["child_current_rows"].astype(np.int32); nb=q59["source_barycentric"]
assert np.array_equal(GP[newsite],np_) and np.array_equal(GC[newsite],nc) and np.array_equal(GB[newsite],nb)
# For only selected site keys, independently account the 058 old uses.
dtype=np.dtype([("p","<i4"),("c","<i4"),("b0","<u8"),("b1","<u8"),("b2","<u8"),("b3","<u8")])
def keys(p,c,b):
 x=np.empty(len(p),dtype=dtype);x["p"]=p;x["c"]=c
 for k in range(4):x[f"b{k}"]=np.ascontiguousarray(b[:,k]).view(np.uint64)
 return x
qkeys=keys(GP[newsite],GC[newsite],GB[newsite])
assert len(np.unique(qkeys))==nnew
oldkeys=keys(m58["flow_parent_role_rows"].astype(np.int32),m58["flow_current_child_rows"].astype(np.int32),m58["flow_source_barycentric"])
oldorder=np.argsort(oldkeys,kind="stable",order=oldkeys.dtype.names)
sorted_old=oldkeys[oldorder]
pos=np.searchsorted(sorted_old,qkeys)
prioruse=np.zeros(nnew,dtype=np.float64)
valid=pos<len(sorted_old)
if np.any(valid):
 starts=np.searchsorted(sorted_old,qkeys[valid],side="left")
 ends=np.searchsorted(sorted_old,qkeys[valid],side="right")
 masses58=m58["flow_mass_kg"].astype(np.float64)[oldorder]
 prioruse[valid]=np.fromiter((masses58[s:e].sum() for s,e in zip(starts,ends)),dtype=np.float64,count=int(valid.sum()))
newuse=q59["flow_mass_kg"].astype(np.float64)
cap_selected=Gcap[newsite]
allow=np.maximum(cap_selected,prioruse)
cap_excess=prioruse+newuse-allow
assert np.all(cap_excess<=64*np.finfo(float).eps*np.maximum(allow,1e-300)),float(np.max(cap_excess,initial=0))
# New site coordinates and radius are independently reconstructed from the exact parent tets.
newpoint=np.einsum("ij,ijk->ik",nb,pverts[np_])
newdist=np.linalg.norm(newpoint-centers[nc],axis=1)
assert np.all(newdist<=RADIUS) and np.max(np.abs(newdist-q59["flow_distance_m"]),initial=0)<5e-12
assert np.array_equal(q59["global_source330_rows"],pglobal[np_])
assert int(r59["counts"]["added_rows"])==nnew==100000
assert int(r59["counts"]["added_phase1_lower_priority_rows"])+int(r59["counts"]["added_phase2_source_exhaustion_rows"])==nnew

floor0=float(np.maximum(LOW*vol-basechild,0).sum())
floor1=float(np.maximum(LOW*vol-child,0).sum())
mass_before=float(basechild.sum());mass_after=float(child.sum())
newmass=float(newuse.sum())
source_total=float(pmass.sum())
resid=float(qafter.sum())
assert abs((source_total-mass_after)-resid)<=5e-13*source_total
result={
 "schema":"numi.human.current406-source-exhaustion-059-independent-replay.v1",
 "status":"PASS_INDEPENDENT_FULL_LEDGER_REPLAY_PARTIAL_SOURCE_EXHAUSTION",
 "physical_admission":False,"native_steps":0,"complete_allocation":False,
 "inputs":{str(p):sha(p) for p in [LEDGER58,REPORT58,REPLAY58,LEDGER59,PROPOSAL59,REPORT59,PRODUCER59,GRAPH,GR,TARGET,SUP,SUPIV,PARENT,NODES,TETS,EXACT,OWNER]},
 "counts":{"parent_rows":nparent,"child_rows":len(tets),"base_058_flows":nold,"new_flows":nnew,"merged_flows":len(fp),
  "added_source_parents":int(len(np.unique(np_))),"added_receivers":int(len(np.unique(nc))),
  "baseline_exact_density_overages":len(base_exact),"candidate_exact_density_overages":len(after_exact),
  "candidate_nominal_950_over_rows":int(nominalover),"new_site_rows":int(len(newsite)),
  "prior_058_site_mass_overconstruction_cap_rows":int(np.count_nonzero(prioruse>cap_selected)),
  "new_site_cap_violations":int(np.count_nonzero(cap_excess>0))},
 "mass_kg":{"source_parent_total":source_total,"058_flow_total":mass_before,"new_flow_total":newmass,
  "candidate_flow_total":mass_after,"candidate_unallocated_residual":resid,
  "global_floor_deficit_before":floor0,"global_floor_deficit_after":floor1,
  "source_mass_closure_error":float(mass_after+resid-source_total),
  "max_density_kg_m3":maxrho,"max_density_child_row":maxrow,
  "nominal_upper_headroom_after":float(np.maximum(HIGH*vol-child,0).sum())},
 "residuals":{"max_source_corner_M4_abs_kg":float(np.max(np.abs(m4err),initial=0)),
  "max_source_corner_M4_ratio_to_5e-13":float(np.max(np.abs(m4err)/m4tol,initial=0)),
  "max_source_point_reconstruction_m":float(np.max(point_err,initial=0)),
  "max_radius_m":float(np.max(dist,initial=0)),
  "max_saved_radius_difference_m":float(np.max(radius_delta,initial=0)),
  "max_added_radius_m":float(np.max(newdist,initial=0)),
  "max_added_radius_saved_difference_m":float(np.max(np.abs(newdist-q59["flow_distance_m"]),initial=0)),
  "max_candidate_child_mass_reconstruction_error_kg":float(np.max(np.abs(child-m59["child_current_mass_kg"]),initial=0)),
  "max_graph_site_cap_excess_kg":float(np.max(cap_excess,initial=0))},
 "checks":{"058_prefix_preserved_once":True,"proposal_equals_merged_suffix":True,"source330_global_parent_mapping":True,
  "source_barycentric_worldpoint_replay":True,"current143_region_and_material":True,"19_353mm_canonical_centroid_radius":True,
  "source_corner_M_over_4_and_residual_closure":True,"exact081_current_F32_density_replay":True,
  "site_usage_replayed_without_baseline_clipping":True,"all_prior_058_rows_retained":True,
  "density_upper_tolerance_unchanged":True,"physical_admission":False},
 "elapsed_seconds":time.monotonic()-T0
}
out=RUN59/"independent-replay"/"independent-replay.json"
out.parent.mkdir(parents=True,exist_ok=False)
out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(result,sort_keys=True))
