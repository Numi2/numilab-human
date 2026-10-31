#!/usr/bin/env python3
# Independent full-ledger replay for attempt 058. This verifier reads frozen
# producer outputs but recomputes row identity, moments, radius, region and
# exact dyadic density from original inputs.
import csv,hashlib,json,time,importlib.util
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
H=ROOT/"fine-physical-volume-706"
S=ROOT/"fractional-semantic-support-090-sparse-graph"
R=S/"fat-held-exchange-031/rebase-current-406/fat-run-003-current406-residual-lp-001"
RUN=R/"attempt-058-exact-nominal-upper-m4-band-exchange/run-001"
MERGE=RUN/"merged-held-ledger.npz"
PROPOSAL=RUN/"local-exchange-proposal.npz"
REPORT=RUN/"report.json"
BASELEDGER=R/"attempt-022-source-exhaustion-reserved-batch/run-001/merged-held-centroid-noncentroid-source-exhaustion.npz"
TARGET=H/"current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
PARENT=S/"fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv"
SRC=ROOT/"native-physical-parent-mass-330/native-source-identity-control-003"
SUP=H/"connected-region-mass-support-143-current406-root/fat-connected-region-support.npz"
SUPIV=SUP.parent/"independent-verification.json"
EXACT=H/"solve_oriented_cavity_milp_081.py"
OUT=RUN/"independent-replay"
RADIUS=.019353
HIGH=950.
M4_REL=5e-13
INNER_M4_REL=.25*M4_REL
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def rows(p):
 with Path(p).open(newline="") as f:return list(csv.DictReader(f))
started=time.monotonic()
report=json.loads(REPORT.read_text())
assert report["status"]=="PASS_LOCAL_CURRENT406_TARGETED_BOUNDED_SLACK_PARTIAL_EXCHANGE"
assert report["physical_admission"] is False and report["native_steps"]==0 and report["complete_allocation"] is False
assert sha(MERGE)==report["result"]["merged_ledger_sha256"]
assert sha(PROPOSAL)==report["result"]["proposal_sha256"]
for p,h in report["inputs"].items(): assert sha(Path(p))==h,(p,sha(Path(p)),h)
with np.load(MERGE,allow_pickle=False) as z: m={k:z[k] for k in z.files}
with np.load(PROPOSAL,allow_pickle=False) as z: q={k:z[k] for k in z.files}
with np.load(BASELEDGER,allow_pickle=False) as z: base={k:z[k] for k in z.files}
parents=rows(PARENT)
pglobal=np.asarray([int(x["global_source330_row"]) for x in parents],np.int64)
pmass_csv=np.asarray([float(x["mass_kg"]) for x in parents],np.float64)
preg=np.asarray([int(x["region"]) for x in parents],np.int32)
pmat=np.asarray([int(x["material"]) for x in parents],np.int32)
nparent=len(parents)
assert nparent==14316 and np.array_equal(np.asarray([int(x["parent_row"]) for x in parents]),np.arange(nparent))
assert np.array_equal(m["parent_source_mass_kg"],pmass_csv)
fp=m["flow_parent_role_rows"].astype(np.int64);fc=m["flow_current_child_rows"].astype(np.int64)
fm=m["flow_mass_kg"].astype(np.float64);fb=m["flow_source_barycentric"].astype(np.float64)
fpoint=m["flow_source_point_world_m"].astype(np.float64)
fd_saved=m["flow_current_centroid_distance_m"].astype(np.float64)
origin=m["flow_origin_code"].astype(np.uint8);baseidx=m["base_ledger_flow_rows"].astype(np.int64)
assert len(fp)==len(fc)==len(fm)==len(fb)==len(fpoint)==len(fd_saved)==len(origin)==len(baseidx)
assert np.all(np.isfinite(fm)) and np.all(fm>0)
assert np.all(np.isfinite(fb)) and np.all(fb>=0) and np.max(np.abs(fb.sum(axis=1)-1),initial=0)<2e-14
assert np.array_equal(m["flow_global_source330_rows"],pglobal[fp])
# Prove every unselected base row is retained once, unchanged; all replacement rows match proposal.
selected=np.asarray(report["selected_parent_rows"],np.int64)
bp=base["flow_parent_role_rows"].astype(np.int64)
selected_mask=np.isin(bp,selected)
expected_base=np.flatnonzero(~selected_mask)
expected_base_ids=base["base_ledger_flow_rows"].astype(np.int64)[expected_base]
retained=origin!=4
assert int(retained.sum())==len(expected_base) and np.array_equal(baseidx[retained],expected_base_ids)
assert np.all(baseidx[~retained]==-1) and np.all(origin[~retained]==4)
for key in ["flow_parent_role_rows","flow_global_source330_rows","flow_current_child_rows","flow_mass_kg","flow_source_barycentric","flow_source_point_world_m","flow_current_centroid_distance_m","flow_origin_code"]:
 assert np.array_equal(m[key][retained],base[key][expected_base]),key
new=~retained
for mk,qk in [("flow_parent_role_rows","source_parent_rows"),("flow_current_child_rows","child_rows"),("flow_mass_kg","flow_mass_kg"),("flow_source_barycentric","source_barycentric"),("flow_source_point_world_m","source_point_world_m"),("flow_current_centroid_distance_m","flow_distance_m")]:
 assert np.array_equal(m[mk][new],q[qk]),mk
assert np.all(q["flow_origin_kind"]==4)
# Source-330 geometry and exact child domain.
nodes=rows(SRC/"compiled-parent-cell-mass.csv.nodes.csv")
xyz=np.asarray([[float(r["rest_"+a+"_m"]) for a in "xyz"] for r in nodes],np.float64)
trows=rows(SRC/"compiled-parent-cell-mass.csv.tets.csv")
gtet=np.asarray([int(r["parent_tet_row"]) for r in trows],np.int64)
tnodes=np.asarray([[int(r[f"parent_node{k}"]) for k in range(4)] for r in trows],np.int64)
tetnodes=np.full((int(max(gtet.max(),pglobal.max()))+1,4),-1,np.int64);tetnodes[gtet]=tnodes
assert np.all(tetnodes[pglobal]>=0)
pverts=xyz[tetnodes[pglobal]]
assert np.all(pmat==16)
src_points=np.einsum("ij,ijk->ik",fb,pverts[fp])
assert np.max(np.abs(src_points-fpoint),initial=0)<2e-12
with np.load(TARGET,allow_pickle=False) as z:
 points=z["points_world_m"].astype(np.float32,copy=False)
 tets=z["tetrahedra"].astype(np.int64,copy=False)
 mat=z["material_indices"].astype(np.int32,copy=False)
assert len(tets)==len(m["child_current_mass_kg"]) and np.all(mat==16)
cent=np.mean(points[tets].astype(np.float64),axis=1)
dist=np.linalg.norm(src_points-cent[fc],axis=1)
assert np.max(np.abs(dist-fd_saved),initial=0)<5e-12 and np.all(dist<=RADIUS)
with np.load(SUP,allow_pickle=False) as z:
 off=z["cell_admissible_region_offsets"].astype(np.int64);regs=z["cell_admissible_region_ids"].astype(np.int32)
region_keys=np.repeat(np.arange(len(tets),dtype=np.int64),np.diff(off))*65536+np.concatenate([regs[off[i]:off[i+1]] for i in range(len(tets))]).astype(np.int64)
region_keys=np.unique(region_keys)
edge_keys=fc*65536+preg[fp];ix=np.searchsorted(region_keys,edge_keys)
assert np.all(ix<len(region_keys)) and np.all(region_keys[np.minimum(ix,len(region_keys)-1)]==edge_keys)
assert json.loads(SUPIV.read_text())["status"]=="PASS_INDEPENDENT_AUTHENTICATED_CONNECTED_SUPPORT"
# Reconstruct source-corner allocations and residual closure over all source parents.
corner=np.column_stack([np.bincount(fp,weights=fm*fb[:,k],minlength=nparent) for k in range(4)])
qafter=m["parent_corner_residual_after_kg"].astype(np.float64)
target=pmass_csv[:,None]*.25
m4err=corner+qafter-target
m4tol=M4_REL*np.maximum(target,1e-300)
m4ratio=np.abs(m4err)/m4tol
assert np.max(m4ratio,initial=0)<=1. and np.all(qafter>=-m4tol)
assert abs((fm.sum()+qafter.sum())-pmass_csv.sum())<=1e-12
# Rebuild child masses independently from all merged rows.
childmass=np.bincount(fc,weights=fm,minlength=len(tets))
stored=m["child_current_mass_kg"].astype(np.float64)
replaytol=5e-10*(np.fromiter((1.0 for _ in tets),dtype=np.float64,count=len(tets))) # replaced below by exact-volume-scaled tolerance
assert len(childmass)==len(stored)
# Exact dyadic cell volumes, signs and full-domain direct density tests.
spec=importlib.util.spec_from_file_location("det081_independent_058",EXACT)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
used=np.unique(tets.ravel());dy={int(i):mod.ip(points[int(i)]) for i in used}
det6=[int(mod.det(*(dy[int(v)] for v in tet))) for tet in tets]
assert all(d>0 for d in det6)
vol=np.fromiter((float(d)/(6.*2.**447) for d in det6),dtype=np.float64,count=len(det6))
massdiff=np.abs(childmass-stored)
mass_tol=5e-10*vol+128*np.finfo(float).eps*np.maximum(np.abs(stored),950.*vol)
assert np.all(massdiff<=mass_tol)
threshold=Fraction.from_float(HIGH+5e-10)
bad=[];maxrho=0.;maxrow=-1
for i,(d,mass) in enumerate(zip(det6,childmass)):
 if Fraction.from_float(float(mass))*6*(1<<447)>threshold*d:bad.append(i)
 rho=float(mass)/(float(d)/(6.*2.**447))
 if rho>maxrho:maxrho,maxrow=rho,i
assert not bad,("exact density violations",bad[:20],len(bad))
with np.load(BASELEDGER,allow_pickle=False) as z: oldchild=z["child_current_mass_kg"].astype(np.float64)
base_bad=[]
for i,(d,mass) in enumerate(zip(det6,oldchild)):
 if Fraction.from_float(float(mass))*6*(1<<447)>threshold*d:base_bad.append(i)
expected=[198545,233459,233636,288320,288360,539645,752551,804459]
assert base_bad==expected
# Selected LP output stayed in the conservative quarter of the original M/4 tolerance.
selected_target=np.asarray(q["selected_parent_target_corner_mass_kg"],dtype=np.float64)
inner_tol=INNER_M4_REL*np.maximum(selected_target,1e-300)
selected_err=q["selected_parent_corner_mass_kg"]+q["selected_parent_remainder_corner_mass_kg"]-selected_target
assert np.all(np.abs(selected_err)<=inner_tol)
target_rows=np.asarray(q["target_children"],np.int64);slack=np.asarray(q["target_floor_slack_kg"],np.float64)
assert np.all(slack>=0) and np.all(slack<=np.maximum(850.*vol[target_rows]-oldchild[target_rows],0.)+1e-14)
assert np.all(childmass[target_rows]+slack>=850.*vol[target_rows]-1e-14)
local=np.asarray(q["local_receiver_rows"],np.int64)
assert np.max(np.abs(childmass[local]-q["local_final_mass_kg"]),initial=0)<=mass_tol[local].max(initial=0)
assert np.all(childmass[local]<=950.*vol[local]+mass_tol[local])
assert np.all(childmass[local]>=np.minimum(oldchild[local],850.*vol[local])-mass_tol[local])
# Unselected child ledger is byte-preserved at the stored per-child mass level.
outside=np.ones(len(tets),bool);outside[local]=False
assert np.array_equal(stored[outside],oldchild[outside])
under_before=float(np.maximum(850.*vol-oldchild,0).sum())
under_after=float(np.maximum(850.*vol-childmass,0).sum())
result={"schema":"numi.human.current406-exact-volume-exchange-independent-replay.v1","status":"PASS_INDEPENDENT_FULL_LEDGER_REPLAY_PARTIAL_EXCHANGE","physical_admission":False,"native_steps":0,"complete_allocation":False,"inputs":{str(p):sha(p) for p in [MERGE,PROPOSAL,REPORT,BASELEDGER,TARGET,PARENT,SUP,SUPIV,EXACT,SRC/"compiled-parent-cell-mass.csv.nodes.csv",SRC/"compiled-parent-cell-mass.csv.tets.csv"]},"counts":{"parents":nparent,"child_cells":len(tets),"merged_flows":len(fp),"retained_old_flows":int(retained.sum()),"new_flows":int(new.sum()),"baseline_exact_overage_rows":len(base_bad),"candidate_exact_overage_rows":len(bad),"selected_parents":len(selected),"local_receivers":len(local)},"mass_kg":{"source_parent_total":float(pmass_csv.sum()),"merged_flow_total":float(fm.sum()),"unallocated_source_residual":float(qafter.sum()),"flow_plus_unallocated_mass_error":float(fm.sum()+qafter.sum()-pmass_csv.sum()),"global_floor_deficit_before":under_before,"global_floor_deficit_after":under_after,"target_slack_before":float(np.maximum(850.*vol[target_rows]-oldchild[target_rows],0).sum()),"target_slack_after":float(slack.sum())},"residuals":{"max_M4_abs_error_kg":float(np.max(np.abs(m4err),initial=0)),"max_M4_error_over_original_5e-13_tolerance":float(np.max(m4ratio,initial=0)),"max_selected_M4_error_over_conservative_quarter_tolerance":float(np.max(np.abs(selected_err)/inner_tol,initial=0)),"max_child_mass_reconstruction_error_kg":float(np.max(massdiff,initial=0)),"max_source_point_reconstruction_error_m":float(np.max(np.abs(src_points-fpoint),initial=0)),"max_radius_m":float(dist.max(initial=0)),"max_radius_replay_difference_m":float(np.max(np.abs(dist-fd_saved),initial=0))},"density":{"method":"exact 081 dyadic F32 determinant; Fraction comparison against 950+5e-10 kg/m3","baseline_overage_rows":base_bad,"candidate_overage_rows":bad,"max_candidate_density_kg_m3":maxrho,"max_density_child_row":maxrow},"checks":{"all_flow_rows_exactly_mapped_to_source330":True,"all_unselected_held_rows_preserved_bytewise":True,"all_new_rows_equal_proposal":True,"all_flow_points_reconstructed_from_parent_barycentrics":True,"all_flow_radii_within_19_353mm":True,"all_flow_regions_allowed_by_authenticated_143":True,"all_parent_corner_mass_plus_residual_within_original_M4_tolerance":True,"selected_exchange_within_quarter_M4_envelope":True,"all_child_masses_reconstructed":True,"all_candidate_cells_within_exact_density_bound":True,"eight_baseline_overages_removed":True,"full_mass_allocation":False},"elapsed_seconds":time.monotonic()-started}
out=OUT/"independent-replay.json"
out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
print(json.dumps(result,sort_keys=True))
