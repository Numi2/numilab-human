#!/usr/bin/env python3
import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
RUN=ROOT/"fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-002-current-region"
CAND=ROOT/"fine-physical-volume-706/current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
SUPPORT=ROOT/"fine-physical-volume-706/connected-region-mass-support-143-current406-root/fat-connected-region-support.npz"
SUPPORT_REPORT=SUPPORT.parent/"independent-verification.json"
PARENT_CSV=ROOT/"fractional-semantic-support-090-sparse-graph/fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-source-parent.csv"
CHILD_CSV=ROOT/"fractional-semantic-support-090-sparse-graph/fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-child-mass.csv"
EXACT_MAP=RUN/"exact-map.npz"; ROW_MAP=RUN/"cell-row-map.npz"
REPORT=RUN/"ledger-report.json"; LEDGER=RUN/"rebased-held-flow-ledger.npz"; OUT=RUN/"independent-ledger-replay-v2.json"
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def relcheck(a,b,tol):
    a=np.asarray(a); b=np.asarray(b)
    if a.shape!=b.shape:return False,float("inf")
    den=np.maximum(np.abs(a),np.abs(b)); d=np.abs(a-b)
    ok=np.all((d==0)|((den>0)&(d<=tol*den)))
    return bool(ok),float(np.max(np.divide(d,den,out=np.zeros_like(d,dtype=np.float64),where=den>0),initial=0))
z=np.load(LEDGER,allow_pickle=False); v=np.load(CAND,allow_pickle=False); s=np.load(SUPPORT,allow_pickle=False)
em=np.load(EXACT_MAP,allow_pickle=False); rmmap=np.load(ROW_MAP,allow_pickle=False)
report=json.loads(REPORT.read_text()); sr=json.loads(SUPPORT_REPORT.read_text())
with open(PARENT_CSV,newline="") as f: parents=list(csv.DictReader(f))
with open(CHILD_CSV,newline="") as f: children=list(csv.DictReader(f))
nparent=int(report["held_ledger_rebase"]["source_parent_count"]); tet=v["tetrahedra"].astype(np.int64,copy=False); ncell=len(tet)
fp=z["flow_parent_role_rows"].astype(np.int64,copy=False); fg=z["flow_global_source330_rows"].astype(np.int64,copy=False)
fc=z["flow_current_child_rows"].astype(np.int64,copy=False); fm=z["flow_mass_kg"].astype(np.float64,copy=False)
fb=z["flow_source_barycentric"].astype(np.float64,copy=False); ret=z["flow_retained"].astype(bool); fd=z["flow_current_centroid_distance_m"].astype(np.float64,copy=False)
assert len(parents)==nparent and [int(x["parent_row"]) for x in parents]==list(range(nparent))
assert len(children)==int(report["held_ledger_rebase"]["baseline_child_count"])
pg=np.array([int(x["global_source330_row"]) for x in parents],dtype=np.int64)
pmat=np.array([int(x["material"]) for x in parents],dtype=np.int64); preg=np.array([int(x["region"]) for x in parents],dtype=np.int64)
pmasscsv=np.array([float(x["mass_kg"]) for x in parents]); pmass=z["parent_source_mass_kg"].astype(np.float64,copy=False)
parent_map_ok=bool(np.array_equal(pg,np.arange(14316,28632,dtype=np.int64)) and np.all(pmat==16))
def_parent_mass_ok,parent_mass_rel=relcheck(pmass,pmasscsv,2e-15)

# Recreate precisely the producer's hybrid volume vector: current signed Float32
# determinants except exact cell identities, which reuse the pinned baseline CSV volume.
pts=v["points_world_m"].astype(np.float64,copy=False); p0=pts[tet[:,0]]
a=pts[tet[:,1]]-p0; b=pts[tet[:,2]]-p0; c=pts[tet[:,3]]-p0
newdet=np.einsum("ij,ij->i",np.cross(a,b),c)
current_vol=np.abs(newdet)/6.0
old_to_new=em["old_to_new_exact"].astype(np.int64,copy=False)
base_vol=np.array([float(x["reference_volume_m3"]) for x in children],dtype=np.float64)
oldrows=np.flatnonzero(old_to_new>=0); newrows=old_to_new[oldrows]
unique_exact=(len(np.unique(newrows))==len(newrows) and np.all((newrows>=0)&(newrows<ncell)))
if unique_exact: current_vol[newrows]=base_vol[oldrows]
exact_identity_count=int(len(newrows))
rowmap_identity=np.array_equal(rmmap["baseline_cell_row_to_current_exact_row"],old_to_new)

valididx=np.flatnonzero(ret); rp=fp[ret]; rc=fc[ret]; flowmass=fm[ret]; rg=fg[ret]; rd=fd[ret]; rb=fb[ret]
volmat=v["material_indices"].astype(np.int64,copy=False); valid_target=(rc>=0)&(rc<ncell); safe_rc=np.clip(rc,0,ncell-1)
material_ok=bool(np.all(valid_target) and np.all(volmat[safe_rc]==pmat[rp]))
R=float(report["held_ledger_rebase"]["distance_hard_radius_m"])
radius_ok=bool(np.all(np.isfinite(rd)) and np.all(rd>=0) and np.all(rd<=R))
flow_ok=bool(np.all(np.isfinite(flowmass)) and np.all(flowmass>0) and np.all(np.isfinite(rb)) and np.all(rb>=0))
baryerr=float(np.max(np.abs(rb.sum(axis=1)-1),initial=0)); baryok=baryerr<=2e-12
offs=s["cell_admissible_region_offsets"].astype(np.int64,copy=False); ids=s["cell_admissible_region_ids"].astype(np.int64,copy=False)
base=int(max(np.max(ids,initial=0),np.max(preg,initial=0)))+1
supportrows=np.repeat(np.arange(ncell,dtype=np.int64),np.diff(offs)); supportkeys=supportrows*base+ids
query=safe_rc*base+preg[rp]; pos=np.searchsorted(supportkeys,query); hit=pos<len(supportkeys)
hit[hit] &= supportkeys[pos[hit]]==query[hit]
regionok=bool(np.all(supportkeys[1:]>supportkeys[:-1]) and np.all(valid_target) and np.all(hit))
childmass=np.bincount(rc,weights=flowmass,minlength=ncell); ledgermass=z["current_cell_mass_kg"].astype(np.float64,copy=False)
cmassok,cmassrel=relcheck(childmass,ledgermass,5e-13)
corner=np.zeros((nparent,4),dtype=np.float64)
np.add.at(corner,(rp[:,None],np.arange(4)[None,:]),flowmass[:,None]*rb)
cornerok,cornerrel=relcheck(corner,z["parent_corner_mass_after_kg"],5e-13)
finitepos=bool(np.all(np.isfinite(current_vol)) and np.all(current_vol>0))
density=np.divide(childmass,current_vol,out=np.zeros_like(childmass),where=current_vol>0)
ledgerdensity=z["current_cell_density_kg_m3"].astype(np.float64,copy=False)
densmatch,densrel=relcheck(density,ledgerdensity,3e-12); capok=bool(np.all(density<=950.0000000001))
under=int(np.count_nonzero((childmass>0)&(density<850)))
deficit=float(np.maximum(850*current_vol-childmass,0).sum())
checks={
"candidate_sha_matches_report":report["candidate_geometry"]["sha256"]==sha(CAND),
"support_sha_matches_report":report["canonical_region_admission"]["support_sha256"]==sha(SUPPORT),
"support_independent_status_pass":sr.get("status")=="PASS_INDEPENDENT_AUTHENTICATED_CONNECTED_SUPPORT",
"support_bound_to_candidate":str(s["volume_sha256"].item())==sha(CAND),
"source_parent_identity_material_map":parent_map_ok,"source_parent_mass_matches_csv":def_parent_mass_ok,
"exact_row_maps_agree":rowmap_identity,"exact_volume_identity_rows_are_unique":unique_exact,
"retained_global_source_map":bool(np.array_equal(rg,pg[rp])),"positive_finite_flow_and_barycentrics":flow_ok,
"barycentric_sum":baryok,"hard_radius":radius_ok,"material_matches_current_cell":material_ok,
"source_region_allowed_by_current_support":regionok,"child_mass_reaggregation":cmassok,
"parent_corner_reaggregation":cornerok,"current_hybrid_volumes_positive":finitepos,
"density_matches_owner_hybrid_volume":densmatch,"density_under_950":capok}
passed=all(checks.values())
result={
"schema":"numi.human.current-volume-held-ledger-independent-replay.v2",
"status":"PASS_INDEPENDENT_CURRENT_HELD_LEDGER_REPLAY_SOURCE_ONLY" if passed else "FAIL_INDEPENDENT_CURRENT_HELD_LEDGER_REPLAY",
"physical_admission":False,"mass_transport":"NOT_ADMITTED","native_steps":0,
"scope":"Independent replay of retained historical Fat held-flow rows against current 406 geometry and independently verified current 143 semantic-region support; not a complete allocation.",
"inputs":{k:{"path":str(p),"sha256":sha(p)} for k,p in [("ledger",LEDGER),("producer_report",REPORT),("candidate",CAND),("current_region_support",SUPPORT),("current_support_independent_report",SUPPORT_REPORT),("source_parent_csv",PARENT_CSV),("baseline_child_volume_csv",CHILD_CSV),("exact_row_map",EXACT_MAP),("cell_row_map",ROW_MAP)]},
"counts":{"current_cells":int(ncell),"current_points":int(len(pts)),"source_parents":nparent,"historical_held_rows":int(len(ret)),"retained_rows":int(ret.sum()),"retained_current_cells":int(np.count_nonzero(childmass)),"exact_identity_rows_reusing_pinned_baseline_volumes":exact_identity_count},
"mass":{"source_parent_total_kg":float(pmass.sum()),"retained_kg":float(flowmass.sum()),"released_kg":float(fm[~ret].sum()),"unallocated_source_kg":float(pmass.sum()-flowmass.sum()),"minimum_positive_density_kg_m3":float(density[childmass>0].min()),"maximum_density_kg_m3":float(density.max()),"positive_mass_cells_below_850":under,"under_850_deficit_kg":deficit,"cells_over_950":int(np.count_nonzero(density>950.0))},
"checks":checks,
"measurements":{"max_barycentric_sum_error":baryerr,"max_relative_source_parent_mass_error":parent_mass_rel,"max_relative_child_mass_reaggregation_error":cmassrel,"max_relative_parent_corner_reaggregation_error":cornerrel,"max_relative_density_recomputation_error":densrel,"max_retained_distance_m":float(rd.max(initial=0)),"radius_limit_m":R,"minimum_hybrid_current_volume_m3":float(current_vol.min()),"retained_rows_allowed_by_current_region_support":int(hit.sum())},
"producer_releases_by_reason":report["held_ledger_rebase"]["release_reason_counts"],
"producer_released_mass_by_reason_kg":report["held_ledger_rebase"]["released_mass_by_reason_kg"],
"producer_max_relative_M4_residual":report["held_ledger_rebase"]["parent_corner_after_max_relative_M4_residual"],
"limitations":["Retained historical rows were not redistributed; this subset does not meet full per-parent M/4 accounting.","Low-density recipients remain underfilled; the deficit is not repaired.","Candidate 406 has physical_admission=false and a residual boundary mismatch."]}
OUT.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({"status":result["status"],"checks":checks,"counts":result["counts"],"mass":result["mass"],"measurements":result["measurements"],"output":str(OUT),"sha256":sha(OUT)},sort_keys=True))

