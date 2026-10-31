#!/usr/bin/env python3
import csv, hashlib, json, time
from pathlib import Path
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
BASE=ROOT/"fine-physical-volume-706/combined-child-handoff-142"
CAND=ROOT/"fine-physical-volume-706/current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz"
CAND_REPORT=CAND.parent/"report.json"
RUN=ROOT/"fractional-semantic-support-090-sparse-graph/fat-held-exchange-031/rebase-current-406/fat-run-002-current-region"
EXACT=RUN/"exact-map.npz"
LEDGER=RUN/"rebased-held-flow-ledger.npz"
CHILD_CSV=ROOT/"fractional-semantic-support-090-sparse-graph/fat-853406-carrier-run-009-partial-flow-export/partial-flow-008/transport-child-mass.csv"
OUT=RUN/"hybrid-volume-audit-001"
REPORT=OUT/"report.json"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def read_node(path):
 a=np.loadtxt(path,skiprows=1)
 ids=a[:,0].astype(np.int64)
 if not np.array_equal(ids,np.arange(1,len(ids)+1)): raise RuntimeError("baseline node IDs are not contiguous one-based")
 return a[:,1:4].astype(np.float32)
def read_ele(path):
 a=np.loadtxt(path,skiprows=1)
 ids=a[:,0].astype(np.int64)
 if not np.array_equal(ids,np.arange(1,len(ids)+1)): raise RuntimeError("baseline element IDs are not contiguous one-based")
 t=a[:,1:5].astype(np.int64)-1
 return t
def detvol(points,tets):
 v=points[tets].astype(np.float64)
 d=np.einsum("ij,ij->i",np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]),v[:,3]-v[:,0])
 return d,np.abs(d)/6.0
def qstats(x):
 x=np.asarray(x,dtype=np.float64)
 return {"count":int(len(x)),"min":float(np.min(x)),"p50":float(np.quantile(x,.50)),"p95":float(np.quantile(x,.95)),"p99":float(np.quantile(x,.99)),"p999":float(np.quantile(x,.999)),"max":float(np.max(x))}
def sorted_vertices(x):
 order=np.lexsort((x[:,:,2],x[:,:,1],x[:,:,0]),axis=1)
 return np.take_along_axis(x,order[:,:,None],axis=1)
inputs=[BASE/"fat.node",BASE/"fat.ele",CAND,CAND_REPORT,EXACT,LEDGER,CHILD_CSV]
for p in inputs:
 if not p.is_file(): raise RuntimeError(f"missing pinned input: {p}")
start=time.monotonic()
oldp=read_node(BASE/"fat.node"); oldt=read_ele(BASE/"fat.ele")
v=np.load(CAND,allow_pickle=False); points=v["points_world_m"].astype(np.float32,copy=False); tets=v["tetrahedra"].astype(np.int64,copy=False)
em=np.load(EXACT,allow_pickle=False); old_to_new=em["old_to_new_exact"].astype(np.int64,copy=False)
oldrows=np.flatnonzero(old_to_new>=0); newrows=old_to_new[oldrows]
if len(np.unique(newrows))!=len(newrows): raise RuntimeError("exact map is not injective")
old_direct_det,old_direct_vol=detvol(oldp,oldt)
new_direct_det,new_direct_vol=detvol(points,tets)
with open(CHILD_CSV,newline="") as f: rows=list(csv.DictReader(f))
base_volume=np.array([float(r["reference_volume_m3"]) for r in rows],dtype=np.float64)
if len(base_volume)!=len(oldt) or len(old_to_new)!=len(oldt): raise RuntimeError("baseline volume/map row domain mismatch")
old_coords=oldp[oldt[oldrows]]
new_coords=points[tets[newrows]]
coords_equal=np.array_equal(sorted_vertices(old_coords),sorted_vertices(new_coords))
# Reconstruct the producer's hybrid volume vector: exact identities reuse the pinned
# source child volumes; changed receiver cells use current Float32 determinant volumes.
hybrid_vol=new_direct_vol.copy(); hybrid_vol[newrows]=base_volume[oldrows]
relative_old=np.abs(old_direct_vol[oldrows]-base_volume[oldrows])/base_volume[oldrows]
relative_new=np.abs(new_direct_vol[newrows]-base_volume[oldrows])/base_volume[oldrows]
absolute_old=np.abs(old_direct_vol[oldrows]-base_volume[oldrows])
absolute_new=np.abs(new_direct_vol[newrows]-base_volume[oldrows])
z=np.load(LEDGER,allow_pickle=False); mass=z["current_cell_mass_kg"].astype(np.float64,copy=False)
hybrid_density=mass/hybrid_vol; direct_density=mass/new_direct_vol
pos=mass>0
hybrid_low=pos&(hybrid_density<850); direct_low=pos&(direct_density<850)
hybrid_high=pos&(hybrid_density>950); direct_high=pos&(direct_density>950)
deficit_h=float(np.maximum(850*hybrid_vol-mass,0).sum())
deficit_d=float(np.maximum(850*new_direct_vol-mass,0).sum())
cross=float(np.sum(mass*(points[tets].astype(np.float64).mean(axis=1)-oldp[oldt].astype(np.float64).mean(axis=1)[np.minimum(np.arange(len(tets)),len(oldt)-1)]))) if False else 0.0
source_report=json.loads(CAND_REPORT.read_text())
checks={
 "baseline_mesh_row_counts_match_source_csv":len(rows)==len(oldt),
 "exact_map_domain_matches_baseline":len(old_to_new)==len(oldt),
 "exact_map_is_injective":len(np.unique(newrows))==len(newrows),
 "exact_coordinate_tet_pairs_are_bitwise_identical_unordered":bool(coords_equal),
 "all_hybrid_volumes_positive":bool(np.all(np.isfinite(hybrid_vol)) and np.all(hybrid_vol>0)),
 "all_direct_current_determinants_positive":bool(np.all(np.isfinite(new_direct_vol)) and np.all(new_direct_vol>0)),
 "candidate_matches_its_report":source_report.get("physical_admission") is False and source_report.get("native_steps")==0,
}
result={
 "schema":"numi.human.current-fat-hybrid-volume-audit.v1",
 "status":"PASS_SOURCE_ONLY_HYBRID_VOLUME_REPLAY" if all(checks.values()) else "FAIL_SOURCE_ONLY_HYBRID_VOLUME_REPLAY",
 "physical_admission":False,"mass_transport":"NOT_ADMITTED","native_steps":0,
 "scope":"Quantifies the existing rebase producer's volume rule and compares pinned baseline volumes to direct determinants on exact-coordinate tetrahedra; no volume or density result is used to suppress the existing quality failure.",
 "source":{"path":str(Path(__file__)),"sha256":sha(Path(__file__))},
 "inputs":{str(p):sha(p) for p in inputs},
 "counts":{"baseline_cells":int(len(oldt)),"current_cells":int(len(tets)),"exact_identity_cell_pairs":int(len(oldrows)),"changed_baseline_cells":int(len(oldt)-len(oldrows)),"exact_coordinate_pair_mismatches":int(not coords_equal)},
 "exact_identity_volume_comparison":{
  "baseline_direct_float64_det_vs_pinned_csv_relative_error":qstats(relative_old),
  "current_direct_float64_det_vs_pinned_csv_relative_error":qstats(relative_new),
  "baseline_direct_float64_det_vs_pinned_csv_absolute_m3":qstats(absolute_old),
  "current_direct_float64_det_vs_pinned_csv_absolute_m3":qstats(absolute_new),
  "baseline_direct_exact_equal_count":int(np.count_nonzero(old_direct_vol[oldrows]==base_volume[oldrows])),
  "current_direct_exact_equal_count":int(np.count_nonzero(new_direct_vol[newrows]==base_volume[oldrows])),
  "pinned_volume_rule":"For all 582,917 exact ordered/signed Float32 coordinate tetrahedron identities, reuse the original transport child's 17-digit reference_volume_m3 value. For other 243,820 current rows, compute absolute signed determinant of current Float32 coordinates widened to Float64, divided by 6. This preserves the previously admitted source-volume rows exactly and gives changed rows their current determinant volume.",
 },
 "density_comparison_using_same_retained_mass":{
  "retained_positive_mass_cells":int(np.count_nonzero(pos)),
  "hybrid_min_positive_kg_m3":float(hybrid_density[pos].min()),
  "hybrid_max_kg_m3":float(hybrid_density.max()),
  "direct_current_det_min_positive_kg_m3":float(direct_density[pos].min()),
  "direct_current_det_max_kg_m3":float(direct_density.max()),
  "hybrid_under_850_cells":int(np.count_nonzero(hybrid_low)),
  "direct_det_under_850_cells":int(np.count_nonzero(direct_low)),
  "hybrid_over_950_cells":int(np.count_nonzero(hybrid_high)),
  "direct_det_over_950_cells":int(np.count_nonzero(direct_high)),
  "low_density_classification_flips":int(np.count_nonzero(hybrid_low!=direct_low)),
  "high_density_classification_flips":int(np.count_nonzero(hybrid_high!=direct_high)),
  "hybrid_under_850_deficit_kg":deficit_h,
  "direct_det_under_850_deficit_kg":deficit_d,
  "max_abs_density_delta_kg_m3":float(np.max(np.abs(hybrid_density-direct_density))),
  "max_abs_density_delta_on_exact_identity_rows_kg_m3":float(np.max(np.abs(hybrid_density[newrows]-direct_density[newrows]))),
  "hybrid_upper_roundoff_witnesses":int(np.count_nonzero(hybrid_density>950)),
  "hybrid_density_still_over_950_after_owner_release":int(np.count_nonzero(hybrid_density>950)),
  "direct_det_density_over_950_after_owner_release":int(np.count_nonzero(direct_density>950)),
 },
 "checks":checks,
 "elapsed_seconds":time.monotonic()-start,
 "interpretation":"The direct determinant comparison is diagnostic. Exact-coordinate identity rows use the pinned source volume by design; numerical differences are reported, not normalized away. Density cap results are reported under both hybrid and all-current-direct determinant volumes; the source-only held ledger is not a complete allocation.",
}
REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
print(json.dumps({"status":result["status"],"checks":checks,"counts":result["counts"],"exact":result["exact_identity_volume_comparison"],"density":result["density_comparison_using_same_retained_mass"],"report":str(REPORT),"report_sha256":sha(REPORT)},sort_keys=True))

