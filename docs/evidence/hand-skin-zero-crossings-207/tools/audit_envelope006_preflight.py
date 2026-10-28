from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,time
import numpy as np
from scipy.spatial import cKDTree
BASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001/inputs/baseline-806.npz")
BASE_AUDIT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001/baseline/audit-159/report.json")
CAND=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/envelope-006-local-lattice/candidate.npz")
OUT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/boolean-prototype-001/audit-envelope-006-preflight.json")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def edge_incidence(f):
 e=np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1)
 _,c=np.unique(e,axis=0,return_counts=True)
 return int(np.count_nonzero(c==1)),int(np.count_nonzero(c>2)),int(np.count_nonzero(c==2))
def edge_counter(p,e):
 out=Counter();bits=np.asarray(p,dtype=np.float32).view(np.uint32).reshape(-1,3)
 for a,b in e:
  aa=tuple(int(q) for q in bits[int(a)]);bb=tuple(int(q) for q in bits[int(b)])
  out[tuple(sorted((aa,bb)))]+=1
 return out
def tri_key(p,tri):
 return tuple(sorted(tuple(int(q) for q in row) for row in np.asarray(p,dtype=np.float32)[np.asarray(tri,dtype=np.int64)].view(np.uint32)))
def main():
 t=time.monotonic()
 audit=json.loads(BASE_AUDIT.read_text())
 with np.load(BASE,allow_pickle=False) as b,np.load(CAND,allow_pickle=False) as c:
  bp=np.asarray(b["points_world_m"],np.float32);bf=np.asarray(b["source_O_faces"],np.int64)
  p=np.asarray(c["points_world_m"],np.float32);f=np.asarray(c["source_O_faces"],np.int64)
  orig=np.asarray(c["envelope_source_original_id"],np.int64);fid=np.asarray(c["envelope_source_face_row"],np.int64)
  base_id=int(np.asarray(c["envelope_baseline_original_id"]).reshape(-1)[0])
  if orig.shape!=(len(f),) or fid.shape!=(len(f),):raise ValueError("provenance mismatch")
  # Bitwise unchanged source triangle rows, keyed by exact original source row plus F32 coordinate bits.
  exact_rows=set()
  for j in np.flatnonzero((orig==base_id)&(fid>=0)&(fid<len(bf))):
   sr=int(fid[j])
   if tri_key(p,f[j])==tri_key(bp,bf[sr]):exact_rows.add(sr)
  changed_rows=set(range(len(bf)))-exact_rows
  # Conservative distance bounds from each original source triangle to all baseline crossing skin faces.
  crossing_rows=set()
  crossing_pair_count=0
  for bank in audit["banks"]:
   result=bank.get("result")
   if not result:continue
   for a,_ in result.get("triangle_pairs",[]):
    crossing_rows.add(int(a));crossing_pair_count+=1
  wrows=np.asarray(sorted(crossing_rows),np.int64)
  if np.any(wrows<0)|np.any(wrows>=len(bf)):raise ValueError("baseline audit references invalid skin face")
  tri=bp[bf].astype(np.float64)
  ctr=tri.mean(axis=1)
  rad=np.sqrt(((tri-ctr[:,None,:])**2).sum(axis=2)).max(axis=1)
  wctr=ctr[wrows];wrad=rad[wrows]
  tree=cKDTree(wctr)
  d,ix=tree.query(ctr,k=1)
  # The witness centroid is a point in its triangle. These bounds use enclosing balls.
  upper=d+rad
  lower=np.maximum(0.,d-rad-float(wrad.max()))
  definitely_outside=set(np.flatnonzero(lower>0.05).tolist())
  definitely_inside=set(np.flatnonzero(upper<=0.05).tolist())
  missing_outside=sorted(definitely_outside-changed_rows)
  changed_outside=sorted(definitely_outside&changed_rows)
  changed_inside=sorted(definitely_inside&changed_rows)
  # Boundary geometry comparison by exact coordinate bits.
  src_e=np.sort(np.concatenate((bf[:,[0,1]],bf[:,[1,2]],bf[:,[2,0]])),axis=1)
  out_e=np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1)
  src_open=src_e[np.unique(src_e,axis=0,return_counts=True)[1]==1] if False else None
  def boundary_rows(e):
   ue,inv,cnt=np.unique(e,axis=0,return_inverse=True,return_counts=True)
   return ue[cnt==1]
  sb=boundary_rows(src_e);ob=boundary_rows(out_e)
  miss=edge_counter(bp,sb)-edge_counter(p,ob);extra=edge_counter(p,ob)-edge_counter(bp,sb)
  # Counts for packed-topology degeneracies.
  opens,nm,two=edge_incidence(f)
  cross=np.cross(p[f][:,1].astype(np.float64)-p[f][:,0],p[f][:,2].astype(np.float64)-p[f][:,0])
  sf=np.sort(f,axis=1);_,dupc=np.unique(sf,axis=0,return_counts=True)
  report={
   "candidate_path":str(CAND),"candidate_sha256":sha(CAND),"baseline_sha256":sha(BASE),
   "candidate_points":int(len(p)),"candidate_faces":int(len(f)),
   "candidate_nonmanifold_edges":nm,"candidate_open_edges":opens,
   "baseline_open_edges":int(len(sb)),"boundary_edges_exact_coordinate_match":not miss and not extra,
   "missing_boundary_edges":int(sum(miss.values())),"extra_boundary_edges":int(sum(extra.values())),
   "repeated_index_faces":int(np.count_nonzero((f[:,0]==f[:,1])|(f[:,1]==f[:,2])|(f[:,2]==f[:,0]))),
   "exact_zero_cross_faces":int(np.count_nonzero(np.all(cross==0.,axis=1))),
   "duplicate_face_groups":int(np.count_nonzero(dupc>1)),
   "baseline_source_face_rows":int(len(bf)),"bitwise_unchanged_source_face_rows":int(len(exact_rows)),
   "retriangulated_or_removed_source_face_rows":int(len(changed_rows)),
   "crossing_pair_occurrences_in_baseline_audit":crossing_pair_count,"unique_baseline_crossing_skin_face_rows":int(len(wrows)),
   "distance_method":"source-face centroids/radii against baseline crossing skin-face centroids; lower bound = center distance - source radius - maximum witness radius; upper bound = center distance + source radius; all distances in metres",
   "outside_50mm_definitely_source_rows":int(len(definitely_outside)),
   "outside_50mm_rows_not_bitwise_preserved":int(len(changed_outside)),
   "outside_50mm_rows_not_bitwise_preserved_first32":changed_outside[:32],
   "within_50mm_definitely_changed_source_rows":int(len(changed_inside)),
   "farthest_changed_source_row_upper_bound_m":float(upper[list(changed_rows)].max()) if changed_rows else 0.,
   "max_definite_outside_lower_bound_m":float(lower.max()),
   "geometry_or_native_admission":False,"elapsed_seconds":time.monotonic()-t,
  }
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()
