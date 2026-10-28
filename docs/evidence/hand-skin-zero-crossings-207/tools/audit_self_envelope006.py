from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np

BASE = Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001/inputs/baseline-806.npz")
CAND = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/envelope-006-local-lattice/candidate.npz")
OUT = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/boolean-prototype-001/audit-envelope-006-self.json")
EXPECTED = "ef51980eea84db5634a4cb61d4b9936d7038bbdf76e08394e464512ad5bea5ee"
EXACT_OWNER = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
PACKAGE_SRC = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/incremental-exact-audit-owner-253/src")

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()

def load_exact():
 if str(PACKAGE_SRC) not in sys.path:sys.path.insert(0,str(PACKAGE_SRC))
 name="numilab_human.connected_skin_bone_exact_159"
 spec=importlib.util.spec_from_file_location(name,EXACT_OWNER)
 if spec is None or spec.loader is None:raise RuntimeError("cannot load pinned exact predicate")
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
 return mod

def main():
 if sha(CAND)!=EXPECTED:raise ValueError("candidate hash mismatch")
 t0=time.monotonic()
 with np.load(CAND,allow_pickle=False) as z:
  p=np.asarray(z["points_world_m"],np.float32)
  f=np.asarray(z["source_O_faces"],np.int64)
  orig=np.asarray(z["envelope_source_original_id"],np.int64)
  fid=np.asarray(z["envelope_source_face_row"],np.int64)
  if orig.shape!=(len(f),) or fid.shape!=(len(f),):raise ValueError("provenance arrays not face aligned")
 module=load_exact()
 vertices=tuple(tuple(Fraction.from_float(float(x)) for x in point) for point in p)
 records=module._records(vertices,tuple(tuple(int(v) for v in face) for face in f))
 result=module._audit_pair(records,records,same_surface=True)
 pairs=[]
 for a,b in result["triangle_pairs"]:
  ra,rb=records[int(a)],records[int(b)]
  points=module.triangle_intersection_points(ra[0],rb[0])
  pairs.append({
   "face_rows":[int(a),int(b)],
   "face_provenance":[
    {"run_original_id":int(orig[int(a)]),"face_id":int(fid[int(a)])},
    {"run_original_id":int(orig[int(b)]),"face_id":int(fid[int(b)])},
   ],
   "intersection_points_fraction":[[str(c) for c in pt] for pt in points],
   "triangle_a_f32":[[float(x) for x in q] for q in p[f[int(a)]]],
   "triangle_b_f32":[[float(x) for x in q] for q in p[f[int(b)]]],
   "triangle_vertex_rows":[f[int(a)].astype(int).tolist(),f[int(b)].astype(int).tolist()],
  })
 report={
  "status":"COMPLETE_RATIONAL_SAME_SURFACE_AUDIT",
  "candidate_path":str(CAND),"candidate_sha256":sha(CAND),
  "expected_candidate_sha256":EXPECTED,
  "exact_predicate_owner":str(EXACT_OWNER),"exact_predicate_sha256":sha(EXACT_OWNER),
  "triangle_count":int(len(f)),
  "exact_triangle_pair_count":int(result["count"]),
  "aabb_candidate_pairs":int(result["aabb_candidate_pairs"]),
  "allowed_shared_vertex_or_edge_pairs":int(result["allowed_shared_vertex_or_edge_pairs"]),
  "triangle_pairs":pairs,
  "complete":True,"geometry_or_native_admission":False,
  "elapsed_seconds":time.monotonic()-t0,
 }
 OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
 print(json.dumps({k:v for k,v in report.items() if k!="triangle_pairs"},indent=2))
 if pairs:print(json.dumps({"pairs":pairs},indent=2))
if __name__=="__main__":main()
