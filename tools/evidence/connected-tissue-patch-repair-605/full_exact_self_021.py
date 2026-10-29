import hashlib,importlib.util,json,sys,time
from fractions import Fraction
from pathlib import Path
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
OUT=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/closed603-merge-020"
C=OUT/"candidate-f32-linked-remesh-typed.npz"
RAW=OUT/"candidate-geometry-final-checkpoint.npz"
BATCH=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/combined-batch-019"
ORIG=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/candidate-f32.npz"
BASESELF=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/exact-native-387-001/self.json"
OPS=P/"closed-envelope-component-repair-603/closed-component-patch-operations.json"
OUTAUD=OUT/"exact-self-021"
EXACT=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py")
PKG=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/incremental-exact-audit-owner-253/src")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
if OUTAUD.exists(): raise SystemExit("output exists")
OUTAUD.mkdir()
sys.path.insert(0,str(PKG))
name="numilab_human.cardiac_cavity_intersections"
spec=importlib.util.spec_from_file_location(name,EXACT);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
with np.load(C,allow_pickle=False) as z:d={k:np.array(z[k],copy=True) for k in z.files}
P32=d["points_world_m"];F=np.asarray(d["faces"],np.int64)
if P32.dtype!=np.float32 or not np.isfinite(P32).all() or F.min()<0 or F.max()>=len(P32):raise ValueError("candidate F32 finite/index check failed")
tri=P32[F]
if np.any(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)==0):raise ValueError("zero-area F32 face before exact audit")
# Map merged output rows to original candidate007 rows for witness attribution.
with np.load(BATCH/"candidate-geometry-final-checkpoint.npz",allow_pickle=False) as z:
 batch_ret=np.array(z["retained_input_face_rows"],np.int64,copy=True)
 batch_removed=np.array(z["removed_input_face_rows"],np.int64,copy=True)
 batch_new=np.array(z["new_faces"],np.int64,copy=True)
with np.load(ORIG,allow_pickle=False) as z:origF=np.array(z["faces"],np.int64,copy=True)
base_to_batch=np.full(len(origF),-1,np.int64);base_to_batch[batch_ret]=np.arange(len(batch_ret),dtype=np.int64)
ops=json.loads(OPS.read_text())["operations"]
remove_current=np.zeros(len(F),bool)
for op in ops:
 rows=np.asarray(op["removed_face_rows"],np.int64)
 mapped=base_to_batch[rows]
 if np.any(mapped<0):raise ValueError("603 base rows absent from batch output")
 remove_current[mapped]=True
current_kept=np.flatnonzero(~remove_current)
if len(current_kept)+16!=len(F):raise ValueError("merged candidate domain mismatch")
current_to_final=np.full(len(F),-1,np.int64);current_to_final[current_kept]=np.arange(len(current_kept),dtype=np.int64)
out_to_base=np.full(len(F),-1,np.int64)
valid=base_to_batch>=0
base_rows=np.flatnonzero(valid)
out_to_base[current_to_final[base_to_batch[base_rows]]]=base_rows
nclosed=len(F)-len(current_kept)
if nclosed!=16:raise ValueError("expected16 closed-component output faces")
with np.load(BASESELF,allow_pickle=False) if False else open(BASESELF) as f:base_self=json.load(f)
base_pairs={tuple(sorted(map(int,x))) for x in base_self["triangle_pairs"]}
t=time.monotonic()
vertices=tuple(tuple(Fraction.from_float(float(v)) for v in xyz) for xyz in P32)
records=mod._records(vertices,F)
res=mod._audit_pair(records,records,same_surface=True)
pairs=np.asarray(res.get("triangle_pairs",[]),np.int64).reshape(-1,2)
if len(pairs):np.save(OUTAUD/"exact-pairs.npy",pairs)
cats={"retained_original_vs_retained_original":0,"ordinary_remesh_vs_retained_original":0,"ordinary_remesh_vs_ordinary_remesh":0,
      "closed603_vs_any":0,"baseline_original_pair_reproduced":0,"other":0}
samples=[]
ordinary_start=len(batch_ret)-16
ordinary_end=len(F)-16
for a,b in pairs:
 a=int(a);b=int(b)
 ac="closed603" if a>=len(F)-16 else ("ordinary_remesh" if a>=ordinary_start else "retained_original")
 bc="closed603" if b>=len(F)-16 else ("ordinary_remesh" if b>=ordinary_start else "retained_original")
 if ac=="closed603" or bc=="closed603":cats["closed603_vs_any"]+=1;cat="closed603_vs_any"
 elif ac==bc=="retained_original":
  cats["retained_original_vs_retained_original"]+=1
  ba,bb=sorted((int(out_to_base[a]),int(out_to_base[b])))
  if (ba,bb) in base_pairs:cats["baseline_original_pair_reproduced"]+=1
  cat="retained_original_vs_retained_original"
 elif ac==bc=="ordinary_remesh":cats["ordinary_remesh_vs_ordinary_remesh"]+=1;cat="ordinary_remesh_vs_ordinary_remesh"
 elif "ordinary_remesh" in (ac,bc):cats["ordinary_remesh_vs_retained_original"]+=1;cat="ordinary_remesh_vs_retained_original"
 else:cats["other"]+=1;cat="other"
 if len(samples)<200:
  samples.append({"pair":[a,b],"category":cat,"source_original_id":[int(d["face_source_original_id"][a]),int(d["face_source_original_id"][b])],
                  "source_face_id":[int(d["face_source_face_id"][a]),int(d["face_source_face_id"][b])],
                  "original_input_row":[int(out_to_base[a]),int(out_to_base[b])]})
report={"schema":"pair-remesh-full-exact-self.v1","candidate_path":str(C),"candidate_sha256":sha(C),
 "raw_geometry_path":str(RAW),"raw_geometry_sha256":sha(RAW),"exact_owner_path":str(EXACT),"exact_owner_sha256":sha(EXACT),
 "algorithm":mod.ALGORITHM,"points":len(P32),"faces":len(F),"full_surface":True,"exact_pair_count":int(res["count"]),
 "aabb_candidate_pairs":int(res.get("aabb_candidate_pairs",0)),"allowed_shared_vertex_or_edge_pairs":int(res.get("allowed_shared_vertex_or_edge_pairs",0)),
 "pair_archive":str(OUTAUD/"exact-pairs.npy") if len(pairs) else None,"categories":cats,"pair_sample_source_rows":samples,
 "baseline_candidate_sha256":"e8e332fb4ab421db6e44f0f5fe638e378646fd63438a318dcf94ad84b7a8ce96",
 "baseline_exact_self_report_sha256":sha(BASESELF),"batch016_result_sha256":sha(BATCH/"result.json"),
 "source603_operation_manifest_sha256":sha(OPS),"elapsed_s":time.monotonic()-t,"physical_admission":False}
(OUTAUD/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:report[k] for k in ("candidate_path","candidate_sha256","points","faces","exact_pair_count","categories","elapsed_s","physical_admission")},sort_keys=True),flush=True)
