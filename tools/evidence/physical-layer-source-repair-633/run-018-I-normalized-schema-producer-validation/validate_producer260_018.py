from pathlib import Path
import hashlib,importlib.util,json,sys,traceback
import numpy as np
BASE=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
RUN=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/reduced-physical-boundary-400/attempt-038-connected-minkowski-counterfactual-023mm/run-017-I-source-schema-normalization")
C=RUN/"output/candidate-I-one-sided-dp-f32-typed-schema-normalized.npz"
V=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/reduced-physical-boundary-400/attempt-038-connected-minkowski-counterfactual-023mm/run-016-I-one-sided-dp-remesh/output/I-one-sided-DP-derived-source.npz")
PROD=BASE/"bounded-parent-registration-314/export_authoring_registered_surface_sidecars_230.py"
OUT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/reduced-physical-boundary-400/attempt-038-connected-minkowski-counterfactual-023mm/run-018-I-normalized-schema-producer-validation/producer260-validation.json")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
r={"schema":"numi.I-one-sided-remesh-producer260-validation.v1","candidate_path":str(C),"candidate_sha256":sha(C),
 "source_view_path":str(V),"source_view_sha256":sha(V),"producer_path":str(PROD),"producer_sha256":sha(PROD),"status":"ERROR"}
try:
 sys.path.insert(0,str(PROD.parent))
 spec=importlib.util.spec_from_file_location("producer260_I017",PROD);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 valid=m.validate_candidate(C,"I",require_f32=True)
 with np.load(C,allow_pickle=False) as z:
  P=z["points_world_m"];F=z["faces"];orig=z["face_source_original_id"];fid=z["face_source_face_id"]
  offsets=z["face_ancestry_offsets"];cache=z["face_ancestry_input_cache_rows"];ao=z["face_ancestry_original_ids"]
  af=z["face_ancestry_face_ids"];ai0=z["face_ancestry_identity0"];ai1=z["face_ancestry_identity1"]
  opmask=orig==61000002
  schema=json.loads(str(z["face_source_identity_schema_by_original_id_json"][0]))
  if offsets.shape!=(len(F)+1,) or offsets[0]!=0 or offsets[-1]!=len(cache):raise ValueError("candidate ancestry CSR offsets invalid")
  if not (len(cache)==len(ao)==len(af)==len(ai0)==len(ai1)):raise ValueError("candidate ancestry CSR columns differ")
  if int(opmask.sum())!=18 or not np.all(np.diff(offsets)[opmask]>0):raise ValueError("derived face rows lack full source ancestry")
 with np.load(V,allow_pickle=False) as z:
  VP=z["points_world_m_f64"];VF=z["faces_f64"];vid=z["face_id"];vi0=z["face_source_identity0"];vi1=z["face_source_identity1"]
  vo=z["face_ancestry_offsets"];vc=z["face_ancestry_input_cache_rows"];vao=z["face_ancestry_original_ids"]
  vaf=z["face_ancestry_face_ids"];vai0=z["face_ancestry_identity0"];vai1=z["face_ancestry_identity1"]
  if len(VF)!=18 or len(VP)!=24 or vo.shape!=(19,) or vo[0]!=0 or vo[-1]!=len(vc):raise ValueError("derived source view geometry/CSR mismatch")
  view_to_candidate=np.full(len(VP),-1,np.int64)
  for j,p in enumerate(VP):
   h=np.flatnonzero(np.all(P==np.asarray(p,np.float32),axis=1))
   if len(h)!=1:raise ValueError(f"source view vertex {j} does not map uniquely to candidate")
   view_to_candidate[j]=h[0]
  if not np.array_equal(F[-18:],view_to_candidate[VF]):raise ValueError("candidate derived faces differ from source view geometry")
  if not np.array_equal(vid,np.arange(18)) or not np.array_equal(fid[opmask],vid):raise ValueError("derived face identity mapping mismatch")
  for i,row in enumerate(np.flatnonzero(opmask)):
   a,b=offsets[row:row+2];c,d=vo[i:i+2]
   cols=((cache[a:b],vc[c:d]),(ao[a:b],vao[c:d]),(af[a:b],vaf[c:d]),(ai0[a:b],vai0[c:d]),(ai1[a:b],vai1[c:d]))
   if any(not np.array_equal(x,y) for x,y in cols):raise ValueError(f"candidate/source-view CSR differs at derived face {i}")
 src_ids=sorted(map(int,np.unique(ao[np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in np.flatnonzero(opmask)])])))
 r.update({"status":"PASS","producer_validation":{"layer":valid["layer"],"role":valid["role"],"vertexCount":valid["vertexCount"],
   "faceCount":valid["faceCount"],"sourceOriginalFaceIDs":valid["sourceOriginalFaceIDs"],"sourceParentLineagePreservedPerFace":valid["sourceParentLineagePreservedPerFace"],
   "derivedAncestry":valid["faceAncestry"].get("61000002")},
   "candidate_CSR":{"face_rows":len(F),"ancestry_entries":len(cache),"nonempty_ancestry_derived_faces":int(opmask.sum()),
   "candidate_and_source_view_CSR_exactly_equal":True,"derived_parent_source_ids_from_CSR":src_ids},
   "candidate_to_source_view_geometry_exact":True,"operation_schema":schema.get("61000002")})
except BaseException as e:
 r.update({"status":"ERROR","error":{"type":type(e).__name__,"message":str(e),"traceback":traceback.format_exc()}})
OUT.write_text(json.dumps(r,indent=2,default=lambda v:v.item() if hasattr(v,"item") else str(v))+"\n")
print(json.dumps(r,indent=2,default=lambda v:v.item() if hasattr(v,"item") else str(v)))
if r["status"]!="PASS":raise SystemExit(1)
