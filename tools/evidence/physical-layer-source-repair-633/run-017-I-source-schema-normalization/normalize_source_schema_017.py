#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,traceback
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
A=P/"reduced-physical-boundary-400/attempt-038-connected-minkowski-counterfactual-023mm"
RUN=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/reduced-physical-boundary-400/attempt-038-connected-minkowski-counterfactual-023mm/run-017-I-source-schema-normalization");OUT=RUN/"output";OUT.mkdir(exist_ok=False)
OLD=A/"run-016-I-one-sided-dp-remesh/output/candidate-I-one-sided-dp-f32-typed.npz"
OLD_REPORT=A/"run-016-I-one-sided-dp-remesh/output/report.json"
SRC_I=P/"reduced-physical-boundary-400/attempt-024-coupled-paired-thickness/source002-I-closed-f32-as-f64.npz"
SRC_SUPPORT=A/"run-005-protected-cap-minthickness/output/I-added-source.npz"
PROD=P/"bounded-parent-registration-314/export_authoring_registered_surface_sidecars_230.py"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,default=lambda v:v.item() if hasattr(v,"item") else str(v))+"\n")
def geometry_digest(P,F):
 h=hashlib.sha256();h.update(np.asarray(P,np.float32).tobytes(order="C"));h.update(np.asarray(F,np.int64).tobytes(order="C"));return h.hexdigest()
report={"schema":"numi.I-source-schema-normalization.v1","status":"PREFLIGHT","physical_admission":False}
try:
 for label,p in [("candidate",OLD),("prior_exact_report",OLD_REPORT),("original_I",SRC_I),("support_primitive",SRC_SUPPORT),("producer260",PROD)]:
  if not p.is_file():raise FileNotFoundError(p)
  report.setdefault("input_pins",{})[label]={"path":str(p),"sha256":sha(p)}
 c=load(OLD);i=load(SRC_I);s=load(SRC_SUPPORT)
 schema=json.loads(str(c["face_source_identity_schema_by_original_id_json"][0]))
 beforeP=np.asarray(c["points_world_m"]);beforeF=np.asarray(c["faces"])
 sourceId=np.asarray(c["face_source_original_id"],np.int64);faceId=np.asarray(c["face_source_face_id"],np.int64)
 ident0=np.asarray(c["face_source_identity0"],np.int64);ident1=np.asarray(c["face_source_identity1"],np.int64)
 proofs={}
 i_ids=np.asarray(i["face_id"],np.int64)
 if len(np.unique(i_ids))!=len(i_ids):raise ValueError("source I face IDs are not unique")
 i_alias=OUT/"source002-I-identity-normalized-view.npz"
 i_out={k:np.asarray(v) for k,v in i.items()}
 i_out["source002_I_closed_face_id"]=i_ids.copy()
 i_out["none"]=np.full(len(i_ids),-1,np.int64)
 np.savez_compressed(i_alias,**i_out)
 r3=sourceId==3
 if not np.all(ident0[r3]==faceId[r3]) or np.any(ident1[r3]!=-1):raise ValueError("source-3 output identity pair does not match exact source face IDs")
 if not set(map(int,faceId[r3])).issubset(set(map(int,i_ids))):raise ValueError("source-3 output face IDs absent from original I primitive")
 schema["3"]["path"]=str(i_alias);schema["3"]["sha256"]=sha(i_alias)
 schema["3"]["identity_normalization"]={"exact_primitive_source_path":str(SRC_I),"exact_primitive_source_sha256":sha(SRC_I),
    "mapping":"source002_I_closed_face_id copies exact sparse face_id; none is constant -1","geometry_and_face_id_rows_copied_unchanged":True}
 proofs["source_3"]={"output_rows":int(r3.sum()),"source_rows":len(i_ids),"candidate_identity0_equals_face_id":True,
    "all_candidate_face_ids_exist_in_original_source":True,"source_geometry_arrays_exactly_copied":True}
 support_ids=np.arange(len(np.asarray(s["faces_f64"])),dtype=np.int64)
 r209=sourceId==209586
 if not np.all(ident0[r209]==faceId[r209]) or np.any(ident1[r209]!=-1):raise ValueError("209586 output identity pair does not match support face rows")
 if not set(map(int,faceId[r209])).issubset(set(support_ids.tolist())):raise ValueError("209586 output face IDs absent from support primitive")
 s_alias=OUT/"I-added-source-identity-normalized-view.npz"
 s_out={k:np.asarray(v) for k,v in s.items()}
 s_out["face_id"]=support_ids.copy()
 s_out["I_protected_support_face_row"]=support_ids.copy()
 s_out["no_unique_parent_face"]=np.full(len(support_ids),-1,np.int64)
 np.savez_compressed(s_alias,**s_out)
 parent_assets=[
  SRC_I,
  A/"run-002/output/inner-region-dilation-f64-stage-f64.npz",
  A/"run-005-protected-cap-minthickness/output/shared-cap-guard-f64.npz",
  A/"run-005-protected-cap-minthickness/output/O-minskin-eroded-f64.npz",
  A/"run-005-protected-cap-minthickness/output/inner-support-protected-f64.npz",
  A/"run-005-protected-cap-minthickness/output/inner-support-clipped-f64.npz",
  SRC_SUPPORT,
  A/"run-003-locality/build_protected_cap_minthickness_005.py",
  A/"run-005-protected-cap-minthickness/output/receipt.json"]
 for p in parent_assets:
  if not p.is_file():raise FileNotFoundError(p)
 info=schema["209586"]
 info["path"]=str(s_alias);info["sha256"]=sha(s_alias)
 info["identity_face_id_key"]="face_id"
 info["derived_parent_assets"]=[{"path":str(p),"sha256":sha(p)} for p in parent_assets]
 info["source_geometry_scope"]="exact existing protected inner support primitive; source coordinates/faces unchanged; source face row identity normalized from exact row order"
 info["source_face_ancestry_policy"]="primitive parent_run_index and parent_face_ids arrays are retained verbatim; candidate full face ancestry CSR is retained verbatim"
 info["identity_normalization"]={"primitive_path":str(SRC_SUPPORT),"primitive_sha256":sha(SRC_SUPPORT),
   "face_id_and_I_protected_support_face_row":"exact source face row index 0..N-1","no_unique_parent_face":"constant -1; no unique measured source face claimed"}
 proofs["source_209586"]={"output_rows":int(r209.sum()),"primitive_rows":len(support_ids),
    "candidate_identity0_equals_support_face_row":True,"all_candidate_face_ids_exist_in_support_primitive":True,
    "primitive_upstream_parent_arrays_retained":True,"parent_assets":len(parent_assets)}
 out=OUT/"candidate-I-one-sided-dp-f32-typed-schema-normalized.npz"
 c["face_source_identity_schema_by_original_id_json"]=np.asarray([json.dumps(schema,sort_keys=True)])
 operation=json.loads(str(c["one_sided_remesh_operation_json"][0]))
 operation["metadata_only_source_schema_normalization"]={"from_candidate_path":str(OLD),"from_candidate_sha256":sha(OLD),
    "geometry_arrays_unchanged":True,"source_3_alias_view":str(i_alias),"source_209586_alias_view":str(s_alias)}
 c["one_sided_remesh_operation_json"]=np.asarray([json.dumps(operation,sort_keys=True)])
 np.savez_compressed(out,**c)
 c2=load(out)
 unchanged={}
 for k in c:
  if k not in ("face_source_identity_schema_by_original_id_json","one_sided_remesh_operation_json"):
   unchanged[k]=bool(np.array_equal(np.asarray(c[k]),np.asarray(c2[k])))
 if not all(unchanged.values()):raise ValueError("metadata-only rewrite changed candidate geometry or another pinned array")
 report.update({"status":"SCHEMA_NORMALIZED","candidate":{"path":str(out),"sha256":sha(out),"faces":len(beforeF),
    "points":len(beforeP),"geometry_digest":geometry_digest(beforeP,beforeF)},
   "source_views":[{"path":str(i_alias),"sha256":sha(i_alias),"source_primitive":str(SRC_I),"geometry_digest":geometry_digest(i["points_world_m_f64"],i["faces_f64"])},
    {"path":str(s_alias),"sha256":sha(s_alias),"source_primitive":str(SRC_SUPPORT),"geometry_digest":geometry_digest(s["points_world_m_f64"],s["faces_f64"])}],
   "identity_proofs":proofs,"all_nonmetadata_candidate_arrays_bitwise_unchanged":all(unchanged.values()),
   "verified_unchanged_array_names":sorted(k for k,v in unchanged.items() if v),
   "metadata_changed_arrays":["face_source_identity_schema_by_original_id_json","one_sided_remesh_operation_json"],
   "producer260_path":str(PROD),"producer260_sha256":sha(PROD)})
 write(OUT/"normalization-receipt.json",report)
 print(json.dumps(report,indent=2))
except BaseException as exc:
 report.update({"status":"ERROR","error":{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()}})
 write(OUT/"normalization-receipt.json",report);print(json.dumps(report,indent=2));raise
