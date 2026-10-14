#!/usr/bin/env python3
from __future__ import annotations
import gc, hashlib, json, signal, sys, time
from pathlib import Path
import numpy as np
BASE=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
AUDIT=BASE/"full-anatomy-delivery-audit-1378"
WORK=AUDIT/"pons-fourth-ventricle-reconstruction-001"
SRC=AUDIT/"forearm492-head016-vessel-seam-integrated-source-002/resting-thorax.nhanatomy"
OLD_SRC=BASE/"brachiocephalic-source-candidate-007-006/resting-thorax.nhanatomy"
FLOOR_NPZ=AUDIT/"pons-fourth-ventricle-reconstruction-001/local-floor-candidate-005/pons-fourth-ventricle-floor-candidate.npz"
FLOOR_REPORT=AUDIT/"pons-fourth-ventricle-reconstruction-001/local-floor-candidate-005/candidate-report.json"
FLOOR_DUAL=AUDIT/"pons-fourth-ventricle-reconstruction-001/local-floor-candidate-005/dual-lineage.json"
CURRENT_PACK=AUDIT/"forearm492-head016-vessel-seam-native-emission-diagnostic-003/native-run/accepted-geometry/step-0.mrvpack"
CURRENT_RECEIPT=AUDIT/"forearm492-head016-vessel-seam-native-emission-diagnostic-003/native-run/accepted-geometry/step-0.receipt.json"
CAND=WORK/"cord-local-clearance-candidate-compile-001/resting-thorax-candidate.nhanatomy"
COMPILE_REPORT=WORK/"cord-local-clearance-candidate-compile-001/compile-report.json"
CAND_NPZ=WORK/"cord-local-clearance-candidate-003/candidate-01.npz"
MANIFEST=BASE/"accepted-capture-inputs-1398-001.json"
LATE=BASE/"back412-skin-native-022-001-60s/baseline/native-run/accepted-geometry"
OUTDIR=WORK/"cord-local-clearance-motion-006"
REPORT=OUTDIR/"report.json"
POSITION_FILE=OUTDIR/"candidate-pose-positions.npz"
sys.path.insert(0,"/Users/n/numi-human-pons-floor-compiler-20261014/src")
from numilab_human import resting_anatomy_verify as av
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human import model as human_model

EXPECTED={
 "source_nha":"2f8af1031b3c5b8cc26866f32935c9b695127a2f357732e566be3f14fdc2666d",
 "historical_source_nha":"3a3693b8de2264127e9190a151dca460002ae3b8a0a09edea312bd4eed73ea42",
 "floor_source_npz":"7c9f5fb245169ce033caf6919c1397b6084acab802c7a745792e282dd024c989",
 "floor_source_report":"0c66ba44381b6dc9011b11b4801fd8d4f9ee9b51984345b7fc8a360696aefda3",
 "floor_dual_lineage":"65fd020482a41b6d85d2ed91bd4d1bf68cb436487af77e093ab0f0cd0fe56458",
 "current_source_pack":"3b82ed8fff3c4eb2bd4dd8dc1f5ccb276fe97a00cb7aa82dff5db6956c2bbb39",
 "current_source_receipt":"62deacd6a6ae9c0e14bafdd78538013d9a901e3217fb302b94927036e7ed061b",
 "compiled_candidate":"9fb3ffb10e458589b2c69779bd1336bfc4dc8e704079a1e68e4b22a2b5b7ba09",
 "compile_report":"efc7efef96eba68da58cc8f8c7cc94ede25255a1a25a579f4ac11038a1f040c8",
 "candidate_npz":"d9798684d4412fd97093ea056d009e2594cbaa40855cab2e1e82b2b84c5bdab6",
 "candidate_gate":"3e04eb8ee9fcc653951b245f285d3a99e84c2b299a1605cb9694018ded5ff33f",
 "manifest":"3df61c8bf202e343a9124588fa94be17e9876db17d1a6ef5dd5fa615aaa7b815",
 "late_pack":"b1c25b89e828ecf62b5b7dfd58ac3a38c9f1d420306e5f9b789d04ec29859c3d",
 "late_receipt":"84a98b1d4d4098c762798517e91248b6f3f47df3b841ce344c04d593f9b58b79",
 "verifier":"de87c97732baed438a50cce4856b602b4abed4dc2c05230bc233004b5e84ba60",
 "model":"29d5664ce1485c520c8ab8cb088f496a82c68e570400030dcaea98b2c9d6450e",
}
def sha(p):
 return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic_json(p,obj):
 t=Path(str(p)+".tmp")
 t.write_text(json.dumps(obj,sort_keys=True,indent=2)+"\n")
 t.replace(p)
def deadline(signum,frame):
 raise TimeoutError("1200-second total bound")

signal.signal(signal.SIGALRM,deadline)
signal.alarm(1200)
t0=time.monotonic()
for p,k in [(SRC,"source_nha"),(OLD_SRC,"historical_source_nha"),(FLOOR_NPZ,"floor_source_npz"),
            (FLOOR_REPORT,"floor_source_report"),(FLOOR_DUAL,"floor_dual_lineage"),
            (CURRENT_PACK,"current_source_pack"),(CURRENT_RECEIPT,"current_source_receipt"),
            (CAND,"compiled_candidate"),(COMPILE_REPORT,"compile_report"),
            (CAND_NPZ,"candidate_npz"),(WORK/"cord-local-clearance-candidate-003/report.json","candidate_gate"),
            (MANIFEST,"manifest"),(LATE/"step-14975.mrvpack","late_pack"),
            (LATE/"step-14975.receipt.json","late_receipt"),(Path(av.__file__),"verifier"),
            (Path(human_model.__file__),"model")]:
 actual=sha(p)
 if actual!=EXPECTED[k]:
  raise SystemExit("input SHA mismatch "+k+": "+actual)
src_header,src_records,src_rows=_parse_payload(SRC.read_bytes())
old_header,old_records,old_rows=_parse_payload(OLD_SRC.read_bytes())
cand_header,cand_records,cand_rows=_parse_payload(CAND.read_bytes())
if list(src_rows)!=list(cand_rows) or len(src_rows)!=525:
 raise SystemExit("compiled ABI5 row inventory mismatch")
src323=src_rows[323]; old323=old_rows[323]; cand323=cand_rows[323]
if (src323["body_index"],src323["layer"])!=(23,1) or (old323["body_index"],old323["layer"])!=(23,1) or not np.array_equal(src323["faces"],cand323["faces"]):
 raise SystemExit("row323 source owner or face order mismatch")
base_xyz=np.asarray(src323["vertices6"][:,:3],dtype=np.float32)
old_xyz=np.asarray(old323["vertices6"][:,:3],dtype=np.float32)
candidate_xyz=np.asarray(cand323["vertices6"][:,:3],dtype=np.float32)
if old_xyz.shape!=(3567,3) or base_xyz.shape!=(9506,3) or candidate_xyz.shape!=base_xyz.shape:
 raise SystemExit("row323 vertex shape mismatch")
if not np.array_equal(np.asarray(src323["vertices6"][:len(old323["vertices6"])]),np.asarray(old323["vertices6"])):
 raise SystemExit("current floor source does not preserve historical source vertex prefix")
floor=np.load(FLOOR_NPZ,allow_pickle=False)
if not np.array_equal(floor["fourth_ventricle_vertices6_f32"],src323["vertices6"]) or not np.array_equal(floor["fourth_ventricle_faces_i64"],src323["faces"]):
 raise SystemExit("current source row323 differs from pinned floor005 candidate")
if floor["fourth_ventricle_new_vertex_parent_face_i64"].shape!=(2848,) or floor["fourth_ventricle_new_vertex_parent_weights_f64"].shape!=(2848,3):
 raise SystemExit("floor005 source-face ancestry arrays malformed")
if floor["fourth_ventricle_face_source_parent_i64"].shape!=(len(src323["faces"]),):
 raise SystemExit("floor005 face ancestry array malformed")
new_parent_faces=np.asarray(floor["fourth_ventricle_new_vertex_parent_face_i64"],dtype=np.int64)
new_parent_weights=np.asarray(floor["fourth_ventricle_new_vertex_parent_weights_f64"],dtype=np.float64)
if np.any(new_parent_faces<0) or np.any(new_parent_faces>=len(old323["faces"])) or not np.isfinite(new_parent_weights).all():
 raise SystemExit("floor005 generated ventricle vertex ancestry outside historical source")
if np.any(new_parent_weights < -1.0e-12) or np.max(np.abs(np.sum(new_parent_weights,axis=1)-1.0),initial=0.0)>1.0e-10:
 raise SystemExit("floor005 generated ventricle vertex parent weights are not positive partitions")
face_source_ids=np.asarray(floor["fourth_ventricle_face_source_surface_stable_id_i64"],dtype=np.int64)
if face_source_ids.shape!=(len(src323["faces"]),) or not set(np.unique(face_source_ids)).issubset({323,345}):
 raise SystemExit("floor005 face source identities malformed")
candidate_slots=np.unique(np.asarray(src323["faces"],dtype=np.int64))
manifest_bytes=MANIFEST.read_bytes()
if hashlib.sha256(manifest_bytes).hexdigest()!=EXPECTED["manifest"]:
 raise SystemExit("accepted-capture manifest hash mismatch")
manifest=json.loads(manifest_bytes)
capture_rows=manifest.get("capture_rows")
expected_steps=[0,3807,4767,5023,5599,6207,7423,8000]
if not isinstance(capture_rows,list) or [r.get("step") for r in capture_rows]!=expected_steps:
 raise SystemExit("N27 capture order mismatch")
captures=[dict(r) for r in capture_rows]
captures.append({"step":14975,"accepted_time_s":None,
 "pack_path":str(LATE/"step-14975.mrvpack"),"recorded_pack_sha256":EXPECTED["late_pack"],
 "receipt_path":str(LATE/"step-14975.receipt.json"),"receipt_sha256":EXPECTED["late_receipt"],
 "capture_family":"N22 late accepted capture"})
with np.load(CAND_NPZ,allow_pickle=False) as z:
 step0_candidate=np.asarray(z["captured_step0_candidate_xyz_f32"],dtype=np.float32).copy()
if step0_candidate.ndim!=2 or step0_candidate.shape[1]!=3:
 raise SystemExit("candidate step0 captured coordinates have wrong shape")
OUTDIR.mkdir(parents=True,exist_ok=True)
if REPORT.exists() or POSITION_FILE.exists():
 raise SystemExit("successor output path is not empty")
report={"schema":"numi.human.pons_cord_retained_pose_pair_gate.v1",
 "status":"running","candidate_admission":False,"native_candidate_run_performed":False,
 "pose_geometry_kind":"counterfactual candidate source geometry transformed by each actual accepted_body_poses[23], audited against exact retained native row12",
 "scope":"row323 versus row12 only; exact pair and bidirectional containment; not an all-neighbor gate",
 "surface_pair":[[51010,323],[51012,12]],
 "source_ancestry":{"historical_source_prefix_vertices":3567,"floor_source_vertices":9506,"floor_source_faces":12676,
                    "historical_prefix_exact":True,"floor005_arrays_exact_current_source":True,
                    "note":"Retained captures use historical row323 topology. Each capture is validated against pinned historical source3a369; current 2f8a/floor005 candidate topology is separately validated against actual source003 step0 emission. Candidate geometry is transformed through actual accepted body23 poses and tested against retained row12; no mapping of new floor faces onto missing historical faces."},
 "pins":dict(EXPECTED),"captures":[]}
position_arrays={}

def pose_for(receipt):
 poses=receipt.get("accepted_body_poses")
 if not isinstance(poses,list):
  raise ValueError("accepted_body_poses missing")
 matches=[r for r in poses if isinstance(r,dict) and r.get("body_index")==23]
 if len(matches)!=1:
  raise ValueError("accepted_body_poses must contain exactly one body_index 23")
 row=matches[0]
 pos=np.asarray(row.get("position_m"),dtype=np.float64)
 quat=np.asarray(row.get("quaternion_xyzw"),dtype=np.float64)
 if pos.shape!=(3,) or quat.shape!=(4,) or not np.isfinite(pos).all() or not np.isfinite(quat).all():
  raise ValueError("body23 accepted pose shape/finite failure")
 return pos,quat
def surface_from_capture(surfaces,key):
 if key not in surfaces:
  raise ValueError("typed target absent: "+repr(key))
 p,f,n,g=surfaces[key]
 return av.prepare_surface(key,p,f,normals=n,source_vertex_ids=g)
try:
 current_receipt=json.loads(CURRENT_RECEIPT.read_text())
 if current_receipt.get("accepted_step")!=0:
  raise ValueError("current-source diagnostic is not step0")
 current_pos,current_q=pose_for(current_receipt)
 current_rot=np.asarray(human_model._myosim_matrix_from_quaternion_xyzw(current_q.tolist()),dtype=np.float64)
 if current_rot.shape!=(3,3) or not np.isfinite(current_rot).all():
  raise ValueError("current-source body23 transform invalid")
 current_surfaces,current_endpoint=av.load_capture(CURRENT_PACK,CURRENT_RECEIPT)
 current_key=(51010,323)
 if current_key not in current_surfaces:
  raise ValueError("current source003 typed row323 missing")
 current_v,current_faces,current_normals,current_gids=current_surfaces[current_key]
 current_local=np.asarray(src323["faces"],dtype=np.int64)
 current_face_gids=np.asarray(current_gids,dtype=np.int64)[np.asarray(current_faces,dtype=np.int64)]
 current_offsets=current_face_gids-current_local
 if not np.all(current_offsets==current_offsets.flat[0]):
  raise ValueError("current-source row323 face-corner map is not a constant offset")
 current_offset=int(current_offsets.flat[0])
 _,current_binding=av.retain_bound_source_faces(
  current_key,current_surfaces[current_key],current_local,current_offset,
  np.arange(len(current_local),dtype=np.int64))
 current_capture_face_order=np.searchsorted(np.asarray(current_gids,dtype=np.int64),current_local+current_offset)
 if np.any(current_capture_face_order>=len(current_gids)) or not np.array_equal(current_capture_face_order,np.asarray(current_faces,dtype=np.int64)):
  raise ValueError("current-source row323 ordered face-to-capture compact-slot mapping mismatch")
 current_slots=np.unique(current_local)
 current_map=np.searchsorted(np.asarray(current_gids,dtype=np.int64),current_slots+current_offset)
 if np.any(current_map>=len(current_gids)) or not np.array_equal(np.asarray(current_gids,dtype=np.int64)[current_map],current_slots+current_offset):
  raise ValueError("current-source row323 used-slot map incomplete")
 current_actual=current_v[current_map].astype(np.float64)
 current_base_expected=base_xyz[current_slots].astype(np.float64)@current_rot.T+current_pos
 current_base_residual=np.linalg.norm(current_actual-current_base_expected,axis=1)
 current_base_max=float(np.max(current_base_residual,initial=0.0))
 if current_base_max>2.0e-6:
  raise ValueError("current source003 baseline emission differs from compiled current source by >2um")
 if step0_candidate.shape!=current_actual.shape:
  raise ValueError("C2 step0 candidate coordinates do not align with current-source used slots")
 current_candidate_expected=(current_actual+(candidate_xyz[current_slots].astype(np.float64)-base_xyz[current_slots].astype(np.float64))@current_rot.T).astype(np.float32)
 step0_parity_error=np.linalg.norm(step0_candidate.astype(np.float64)-current_candidate_expected.astype(np.float64),axis=1)
 step0_parity_max=float(np.max(step0_parity_error,initial=0.0))
 if step0_parity_max>2.0e-6:
  raise ValueError("candidate step0 field differs from actual-source-emission-based C2 prediction by >2um")
 report["source_emission_preflight"]={"current_source_step":0,"body_index":23,"face_global_offset":current_offset,
   "used_source_vertices":int(len(current_slots)),"baseline_source_to_actual_emission_max_residual_m":current_base_max,
   "candidate_step0_npz_max_error_m":step0_parity_max,"passed":True}
 atomic_json(REPORT,report)
 del current_surfaces,current_v,current_faces,current_normals,current_gids,current_actual,current_base_expected
 gc.collect()
 for caprow in captures:
  step=int(caprow["step"])
  pack=Path(caprow["pack_path"])
  receipt_path=Path(caprow["receipt_path"])
  if sha(pack)!=caprow["recorded_pack_sha256"] or sha(receipt_path)!=caprow["receipt_sha256"]:
   raise ValueError("capture pin mismatch at step "+str(step))
  cap_receipt=json.loads(receipt_path.read_text())
  if cap_receipt.get("accepted_step")!=step:
   raise ValueError("capture receipt step mismatch")
  pose_pos,pose_q=pose_for(cap_receipt)
  matrix=np.asarray(human_model._myosim_matrix_from_quaternion_xyzw(pose_q.tolist()),dtype=np.float64)
  if matrix.shape!=(3,3) or not np.isfinite(matrix).all():
   raise ValueError("body23 transform invalid")
  rotation=matrix
  transformed_old=old_xyz.astype(np.float64)@rotation.T+pose_pos
  surfaces,endpoint=av.load_capture(pack,receipt_path)
  if endpoint.get("accepted_step")!=step:
   raise ValueError("capture loader step mismatch")
  key_v=(51010,323); key_c=(51012,12)
  if key_v not in surfaces or key_c not in surfaces:
   raise ValueError("required typed source/target missing")
  captured_v,captured_faces,captured_normals,captured_gids=surfaces[key_v]
  old_face_local=np.asarray(old323["faces"],dtype=np.int64)
  face_gids=np.asarray(captured_gids,dtype=np.int64)[np.asarray(captured_faces,dtype=np.int64)]
  offsets=face_gids-old_face_local
  if not np.all(offsets==offsets.flat[0]):
   raise ValueError("historical row323 face-corner global/local correspondence is not one constant offset")
  offset=int(offsets.flat[0])
  _,historical_binding=av.retain_bound_source_faces(
   key_v,surfaces[key_v],old_face_local,offset,
   np.arange(len(old_face_local),dtype=np.int64))
  captured_face_order=np.searchsorted(np.asarray(captured_gids,dtype=np.int64),old_face_local+offset)
  if np.any(captured_face_order>=len(captured_gids)) or not np.array_equal(captured_face_order,np.asarray(captured_faces,dtype=np.int64)):
   raise ValueError("historical row323 ordered face-to-capture compact-slot mapping mismatch")
  old_slots=np.unique(old_face_local)
  mapped_gids=old_slots+offset
  capture_idx=np.searchsorted(np.asarray(captured_gids,dtype=np.int64),mapped_gids)
  if np.any(capture_idx>=len(captured_gids)) or not np.array_equal(np.asarray(captured_gids,dtype=np.int64)[capture_idx],mapped_gids):
   raise ValueError("historical row323 used-slot map is incomplete")
  actual_old=captured_v[capture_idx].astype(np.float64)
  residual=np.linalg.norm(actual_old-transformed_old[old_slots],axis=1)
  max_res=float(np.max(residual,initial=0.0))
  if max_res>2.0e-6:
   raise ValueError("historical source-to-capture residual exceeds 2um at step "+str(step)+": "+str(max_res))
  candidate_world=(candidate_xyz.astype(np.float64)@rotation.T+pose_pos).astype(np.float32)
  step0_max=None
  vent=av.prepare_surface(key_v,candidate_world,src323["faces"],source_vertex_ids=np.arange(len(candidate_world),dtype=np.int64))
  cord=surface_from_capture(surfaces,key_c)
  if not vent.closed or not cord.closed:
   raise ValueError("required source/target surface is not closed at step "+str(step))
  pair=av.audit_surface_pair(vent,cord)
  if pair.get("containment"):
   containment=pair["containment"]
   if len(containment)!=2:
    raise ValueError("bidirectional containment result malformed")
  elif pair.get("aabb_disjoint"):
   containment=[{"status":"aabb_disjoint_outside","inside_component_witnesses":[]},
                {"status":"aabb_disjoint_outside","inside_component_witnesses":[]}]
  else:
   raise ValueError("bidirectional containment was not evaluated")
  row={"step":step,"accepted_time_s":caprow.get("accepted_time_s"),
       "pose_body_index":23,"pose_position_m":pose_pos.tolist(),"pose_quaternion_xyzw":pose_q.tolist(),
       "historical_source_face_global_offset":offset,"historical_used_source_vertex_count":int(len(old_slots)),
       "historical_source_to_capture_max_residual_m":max_res,
       "step0_candidate_npz_max_error_m":step0_max,
       "candidate_max_delta_from_source_m":float(np.max(np.linalg.norm(candidate_xyz-base_xyz,axis=1),initial=0.0)),
       "pair":pair,"containment":containment,"passed":bool(pair.get("passed",False))}
  report["captures"].append(row)
  position_arrays[f"step_{step}"]=candidate_world[candidate_slots].astype(np.float32)
  report["status"]="running"; report["completed_captures"]=len(report["captures"])
  report["elapsed_s"]=time.monotonic()-t0
  atomic_json(REPORT,report)
  np.savez_compressed(POSITION_FILE,**position_arrays)
  del surfaces,vent,cord,pair
  gc.collect()
  if not row["passed"]:
   report["status"]="rejected_first_failure"; report["failed_step"]=step
   atomic_json(REPORT,report)
   break
 else:
  report["status"]="passed_scoped_counterfactual_pair_gate"
  report["passed"]=True
  report["completed_captures"]=len(report["captures"])
  atomic_json(REPORT,report)
except BaseException as exc:
 report["status"]="failed_or_interrupted"
 report["error_type"]=type(exc).__name__
 report["error"]=str(exc)
 report["elapsed_s"]=time.monotonic()-t0
 atomic_json(REPORT,report)
 raise
finally:
 signal.alarm(0)
