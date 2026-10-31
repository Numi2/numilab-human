#!/usr/bin/env python3
"""Compose face-only singleton output maps through paired retirement and exact c157 geometry identity."""
from pathlib import Path
import hashlib,json
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
R=H/"paired-inferred-envelope-resolution-reduction-293-root"
RUN=H/"native-apex-D-singleton-after-paired-293-root/run-001"
D0=R/"candidate-D-paired-resolution-reduced.npz"
MAP=R/"paired-retirement-maps.npz"
D282=H/"native-apex-D-ears-282-root/run-001/typed-001/candidate-D-typed-current.npz"
D015=H/"current-D-proof-015/candidate-D-saved-op-replay-source-mapped.npz"
C157=H/"current-volume-boundary-remesh-032/derived-D-pY-edge-split-001/candidate-D-pY-boundary-split.npz"
C157T=H/"current-volume-boundary-remesh-032/derived-D-pY-edge-split-001/registered-217-adapter-001/candidate-D-pY-boundary-split-registered.npz"
RAW=RUN/"candidate-D-native-apex-connected-raw.npz"
RAWREP=RUN/"report.json"
TYPED=RUN/"typed-001/candidate-D-typed-current.npz"
TRECEIPT=RUN/"typed-001/receipt.json"
OUT=RUN/"composed-row-maps-001.npz"
REC=RUN/"composed-row-maps-001.json"
PINS={D0:"9a66be5a75d097647d8ff6c0b8a7ee447faa687f2434d146709d6ae5fa316ff5",MAP:"6c9a38306195d0cce8d96d869a9e2cbdede87c0c9a477cb399e9ee24a3b1abca",D282:"a8845bc3ab5106fa08e77ecd37f7bc65ffe9991961f68351c444e2cf6b1a6866",D015:"15afa8eecdaeff456cda25ef26890da60d60dd06e0412ed2035c77a8cbeb460f",C157:"c157370a5389bdd54552abaffc991325b12bfafd2fed9772a639df3ac5d38487",C157T:"03a017eb723ac5a6f72435d3a5369c2693724d3dc349f1078c39002f2d799f98",RAW:"d88b9d5bf5564933cc717c13a146a5fb38b329bc508ca4eac60bafd1c73a0d51",TYPED:"214f888227a38da66bbbbeff70932927f393f9d6b34b1a2690f328a01b6a4dca",RAWREP:"ed2fc12e9e6733b447859a45d206881b2c5579d75ac7084ffdf99b78d2baf457",TRECEIPT:"b8afa27dfab89ba7beaf039030201029422c42d3fd4415bce3cac9c1ae2157ac"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def cyclic(a,b):
 a=tuple(map(int,a));b=tuple(map(int,b))
 return any(a[i:]+a[:i]==b for i in range(3))
def main():
 for p,h in PINS.items():
  if sha(p)!=h:raise RuntimeError(f"pin mismatch {p}: {sha(p)}")
 d0=load(D0);maps=load(MAP);d282=load(D282);d015=load(D015);c=load(C157T);raw=load(RAW);final=load(TYPED)
 rep=json.loads(RAWREP.read_text());trep=json.loads(TRECEIPT.read_text())
 if rep["physical_admission"] or rep["native_steps"]!=0 or rep["joint_complete_surface_walls_all185_pairs"]!=0 or rep["joint_changed_self_pairs"]!=0:raise RuntimeError("raw proposal admission/context gate mismatch")
 if trep["status"]!="PASS_CURRENT_ROW_TYPED_ANCESTRY_ADAPTER_ONLY" or trep["physical_admission"] or trep["native_steps"]!=0:raise RuntimeError("typed adapter receipt mismatch")
 # Face-only singleton operation: point coordinates remain exact and point maps compose without geometry matching.
 if not np.array_equal(final["points_world_m"],d0["points_world_m"]) or not np.array_equal(final["points_world_m_f32"],d0["points_world_m_f32"]):raise RuntimeError("singleton unexpectedly changed point arrays")
 raw_pm=np.asarray(raw["point_output_to_parent_rows"],np.int64)
 raw_fm=np.asarray(raw["face_output_to_parent_rows"],np.int64)
 if not np.array_equal(raw_pm,np.arange(len(d0["points_world_m"]),dtype=np.int64)):raise RuntimeError("nonidentity singleton point map")
 paired_p=maps["D_output_point_to_parent282_row"].astype(np.int64)
 paired_f=maps["D_output_face_to_parent282_row"].astype(np.int64)
 final_to_d282_p=paired_p[raw_pm]
 final_to_d282_f=np.full(len(raw_fm),-1,np.int64);ret=raw_fm>=0
 final_to_d282_f[ret]=paired_f[raw_fm[ret]]
 if not np.array_equal(final["quality_star_point_output_to_input_row"],d282["quality_star_point_output_to_input_row"][final_to_d282_p]):raise RuntimeError("D282/paired/final source point maps do not compose")
 final_to_d015_p=final["quality_star_point_output_to_input_row"].astype(np.int64)
 final_to_d015_f=final["quality_star_face_output_to_input_row"].astype(np.int64)
 d282_source_face=d282["quality_star_face_output_to_input_row"].astype(np.int64)
 check=final_to_d282_f>=0
 if not np.array_equal(d282_source_face[final_to_d282_f[check]],final_to_d015_f[check]):raise RuntimeError("final face source IDs do not compose through D282")
 # Exact F32 point map from final subset into c157; no nearest-coordinate matching is used.
 ccoord={}
 for i,p in enumerate(c["points_world_m_f32"]):
  ccoord.setdefault(np.asarray(p,dtype=np.float32).tobytes(),[]).append(i)
 final_to_c157_p=np.full(len(final["points_world_m_f32"]),-1,np.int64);amb=[]
 for i,p in enumerate(final["points_world_m_f32"]):
  hit=ccoord.get(np.asarray(p,dtype=np.float32).tobytes(),[])
  if len(hit)==1:final_to_c157_p[i]=hit[0]
  elif len(hit)>1:amb.append((i,hit))
 if np.any(final_to_c157_p<0) or amb:raise RuntimeError(f"exact c157 point map incomplete/ambiguous ({np.count_nonzero(final_to_c157_p<0)}, {len(amb)})")
 c157_to_final_p=np.full(len(c["points_world_m_f32"]),-1,np.int64)
 c157_to_final_p[final_to_c157_p]=np.arange(len(final_to_c157_p),dtype=np.int64)
 # Face crosswalk is exact triangle identity only. Unmatched rows remain -1, with complete typed unions retained separately.
 cfaces={}
 for i,t in enumerate(c["faces"]):
  ft=c157_to_final_p[t]
  if np.all(ft>=0):cfaces.setdefault(tuple(sorted(map(int,ft))),[]).append(i)
 final_to_c157_f=np.full(len(final["faces"]),-1,np.int64);final_to_c157_orientation=np.zeros(len(final["faces"]),np.int8);multi=[]
 for i,t in enumerate(final["faces"]):
  hits=cfaces.get(tuple(sorted(map(int,t))),[])
  if len(hits)==1:
   j=hits[0];final_to_c157_f[i]=j
   mapped_c_tri=c157_to_final_p[c["faces"][j]]
   if cyclic(t, mapped_c_tri):final_to_c157_orientation[i]=1
   elif cyclic((int(t[0]),int(t[2]),int(t[1])),mapped_c_tri):final_to_c157_orientation[i]=-1
   else:raise RuntimeError("sorted triangle match did not preserve cyclic orientation")
  elif len(hits)>1:multi.append((i,hits))
 if multi:raise RuntimeError("exact c157 face mapping ambiguous")
 c157_to_final_f=np.full(len(c["faces"]),-1,np.int64);ok=final_to_c157_f>=0;c157_to_final_f[final_to_c157_f[ok]]=np.flatnonzero(ok)
 np.savez_compressed(OUT,
  final_output_point_to_paired_D_row=raw_pm,
  final_output_face_to_paired_D_row=raw_fm,
  final_output_point_to_D282_row=final_to_d282_p,
  final_output_face_to_D282_row=final_to_d282_f,
  final_output_point_to_D015_row=final_to_d015_p,
  final_output_face_to_D015_row=final_to_d015_f,
  final_output_point_to_c157_row=final_to_c157_p,
  c157_point_to_final_output_row=c157_to_final_p,
  final_output_face_to_c157_exact_triangle_row=final_to_c157_f,
  final_output_face_to_c157_orientation=final_to_c157_orientation,
  c157_face_to_final_output_exact_triangle_row=c157_to_final_f)
 report={
  "schema":"numi.paired-retirement-singleton-composed-row-maps.v1","status":"PASS_SOURCE_ONLY_ROW_MAP_COMPOSITION","physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
  "inputs":{str(p):sha(p) for p in PINS},
  "paired_retirement_receipt":{"path":str(R/"receipt.json"),"sha256":sha(R/"receipt.json")},
  "paired_operation":{"path":str(R/"operation.json"),"sha256":sha(R/"operation.json")},
  "singleton_raw_report":{"path":str(RAWREP),"sha256":sha(RAWREP)},
  "singleton_typed_receipt":{"path":str(TRECEIPT),"sha256":sha(TRECEIPT)},
  "final_typed_candidate":{"path":str(TYPED),"sha256":sha(TYPED)},
  "maps":{"path":str(OUT),"sha256":sha(OUT),"keys":["final_output_point_to_paired_D_row","final_output_face_to_paired_D_row","final_output_point_to_D282_row","final_output_face_to_D282_row","final_output_point_to_D015_row","final_output_face_to_D015_row","final_output_point_to_c157_row","c157_point_to_final_output_row","final_output_face_to_c157_exact_triangle_row","final_output_face_to_c157_orientation","c157_face_to_final_output_exact_triangle_row"]},
  "retirement_rows":{"paired_D282_points":[56714,56715,56716,56717],"paired_D282_faces":[62075,62077,162298,162299],"singleton_D282_faces":[119513,119514,238473,238572,238573,238574],"singleton_paired_D_faces":rep["paired_retirement_input"]["singleton_D0_parent_face_rows"],"singleton_final_generated_face_rows":next(op["new_output_face_rows"] for op in rep["accepted_operations"] if op["role"]=="D")},
  "counts":{"final_D_points":len(final_to_d282_p),"final_D_faces":len(final_to_d282_f),"faces_exactly_shared_with_c157":int(np.count_nonzero(final_to_c157_f>=0)),"final_faces_without_exact_c157_triangle":int(np.count_nonzero(final_to_c157_f<0)),"c157_faces_without_exact_final_triangle":int(np.count_nonzero(c157_to_final_f<0)),"c157_points_retired_from_final":int(np.count_nonzero(c157_to_final_p<0))},
  "checks":{"final_points_bitwise_equal_paired_D0":True,"final_points_exactly_mapped_to_D282":True,"final_to_D015_point_map_matches_D282_composition":True,"exact_unique_F32_point_crosswalk_to_c157":True,"c157_face_crosswalk_is_exact_triangle_identity_only":True,"unmatched_c157_faces_remain_minus_one_no_nearest_or_fabricated_face_identity":True,"full_typed_patch_union_remains_in_final_candidate":True,"all185_context_and_joint_self_receipts_pinned":True},
  "limitations":["c157 is a sibling geometry/topology base, not the immediate parent of the D282 branch; exact c157 face map is partial where triangle topology differs.","Physical combined-shell closure, Fat mass redistribution, volume rebind, and native qualification remain pending."]
 }
 REC.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"status":report["status"],"maps_path":str(OUT),"maps_sha256":sha(OUT),"receipt_path":str(REC),"receipt_sha256":sha(REC),"counts":report["counts"],"final_candidate_sha256":sha(TYPED)}))
if __name__=="__main__":main()
