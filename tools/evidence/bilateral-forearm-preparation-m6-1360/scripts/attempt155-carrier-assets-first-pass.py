#!/usr/bin/env python3
import hashlib, importlib.util, json, pickle, struct, sys, time
from fractions import Fraction as F
from pathlib import Path
import numpy as np

ROOT=Path("/Users/n/numi-human-fds132-m6-1350")
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
OUT=ROOT/"attempt-155-right-candidate150-carrier-assets-001"
ASSET=OUT/"assets"
CAND104=ROOT/"attempt-104-right-witness-band-refinement-001/result-001/candidate.pkl"
CK104=ROOT/"attempt-104-right-witness-band-refinement-001/result-001/construction-checkpoint.json"
CAND122=ROOT/"attempt-122-right-cap-boundary-directed-contractions-001/result-001/candidate.pkl"
CAND150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/candidate.pkl"
CHECK150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/construction-checkpoint.json"
SUMMARY150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/summary.json"
SOURCE150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/source-critical.json"
F64_150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/current8000-f64-critical.json"
F32SEP150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/current8000-f32-separate-critical.json"
F32FMA150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/current8000-f32-fma-critical.json"
REST150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/new-midpoint-rest-check.json"
REPLAY154=ROOT/"attempt-154-exact-source-route-field-replay-001/report.json"
CONSTR122=ROOT/"attempt-122-right-cap-boundary-directed-contractions-001/result-001/construction-report.json"
SOURCE122=ROOT/"attempt-122-right-cap-boundary-directed-contractions-001/result-001/audit-001/audit-report.json"
BASE=A/"forearm-shared-field-native-012-diagonal097/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
SOURCE=A/"temporalis164-native-m6-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
REST125=ROOT/"attempt-125-right-rest-registration-check-001/report.json"
F32=ROOT/"attempt-123-right-native-like-f32-audit-001/result-005/report.json"
ENCROOT=Path("/Users/n/numi-human-left-pq-publication-1344/src")
HELPER=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
OUT.mkdir(exist_ok=True)
ASSET.mkdir(exist_ok=True)
sys.path.insert(0,str(ENCROOT))
from numilab_human import passive_tissue_parent_carriers as enc
from numilab_human import passive_attachment_composition as pc
START=time.monotonic()

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def req(x,m):
 if not x:raise RuntimeError(m)
def oriented_key(t):
 t=tuple(map(int,t))
 return min(t,t[1:]+t[:1],t[2:]+t[:2])

def main():
 inputs=(Path(__file__),CAND104,CK104,CAND122,CONSTR122,SOURCE122,CAND150,CHECK150,SUMMARY150,SOURCE150,F64_150,F32SEP150,F32FMA150,REST150,REPLAY154,BASE,SOURCE,REST125,HELPER,ENCROOT/"numilab_human/passive_tissue_parent_carriers.py")
 for p in inputs:req(p.exists(),f"missing input {p}")
 hashes={str(p):sha(p) for p in inputs}
 req(hashes[str(CAND104)]=="28ab429feb10615a5cfbd2345148d6b0619a7e75b41301b3f86f0a244202ad6d","candidate104 hash changed")
 req(hashes[str(CAND122)]=="aeeaac9444ef873624d7a0c9e667ffa694b0cb1509ee89e3ccdea8acbc665f94","candidate122 hash changed")
 req(hashes[str(CAND150)]=="8607d93a055a2045e823f80b4e04b6d6dcd6f5d8bfa95b442485a35b555918b1","candidate150 hash changed")
 source150=json.loads(SOURCE150.read_text());f64_150=json.loads(F64_150.read_text());summary150=json.loads(SUMMARY150.read_text());replay154=json.loads(REPLAY154.read_text())
 req(source150["candidate_sha256"]==hashes[str(CAND150)] and all(source150["rows"][str(s)]["self_count"]==0 for s in (131,133)) and source150["union"]["self_count"]==0,"candidate150 exact source gate is not passing")
 req(f64_150["candidate_sha256"]==hashes[str(CAND150)] and all(f64_150["rows"][str(s)]["self_count"]==0 for s in (131,133)) and f64_150["union"]["self_count"]==0,"candidate150 Float64 current8000 gate is not passing")
 req(summary150["candidate_sha256"]==hashes[str(CAND150)] and summary150["candidate_admission"] is False and summary150["gpu_launched"] is False,"candidate150 status/identity changed")
 req(replay154["candidate150_sha256"]==hashes[str(CAND150)] and all(replay154["candidate150_exact_source_replay"][str(s)]["changed_scalar_entry_count"]==0 for s in (131,133)),"canonical exact-coordinate route-field replay does not pass")
 req(json.loads(SOURCE122.read_text()).get("status")=="passed_bounded_exact_source_and_late_pose_geometry","candidate122 exact source audit not passed")
 c104=pickle.loads(CAND104.read_bytes());c122=pickle.loads(CAND122.read_bytes());c150=pickle.loads(CAND150.read_bytes())
 core=list(map(int,c150["core_body_indices"]))
 req(core==list(map(int,c104["core_body_indices"])) and core==list(map(int,c122["core_body_indices"])) and len(core)==21 and core==sorted(set(core)),"full 21-core order changed")
 delta={"input_candidate104_path":str(CAND104),"input_candidate104_sha256":hashes[str(CAND104)],"output_candidate122_path":str(CAND122),"output_candidate122_sha256":hashes[str(CAND122)],"source_conditioning_report_path":str(CONSTR122),"source_conditioning_report_sha256":hashes[str(CONSTR122)],"rows":{}}
 for sid in (131,133):
  old,new=c104["surfaces"][sid],c122["surfaces"][sid]
  oldg=list(map(int,old["global_vertex_ids"]));newg=list(map(int,new["global_vertex_ids"]))
  req(len(set(oldg))==len(oldg) and len(set(newg))==len(newg),"duplicate active global ID")
  oldpts={g:tuple(F(x) for x in c104["global_vertices_exact_m"][g]) for g in oldg}
  newpts={g:tuple(F(x) for x in c122["global_vertices_exact_m"][g]) for g in newg}
  common=set(oldpts)&set(newpts)
  maxcoord=max((max(abs(a-b) for a,b in zip(oldpts[g],newpts[g])) for g in common),default=F(0))
  oldw={g:tuple(float(x) for x in w) for g,w in zip(oldg,old["core_weights_float64"])}
  neww={g:tuple(float(x) for x in w) for g,w in zip(newg,new["core_weights_float64"])}
  unchanged_w=sum(oldw[g]==neww[g] for g in common)
  oldfaces={oriented_key(t) for t in old["faces_global"]}
  newfaces={oriented_key(t) for t in new["faces_global"]}
  delta["rows"][str(sid)]={"candidate104_listed_vertices":len(oldg),"candidate122_listed_vertices":len(newg),"candidate104_face_incident_vertices":len(set(map(int,np.asarray(old["faces"]).ravel()))),"candidate122_face_incident_vertices":len(set(map(int,np.asarray(new["faces"]).ravel()))),"candidate122_listed_unreferenced_vertices":len(newg)-len(set(map(int,np.asarray(new["faces"]).ravel()))),"active_vertex_ids_removed":len(set(oldg)-set(newg)),"active_vertex_ids_added":len(set(newg)-set(oldg)),"candidate104_oriented_face_count":len(oldfaces),"candidate122_oriented_face_count":len(newfaces),"oriented_face_keys_removed":len(oldfaces-newfaces),"oriented_face_keys_added":len(newfaces-oldfaces),"common_global_vertex_ids":len(common),"common_vertex_exact_source_coordinate_max_delta_m":[maxcoord.numerator,maxcoord.denominator],"common_global_vertex_ids_with_exactly_unchanged_21_core_weights":unchanged_w,"all_common_vertex_coordinates_unchanged":maxcoord==0,"all_common_vertex_weights_unchanged":unchanged_w==len(common)}
  req(maxcoord==0 and unchanged_w==len(common),"candidate104-to122 source coordinates/field weights changed unexpectedly")
 cons=json.loads(CONSTR122.read_text())
 delta["accepted_contractions"]=len(cons["accepted_contractions"])
 delta["configured_displacement_guard_from_candidate104_m"]=cons["max_cumulative_displacement_from104_m"]
 delta["maximum_accepted_cumulative_displacement_from_candidate104_m"]=max(float(x["cumulative_max_displacement_from_candidate104_m"]) for x in cons["accepted_contractions"])
 delta["source_geometry_change"]="Paired-face endpoint reassignment and face deletion in generated shared cap; all retained global source coordinates and 21-core field arrays remain unchanged. Surface tessellation changes, so no exact surface coverage equivalence is claimed."
 delta_path=OUT/"construction-delta-104-to-122.json"
 delta_path.write_text(json.dumps(delta,sort_keys=True,indent=2)+chr(10))
 delta150={"input_candidate122_path":str(CAND122),"input_candidate122_sha256":hashes[str(CAND122)],"output_candidate150_path":str(CAND150),"output_candidate150_sha256":hashes[str(CAND150)],"midpoint_field_correction_report_path":str(CHECK150),"midpoint_field_correction_report_sha256":hashes[str(CHECK150)],"rows":{}}
 for sid in (131,133):
  old,new=c122["surfaces"][sid],c150["surfaces"][sid]
  oldg=list(map(int,old["global_vertex_ids"]));newg=list(map(int,new["global_vertex_ids"]))
  oldloc={g:i for i,g in enumerate(oldg)};newloc={g:i for i,g in enumerate(newg)}
  common=set(oldloc)&set(newloc);added=sorted(set(newloc)-set(oldloc));removed=sorted(set(oldloc)-set(newloc))
  unchanged_coords=all(tuple(old["vertices_exact_m"][oldloc[g]])==tuple(new["vertices_exact_m"][newloc[g]]) for g in common)
  unchanged_weights=all(tuple(map(float,old["core_weights_float64"][oldloc[g]]))==tuple(map(float,new["core_weights_float64"][newloc[g]])) for g in common)
  oldfaces={oriented_key(t) for t in old["faces_global"]};newfaces={oriented_key(t) for t in new["faces_global"]}
  delta150["rows"][str(sid)]={"candidate122_vertices":len(oldg),"candidate150_vertices":len(newg),"common_vertices":len(common),"added_vertex_ids":added,"removed_vertex_ids":removed,"common_exact_coordinates_unchanged":unchanged_coords,"common_full21_weights_unchanged":unchanged_weights,"candidate122_face_count":len(oldfaces),"candidate150_face_count":len(newfaces),"oriented_faces_added":len(newfaces-oldfaces),"oriented_faces_removed":len(oldfaces-newfaces)}
  req(len(added)==1 and not removed and unchanged_coords and unchanged_weights,"candidate150 changed inherited coordinates/weights beyond one new midpoint")
 delta150_path=OUT/"construction-delta-122-to-150.json"
 delta150_path.write_text(json.dumps(delta150,sort_keys=True,indent=2)+chr(10))
 # Bind carriers to exactly the source table consumed by native012.
 hsp=importlib.util.spec_from_file_location("fwd_126",HELPER)
 h=importlib.util.module_from_spec(hsp);sys.modules[hsp.name]=h;hsp.loader.exec_module(h)
 base=h.load_tissue(BASE);source=h.load_tissue(SOURCE)
 hdr=struct.unpack_from("<8s6I32s",base["raw"],0);src_hdr=struct.unpack_from("<8s6I32s",source["raw"],0)
 req(hdr[0]==bytes.fromhex("4e48544953533400") and hdr[1]==5 and src_hdr[0]==bytes.fromhex("4e48544953533400") and src_hdr[1]==5,"NHTISS4 ABI mismatch")
 req(base["raw"][28:64]==source["raw"][28:64],"native012/source registration or source digest differs")
 base_bind={};source_bind={}
 for i,row in enumerate(base["bindings"]):
  c=int(row["core"]);rawrow=base["raw"][base["bo"]+36*i:base["bo"]+36*(i+1)]
  req(c not in base_bind or base_bind[c]==rawrow,f"conflicting repeated native012 binding core {c}")
  base_bind.setdefault(c,rawrow)
 for i,row in enumerate(source["bindings"]):
  c=int(row["core"]);rawrow=source["raw"][source["bo"]+36*i:source["bo"]+36*(i+1)]
  req(c not in source_bind or source_bind[c]==rawrow,f"conflicting repeated source binding core {c}")
  source_bind.setdefault(c,rawrow)
 table=[]
 for c in core:
  req(c in base_bind and c in source_bind,"missing canonical binding row for route core")
  req(base_bind[c]==source_bind[c],f"native012 and source binding mismatch for core {c}")
  table.append(base_bind[c])
 binding_bytes=b"".join(table)
 # Encode the complete positive route field; the encoder performs no truncation or normalization.
 rows=[];geometry_summary={};encoded_rows={}
 for sid in (131,133):
  sf=c150["surfaces"][sid]
  listed_vertices=np.asarray(sf["vertices_float32_m"],dtype=np.float32)
  listed_faces=np.asarray(sf["faces"],dtype=np.int64)
  listed_weights=np.asarray(sf["core_weights_float64"],dtype=np.float64)
  listed_gids=np.asarray(sf["global_vertex_ids"],dtype=np.uint64)
  req(listed_vertices.shape==(len(listed_gids),3) and listed_weights.shape==(len(listed_gids),21),"candidate geometry/full field shape mismatch")
  req(np.isfinite(listed_vertices).all() and np.all(listed_weights>0) and np.isfinite(listed_weights).all() and np.max(np.abs(listed_weights.sum(axis=1)-1))<1e-12,"candidate route field is not a positive finite partition")
  used_local=np.asarray(sorted(set(map(int,listed_faces.ravel()))),dtype=np.int64)
  req(len(used_local)>0 and np.all(used_local>=0) and int(used_local.max())<len(listed_vertices),"face-incident vertex filter invalid")
  remap=np.full(len(listed_vertices),-1,dtype=np.int64);remap[used_local]=np.arange(len(used_local),dtype=np.int64)
  faces=remap[listed_faces]
  req(np.all(faces>=0),"face references a removed vertex")
  vertices=listed_vertices[used_local]
  weights=listed_weights[used_local]
  gids=listed_gids[used_local]
  unreferenced_gids=listed_gids[~np.isin(np.arange(len(listed_gids)),used_local)]
  normals=h.vertex_normals(vertices,faces)
  vertices6=np.concatenate((vertices,normals),axis=1).astype("<f4")
  geom=ASSET/f"stable-{sid}-candidate150-source-geometry-field.npz"
  np.savez_compressed(geom,vertices6=vertices6,vertices_float32_m=vertices,faces_local=faces.astype("<u4"),faces_global=np.asarray(sf["faces_global"],dtype="<u8"),global_vertex_ids=gids,listed_unreferenced_global_ids=unreferenced_gids,core_body_indices=np.asarray(core,dtype="<u4"),core_weights_float64=weights,parent_ids=np.asarray(sf["parent_ids"],dtype="<u4"),parent_faces=np.asarray(sf["parent_faces"],dtype="<u4"),parent_input_local_face_indices=np.asarray(sf["parent_input_local_face_indices"],dtype="<u4"),roles=np.asarray(sf["roles"],dtype="<u1"),target_band=np.asarray(sf["target_band"]))
  rows.append(enc.CarrierRow(sid,vertices6,faces,binding_bytes,weights));encoded_rows[sid]=(gids,vertices,weights)
  geometry_summary[str(sid)]={"path":str(geom),"sha256":sha(geom),"candidate150_listed_vertex_count":len(listed_vertices),"face_incident_vertex_count":len(vertices),"dropped_unreferenced_listed_vertex_count":len(unreferenced_gids),"dropped_unreferenced_global_ids":list(map(int,unreferenced_gids)),"face_count":len(faces),"field_core_count":int(weights.shape[1]),"positive_terms_per_vertex_min":int(np.count_nonzero(weights>0,axis=1).min()),"positive_terms_per_vertex_max":int(np.count_nonzero(weights>0,axis=1).max()),"normal_source":"Area-weighted unit normals recomputed from candidate150 source-coordinate triangles; no posed normals.","parent_fields_preserved":True}
 payload,surfaces,proof=enc.encode_carriers(rows,registration_fingerprint=int(hdr[6]),source_digest=bytes(hdr[7]))
 carrier=ASSET/"right-131-133-full21-route-field-carriers.nhtissue";carrier.write_bytes(payload)
 # Verify the payload using the same owner decoder and every row-local term.
 rt=pc._read_nhtiss4(carrier)
 req(rt["raw"]==payload and bytes(rt["raw"][28:64])==base["raw"][28:64],"carrier source registration/digest roundtrip mismatch")
 for cr,record,surface in zip(rows,rt["records"],surfaces):
  row=pc._row_slices(rt,record);got=pc._biceps_row_arrays(row)
  req(row["binding_bytes"]==binding_bytes and np.array_equal(got["vertices6"],np.repeat(cr.vertices6,21,axis=0)),"carrier vertex/binding roundtrip mismatch")
  req(np.array_equal(got["binding_indices"][:,0],np.tile(np.arange(21),len(cr.vertices6))) and np.array_equal(got["weights"][:,0],np.ones(len(got["weights"]),np.float32)),"carrier core-slot roundtrip mismatch")
  req(np.array_equal(got["faces"],cr.faces*21),"carrier face roundtrip mismatch")
  req(len(surface["vertices"])==len(cr.vertices6) and all(len(v)==21 for v in surface["vertices"]),"parent surface term cardinality mismatch")
  for vi,terms in enumerate(surface["vertices"]):
   req([t[:3] for t in terms]==[[0,cr.stable_id,vi*21+j] for j in range(21)],"parent term order/index mismatch")
   req(np.array_equal(np.asarray([t[3] for t in terms],np.float64),cr.weights[vi]),"parent coefficients differ from full field")
 # Shared interface coordinates and all route weights are canonical at both child rows.
 left={int(g):(tuple(map(int,p.view(np.uint32))),tuple(map(float,w))) for g,p,w in zip(*encoded_rows[131])}
 right={int(g):(tuple(map(int,p.view(np.uint32))),tuple(map(float,w))) for g,p,w in zip(*encoded_rows[133])}
 common=set(left)&set(right);req(common and all(left[g]==right[g] for g in common),"shared seam point/field bytes differ")
 terms0=ASSET/"right-parent-surface-terms-source-index-0.json"
 terms0.write_text(json.dumps({"schema":"numi.human.right-fds131-fdp133-full21-source-parent-terms.v1","candidate_path":str(CAND122),"candidate_sha256":hashes[str(CAND122)],"carrier_payload_path":str(carrier),"carrier_payload_sha256":sha(carrier),"surfaces":surfaces},sort_keys=True,separators=(",",":"))+chr(10))
 surfaces1=[{"stable_id":s["stable_id"],"vertices":[[[1,*term[1:]] for term in vertex] for vertex in s["vertices"]]} for s in surfaces]
 terms1=ASSET/"right-parent-surface-terms-native013-source-index-1.json"
 terms1.write_text(json.dumps({"schema":"numi.human.right-fds131-fdp133-full21-source-parent-terms-native013.v1","candidate_path":str(CAND122),"candidate_sha256":hashes[str(CAND122)],"carrier_payload_path":str(carrier),"carrier_payload_sha256":sha(carrier),"source_payload_index":1,"index_shift_assumption":"native012 left shared-field carrier remains source_payload_index 0; right carrier appended at index 1. Recompute if merged source_payload ordering differs.","surfaces":surfaces1},sort_keys=True,separators=(",",":"))+chr(10))
 rest=json.loads(REST125.read_text()); max_rest=max(rest["measured"][mode][sid]["max_um"] for mode in ("fma","separate") for sid in ("131","133"))
 midpoint_rest=json.loads(REST150.read_text())
 f32sep=json.loads(F32SEP150.read_text());f32fma=json.loads(F32FMA150.read_text())
 replay_rows=replay154["candidate150_exact_source_replay"]
 require_rest=max(max_rest,float(midpoint_rest["new_midpoint_max_um"]))
 preparation={"schema":"numi.human.right-fds131-fdp133-full21-route-field-carrier-preparation-155.v1","status":"prepared_assets_not_native_or_anatomically_admitted","gpu_launched":False,"candidate_admission":False,"description":"Candidate150 right FDS/FDP rows with complete positive 21-core route-reference fields, packaged as load-only NHTISS4 carriers plus compact source geometry/ancestry arrays. Route controls are source-authored force-route references, not measured passive tissue attachments; physical MyoSim routes are unchanged.","candidate":{"candidate104_path":str(CAND104),"candidate104_sha256":hashes[str(CAND104)],"structural_parent122_path":str(CAND122),"structural_parent122_sha256":hashes[str(CAND122)],"candidate150_path":str(CAND150),"candidate150_sha256":hashes[str(CAND150)],"source_gate_path":str(SOURCE150),"source_gate_sha256":hashes[str(SOURCE150)],"current8000_f64_path":str(F64_150),"current8000_f64_sha256":hashes[str(F64_150)],"summary_path":str(SUMMARY150),"summary_sha256":hashes[str(SUMMARY150)],"construction_delta_104_to_122_path":str(delta_path),"construction_delta_104_to_122_sha256":sha(delta_path),"construction_delta_122_to_150_path":str(delta150_path),"construction_delta_122_to_150_sha256":sha(delta150_path)},"canonical_field_replay":{"report_path":str(REPLAY154),"report_sha256":hashes[str(REPLAY154)],"evaluator_source_path":str(ROOT/"attempt-095-right-registered-route-field-001.py"),"evaluator_source_sha256":hashes[str(ROOT/"attempt-095-right-registered-route-field-001.py")],"registered_transform_source_path":str(ROOT/"registration-source-b1b410ad.json"),"registered_transform_sha256":hashes[str(ROOT/"registration-source-b1b410ad.json")],"exact_candidate150_coordinates_match_all_full21_weights":True,"stored_float32_position_max_quantization_um":max(float(replay_rows[s]["max_exact_to_float32_coordinate_delta_um"]) for s in ("131","133")),"weight_replay_is_at_exact_rational_source_coordinates":True},"source_gate":{"rows":{s:source150["rows"][s]["self_count"] for s in ("131","133")},"union_pairs":source150["union"]["self_count"],"union_components":source150["union"]["components"],"scope":"Exact source-coordinate predicates on the candidate partition; does not prove native posed admission."},"current8000_ideal":{"rows":{s:f64_150["rows"][s]["self_count"] for s in ("131","133")},"union_pairs":f64_150["union"]["self_count"],"arithmetic":f64_150["arithmetic"],"pose_pack_sha256":f64_150["pose_pack_sha256"],"pose_receipt_sha256":f64_150["pose_receipt_sha256"]},"cpu_float32_diagnostic":{"native_gpu_launched":False,"separate_accumulation":{"row_self_pairs":{s:f32sep["rows"][s]["self_count"] for s in ("131","133")},"row_degenerate_faces":{s:f32sep["rows"][s]["degenerate_count"] for s in ("131","133")},"union_pairs":f32sep["union"]["self_count"],"union_degeneracy":"not computed","report_path":str(F32SEP150),"report_sha256":hashes[str(F32SEP150)]},"fma_accumulation":{"row_self_pairs":{s:f32fma["rows"][s]["self_count"] for s in ("131","133")},"row_degenerate_faces":{s:f32fma["rows"][s]["degenerate_count"] for s in ("131","133")},"union_pairs":f32fma["union"]["self_count"],"union_degeneracy":"not computed","report_path":str(F32FMA150),"report_sha256":hashes[str(F32FMA150)]}},"registered_rest_preflight":{"inherited_candidate122_max_um":float(max_rest),"inherited_report_path":str(REST125),"inherited_report_sha256":hashes[str(REST125)],"new_midpoint_max_um":float(midpoint_rest["new_midpoint_max_um"]),"new_midpoint_report_path":str(REST150),"new_midpoint_report_sha256":hashes[str(REST150)],"combined_bound_um":require_rest,"owner_gate_um":2.0,"scope":"CPU rest registration evidence only; native composition performs its own full-row loader preflight."},"source_inputs":{"native012_child_payload_path":str(BASE),"native012_child_payload_sha256":hashes[str(BASE)],"canonical_source_payload_path":str(SOURCE),"canonical_source_payload_sha256":hashes[str(SOURCE)],"registration_fingerprint":int(hdr[6]),"source_digest_hex":bytes(hdr[7]).hex(),"selected_21_core_binding_records_byte_equal_between_payloads":True,"canonical_core_binding_order":core},"geometry_rows":geometry_summary,"carrier_payload":{"path":str(carrier),"sha256":sha(carrier),"bytes":len(payload),"encoder_proof":proof,"row_count":2,"source_vertex_copies":sum(len(r.vertices6)*21 for r in rows),"all21_positive_influences_per_child_vertex":True,"payload_is_load_only_not_rendered_anatomy":True},"parent_terms":{"source_index0_path":str(terms0),"source_index0_sha256":sha(terms0),"native013_source_index1_path":str(terms1),"native013_source_index1_sha256":sha(terms1),"child_payload_sha256":"not available until root composition","surface_vertex_order_contract":"Terms follow candidate150 compact face-incident local vertex order. The geometry NPZ stores the matching global_vertex_ids and remapped faces; preserve this order or remap term indices."},"shared_interface":{"candidate_global_vertex_count":len(common),"source_position_and_all21_route_coefficients_bit_identical":True,"encoder_checked_same_coordinate_field_bytes":True},"inputs_sha256":hashes,"inputs":hashes,"elapsed_seconds":time.monotonic()-START}
 report=OUT/"preparation.json";report.write_text(json.dumps(preparation,sort_keys=True,indent=2)+chr(10))
 print(json.dumps({"status":preparation["status"],"report":str(report),"report_sha256":sha(report),"carrier_sha256":sha(carrier),"geometry_rows":geometry_summary,"terms1_sha256":sha(terms1),"candidate_sha256":sha(CAND150),"candidate_admission":False,"cpu_f32_union_pairs":{"separate":f32sep["union"]["self_count"],"fma":f32fma["union"]["self_count"]}},sort_keys=True))
if __name__=="__main__":main()
