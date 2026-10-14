import hashlib,json,sys
from pathlib import Path
import numpy as np
BASE=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
WORK=BASE/"pons-fourth-ventricle-reconstruction-001"
D=WORK/"cord-local-clearance-candidate-003"
SRC=BASE/"forearm492-head016-vessel-seam-integrated-source-002/resting-thorax.nhanatomy"
NPZ=D/"candidate-01.npz"; GATE=D/"report.json"
OUTDIR=WORK/"cord-local-clearance-candidate-compile-001"
OUT=OUTDIR/"resting-thorax-candidate.nhanatomy"; REPORT=OUTDIR/"compile-report.json"
PARSER=Path("/Users/n/numi-human-pons-floor-compiler-20261014/src/numilab_human/resting_anatomy_interface_patch.py")
DECODER=Path("/Users/n/numi-human-pons-floor-compiler-20261014/src/numilab_human/resting_pleura_proxy.py")
ENCODER=Path("/Users/n/numi-human-pons-floor-compiler-20261014/src/numilab_human/resting_intercostal_mesh_arrangement.py")
sys.path.insert(0,"/Users/n/numi-human-pons-floor-compiler-20261014/src")
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
pins={"source_nha":"2f8af1031b3c5b8cc26866f32935c9b695127a2f357732e566be3f14fdc2666d","candidate_npz":"d9798684d4412fd97093ea056d009e2594cbaa40855cab2e1e82b2b84c5bdab6","candidate_gate_report":"3e04eb8ee9fcc653951b245f285d3a99e84c2b299a1605cb9694018ded5ff33f","parser":"aad523b4606ff17e7aa3ada0ac9c9c519f85597393cd4ea86492852e5b632add","decoder":"d874f599b8aeb9e9f02318fbae30ed5bf017920170d41c8b775cc3a9f792d683","encoder":"fdedd40342ce9d8b14b7497dd1a188425aae2d28b2e623a24fb1a6f81f375447"}
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for p,k in [(SRC,"source_nha"),(NPZ,"candidate_npz"),(GATE,"candidate_gate_report"),(PARSER,"parser"),(DECODER,"decoder"),(ENCODER,"encoder")]:
 if sha(p)!=pins[k]: raise SystemExit(f"pin mismatch {k}: {sha(p)}")
gate=json.loads(GATE.read_text())
if gate.get("status")!="failed" or len(gate.get("candidates",[]))!=1: raise SystemExit("unexpected gate report")
c=gate["candidates"][0]
if not (c["closed"] and c["self_count"]==0 and c["pair_passed"] and c["unexpected_pair_count"]==0 and c["face_arrays_unchanged"] and c["cap_vertex_bytes_unchanged"] and all(x["status"]=="checked_exact_rational_metres" and not x["inside_component_witnesses"] for x in c["containment"])):
 raise SystemExit("candidate geometry gate incomplete")
with np.load(NPZ,allow_pickle=False) as z:
 xyz6=np.asarray(z["vertices6_f32"],dtype="<f4").copy()
 faces=np.asarray(z["faces_i64"],dtype=np.int64).copy()
 slots=np.asarray(z["changed_source_vertex_ids_i64"],dtype=np.int64).copy()
 cfaces=np.asarray(z["changed_source_face_ids_i64"],dtype=np.int64).copy()
raw=SRC.read_bytes(); header,records,rows=_parse_payload(raw)
if len(rows)!=525 or 323 not in rows: raise SystemExit("unexpected ABI5 row inventory")
old=rows[323]
if (old["body_index"],old["layer"])!=(23,1): raise SystemExit("row323 owner/type mismatch")
if old["vertices6"].shape!=xyz6.shape or not np.array_equal(old["faces"],faces): raise SystemExit("candidate topology differs from current row323")
if not np.isfinite(xyz6).all() or len(slots)!=c["changed_source_slots"] or len(cfaces)!=c["changed_source_faces"]: raise SystemExit("candidate arrays/ancestry invalid")
new=dict(old); new["vertices6"]=xyz6
payload,proof=encode_payload_with_row_replacements(raw,{323:new})
OUTDIR.mkdir(parents=True,exist_ok=True)
if OUT.exists() or REPORT.exists(): raise SystemExit("refusing to overwrite serializer output")
OUT.write_bytes(payload)
oh,orec,after=_parse_payload(payload)
if list(after)!=list(rows): raise SystemExit("stable-ID order changed")
untouched=[]
for sid,a in rows.items():
 if sid==323: continue
 b=after[sid]
 if a["record"]!=b["record"] or a["vertices6"].tobytes()!=b["vertices6"].tobytes() or a["faces"].tobytes()!=b["faces"].tobytes(): raise SystemExit(f"unmodified row changed {sid}")
 untouched.append(sid)
if len(untouched)!=524 or not np.array_equal(after[323]["faces"],old["faces"]) or not np.array_equal(after[323]["vertices6"],xyz6): raise SystemExit("final decoded candidate mismatch")
report={"schema":"numi.human.pons_cord_local_clearance_compile.v1","status":"source_candidate_serialized","candidate_admission":False,"native_or_motion_confirmation":False,"physical_mass_or_state_added":False,"scope":"ABI5 row323 vertices only; ordered faces and owner retained; all other 524 decoded rows byte-identical","pins":{**pins,"serializer_script":sha(Path(__file__)),"output_nha":sha(OUT)},"output":str(OUT),"candidate_npz":str(NPZ),"stable_id":323,"body_index":23,"layer":1,"ordered_faces_unchanged":True,"changed_vertex_slots":len(slots),"changed_geometry_support_faces":len(cfaces),"floor_cap_vertices_unchanged":True,"abi5_encode_proof":proof,"decoded_rows":len(after),"untouched_rows":len(untouched),"untouched_ids_sha256":hashlib.sha256(np.asarray(untouched,dtype="<u4").tobytes()).hexdigest(),"source_gate":{"self_count":c["self_count"],"self_pairs_sha256":c["self_pair_sha256"],"unexpected_cord_pairs":c["unexpected_pair_count"],"unexpected_pairs_sha256":c["unexpected_pair_sha256"],"containment":c["containment"],"max_displacement_m":c["max_source_vertex_delta_m"],"ambient_jacobian_lower_bound":c["ambient_jacobian_lower_bound"],"signed_volume_before_m3":c["signed_volume_before_m3"],"signed_volume_after_m3":c["signed_volume_after_m3"],"signed_volume_delta_m3":c["signed_volume_delta_m3"]},"limitation":"source serialization only; native and retained-pose confirmation remains required before admission"}
REPORT.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print(json.dumps({"output":str(OUT),"output_sha256":sha(OUT),"report":str(REPORT),"report_sha256":sha(REPORT),"script_sha256":sha(Path(__file__)),"untouched_rows":len(untouched),"proof":proof},sort_keys=True))
