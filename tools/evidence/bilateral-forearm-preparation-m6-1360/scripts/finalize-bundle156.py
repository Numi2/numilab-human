#!/usr/bin/env python3
import hashlib, json, os, pickle, struct
from pathlib import Path
import numpy as np
ROOT=Path("/Users/n/numi-human-fds132-m6-1350")
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
SRC_OUT=ROOT/"attempt-155-right-candidate150-carrier-assets-001"
OUT=ROOT/"attempt-156-right-candidate150-carrier-assets-001"
ASSET=OUT/"assets"
CAND104=ROOT/"attempt-104-right-witness-band-refinement-001/result-001/candidate.pkl"
CK104=ROOT/"attempt-104-right-witness-band-refinement-001/result-001/construction-checkpoint.json"
CAND122=ROOT/"attempt-122-right-cap-boundary-directed-contractions-001/result-001/candidate.pkl"
CONSTR122=ROOT/"attempt-122-right-cap-boundary-directed-contractions-001/result-001/construction-report.json"
SOURCE122=ROOT/"attempt-122-right-cap-boundary-directed-contractions-001/result-001/audit-001/audit-report.json"
CAND150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/candidate.pkl"
CHECK150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/construction-checkpoint.json"
SUMMARY150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/summary.json"
SOURCE150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/source-critical.json"
F64_150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/current8000-f64-critical.json"
F32SEP150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/current8000-f32-separate-critical.json"
F32FMA150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/current8000-f32-fma-critical.json"
REST150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/new-midpoint-rest-check.json"
REPLAY154=ROOT/"attempt-154-exact-source-route-field-replay-001/report.json"
ROUTE_SOURCE=ROOT/"attempt-095-right-registered-route-field-001.py"
ROUTE_FIELD=ROOT/"attempt-095-right-registered-route-field-001/result-001/right-route-anchor-reference-field-001.npz"
REG=ROOT/"registration-source-b1b410ad.json"
REG_PROOF=ROOT/"attempt-088-registered-route-field-refined-2mm-001/result-001/registration-frame-proof.json"
BASE=A/"forearm-shared-field-native-012-diagonal097/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
SOURCE=A/"temporalis164-native-m6-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
REST125=ROOT/"attempt-125-right-rest-registration-check-001/report.json"
HELPER=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
ENCODER=Path("/Users/n/numi-human-left-pq-publication-1344/src/numilab_human/passive_tissue_parent_carriers.py")
GENERATOR=ROOT/"attempt-155-right-candidate150-carrier-assets-001.py"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
    if not v: raise RuntimeError(m)
def write_json(path,value,compact=False):
    path.write_text(json.dumps(value,sort_keys=True,separators=(",",":")) + "\n" if compact else json.dumps(value,sort_keys=True,indent=2)+"\n")
def main():
    require(not OUT.exists(),"preserve any previous attempt156 output")
    OUT.mkdir();ASSET.mkdir()
    names=["right-131-133-full21-route-field-carriers.nhtissue","stable-131-candidate150-source-geometry-field.npz","stable-133-candidate150-source-geometry-field.npz"]
    for name in names:
        src=SRC_OUT/"assets"/name;dst=ASSET/name
        require(src.is_file(),"missing attempt155 serialized asset: "+str(src))
        os.link(src,dst)
    carrier=ASSET/names[0]
    # Rebind copied sidecars to the hardlinked attempt156 carrier; source term data itself is unchanged.
    terms0_src=SRC_OUT/"assets/right-parent-surface-terms-source-index-0.json"
    terms1_src=SRC_OUT/"assets/right-parent-surface-terms-native013-source-index-1.json"
    terms0=json.loads(terms0_src.read_text());terms1=json.loads(terms1_src.read_text())
    for d in (terms0,terms1):
        d["carrier_payload_path"]=str(carrier);d["carrier_payload_sha256"]=sha(carrier)
    terms0_path=ASSET/terms0_src.name;terms1_path=ASSET/terms1_src.name
    write_json(terms0_path,terms0,compact=True);write_json(terms1_path,terms1,compact=True)
    # Keep the emitted NPZs byte-identical to attempt155 and verify their exact candidate mappings.
    cp=pickle.loads(CAND150.read_bytes());geometry={};rows={}
    expected_keys={"vertices6","vertices_float32_m","faces_local","faces_global","global_vertex_ids","listed_unreferenced_global_ids","core_body_indices","core_weights_float64","parent_ids","parent_faces","parent_input_local_face_indices","roles","target_band"}
    for sid in (131,133):
        p=ASSET/f"stable-{sid}-candidate150-source-geometry-field.npz"
        with np.load(p,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
        require(set(d)==expected_keys,f"stable{sid} NPZ schema mismatch")
        sf=cp["surfaces"][sid];listed=np.asarray(sf["vertices_float32_m"],dtype=np.float32);faces_full=np.asarray(sf["faces"],dtype=np.int64)
        used=np.asarray(sorted(set(map(int,faces_full.ravel()))),dtype=np.int64);remap=np.full(len(listed),-1,dtype=np.int64);remap[used]=np.arange(len(used))
        faces=remap[faces_full];gids=np.asarray(sf["global_vertex_ids"],dtype=np.uint64)[used]
        weights=np.asarray(sf["core_weights_float64"],dtype=np.float64)[used]
        require(np.array_equal(d["faces_local"],faces.astype("<u4")),f"stable{sid} compact face mapping mismatch")
        require(np.array_equal(d["global_vertex_ids"],gids),f"stable{sid} compact global ID mapping mismatch")
        require(np.array_equal(d["vertices_float32_m"],listed[used]),f"stable{sid} Float32 position mapping mismatch")
        require(np.array_equal(d["core_weights_float64"],weights),f"stable{sid} 21-term coefficient mapping mismatch")
        require(np.array_equal(d["faces_global"],np.asarray(sf["faces_global"],dtype="<u8")),f"stable{sid} global face mapping mismatch")
        require(np.array_equal(d["global_vertex_ids"][d["faces_local"]],d["faces_global"]),f"stable{sid} local/global face parity failure")
        require(d["core_weights_float64"].shape==(len(gids),21) and np.all(d["core_weights_float64"]>0) and np.isfinite(d["core_weights_float64"]).all(),f"stable{sid} field support invalid")
        require(np.max(np.abs(np.sum(d["core_weights_float64"],axis=1)-1.0))<1e-12,f"stable{sid} coefficient partition failure")
        require(len(d["parent_ids"])==len(d["parent_faces"])==len(d["parent_input_local_face_indices"])==len(d["roles"])==len(d["faces_local"]),f"stable{sid} face ancestry cardinality mismatch")
        geometry[str(sid)]={"path":str(p),"sha256":sha(p),"candidate150_listed_vertex_count":len(listed),"face_incident_vertex_count":len(gids),"dropped_unreferenced_listed_vertex_count":len(listed)-len(gids),"dropped_unreferenced_global_ids":list(map(int,d["listed_unreferenced_global_ids"])),"face_count":len(faces),"field_core_count":21,"positive_terms_per_vertex_min":int(np.count_nonzero(weights>0,axis=1).min()),"positive_terms_per_vertex_max":int(np.count_nonzero(weights>0,axis=1).max()),"npz_keys":sorted(d),"normal_source":"Area-weighted unit normals recomputed from candidate150 source-coordinate triangles; no posed normals.","parent_fields_preserved":True}
        rows[sid]=d
    # Validate source-index-1 term shape and complete coefficients without altering its order.
    require([int(s["stable_id"]) for s in terms1["surfaces"]]==[131,133],"native013 terms row order changed")
    require(int(terms1["source_payload_index"])==1,"right runtime source index is not 1")
    for surf in terms1["surfaces"]:
        sid=int(surf["stable_id"]);w=rows[sid]["core_weights_float64"]
        require(len(surf["vertices"])==len(w),f"stable{sid} term vertex cardinality mismatch")
        # Check all 21 terms per vertex against the emitted compact geometry array.
        for vi,terms in enumerate(surf["vertices"]):
            require(len(terms)==21 and all(int(t[0])==1 and int(t[1])==sid and int(t[2])==vi*21+j and float(t[3])==float(w[vi,j]) for j,t in enumerate(terms)),f"stable{sid} source-index-1 term differs at local vertex {vi}")
    left={int(g):(p.tobytes(),w.tobytes()) for g,p,w in zip(rows[131]["global_vertex_ids"],rows[131]["vertices_float32_m"],rows[131]["core_weights_float64"])}
    right={int(g):(p.tobytes(),w.tobytes()) for g,p,w in zip(rows[133]["global_vertex_ids"],rows[133]["vertices_float32_m"],rows[133]["core_weights_float64"])}
    common=set(left)&set(right);require(common and all(left[g]==right[g] for g in common),"shared interface coordinates or full21 fields differ")
    # Pin every source used to build or verify this bundle, including the actual 095 evaluator.
    inputs=[Path(__file__),GENERATOR,CAND104,CK104,CAND122,CONSTR122,SOURCE122,CAND150,CHECK150,SUMMARY150,SOURCE150,F64_150,F32SEP150,F32FMA150,REST150,REPLAY154,ROUTE_SOURCE,ROUTE_FIELD,REG,REG_PROOF,BASE,SOURCE,REST125,HELPER,ENCODER]
    for p in inputs:require(p.is_file(),"missing pinned input "+str(p))
    hashes={str(p):sha(p) for p in inputs}
    require(hashes[str(CAND104)]=="28ab429feb10615a5cfbd2345148d6b0619a7e75b41301b3f86f0a244202ad6d","candidate104 pin changed")
    require(hashes[str(CAND122)]=="aeeaac9444ef873624d7a0c9e667ffa694b0cb1509ee89e3ccdea8acbc665f94","candidate122 pin changed")
    require(hashes[str(CAND150)]=="8607d93a055a2045e823f80b4e04b6d6dcd6f5d8bfa95b442485a35b555918b1","candidate150 pin changed")
    source=json.loads(SOURCE150.read_text());ideal=json.loads(F64_150.read_text());sep=json.loads(F32SEP150.read_text());fma=json.loads(F32FMA150.read_text());rest_old=json.loads(REST125.read_text());rest_new=json.loads(REST150.read_text());replay=json.loads(REPLAY154.read_text())
    require(source["candidate_sha256"]==hashes[str(CAND150)] and source["union"]["self_count"]==0 and all(source["rows"][str(s)]["self_count"]==0 for s in (131,133)),"candidate150 source gate not passing")
    require(ideal["candidate_sha256"]==hashes[str(CAND150)] and ideal["union"]["self_count"]==0 and all(ideal["rows"][str(s)]["self_count"]==0 for s in (131,133)),"candidate150 ideal current8000 gate not passing")
    require(replay["candidate150_sha256"]==hashes[str(CAND150)] and all(replay["candidate150_exact_source_replay"][str(s)]["changed_scalar_entry_count"]==0 for s in ("131","133")),"candidate150 exact route replay not passing")
    require(json.loads(SUMMARY150.read_text())["candidate_admission"] is False,"candidate150 was marked admitted")
    oldmax=max(float(rest_old["measured"][m][s]["max_um"]) for m in ("fma","separate") for s in ("131","133"))
    newmax=float(rest_new["new_midpoint_max_um"])
    raw=BASE.read_bytes();hdr=struct.unpack_from("<8s6I32s",raw,0)
    preparation={"schema":"numi.human.right-fds131-fdp133-full21-route-field-carrier-preparation-156.v1","status":"prepared_assets_not_native_or_anatomically_admitted","gpu_launched":False,"candidate_admission":False,"description":"Candidate150 right FDS/FDP rows packaged as load-only NHTISS4 carriers and compact source geometry/ancestry arrays. The full positive 21-core inferred route-reference field is retained. Route controls are authored force-route references, not measured passive tissue attachments; physical MyoSim routes are unchanged.","candidate":{"candidate104_path":str(CAND104),"candidate104_sha256":hashes[str(CAND104)],"structural_parent122_path":str(CAND122),"structural_parent122_sha256":hashes[str(CAND122)],"candidate150_path":str(CAND150),"candidate150_sha256":hashes[str(CAND150)],"source_gate_path":str(SOURCE150),"source_gate_sha256":hashes[str(SOURCE150)],"current8000_f64_path":str(F64_150),"current8000_f64_sha256":hashes[str(F64_150)],"summary_path":str(SUMMARY150),"summary_sha256":hashes[str(SUMMARY150)]},"canonical_field_replay":{"report_path":str(REPLAY154),"report_sha256":hashes[str(REPLAY154)],"evaluator_source_path":str(ROUTE_SOURCE),"evaluator_source_sha256":hashes[str(ROUTE_SOURCE)],"registered_transform_source_path":str(REG),"registered_transform_sha256":hashes[str(REG)],"exact_candidate150_coordinates_match_all_full21_weights":True,"stored_float32_position_max_quantization_um":max(float(replay["candidate150_exact_source_replay"][str(s)]["max_exact_to_float32_coordinate_delta_um"]) for s in ("131","133")),"weight_replay_is_at_exact_rational_source_coordinates":True},"source_gate":{"rows":{str(s):source["rows"][str(s)]["self_count"] for s in (131,133)},"union_pairs":source["union"]["self_count"],"union_components":source["union"]["components"],"scope":"Exact source-coordinate predicates; not a native posed admission."},"current8000_ideal":{"rows":{str(s):ideal["rows"][str(s)]["self_count"] for s in (131,133)},"union_pairs":ideal["union"]["self_count"],"arithmetic":ideal["arithmetic"],"pose_pack_sha256":ideal["pose_pack_sha256"],"pose_receipt_sha256":ideal["pose_receipt_sha256"]},"cpu_float32_diagnostic":{"native_gpu_launched":False,"separate_accumulation":{"row_self_pairs":{str(s):sep["rows"][str(s)]["self_count"] for s in (131,133)},"row_degenerate_faces":{str(s):sep["rows"][str(s)]["degenerate_count"] for s in (131,133)},"union_pairs":sep["union"]["self_count"],"union_degeneracy":"not computed","report_path":str(F32SEP150),"report_sha256":hashes[str(F32SEP150)]},"fma_accumulation":{"row_self_pairs":{str(s):fma["rows"][str(s)]["self_count"] for s in (131,133)},"row_degenerate_faces":{str(s):fma["rows"][str(s)]["degenerate_count"] for s in (131,133)},"union_pairs":fma["union"]["self_count"],"union_degeneracy":"not computed","report_path":str(F32FMA150),"report_sha256":hashes[str(F32FMA150)]}},"registered_rest_preflight":{"inherited_candidate122_max_um":oldmax,"inherited_report_path":str(REST125),"inherited_report_sha256":hashes[str(REST125)],"new_midpoint_max_um":newmax,"new_midpoint_report_path":str(REST150),"new_midpoint_report_sha256":hashes[str(REST150)],"combined_bound_um":max(oldmax,newmax),"owner_gate_um":2.0,"scope":"CPU rest registration evidence only; native composition performs its own full-row loader preflight."},"source_inputs":{"native012_child_payload_path":str(BASE),"native012_child_payload_sha256":hashes[str(BASE)],"canonical_source_payload_path":str(SOURCE),"canonical_source_payload_sha256":hashes[str(SOURCE)],"registration_fingerprint":int(hdr[6]),"source_digest_hex":bytes(hdr[7]).hex()},"geometry_rows":geometry,"carrier_payload":{"path":str(carrier),"sha256":sha(carrier),"bytes":carrier.stat().st_size,"row_count":2,"source_vertex_copies":sum(len(v["vertices6"])*21 for v in rows.values()),"all21_positive_influences_per_child_vertex":True,"payload_is_load_only_not_rendered_anatomy":True,"serialization_reused_from_attempt155":True},"parent_terms":{"source_index0_path":str(terms0_path),"source_index0_sha256":sha(terms0_path),"native013_source_index1_path":str(terms1_path),"native013_source_index1_sha256":sha(terms1_path),"source_index1":1,"child_payload_sha256":"not available until root composition","surface_vertex_order_contract":"Terms follow candidate150 compact face-incident local vertex order; the geometry NPZ arrays carry matching global_vertex_ids and remapped faces."},"shared_interface":{"candidate_global_vertex_count":len(common),"source_position_and_all21_route_coefficients_bit_identical":True},"inputs_sha256":hashes,"inputs":hashes}
    report=OUT/"preparation.json";write_json(report,preparation)
    print(json.dumps({"status":preparation["status"],"report":str(report),"report_sha256":sha(report),"candidate_sha256":hashes[str(CAND150)],"carrier_sha256":sha(carrier),"geometry_rows":geometry,"source_index1_terms_path":str(terms1_path),"source_index1_terms_sha256":sha(terms1_path),"source_index0_terms_sha256":sha(terms0_path),"shared_global_vertices":len(common),"cpu_f32_union_pairs":{"separate":sep["union"]["self_count"],"fma":fma["union"]["self_count"]}},sort_keys=True))
if __name__=="__main__":main()
