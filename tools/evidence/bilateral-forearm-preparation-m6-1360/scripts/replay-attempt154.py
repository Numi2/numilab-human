#!/usr/bin/env python3
import hashlib, importlib.util, json, pickle, sys
from pathlib import Path
import numpy as np
ROOT=Path("/Users/n/numi-human-fds132-m6-1350")
CAND122=ROOT/"attempt-122-right-cap-boundary-directed-contractions-001/result-001/candidate.pkl"
CAND150=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001/result-001/candidate.pkl"
SRC=ROOT/"attempt-095-right-registered-route-field-001.py"
FIELD=ROOT/"attempt-095-right-registered-route-field-001/result-001/right-route-anchor-reference-field-001.npz"
REG=ROOT/"registration-source-b1b410ad.json"
REG_PROOF=ROOT/"attempt-088-registered-route-field-refined-2mm-001/result-001/registration-frame-proof.json"
OUT=ROOT/"attempt-154-exact-source-route-field-replay-001"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replay(candidate_path, expected_sha):
    if sha(candidate_path)!=expected_sha: raise RuntimeError(f"candidate hash mismatch {candidate_path}")
    cp=pickle.loads(candidate_path.read_bytes())
    out={}
    for sid in (131,133):
        s=cp["surfaces"][sid]
        exact=s["vertices_exact_m"]
        ids=np.asarray(s["global_vertex_ids"],dtype=np.uint64)
        rounded=np.asarray(s["vertices_float32_m"],dtype=np.float32)
        stored=np.asarray(s["core_weights_float64"],dtype=np.float64)
        exact64=np.asarray([[float(x) for x in p] for p in exact],dtype=np.float64)
        quant_delta=np.abs(exact64-rounded.astype(np.float64))
        world=exact64 @ G[:,:3].T + G[:,3]
        expected=route_field(world,controls,control_core,core)
        delta=np.abs(stored-expected); vmax=np.max(delta,axis=1)
        i=int(np.argmax(vmax))
        out[str(sid)]={"vertex_count":len(ids),"scalar_entry_count":int(stored.size),"exact_scalar_entry_count":int(np.count_nonzero(stored==expected)),"changed_scalar_entry_count":int(np.count_nonzero(stored!=expected)),"vertices_with_any_nonidentical_component":int(np.count_nonzero(vmax>0)),"max_abs_weight_delta":float(np.max(delta)),"rms_abs_weight_delta":float(np.sqrt(np.mean(delta*delta))),"worst_global_vertex_id":int(ids[i]),"worst_vertex_max_abs_delta":float(vmax[i]),"max_exact_to_float32_coordinate_delta_um":float(np.max(np.linalg.norm(quant_delta,axis=1))*1e6),"vertices_with_exact_vs_float32_coordinate_difference":int(np.count_nonzero(np.any(quant_delta!=0,axis=1))),"sum_error_max":float(np.max(np.abs(np.sum(stored,axis=1)-1.0)))}
    return out
spec=importlib.util.spec_from_file_location("attempt095",SRC);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
z=np.load(FIELD,allow_pickle=False);core=list(map(int,z["route_core_body_indices"]));controls=np.asarray(z["route_control_world_positions_m"],dtype=np.float64);control_core=np.asarray(z["route_control_core_body_indices"],dtype=np.int64)
reg=json.loads(REG.read_text());G=np.asarray(reg["coordinate_system"]["global_source_mm_to_myosim_world_m"],dtype=np.float64)[:3].copy();G[:,:3]*=1000.0
if sha(REG)!="b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e":raise RuntimeError("registration hash mismatch")
if sha(REG_PROOF)!="d9d2256d1c50d3ffbbcde6718c081bf4bd8e858973628ab81fae42debbeca0fb":raise RuntimeError("registration proof hash mismatch")
route_field=m.route_field
report={"schema":"numi.human.right-exact-source-coordinate-route-replay-154.v1","status":"complete","evaluator":"Exact attempt095 route_field; exact rational candidate source coordinates converted to Float64 before the pinned source-m-to-registered-world transform. This is distinct from the rounded Float32 payload coordinates used for native storage.","route_field_source_sha256":sha(SRC),"route_field_npz_sha256":sha(FIELD),"registration_authoring_sha256":sha(REG),"registration_proof_sha256":sha(REG_PROOF),"candidate122_sha256":sha(CAND122),"candidate150_sha256":sha(CAND150),"candidate122_exact_source_replay":replay(CAND122,"aeeaac9444ef873624d7a0c9e667ffa694b0cb1509ee89e3ccdea8acbc665f94"),"candidate150_exact_source_replay":replay(CAND150,"8607d93a055a2045e823f80b4e04b6d6dcd6f5d8bfa95b442485a35b555918b1")}
out=OUT/"report.json";out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
print(json.dumps({"path":str(out),"sha256":sha(out),"candidate122":report["candidate122_exact_source_replay"],"candidate150":report["candidate150_exact_source_replay"]},sort_keys=True))
