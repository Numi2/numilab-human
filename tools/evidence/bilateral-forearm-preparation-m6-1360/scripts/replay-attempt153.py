#!/usr/bin/env python3
import hashlib, importlib.util, json, math, pickle, sys
from pathlib import Path
import numpy as np
ROOT=Path("/Users/n/numi-human-fds132-m6-1350")
CAND=ROOT/"attempt-122-right-cap-boundary-directed-contractions-001/result-001/candidate.pkl"
SRC=ROOT/"attempt-095-right-registered-route-field-001.py"
FIELD=ROOT/"attempt-095-right-registered-route-field-001/result-001/right-route-anchor-reference-field-001.npz"
REG=ROOT/"registration-source-b1b410ad.json"
REG_PROOF=ROOT/"attempt-088-registered-route-field-refined-2mm-001/result-001/registration-frame-proof.json"
OUT=ROOT/"attempt-153-candidate122-exact-route-field-replay-001"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    expected_hashes={str(CAND):"aeeaac9444ef873624d7a0c9e667ffa694b0cb1509ee89e3ccdea8acbc665f94",str(FIELD):"7f9c61b6ad804ab859f29fa44fc7d2ea2bc884f7b8468b441749c02648fc6146"}
    for p,h in expected_hashes.items():
        if sha(p)!=h: raise RuntimeError(f"input hash changed: {p}")
    spec=importlib.util.spec_from_file_location("attempt095",SRC)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    z=np.load(FIELD,allow_pickle=False)
    core=list(map(int,z["route_core_body_indices"]))
    controls=np.asarray(z["route_control_world_positions_m"],dtype=np.float64)
    control_core=np.asarray(z["route_control_core_body_indices"],dtype=np.int64)
    reg=json.loads(REG.read_text())
    if sha(REG)!="b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e": raise RuntimeError("authoring registration hash mismatch")
    G=np.asarray(reg["coordinate_system"]["global_source_mm_to_myosim_world_m"],dtype=np.float64)[:3].copy()
    G[:,:3]*=1000.0
    cp=pickle.loads(CAND.read_bytes())
    if list(map(int,cp["core_body_indices"]))!=core: raise RuntimeError("canonical 21-core order mismatch")
    rows={}
    for sid in (131,133):
        s=cp["surfaces"][sid]
        ids=np.asarray(s["global_vertex_ids"],dtype=np.uint64)
        pts=np.asarray(s["vertices_float32_m"],dtype=np.float32)
        stored=np.asarray(s["core_weights_float64"],dtype=np.float64)
        if pts.shape!=(len(ids),3) or stored.shape!=(len(ids),len(core)): raise RuntimeError(f"shape mismatch stable{sid}")
        if len(set(map(int,ids)))!=len(ids): raise RuntimeError(f"duplicate global vertex ID stable{sid}")
        # Match attempt095 exactly: float32 source points -> float64 -> row-vector matrix transform.
        world=pts.astype(np.float64) @ G[:,:3].T + G[:,3]
        replay=m.route_field(world,controls,control_core,core)
        delta=np.abs(stored-replay)
        vertex_max=np.max(delta,axis=1)
        changed=vertex_max!=0.0
        worst=int(np.argmax(vertex_max))
        rows[str(sid)]={
            "vertex_count":int(len(ids)),"scalar_entry_count":int(stored.size),
            "exact_scalar_entry_count":int(np.count_nonzero(stored==replay)),
            "changed_scalar_entry_count":int(np.count_nonzero(stored!=replay)),
            "vertices_with_any_nonidentical_component":int(np.count_nonzero(changed)),
            "max_absolute_component_delta":float(np.max(delta)),
            "rms_absolute_component_delta":float(np.sqrt(np.mean(delta*delta))),
            "worst_global_vertex_id":int(ids[worst]),
            "worst_vertex_max_delta":float(vertex_max[worst]),
            "stored_sum_max_abs_from_one":float(np.max(np.abs(np.sum(stored,axis=1)-1.0))),
            "replay_sum_max_abs_from_one":float(np.max(np.abs(np.sum(replay,axis=1)-1.0))),
            "max_abs_delta_by_decimal_bucket":{str(power):int(np.count_nonzero((delta>0)&(delta<=10.0**power))) for power in (-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5)},
        }
    report={"schema":"numi.human.right-candidate122-exact-attempt095-route-field-replay-151.v1","status":"complete","candidate_sha256":sha(CAND),"route_field_source_sha256":sha(SRC),"route_field_npz_sha256":sha(FIELD),"registration_authoring_sha256":sha(REG),"registration_proof_path":str(REG_PROOF),"registration_proof_sha256":sha(REG_PROOF),"registration_proof_expected_sha256":"d9d2256d1c50d3ffbbcde6718c081bf4bd8e858973628ab81fae42debbeca0fb","canonical_core_order":core,"evaluator":"Exact imported route_field function from attempt095; exact source transform order points.astype(float64) @ G[:,:3].T + G[:,3], controls from pinned 095 NPZ; no changed weights are written.","rows":rows}
    out=OUT/"report.json";out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"path":str(out),"sha256":sha(out),"rows":rows},sort_keys=True))
if __name__=="__main__":main()
