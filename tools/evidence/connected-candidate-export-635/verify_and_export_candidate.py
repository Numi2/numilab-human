#!/usr/bin/env python3
"""Verify a same-topology 006 connected fit, then reuse its existing asset compiler.

Run via physical-transfer-209/run_geometry_serial.py. This prepares existing
registered-volume CSVs only. Exact contact, conservative native mass admission,
and motion qualification remain independent mandatory stages.
"""
from __future__ import annotations
import argparse,hashlib,json,pathlib,subprocess,sys,time
import numpy as np
P=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
BASE=P/"reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz"
BASE_HASH="888eb12450d0e69fb0ad3f5998159ab7a0a20d3e70b13c42ba416c61cbbede63"
EXPORTER=P/"operational-reference-map-501/export_refined_coarse_registered_volume_433.py"
EXPORTER_HASH="ad63b10701c42b37b1aa6aa79e0dc0c9f7ac87cdea7622f65582f9d1a63af2b1"
IMMUTABLE_KEYS=("volume_tetrahedra","material_index","geometric_donor_body","anatomical_region_id",
"stable_source_face_id","parent_volume_node_row","child_parent_tet_row","full86_weights",
"coarse_boundary_faces_O","coarse_boundary_faces_I","coarse_boundary_faces_D",
"refinement_node_source_offsets","refinement_node_source_rows","refinement_node_source_weights",
"refinement_child_source_tet_rows","refinement_child_source_measure_fractions","pinned_volume_node_rows")
def digest(f):
    h=hashlib.sha256()
    with pathlib.Path(f).open("rb") as s:
        for b in iter(lambda:s.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def same_bytes(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def six_volumes(points,tets):
    a,b,c,d=np.moveaxis(np.asarray(points,dtype=np.float64)[tets],1,0)
    return np.einsum("ij,ij->i",np.cross(b-a,c-a),d-a)
def check_candidate(base,candidate):
    for k in IMMUTABLE_KEYS:
        if k not in candidate or not same_bytes(base[k],candidate[k]):
            raise ValueError("changed immutable topology, identity, mass ancestry, binding, or protected-row field: "+k)
    x=candidate["points_current_world_m_f32"]
    ref=candidate["points_reference_world_m_f64"]
    if x.dtype!=np.dtype("float32") or ref.dtype!=np.dtype("float64") or x.shape!=base["points_current_world_m_f32"].shape or ref.shape!=x.shape:
        raise ValueError("invalid current/reference coordinate shape or scalar policy")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(ref)):
        raise ValueError("nonfinite current/reference coordinates")
    if not same_bytes(x,ref.astype(np.float32)):
        raise ValueError("current coordinates are not the emitted Float32 operational-reference alias")
    for k in ("points_donor_local_m_f64","target_current_world_m"):
        if k in candidate and not np.all(np.isfinite(candidate[k])):
            raise ValueError("nonfinite derived coordinate field: "+k)
    pins=np.asarray(base["pinned_volume_node_rows"],dtype=np.int64)
    if not same_bytes(x[pins],base["points_current_world_m_f32"][pins]):
        raise ValueError("protected back/head-ring row coordinates changed")
    t=np.asarray(candidate["volume_tetrahedra"],dtype=np.int64)
    v=six_volumes(x,t)
    invalid=np.flatnonzero(~np.isfinite(v)|(v<=0.0))
    if len(invalid):
        raise ValueError("invalid emitted tetrahedra: "+repr(invalid[:20].tolist())+" count="+str(len(invalid)))
    moved=np.flatnonzero(np.any(x!=base["points_current_world_m_f32"],axis=1))
    return {"positive_tetrahedra":len(t),"minimum_six_volume_m3":float(v.min()),
            "moved_node_count":len(moved),"maximum_node_displacement_m":float(np.max(np.linalg.norm(x.astype(np.float64)-base["points_current_world_m_f32"],axis=1))),
            "protected_rows_unchanged":len(pins),"immutable_fields_checked":list(IMMUTABLE_KEYS)}
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-directory",type=pathlib.Path,required=True)
    ap.add_argument("--out",type=pathlib.Path,required=True)
    ap.add_argument("--check-only",action="store_true")
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    record={"physical_admission":False,"runtime":0,"accepted_steps":0,"started_unix":time.time()}
    try:
        if digest(BASE)!=BASE_HASH or digest(EXPORTER)!=EXPORTER_HASH:
            raise ValueError("baseline or reused exporter changed")
        path=args.candidate_directory/"candidate.npz"
        pin=digest(path);record["candidate"]={"path":str(path),"sha256":pin}
        with np.load(BASE,allow_pickle=False) as base,np.load(path,allow_pickle=False) as candidate:
            record["cheap_connected_volume_gate"]=check_candidate(base,candidate)
        record["status"]="PASS_CHEAP_CONNECTED_VOLUME_GATE"
        if not args.check_only:
            inputs={"--candidate-npz":path,"--candidate-node":args.candidate_directory/"candidate.node",
                    "--candidate-node-map":args.candidate_directory/"coarse-node-to-fine-surface.csv",
                    "--inserted-node-ancestry":args.candidate_directory/"inserted-node-ancestry.csv"}
            record["input_sha256"]={str(f):digest(f) for f in inputs.values()}
            argv=[sys.executable,str(EXPORTER)]
            for k,f in inputs.items():argv.extend((k,str(f)))
            argv.extend(("--capture",str(P/"native-physical-parent-mass-330/native-source-identity-control-003"),
                         "--out",str(args.out/"registered-assets"),
                         "--maximum-mass-transport-distance-m","0.019353","--candidate-operational-reference"))
            record["argv"]=argv;record["exporter_sha256"]=EXPORTER_HASH
            started=time.monotonic()
            with (args.out/"export.stdout").open("wb") as so,(args.out/"export.stderr").open("wb") as se:
                result=subprocess.run(argv,stdout=so,stderr=se)
            record["export_wall_s"]=time.monotonic()-started;record["export_exit_code"]=result.returncode
            if result.returncode:raise RuntimeError("existing asset compiler rejected candidate; see export.stderr")
            receipt=json.loads((args.out/"registered-assets/receipt.json").read_text())
            for f,h in receipt["output_sha256"].items():
                if digest(f)!=h:raise ValueError("asset output changed: "+f)
            if receipt.get("registration_domain")!="candidateOperationalReference" or receipt.get("physical_admission") is not False:
                raise ValueError("wrong source-only operational-reference asset contract")
            record["verified_asset_outputs"]=len(receipt["output_sha256"])
            record["status"]="PASS_REGISTERED_ASSET_PREPARATION_ONLY"
            if any(digest(f)!=h for f,h in record["input_sha256"].items()):
                raise ValueError("input changed during export")
        if digest(path)!=pin:raise ValueError("candidate changed during gate/export")
        code=0
    except Exception as e:
        record["status"]="REJECTED";record["error_type"]=type(e).__name__;record["error"]=str(e);code=1
    record["finished_unix"]=time.time()
    (args.out/"result.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print(json.dumps(record,sort_keys=True));return code
if __name__=="__main__":raise SystemExit(main())
