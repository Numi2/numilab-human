#!/usr/bin/env python3
"""Read-only source-I face/cap screen for an integer-F32 fit report."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
import numpy as np

SCHEMA="numi.h706.integer-f32-source-I-face-screen.v1"
SOURCE_CAP=250.0e-6
AREA_BOUNDS=(0.5,2.0)
COS_MIN=0.5

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def file_pin(path):
    p=Path(path).resolve()
    return {"path":str(p),"sha256":sha(p)}

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def exact_bytes(a):
    return np.ascontiguousarray(a).tobytes()

def same(a,b):
    x,y=np.asarray(a),np.asarray(b)
    return x.dtype==y.dtype and x.shape==y.shape and exact_bytes(x)==exact_bytes(y)

def save_npz_verify(path,payload):
    np.savez_compressed(path,**payload)
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(payload) or len(z.files)!=len(set(z.files)):
            raise ValueError("diagnostic NPZ key mismatch or duplicate central names")
        for k,v in payload.items():
            if not same(v,z[k]): raise ValueError("diagnostic NPZ byte mismatch: "+k)
    with zipfile.ZipFile(path) as z:
        names=[n[:-4] for n in z.namelist() if n.endswith(".npy")]
        if len(names)!=len(set(names)): raise ValueError("duplicate .npy central-directory entry")
    return file_pin(path)

def run(report_path,state_path,output_dir):
    report_path,state_path,output_dir=map(lambda p:Path(p).resolve(),(report_path,state_path,output_dir))
    if output_dir.exists(): raise FileExistsError(output_dir)
    report_raw=report_path.read_text()
    report=json.loads(report_raw)
    state_raw=json.loads(state_path.read_text())
    state_sha=sha(state_path)
    source=state_raw.get("source_surfaces",{}).get("I")
    if not isinstance(source,dict): raise ValueError("current state lacks typed I source pin")
    source_pin={"path":str(Path(source["path"]).resolve()),"sha256":str(source["sha256"])}
    if not Path(source_pin["path"]).is_file() or sha(source_pin["path"])!=source_pin["sha256"]:
        raise ValueError("current typed I source pin mismatch")
    report_input=report.get("input",{})
    debug_pin={"path":str(Path(report_input.get("path","")).resolve()),"sha256":str(report_input.get("sha256",""))}
    if not Path(debug_pin["path"]).is_file() or sha(debug_pin["path"])!=debug_pin["sha256"]:
        raise ValueError("1015 input debug pin mismatch")
    producer_path=Path(__file__).resolve().parent.parent/"fit_integer_f32_quality.py"
    producer_sha=str(report.get("producer_sha256",""))
    if not producer_path.is_file() or sha(producer_path)!=producer_sha:
        raise ValueError("1015 producer source hash mismatch")
    if report.get("source_prefix_positions_unchanged_except_declared_I_updates") is not True:
        raise ValueError("fit report does not bind its declared I-only source update scope")
    records=report.get("variable_records")
    base=np.asarray(report.get("base_variables_f32"),dtype=np.float32)
    final=np.asarray(report.get("final_variables_f32"),dtype=np.float32)
    if not isinstance(records,list) or base.ndim!=2 or final.shape!=base.shape or base.shape!=(len(records),3):
        raise ValueError("1015 F32 variable arrays and record rows are misaligned")
    ids=[i for i,r in enumerate(records) if r.get("kind")=="I_surface"]
    if len(ids)!=6: raise ValueError("expected exactly six declared I surface variables")
    rows=[int(records[i]["source_row"]) for i in ids]
    if len(rows)!=len(set(rows)): raise ValueError("duplicate I source row in fit variable records")
    I=load_npz(Path(source_pin["path"]))
    ip=np.asarray(I["points_world_m_f32"],dtype=np.float32)
    if not np.array_equal(np.asarray(I["points_world_m"],dtype=np.float64).astype(np.float32).view(np.uint32),ip.view(np.uint32)):
        raise ValueError("current I source F64/F32 lanes differ")
    if min(rows)<0 or max(rows)>=len(ip): raise ValueError("fit I source row is out of range")
    updated=ip.copy()
    moves=[]
    for vi,sr in zip(ids,rows):
        rec=records[vi]
        if not np.array_equal(base[vi].view(np.uint32),ip[sr].view(np.uint32)):
            raise ValueError("fit base F32 does not match current1009 I row "+str(sr))
        recbase=np.asarray(rec.get("base"),dtype=np.float32)
        if not np.array_equal(recbase.view(np.uint32),base[vi].view(np.uint32)):
            raise ValueError("variable record base disagrees with base_variables_f32 at "+str(vi))
        candidate=final[vi]
        cap=float(rec.get("cap_m",float("nan")))
        move=float(np.linalg.norm(candidate.astype(np.float64)-ip[sr].astype(np.float64)))
        if not math.isfinite(cap) or cap<0 or cap>SOURCE_CAP+1e-15:
            raise ValueError("declared source I cap invalid at row "+str(sr))
        moves.append({"variable_index":int(vi),"source_row":int(sr),"cap_m":cap,"move_m":move,
                      "base_f32":base[vi].tolist(),"final_f32":candidate.tolist(),
                      "cap_pass":bool(move<=cap+1e-15 and move<=SOURCE_CAP+1e-15)})
        updated[sr]=candidate
    faces=np.asarray(I["faces"],dtype=np.int64)
    if faces.size and (faces.min()<0 or faces.max()>=len(ip)):
        raise ValueError("current I face references invalid point row")
    incident=np.flatnonzero(np.isin(faces,np.asarray(rows,dtype=np.int64)).any(axis=1)).astype(np.int64)
    face_rows=faces[incident]
    used=np.unique(face_rows)
    local=np.searchsorted(used,face_rows)
    old_xyz=ip[face_rows].astype(np.float64)
    new_xyz=updated[face_rows].astype(np.float64)
    old_area=np.cross(old_xyz[:,1]-old_xyz[:,0],old_xyz[:,2]-old_xyz[:,0])
    new_area=np.cross(new_xyz[:,1]-new_xyz[:,0],new_xyz[:,2]-new_xyz[:,0])
    old_norm=np.linalg.norm(old_area,axis=1)
    new_norm=np.linalg.norm(new_area,axis=1)
    ratio=new_norm/np.maximum(old_norm,np.finfo(np.float64).tiny)
    cosine=np.einsum("ij,ij->i",old_area,new_area)/(np.maximum(old_norm,np.finfo(np.float64).tiny)*np.maximum(new_norm,np.finfo(np.float64).tiny))
    cap_pass=all(x["cap_pass"] for x in moves)
    face_pass=bool(np.isfinite(ratio).all() and np.isfinite(cosine).all()
                   and np.all((ratio>=AREA_BOUNDS[0])&(ratio<=AREA_BOUNDS[1]))
                   and np.all(cosine>=COS_MIN))
    passed=cap_pass and face_pass
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=".source-face-screen-",dir=output_dir.parent))
    try:
        source_json=json.dumps(source_pin,sort_keys=True)
        payload={
          "source_point_rows":used.astype(np.int64),
          "incident_source_face_rows":incident.astype(np.int64),
          "incident_faces_local":local.astype(np.int64),
          "source_points_original_f32":ip[used].copy(),
          "source_points_candidate_f32":updated[used].copy(),
          "updated_source_rows":np.asarray(rows,np.int64),
          "updated_variable_indices":np.asarray(ids,np.int64),
          "updated_base_f32":base[ids].copy(),
          "updated_final_f32":final[ids].copy(),
          "initial_area_vectors":old_area,
          "candidate_area_vectors":new_area,
          "face_area_ratio":ratio,
          "face_orientation_cosine":cosine,
          "source_pin_json":np.asarray(source_json),
          "fit_report_json":np.asarray(report_raw),
          "fit_report_path":np.asarray(str(report_path)),
          "fit_report_sha256":np.asarray(sha(report_path)),
          "input_state_path":np.asarray(str(state_path)),
          "input_state_sha256":np.asarray(state_sha),
          "input_debug_path":np.asarray(debug_pin["path"]),
          "input_debug_sha256":np.asarray(debug_pin["sha256"]),
        }
        npz_pin=save_npz_verify(stage/"source-face-screen-input.npz",payload)
        result={
          "schema":SCHEMA,"status":"PASS_SOURCE_I_FACE_CAP_SCREEN" if passed else "REJECT_SOURCE_I_FACE_OR_CAP_SCREEN",
          "fit_report":{"path":str(report_path),"sha256":sha(report_path),"status":report.get("status"),
                        "final_deficit":report.get("final_deficit"),"final_roles":report.get("final_roles")},
          "input_state":{"path":str(state_path),"sha256":state_sha},
          "source_I":source_pin,"fit_input_debug":debug_pin,
          "fit_producer":{"path":str(producer_path),"sha256":producer_sha},
          "updated_I_rows":moves,"updated_source_row_count":len(rows),
          "incident_source_face_count":int(len(incident)),"unique_source_point_count":int(len(used)),
          "area_ratio_bounds":list(AREA_BOUNDS),"min_area_ratio":float(np.min(ratio,initial=float("inf"))),
          "max_area_ratio":float(np.max(ratio,initial=float("-inf"))),
          "min_orientation_cosine":float(np.min(cosine,initial=float("inf"))),
          "max_orientation_cosine":float(np.max(cosine,initial=float("-inf"))),
          "cap_pass":cap_pass,"face_screen_pass":face_pass,
          "diagnostic_npz":npz_pin,
          "screen_scope":"six I source points and every incident current-I triangle only; no volumes loaded",
          "candidate_physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
        }
        (stage/"source-face-screen-report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        os.replace(stage,output_dir)
        return result
    finally:
        if stage.exists(): shutil.rmtree(stage,ignore_errors=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report",type=Path,required=True)
    ap.add_argument("--state",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    print(json.dumps(run(a.report,a.state,a.output),indent=2,sort_keys=True))

if __name__=="__main__": main()

