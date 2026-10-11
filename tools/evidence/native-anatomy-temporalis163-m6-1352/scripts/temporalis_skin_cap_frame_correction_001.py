#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

W=Path("/Users/n/numi-human-exact-normalization-m6-1348")
R=Path("/Users/n/numi-human-retained-delivery-20261009")
A=R/"anatomy-completion-1276"
BASE=W/"reports/temporalis-clearance-002"
OUT=BASE/"skin-cap-frame-correction-001/run-002/result-001"
TARGET=BASE/"skin-cap-center-audit-017/run-001/result-001/temporary-classifier-target.npz"
OLD_REPORT=BASE/"skin-cap-center-audit-017/run-001/result-001/report.json"
CAND=BASE/"run-001/result-001/stable-163-right-temporalis-bone-shell-offset-10um.npz"
PAIRS=BASE/"run-001/result-001/pairs-by-pose.json"
OLD_PACK=R/"anatomy-completion-1276/right-edl-ehl-native-mtp-two-001/baseline/native-run/accepted-geometry/step-0.mrvpack"
OLD_RECEIPT=OLD_PACK.with_suffix(".receipt.json")
CURRENT=A/"temporalis163-native-m6-001/baseline/native-run/accepted-geometry"
PACK=CURRENT/"step-0.mrvpack"
RECEIPT=CURRENT/"step-0.receipt.json"
PYTHON="/Users/n/numi-human-prep-venv-20261005/bin/python3.13"
sys.path.insert(0,str(W/"src"))
from numilab_human import common_atlas_skin_clearance as ca
from numilab_human import cardiac_cavity_intersections as ci

def req(ok,msg):
    if not ok: raise RuntimeError(msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
    return h.hexdigest()
def bounds(a):
    a=np.asarray(a,dtype=np.float64)
    return {"min_m":a.min(0).tolist(),"max_m":a.max(0).tolist()}
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    progress=OUT/"progress.jsonl"
    receipt=json.loads(RECEIPT.read_text())
    req(receipt.get("physical_endpoint")=="accepted" and int(receipt.get("accepted_step",-1))==0,
        "current native M6 step0 is not an accepted endpoint")
    pins={str(p):sha(p) for p in [TARGET,OLD_REPORT,CAND,PAIRS,OLD_PACK,OLD_RECEIPT,PACK,RECEIPT,
        W/"src/numilab_human/common_atlas_skin_clearance.py",
        W/"src/numilab_human/cardiac_cavity_intersections.py"]}
    req(pins[str(TARGET)]=="6f875343aa7d4f4a51e46de3ff05276de7ad7373a3c8d42397d9e9edb4c916f3","temporary closure target pin changed")
    req(pins[str(OLD_PACK)]=="d92a1e1467ee6b75025990f2fb46508a6fad54effd9b97081b261897180c2dde","original candidate pack pin changed")
    req(pins[str(OLD_RECEIPT)]=="6899c6f46aef0bc0d93eaa990a29bdd53aa6e8d1459248401789ab83d3087968","original candidate receipt pin changed")
    old=json.loads(OLD_REPORT.read_text())
    req(old.get("temporary_classifier_target_artifact",{}).get("sha256")==pins[str(TARGET)],"old target report and target artifact do not match")
    xyz,surfaces,pack_info=ca._pack_surfaces(PACK,{(51007,1)})
    req((51007,1) in surfaces,"current native pack omits runtime skin surface 51007:1")
    skin_faces=np.asarray(surfaces[(51007,1)]["faces"],dtype=np.int64)
    with np.load(TARGET,allow_pickle=False) as z:
        target_v=np.asarray(z["vertices_f32"],dtype="<f4")
        target_f=np.asarray(z["faces_i64"],dtype=np.int64)
    source_face_count=int(old["runtime_skin_contract"]["source_faces"])
    req(source_face_count==109211 and len(skin_faces)==source_face_count,"skin source face population changed")
    req(target_f.shape==(109326,3) and np.array_equal(target_f[:source_face_count].shape,skin_faces.shape),"temporary target source-face section changed")
    target_source_tri=target_v[target_f[:source_face_count]]
    current_skin_tri=xyz[skin_faces]
    delta=target_source_tri.astype(np.float64)-current_skin_tri.astype(np.float64)
    max_abs=float(np.max(np.abs(delta)))
    rms=float(np.sqrt(np.mean(delta*delta)))
    req(max_abs==0.0 and rms==0.0 and np.array_equal(target_source_tri,current_skin_tri),
        "temporary authoring target source skin differs from current native skin frame")
    cap_f=target_f[source_face_count:]
    cap_vids=np.unique(cap_f)
    cap_points=target_v[cap_vids]
    skin_vids=np.unique(skin_faces)
    skin_points=xyz[skin_vids]
    cap_to_skin=np.asarray(cKDTree(skin_points).query(cap_points,k=1)[0],dtype=np.float64)
    with np.load(CAND,allow_pickle=False) as z:
        local=np.asarray(z["positions_m"],dtype="<f4")
        world=np.asarray(z["step0_world_f32"],dtype="<f4")
        cand_f=np.asarray(z["faces"],dtype=np.int64)
    req(world.shape==local.shape and len(cand_f)==10994,"pinned temporalis source shape changed")
    pair_doc=json.loads(PAIRS.read_text())
    row=next(x for x in pair_doc if int(x["step"])==0)
    skin_pairs=np.asarray(row["pairs"]["51007:1"],dtype=np.int64)
    contact_face_rows=np.unique(skin_pairs[:,0])
    contact_vids=np.unique(cand_f[contact_face_rows])
    contact_world=np.asarray(world[contact_vids],dtype="<f4")
    gap=np.maximum(np.maximum(cap_points.min(0)-contact_world.max(0),contact_world.min(0)-cap_points.max(0)),0.0)
    nearest=float(cKDTree(contact_world).query(cap_points,k=1)[0].min())
    world_records=ci._records([ci.float32_point_lattice_key(tuple(float(x) for x in p)) for p in world],cand_f.tolist())
    cap_records0=ci._records([ci.float32_point_lattice_key(tuple(float(x) for x in p)) for p in target_v],cap_f.tolist())
    cap_records=[(r[0],r[1],r[2],int(i),r[4]) for i,r in enumerate(cap_records0)]
    cap_audit=ci._audit_record_pairs(
        ((a,b) for a,b in ci._query_surface_aabb(cap_records,ci._prepare_surface_aabb(world_records))),
        same_surface=False)
    wrong_frame={"candidate_positions_m_bounds":bounds(local),"cap_bounds":bounds(cap_points),
        "candidate_positions_m_frame":"registered row-local coordinates; must not be compared with world-frame cap vertices"}
    req(float(local[:,2].min())>1.0 and float(world[:,2].max())<0.3,
        "expected local/world frame discriminator did not hold")
    req(cap_audit["count"]==0,"cap fans exactly intersect the correctly transformed full candidate")
    report={"schema":"offline.temporalis.skin-cap-frame-correction.v1",
        "status":"frame_verified_corrected_distance_receipt",
        "supersedes":{"report_path":str(OLD_REPORT),"report_sha256":pins[str(OLD_REPORT)],
            "incorrect_distance_field":"repair_neighborhood_exclusion.cap_contact_aabb_lower_bound_m",
            "incorrect_reported_value_m":old["repair_neighborhood_exclusion"]["cap_contact_aabb_lower_bound_m"],
            "cause":"the prior audit compared world-frame scalp cap vertices with row-local CAND.positions_m (z about 1.47 to 1.59 m), instead of CAND.step0_world_f32"},
        "coordinate_frame_verification":{"current_native_endpoint":"accepted step 0 of temporalis163-native-m6-001",
            "skin_key":[51007,1],"current_pack_sha256":pins[str(PACK)],"current_receipt_sha256":pins[str(RECEIPT)],
            "current_pack_info":pack_info,"source_skin_faces":source_face_count,
            "source_skin_referenced_vertices":int(len(skin_vids)),"temporary_target_cap_faces":int(len(cap_f)),
            "temporary_target_cap_referenced_vertices":int(len(cap_vids)),
            "source_skin_triangle_coordinate_max_abs_difference_m":max_abs,
            "source_skin_triangle_coordinate_rms_difference_m":rms,
            "source_skin_triangle_coordinate_sha256_target_ordered":hashlib.sha256(target_source_tri.tobytes()).hexdigest(),
            "source_skin_triangle_coordinate_sha256_current_ordered":hashlib.sha256(current_skin_tri.tobytes()).hexdigest(),
            "temporary_target_source_skin_equals_current_native_skin":True,
            "target_bounds":bounds(target_v),"current_skin_bounds":bounds(skin_points),
            "cap_bounds":bounds(cap_points),"cap_vertices_to_current_skin_vertices_nearest_max_m":float(cap_to_skin.max()),
            "local_vs_world_candidate_bounds":wrong_frame},
        "corrected_repair_neighborhood_exclusion":{"contact_source_candidate_sha256":pins[str(CAND)],
            "contact_pair_table_sha256":pins[str(PAIRS)],"source_candidate_face_count":int(len(cand_f)),
            "skin_contact_pair_count":int(len(skin_pairs)),"unique_contact_candidate_face_count":int(len(contact_face_rows)),
            "unique_contact_candidate_vertex_count":int(len(contact_vids)),
            "contact_region_coordinate_frame":"CAND.step0_world_f32, verified against current target skin triangle coordinates",
            "cap_contact_aabb_axis_gaps_m":gap.tolist(),"cap_contact_aabb_lower_bound_m":float(np.linalg.norm(gap)),
            "minimum_cap_vertex_to_contact_vertex_distance_m":nearest,
            "exact_cap_fan_vs_full_world_candidate_triangle_pair_count":int(cap_audit["count"]),
            "runtime_skin_modified":False,"runtime_ocular_openings_changed":False,
            "temporary_target_used_at_runtime":False},
        "input_sha256":{**pins,str(Path(__file__)):sha(Path(__file__))},
        "runtime_skin_modified":False,"runtime_ocular_openings_changed":False,
        "temporary_target_used_at_runtime":False}
    out=OUT/"report.json"
    with out.open("w") as f: json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    fd=os.open(OUT,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    print(json.dumps({"event":"complete","report":str(out),"report_sha256":sha(out),
        "skin_max_abs_diff_m":max_abs,"skin_rms_diff_m":rms,
        "cap_contact_aabb_lower_bound_m":float(np.linalg.norm(gap)),
        "cap_to_contact_vertex_min_m":nearest,"cap_full_candidate_exact_pairs":int(cap_audit["count"])},sort_keys=True),flush=True)
if __name__=="__main__":main()

