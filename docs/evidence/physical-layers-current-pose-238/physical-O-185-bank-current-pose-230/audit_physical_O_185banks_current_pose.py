#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, importlib.util, json, sys, time
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D")
W=ROOT/"connected-obstacle-207/physical-transfer-209"
OUT=W/"root-connected-layer-fit-219/canonical-registration-diagnostic-228/attempt-008-physicalO-currentpose-185bank"
O_PATH=W/"physical-outer-direct-placement-235/O-direct-native-placement.npz"
O_RECEIPT=W/"physical-outer-direct-placement-235/report.json"
POSE_PATH=W/"P/native-kind-aware-source-preflight-226/run-006-native-emission/registered-skin-poses-100.json"
BANKHELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
EXACT=E/"collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(f"could not import {path}")
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()

def records(exact,triangles,face_ids):
    pts=tuple(tuple(Fraction.from_float(float(x)) for x in p) for p in np.asarray(triangles).reshape(-1,3))
    faces=np.arange(len(pts),dtype=np.int64).reshape(-1,3)
    raw=exact._records(pts,faces)
    return [(r[0],r[1],r[2],int(face_ids[i]),r[4]) for i,r in enumerate(raw)]

def current_world_to_body(q_xyzw):
    q=np.asarray(q_xyzw,dtype=np.float32).astype(np.float64).copy()
    q[:3]*=-1.0
    def rotate(v):
        return v+2.0*np.cross(q[:3],np.cross(q[:3],v)+q[3]*v)
    return np.column_stack([rotate(v) for v in np.eye(3)])

start=time.monotonic()
O_receipt=json.loads(O_RECEIPT.read_text())
if sha(O_PATH)!=O_receipt.get("output_sha256"):raise ValueError("physical O file SHA differs from direct-placement receipt")
z=np.load(O_PATH,allow_pickle=False)
P=np.asarray(z["points_world_m"],dtype=np.float32).astype(np.float64)
F=np.asarray(z["source_O_faces"],dtype=np.int64)
if not np.array_equal(F,np.asarray(z["faces"],dtype=np.int64)):raise ValueError("physical O faces/source_O_faces mismatch")
if not np.isfinite(P).all() or F.min()<0 or F.max()>=len(P):raise ValueError("physical O finite/index gate failed")
if len(P)!=32002 or len(F)!=63959:raise ValueError("physical O shape differs from receipt")

sys.path.insert(0,str(E/"incremental-exact-audit-owner-253/src"))
helper=module("connected_bone_bank_current_pose_235",BANKHELPER)
exact=module("numilab_human.connected_skin_bone_exact_current_pose_235",EXACT)
T,ids,gids,bodies,M39,offsets,positions39,bankkeys,pins,unused=helper.read_bank_inputs(BANKDIR,E)
if len(bankkeys)!=185:raise ValueError(f"expected 185 typed banks, got {len(bankkeys)}")
native_pose=json.loads(POSE_PATH.read_text())
native_bodies={int(x["body_index"]):x for x in native_pose["all_bodies"]}
bank_body_ids=set(int(x) for x in np.unique(ids[:,0]))
missing=sorted(bank_body_ids-set(native_bodies))
if missing:raise ValueError(f"native all_bodies pose missing bank bodies {missing}")
pose_compare=[]
Mcur={};Pcur={}
for body in sorted(bank_body_ids):
    cur=native_bodies[body]["current"]
    qpos=np.asarray(cur["position_world_m"],dtype=np.float32)
    qquat=np.asarray(cur["orientation_world_xyzw"],dtype=np.float32)
    bpos=np.asarray([float(bodies[body][f"position_{c}_m"]) for c in "xyz"],dtype=np.float32)
    bquat=np.asarray([float(bodies[body][f"orientation_q{c}"]) for c in "xyzw"],dtype=np.float32)
    qM=current_world_to_body(qquat)
    Mcur[body]=qM;Pcur[body]=qpos.astype(np.float64)
    pose_compare.append({"body":body,"position_bitwise_equal_to_039":bool(np.array_equal(qpos.view(np.uint32),bpos.view(np.uint32))),"orientation_bitwise_equal_to_039":bool(np.array_equal(qquat.view(np.uint32),bquat.view(np.uint32))),"position_delta_m":float(np.linalg.norm(qpos.astype(np.float64)-bpos.astype(np.float64))),"matrix_max_abs_delta":float(np.max(np.abs(qM-M39[body])))})
if any(x["matrix_max_abs_delta"]>1e-12 for x in pose_compare):raise ValueError("native and pinned 039 rotations differ; explicit transform review required")
local_cache={}
face_ids=np.arange(len(F),dtype=np.int64)
bank_rows=[];witness_arrays={};aabb_active=0;exact_total=0
for n,(body0,bone0) in enumerate(bankkeys,1):
    body,bone=int(body0),int(bone0)
    if body not in local_cache:
        local_cache[body]=(P-Pcur[body])@Mcur[body].T-np.asarray(offsets[body],dtype=np.float64)
    local=local_cache[body]
    S=local[F]
    mask=(ids[:,0]==body)&(ids[:,1]==bone)
    B=np.asarray(T[mask],dtype=np.float64)
    stable_ids=np.asarray(ids[mask,3],dtype=np.int64)
    if not len(B):raise ValueError(f"empty typed bank {body}/{bone}")
    slo,shi=S.min(1),S.max(1);blo,bhi=B.min((0,1)),B.max((0,1))
    active=np.flatnonzero(np.all(shi>=blo,axis=1)&np.all(slo<=bhi,axis=1))
    if not len(active):
        row={"body":body,"stable_bone_id":bone,"bank_triangles":len(B),"surface_face_count":len(F),"surface_aabb_active":0,"bone_aabb_active":0,"exact_aabb_candidate_pairs":0,"exact_crossing_pairs":0,"witness_pairs":[]}
    else:
        rlo,rhi=slo[active].min(0),shi[active].max(0)
        bactive=np.flatnonzero(np.all(B.max(1)>=rlo,axis=1)&np.all(B.min(1)<=rhi,axis=1))
        sr=records(exact,S[active],face_ids[active]);br=records(exact,B[bactive],stable_ids[bactive])
        rr=exact._audit_pair(sr,br,same_surface=False)
        pairs=np.asarray(rr.get("triangle_pairs",[]),dtype=np.int64).reshape(-1,2)
        exact_count=int(rr.get("count",len(pairs)))
        exact_total+=exact_count;aabb_active+=1
        row={"body":body,"stable_bone_id":bone,"bank_triangles":len(B),"surface_face_count":len(F),"surface_aabb_active":int(len(active)),"bone_aabb_active":int(len(bactive)),"exact_aabb_candidate_pairs":int(rr.get("aabb_candidate_pairs",0)),"allowed_shared_vertex_or_edge_pairs":int(rr.get("allowed_shared_vertex_or_edge_pairs",0)),"exact_crossing_pairs":exact_count,"witness_pair_count":int(len(pairs)),"witness_pairs_sample":pairs[:50].tolist()}
        if len(pairs):witness_arrays[f"{body}_{bone}"]=pairs
    bank_rows.append(row)
    (OUT/"progress.json").write_text(json.dumps({"completed_banks":n,"expected_banks":185,"last_body":body,"last_bone":bone,"exact_crossing_pairs_so_far":exact_total,"elapsed_s":time.monotonic()-start})+"\n")

if witness_arrays:np.savez_compressed(OUT/"all-185-crossing-pairs.npz",**witness_arrays)
report={"schema":"physical-O235-native-pose-exact-185-typed-bone-bank-crossing.v1","status":"COMPLETE_EXACT_FULL_PHYSICAL_O_185_BANK_CROSSING_AUDIT","scope":"all 63,959 physical O faces against all 185 exact typed bone banks; exact rational triangle predicate after approved body-local frame conversion; no solid containment or admission","physical_O_path":str(O_PATH),"physical_O_sha256":sha(O_PATH),"physical_O_receipt_path":str(O_RECEIPT),"physical_O_receipt_sha256":sha(O_RECEIPT),"physical_O_self_pairs_previously_verified":int(O_receipt["exact_self_pairs"]),"physical_O_points":int(len(P)),"physical_O_faces":int(len(F)),"native_pose_path":str(POSE_PATH),"native_pose_sha256":sha(POSE_PATH),"native_all_bodies_count":len(native_bodies),"bank_body_ids":len(bank_body_ids),"missing_bank_pose_ids":missing,"pose_rebind_policy":"retain exact pinned 039 local triangles and COM offsets; use current position and orientation quaternion from native run-006 all_bodies (Float32 fields promoted to Float64) to derive world_to_body for every bank body","pose_comparison_to_pinned039":{"body_count":len(pose_compare),"position_bitwise_matches":sum(x["position_bitwise_equal_to_039"] for x in pose_compare),"orientation_bitwise_matches":sum(x["orientation_bitwise_equal_to_039"] for x in pose_compare),"max_position_delta_m":max(x["position_delta_m"] for x in pose_compare),"max_matrix_delta":max(x["matrix_max_abs_delta"] for x in pose_compare),"per_body":pose_compare},"exact_owner_path":str(EXACT),"exact_owner_sha256":sha(EXACT),"bank_parser_path":str(BANKHELPER),"bank_parser_sha256":sha(BANKHELPER),"bank_input_pins":pins,"bank_count":len(bank_rows),"expected_bank_count":185,"aabb_active_bank_count":aabb_active,"exact_crossing_pair_count":exact_total,"banks_with_crossings":[r for r in bank_rows if r["exact_crossing_pairs"]],"banks":bank_rows,"pair_witness_archive":str(OUT/"all-185-crossing-pairs.npz") if witness_arrays else None,"elapsed_s":time.monotonic()-start,"physical_admission":False,"limitations":["This audits exact surface triangle crossings only; no all-space solid-containment certificate, layer thickness/volume-cell/mass transfer, dynamics, or physical admission is established."]}
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
(OUT/"progress.json").unlink(missing_ok=True)
print(json.dumps({"status":report["status"],"physical_O_faces":len(F),"bank_count":len(bank_rows),"aabb_active_bank_count":aabb_active,"exact_crossing_pair_count":exact_total,"banks_with_crossings":[[r["body"],r["stable_bone_id"],r["exact_crossing_pairs"]] for r in bank_rows if r["exact_crossing_pairs"]],"elapsed_s":report["elapsed_s"],"report":str(OUT/"report.json")},indent=2))
