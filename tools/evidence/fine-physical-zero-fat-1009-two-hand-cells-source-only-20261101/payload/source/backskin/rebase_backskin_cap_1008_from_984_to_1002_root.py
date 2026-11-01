#!/usr/bin/env python3
"""Compose and independently replay the 57482-59961 I flip with retained Skin/Fat domains."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
from itertools import combinations
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
BASE=H/"quality-partial-cavity-684-root"
OWNER=BASE/"paired_connected_boundary_patch_batch_883_targeted_source_edge_innerflip_root.py"
BASE_PARENT=H/"all-quality-984-retained-I-quad-qualified-root/verified-state.json"
PARENT=H/"all-quality-1002-zero-fat-qualified-root/verified-state.json"
PLAN=BASE/"backskin_cap_domains_990_linkclosure.json"
LOCAL=H/"all-quality-972-backskin-cap-localproof-983/local-feasibility-receipt.json"
DISTANCE=H/"all-quality-972-backskin-cap-localproof-983/surface-distance.json"
ORIENT=H/"all-quality-972-backskin-cap-localproof-983/oriented-equivalence-receipt.json"
OUT=H/"all-quality-1008-backskin-cap-rebased1002-root"
REUSE_DIR=H/"all-quality-1007-backskin-cap-rebased1002-root"
REUSE_EXPECTED={"candidate-I-typed-current.npz":"b98c823893ec2715b114482f0f6a0c6dccf820d0cf99af937520b03c6f2d3e05",
 "skin-volume-proposal.npz":"ea5c40905d9137d929e1254843bbdcb8db9bc7eebb4d73b5916b5d7fcbbe3a44",
 "fat-volume-proposal.npz":"4048abfd126bfe6826a7b3a2923ed74ec7736dedac25daf4618781b9bb03742b"}
BACK_REPLAY=H/"all-quality-992-backskin-cap-candidate-root/independent-replay.json"
BACK_STATE=H/"all-quality-992-backskin-cap-candidate-root/candidate-state-proposal.json"
BACK_REPORT=H/"all-quality-992-backskin-cap-candidate-root/report.json"
BACK_JOURNAL=H/"all-quality-992-backskin-cap-candidate-root/operation-journal.json"
STRICT_DISTANCE=H/"all-quality-992-backskin-cap-candidate-root/independent-3d-distance-replay-994.json"
EXPECTED_OWNER_SHA="7bde558d69512ab9edc41748ccda728ee510c75c4ec840a297eae65345385c54"
EXPECTED_PARENT_SHA="6edccfc0e81c1a15d9be88b212609a8f6bb1bb54deab2807f5ffd10ebf0ddab7"
EXPECTED_BASE_SHA="ff37d39728b34d91887060f8742d54121a4ecb69c76dbc2f89f2b88bb9e0781c"
EXPECTED_PLAN_SHA="182ad2537eebb2f3bf638a187ac63f5917a47e1e9ede23c87b464b09547ec0d7"
EXPECTED_LOCAL_SHA="271aca93297c86b0b3c205731dc3d27ae820398ba84206599cca513a956a061c"
EXPECTED_DISTANCE_SHA="7f5b9bc7788577f4a35581900f091495dde93e91f368983b6edcb2861968e1de"
EXPECTED_ORIENT_SHA="9b15f20ace703aecf31796de647a96387817367bcd852126d7cf44c11e80d13c"
EXPECTED_BACK_STATE_SHA="b9a45bb0b1a467839dd6ca0f7abe049453d28ffe0a343d368778d41958bc9733"
EXPECTED_BACK_REPORT_SHA="6f8cec52b9cfea4046ba820291491b2875398b0b4d6ecd4409a97b3f053849c5"
EXPECTED_BACK_JOURNAL_SHA="50ee0410380ca8dfbc850358eee1aa2820c2e3c64c8b97e19eebf5f11d776f61"
EXPECTED_BACK_REPLAY_SHA="14f5194975ea933205a8cb459b07723e8aa38451f521cca25726bcb1f99835c3"
EXPECTED_STRICT_DISTANCE_SHA="ec10bce966ab1c13ee9527587c18bdc2d89a3d1d1d1d00bcbd51db49740e8e05"
STRICT_DISTANCE_CAP_M=50e-6
BASE_SOURCE_ROWS=np.asarray([52337,52342],np.int64)
SOURCE_ROWS=None
SOURCE_EDGE=np.asarray([57482,59961],np.int64)
OLD_SOURCE=np.asarray([[57482,60059,59961],[57482,59961,59941]],np.int64)
NEW_SOURCE=np.asarray([[57482,60059,59941],[59961,59941,60059]],np.int64)
INNER_3D_CAP_M=250e-6

def sha(path):
    return hashlib.file_digest(Path(path).open("rb"),"sha256").hexdigest()

def pin(path):
    p=Path(path).resolve()
    return {"path":str(p),"sha256":sha(p)}

def json_default(value):
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,Path): return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")

if OUT.exists():
    raise FileExistsError(f"candidate directory already exists: {OUT}")
for path,want in ((OWNER,EXPECTED_OWNER_SHA),(PARENT,EXPECTED_PARENT_SHA),(BASE_PARENT,EXPECTED_BASE_SHA),(PLAN,EXPECTED_PLAN_SHA),
                  (LOCAL,EXPECTED_LOCAL_SHA),(DISTANCE,EXPECTED_DISTANCE_SHA),(ORIENT,EXPECTED_ORIENT_SHA),
                  (BACK_STATE,EXPECTED_BACK_STATE_SHA),(BACK_REPORT,EXPECTED_BACK_REPORT_SHA),(BACK_JOURNAL,EXPECTED_BACK_JOURNAL_SHA),
                  (BACK_REPLAY,EXPECTED_BACK_REPLAY_SHA),(STRICT_DISTANCE,EXPECTED_STRICT_DISTANCE_SHA)):
    if sha(path)!=want:
        raise ValueError(f"pinned input hash mismatch: {path}")

spec=importlib.util.spec_from_file_location("owner883_backskin991",OWNER)
B=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=B
spec.loader.exec_module(B)
ENG=B.ENG
T=B.T
M=B.M
A=B.A
root,st=B._source_state(PARENT)
base_root,base_st=B._source_state(BASE_PARENT)
plan=json.loads(PLAN.read_text())
local=json.loads(LOCAL.read_text())
distance=json.loads(DISTANCE.read_text())
orient=json.loads(ORIENT.read_text())
backroot,backst=B._source_state(BACK_STATE)
backreport=json.loads(BACK_REPORT.read_text());backjournal=json.loads(BACK_JOURNAL.read_text());backreplay=json.loads(BACK_REPLAY.read_text())
strict_distance=json.loads(STRICT_DISTANCE.read_text())
if backst.get("source_state",{}).get("sha256")!=EXPECTED_BASE_SHA or backjournal.get("parent_state",{}).get("sha256")!=EXPECTED_BASE_SHA:
    raise ValueError("back proposal/source journal is not pinned to exact984 parent")
if backreport.get("operation_journal",{}).get("sha256")!=EXPECTED_BACK_JOURNAL_SHA:
    raise ValueError("back report does not pin its immutable operation journal")
if backreport.get("source_candidate",{}).get("sha256")!=backst["source_surfaces"]["I"]["sha256"]:
    raise ValueError("back report source pin differs from its typed candidate state")
for vr in ("Skin","Fat"):
    if backreport.get("volume_candidates",{}).get(vr,{}).get("sha256")!=backst[vr+"_current"].get("sha256"):
        raise ValueError(vr+" back report volume pin differs from its typed candidate state")
if backst.get("backskin_cap_transfer",{}).get("independent_replay_status")!="PASS_INDEPENDENT_BACKSKIN_CAP_REPLAY" or backreplay.get("status")!="PASS_INDEPENDENT_BACKSKIN_CAP_REPLAY":
    raise ValueError("back proposal parent independent replay is not passing")
if backreport.get("independent_replay",{}).get("sha256")!=EXPECTED_BACK_REPLAY_SHA:
    raise ValueError("back report does not pin its independent replay")
if strict_distance.get("status")!="PASS" or strict_distance.get("parent_state",{}).get("sha256")!=EXPECTED_BASE_SHA:
    raise ValueError("independent source-patch distance replay is not passing on984")
if strict_distance.get("candidate_state",{}).get("sha256")!=EXPECTED_BACK_STATE_SHA:
    raise ValueError("strict distance replay is not bound to exact back proposal")
if strict_distance.get("candidate_surface",{}).get("sha256")!=backst["source_surfaces"]["I"]["sha256"]:
    raise ValueError("strict distance replay is not bound to exact back I surface")
if float(strict_distance.get("method",{}).get("cap_m",np.inf))!=STRICT_DISTANCE_CAP_M:
    raise ValueError("strict independent 50um distance cap mismatch")
strict_distance_upper=max(float(strict_distance["method"]["directional_results"][d]["upper_bound_m"]) for d in ("old_to_new","new_to_old"))
if strict_distance_upper>STRICT_DISTANCE_CAP_M:
    raise ValueError("independent current 50um source distance bound exceeded")
if plan.get("status")!="PASS_EXACTKEY_REBASED_CLOSED_REGIONS_WITH_1PCT_CAP":
    raise ValueError("retained full-domain plan is not a passing exact-key closure")
if plan.get("parent984",{}).get("sha256")!=EXPECTED_BASE_SHA:
    raise ValueError("closure plan is not pinned to current984")
if local.get("status")!="PASS_LOCAL_CANDIDATE_PROOF_PENDING_GLOBAL_AUDIT":
    raise ValueError("local Skin/Fat interface-transfer proof is not passing")
parent972=H/"all-quality-972-last-fat-fixed-775-qualified-root/verified-state.json"
if sha(parent972)!=plan.get("parent972",{}).get("sha256") or local.get("input",{}).get("state_sha256")!=sha(parent972):
    raise ValueError("local feasibility proof is not bound to its exact972 parent")
root972,st972=B._source_state(parent972)
for vr,key in (("Skin","Skin_sha256"),("Fat","Fat_sha256")):
    if sha(st972[vr+"_current"]["path"])!=local.get("input",{}).get(key):
        raise ValueError("local feasibility proof "+vr+" volume pin differs from972")
if sha(st972["source_surfaces"]["I"]["path"])!=local.get("input",{}).get("I_sha256"):
    raise ValueError("local feasibility proof I surface pin differs from972")
if orient.get("input_state",{}).get("sha256")!=sha(parent972):
    raise ValueError("oriented neighbor receipt is not bound to the same972 parent")
if distance.get("status")!="PASS_BIDIRECTIONAL_PATCH_DISTANCE_UNDER_250UM":
    raise ValueError("actual 3D source patch distance proof is not passing")
if orient.get("status")!="PASS_LOCAL_SKIN_CAP_ORIENTED_NEIGHBORS":
    raise ValueError("local Skin oriented-neighbor proof is not passing")
if local.get("scope",{}).get("geometry_files_written") is not False:
    raise ValueError("local proof unexpectedly claims candidate geometry output")
if local.get("scope",{}).get("physical_admission") is not False:
    raise ValueError("local proof improperly claims physical admission")
if float(distance.get("construction_distance_cap_m",np.inf))!=INNER_3D_CAP_M:
    raise ValueError("legacy250um feasibility proof cap differs from its own declaration")

tic=time.monotonic()
sources={r:B.load_npz(B.check_ref(st["source_surfaces"][r],"current typed "+r)) for r in ("O","I","D")}
surface=sources["I"]
srcfaces=np.asarray(surface["faces"],np.int64)
srcpoints=np.asarray(surface["points_world_m_f32"],np.float32)
def face_row_lookup(faces,label):
    out={}
    for i,f in enumerate(np.asarray(faces,np.int64)):
        key=tuple(sorted(map(int,f)))
        out.setdefault(key,[]).append(int(i))
    if any(len(v)!=1 for v in out.values()): raise ValueError(label+" has duplicate canonical source faces")
    return out
base_surface=B.load_npz(B.check_ref(base_st["source_surfaces"]["I"],"984 I base"))
base_faces=np.asarray(base_surface["faces"],np.int64)
if not np.array_equal(base_faces[BASE_SOURCE_ROWS],OLD_SOURCE):
    raise ValueError("pinned984 I source rows do not contain exact old triangles")
lookup_src=face_row_lookup(srcfaces,"1002 I source")
SOURCE_ROWS=np.asarray([lookup_src.get(tuple(sorted(map(int,f))),[-1])[0] for f in OLD_SOURCE],np.int64)
if np.any(SOURCE_ROWS<0) or not np.array_equal(srcfaces[SOURCE_ROWS],OLD_SOURCE):
    raise ValueError("exact old I source face keys/orientations do not map to1002")
back_surface=B.load_npz(B.check_ref(backst["source_surfaces"]["I"],"back 984 I candidate"))
expected_back_faces=base_faces.copy();expected_back_faces[BASE_SOURCE_ROWS]=NEW_SOURCE
if not np.array_equal(np.asarray(back_surface["faces"],np.int64),expected_back_faces):
    raise ValueError("back candidate I source geometry does not reconstruct from exact984 faces")
if not np.array_equal(np.asarray(back_surface["points_world_m_f32"],np.float32).view(np.uint32),
                      np.asarray(surface["points_world_m_f32"],np.float32).view(np.uint32)):
    raise ValueError("back candidate and1002 I F32 point bytes differ")
for ck in ("points_world_m",):
    if ck in back_surface or ck in surface:
        if ck not in back_surface or ck not in surface or not np.array_equal(np.asarray(back_surface[ck]).view(np.uint8),np.asarray(surface[ck]).view(np.uint8)):
            raise ValueError("back candidate and1002 I full-precision point bytes differ")
if not np.array_equal(np.asarray(plan["source_faces_exact_current984"],np.int64),OLD_SOURCE):
    raise ValueError("closure plan old source triangles do not match pinned984")
if not np.array_equal(np.asarray(plan["source_faces_exact_current984"],np.int64),OLD_SOURCE):
    raise ValueError("990 plan current I faces do not match fixed operation")
if not np.array_equal(np.asarray(plan["source_faces_new_oriented"],np.int64),NEW_SOURCE):
    raise ValueError("990 plan proposed I triangles differ from local proof")
if not np.array_equal(np.asarray(local["source_operation"]["face_rows"],np.int64),BASE_SOURCE_ROWS):
    raise ValueError("local feasibility receipt source rows differ")
if not np.array_equal(np.asarray(local["source_operation"]["old_faces"],np.int64),OLD_SOURCE):
    raise ValueError("local feasibility receipt old I triangles differ")
if not np.array_equal(np.asarray(local["source_operation"]["new_faces"],np.int64),NEW_SOURCE):
    raise ValueError("local feasibility receipt new I triangles differ")
if not np.array_equal(np.asarray(distance["old_faces"],np.int64),OLD_SOURCE) or not np.array_equal(np.asarray(distance["new_faces"],np.int64),NEW_SOURCE):
    raise ValueError("3D distance proof does not cover exact source triangles")
proof_surface_path=Path(distance["I"]["path"])
if sha(proof_surface_path)!=distance["I"]["sha256"]:
    raise ValueError("3D distance proof source surface pin mismatch")
proof_surface=B.load_npz(proof_surface_path)
if not np.array_equal(np.asarray(proof_surface["points_world_m_f32"],np.float32).view(np.uint32),srcpoints.view(np.uint32)):
    raise ValueError("legacy 3D proof point coordinates are not bitwise identical to current1002 I")
if not np.array_equal(np.asarray(proof_surface["faces"],np.int64)[BASE_SOURCE_ROWS],OLD_SOURCE):
    raise ValueError("legacy 3D proof source-face rows do not match base984")
strict_f32_hash=hashlib.sha256(np.ascontiguousarray(srcpoints).tobytes()).hexdigest()
strict_f64_hash=hashlib.sha256(np.ascontiguousarray(np.asarray(surface["points_world_m"],np.float64)).tobytes()).hexdigest()
if strict_distance.get("points_world_m_f32_sha256")!=strict_f32_hash or strict_distance.get("points_world_m_f64_sha256")!=strict_f64_hash:
    raise ValueError("independent 50um distance proof coordinate view hashes differ from current1002")
if not np.array_equal(np.asarray(strict_distance["old_faces"],np.int64),OLD_SOURCE) or not np.array_equal(np.asarray(strict_distance["new_faces"],np.int64),NEW_SOURCE):
    raise ValueError("strict 50um proof does not bind exact old/new faces")
if len({tuple(sorted(map(int,f))) for f in srcfaces})!=len(srcfaces):
    raise ValueError("current I source contains duplicate canonical faces")
source_face_keys={tuple(sorted(map(int,f))) for f in srcfaces}
if any(tuple(sorted(map(int,f))) in source_face_keys for f in NEW_SOURCE):
    raise ValueError("new I source face already exists")
base_face_keys={tuple(sorted(map(int,f))) for f in base_faces}
current_face_keys={tuple(sorted(map(int,f))) for f in srcfaces}
prior_removed_source=base_face_keys-current_face_keys
prior_added_source=current_face_keys-base_face_keys
back_old_face_keys={tuple(sorted(map(int,f))) for f in OLD_SOURCE}
back_new_face_keys={tuple(sorted(map(int,f))) for f in NEW_SOURCE}
if (prior_removed_source|prior_added_source)&(back_old_face_keys|back_new_face_keys):
    raise ValueError("prior1002 source-face change conflicts with back source-face operation")
prior_source_vertices=set(map(int,np.asarray(base_faces)[[i for i,f in enumerate(base_faces) if tuple(sorted(map(int,f))) in prior_removed_source]].ravel()))
prior_source_vertices.update(map(int,np.asarray(srcfaces)[[i for i,f in enumerate(srcfaces) if tuple(sorted(map(int,f))) in prior_added_source]].ravel()))
if prior_source_vertices & set(map(int,np.asarray(OLD_SOURCE).ravel())):
    raise ValueError("prior1002 source flip shares a vertex with back source operation")
def oriented_edge_chain(fs):
    out=Counter()
    for f in np.asarray(fs,np.int64).reshape((-1,3)):
        a,b,c=map(int,f)
        for u,v in ((a,b),(b,c),(c,a)):
            key=tuple(sorted((u,v)))
            out[key]+=1 if (u,v)==key else -1
    return {k:v for k,v in out.items() if v}
if oriented_edge_chain(OLD_SOURCE)!=oriented_edge_chain(NEW_SOURCE):
    raise ValueError("new I triangles do not preserve the exact oriented quad boundary")
if sum(1 for f in OLD_SOURCE if set(SOURCE_EDGE).issubset(set(map(int,f))))!=2:
    raise ValueError("old I faces do not share the selected source diagonal")
if sum(1 for f in NEW_SOURCE if set((60059,59941)).issubset(set(map(int,f))))!=2:
    raise ValueError("new I faces do not share the opposite source diagonal")

volumes={vr:B.B.volume_cache(st[vr+"_current"]) for vr in ("Skin","Fat")}
base_volumes={vr:B.B.volume_cache(base_st[vr+"_current"]) for vr in ("Skin","Fat")}
def cell_key(tet,material): return (tuple(sorted(map(int,tet))),int(material))
def row_lookup(cache,label):
    d={}
    for i,(tet,mat) in enumerate(zip(np.asarray(cache["tets"],np.int64),np.asarray(cache["mat"]))):
        k=cell_key(tet,mat)
        if k in d: raise ValueError(label+" has duplicate tet/material keys")
        d[k]=int(i)
    return d
row_maps={};full_domain_rows_by_role={};prior_volume_delta_by_role={}
for vr in ("Skin","Fat"):
    bl=row_lookup(base_volumes[vr],"984 "+vr);cl=row_lookup(volumes[vr],"1002 "+vr)
    base_keys=set(bl);current_keys=set(cl)
    removed_prior=base_keys-current_keys;added_prior=current_keys-base_keys
    prior_volume_delta_by_role[vr]={"removed_keys":removed_prior,"added_keys":added_prior}
    back_full_base_rows=np.asarray(plan["roles"][vr]["full_domain_cell_rows"],np.int64)
    back_full_keys={cell_key(base_volumes[vr]["tets"][int(r)],base_volumes[vr]["mat"][int(r)]) for r in back_full_base_rows}
    if removed_prior & back_full_keys:
        raise ValueError(vr+" prior1002 accepted cell patch intersects back full-domain cells")
    row_maps[vr]={"base_to_current":cl}
def map_base_rows(role,rows,what):
    result=[];lookup=row_maps[role]["base_to_current"];base=base_volumes[role]
    for r in map(int,rows):
        k=cell_key(base["tets"][r],base["mat"][r])
        if k not in lookup: raise ValueError(role+" "+what+" exact tet/material key is absent from1002")
        result.append(lookup[k])
    if len(result)!=len(set(result)): raise ValueError(role+" "+what+" key map is not injective")
    return np.asarray(result,np.int64)
source_maps={vr:B.active_map(surface,c,"I/"+vr) for vr,c in volumes.items()}
target_updates={}
target_face_replacements={}
target_rows_by_role={}
orientation_by_role={}
selected_by_role={}
new_tets_by_role={}
full_domain_metrics={}
def face_owners(cache,face):
    ids=np.asarray(B.B.incident_cells(cache,tuple(map(int,face))),np.int64)
    target=set(map(int,face))
    return sorted(int(r) for r in ids if target.issubset(set(map(int,cache["tets"][int(r)]))))
def same_oriented_triangle(a,b):
    aa=tuple(map(int,a));bb=tuple(map(int,b))
    if set(aa)!=set(bb):return False
    return any(aa==bb[i:]+bb[:i] for i in range(3))
def exact_det_sum(points,tets):
    tt=np.asarray(tets,np.int64).reshape((-1,4))
    if not len(tt):return 0
    fp=np.asarray(points,np.float32)
    cache={int(v):ENG.S.S.ip(fp[int(v)]) for v in np.unique(tt.ravel())}
    return sum(int(ENG.S.S.det(*(cache[int(v)] for v in tet))) for tet in tt)

# Bind current parent target faces by exact source-point maps and orientation.
for vr,c in volumes.items():
    cross=np.asarray(source_maps[vr],np.int64)
    if np.any(cross[np.r_[OLD_SOURCE,NEW_SOURCE].ravel()]<0):
        raise ValueError(vr+" active source-to-volume map is missing a changed-face vertex")
    oldvf=cross[OLD_SOURCE]
    newvf=cross[NEW_SOURCE]
    lookup=B.B.face_lookup(np.asarray(c["target"],np.int64))
    target_rows=[];rels=[]
    for f in oldvf:
        hits=lookup.get(tuple(sorted(map(int,f))),[])
        if len(hits)!=1:
            raise ValueError(vr+" old I face does not map to one unique current target row")
        row=int(hits[0]);target_rows.append(row)
        rels.append(int(ENG.rel(np.asarray(c["target"],np.int64)[row],f)))
    if len(set(rels))!=1:
        raise ValueError(vr+" source-to-target orientation differs across flipped pair")
    mapped_new=newvf.copy()
    if rels[0]<0:
        mapped_new=mapped_new[:,[0,2,1]]
    planned_old=np.asarray(plan["roles"][vr]["old_target_faces_volume_rows"],np.int64)
    current_old=np.asarray([c["target"][r] for r in target_rows],np.int64)
    if any(tuple(sorted(map(int,a)))!=tuple(sorted(map(int,b))) or not same_oriented_triangle(a,b) for a,b in zip(current_old,planned_old)):
        raise ValueError(vr+" exact old target face keys/orientations do not map to1002")
    if not np.array_equal(mapped_new,np.asarray(plan["roles"][vr]["new_target_faces_volume_rows"],np.int64)):
        raise ValueError(vr+" new volume target triangles differ from exact-key domain plan")
    target_lookup=B.B.face_lookup(np.asarray(c["target"],np.int64))
    oldrowset=set(target_rows)
    for f in mapped_new:
        existing=target_lookup.get(tuple(sorted(map(int,f))),[])
        if any(int(r) not in oldrowset for r in existing):
            raise ValueError(vr+" replacement target face collides with a retained current target face")
    target_updates[vr]=[(target_rows[j],mapped_new[j].copy()) for j in range(2)]
    target_face_replacements[vr]=[(np.asarray(c["target"][target_rows[j]],np.int64).copy(),mapped_new[j].copy()) for j in range(2)]
    target_rows_by_role[vr]=np.asarray(target_rows,np.int64)
    orientation_by_role[vr]=int(rels[0])
    selected=map_base_rows(vr,plan["roles"][vr]["seed_rows"],"seed")
    full=map_base_rows(vr,plan["roles"][vr]["full_domain_cell_rows"],"full-domain")
    full_domain_rows_by_role[vr]=full
    base_cross=np.asarray(B.active_map(base_surface,base_volumes[vr],"984 I/"+vr),np.int64)
    current_cross=np.asarray(source_maps[vr],np.int64)
    if np.any(base_cross[SOURCE_EDGE]<0) or np.any(current_cross[SOURCE_EDGE]<0):
        raise ValueError(vr+" source-edge endpoints are not active in both exact point maps")
    base_star=np.asarray(B.B.full_edge_star(base_volumes[vr],int(base_cross[SOURCE_EDGE[0]]),int(base_cross[SOURCE_EDGE[1]])),np.int64)
    current_star=np.asarray(B.B.full_edge_star(c,int(current_cross[SOURCE_EDGE[0]]),int(current_cross[SOURCE_EDGE[1]])),np.int64)
    if set(map(int,current_star))!=set(map(int,selected)):
        raise ValueError(vr+" exact current endpoint full-star differs from selected seed tet/material keys")
    base_star_keys={cell_key(base_volumes[vr]["tets"][int(r)],base_volumes[vr]["mat"][int(r)]) for r in base_star}
    current_star_keys={cell_key(c["tets"][int(r)],c["mat"][int(r)]) for r in current_star}
    if base_star_keys!=current_star_keys:
        raise ValueError(vr+" exact endpoint-star key set changed between984 and1002")
    if len(np.unique(selected))!=len(selected) or len(np.unique(full))!=len(full):
        raise ValueError(vr+" exact-key selected/domain rows are duplicated")
    if not set(map(int,selected)).issubset(set(map(int,full))):
        raise ValueError(vr+" complete conforming domain does not contain every retired seed cell")
    if np.any(selected<0) or np.any(selected>=len(c["tets"])) or np.any(full<0) or np.any(full>=len(c["tets"])):
        raise ValueError(vr+" closure rows are out of range in current984")
    if len(np.unique(np.asarray(c["mat"])[selected]))!=1:
        raise ValueError(vr+" selected replacement seed crosses material ownership")
    selected_by_role[vr]=selected
    old_seed=np.asarray(c["tets"][selected],np.int64)
    expected_keys=plan["exact_seed_tet_rows_972_to_984"][vr]["exact_tet_keys"]
    if sorted([sorted(map(int,t)) for t in old_seed])!=sorted([sorted(map(int,t)) for t in expected_keys]):
        raise ValueError(vr+" exact seed tetrahedron keys do not match 972→984 plan")
    full_tets=np.asarray(c["tets"][full],np.int64)
    old_boundary=np.asarray(T.m.boundary(full_tets),np.int64)
    old_chain=ENG.chain(old_boundary)
    for f in np.asarray(plan["roles"][vr]["old_target_faces_volume_rows"],np.int64):
        if old_chain.get(tuple(sorted(map(int,f))),0)!=ENG.fsign(f):
            raise ValueError(vr+" old source target face absent or reversed on declared full-domain boundary")
    new_boundary=np.asarray(ENG.oriented_boundary_replace(old_boundary,target_face_replacements[vr]),np.int64)
    is_closed,open_count=B.checked_closed_boundary(new_boundary,vr+" candidate changed target boundary")
    mt=T.manifold_topology(new_boundary)
    if not is_closed or open_count or mt.get("bad_vertex_links") or mt.get("boundary_components")!=1:
        raise ValueError(vr+" changed full-domain boundary is not one closed manifold")
    oldsum=exact_det_sum(c["points"],full_tets)
    oldflux=int(ENG.exact_boundary_flux(c["points"],old_boundary))
    newflux=int(ENG.exact_boundary_flux(c["points"],new_boundary))
    if oldsum!=oldflux:
        raise ValueError(vr+" old closed-domain exact volume/flux mismatch")
    delta=int(newflux-oldflux)
    if delta!=int(plan["roles"][vr]["full_domain_exact_signed_delta"]):
        raise ValueError(vr+" recomputed full-domain exact delta differs from 990 receipt")
    local_fraction=abs(delta)/float(oldsum)
    global_flux=int(ENG.exact_boundary_flux(c["points"],c["target"]))
    global_fraction=abs(delta)/float(global_flux)
    if local_fraction>0.01:
        raise ValueError(vr+" recomputed complete-region exact 1% volume budget exceeded")
    if global_fraction>1e-4:
        raise ValueError(vr+" recomputed whole-region exact 1e-4 volume budget exceeded")
    if abs(delta)/float(oldsum)>float(plan["caps"]["local_fraction"]):
        raise ValueError(vr+" local fraction exceeds pinned plan cap")
    keep_full=np.asarray([int(r) not in set(map(int,selected)) for r in full],bool)
    if vr=="Skin":
        new_local=np.empty((0,4),np.int64)
    else:
        if np.asarray(local["Fat_5point_cavity"]["old_tets"],np.int64).shape != np.asarray(old_seed).shape:
            raise ValueError("Fat local proof old seed shape mismatch")
        proof_old=np.asarray(local["Fat_5point_cavity"]["old_tets"],np.int64)
        if sorted([sorted(map(int,t)) for t in proof_old])!=sorted([sorted(map(int,t)) for t in old_seed]):
            raise ValueError("Fat exact old seed tets differ from local 785 proof")
        new_local=np.asarray(local["Fat_5point_cavity"]["new_tets"],np.int64).reshape((-1,4))
        if not set(map(int,new_local.ravel())).issubset(set(map(int,np.unique(old_seed.ravel())))):
            raise ValueError("Fat local solver output uses a point outside the five-point seed")
    new_tets_by_role[vr]=new_local
    delta_local=exact_det_sum(c["points"],new_local)-exact_det_sum(c["points"],old_seed)
    if delta_local!=delta:
        raise ValueError(vr+" seed replacement exact determinant delta differs from full-domain flux")
    if int(plan["roles"][vr]["full_domain_exact_signed_delta"])!=delta:
        raise ValueError(vr+" plan exact regional delta changed")
    combined=np.vstack((full_tets[keep_full],new_local)).astype(np.int64)
    if ENG.chain(np.asarray(T.m.boundary(combined),np.int64))!=ENG.chain(new_boundary):
        raise ValueError(vr+" retained conforming region does not realize changed target boundary")
    combined_sum=exact_det_sum(c["points"],combined)
    if combined_sum!=newflux:
        raise ValueError(vr+" retained conforming region exact tet sum does not equal target flux")
    dd,qq=M.measurements(c["points"],combined)
    if np.any(np.asarray(dd)<=0) or not np.all(np.asarray(M.passes(dd,qq),bool)):
        raise ValueError(vr+" retained conforming region has a nonpositive/native-bad cell")
    oldbad=int(np.count_nonzero(~np.asarray(M.passes(c["det"][full],c["q"][full]),bool)))
    full_domain_metrics[vr]={"full_rows":full,"seed_rows":selected,"cell_count":int(len(full)),
       "point_count":int(len(np.unique(full_tets.ravel()))),"old_det6_sum":oldsum,"old_flux_det6":oldflux,
       "new_flux_det6":newflux,"exact_delta_det6":delta,"local_volume_fraction":local_fraction,
       "global_volume_fraction":global_fraction,"old_bad_cells":oldbad,"new_bad_cells":0,
       "old_boundary_face_count":int(len(old_boundary)),"new_boundary_face_count":int(len(new_boundary)),
       "old_boundary_closed":True,"new_boundary_closed":True,"retained_cells_native_pass":True}
# Skin cap removal and local Fat exact5point MILP are the paired operation.
if sum(int(full_domain_metrics[r]["exact_delta_det6"]) for r in ("Skin","Fat"))!=0:
    raise ValueError("paired Skin/Fat exact regional volume deltas do not cancel")
skin_cell=int(selected_by_role["Skin"][0])
skin_tet=np.asarray(volumes["Skin"]["tets"][skin_cell],np.int64)
if sorted(map(int,skin_tet))!=sorted(map(int,local["Skin_cap"]["tet"])):
    raise ValueError("current984 Skin seed cap tet differs from local oriented-neighbor proof")
if len(selected_by_role["Skin"])!=1 or len(selected_by_role["Fat"])!=2:
    raise ValueError("expected exact one-cell Skin cap and two-cell Fat seed")
# Independently show the two proposed Skin faces are exactly exposed by cap retirement.
for nf in np.asarray(plan["roles"]["Skin"]["new_target_faces_volume_rows"],np.int64):
    owners=face_owners(volumes["Skin"],nf)
    if len(owners)!=2 or skin_cell not in owners:
        raise ValueError("Skin new target face is not exactly the cap plus one retained neighbor")
    neighbor=next(int(r) for r in owners if int(r)!=skin_cell)
    outward=np.asarray(T.m.boundary(volumes["Skin"]["tets"][neighbor:neighbor+1]),np.int64)
    matches=[f for f in outward if tuple(sorted(map(int,f)))==tuple(sorted(map(int,nf)))]
    if len(matches)!=1 or not same_oriented_triangle(nf,matches[0]):
        raise ValueError("Skin retained neighbor oriented face is not the declared new target face")
# Verify independent source-to-volume face topology relation and simple 2-to-2 face count.
for vr in ("Skin","Fat"):
    if len(target_updates[vr])!=2 or orientation_by_role[vr] not in (-1,1):
        raise ValueError(vr+" source flip target-face pairing malformed")
# Recompute exact old/new det sums and native gate for the local Fat candidate.
fat_new=new_tets_by_role["Fat"]
fatdet,fatq=M.measurements(volumes["Fat"]["points"],fat_new)
if np.any(fatdet<=0) or not np.all(np.asarray(M.passes(fatdet,fatq),bool)):
    raise ValueError("Fat 785 replacement tets fail native quality/positive gates")
if len(fat_new)!=2:
    raise ValueError("Fat local solver did not return exactly two replacement tets")
# Compose one immutable source patch with each paired role's exact seed replacement.
candidate={
    "face_rows":SOURCE_ROWS.tolist(),"new_faces":NEW_SOURCE.copy(),"edge":SOURCE_EDGE.tolist(),
    "quad":set(map(int,OLD_SOURCE.ravel())),"shared":[57482,59961],"opposite":[60059,59941],
    "source_patch_kind":"edge_2to2","patch_boundary_cycle":[57482,60059,59961,59941],
    "quad_hausdorff":{"hausdorff_upper_bound_m":float(strict_distance_upper),
        "proof_method":strict_distance["method"]["description"],
        "distance_report":pin(STRICT_DISTANCE),"legacy_250um_feasibility_report":pin(DISTANCE),"oriented_equivalence_receipt":pin(ORIENT)},
    "surface_hausdorff_limit_m":STRICT_DISTANCE_CAP_M,"inner_back_touched":True,
    "inner_back_hausdorff":{"hausdorff_upper_bound_m":float(strict_distance_upper),
        "proof_method":strict_distance["method"]["description"],
        "distance_report":pin(STRICT_DISTANCE),"cap_m":STRICT_DISTANCE_CAP_M},
    "other_role_protected_volume_rows_touched":{"Skin":[],"Fat":[]}
}
patch_volumes={}
for vr,c in volumes.items():
    selected=selected_by_role[vr]
    newt=new_tets_by_role[vr]
    target_replacements=target_updates[vr]
    oldbad=int(np.count_nonzero(~np.asarray(M.passes(c["det"][selected],c["q"][selected]),bool)))
    if len(newt):
        newdet,newq=M.measurements(c["points"],newt)
        newbad=int(np.count_nonzero(~np.asarray(M.passes(newdet,newq),bool)))
    else:
        newdet=np.empty((0,),np.float64);newq=np.empty((0,),np.float64);newbad=0
    exact_delta=exact_det_sum(c["points"],newt)-exact_det_sum(c["points"],c["tets"][selected])
    patch_volumes[vr]={
        "cell_rows":selected,"replacement_tets":newt,"exact_det6_delta":exact_delta,
        "old_bad":oldbad,"new_bad":newbad,"material":int(np.asarray(c["mat"])[selected[0]]),
        "target_replacements":target_replacements,
        "candidate_operations":[{"orientation_relation":orientation_by_role[vr],"inner_back_touched":True,
             "inner_back_hausdorff":candidate["inner_back_hausdorff"]}],
        "inner_back_hausdorff_max_m":float(strict_distance_upper),
        "local_volume_fraction":float(full_domain_metrics[vr]["local_volume_fraction"]),
        "global_volume_fraction":float(full_domain_metrics[vr]["global_volume_fraction"])
    }
accepted=[{"group":[candidate],"volumes":patch_volumes,
    "selection":"paired inner-I source diagonal flip with closed retained-domain Skin/Fat proof"}]
combined_bad_before=sum(int(np.count_nonzero(volumes[r]["bad"])) for r in ("Skin","Fat"))

# Compose typed I source arrays and paired volume arrays using the pinned existing owner.
surface_arrays,face_ops=B.compose_surface_batch(surface,accepted,st["source_surfaces"]["I"]["path"])
if not np.array_equal(np.asarray(surface_arrays["faces"],np.int64)[SOURCE_ROWS],NEW_SOURCE):
    raise ValueError("source composer did not write the exact declared new I faces")
if not np.array_equal(ENG.topology_sig(srcfaces),ENG.topology_sig(np.asarray(surface_arrays["faces"],np.int64))):
    raise ValueError("source 2-to-2 face flip changed the I mesh topology signature")
OUT.mkdir(parents=True)
rebase_path=OUT/"exact-key-rebase-1002.json"
def tet_key_json(tet,mat): return {"vertices":sorted(map(int,tet)),"material":int(mat)}
rebase_receipt={"schema":"h706.backskin-cap-exact-key-rebase-1002.v1","status":"PASS_EXACT_KEY_REBASE_AND_DISJOINTNESS",
 "base984":pin(BASE_PARENT),"current1002":pin(PARENT),"back_candidate_state":pin(BACK_STATE),
 "source":{"base_face_rows":BASE_SOURCE_ROWS.tolist(),"current_face_rows":SOURCE_ROWS.tolist(),
  "old_faces":OLD_SOURCE.tolist(),"new_faces":NEW_SOURCE.tolist(),"prior_removed_face_count":len(prior_removed_source),
  "prior_added_face_count":len(prior_added_source),"prior_changed_vertex_intersection":0},
 "roles":{vr:{"base_seed_rows":np.asarray(plan["roles"][vr]["seed_rows"],np.int64).tolist(),
  "current_seed_rows":selected_by_role[vr].tolist(),
  "seed_keys":[{"base_row":int(br),"current_row":int(cr),**tet_key_json(base_volumes[vr]["tets"][int(br)],base_volumes[vr]["mat"][int(br)])}
                for br,cr in zip(plan["roles"][vr]["seed_rows"],selected_by_role[vr])],
  "base_full_domain_rows":np.asarray(plan["roles"][vr]["full_domain_cell_rows"],np.int64).tolist(),
  "current_full_domain_rows":full_domain_rows_by_role[vr].tolist(),
  "full_domain_key_count":len(full_domain_rows_by_role[vr]),
  "prior_removed_cell_key_count":len(prior_volume_delta_by_role[vr]["removed_keys"]),
  "prior_added_cell_key_count":len(prior_volume_delta_by_role[vr]["added_keys"]),
  "prior_removed_keys_intersect_full_domain":0,
  "current_target_rows_by_source_face":target_rows_by_role[vr].tolist(),
  "current_target_faces":np.asarray([volumes[vr]["target"][int(r)] for r in target_rows_by_role[vr]],np.int64).tolist(),
  "target_faces_after":np.asarray([f for _,f in target_updates[vr]],np.int64).tolist()} for vr in ("Skin","Fat")},
 "source_and_volume_point_views_bitwise_unchanged":True,"all maps use exact face keys or sorted tetrahedron+material keys":True}
rebase_path.write_text(json.dumps(rebase_receipt,indent=2,sort_keys=True,default=json_default)+"\n")
def write_npz_exact(parent_ref,dest,arrays,label):
    parent_path=Path(parent_ref["path"])
    if sha(parent_path)!=parent_ref["sha256"]:
        raise ValueError(label+" immediate parent archive pin mismatch")
    parent_arrays=B.load_npz(B.check_ref(parent_ref,label+" immediate parent"))
    if not set(parent_arrays).issubset(set(arrays)):
        raise ValueError(label+" output archive drops one or more immediate-parent keys")
    clean={}
    for key,value in arrays.items():
        a=np.asarray(value)
        if a.dtype.kind=="O":
            raise ValueError(label+" output contains an object-typed lane: "+key)
        clean[key]=a
    with Path(dest).open("wb") as stream:
        np.savez_compressed(stream,**clean)
    readback=B.load_npz(dest)
    if set(readback)!=set(clean):
        raise ValueError(label+" serialized archive key set differs")
    for key,a in clean.items():
        b=np.asarray(readback[key])
        if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes(order="C")!=b.tobytes(order="C"):
            raise ValueError(label+" serialized dtype/shape/value mismatch: "+key)
    return readback
def assert_existing_archive_exact(path,arrays,label):
    if sha(path)!=REUSE_EXPECTED[Path(path).name]: raise ValueError(label+" reused archive hash mismatch")
    readback=B.load_npz(path)
    if set(readback)!=set(arrays): raise ValueError(label+" reused archive key set mismatch")
    for key,want in arrays.items():
        a=np.asarray(want);b=np.asarray(readback[key])
        if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes(order="C")!=b.tobytes(order="C"):
            raise ValueError(label+" reused archive fresh-composition byte mismatch for "+key)
    return readback
srcpath=REUSE_DIR/"candidate-I-typed-current.npz"
source_read=assert_existing_archive_exact(srcpath,surface_arrays,"I typed surface")
source_ref=pin(srcpath)
role_refs={r:dict(st["source_surfaces"][r]) for r in ("O","I","D")}
role_refs["I"]=source_ref
args=SimpleNamespace(role="I",maximum_local_volume_fraction=.01,maximum_global_volume_fraction=1e-4)
output_volumes={};volume_metrics={};volume_refs={}
for vr in ("Skin","Fat"):
    arrays,metrics=B.compose_volume_batch(vr,volumes[vr],accepted,source_ref,role_refs,args)
    if vr=="Skin":
        arrays["connected_boundary_patch_empty_target_retirement_cell_rows"]=selected_by_role[vr].copy()
        arrays["connected_boundary_patch_empty_target_retirement_source_face_rows"]=SOURCE_ROWS.copy()
        arrays["connected_boundary_patch_empty_target_retirement_exact_det6_delta"]=np.asarray([str(patch_volumes[vr]["exact_det6_delta"])])
        arrays["connected_boundary_patch_empty_target_retirement_new_target_faces"]=np.asarray(plan["roles"][vr]["new_target_faces_volume_rows"],np.int64)
        arrays["connected_boundary_patch_empty_target_retirement_semantics"]=np.asarray(["explicit one-cell local shell retirement; full closed region retained/replayed"])
    output_volumes[vr]=arrays
    volume_metrics[vr]=metrics
    vpath=REUSE_DIR/(vr.lower()+"-volume-proposal.npz")
    output_volumes[vr]=assert_existing_archive_exact(vpath,arrays,vr+" volume")
    volume_refs[vr]=pin(vpath)
combined_bad_after=sum(int(volume_metrics[r]["native_bad_after"]) for r in ("Skin","Fat"))
if combined_bad_after>=combined_bad_before:
    raise ValueError("paired whole-role native bad count did not strictly decrease")

# Independent output replay from original parent arrays and serialized candidate assets.
def independent_replay():
    parent_surface=B.load_npz(B.check_ref(st["source_surfaces"]["I"],"replay parent I"))
    out_surface=B.load_npz(srcpath)
    pfaces=np.asarray(parent_surface["faces"],np.int64)
    ofaces=np.asarray(out_surface["faces"],np.int64)
    expected=pfaces.copy();expected[SOURCE_ROWS]=NEW_SOURCE
    if not np.array_equal(ofaces,expected):
        raise ValueError("independent source face-row replay mismatch")
    if not np.array_equal(np.asarray(parent_surface["points_world_m_f32"],np.float32).view(np.uint32),
                          np.asarray(out_surface["points_world_m_f32"],np.float32).view(np.uint32)):
        raise ValueError("independent source point-coordinate replay mismatch")
    if not np.array_equal(np.asarray(parent_surface["points_world_m"],np.float64).view(np.uint8),
                          np.asarray(out_surface["points_world_m"],np.float64).view(np.uint8)):
        raise ValueError("independent full-precision source point replay mismatch")
    if not np.array_equal(oriented_edge_chain(pfaces[SOURCE_ROWS]),oriented_edge_chain(NEW_SOURCE)):
        raise ValueError("independent source oriented patch boundary mismatch")
    # Revalidate the 3D proof on the exact local patch and current point bytes.
    if not np.array_equal(np.asarray(parent_surface["points_world_m_f32"],np.float32).view(np.uint32),
                          np.asarray(proof_surface["points_world_m_f32"],np.float32).view(np.uint32)):
        raise ValueError("independent source 3D proof coordinate pin mismatch")
    if strict_distance["status"]!="PASS" or strict_distance_upper>STRICT_DISTANCE_CAP_M:
        raise ValueError("independent source 50um distance replay failed")
    if not np.array_equal(np.asarray(strict_distance["old_faces"],np.int64),OLD_SOURCE) or not np.array_equal(np.asarray(strict_distance["new_faces"],np.int64),NEW_SOURCE):
        raise ValueError("independent source 50um distance replay face binding failed")
    changed=set(map(int,SOURCE_ROWS))
    old_union=set(tuple(map(int,x)) for r in SOURCE_ROWS for x in A.rows_union(parent_surface,int(r)))
    old_anc=set(tuple(map(int,x)) for r in SOURCE_ROWS for x in A.rows_anc(parent_surface,int(r)))
    p_qf=np.asarray(parent_surface["quality_star_face_output_to_input_row"],np.int64)
    o_qf=np.asarray(out_surface["quality_star_face_output_to_input_row"],np.int64)
    uoff=np.asarray(out_surface["quality_star_patch_union_offsets"],np.int64)
    aoff=np.asarray(out_surface["face_ancestry_offsets"],np.int64)
    for r in range(len(pfaces)):
        if r in changed:
            if int(o_qf[r])!=-1:raise ValueError("changed source face retains direct parent row")
            lo,hi=map(int,uoff[r:r+2])
            got={tuple(int(out_surface[k][j]) for k in ENG.TUN) for j in range(lo,hi)}
            if got!=old_union:raise ValueError("changed source face typed patch union mismatch")
            lo,hi=map(int,aoff[r:r+2])
            got={tuple(int(out_surface[k][j]) for k in ENG.ANC) for j in range(lo,hi)}
            if got!=old_anc:raise ValueError("changed source face ancestry mismatch")
        else:
            if not np.array_equal(ofaces[r],pfaces[r]) or int(o_qf[r])!=int(p_qf[r]):
                raise ValueError("unrelated source face geometry/map changed")
            if A.rows_union(out_surface,r)!=A.rows_union(parent_surface,r) or A.rows_anc(out_surface,r)!=A.rows_anc(parent_surface,r):
                raise ValueError("unrelated source face typed ancestry changed")
    if not np.array_equal(np.asarray(parent_surface["source_input_vertex_quotient"]),
                          np.asarray(out_surface["source_input_vertex_quotient"])):
        raise ValueError("input-domain source vertex quotient changed")
    if not np.array_equal(np.asarray(out_surface["source_current_candidate_path"]).reshape(-1),
                          np.asarray([str(Path(st["source_surfaces"]["I"]["path"]).resolve())])):
        raise ValueError("output source current-candidate path must point to immediate parent")
    if not np.array_equal(np.asarray(out_surface["source_current_candidate_sha256"]).reshape(-1),
                          np.asarray([str(st["source_surfaces"]["I"]["sha256"])])):
        raise ValueError("output source current-candidate digest must pin immediate parent")
    # Every unchanged or output-domain point lane remains tied to the parent point namespace.
    source_mutable={"faces","quality_star_face_output_to_input_row","quality_star_patch_union_offsets",
        *ENG.TUN,"face_ancestry_offsets",*ENG.ANC,*ENG.TSC,"face_parent_source_face_rows","face_exact_source_row",
        "source_current_candidate_path","source_current_candidate_sha256","typed_parent_candidate_path",
        "typed_parent_candidate_sha256","connected_boundary_patch_batch_semantics",
        "connected_boundary_patch_operation_face_rows","connected_boundary_patch_operation_face_parent_offsets",
        "connected_boundary_patch_operation_face_parent_union","connected_boundary_patch_operation_face_ancestry_offsets",
        "connected_boundary_patch_operation_face_ancestry","connected_boundary_patch_operation_patch_face_offsets",
        "connected_boundary_patch_operation_patch_face_rows"}
    for key in parent_surface:
        if key in source_mutable or key not in out_surface:continue
        a=np.asarray(parent_surface[key]);b=np.asarray(out_surface[key])
        if a.dtype.kind in "f" and b.dtype.kind in "f":
            if not np.array_equal(a.view(np.uint8),b.view(np.uint8)):raise ValueError("unrelated I array changed: "+key)
        elif not np.array_equal(a,b):
            raise ValueError("unrelated I array changed: "+key)
    result_roles={}
    for vr in ("Skin","Fat"):
        c=volumes[vr]
        parent_arr=B.load_npz(B.check_ref(st[vr+"_current"],"replay parent "+vr))
        out=B.load_npz(volume_refs[vr]["path"])
        t0=np.asarray(parent_arr["tetrahedra"],np.int64)
        mat0=np.asarray(parent_arr["material_indices"])
        selected=selected_by_role[vr]
        keep=np.ones(len(t0),bool);keep[selected]=False
        newt=new_tets_by_role[vr]
        newmat=np.full(len(newt),int(mat0[selected[0]]),dtype=mat0.dtype)
        expected_t=np.vstack((t0[keep],newt)).astype(np.int64)
        expected_mat=np.r_[mat0[keep],newmat]
        if not np.array_equal(np.asarray(out["tetrahedra"],np.int64),expected_t):
            raise ValueError(vr+" independent connectivity replay mismatch")
        if not np.array_equal(np.asarray(out["material_indices"]),expected_mat):
            raise ValueError(vr+" independent material ancestry replay mismatch")
        if not np.array_equal(np.asarray(out["points_world_m"],np.float32).view(np.uint32),
                              np.asarray(parent_arr["points_world_m"],np.float32).view(np.uint32)):
            raise ValueError(vr+" volume point bytes changed")
        output_to_input=np.r_[np.flatnonzero(keep).astype(np.int64),np.full(len(newt),-1,np.int64)]
        input_to_output=np.full(len(t0),-1,np.int64);input_to_output[keep]=np.arange(int(keep.sum()),dtype=np.int64)
        if not np.array_equal(np.asarray(out["output_cell_to_input_cell"],np.int64),output_to_input):
            raise ValueError(vr+" direct output-cell ancestry map mismatch")
        if not np.array_equal(np.asarray(out["input_cell_to_output_cell"],np.int64),input_to_output):
            raise ValueError(vr+" direct input-cell ancestry map mismatch")
        if not np.array_equal(np.asarray(out["connected_boundary_patch_operation_cell_rows"],np.int64),selected):
            raise ValueError(vr+" operation seed-cell ledger mismatch")
        rpo=np.asarray(out["replacement_parent_offsets"],np.int64)
        rpr=np.asarray(out["replacement_parent_cell_rows"],np.int64)
        if len(rpo)!=len(newt)+1 or rpo[0]!=0 or rpo[-1]!=len(rpr):
            raise ValueError(vr+" replacement-parent CSR malformed")
        for j in range(len(newt)):
            lo,hi=map(int,rpo[j:j+2])
            if set(map(int,rpr[lo:hi]))!=set(map(int,selected)):
                raise ValueError(vr+" replacement tet lacks exact seed-parent union")
        if vr=="Skin":
            if len(newt)!=0:raise ValueError("Skin retirement must have an empty local target")
            if not np.array_equal(np.asarray(out["connected_boundary_patch_empty_target_retirement_cell_rows"],np.int64),selected):
                raise ValueError("Skin explicit empty-target retirement cell list mismatch")
            if not np.array_equal(np.asarray(out["connected_boundary_patch_empty_target_retirement_source_face_rows"],np.int64),SOURCE_ROWS):
                raise ValueError("Skin empty-target retirement lacks exact source face rows")
        target_key=str(c["target_key"])
        target0=np.asarray(c["target"],np.int64)
        expected_target=target0.copy()
        for tr,nf in target_updates[vr]:expected_target[int(tr)]=np.asarray(nf,np.int64)
        target1=np.asarray(out[target_key],np.int64)
        if not np.array_equal(target1,expected_target):raise ValueError(vr+" exact current target-face replay mismatch")
        actual_boundary=np.asarray(T.m.boundary(expected_t),np.int64)
        if ENG.chain(np.asarray(out["boundary_faces"],np.int64))!=ENG.chain(actual_boundary):
            raise ValueError(vr+" actual target-boundary chain differs from output tets")
        if ENG.chain(actual_boundary)!=ENG.chain(target1):
            raise ValueError(vr+" full volume does not realize the complete changed target chain")
        if len(np.unique(np.sort(expected_t,axis=1),axis=0))!=len(expected_t):
            raise ValueError(vr+" output contains duplicate tetrahedra")
        det,q=M.measurements(c["points"],expected_t)
        if np.any(det<=0):raise ValueError(vr+" output has a nonpositive native determinant")
        passed=np.asarray(M.passes(det,q),bool)
        oldpassed=np.asarray(M.passes(c["det"],c["q"]),bool)
        old_bad=int(np.count_nonzero(~oldpassed));new_bad=int(np.count_nonzero(~passed))
        if new_bad>old_bad:raise ValueError(vr+" whole-role native bad count regressed")
        if np.any(oldpassed[keep]&~passed[:int(keep.sum())]):
            raise ValueError(vr+" a retained formerly-good tetrahedron regressed")
        if len(newt) and not np.all(passed[-len(newt):]):
            raise ValueError(vr+" new replacement cells fail native quality")
        flux0=int(ENG.exact_boundary_flux(c["points"],target0))
        flux1=int(ENG.exact_boundary_flux(c["points"],target1))
        delta=flux1-flux0
        if delta!=int(patch_volumes[vr]["exact_det6_delta"]):
            raise ValueError(vr+" full-volume exact target-flux delta differs from seed operation")
        # Reconstruct the separately declared conforming domain and its exact cap.
        fullrows=full_domain_rows_by_role[vr]
        selected_set=set(map(int,selected))
        oldfull=t0[fullrows]
        localkeep=np.asarray([int(r) not in selected_set for r in fullrows],bool)
        region_new=np.vstack((oldfull[localkeep],newt)).astype(np.int64)
        oldbd=np.asarray(T.m.boundary(oldfull),np.int64)
        newbd=np.asarray(ENG.oriented_boundary_replace(oldbd,target_face_replacements[vr]),np.int64)
        if ENG.chain(np.asarray(T.m.boundary(region_new),np.int64))!=ENG.chain(newbd):
            raise ValueError(vr+" retained full-domain regional target chain mismatch")
        oldsum=exact_det_sum(c["points"],oldfull);newsum=exact_det_sum(c["points"],region_new)
        regionflux=int(ENG.exact_boundary_flux(c["points"],newbd))
        if oldsum!=int(plan["roles"][vr]["full_domain_old_exact_det6_sum"]):
            raise ValueError(vr+" full-domain old exact det6 sum differs from 990 receipt")
        if newsum!=regionflux or newsum-oldsum!=delta:
            raise ValueError(vr+" full-domain exact signed volume/flux replay mismatch")
        region_fraction=abs(delta)/float(oldsum)
        global_fraction=abs(delta)/float(flux0)
        if region_fraction>.01 or global_fraction>1e-4:
            raise ValueError(vr+" full-domain or global exact volume budget replay failed")
        result_roles[vr]={"native_bad_before":old_bad,"native_bad_after":new_bad,
            "new_seed_cells_native_pass":True,"retained_formerly_good_cells_nonregressing":True,
            "full_target_chain":True,"whole_exact_target_flux_delta":str(delta),
            "full_domain_cells":int(len(fullrows)),"full_domain_native_pass":True,
            "full_domain_old_exact_sum":str(oldsum),"full_domain_new_exact_sum":str(newsum),
            "full_domain_exact_delta":str(delta),"local_volume_fraction":region_fraction,
            "global_volume_fraction":global_fraction,"empty_target_retirement":vr=="Skin"}
    if sum(int(result_roles[r]["whole_exact_target_flux_delta"]) for r in ("Skin","Fat"))!=0:
        raise ValueError("independent paired Skin/Fat exact target-flux deltas do not cancel")
    if sum(int(result_roles[r]["native_bad_after"]) for r in ("Skin","Fat"))>=sum(int(result_roles[r]["native_bad_before"]) for r in ("Skin","Fat")):
        raise ValueError("independent paired whole-role native bad count did not strictly decrease")
    return {"schema":"h706.backskin-cap-candidate-independent-replay.v1",
        "status":"PASS_INDEPENDENT_BACKSKIN_CAP_REPLAY",
        "source_face_rows":SOURCE_ROWS.tolist(),"source_old_faces":OLD_SOURCE.tolist(),"source_new_faces":NEW_SOURCE.tolist(),
        "source_topology_preserved":True,"source_point_bytes_unchanged":True,
        "bidirectional_3d_source_patch_distance":float(strict_distance_upper),
        "bidirectional_3d_distance_cap_m":STRICT_DISTANCE_CAP_M,"distance_report":pin(STRICT_DISTANCE),
        "legacy_250um_feasibility_report":pin(DISTANCE),
        "oriented_neighbor_replay":True,"full_typed_source_face_ancestry":True,
        "outer_O_source_unchanged":True,"all_source_and_volume_points_unchanged":True,
        "roles":result_roles,"paired_exact_volume_cancellation":True,
        "full_global_target_chains":True,"native_thresholds_unchanged":True,
        "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}

replay=independent_replay()
replay_path=OUT/"independent-replay.json"
replay_path.write_text(json.dumps(replay,indent=2,sort_keys=True,default=json_default)+"\n")
journal={
    "schema":"h706.backskin-cap-retained-domain-operation.v1",
    "status":"PASS_LOCAL_AND_FULL_DOMAIN_REPLAY_PENDING_775_808",
    "parent_state":pin(PARENT),"base_parent984":pin(BASE_PARENT),"source_I_parent":pin(st["source_surfaces"]["I"]["path"]),
    "source_owner":pin(OWNER),"closure_plan":pin(PLAN),"local_feasibility_receipt":pin(LOCAL),"exact_key_rebase_receipt":pin(rebase_path),
    "distance_report":pin(STRICT_DISTANCE),"legacy_250um_feasibility_report":pin(DISTANCE),"oriented_equivalence_receipt":pin(ORIENT),
    "operation":{"role":"I","kind":"2-to-2 source edge flip","source_edge":SOURCE_EDGE.tolist(),
       "source_face_rows":SOURCE_ROWS.tolist(),"old_source_faces":OLD_SOURCE.tolist(),"new_source_faces":NEW_SOURCE.tolist(),
       "all_source_points_fixed":True,"outer_O_back_fixed":True,"inner_back_3d_distance_upper_m":float(strict_distance_upper),
       "inner_back_3d_distance_cap_m":STRICT_DISTANCE_CAP_M,"distance_method":strict_distance["method"]["description"],
       "independent_3d_distance_replay":pin(STRICT_DISTANCE),"legacy_250um_feasibility_report":pin(DISTANCE)},
    "roles":{vr:{"retired_seed_rows":selected_by_role[vr].tolist(),
       "old_seed_tets":np.asarray(volumes[vr]["tets"][selected_by_role[vr]],np.int64).tolist(),
       "new_seed_tets":new_tets_by_role[vr].tolist(),"replacement_parent_union":selected_by_role[vr].tolist(),
       "full_closed_region_rows":full_domain_metrics[vr]["full_rows"].tolist(),
       "full_closed_region_cell_count":full_domain_metrics[vr]["cell_count"],
       "full_closed_region_point_count":full_domain_metrics[vr]["point_count"],
       "full_closed_region_exact_delta_det6":str(full_domain_metrics[vr]["exact_delta_det6"]),
       "local_volume_fraction":full_domain_metrics[vr]["local_volume_fraction"],
       "global_volume_fraction":full_domain_metrics[vr]["global_volume_fraction"],
       "old_native_bad":full_domain_metrics[vr]["old_bad_cells"],"new_native_bad":0,
       "full_region_old_boundary_closed":True,"full_region_new_boundary_closed":True,
       "empty_target_retirement":vr=="Skin",
       "target_row_replacements":[{"row":int(r),"old":np.asarray(volumes[vr]["target"][int(r)],np.int64).tolist(),
                                   "new":np.asarray(f,np.int64).tolist()} for r,f in target_updates[vr]],
       "material":int(np.asarray(volumes[vr]["mat"])[selected_by_role[vr][0]])} for vr in ("Skin","Fat")},
    "paired_exact_signed_delta_det6":"0","independent_replay_status":replay["status"],
    "global_185_audit":"PENDING","source_crossing_audit":"PENDING",
    "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"
}
journal_path=OUT/"operation-journal.json"
journal_path.write_text(json.dumps(journal,indent=2,sort_keys=True,default=json_default)+"\n")
# State is an unqualified source-only proposal; no parent PASS pointer is copied as current evidence.
source_ref=pin(srcpath)
role_refs["I"]=source_ref
so=dict(st)
so["source_surfaces"]=role_refs
for vr in ("Skin","Fat"):
    old=dict(so[vr+"_current"])
    new={**old,**volume_refs[vr],"cells":int(len(output_volumes[vr]["tetrahedra"])),
         "native_quality_failures":int(volume_metrics[vr]["native_bad_after"]),"nonpositive":0,
         "boundary_source_binding_status":"PENDING_FRESH_775_808_SOURCE_AUDIT"}
    so[vr+"_current"]=new
    so[vr]=dict(new)
for key in ("exact_context","independent_replay","independent_point_replay","independent_allrole_local_replay",
            "independent_target_replay","source_evidence","source_audit","source_crossing_audit",
            "native_qualification","mass_mapping_receipt","boundary_source_binding_status"):
    so.pop(key,None)
so.update({"source_state":pin(PARENT),"source_audit_status":"PENDING_GLOBAL_185_AUDIT",
    "source_crossing_audit_status":"PENDING_GLOBAL_185_AUDIT","physical_admission":False,"native_steps":0,
    "mass_mapping":"NOT_RUN","status":"SOURCE_ONLY_BACKSKIN_CAP_TRANSFER_PENDING_775_808_AND_MASS",
    "backskin_cap_transfer":{"source_owner":pin(OWNER),"source_operation":journal["operation"],
       "parent_state":pin(PARENT),"base_parent984":pin(BASE_PARENT),"retained_domain_plan":pin(PLAN),"local_proof":pin(LOCAL),"exact_key_rebase_receipt":pin(rebase_path),
       "3d_distance_report":pin(STRICT_DISTANCE),"legacy_250um_feasibility_report":pin(DISTANCE),"oriented_equivalence_receipt":pin(ORIENT),
       "operation_journal":pin(journal_path),"independent_replay":pin(replay_path),
       "independent_replay_status":replay["status"],"global_185_audit":"REQUIRED_NOT_RUN",
       "paired_exact_signed_delta_det6":"0","all_point_coordinates_unchanged":True,
       "outer_O_back_source_fixed":True,"native_thresholds_unchanged":True,
       "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}})
if "current_verified_state_20261031" in root:
    stateout=dict(root);stateout["current_verified_state_20261031"]=so
else:
    stateout=so
state_path=OUT/"candidate-state-proposal.json"
state_path.write_text(json.dumps(stateout,indent=2,sort_keys=True,default=json_default)+"\n")
report={"schema":"h706.backskin-cap-retained-domain-candidate.v1",
    "status":"PASS_INDEPENDENT_REPLAY_PENDING_FRESH_775_808",
    "parent_state":pin(PARENT),"base_parent984":pin(BASE_PARENT),"source_candidate":source_ref,"volume_candidates":volume_refs,
    "candidate_state":pin(state_path),"closure_plan":pin(PLAN),"local_proof":pin(LOCAL),"exact_key_rebase_receipt":pin(rebase_path),
    "reused_composition_archives":{k:{"path":str((REUSE_DIR/k).resolve()),"sha256":sha(REUSE_DIR/k)} for k in REUSE_EXPECTED},
    "distance_report":pin(STRICT_DISTANCE),"legacy_250um_feasibility_report":pin(DISTANCE),"oriented_equivalence_receipt":pin(ORIENT),
    "operation_journal":pin(journal_path),"independent_replay":pin(replay_path),
    "independent_replay_status":replay["status"],"source_old_faces":OLD_SOURCE.tolist(),
    "source_new_faces":NEW_SOURCE.tolist(),"source_face_rows":SOURCE_ROWS.tolist(),
    "source_3d_symmetric_distance_upper_m":float(strict_distance_upper),
    "source_3d_distance_cap_m":STRICT_DISTANCE_CAP_M,"strict_independent_distance_replay":pin(STRICT_DISTANCE),
    "legacy_250um_feasibility_report":pin(DISTANCE),
    "role_metrics":volume_metrics,"full_domain_metrics":full_domain_metrics,
    "paired_quality_bad_before":combined_bad_before,"paired_quality_bad_after":combined_bad_after,
    "paired_exact_signed_delta_det6":"0","full_global_185_audit":"PENDING",
    "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
    "exact_key_rebase_policy":"source faces by canonical triangle keys/orientation; volume seed and full-domain cells by sorted tetrahedron vertices plus material integer; all arrays regenerated from immediate parent1002",
    "previous1002_change_disjointness":{"checked_by_exact_cell_key_and_source_face_key":True,"prior_source_removed_face_count":len(prior_removed_source),"prior_source_added_face_count":len(prior_added_source),"prior_source_vertex_intersection":0,"prior_volume_key_deltas":{vr:{"removed":len(prior_volume_delta_by_role[vr]["removed_keys"]),"added":len(prior_volume_delta_by_role[vr]["added_keys"]),"full_domain_intersection":0} for vr in ("Skin","Fat")},"parent984_base":pin(BASE_PARENT)},
    "elapsed_s":time.monotonic()-tic}
report_path=OUT/"report.json"
report_path.write_text(json.dumps(report,indent=2,sort_keys=True,default=json_default)+"\n")
print(json.dumps(report,indent=2,sort_keys=True,default=json_default),flush=True)
print("candidate_state",state_path,flush=True)
print("elapsed_s",time.monotonic()-tic,flush=True)
