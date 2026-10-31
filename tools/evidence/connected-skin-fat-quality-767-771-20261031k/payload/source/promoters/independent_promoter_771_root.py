#!/usr/bin/env python3
"""Disk-backed independent verifier/promoter for bounded 750 proposals.

The input state and proposal journal stay separate on disk. No verified-state is
written until source pins, output pins, lineage, exact cavity certificates,
readback replay, and inherited source-boundary status all pass.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PRODUCER750=HERE/"batch_interior_steiner_kernel_cone_quality_767_root.py"
EXPECTED_750_SHA256="c9d11fbcc2403755f283a9773053dd8e91c4d81dde8f3594c3f29be208a7da75"
def sha(path):
    return hashlib.file_digest(Path(path).open("rb"),"sha256").hexdigest()
def pin(path):
    p=Path(path)
    return {"path":str(p),"sha256":sha(p)}
def imp(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module
if sha(PRODUCER750)!=EXPECTED_750_SHA256:
    raise RuntimeError("pinned 750 producer SHA mismatch")
V=imp("steiner_767_for_independent_promoter",PRODUCER750)
if Path(V.__file__).resolve()!=PRODUCER750.resolve():
    raise RuntimeError("independent promoter loaded unexpected producer path")
STATE_WRAPPER="current_verified_state_20261031"
PASS_STATUS="PASS_INTERIOR_STEINER_KERNEL_CONE_BATCH"
NOOP_STATUSES={
    "NO_NATIVE_BAD_CELLS",
    "NO_VALID_BOUNDED_PARTIAL_FIXED_BOUNDARY_QUALITY_PROGRESS",
    "SKIPPED_GLOBAL_TIME_BUDGET",
}
EXPECTED_CONTEXT_SCHEMA="h706.combined-current-exact-surface-diff-audit.v1"
EXPECTED_CONTEXT_STATUS="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185"
EXPECTED_CONFIG_SCHEMA="h706.combined-current-exact-surface-diff-config.v1"
EXPECTED_ALL185_SCHEMA="existing-bone-record-remesh-barrier.v1"
EXPECTED_ALL185_COUNT=938272
EXPECTED_ALL185_NATIVE_SHA="e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5"
EXPECTED_ALL185_MANIFEST_SHA="0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c"
SURFACE_NAMES=("D","I","O")

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def _one(record,key):
    if not isinstance(record,dict) or key not in record:
        raise ValueError("missing pin field: "+key)
    return record[key]

def _verify_pin(record,expected_path=None,label="artifact"):
    path=Path(_one(record,"path"))
    digest=str(_one(record,"sha256"))
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual=sha(path)
    if digest!=actual:
        raise ValueError(f"{label} SHA mismatch: {path}")
    if expected_path is not None and path.resolve()!=Path(expected_path).resolve():
        raise ValueError(f"{label} path mismatch: {path}")
    return path,digest

def _unwrap_state(raw):
    if STATE_WRAPPER in raw:
        return raw[STATE_WRAPPER],True
    return raw,False

def _check_not_747(path,what):
    if "747" in str(path).lower():
        raise ValueError(f"{what} points to the unpromoted 747 branch: {path}")

def _scalar_int(arr,key):
    if key not in arr:
        raise ValueError("missing integer metadata array: "+key)
    x=np.asarray(arr[key])
    if x.size!=1:
        raise ValueError("integer metadata is not scalar: "+key)
    return int(x.reshape(-1)[0])

def _scalar_str(arr,key):
    if key not in arr:
        raise ValueError("missing string metadata array: "+key)
    x=np.asarray(arr[key])
    if x.size!=1:
        raise ValueError("string metadata is not scalar: "+key)
    return str(x.reshape(-1)[0])

def _source_boundary_check(role,entry,parent_npz):
    srcpath=Path(entry["path"])
    p=np.asarray(parent_npz["points_world_m"],np.float32)
    t=np.asarray(parent_npz["tetrahedra"],np.int64)
    mats=np.asarray(parent_npz["material_indices"])
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        raise ValueError(f"{role} source points are invalid")
    if "points_world_m_f32" in parent_npz and not np.array_equal(
            np.asarray(parent_npz["points_world_m_f32"],np.float32).view(np.uint32),p.view(np.uint32)):
        raise ValueError(f"{role} source F32 point alias differs from its bound point array")
    if t.ndim!=2 or t.shape[1]!=4 or mats.shape!=(len(t),):
        raise ValueError(f"{role} source cells/materials are invalid")
    declared_nonpositive=entry.get("nonpositive",parent_npz.get("nonpositive",None))
    if declared_nonpositive is None or int(declared_nonpositive)!=0:
        raise ValueError(f"{role} source does not inherit nonpositive=0")
    declared_nodes=entry.get("nodes")
    if declared_nodes is not None and int(declared_nodes)!=len(p):
        raise ValueError(f"{role} source node count differs from the verified state")
    declared_cells=entry.get("cells")
    if declared_cells is not None and int(declared_cells)!=len(t):
        raise ValueError(f"{role} source cell count differs from the verified state")
    target,target_key=V.boundary_target(parent_npz)
    actual=V.T.m.boundary(t)
    if "boundary_faces" in parent_npz and not np.array_equal(
            V.T.m.boundary_identity(np.asarray(parent_npz["boundary_faces"],np.int64)),
            V.T.m.boundary_identity(actual)):
        raise ValueError(f"{role} stored source boundary_faces differs from the volume boundary")
    if not np.array_equal(V.T.m.boundary_identity(actual),V.T.m.boundary_identity(target)):
        raise ValueError(f"{role} source boundary does not match {target_key}")
    bad,score,det,q=V.quality_score(p,t)
    if np.any(det<=0):
        raise ValueError(f"{role} source has a nonpositive measured cell")
    declared_bad=entry.get("native_quality_failures")
    if declared_bad is None or int(declared_bad)!=bad:
        raise ValueError(f"{role} source native-bad count differs from verified state")
    return {
        "source_path":str(srcpath),"points":p,"tets":t,"materials":mats,
        "target":target,"target_key":target_key,"bad":int(bad),"deficit":float(score),
        "native_positive":True,"nonpositive_inherited":0,
        "boundary_match":True,
    }

def _reconstruct_operations(parent_npz,readback,role):
    oldt=np.asarray(parent_npz["tetrahedra"],np.int64)
    oldmat=np.asarray(parent_npz["material_indices"])
    oldp=np.asarray(parent_npz["points_world_m"],np.float32)
    outt=np.asarray(readback["tetrahedra"],np.int64)
    outmat=np.asarray(readback["material_indices"])
    outp=np.asarray(readback["points_world_m"],np.float32)
    opcount=_scalar_int(readback,"quality_fixed_boundary_operation_count")
    if opcount<=0:
        raise ValueError(f"{role} accepted result has no recorded operations")
    offsets=np.asarray(readback["quality_fixed_boundary_operation_parent_offsets"],np.int64)
    rows=np.asarray(readback["quality_fixed_boundary_operation_parent_cell_rows"],np.int64)
    if offsets.shape!=(opcount+1,) or offsets[0]!=0 or offsets[-1]!=len(rows) or np.any(np.diff(offsets)<=0):
        raise ValueError(f"{role} operation-parent CSR is malformed")
    op_rows=[]
    removed=set()
    for oi in range(opcount):
        part=rows[offsets[oi]:offsets[oi+1]]
        if len(part)==0 or np.any(part<0) or np.any(part>=len(oldt)):
            raise ValueError(f"{role} operation {oi} has invalid old-cell rows")
        values=list(map(int,part))
        if values!=sorted(set(values)):
            raise ValueError(f"{role} operation {oi} old-cell rows are not sorted unique")
        if removed.intersection(values):
            raise ValueError(f"{role} operations are not disjoint")
        removed.update(values);op_rows.append(values)
    removed_rows=np.asarray(sorted(removed),np.int64)
    retained_mask=np.ones(len(oldt),bool);retained_mask[removed_rows]=False
    retained=np.flatnonzero(retained_mask)
    replacement=np.asarray(readback["replacement_tetrahedra"],np.int64)
    if replacement.ndim!=2 or replacement.shape[1]!=4:
        raise ValueError(f"{role} replacement connectivity is malformed")
    tet_offsets=np.asarray(readback["replacement_tet_offsets"],np.int64)
    if (tet_offsets.shape!=(opcount+1,) or tet_offsets[0]!=0 or
            tet_offsets[-1]!=len(replacement) or np.any(np.diff(tet_offsets)<=0)):
        raise ValueError(f"{role} replacement tet offsets are malformed")
    point_rows=np.asarray(readback["quality_steiner_point_output_rows"],np.int64)
    seeds=np.asarray(readback["quality_fixed_boundary_operation_seed_rows"],np.int64)
    methods=np.asarray(readback["quality_fixed_boundary_operation_methods"])
    selection=np.asarray(readback["quality_steiner_cavity_selection_methods"])
    if any(x.shape!=(opcount,) for x in (point_rows,seeds,methods,selection)):
        raise ValueError(f"{role} operation metadata counts disagree")
    if _scalar_int(readback,"quality_steiner_operation_count")!=opcount:
        raise ValueError(f"{role} Steiner operation count alias mismatch")
    if len(outp)!=len(oldp)+opcount:
        raise ValueError(f"{role} output point count does not match appended Steiner points")
    if len(outt)!=len(retained)+len(replacement):
        raise ValueError(f"{role} retained/replacement cell counts do not explain output")
    if not np.array_equal(outt[len(retained):],replacement):
        raise ValueError(f"{role} output replacement suffix differs from recorded connectivity")
    if not np.array_equal(outmat[:len(retained)],oldmat[retained]):
        raise ValueError(f"{role} retained material prefix is not canonical")
    ops=[]
    for oi,old_rows in enumerate(op_rows):
        start,end=map(int,tet_offsets[oi:oi+2])
        block=replacement[start:end]
        seed=int(seeds[oi])
        if seed not in old_rows:
            raise ValueError(f"{role} operation seed is not in its old-cell union")
        old_materials=np.unique(oldmat[np.asarray(old_rows,np.int64)])
        block_materials=np.unique(outmat[len(retained)+start:len(retained)+end])
        if len(old_materials)!=1 or len(block_materials)!=1 or int(old_materials[0])!=int(block_materials[0]):
            raise ValueError(f"{role} operation {oi} changes or mixes material")
        apex_row=int(point_rows[oi])
        if apex_row!=len(oldp)+oi:
            raise ValueError(f"{role} operation {oi} has a noncanonical Steiner point row")
        if not np.isfinite(outp[apex_row]).all():
            raise ValueError(f"{role} operation {oi} has a nonfinite Steiner point")
        ops.append({
            "old_cell_rows":old_rows,
            "new_tetrahedra":block.tolist(),
            "replacement_tet_range":[start,end],
            "seed_cell_row":seed,
            "proposal_method":str(methods[oi]),
            "candidate_selection_method":str(selection[oi]),
            "apex_point_output_row":apex_row,
            "apex_point_f32":outp[apex_row].tolist(),
            "material_index":int(block_materials[0]),
        })
    return retained,ops

def _crosscheck_journal_operations(role,role_result,ops):
    listed=role_result.get("accepted_operations")
    if not isinstance(listed,list) or len(listed)!=len(ops):
        raise ValueError(f"{role} journal operation list does not match NPZ operation count")
    for i,(saved,rebuilt) in enumerate(zip(listed,ops)):
        checks=(
            ("old_cell_rows",rebuilt["old_cell_rows"]),
            ("new_tetrahedra",rebuilt["new_tetrahedra"]),
            ("seed_cell_row",rebuilt["seed_cell_row"]),
            ("proposal_method",rebuilt["proposal_method"]),
            ("candidate_selection_method",rebuilt["candidate_selection_method"]),
            ("apex_point_output_row",rebuilt["apex_point_output_row"]),
            ("material_index",rebuilt["material_index"]),
        )
        for key,expected in checks:
            if key not in saved or saved[key]!=expected:
                raise ValueError(f"{role} journal operation {i} differs from NPZ: {key}")
        if not np.array_equal(np.asarray(saved.get("apex_point_f32"),np.float32),
                              np.asarray(rebuilt["apex_point_f32"],np.float32)):
            raise ValueError(f"{role} journal operation {i} Steiner point differs from NPZ")


def _source_pins(s):
    rows=s.get("source_surfaces")
    if not isinstance(rows,dict) or set(rows)!=set(SURFACE_NAMES):
        raise ValueError("state lacks exact D/I/O source surface pins")
    out={}
    for n in SURFACE_NAMES:
        p,h=_verify_pin(rows[n],label="source surface "+n);_check_not_747(p,"source surface")
        out[n]={"path":str(p.resolve()),"sha256":h}
    return out

def _context_anchor(p,s,spins):
    rec=s.get("exact_context")
    if not isinstance(rec,dict):return None
    rp,rh=_verify_pin(rec,label="exact-context report");_check_not_747(rp,"exact-context")
    r=json.loads(rp.read_text())
    if r.get("schema")!=EXPECTED_CONTEXT_SCHEMA or r.get("status")!=EXPECTED_CONTEXT_STATUS:
        raise ValueError("inherited exact-context report is not the all185 PASS")
    if r.get("physical_admission") is not False or int(r.get("native_steps",-1))!=0:
        raise ValueError("exact-context report admission metadata mismatch")
    cp,ch=_verify_pin(r.get("config"),label="exact-context config")
    c=json.loads(cp.read_text())
    if c.get("schema")!=EXPECTED_CONFIG_SCHEMA or set(c.get("candidate_surfaces",{}))!=set(SURFACE_NAMES):
        raise ValueError("exact-context candidate config is malformed")
    for n in SURFACE_NAMES:
        q,h=_verify_pin(c["candidate_surfaces"][n],label="context candidate "+n)
        if {"path":str(q.resolve()),"sha256":h}!=spins[n]:raise ValueError("context candidate surface pin mismatch: "+n)
        rr=r.get("roles",{}).get(n,{})
        q,h=_verify_pin(rr.get("candidate"),label="context role candidate "+n)
        if {"path":str(q.resolve()),"sha256":h}!=spins[n]:raise ValueError("context role candidate mismatch: "+n)
        _verify_pin(rr.get("baseline"),label="context role baseline "+n)
        if rr.get("points",{}).get("f32_coordinate_bytes_unchanged") is not True:
            raise ValueError("context F32 identity proof missing: "+n)
    a=r.get("all185_receipt",{})
    if (a.get("schema")!=EXPECTED_ALL185_SCHEMA or int(a.get("record_count",-1))!=EXPECTED_ALL185_COUNT or
        a.get("native_sha256")!=EXPECTED_ALL185_NATIVE_SHA or
        a.get("pinned_manifest_sha256")!=EXPECTED_ALL185_MANIFEST_SHA):
        raise ValueError("all185 exact-context receipt pin mismatch")
    return {"state":pin(p),"report":{"path":str(rp.resolve()),"sha256":rh},
            "config":{"path":str(cp.resolve()),"sha256":ch},"status":r["status"],
            "record_count":EXPECTED_ALL185_COUNT,"native_sha256":a["native_sha256"],
            "manifest_sha256":a["pinned_manifest_sha256"]}

def _authenticated_source_proof(path,state):
    pins=_source_pins(state);seen=set();chain=[];p=Path(path);s=state
    for _ in range(40):
        p=p.resolve()
        if p in seen:raise ValueError("source proof chain cycle")
        seen.add(p);_check_not_747(p,"source proof state")
        if _source_pins(s)!=pins:raise ValueError("source surface pins changed in proof chain")
        if "surface_crossings" in s and int(s["surface_crossings"])!=0:raise ValueError("source proof records crossings")
        if "source_metadata_current_step_only" in s and s["source_metadata_current_step_only"] is not True:
            raise ValueError("source proof is not current-step-only")
        chain.append({"path":str(p),"sha256":sha(p),"status":s.get("boundary_source_binding_status")})
        anchor=_context_anchor(p,s,pins)
        if anchor:return {"status":"PASS_AUTHENTICATED_INHERITED_EXACT_CONTEXT","source_surfaces":pins,"anchor":anchor,"chain":chain}
        link=s.get("source_crossing_evidence_inherited_from_unchanged_surfaces")
        if not isinstance(link,dict):link=s.get("source_state")
        if not isinstance(link,dict):raise ValueError("no authenticated exact-context anchor in source proof chain")
        pp,_=_verify_pin(link,label="source proof parent");_check_not_747(pp,"source proof parent")
        raw=json.loads(pp.read_text());s,_=_unwrap_state(raw);p=pp
    raise ValueError("source proof chain depth exceeded")

def _facekey(f):return tuple(sorted(map(int,f)))

def _source_crosswalk(role,pins,vol):
    from collections import Counter
    for n in SURFACE_NAMES:
        p,h=_verify_pin(pins[n],label="typed source "+n)
        for k,v in ((f"typed_{n}_candidate_path",str(p)),(f"typed_{n}_candidate_sha256",h)):
            if k not in vol or str(np.asarray(vol[k]).reshape(-1)[0])!=v:raise ValueError("typed volume binding mismatch: "+k)
    xyz=np.asarray(vol.get("points_world_m_f32",vol["points_world_m"]),np.float32)
    target=np.asarray(vol.get("desired_boundary_faces",V.boundary_target(vol)[0]),np.int64)
    keys=[_facekey(f) for f in target];counts=Counter(keys)
    if any(v!=1 for v in counts.values()):raise ValueError("duplicate desired boundary face")
    idx={k:i for i,k in enumerate(keys)}
    roles=("O","I") if role=="Skin" else ("I","D") if role=="Fat" else None
    if roles is None:raise ValueError("invalid volume role")
    prefix=role.lower()+"_current_target_rebind_";covered=[];surface_counts={};orient={}
    for n in roles:
        surf=load_npz(pins[n]["path"]);sp=np.asarray(surf["points_world_m_f32"],np.float32);sf=np.asarray(surf["faces"],np.int64)
        mapkey=prefix+n+"_point_to_volume_rows";m=np.asarray(vol[mapkey],np.int64)
        if m.shape!=(len(sp),):raise ValueError("surface-to-volume point map shape mismatch")
        used=np.unique(sf.reshape(-1));present=np.flatnonzero(m>=0)
        if np.any(m < -1) or np.any(m[present]>=len(xyz)) or np.any(used>=len(sp)):raise ValueError("invalid source point map row")
        if np.any(m[used]<0) or len(np.unique(m[used]))!=len(used):raise ValueError("used source point map not injective")
        if not np.array_equal(sp[present].view(np.uint32),xyz[m[present]].view(np.uint32)):raise ValueError("source F32 coordinates differ")
        signs=[]
        for f in m[sf]:
            key=_facekey(f)
            if counts.get(key)!=1:raise ValueError("source face missing from desired boundary")
            ref=target[idx[key]].tolist();perm=[ref.index(int(v)) for v in f]
            inv=sum(perm[i]>perm[j] for i in range(3) for j in range(i+1,3));signs.append(-1 if inv%2 else 1)
            covered.append(key)
        if len(set(signs))>1:raise ValueError("source surface winding inconsistent")
        orient[n]=signs[0] if signs else 0;surface_counts[n]=len(sf)
    wkey=prefix+"wall_faces_volume_rows";ckey=prefix+"wall_face_points_world_m_f32"
    w=np.asarray(vol[wkey],np.int64);wc=np.asarray(vol[ckey],np.float32)
    if w.ndim!=2 or w.shape[1]!=3 or wc.shape!=(len(w),3,3):raise ValueError("wall crosswalk shape mismatch")
    if np.any(w<0) or np.any(w>=len(xyz)) or not np.array_equal(wc.view(np.uint32),xyz[w].view(np.uint32)):
        raise ValueError("wall crosswalk F32 mismatch")
    ws=[]
    for f in w:
        key=_facekey(f)
        if counts.get(key)!=1:raise ValueError("wall face missing from desired boundary")
        ref=target[idx[key]].tolist();perm=[ref.index(int(v)) for v in f]
        inv=sum(perm[i]>perm[j] for i in range(3) for j in range(i+1,3));ws.append(-1 if inv%2 else 1);covered.append(key)
    if len(set(ws))>1 or len(set(covered))!=len(covered) or set(covered)!=set(keys):
        raise ValueError("source/wall surfaces do not cover target boundary exactly")
    return {"status":"PASS_FRESH_TYPED_SOURCE_VOLUME_CROSSWALK","surface_faces":surface_counts,
            "wall_faces":len(w),"target_faces":len(target),"orientation":orient,
            "wall_orientation":ws[0] if ws else 0}

def verify_role(role,parent_state,entry,role_result,state_sha,source_surface_pins=None):
    status=str(role_result.get("status",""))
    source_path=Path(entry["path"])
    _check_not_747(source_path,f"{role} source candidate")
    source_record=role_result.get("input")
    if status!="SKIPPED_GLOBAL_TIME_BUDGET":
        source_pin_path,source_file_sha=_verify_pin(source_record,source_path,f"{role} source input")
    else:
        source_file_sha=sha(source_path)
        if entry.get("sha256")!=source_file_sha:
            raise ValueError(f"{role} skipped source SHA differs from verified state")
    if entry.get("sha256")!=source_file_sha:
        raise ValueError(f"{role} verified-state source SHA mismatch")
    if role_result.get("state_input_sha256") not in (None,state_sha):
        raise ValueError(f"{role} 750 role result is bound to another state")
    parent_npz=load_npz(source_path)
    if source_surface_pins is not None:
        crosswalk_before=_source_crosswalk(role,source_surface_pins,parent_npz)
    else:crosswalk_before=None
    src=_source_boundary_check(role,entry,parent_npz)
    if status in NOOP_STATUSES:
        reported_before=role_result.get("native_bad_before",role_result.get("bad_before",src["bad"]))
        reported_after=role_result.get("native_bad_after",role_result.get("bad_after",src["bad"]))
        if int(reported_before)!=src["bad"] or int(reported_after)!=src["bad"]:
            raise ValueError(f"{role} no-op result changed native-bad count")
        if "output" in role_result or "accepted_operations" in role_result:
            raise ValueError(f"{role} no-op result unexpectedly carries proposal output")
        return {"role":role,"status":"PASS_INHERITED_SOURCE_NOOP","source_crosswalk":crosswalk_before,
                "native_bad_before":src["bad"],"native_bad_after":src["bad"],
                "source":V.pin(source_path),"nonpositive":0,"oriented_boundary":True,
                "proposal_replayed":False}
    if status!=PASS_STATUS:
        raise ValueError(f"{role} unsupported 750 result status: {status}")
    output_path,_=_verify_pin(role_result.get("output"),label=f"{role} proposal NPZ")
    _check_not_747(output_path,f"{role} proposal")
    output_npz=load_npz(output_path)
    target,target_key=V.boundary_target(parent_npz)
    crosswalk_after=(_source_crosswalk(role,source_surface_pins,output_npz)
                     if source_surface_pins is not None else None)
    if crosswalk_after!=crosswalk_before:raise ValueError(f"{role} source crosswalk changed in proposal")
    retained,ops=_reconstruct_operations(parent_npz,output_npz,role)
    _crosscheck_journal_operations(role,role_result,ops)
    old_bad=src["bad"]
    new_bad,_,_,_=V.quality_score(np.asarray(output_npz["points_world_m"],np.float32),
                                  np.asarray(output_npz["tetrahedra"],np.int64))
    if int(role_result.get("native_bad_before",-1))!=old_bad:
        raise ValueError(f"{role} journal native-bad before count differs")
    if int(role_result.get("native_bad_after",-1))!=new_bad:
        raise ValueError(f"{role} journal native-bad after count differs")
    if new_bad>=old_bad:
        raise ValueError(f"{role} proposal does not strictly lower native-bad count")
    for field,expected in (("input_points",len(src["points"])),("input_cells",len(src["tets"])),
                           ("output_points",len(output_npz["points_world_m"])),
                           ("output_cells",len(output_npz["tetrahedra"]))):
        if field in role_result and int(role_result[field])!=int(expected):
            raise ValueError(f"{role} journal count mismatch: {field}")
    readback_record=role_result.get("independent_readback")
    readback_path,_=_verify_pin(readback_record,label=f"{role} 750 independent readback")
    journal_replay=json.loads(readback_path.read_text())
    replay=V.independent_readback_replay(
        parent_npz,output_npz,ops,retained,target,old_bad,new_bad,source_path,source_file_sha)
    if journal_replay!=replay:
        raise ValueError(f"{role} pinned 750 readback receipt differs from independent external replay")
    if replay.get("status")!="PASS_INDEPENDENT_STEINER_CAVITY_READBACK":
        raise ValueError(f"{role} independent readback did not pass")
    if int(entry.get("nodes",len(src["points"])))!=len(src["points"]):
        raise ValueError(f"{role} source node count is inconsistent")
    if len(output_npz["points_world_m"])!=len(src["points"])+len(ops):
        raise ValueError(f"{role} promoted node count does not include every appended apex")
    return {
        "role":role,"status":"PASS_EXTERNAL_INDEPENDENT_REPLAY",
        "source":V.pin(source_path),"proposal":V.pin(output_path),
        "readback":V.pin(readback_path),
        "native_bad_before":old_bad,"native_bad_after":new_bad,
        "new_steiner_points":len(ops),"nodes":int(len(output_npz["points_world_m"])),
        "cells":int(len(output_npz["tetrahedra"])),"nonpositive":0,
        "target_boundary_key":target_key,"oriented_boundary_unchanged":True,
        "all_untouched_source_and_typed_bindings_equal":True,
        "all_operation_and_point_parent_csr_reconstructed":True,
        "source_crosswalk":crosswalk_after,
        "proposal_replayed":True,"physical_admission":False,"native_steps":0,
    }

def _journal_role_map(journal):
    rows=journal.get("role_results")
    if not isinstance(rows,list):raise ValueError("750 journal lacks role_results")
    out={}
    for row in rows:
        role=str(row.get("role",""))
        if role not in ("Skin","Fat") or role in out:
            raise ValueError("750 journal contains unknown or duplicate role")
        out[role]=row
    return out

def verify_and_promote(state_path,journal_path,output_path,receipt_path):
    state_path=Path(state_path);journal_path=Path(journal_path)
    state_pin=pin(state_path);journal_pin=pin(journal_path)
    _check_not_747(state_path,"input state")
    raw=json.loads(state_path.read_text());state,wrapped=_unwrap_state(raw)
    source_proof=_authenticated_source_proof(state_path,state)
    report=json.loads(journal_path.read_text())
    report_state=report.get("source_state",{})
    if Path(str(report_state.get("path",""))).resolve()!=state_path.resolve() or report_state.get("sha256")!=state_pin["sha256"]:
        raise ValueError("750 journal is not bound to the supplied source verified-state")
    producer=report.get("producer",{})
    _verify_pin(producer,PRODUCER750,"750 producer source")
    if producer.get("sha256")!=EXPECTED_750_SHA256:
        raise ValueError("750 journal producer hash is not the pinned source")
    if report.get("native_thresholds_unchanged") is not True:
        raise ValueError("750 journal does not preserve native thresholds")
    role_results=_journal_role_map(report)
    expected_roles={r for r in ("Skin","Fat") if r+"_current" in state}
    if set(role_results)!=expected_roles:
        raise ValueError("750 journal role coverage differs from the supplied verified-state")
    updated=dict(state)
    updated['source_state']=state_pin
    role_receipts=[]
    for role,role_result in role_results.items():
        key=role+"_current"
        if key not in state:raise ValueError(f"{role} result has no matching input current role")
        entry=state[key]
        receipt=verify_role(role,state,entry,role_result,state_pin["sha256"],source_proof["source_surfaces"])
        role_receipts.append(receipt)
        if receipt["status"]=="PASS_EXTERNAL_INDEPENDENT_REPLAY":
            updated[key]={**entry,
                "path":receipt["proposal"]["path"],
                "sha256":receipt["proposal"]["sha256"],
                "nodes":receipt["nodes"],"cells":receipt["cells"],
                "native_quality_failures":receipt["native_bad_after"],
                "nonpositive":0,"oriented_volume_target_boundary_equal":True,
                "target_boundary_match":True,
            }
    # Roles omitted or skipped remain bound to the input verified state.
    updated["physical_admission"]=False
    updated["native_steps"]=0
    if "independent_replay" in updated:
        updated["quality_steiner_760_parent_independent_replay"]=updated.pop("independent_replay")
    if "exact_context" in updated:
        updated["quality_steiner_768_parent_exact_context"]=updated.pop("exact_context")
    updated["quality_steiner_768_promotion"]={
        "schema":"h706.independent-external-steiner-promoter.v3",
        "status":"PASS_EXTERNAL_INDEPENDENT_REPLAY",
        "source_state":state_pin,"proposal_journal":journal_pin,
        "producer":producer,"role_receipts":role_receipts,
        "source_proof":source_proof,
        "source_boundary_status":"PASS_AUTHENTICATED_INHERITED_EXACT_CONTEXT_AND_FRESH_CROSSWALK",
        "native_admission":False,"physical_admission":False,"native_steps":0,
        "mass_mapping":"NOT_RUN",
    }
    final_state=dict(raw)
    if wrapped:
        final_state[STATE_WRAPPER]=updated
    else:
        final_state=updated
    output_path=Path(output_path);receipt_path=Path(receipt_path)
    if output_path.resolve()==receipt_path.resolve():
        raise ValueError("verified-state and promoter receipt paths must differ")
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite an existing promoter artifact")
    output_path.parent.mkdir(parents=True,exist_ok=True)
    receipt_path.parent.mkdir(parents=True,exist_ok=True)
    # No state is written before all source, proposal, and readback checks pass.
    tmp=output_path.with_name(output_path.name+".tmp")
    tmp.write_text(json.dumps(final_state,indent=2,sort_keys=True)+"\n")
    os.replace(tmp,output_path)
    receipt={
        "schema":"h706.independent-external-steiner-promoter.v3",
        "status":"PASS_EXTERNAL_INDEPENDENT_REPLAY",
        "source_state":state_pin,"proposal_journal":journal_pin,
        "source_proof":source_proof,
        "producer":producer,"verified_state":pin(output_path),
        "role_receipts":role_receipts,"physical_admission":False,
        "native_steps":0,"mass_mapping":"NOT_RUN",
        "elapsed_s":None,
    }
    receipt["elapsed_s"]=float(time.monotonic()-_START)
    receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt

_START=time.monotonic()

def _negative_control_fixture(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    root3=np.sqrt(3.0);root23=np.sqrt(2.0/3.0)
    outer_points=np.asarray([[0.,0.,0.],[1.,0.,0.],[.5,root3/2,0.],
                             [.5,root3/6,root23]],np.float32)
    outer=V.M.orient(outer_points,np.asarray([[0,1,2,3]],np.int64))
    faces=V.KERNEL.K734.oriented_boundary(outer)
    near=np.asarray([.5,root3/6,1.0e-5],np.float32)
    points=np.vstack((outer_points,near))
    old=V.M.orient(points,np.column_stack((np.full(len(faces),4,np.int64),faces)))
    source_path=output/"fixture-source.npz"
    np.savez_compressed(source_path,points_world_m=points,points_world_m_f32=points.copy(),
        tetrahedra=old,material_indices=np.full(len(old),17,np.int32),
        boundary_faces=faces,desired_boundary_faces=faces,nonpositive=np.asarray([0],np.int64),
        output_point_parent_offsets=np.asarray([0,1,1,1,1,1],np.int64),
        output_point_parent_input_cell_rows=np.asarray([2],np.int64),
        typed_I_candidate_path=np.asarray(["/synthetic/typed-I-surface.npz"]),
        typed_I_candidate_sha256=np.asarray(["a"*64]),
        source_context_anchor_rows=np.asarray([11,19,23],np.int64))
    old_bad=V.quality_score(points,old)[0]
    entry={"path":str(source_path),"sha256":sha(source_path),
           "native_quality_failures":old_bad,"nonpositive":0,
           "nodes":len(points),"cells":len(old)}
    state={"boundary_source_binding_status":"PENDING_METADATA_REBIND_NO_GEOMETRY_CHANGE",
           "physical_admission":False,"native_steps":0,"Skin_current":entry}
    state_path=output/"fixture-source-state.json"
    state_path.write_text(json.dumps(state,indent=2)+"\n")
    state_sha=sha(state_path);state["_state_sha256"]=state_sha
    args=argparse.Namespace(maximum_old_cells=24,maximum_star_cells=128,maximum_points=192,
        maximum_attempts=32,maximum_operations=1,construction_margin=1.0e-3)
    run_result=V.run_role("Skin",entry,state,output/"baseline-producer",
                          time.monotonic()+20.0,args)
    if run_result.get("status")!=PASS_STATUS:
        raise AssertionError("negative-control fixture proposal did not pass producer")
    journal={"source_state":V.pin(state_path),"producer":V.pin(PRODUCER750),
             "native_thresholds_unchanged":True,"role_results":[run_result]}
    journal_path=output/"fixture-journal.json"
    journal_path.write_text(json.dumps(journal,indent=2,sort_keys=True)+"\n")
    role_entry=entry
    baseline=verify_role("Skin",state,role_entry,run_result,state_sha)
    if baseline.get("status")!="PASS_EXTERNAL_INDEPENDENT_REPLAY":
        raise AssertionError("baseline external promoter replay failed")
    try:
        verify_and_promote(state_path,journal_path,
            output/"promoted-verified-state.json",output/"external-promoter-receipt.json")
    except ValueError as exc:
        if "exact D/I/O source surface pins" not in str(exc):raise
        promotion={"status":"PASS_REJECTED_SYNTHETIC_MISSING_SOURCE_PROOF","reason":str(exc)}
    else:raise AssertionError("synthetic fixture without source proof was incorrectly promotable")

    source_npz=load_npz(source_path)
    output_npz=load_npz(run_result["output"]["path"])
    controls=[]

    def reject_mutant(name,mutate):
        mutant={k:np.asarray(v).copy() for k,v in output_npz.items()}
        mutate(mutant)
        badpath=output/(name+".npz")
        np.savez_compressed(badpath,**mutant)
        trial=dict(run_result)
        trial["output"]=pin(badpath)
        try:
            verify_role("Skin",state,role_entry,trial,state_sha)
        except Exception as exc:
            controls.append({"name":name,"status":"REJECTED_AS_EXPECTED","reason":str(exc)})
            return
        raise AssertionError("negative control was incorrectly accepted: "+name)

    def mutate_old_prefix_csr(a):
        a["output_point_parent_offsets"][0]=1
    reject_mutant("mutated-old-prefix-csr",mutate_old_prefix_csr)

    def mutate_source_binding(a):
        x=np.asarray(a["typed_I_candidate_sha256"]).copy()
        x[0]=str(x[0])[:-1]+("b" if str(x[0])[-1]!="b" else "c")
        a["typed_I_candidate_sha256"]=x
    reject_mutant("mutated-unrelated-source-binding",mutate_source_binding)

    def mutate_one_point(a):
        row=int(np.asarray(a["quality_steiner_point_output_rows"])[0])
        xyz=np.asarray(a["points_world_m"],np.float32).copy()
        xyz[row,0]=np.nextafter(xyz[row,0],np.float32(np.inf))
        a["points_world_m"]=xyz
        if "points_world_m_f32" in a:a["points_world_m_f32"]=xyz.copy()
        # Leave the producer journal witness pinned to the original point;
        # the external check must reject this mutated proposal copy.
    reject_mutant("mutated-one-point",mutate_one_point)

    def mutate_ancestry_union(a):
        vals=np.asarray(a["replacement_parent_cell_rows"],np.int64).copy()
        vals[0]=int(vals[0])+1
        a["replacement_parent_cell_rows"]=vals
    reject_mutant("mutated-one-ancestry-union",mutate_ancestry_union)
    return {
        "schema":"h706.independent-external-steiner-promoter-controls.v1",
        "status":"PASS_EXTERNAL_PROMOTER_NEGATIVE_CONTROLS_AND_MISSING_PROOF_REJECTION",
        "baseline":baseline,"promotion":promotion,
        "negative_controls":controls,
        "all_four_mutations_rejected":len(controls)==4 and all(x["status"]=="REJECTED_AS_EXPECTED" for x in controls),
        "synthetic_missing_source_proof_rejected":promotion.get("status")=="PASS_REJECTED_SYNTHETIC_MISSING_SOURCE_PROOF",
        "source_zero_and_boundary_inherited":True,
        "actual_geometry_batch":False,"physical_admission":False,"native_steps":0,
    }

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state",type=Path)
    ap.add_argument("--journal",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--receipt",type=Path)
    ap.add_argument("--fixture",action="store_true")
    args=ap.parse_args()
    if args.fixture:
        if args.state or args.journal:
            ap.error("--fixture cannot be combined with --state or --journal")
        out=args.output or Path("/tmp/h706-steiner-768-controls-"+str(os.getpid()))
        print(json.dumps(_negative_control_fixture(out),indent=2,sort_keys=True))
        return
    if args.state is None or args.journal is None:
        ap.error("--state and --journal are required")
    output=args.output or args.journal.parent/"verified-state-768.json"
    receipt=args.receipt or args.journal.parent/"independent-promoter-768-report.json"
    result=verify_and_promote(args.state,args.journal,output,receipt)
    print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__":
    main()

