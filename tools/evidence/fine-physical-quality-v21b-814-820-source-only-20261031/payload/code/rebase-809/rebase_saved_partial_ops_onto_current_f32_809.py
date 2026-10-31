#!/usr/bin/env python3
"""Rebase retained partial-quality Steiner operations onto a new F32 point fit.

This is a source-only preparation tool. It never runs the cone solver: saved
apexes are replayed against the supplied current point coordinates, invalid
operations are dropped, and all accepted output maps are rebuilt from that
current parent.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PRODUCER_PATH=BASE/"quality-partial-cavity-684-root"/"batch_interior_steiner_partial_quality_795_root.py"
PRODUCER_SHA="4e35df53981c2fedc0f03b268138ebf74378a09e0c55d1f7a64b8e09c813f3bf"
PROMOTER_PATH=BASE/"quality-partial-cavity-684-root"/"independent_promoter_796_795_root.py"
PROMOTER_SHA="320fcbb6f1eb8147c67a27474c4b23100e92bca060122b3dffc37aa851bbcfec"
SOURCE_806_REPORT_SHA="2d6b25746661c9b98b29b5ebc930f1c1262892bca320829d1ec4a9af8133e44c"
EXPECTED_INPUT_STATE_SHA="6ff110ecb9e938d5f86aa4e864d1a7ebf20ab5b0dfb3b4773dcbf0ac6ddb5d28"
REBASE_SCHEMA="h706.partial-quality-steiner-current-point-rebase.v1"
PASS_ROLE="PASS_INTERIOR_STEINER_PARTIAL_QUALITY_CONE_BATCH"
NOOP_ROLE="NO_VALID_BOUNDED_PARTIAL_FIXED_BOUNDARY_QUALITY_PROGRESS"
STATE_WRAPPER="current_verified_state_20261031"

def imp(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
def sha(path):
    return hashlib.file_digest(Path(path).open("rb"),"sha256").hexdigest()
def pin(path):
    p=Path(path)
    return {"path":str(p.resolve()),"sha256":sha(p)}
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}
def _json(path):
    return json.loads(Path(path).read_text())
def _check_pin(row,label):
    if not isinstance(row,dict) or not row.get("path") or not row.get("sha256"):
        raise ValueError(label+" missing path/SHA pin")
    p=Path(row["path"])
    if not p.is_file() or sha(p)!=row["sha256"]:
        raise ValueError(label+" missing or changed: "+str(p))
    return p
def _unwrap(raw):
    if STATE_WRAPPER in raw:
        return raw[STATE_WRAPPER],True
    return raw,False

def _authenticated_ancestor_chain(state_path,state,target_pin):
    target_path=Path(target_pin["path"]).resolve()
    if sha(target_path)!=target_pin["sha256"]:
        raise ValueError("required 799 ancestor pin is missing or changed")
    chain=[pin(state_path)]
    current=state
    seen={Path(state_path).resolve()}
    for depth in range(8):
        link=current.get("source_state")
        if not isinstance(link,dict):
            raise ValueError("current state ancestry ended before the pinned 799 parent")
        parent_path=_check_pin(link,"verified-state ancestor")
        parent_path=parent_path.resolve()
        if parent_path in seen:
            raise ValueError("verified-state ancestry contains a cycle")
        seen.add(parent_path)
        normalized={"path":str(parent_path),"sha256":sha(parent_path)}
        chain.append(normalized)
        if parent_path==target_path and normalized["sha256"]==target_pin["sha256"]:
            return chain
        current,_=_unwrap(_json(parent_path))
    raise ValueError("current state ancestry exceeded the bounded chain depth")

if not PRODUCER_PATH.is_file() or sha(PRODUCER_PATH)!=PRODUCER_SHA:
    raise RuntimeError("pinned 795 producer mismatch")
if not PROMOTER_PATH.is_file() or sha(PROMOTER_PATH)!=PROMOTER_SHA:
    raise RuntimeError("pinned 796 promoter mismatch")
V=imp("rebase_v795_owner",PRODUCER_PATH)
P=imp("rebase_v796_promoter_owner",PROMOTER_PATH)

def _assert_same_row_domain(saved,current,role):
    sp=np.asarray(saved["points_world_m"],np.float32)
    cp=np.asarray(current["points_world_m"],np.float32)
    st=np.asarray(saved["tetrahedra"],np.int64)
    ct=np.asarray(current["tetrahedra"],np.int64)
    sm=np.asarray(saved["material_indices"])
    cm=np.asarray(current["material_indices"])
    if sp.ndim!=2 or sp.shape[1]!=3 or cp.shape!=sp.shape or not np.isfinite(cp).all():
        raise ValueError(f"{role} old/current point row domains differ")
    if st.shape!=ct.shape or not np.array_equal(st,ct):
        raise ValueError(f"{role} old/current tetrahedron row connectivity differs")
    if sm.shape!=cm.shape or not np.array_equal(sm,cm):
        raise ValueError(f"{role} old/current material rows differ")
    if not np.array_equal(V.T.m.boundary_identity(V.T.m.boundary(st)),
                          V.T.m.boundary_identity(V.T.m.boundary(ct))):
        raise ValueError(f"{role} old/current derived boundary differs")
    target0,key0=V.boundary_target(saved)
    target1,key1=V.boundary_target(current)
    if key0!=key1 or not np.array_equal(V.T.m.boundary_identity(target0),
                                        V.T.m.boundary_identity(target1)):
        raise ValueError(f"{role} desired boundary row domain differs")
    if "points_world_m_f32" in current and not np.array_equal(
            np.asarray(current["points_world_m_f32"],np.float32).view(np.uint32),
            cp.view(np.uint32)):
        raise ValueError(f"{role} current F32 point alias mismatch")
    changed=int(np.count_nonzero(np.any(sp.view(np.uint32)!=cp.view(np.uint32),axis=1)))
    return {"old_point_count":len(sp),"current_point_count":len(cp),
            "cell_count":len(st),"material_rows_equal":True,"tetrahedron_rows_equal":True,
            "target_boundary_equal":True,"changed_f32_point_rows":changed}

def _validated_saved_ops(role,saved_parent,saved_output,role_result):
    if role_result.get("status")!=PASS_ROLE:
        raise ValueError(f"{role} saved result is not an accepted partial-quality batch")
    ops_list=role_result.get("accepted_operations")
    if not isinstance(ops_list,list) or not ops_list:
        raise ValueError(f"{role} saved result has no operations")
    retained,rebuilt=P._reconstruct_operations(saved_parent,saved_output,role)
    P._crosscheck_journal_operations(role,role_result,rebuilt)
    if len(rebuilt)!=len(ops_list):
        raise ValueError(f"{role} saved operation count mismatch")
    readback_path,_=P._verify_pin(role_result.get("independent_readback"),label=f"{role} saved 806 independent readback")
    old_target,_=V.boundary_target(saved_parent)
    old_bad,old_score,_,_=V.quality_score(np.asarray(saved_parent["points_world_m"],np.float32),
                                          np.asarray(saved_parent["tetrahedra"],np.int64))
    saved_out_bad,saved_out_score,_,_=V.quality_score(np.asarray(saved_output["points_world_m"],np.float32),
                                                       np.asarray(saved_output["tetrahedra"],np.int64))
    replay=V.independent_readback_replay(saved_parent,saved_output,rebuilt,retained,
        old_target,old_bad,saved_out_bad,role_result["input"]["path"],role_result["input"]["sha256"])
    if _json(readback_path)!=replay or replay.get("status")!="PASS_INDEPENDENT_STEINER_CAVITY_READBACK":
        raise ValueError(f"{role} saved 806 independent readback did not authenticate")
    for i,(saved,op) in enumerate(zip(ops_list,rebuilt)):
        rows=np.asarray(op["old_cell_rows"],np.int64)
        t=np.asarray(saved_parent["tetrahedra"],np.int64)
        if np.any(rows<0) or np.any(rows>=len(t)) or len(np.unique(rows))!=len(rows):
            raise ValueError(f"{role} operation {i} old row range invalid")
        vertices=np.unique(t[rows])
        if saved.get("old_vertex_rows")!=vertices.tolist():
            raise ValueError(f"{role} operation {i} old vertex crosswalk mismatch")
        if sorted(set(op["old_cell_rows"]))!=op["old_cell_rows"]:
            raise ValueError(f"{role} operation {i} old rows are not canonical")
        apex=int(op["apex_point_output_row"])
        if apex!=len(saved_parent["points_world_m"])+i:
            raise ValueError(f"{role} operation {i} saved apex row is not canonical")
        if not np.array_equal(np.asarray(saved_output["points_world_m"],np.float32)[apex].view(np.uint32),
                              np.asarray(op["apex_point_f32"],np.float32).view(np.uint32)):
            raise ValueError(f"{role} operation {i} saved apex is not authenticated by 806 output")
    return retained,rebuilt

def _rebase_ops(role,current,ops):
    p=np.asarray(current["points_world_m"],np.float32)
    t=np.asarray(current["tetrahedra"],np.int64)
    mat=np.asarray(current["material_indices"])
    old_bad,old_score,old_det,old_q=V.quality_score(p,t)
    if np.any(old_det<=0):
        raise ValueError(f"{role} current input has nonpositive measured cells")
    base_index=V._point_key_index(p)
    accepted=[];rejected=[];reserved_cells=set();accepted_keys=set();appended=[]
    exact_cache={}
    for oi,source_op in enumerate(ops):
        reason=None
        rows=np.asarray(source_op["old_cell_rows"],np.int64)
        if rows.ndim!=1 or not len(rows) or np.any(rows<0) or np.any(rows>=len(t)):
            reason="invalid_old_cell_rows"
        elif set(map(int,rows))&reserved_cells:
            reason="overlapping_saved_cavities"
        elif not np.array_equal(rows,np.asarray(sorted(set(map(int,rows))),np.int64)):
            reason="noncanonical_old_cell_rows"
        if reason is not None:
            rejected.append({"operation":oi,"reason":reason})
            continue
        old=np.asarray(t[rows],np.int64)
        verts=np.unique(old.reshape(-1))
        if not V.exact_positive(p,old,exact_cache):
            rejected.append({"operation":oi,"reason":"current_old_cavity_not_exact_positive"})
            continue
        if len(np.unique(mat[rows]))!=1 or int(np.unique(mat[rows])[0])!=int(source_op["material_index"]):
            rejected.append({"operation":oi,"reason":"current_cavity_material_mismatch"})
            continue
        apex=np.asarray(source_op["apex_point_f32"],np.float32).reshape(3)
        if not np.isfinite(apex).all():
            rejected.append({"operation":oi,"reason":"nonfinite_saved_apex"})
            continue
        key=V._point_key(apex)
        if V._point_key_present(base_index,key) or key in accepted_keys:
            rejected.append({"operation":oi,"reason":"saved_apex_duplicates_current_or_accepted_point"})
            continue
        saved_global=np.asarray(source_op["new_tetrahedra"],np.int64)
        saved_apex=int(source_op["apex_point_output_row"])
        next_point_row=len(p)+len(accepted)
        new=saved_global.copy()
        if not np.all(new==saved_global):
            raise AssertionError("copy mismatch")
        if not np.any(new==saved_apex):
            rejected.append({"operation":oi,"reason":"saved_replacement_does_not_reference_its_apex"})
            continue
        new[new==saved_apex]=next_point_row
        if np.any(new<0) or np.any(new>=next_point_row+1):
            rejected.append({"operation":oi,"reason":"saved_replacement_references_unknown_point"})
            continue
        if np.any((new>=len(p)) & (new!=next_point_row)):
            rejected.append({"operation":oi,"reason":"saved_replacement_references_prior_806_steiner_row"})
            continue
        used_old=np.unique(new[new<len(p)])
        if not set(map(int,used_old)).issubset(set(map(int,verts))):
            rejected.append({"operation":oi,"reason":"saved_replacement_uses_unknown_old_vertex"})
            continue
        trial_points=np.vstack((p,np.asarray(appended+[apex],np.float32)))
        if not V.exact_positive(trial_points,new,exact_cache):
            rejected.append({"operation":oi,"reason":"current_rebased_cells_not_exact_positive"})
            continue
        if V.M.chain(old)!=V.M.chain(new):
            rejected.append({"operation":oi,"reason":"current_rebased_oriented_chain_mismatch"})
            continue
        ok,why=V.M.certify(trial_points,old,new)
        if not ok:
            rejected.append({"operation":oi,"reason":"current_rebased_exact_certificate_"+str(why)})
            continue
        before_bad,before_score,_,_=V.quality_score(p,old)
        after_bad,after_score,_,_=V.quality_score(trial_points,new)
        if before_bad<=0 or not ((after_bad<before_bad) or
                                  (after_bad==before_bad and after_score<before_score)):
            rejected.append({"operation":oi,"reason":"current_coordinates_no_local_lexicographic_progress",
                "current_bad_before":before_bad,"current_bad_after":after_bad,
                "current_log_deficit_before":before_score,"current_log_deficit_after":after_score})
            continue
        rec=dict(source_op)
        rec.update({
            "apex_point_f32":apex.tolist(),"apex_point_output_row":int(next_point_row),
            "new_tetrahedra":new.tolist(),"material_index":int(mat[rows[0]]),
            "old_cell_rows":rows.astype(int).tolist(),"old_vertex_rows":verts.astype(int).tolist(),
            "full_cavity_parent_rows":rows.astype(int).tolist(),
            "old_cell_count":int(len(rows)),"new_cell_count":int(len(new)),
            "bad_before":int(before_bad),"bad_after":int(after_bad),
            "log_deficit_before":float(before_score),"log_deficit_after":float(after_score),
            "exact_cavity_certificate":str(why),"exact_positive_new_cells":True,
            "all_new_cells_native_pass":bool(np.all(V.M.passes(*V.M.measurements(trial_points,new)))),
            "exact_volume_conserved":True,"oriented_chain_matches_input":True,
            "exact_cavity_certificate_on_current_parent":True,
            "rebased_from_saved_operation_index":int(oi),
        })
        rec["replacement_tet_range"]=[0,0]
        appended.append(apex.copy());accepted.append(rec);reserved_cells.update(map(int,rows));accepted_keys.add(key)
    cursor=0
    for op in accepted:
        n=len(op["new_tetrahedra"]);op["replacement_tet_range"]=[cursor,cursor+n];cursor+=n
    return accepted,rejected,old_bad,old_score,p, t, mat

def rebase_role(role,current_entry,current_state_sha,current_path,saved_parent_path,
                saved_output_path,saved_role_result,outdir):
    source_path=Path(current_path);source_sha=sha(source_path)
    if current_entry.get("path") and Path(current_entry["path"]).resolve()!=source_path.resolve():
        raise ValueError(f"{role} current path differs from supplied verified state")
    if current_entry.get("sha256")!=source_sha:
        raise ValueError(f"{role} current state SHA differs from current NPZ")
    current=load_npz(source_path)
    saved_parent_path=Path(saved_parent_path);saved_parent_sha=sha(saved_parent_path)
    if saved_role_result.get("input",{}).get("path")!=str(saved_parent_path):
        if Path(saved_role_result.get("input",{}).get("path","/invalid")).resolve()!=saved_parent_path.resolve():
            raise ValueError(f"{role} saved journal input path mismatch")
    if saved_role_result.get("input",{}).get("sha256")!=saved_parent_sha:
        raise ValueError(f"{role} saved journal input SHA mismatch")
    saved_parent=load_npz(saved_parent_path)
    topology=_assert_same_row_domain(saved_parent,current,role)
    saved_output_path=Path(saved_output_path);saved_output_sha=sha(saved_output_path)
    if saved_role_result.get("output",{}).get("path")!=str(saved_output_path):
        if Path(saved_role_result.get("output",{}).get("path","/invalid")).resolve()!=saved_output_path.resolve():
            raise ValueError(f"{role} saved journal output path mismatch")
    if saved_role_result.get("output",{}).get("sha256")!=saved_output_sha:
        raise ValueError(f"{role} saved output SHA mismatch")
    saved_output=load_npz(saved_output_path)
    _,saved_ops=_validated_saved_ops(role,saved_parent,saved_output,saved_role_result)
    ops,rejected,old_bad,old_score,p,t,mat=_rebase_ops(role,current,saved_ops)
    declared_bad=current_entry.get("native_quality_failures")
    if declared_bad is None or int(declared_bad)!=int(old_bad):
        raise ValueError(f"{role} current state native-bad count differs from F32 replay")
    declared_nonpositive=current_entry.get("nonpositive",current.get("nonpositive",None))
    if declared_nonpositive is None or int(np.asarray(declared_nonpositive).reshape(-1)[0])!=0:
        raise ValueError(f"{role} current state does not inherit nonpositive=0")
    if not ops:
        return {
          "role":role,"status":NOOP_ROLE,"input":pin(source_path),
          "state_input_sha256":current_state_sha,"input_cells":int(len(t)),
          "input_points":int(len(p)),"native_bad_before":int(old_bad),
          "native_bad_after":int(old_bad),"log_deficit_before":float(old_score),
          "log_deficit_after":float(old_score),"rebase_rejections":rejected,
          "rebase_source_operations":len(rejected),"topology_compatibility":topology,
          "rebase_source_journal":pin(Path(saved_role_result["_journal_path"])),
          "rebase_source_output":pin(saved_output_path),"rebase_source_input":pin(saved_parent_path),
          "native_thresholds_unchanged":True,"native_admission":False,
          "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
    new_points=np.asarray([o["apex_point_f32"] for o in ops],np.float32).reshape((-1,3))
    out_points=np.vstack((p,new_points))
    removed=np.asarray(sorted({int(x) for op in ops for x in op["old_cell_rows"]}),np.int64)
    if not len(removed):raise ValueError(f"{role} rebased operation set removes no cells")
    keep=np.ones(len(t),bool);keep[removed]=False;retained=np.flatnonzero(keep)
    blocks=[np.asarray(op["new_tetrahedra"],np.int64) for op in ops]
    replacement=np.vstack(blocks)
    replacement_mat=np.concatenate([np.full(len(b),int(op["material_index"]),mat.dtype) for op,b in zip(ops,blocks)])
    out_t=np.vstack((t[keep],replacement))
    out_mat=np.concatenate((mat[keep],replacement_mat))
    det,q=V.M.measurements(out_points,out_t)
    if np.any(det<=0):raise ValueError(f"{role} rebased output has nonpositive measured cells")
    if not V.exact_positive(out_points,replacement):
        raise ValueError(f"{role} rebased output exact positivity failed")
    boundary=V.T.m.boundary(out_t)
    target,target_key=V.boundary_target(current)
    if not np.array_equal(V.T.m.boundary_identity(boundary),V.T.m.boundary_identity(target)):
        raise ValueError(f"{role} rebased output changed the target boundary")
    new_bad,new_score,_,_=V.quality_score(out_points,out_t)
    if not ((new_bad<old_bad) or (new_bad==old_bad and new_score<old_score)):
        raise ValueError(f"{role} rebased accepted operations do not improve global quality")
    origin=np.r_[retained,np.full(len(replacement),-1,np.int64)]
    inverse=np.full(len(t),-1,np.int64);inverse[retained]=np.arange(len(retained),dtype=np.int64)
    repl_parent_offsets=[0];repl_parent_rows=[]
    repl_tet_offsets=[0];op_parent_offsets=[0];op_parent_rows=[]
    point_offsets,point_rows=V._point_parent_csr(current,ops,len(p))
    for op,block in zip(ops,blocks):
        rows=sorted(set(map(int,op["old_cell_rows"])))
        for _ in block:
            repl_parent_rows.extend(rows);repl_parent_offsets.append(len(repl_parent_rows))
        op_parent_rows.extend(rows);op_parent_offsets.append(len(op_parent_rows))
        repl_tet_offsets.append(repl_tet_offsets[-1]+len(block))
    payload=dict(current)
    payload["points_world_m"]=out_points.astype(np.asarray(current["points_world_m"]).dtype,copy=False)
    if "points_world_m_f32" in current:payload["points_world_m_f32"]=out_points.copy()
    updates={
      "tetrahedra":out_t,"material_indices":out_mat,"boundary_faces":boundary,
      "desired_boundary_faces":np.asarray(target,np.int64),
      "output_cell_to_input_cell":origin,"input_cell_to_output_cell":inverse,
      "retained_input_cell_rows":retained,"replacement_tetrahedra":replacement,
      "replacement_parent_offsets":np.asarray(repl_parent_offsets,np.int64),
      "replacement_parent_cell_rows":np.asarray(repl_parent_rows,np.int64),
      "replacement_tet_offsets":np.asarray(repl_tet_offsets,np.int64),
      "quality_fixed_boundary_operation_parent_offsets":np.asarray(op_parent_offsets,np.int64),
      "quality_fixed_boundary_operation_parent_cell_rows":np.asarray(op_parent_rows,np.int64),
      "quality_fixed_boundary_operation_count":np.asarray([len(ops)],np.int64),
      "quality_fixed_boundary_operation_methods":np.asarray([op["proposal_method"] for op in ops]),
      "quality_fixed_boundary_operation_seed_rows":np.asarray([op["seed_cell_row"] for op in ops],np.int64),
      "quality_fixed_boundary_input_candidate_path":np.asarray([str(source_path)]),
      "quality_fixed_boundary_input_candidate_sha256":np.asarray([source_sha]),
      "quality_milp_operation_parent_offsets":np.asarray([0],np.int64),
      "quality_milp_operation_parent_cell_rows":np.empty(0,np.int64),
      "quality_milp_partial_bad_replacement":np.asarray([False]),
      "quality_milp_operation_count":np.asarray([0],np.int64),
      "quality_milp_operation_seed_rows":np.empty(0,np.int64),
      "quality_milp_input_candidate_path":np.asarray([str(source_path)]),
      "quality_milp_input_candidate_sha256":np.asarray([source_sha]),
      "quality_milp_input_node_to_output_node":np.arange(len(p),dtype=np.int64),
      "quality_milp_output_node_to_input_node":np.r_[np.arange(len(p),dtype=np.int64),np.full(len(new_points),-1,np.int64)],
      "quality_steiner_point_output_rows":np.asarray([o["apex_point_output_row"] for o in ops],np.int64),
      "quality_steiner_operation_count":np.asarray([len(ops)],np.int64),
      "quality_steiner_cavity_selection_methods":np.asarray([o["candidate_selection_method"] for o in ops]),
      "quality_steiner_kernel_helper_path":np.asarray([str(V.KERNEL_PATH)]),
      "quality_steiner_kernel_helper_sha256":np.asarray([V.EXPECTED_KERNEL_SHA256]),
      "current_geometry_parent_path":np.asarray([str(source_path)]),
      "current_geometry_parent_sha256":np.asarray([source_sha]),
      "canonical_parent_union_operation_order":np.asarray("interior-steiner-kernel-cone-full-original-cavity-union-v1"),
    }
    updates.update(V._point_map_updates(current,len(p),len(new_points),point_offsets,point_rows))
    if "source_interior_vertex_quotient" in current:
        updates["source_interior_vertex_quotient"]=np.r_[np.asarray(current["source_interior_vertex_quotient"],np.int64),np.full(len(new_points),-1,np.int64)]
    if "metadata_finalizer_point_map_status" in current:
        updates["metadata_finalizer_point_map_status"]=np.asarray("APPENDED_INTERIOR_STEINER_POINTS_NO_SOURCE_BINDING")
    if "metadata_finalizer_point_parent_status" in current:
        updates["metadata_finalizer_point_parent_status"]=np.asarray("COMPLETE_FULL_CAVITY_PARENT_CELL_CSR")
    updates.update(V._historical_point_csr_updates(current,len(p),source_path,source_sha))
    payload.update(updates)
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=False)
    output_path=outdir/f"{role.lower()}-rebased-partial-quality.npz"
    np.savez_compressed(output_path,**payload)
    output_sha=sha(output_path);readback=load_npz(output_path)
    if sha(output_path)!=output_sha:raise ValueError(f"{role} output changed during readback")
    replay=V.independent_readback_replay(current,readback,ops,retained,target,old_bad,new_bad,source_path,source_sha)
    readback_path=outdir/f"{role.lower()}-independent-rebase-readback.json"
    readback_path.write_text(json.dumps(replay,indent=2,sort_keys=True)+"\n")
    return {
      "role":role,"status":PASS_ROLE,"input":pin(source_path),"output":pin(output_path),
      "independent_readback":pin(readback_path),"state_input_sha256":current_state_sha,
      "input_cells":int(len(t)),"output_cells":int(len(out_t)),"input_points":int(len(p)),
      "output_points":int(len(out_points)),"new_steiner_points":int(len(new_points)),
      "native_bad_before":int(old_bad),"native_bad_after":int(new_bad),
      "log_deficit_before":float(old_score),"log_deficit_after":float(new_score),
      "parent_declared_nonpositive":int(current_entry_nonpositive(payload,current)),
      "exact_nonpositive_new_cells":0,"all_output_measured_positive":True,
      "all_replacement_cells_exact_positive":True,"all_new_cells_native_pass":bool(np.all(V.M.passes(*V.M.measurements(out_points,replacement)))),
      "strict_native_bad_count_drop":bool(new_bad<old_bad),"target_boundary_key":target_key,
      "oriented_target_boundary_unchanged":True,"all_new_cells_same_material_as_cavity":True,
      "complete_replacement_parent_union_csr":True,"complete_steiner_point_parent_union_csr":True,
      "immediate_old_point_parent_csr_empty":True,"historical_parent_point_csr_archived":True,
      "exact_volume_conserved_per_cavity":True,"accepted_operations":ops,
      "rebase_rejections":rejected,"rebase_source_operations":len(rejected)+len(ops),
      "topology_compatibility":topology,"rebase_source_journal":pin(Path(saved_role_result["_journal_path"])),
      "rebase_source_output":pin(saved_output_path),"rebase_source_input":pin(saved_parent_path),
      "native_thresholds_unchanged":True,"native_admission":False,"physical_admission":False,
      "native_steps":0,"mass_mapping":"NOT_RUN"}

def current_entry_nonpositive(payload,current):
    v=current.get("nonpositive",np.asarray([0]))
    return int(np.asarray(v).reshape(-1)[0]) if np.asarray(v).size else 0

def run(state_path,journal_path,output_path):
    started=time.monotonic()
    state_path=Path(state_path).resolve();journal_path=Path(journal_path).resolve()
    if sha(journal_path)!=SOURCE_806_REPORT_SHA:
        raise ValueError("saved 806 journal SHA is not the pinned accepted result")
    raw_state=_json(state_path);state,wrapped=_unwrap(raw_state)
    if state.get("schema")!="h706.combined-current-exact-surface-diff-config.v1":
        raise ValueError("current verified-state schema mismatch")
    if state.get("source_state",{}).get("sha256") is None:
        raise ValueError("current verified-state has no parent-state pin")
    _check_pin(state.get("source_state"),"current source state")
    _check_pin(state.get("exact_context"),"807 exact-context report")
    journal=_json(journal_path)
    if journal.get("schema")!="h706.interior-steiner-adaptive-partial-quality-cone-batch.v1" or journal.get("status")!="PASS_WITH_PARTIAL_NATIVE_DEFECTS":
        raise ValueError("806 journal status/schema is not the accepted partial-quality batch")
    saved_source=_check_pin(journal.get("source_state"),"806 source state")
    if sha(saved_source)!=EXPECTED_INPUT_STATE_SHA:
        raise ValueError("806 journal is not bound to the pinned 799 state")
    source_ancestry_chain=_authenticated_ancestor_chain(state_path,state,journal["source_state"])
    if journal.get("native_thresholds_unchanged") is not True or journal.get("physical_admission") is not False or int(journal.get("native_steps",-1))!=0:
        raise ValueError("806 journal does not preserve source-only status")
    if Path(journal["producer"]["path"]).resolve()!=PRODUCER_PATH.resolve() or journal["producer"]["sha256"]!=PRODUCER_SHA:
        raise ValueError("806 source producer pin mismatch")
    current_state_pin=pin(state_path)
    roles={x["role"]:x for x in journal.get("role_results",[])}
    results=[];role_rebases=[]
    output_path=Path(output_path).resolve()
    if output_path.exists():raise FileExistsError("refusing to overwrite rebase output")
    output_path.mkdir(parents=True)
    updated=dict(state);updated["source_state"]=current_state_pin
    ordered=sorted(roles,key=lambda r:r)
    for role in ordered:
        if role not in ("Fat","Skin"):raise ValueError("806 journal contains unknown role")
        key=role+"_current"
        if key not in state:raise ValueError("current state missing role "+role)
        current_entry=state[key]
        current_path=_check_pin(current_entry,role+" current NPZ")
        old_state_raw=_json(saved_source)
        old_state,old_wrapped=_unwrap(old_state_raw)
        if key not in old_state:raise ValueError("799 state missing role "+role)
        saved_input_path=_check_pin(old_state[key],role+" 799 input NPZ")
        rr=roles[role]
        if rr.get("input",{}).get("path")!=str(saved_input_path.resolve()) or rr.get("input",{}).get("sha256")!=sha(saved_input_path):
            raise ValueError(role+" 806 input does not match pinned 799 state")
        saved_output_path=_check_pin(rr.get("output"),role+" 806 output NPZ")
        rr_for_rebase=dict(rr);rr_for_rebase["_journal_path"]=str(journal_path)
        role_out=output_path/role.lower()
        role_result=rebase_role(role,current_entry,current_state_pin["sha256"],current_path,
            saved_input_path,saved_output_path,rr_for_rebase,role_out)
        results.append(role_result)
        if role_result["status"]==PASS_ROLE:
            updated[key]={**current_entry,**role_result["output"],"nodes":role_result["output_points"],
                "cells":role_result["output_cells"],"native_quality_failures":role_result["native_bad_after"],
                "nonpositive":0,"oriented_volume_target_boundary_equal":True,
                "target_boundary_match":True}
        role_rebases.append({"role":role,"status":
            "PASS_REBASED_ON_CURRENT_F32" if role_result["status"]==PASS_ROLE else "PASS_NO_OP_REBASED_CURRENT_F32",
            "old_806_operation_count":len(rr.get("accepted_operations",[])),
            "replayed_operation_count":len(role_result.get("accepted_operations",[])),
            "rejected_operation_count":len(role_result.get("rebase_rejections",[])),
            "topology_compatibility":role_result["topology_compatibility"],
            "source_806_result":pin(journal_path),"source_806_output":role_result["rebase_source_output"],
            "current_parent":role_result["input"],
            **({"rebased_output":role_result["output"],"independent_readback":role_result["independent_readback"]}
               if role_result["status"]==PASS_ROLE else {})})
    updated["physical_admission"]=False;updated["native_steps"]=0;updated["mass_mapping"]="NOT_RUN"
    updated.pop("independent_replay",None);updated.pop("exact_context",None)
    if "independent_point_replay" in updated:
        updated["quality_steiner_current_parent_independent_point_replay"]=updated.pop("independent_point_replay")
    if "exact_context" in state:
        updated["quality_steiner_current_parent_exact_context"]=state["exact_context"]
    if wrapped:
        state_out_raw=dict(raw_state);state_out_raw[STATE_WRAPPER]=updated
    else:state_out_raw=updated
    state_out=output_path/"state.json"
    state_out.write_text(json.dumps(state_out_raw,indent=2,sort_keys=True)+"\n")
    report={
      "schema":"h706.interior-steiner-adaptive-partial-quality-cone-batch.v1",
      "status":"PASS_WITH_PARTIAL_NATIVE_DEFECTS" if any(r.get("status")==PASS_ROLE for r in results) else "NO_VALID_BOUNDED_PROGRESS","source_state":current_state_pin,
      "output_state":pin(state_out),"producer":pin(PRODUCER_PATH),
      "exact_quality_loader":pin(V.SOLVER_PATH),"topology_owner":pin(V.TOP_PATH),
      "kernel_cone_helper":pin(V.KERNEL_PATH),"role_results":results,
      "role_schedule":[{"role":r["role"],"budget_seconds":0.0,"budget_policy":"SAVED_OPERATION_REBASE_NO_SOLVER"}
                       for r in results],
      "roles_sorted_by_measured_source_bad_descending":False,
      "balanced_per_role_deadlines":False,"cavity_selection_methods":["replayed_saved_current_point_f32_operations"],
      "maximum_vertex_star_cells":0,"native_thresholds_unchanged":True,"all_cells_positive":True,
      "native_admission":False,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
      "rebase_manifest":{"schema":REBASE_SCHEMA,"status":"PASS_REBASE_VERIFIED_TO_CURRENT_F32",
        "saved_journal":pin(journal_path),"saved_journal_source_state":journal["source_state"],
        "current_state":current_state_pin,"current_state_sha256":sha(state_path),
        "source_ancestry_chain":source_ancestry_chain,
        "old_input_state_sha256":EXPECTED_INPUT_STATE_SHA,"rebase_source":"authenticated_806_saved_operations",
        "topology_policy":"EXACT_TETRAHEDRA_MATERIAL_AND_TARGET_BOUNDARY_ROW_EQUALITY",
        "current_coordinates_used":True,"operation_reoptimization":"NOT_RUN",
        "failed_operations_pruned":True,"role_rebases":role_rebases,
        "rebase_builder":pin(Path(__file__)),"independent_promoter":pin(PROMOTER_PATH),
        "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"},
      "elapsed_s":time.monotonic()-started,"time_budget_s":0.0}
    report_path=output_path/"report.json"
    report_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state",type=Path,required=True)
    ap.add_argument("--saved-journal",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    report=run(args.state,args.saved_journal,args.output)
    print(json.dumps({"status":report["status"],"report":pin(args.output/"report.json"),
        "output_state":report["output_state"],"roles":[{"role":r["role"],
        "status":r["status"],"native_bad_before":r["native_bad_before"],
        "native_bad_after":r["native_bad_after"],"accepted":len(r["accepted_operations"]),
        "rejected":len(r["rebase_rejections"])} for r in report["role_results"]],
        "physical_admission":False,"native_steps":0},sort_keys=True))
if __name__=="__main__":main()

