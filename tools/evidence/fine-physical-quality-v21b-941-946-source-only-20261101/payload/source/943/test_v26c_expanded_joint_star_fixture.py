from pathlib import Path
import hashlib, importlib.util, json, sys, tempfile
import numpy as np
PTH=Path(__file__).resolve().parent
PRODUCER=PTH/"repair_coupled_quality_points_722_v26c_expanded_joint_stars.py"
VERIFIER=PTH/"verify_ordinary_allrole_v26c_expanded_joint_stars.py"
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
P=load("v26c_fixture_producer",PRODUCER);V=load("v26c_fixture_verifier",VERIFIER);Q=P.Q
s=2*np.sqrt(2);qg=(1+2e-4)*1e-5;qb=(1-2e-4)*1e-5;hg=qg*s;hb=qb*s
points=np.asarray([[.25,.25,hg],[0,0,0],[1,0,0],[0,1,0],
                   [0,0,hg+hb],[1,0,hg+hb],[0,1,hg+hb],
                   [1.25,.25,0],[.25,1.25,0],[.25,.25,1]],np.float32)
tets=np.asarray([[0,1,3,2],[0,4,5,6],[0,7,8,9]],np.int64)
materials=np.asarray([16,16,16],np.int32);groups=np.asarray([0,1,1],np.int8)
d,q=Q._M.measurements(points,tets);good=np.asarray(Q._M.passes(d,q),bool)
assert good.tolist()==[True,False,True],(d.tolist(),q.tolist(),good.tolist())
move=3e-4*1e-5*s
orig_lp=Q.propose_quality_repair;orig_node=P._single_node_quality_candidate
orig_joint=P.JOINT.try_joint_small_patch_slsqp
offsets,by_node=P._point_cell_incidence(tets,len(points))
def stalled_lp(p,t,mat,free,cap,**kwargs):
    return np.asarray(p,np.float32).copy(),{"status":"NO_LEXICOGRAPHIC_PROGRESS","solver_status":9}
def raw_node(p,t,node,node_cells,cap,current_good,base_good,separation_rows,deadline,*,return_float64_candidate=False):
    assert return_float64_candidate is True
    c=np.asarray(p,np.float64).copy();c[int(node),2]-=move
    return c,{"status":"RAW_FLOAT64_NODE_CANDIDATE_FOR_F32_BACKTRACK","optimizer_success":True,"fixture":"node-full-step-regression-half-step-progress"}
def no_node(p,t,node,node_cells,cap,current_good,base_good,separation_rows,deadline,*,return_float64_candidate=False):
    return None,{"status":"FIXTURE_NO_NODE_CANDIDATE"}
def raw_joint(p,t,free_rows,seed_bad_cell_rows,current_passed,baseline_passed,remaining_l1_cap_by_point,prior_failures,**kwargs):
    if len(free_rows)<=4:
        return None,{"status":"FIXTURE_SEED_CORNERS_NO_GAIN","seed_bad_cells":[1],"free_rows":list(map(int,free_rows))}
    c=np.asarray(p,np.float64).copy();c[0,2]-=move
    return c,{"status":"RAW_FLOAT64_CANDIDATE_FOR_F32_BACKTRACK","affected_cells":[0,1,2],
        "seed_bad_cells":[1],"free_rows":list(map(int,free_rows)),"optimizer_success":True,
        "maximum_float64_step_l1_m":float(move),
        "fixture":"expanded-star-full-step-regression-half-step-progress"}
meta={"f32_candidate_backtracking_helper":{"path":str(P._F32_BACKTRACK_PATH.resolve()),"sha256":P._F32_BACKTRACK_SHA},
 "f32_candidate_backtracking_policy":"raw float64 single-node, 2-4 seed-corner and 8-16 authorized seed-vertex-star SLSQP candidates are checked at alpha 1,1/2,...,1/64 on complete incident stars; existing F32 native, exact sign, separation, cap, and role count/score gates remain unchanged"}
results=[]
try:
    Q.propose_quality_repair=stalled_lp
    P._single_node_quality_candidate=raw_node
    patch={"patch":0,"bad_component":0,"seed_bad_cells":[1],"bad_cluster_cells":[1],"affected_cells":[0,1,2],"free_rows":[0]}
    cand,node_diag=P.propose_remainder_block_local(points,tets,materials,groups,d,q,good,[patch],
        trust_cap_m=2e-6,seconds=2.,patch_seconds=1.5,offsets=offsets,by_node=by_node,
        node_fallbacks_per_patch=1,ordinary_mode=False,extended_250um_construction_trial=False,
        joint_small_patch_fallback=False)
    assert node_diag["status"]=="PASS_F32_BLOCK_LOCAL_TARGET_QUALITY_PROPOSAL",node_diag
    assert node_diag["accepted_single_node_fallbacks"]==1,node_diag
    assert node_diag["accepted_f32_backtracked_proposals"]==1,node_diag
    nr=node_diag["local_patch_attempts"][0]["single_node_fallback"]["nodes"][0]
    nb=nr["f32_backtracking"]
    assert nb["alpha"]==.5 and nb["attempts"][0]["status"]=="REJECT_F32_GOOD_CELL_REGRESSION",nb
    assert int(node_diag["bad_after"])==1 and node_diag["bad_score_after"]<node_diag["bad_score_before"],node_diag
    assert np.array_equal(cand[1:].view(np.uint32),points[1:].view(np.uint32))
    assert np.array_equal(Q._M.passes(*Q._M.measurements(cand,tets)),good)
    replay_node=V.verify_f32_backtracking_diagnostics(meta,node_diag)
    assert replay_node["accepted_events"]==1 and replay_node["independent_alpha_full_star_replay"]
    results.append({"kind":"one-node","alpha":nb["alpha"],"attempts":len(nb["attempts"]),"bad_after":node_diag["bad_after"],"independent":"PASS"})

    P._single_node_quality_candidate=no_node
    P.JOINT.try_joint_small_patch_slsqp=raw_joint
    patch_joint={"patch":0,"bad_component":0,"seed_bad_cells":[1],"bad_cluster_cells":[1],"affected_cells":[0,1,2],"free_rows":list(range(10))}
    candidates=P._joint_seed_candidates_v26c(np.asarray([1]),np.asarray(patch_joint["free_rows"]),points,tets,good,d,q,
        np.zeros(len(points),np.float64),2e-6,offsets,by_node,max_free_nodes=16)
    assert len(candidates)==2 and candidates[0][3]=="seed-corners" and candidates[1][3]=="expanded-seed-vertex-star",candidates
    assert len(candidates[1][2])==10 and set(candidates[1][2]).issubset(set(range(10))),candidates[1]
    cand2,joint_diag=P.propose_remainder_block_local(points,tets,materials,groups,d,q,good,[patch_joint],
        trust_cap_m=2e-6,seconds=2.,patch_seconds=1.5,offsets=offsets,by_node=by_node,
        node_fallbacks_per_patch=4,ordinary_mode=False,extended_250um_construction_trial=False,
        joint_small_patch_fallback=True,joint_fallbacks_per_patch=2,joint_patch_seconds=.95,joint_maxiter=10)
    assert joint_diag["status"]=="PASS_F32_BLOCK_LOCAL_TARGET_QUALITY_PROPOSAL",joint_diag
    assert joint_diag["accepted_joint_small_patch_fallbacks"]==1,joint_diag
    assert joint_diag["accepted_f32_backtracked_proposals"]==1,joint_diag
    jr=next(x for x in joint_diag["local_patch_attempts"][0]["joint_multi_node_slsqp_fallback"]["reports"] if str(x.get("status","")).startswith("ACCEPT_"))
    jb=jr["f32_backtracking"]
    assert jb["alpha"]==.5 and jb["attempts"][0]["status"]=="REJECT_F32_GOOD_CELL_REGRESSION",jb
    assert int(joint_diag["bad_after"])==1 and joint_diag["bad_score_after"]<joint_diag["bad_score_before"],joint_diag
    assert np.array_equal(cand2[[1,2,3,7,8,9]].view(np.uint32),points[[1,2,3,7,8,9]].view(np.uint32))
    assert np.array_equal(Q._M.passes(*Q._M.measurements(cand2,tets)),good)
    joint_meta={**meta,"maximum_displacement_m":2e-6,"joint_small_patch_fallback":joint_diag["joint_small_patch_fallback"]}
    replay_joint,verified_joint_ids=V.verify_f32_backtracking_diagnostics(joint_meta,joint_diag,return_verified_record_ids=True)
    assert replay_joint["accepted_events"]==1 and replay_joint["independent_alpha_full_star_replay"]
    assert not ({"full_point_star_verified","all_local_exact_f32_determinants_positive","all_attached_separation_witnesses_rechecked"} & set(jr))
    joint_policy=V.verify_joint_fallback_diagnostics(joint_meta,joint_diag,verified_joint_ids,tets,[patch_joint])
    assert joint_policy["count_stall_policy_verified"] is True and joint_policy["accepted"]==1,joint_policy
    results.append({"kind":"joint-seed-4-plus-expanded-10-node","alpha":jb["alpha"],"attempts":len(jb["attempts"]),"bad_after":joint_diag["bad_after"],"independent":"PASS"})
finally:
    Q.propose_quality_repair=orig_lp;P._single_node_quality_candidate=orig_node
    P.JOINT.try_joint_small_patch_slsqp=orig_joint
with tempfile.TemporaryDirectory(prefix="v26c-cow-control-") as td:
    parent=Path(td)/"parent.npz";child=Path(td)/"child.npz"
    keep=np.asarray([[1.25,-0.5,0.0],[4.0,7.0,9.0]],np.float32)
    changed=np.asarray([[9,8,7],[6,5,4]],np.int16)
    np.savez_compressed(parent,keep=keep,replaced=changed,delete_me=np.asarray([8],np.uint8))
    parent_sha=sha(parent)
    full_child={"keep":keep,"replaced":changed+1,"new_key":np.asarray([3,1,4],np.int8)}
    delta,deleted,source_summary=P.cow_payload_delta(parent,full_child)
    assert set(delta)=={"replaced","new_key"} and deleted==["delete_me"]
    assert source_summary["unchanged_member_count"]==1
    no_op_full={"keep":keep,"replaced":changed,"delete_me":np.asarray([8],np.uint8)}
    no_op_delta,no_op_deleted,no_op_summary=P.cow_payload_delta(parent,no_op_full)
    assert no_op_delta=={} and no_op_deleted==[] and no_op_summary["unchanged_member_count"]==3
    no_op_path=Path(td)/"no-op-clone.npz"
    no_op_receipt=P.COW_COMPLETE.write_npz_clone_complete(parent,no_op_path,no_op_full)
    assert no_op_receipt["replaced_keys"]==[] and no_op_receipt["new_keys"]==[] and no_op_receipt["deleted_keys"]==[]
    assert no_op_receipt["unchanged_compressed_members_bitwise_verified"]==3
    cow_receipt=P.COW_COMPLETE.write_npz_clone_complete(parent,child,full_child)
    assert sha(parent)==parent_sha and cow_receipt["parent_unchanged"] and cow_receipt["apfs_clonefile_used"]
    assert cow_receipt["full_copy_fallback"] is False and cow_receipt["zip_crc"]=="PASS"
    with np.load(child,allow_pickle=False) as z:
        assert np.array_equal(z["keep"],keep) and np.array_equal(z["replaced"],changed+1)
        assert np.array_equal(z["new_key"],np.asarray([3,1,4],np.int8))
    basepin={"path":str(parent.resolve()),"sha256":parent_sha}
    childpin={"path":str(child.absolute()),"sha256":sha(child)}
    receipt_path=Path(td)/"source-D-receipt.json"
    receipt_path.write_text(json.dumps(cow_receipt,indent=2,sort_keys=True)+"\n")
    receipts={"source_D":{"path":str(receipt_path.resolve()),"sha256":sha(receipt_path)}}
    delta_summaries={"source_D":source_summary}
    currentpins={}
    for role in ("Skin","Fat"):
        vdest=Path(td)/(role.lower()+"-volume.npz")
        vfull={"keep":keep,"replaced":changed+2,"new_key":np.asarray([2,7,1],np.int8)}
        vdelta,vdeleted,vsummary=P.cow_payload_delta(parent,vfull)
        vr=P.COW_COMPLETE.write_npz_clone_complete(parent,vdest,vfull)
        delta_summaries["volume_"+role]=vsummary
        vrpath=Path(td)/(role.lower()+"-volume-receipt.json")
        vrpath.write_text(json.dumps(vr,indent=2,sort_keys=True)+"\n")
        currentpins[role]={"path":str(vdest.absolute()),"sha256":sha(vdest)}
        receipts["volume_"+role]={"path":str(vrpath.resolve()),"sha256":sha(vrpath)}
    fake_parent={"source_surfaces":{"O":basepin,"I":basepin,"D":basepin},
        "Skin_current":basepin,"Fat_current":basepin}
    fake_candidate={"source_surfaces":{"O":basepin,"I":basepin,"D":childpin},
        "Skin_current":currentpins["Skin"],"Fat_current":currentpins["Fat"],
        "npz_cow_append_writer":{"path":str(P._COW_COMPLETE_PATH.resolve()),"sha256":sha(P._COW_COMPLETE_PATH)},
        "npz_cow_underlying_npz_writer":{"path":str(P._COW_NPZ_PATH.resolve()),"sha256":sha(P._COW_NPZ_PATH)},
        "npz_cow_append_policy":"Complete desired NPZ payloads are diffed by the pinned 926b adapter using pinned 925 dtype/shape/order-A byte equality; only changed/new arrays and deletions are emitted through APFS clone-only updates; no full-copy fallback",
        "npz_cow_writer_receipts":receipts,"npz_cow_payload_deltas":delta_summaries}
    cow_validation=V.verify_cow_npz_receipts(fake_candidate,fake_parent)
    assert cow_validation["verified_archive_count"]==3 and cow_validation["apfs_clone_only"]
report={"schema":"h706.v26c-expanded-joint-star-producer-fixture.v1","status":"PASS_EXPANDED_STAR_INDEPENDENT_ALPHA_REPLAY_AND_COW",
 "producer":{"path":str(PRODUCER.resolve()),"sha256":sha(PRODUCER)},
 "joint_helper":{"path":str(P._JOINT_PATH.resolve()),"sha256":sha(P._JOINT_PATH)},
 "backtracking_helper":{"path":str(P._F32_BACKTRACK_PATH.resolve()),"sha256":sha(P._F32_BACKTRACK_PATH)},
 "verifier":{"path":str(VERIFIER.resolve()),"sha256":sha(VERIFIER)},
 "cases":results,"joint_acceptance_derived_from_independent_nested_f32_replay":"PASS",
 "full_global_star_connectivity_control":"PASS","full_star_f32_exact_and_native_gates":"PASS",
 "rejected_alpha_full_good_regression_then_half_alpha_progress":"PASS",
 "native_thresholds_caps_and_source_policy_unchanged":True,
 "cow_emitter":{"path":str(P._COW_COMPLETE_PATH.resolve()),"sha256":sha(P._COW_COMPLETE_PATH),
     "underlying_writer_path":str(P._COW_NPZ_PATH.resolve()),"underlying_writer_sha256":sha(P._COW_NPZ_PATH),
     "parent_unchanged":True,"readback_exact":True,"apfs_clonefile_used":bool(cow_receipt["apfs_clonefile_used"]),
     "full_copy_fallback":cow_receipt["full_copy_fallback"],"zip_crc":cow_receipt["zip_crc"],
     "independent_receipt_verifier":"PASS","verified_archive_count":cow_validation["verified_archive_count"],
     "archive_diffs":[{k:v for k,v in row.items() if k!="receipt"} for row in cow_validation["receipts"]],
     "no_op_delta_control":{"status":"PASS","empty_delta":True,"unchanged_members":no_op_receipt["unchanged_compressed_members_bitwise_verified"]}},
 "expanded_seed_vertex_star_policy":"PASS",
 "scene_trial":"NOT_RUN","physical_admission":False,"native_steps":0}
out=PTH/"test_v26c_expanded_joint_star_fixture-report.json"
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":report["status"],"cases":results,"physical_admission":False},sort_keys=True))
