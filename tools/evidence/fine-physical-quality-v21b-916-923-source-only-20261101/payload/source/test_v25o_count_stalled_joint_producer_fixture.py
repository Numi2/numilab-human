from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np
PTH=Path(__file__).resolve().parent
PRODUCER=PTH/"repair_coupled_quality_points_722_v25o_count_stall_joint.py"
VERIFIER=PTH/"verify_ordinary_allrole_v25o_count_stall_joint.py"
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
P=load("v25o_fixture_producer",PRODUCER);V=load("v25o_fixture_verifier",VERIFIER);Q=P.Q
points=np.asarray([[0,0,0],[1e-3,0,0],[0,1e-3,0],[0,0,1e-9],[1e-4,0,-.9e-3]],np.float32)
tets=np.asarray([[0,1,2,3],[0,2,1,4]],np.int64)
materials=np.asarray([16,16],np.int32);groups=np.asarray([0,1],np.int8)
d,q=Q._M.measurements(points,tets);passed=np.asarray(Q._M.passes(d,q),bool)
assert passed.tolist()==[False,True],(d,q,passed)
patch={"patch":0,"bad_component":0,"seed_bad_cells":[0],"bad_cluster_cells":[0],"free_rows":[2,3],"affected_cells":[0,1]}
offsets,by_node=P._point_cell_incidence(tets,len(points))
orig_lp=Q.propose_quality_repair;orig_node=P._single_node_quality_candidate
lp_calls=[];node_calls=[]
def stalled_lp(p,t,mat,free,cap,**kwargs):
    lp_calls.append({"cells":len(t),"free":int(np.count_nonzero(free))})
    return np.asarray(p,np.float32).copy(),{"status":"NO_LEXICOGRAPHIC_PROGRESS","solver_status":9}
def score_only_node(p,t,node,node_cells,cap,current_good,base_good,separation_rows,deadline):
    node_calls.append(int(node))
    if int(node)!=3:return None,{"status":"FIXTURE_OTHER_NODE_SKIPPED"}
    x=np.asarray(p,np.float32).copy();x[3,2]=np.float32(1.5e-9)
    nd,nq=Q._M.measurements(x,t[node_cells]);ng=np.asarray(Q._M.passes(nd,nq),bool)
    assert not bool(ng[np.flatnonzero(np.asarray(node_cells)==0)[0]])
    assert np.all(current_good[np.asarray(node_cells)]<=ng)
    assert _score(Q,nq,nd)<_score(Q,Q._M.measurements(p,t[node_cells])[1],Q._M.measurements(p,t[node_cells])[0])
    return x,{"fixture":"SCORE_ONLY_NODE_CANDIDATE"}
def _score(Q,q,d):
    return float(P._global_log_deficit(q,d))
try:
    Q.propose_quality_repair=stalled_lp;P._single_node_quality_candidate=score_only_node
    candidate,diag=P.propose_remainder_block_local(points,tets,materials,groups,d,q,passed,[patch],
        trust_cap_m=5e-4,seconds=2.0,patch_seconds=1.5,offsets=offsets,by_node=by_node,
        node_fallbacks_per_patch=4,ordinary_mode=True,joint_small_patch_fallback=True,
        joint_fallbacks_per_patch=2,joint_patch_seconds=.95,joint_maxiter=30)
    assert diag["status"]=="PASS_F32_BLOCK_LOCAL_TARGET_QUALITY_PROPOSAL",diag
    assert diag["accepted_joint_small_patch_fallbacks"]==1,diag
    assert diag["bad_before"]==1 and diag["bad_after"]==0,diag
    assert len(lp_calls)==1 and 3 in node_calls,(lp_calls,node_calls)
    got=np.asarray(Q._M.passes(*Q._M.measurements(candidate,tets)),bool)
    assert got.tolist()==[True,True],got
    assert all(int(x)>0 for x in Q._exact_positive(candidate,tets))
    assert np.array_equal(candidate[[0,1,4]].view(np.uint32),points[[0,1,4]].view(np.uint32))
    record=diag["local_patch_attempts"][0]
    assert record["block_lp_retry"]["attempted"] and not record["block_lp_retry"]["accepted"]
    assert record["single_node_fallback"]["accepted_count"]==1
    assert record["single_node_fallback"]["nodes"][-1]["status"]=="ACCEPT_TIED_SCORE_REDUCTION"
    joint=record["joint_multi_node_slsqp_fallback"]
    assert joint["accepted_count"]==1 and joint["reports"][0]["prior_failures"]["node_fallback_accepted"]==1
    independent=V.verify_joint_fallback_diagnostics({"joint_small_patch_fallback":diag["joint_small_patch_fallback"],"maximum_displacement_m":5e-4},diag)
    assert independent["count_stall_policy_verified"] is True and independent["accepted"]==1,independent
    # A prior with actual count progress must be refused by the pinned helper.
    bad_prior=dict(joint["reports"][0]["prior_failures"]);bad_prior["node_bad_count_after"]=0
    bad_candidate,bad_diag=P.JOINT.try_joint_small_patch_slsqp(points,tets,np.asarray([2,3]),[0],passed,passed,
        np.full(len(points),5e-4),bad_prior,separation_rows=None,allowed_free_rows=np.asarray([2,3]),
        incidence_offsets=offsets,incidence_by_node=by_node,maxiter=5,time_limit_s=.05)
    assert bad_candidate is None and bad_diag["status"]=="SKIP_PRIOR_NATIVE_BAD_COUNT_CHANGED",bad_diag
finally:
    Q.propose_quality_repair=orig_lp;P._single_node_quality_candidate=orig_node
report={"schema":"h706.v25o-count-stalled-joint-producer-fixture.v1","status":"PASS_END_TO_END_COUNT_STALLED_SCORE_ONLY_NODE_FIXTURE",
 "fixture":{"path":str(Path(__file__).resolve()),"sha256":sha(__file__)},
 "producer":{"path":str(PRODUCER),"sha256":sha(PRODUCER)},
 "helper":{"path":str(P._JOINT_PATH),"sha256":sha(P._JOINT_PATH)},
 "independent_verifier":{"path":str(VERIFIER),"sha256":sha(VERIFIER)},
 "lp":{"attempts":len(lp_calls),"accepted":False,"status":"NO_LEXICOGRAPHIC_PROGRESS"},
 "node":{"attempted":len(node_calls),"accepted_score_only":1,"node":3},
 "joint":{"attempts":joint["attempt_count"],"accepted":joint["accepted_count"],"status":joint["reports"][0]["status"]},
 "native_bad_before":1,"native_bad_after":0,"good_neighbor_preserved":True,
 "exact_local_f32_signs_positive":True,"fixed_points_bitwise_unchanged":[0,1,4],
 "independent_count_stall_policy":"PASS","count_progress_prior_rejected":"PASS",
 "full_scene_775_807_805_boundary_flux_volume_audits":"NOT_RUN_REQUIRED","physical_admission":False,"native_steps":0}
out=PTH/"test_v25o_count_stalled_joint_producer_fixture-report.json"
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":report["status"],"before":1,"after":0,"joint":joint["accepted_count"],"verifier":"PASS","physical_admission":False},sort_keys=True))
