from pathlib import Path
import importlib.util
import json
import sys
import numpy as np

PRODUCER=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/coupled-anchor-collapse-780-root/ordinary-allrole-v25m-back-rowfix/repair_coupled_quality_points_722_v25m_back_rowfix.py')
spec=importlib.util.spec_from_file_location('v25m_fixture_producer',PRODUCER)
P=importlib.util.module_from_spec(spec);sys.modules[spec.name]=P;spec.loader.exec_module(P)
Q=P.Q
points=np.asarray([[0,0,0],[1e-3,0,0],[0,1e-3,0],[0,0,1e-9],[1e-4,0,-.9e-3]],np.float32)
tets=np.asarray([[0,1,2,3],[0,2,1,4]],np.int64)
materials=np.asarray([16,16],np.int32);groups=np.asarray([0,1],np.int8)
d,q=Q._M.measurements(points,tets);passed=np.asarray(Q._M.passes(d,q),bool)
assert passed.tolist()==[False,True]
patch={'patch':0,'bad_component':0,'seed_bad_cells':[0],'bad_cluster_cells':[0],
       'free_rows':[2,3],'affected_cells':[0,1]}
offsets,by_node=P._point_cell_incidence(tets,len(points))
orig_lp=Q.propose_quality_repair;orig_node=P._single_node_quality_candidate
lp_calls=[];node_calls=[]
def stalled_lp(p,t,mat,free,cap,**kwargs):
    lp_calls.append({'cells':len(t),'free':int(np.count_nonzero(free))})
    return np.asarray(p,np.float32).copy(),{'status':'NO_LEXICOGRAPHIC_PROGRESS','solver_status':9}
def stalled_node(*args,**kwargs):
    node_calls.append(int(args[2]))
    return None,{'status':'NODE_FALLBACK_STALLED'}
try:
    Q.propose_quality_repair=stalled_lp;P._single_node_quality_candidate=stalled_node
    candidate,diag=P.propose_remainder_block_local(points,tets,materials,groups,d,q,passed,[patch],
        trust_cap_m=5e-4,seconds=1.0,patch_seconds=.8,offsets=offsets,by_node=by_node,
        node_fallbacks_per_patch=4,ordinary_mode=True,joint_small_patch_fallback=True,
        joint_fallbacks_per_patch=1,joint_patch_seconds=.2,joint_maxiter=20)
    assert diag['status']=='PASS_F32_BLOCK_LOCAL_TARGET_QUALITY_PROPOSAL',diag
    assert diag['accepted_joint_small_patch_fallbacks']==1,diag
    assert diag['bad_before']==1 and diag['bad_after']==0,diag
    assert set(diag['quality_log_deficit_by_role_before'])=={'Skin','Fat'}
    assert set(diag['quality_log_deficit_by_role_after'])=={'Skin','Fat'}
    assert abs(sum(diag['quality_log_deficit_by_role_before'].values())-diag['bad_score_before'])<1.e-8
    assert abs(sum(diag['quality_log_deficit_by_role_after'].values())-diag['bad_score_after'])<1.e-8
    assert diag['skin_log_deficit_score_after']<diag['skin_log_deficit_score_before']
    assert len(lp_calls)==1 and len(node_calls)>=1,(lp_calls,node_calls)
    assert np.asarray(Q._M.passes(*Q._M.measurements(candidate,tets)),bool).tolist()==[True,True]
    assert all(int(x)>0 for x in Q._exact_positive(candidate,tets))
    assert np.array_equal(candidate[[0,1,4]].view(np.uint32),points[[0,1,4]].view(np.uint32))
    record=diag['local_patch_attempts'][0]
    assert record['block_lp_retry']['attempted'] and not record['block_lp_retry']['accepted']
    assert record['single_node_fallback']['accepted_count']==0
    joint=record['joint_multi_node_slsqp_fallback']
    assert joint['accepted_count']==1 and joint['reports'][0]['prior_failures']['node_fallback_accepted']==0
    passed_diag=diag
    # If disabled, the stalled LP/node path cannot invoke joint SLSQP.
    lp_calls.clear();node_calls.clear()
    no_candidate,no_diag=P.propose_remainder_block_local(points,tets,materials,groups,d,q,passed,[patch],
        trust_cap_m=5e-4,seconds=1.0,patch_seconds=.8,offsets=offsets,by_node=by_node,
        node_fallbacks_per_patch=4,ordinary_mode=True,joint_small_patch_fallback=False)
    assert no_diag['status']=='NO_LEXICOGRAPHIC_PROGRESS' and no_diag['accepted_joint_small_patch_fallbacks']==0,no_diag
    assert np.array_equal(no_candidate.view(np.uint32),points.view(np.uint32))
finally:
    Q.propose_quality_repair=orig_lp;P._single_node_quality_candidate=orig_node

def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
report={'schema':'h706.v25m-back-rowfix-producer-fixture.v1','status':'PASS_END_TO_END_TINY_PRODUCER_FIXTURE',
 'producer':{'path':str(PRODUCER),'sha256':sha(PRODUCER)},'joint_helper':{'path':str(P._JOINT_PATH),'sha256':sha(P._JOINT_PATH)},
 'lp_calls':lp_calls,'node_calls':node_calls,'joint_diag_status':joint['reports'][0]['status'],
 'joint_bad_before':passed_diag['bad_before'],'joint_bad_after':passed_diag['bad_after'],
 'joint_fallbacks':passed_diag['accepted_joint_small_patch_fallbacks'],
 'analytic_jacobian_control':'PASS in pinned joint helper fixture',
 'per_role_log_deficit_score_fields':'PASS; Skin and Fat scores sum to global deficit before/after',
 'good_neighbor_preserved':bool(passed_diag['bad_after']==0),'exact_local_f32_positive':True,
 'fixed_points_bitwise_unchanged':[0,1,4],'fallback_disabled_control':'PASS',
 'physical_admission':False,'native_steps':0,'full_805_and_global_boundary_flux_volume_audit':'NOT_RUN_REQUIRED'}
path=PRODUCER.parent/'test_v25m_back_rowfix_producer_fixture-final-report.json'
path.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':report['status'],'producer_sha':report['producer']['sha256'],'helper_sha':report['joint_helper']['sha256'],'bad_before':report['joint_bad_before'],'bad_after':report['joint_bad_after'],'accepted_joint':report['joint_fallbacks'],'full_audits':'NOT RUN'},sort_keys=True))
