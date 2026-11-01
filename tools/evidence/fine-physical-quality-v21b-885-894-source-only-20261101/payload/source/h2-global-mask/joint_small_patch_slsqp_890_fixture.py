import importlib.util
import pathlib
import sys
import json
import numpy as np
from scipy import optimize

MODULE=pathlib.Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/coupled-anchor-collapse-780-root/ordinary-allrole-v25h-unified-mixed/joint_small_patch_slsqp_890.py')
spec=importlib.util.spec_from_file_location('joint890',MODULE)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
Q=mod.Q

# One q-bad sliver and one good neighbor share movable rows; the optimization
# must improve the sliver without regressing the neighbor.
p=np.asarray([[0,0,0],[1e-3,0,0],[0,1e-3,0],[0,0,1e-9],[1e-4,0,-.9e-3]],np.float32)
t=np.asarray([[0,1,2,3],[0,2,1,4]],np.int64)
d,q=Q._M.measurements(p,t);passed=np.asarray(Q._M.passes(d,q),bool)
assert passed.tolist()==[False,True],(d,q,passed)
caps=np.zeros(len(p),np.float64);caps[[2,3,4]]=5e-4
prior={'lp_attempted':True,'node_fallback_attempted':True,'node_fallback_accepted':0,'lp_status':'LP_NO_CANDIDATE'}
orig_minimize=optimize.minimize
jac_check={}
def checked_minimize(fun,x0,method,jac,bounds,constraints,options):
    z=np.zeros_like(np.asarray(x0,np.float64));n=len(z)//2
    z[:n]=np.linspace(.02,.07,n);z[n:]=.6
    h=2e-6
    fnum=np.empty(len(z))
    for j in range(len(z)):
        xp=z.copy();xm=z.copy();xp[j]+=h;xm[j]-=h
        fnum[j]=(fun(xp)-fun(xm))/(2*h)
    fan=np.asarray(jac(z),np.float64)
    cfun=constraints[0]['fun'];cjac=constraints[0]['jac']
    c0=np.asarray(cfun(z),np.float64);jan=np.asarray(cjac(z),np.float64);jnum=np.empty_like(jan)
    for j in range(len(z)):
        xp=z.copy();xm=z.copy();xp[j]+=h;xm[j]-=h
        jnum[:,j]=(np.asarray(cfun(xp))-np.asarray(cfun(xm)))/(2*h)
    ferr=float(np.max(np.abs(fnum-fan)/(1+np.maximum(np.abs(fnum),np.abs(fan)))))
    cerr=float(np.max(np.abs(jnum-jan)/(1+np.maximum(np.abs(jnum),np.abs(jan)))))
    assert ferr<2e-5,(ferr,fnum,fan)
    assert cerr<2e-5,(cerr,jnum,jan)
    jac_check.update(objective_scaled_max_error=ferr,constraint_scaled_max_error=cerr)
    return orig_minimize(fun,x0,method=method,jac=jac,bounds=bounds,constraints=constraints,options=options)
optimize.minimize=checked_minimize
incidence_offsets,incidence_by_node=mod._OWNER._point_cell_incidence(t,len(p))
candidate,diag=mod.try_joint_small_patch_slsqp(p,t,[2,3],[0],passed,passed,caps,prior,
    incidence_offsets=incidence_offsets,incidence_by_node=incidence_by_node,maxiter=30,time_limit_s=1.5)
optimize.minimize=orig_minimize
assert candidate is not None,diag
assert diag['status']=='PASS_JOINT_PATCH_PROPOSAL',diag
assert diag['bad_after']==0 and diag['bad_before']==1,diag
assert np.asarray(Q._M.passes(*Q._M.measurements(candidate,t)),bool).tolist()==[True,True]
assert all(int(v)>0 for v in Q._exact_positive(candidate,t))
assert np.array_equal(candidate[[0,1,4]].view(np.uint32),p[[0,1,4]].view(np.uint32))

# Full witness lock on both free vertices must fail closed.
rows=[]
for node in [2,3]:
    for ax in range(3):
        v=np.zeros((1,3));v[0,ax]=1
        rows.append({'point_rows':[node],'vectors':v,'origin':p[node].astype(np.float64),'upper_bound':0.})
        v2=-v
        rows.append({'point_rows':[node],'vectors':v2,'origin':p[node].astype(np.float64),'upper_bound':0.})
blocked,bdiag=mod.try_joint_small_patch_slsqp(p,t,[2,3],[0],passed,passed,caps,prior,separation_rows=rows,
    incidence_offsets=incidence_offsets,incidence_by_node=incidence_by_node,maxiter=20,time_limit_s=1.0)
assert blocked is None,(bdiag,blocked)
assert bdiag['status']!='PASS_JOINT_PATCH_PROPOSAL',bdiag

# Eligibility is strict: a row absent from the seed bad cell cannot be freed.
skip,sdiag=mod.try_joint_small_patch_slsqp(p,t,[2,4],[0],passed,passed,caps,prior,
    incidence_offsets=incidence_offsets,incidence_by_node=incidence_by_node,maxiter=10,time_limit_s=1.0)
assert skip is None and sdiag['status']=='REJECT_FREE_ROW_NOT_A_SEED_BAD_CELL_CORNER',sdiag

# A previous accepted node fallback makes this method ineligible.
skip2,sdiag2=mod.try_joint_small_patch_slsqp(p,t,[2,3],[0],passed,passed,caps,{**prior,'node_fallback_accepted':1},
    incidence_offsets=incidence_offsets,incidence_by_node=incidence_by_node,maxiter=10,time_limit_s=1.0)
assert skip2 is None and sdiag2['status']=='SKIP_NOT_AFTER_STALLED_LP_AND_NODE_FALLBACK',sdiag2
result={'schema':'h706.joint-small-patch-slsqp.890-fixture.v1','status':'PASS_TINY_FIXTURE','helper_path':str(MODULE),'helper_sha256':mod._sha(MODULE),'fixture_source':str(pathlib.Path(__file__).resolve()),'jacobian_checks':jac_check,'success_diag':diag,'witness_control':bdiag,'scope_control':sdiag,'history_control':sdiag2,'physical_admission':False,'native_steps':0}
out=MODULE.parent/'joint_small_patch_slsqp_890_fixture-report.json'
out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('schema','status','helper_path','helper_sha256','jacobian_checks','physical_admission','native_steps')},sort_keys=True))
