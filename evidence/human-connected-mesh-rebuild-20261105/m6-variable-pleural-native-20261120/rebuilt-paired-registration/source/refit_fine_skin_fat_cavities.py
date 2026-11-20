"""Test determinant-constrained fine cavity fits with every exterior row fixed."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, eye, hstack, vstack

from check_current_parent_volume import volumes
from rebind_current_skin import sha256

data=Path('fine-transfer-screen-map.npz')
with np.load(data,allow_pickle=False) as x:
    original=x['old_world_m']
    starting=original+x['fine_node_delta_m']
    roles=x['roles']
compiled=Path('../captured-fat-sliver-495/proposal/compiler/volume-export/volume/mesh.ele')
rows=np.loadtxt(compiled,skiprows=1,dtype=np.int32)
tet,material=rows[:,1:5]-1,rows[:,5]
assert len(tet)==1448830 and set(material)=={15,16}

def f32(x):
    return x.astype(np.float32).astype(float)

def all_quality(x):
    result=np.empty(len(tet))
    x=f32(x)
    for lo in range(0,len(tet),120_000):
        r=slice(lo,lo+120_000)
        result[r]=volumes(x,tet[r])[1]
    return result

base_quality=all_quality(original)
q=all_quality(starting)
invalid=np.flatnonzero(q<=0)
assert len(invalid)==195 and np.min(base_quality)>0
print('baseline minimum Float32 quality',float(base_quality.min()),flush=True)
fixed=roles=='O'
obstructed=invalid[np.all(fixed[tet[invalid]],axis=1)]
print('fixed-exterior tetrahedra among invalid rows',len(obstructed),flush=True)
history=[]
current=starting.copy()

def propose(reference, free, incident, goal, step):
    tx=tet[incident]
    p=reference[tx]
    u,v,w=np.moveaxis(p[:,1:]-p[:,0,None,:],1,0)
    jac1,jac2,jac3=np.cross(v,w),np.cross(w,u),np.cross(u,v)
    det=np.einsum('ij,ij->i',u,jac1)
    scale=np.maximum.reduce([np.linalg.norm(p[:,i]-p[:,j],axis=1)
                             for i in range(4) for j in range(i)])**3
    jac=np.stack((-jac1-jac2-jac3,jac1,jac2,jac3),axis=1)
    node=np.full(len(current),-1,dtype=int)
    node[free]=np.arange(len(free))
    ids=node[tx]
    a=np.flatnonzero(ids.ravel()>=0)
    coords=3*ids.ravel()[a,None]+np.arange(3)
    rr=np.repeat(a//4,3)
    jj=np.concatenate((coords,),axis=0).ravel()
    vv=(jac/scale[:,None,None]).reshape(-1,3)[a].ravel()*1e-3
    nv=3*len(free)
    linear=coo_matrix((vv,(rr,jj)),shape=(len(p),nv)).tocsr()
    zero=linear.copy()
    zero.data[:]=0.
    ident=eye(nv,format='csr')
    abs_value=vstack((hstack((ident,-ident)),hstack((-ident,-ident))),format='csr')
    limiter=vstack((hstack((-linear,zero)),abs_value),format='csr')
    rhs=np.concatenate((det/scale-goal,np.zeros(2*nv)))
    result=linprog(np.concatenate((np.zeros(nv),np.ones(nv))),
                   A_ub=limiter,b_ub=rhs,
                   bounds=[(-step,step)]*nv+[(0,step)]*nv,
                   method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    return result

for iteration in range(5):
    q=all_quality(current)
    broken=np.flatnonzero(q<=0)
    if not len(broken):
        break
    eligible=broken[~np.all(fixed[tet[broken]],axis=1)]
    if not len(eligible):
        break
    region=np.unique(tet[eligible])
    if iteration>1:
        neighborhood=np.flatnonzero(np.any(np.isin(tet,region),axis=1))
        region=np.unique(tet[neighborhood])
    free=region[~fixed[region]]
    incident=np.flatnonzero(np.any(np.isin(tet,free),axis=1))
    goal=np.minimum(base_quality[incident]*.50,5e-6)
    # Close the division by the longest edge with a fresh exact determinant
    # evaluation each trial. All original exterior node positions stay fixed.
    event=dict(iteration=iteration,free_nodes=len(free),incident_cells=len(incident),steps=[])
    best=None
    for step in (.06,.18,.4,.75):
        result=propose(current,free,incident,goal,step)
        trial=dict(trust_mm=step,linear_status=int(result.status))
        event['steps'].append(trial)
        if not result.success:
            continue
        for a in (1.,.5,.25):
            candidate=current.copy()
            candidate[free]+=a*result.x[:3*len(free)].reshape(-1,3)*1e-3
            q1=all_quality(candidate)
            score=(int(np.count_nonzero(q1<=0)),float(-np.min(q1)))
            if best is None or score<best[0]:
                best=(score,candidate,a,step)
        if best and best[0][0] < len(broken):
            break
    history.append(event)
    if best is None or best[0][0]>=len(broken):
        print(event,flush=True)
        break
    event.update(accepted_invalid=best[0][0],fraction=best[2],trust_mm=best[3])
    print(event,flush=True)
    current=best[1]

assert np.array_equal(current[fixed],starting[fixed])
q=all_quality(current)
final_failed=np.flatnonzero(q<=0)
changed=np.flatnonzero(np.any(current!=starting,axis=1))
ledger={}
for index,density in ((15,1000.),(16,900.)):
    sums=[]
    for p in (original,current):
        volume=0.
        p=f32(p)
        for lo in range(0,len(tet),120_000):
            r=slice(lo,lo+120_000)
            volume+=float(volumes(p,tet[r][material[r]==index])[0].sum())
        sums.append(volume*density)
    ledger[index]={'original_kg':sums[0],'candidate_kg':sums[1]}
report=dict(input_sha256=sha256(data),cells=len(tet),baseline_invalid=0,
            transferred_invalid=len(invalid),remaining_invalid=len(final_failed),
            remaining_invalid_cell_rows=final_failed[:400].tolist(),exterior_only_invalid=obstructed.tolist(),
            adjusted_nodes=len(changed),max_adjustment_m=float(np.linalg.norm(current-starting,axis=1).max()),
            minimum_normalized_determinant=float(q.min()),local_history=history,material_volume_ledger=ledger,
            crosses_verified=False,semantic_mass_compiled=False,physical_admission=False)
Path('fine-cavity-refit.json').write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed('fine-cavity-refit.npz',changed_node_rows=changed,
                    fitted_world_m=current[changed],original_transferred_world_m=starting[changed])
print({k:v for k,v in report.items() if k not in ('remaining_invalid_cell_rows','exterior_only_invalid','local_history')},flush=True)
