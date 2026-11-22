"""Solve remaining local Float32 lattice constraints and recheck the original nonlinear tetrahedra."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import csr_matrix,eye,hstack,vstack
from construct_registered_donor_field import errors
from screen_registered_dense_posture import load_geometry,measure
from recover_material_thickness import audit
from untangle_registered_donor_field import differentials
from fit_connected_material_cells import groups,severity


def solve(y,t,ref,nodes,original,radius,fixed=None):
    ids=np.searchsorted(nodes,t).clip(max=len(nodes)-1);mask=nodes[ids]==t
    p=y[nodes].astype('f4');ulps=np.abs(np.spacing(p)).astype(float)
    if not np.all(ulps>0):raise ValueError('invalid emission')
    current=(y[nodes]-original[nodes])
    lower=np.maximum(-64,np.ceil((-radius-current)/ulps)).ravel()
    upper=np.minimum(64,np.floor((radius-current)/ulps)).ravel()
    if fixed is not None:
        lower[2::3]=np.maximum(lower[2::3],np.maximum(-64,np.ceil(-y[nodes,2]/ulps[:,2])))
        lock=np.repeat(fixed[nodes],3);lower[lock]=0.;upper[lock]=0.
    if np.any(lower>upper):raise ValueError('outside construction box')
    det,grad,_=differentials(y,t);n=3*len(nodes)
    # Each row is the oriented margin in material units, and its derivative under
    # one integer step at each moving Float32 coordinate. Ignore only unreachable rows.
    parts=[];rhs=[]
    def inequality(g,gap,margin):
        coeff=np.where(mask[:,:,None],g*ulps[ids],0.)
        border=np.where(gap>0,np.minimum(margin,gap*.2),margin)
        b=border-gap
        limit=np.maximum(abs(lower),abs(upper)).reshape(-1,3)
        reachable=b>=-(abs(coeff)*limit[ids]).sum((1,2))
        if not reachable.any():return
        indices=(3*ids[:,:,None]+np.arange(3))[reachable]
        c=coeff[reachable];v=b[reachable]
        scale=np.maximum(abs(c).sum((1,2)),1e-40)
        parts.append(csr_matrix(((c/scale[:,None,None]).ravel(),indices.ravel(),
                   np.arange(0,c.size+1,12)),shape=(len(v),n)))
        rhs.append(v/scale)
    inequality(grad,det-.4*ref,.004*ref);inequality(-grad,2.5*ref-det,.004*ref)
    inequality(grad,det-6e-18,np.full(len(t),4e-20))
    v=y[t]
    for i in range(4):
        for j in range(i):
            e=v[:,i]-v[:,j];l=np.linalg.norm(e,axis=1);g=grad.copy();d=(3e-5*l)[:,None]*e
            g[:,i]-=d;g[:,j]+=d
            inequality(g,det-1e-5*l**3,.008e-5*l**3)
    A=vstack(parts,format='csr');b=np.concatenate(rhs)
    band=vstack([eye(n),-eye(n)],format='csr')
    cons=vstack([hstack([A,csr_matrix(A.shape)]),hstack([-band,vstack([eye(n),eye(n)])])],format='csc')
    res=milp(c=np.r_[np.zeros(n),np.ones(n)],integrality=np.r_[np.ones(n),np.zeros(n)],
             bounds=Bounds(np.r_[lower,np.zeros(n)],np.r_[upper,np.maximum(-lower,upper)]),
             constraints=LinearConstraint(cons,np.r_[b,np.zeros(2*n)],np.inf),
             options={'time_limit':4.,'mip_rel_gap':.10})
    if res.x is None:return None
    z=(y[nodes]+np.rint(res.x[:n]).reshape(-1,3)*ulps).astype('f4').astype(float)
    return z


def main(full,donor,seed,warm,out):
    x,target,owner,t,mat,outer,_,pins=load_geometry(full,donor)
    for f in (seed,warm):
        if json.loads((f/'result.json').read_text())['source']!=pins:raise ValueError('wrong source')
    exact=x+np.load(seed/'displacement.npy');y=(x+np.load(warm/'displacement.npy')).astype('f4').astype(float)
    ref=np.concatenate([differentials(x,t[k:k+32768])[0] for k in range(0,len(t),32768)])
    boundary=seed/'construction-boundary.npz';fixed=None
    if boundary.exists():
        data=np.load(boundary);fixed=np.zeros(len(x),bool);fixed[data['node']]=True
        if not np.all(data['axis']) or np.any(y[:,2]<0) or not np.array_equal(y[data['node']],data['value']):raise ValueError('unmatched initial anchor/bed')
    history=[]
    for it in range(6):
        J,s,d,bad=audit(y,t,ref);log=dict(iteration=it,failed=int(bad.sum()),inverted=int((d<=0).sum()))
        history.append(log);print(log,flush=True)
        if not bad.any():break
        active=bad.copy()
        if it>=1:
            touch=np.zeros(len(x),bool);touch[t[bad].ravel()]=True
            active|=touch[t].any(1)
        changes=0
        for nodes,incident in groups(t,active):
            part=t[incident];r=ref[incident];start=y[nodes].copy()
            proposal=solve(y,part,r,nodes,exact,2.e-6,fixed)
            if proposal is None:continue
            y[nodes]=proposal;j,h,det,f=audit(y,part,r)
            def merit(a,b):return (int(a.sum()),float(np.linalg.norm(b)))
            if merit(f,severity(j,h,det,r))<merit(bad[incident],severity(J[incident],s[incident],d[incident],r)):changes+=1
            else:y[nodes]=start
        if it>=1 and not changes:break
    j,h,det,bad=audit(y,t,ref)
    out.mkdir(exist_ok=False);np.save(out/'displacement.npy',y-x)
    if fixed is not None:
        import shutil
        if np.any(y[:,2]<0) or not np.array_equal(y[data['node']],data['value']):raise ValueError('lattice constraints lost')
        shutil.copyfile(boundary,out/boundary.name)
    result=dict(source=pins,history=history,seed=str(seed),warm=str(warm),failed=int(bad.sum()),min_J=float(j.min()),max_J=float(j.max()),
                quality=measure(y,owner,t,mat),failure_by_role={str(k):int(np.count_nonzero(bad&(mat==k))) for k in np.unique(mat)},
                correction_to_seed=errors(y,exact),discrepancy_to_rigid_donors=errors(y,target),outer_error=errors(y[outer],target[outer]),
                scope='Emitted Float32 initial-pose coordinates. Original mass mapping, constitutive reference and tetrahedral connectivity are unchanged; loading, anatomical clearance and motion require separate checks.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main(*(Path(v).resolve() for v in sys.argv[1:]))
