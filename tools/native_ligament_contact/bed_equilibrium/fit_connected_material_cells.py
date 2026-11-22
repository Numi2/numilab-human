"""Constrained construction of initial coordinates on the original joined skin/fat volume."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack, eye
from scipy.sparse.csgraph import connected_components
from construct_registered_donor_field import errors
from screen_registered_dense_posture import load_geometry, measure
from untangle_registered_donor_field import differentials
from recover_material_thickness import audit


def severity(j, shape, det, ref):
    return np.maximum.reduce([.4-j,j-2.5,(1e-5-shape)/1e-5,(6e-18-det)/ref,np.zeros(len(j))])


def fit_component(y, initial, tet, ref, nodes, trust_m, radius_m, locked=None):
    # Each moving node belongs to precisely one group. All its incident cells are here.
    moving=np.searchsorted(nodes,tet).clip(max=len(nodes)-1)
    present=nodes[moving]==tet
    det,grad,span=differentials(y,tet)
    grad[~present]=0
    norm=np.sqrt(np.sum(grad**2,axis=(1,2)))
    if np.any(norm<=0):raise ValueError('inconsistent movable incidence or degenerate gradients')
    lower=np.maximum(6.12e-18,.42*ref)
    upper=2.47*ref
    def block(g, margin):
        g=np.where(present[:,:,None],g,0.)
        size=np.linalg.norm(g.reshape(len(tet),12),axis=1)
        normalizer=np.maximum(size,1.e-35)
        a=csr_matrix(((g/normalizer[:,None,None]).ravel(),(moving[:,:,None]*3+np.arange(3)).ravel(),
                      np.arange(0,12*len(tet)+1,12)),shape=(len(tet),3*len(nodes)))
        a.eliminate_zeros()
        b=margin/normalizer/trust_m
        active=b<=np.asarray(abs(a).sum(axis=1)).ravel()+1e-8
        return a[active],b[active],(ref/normalizer/trust_m)[active]
    parts=[block(-grad,det-lower),block(grad,upper-det)]
    # Shape uses the longest current edge. Constrain every competing edge directly;
    # for a sliver, shortening that edge is independent of increasing its volume.
    vertices=y[tet];quality=1.01e-5
    for a in range(4):
        for b0 in range(a):
            edge=vertices[:,a]-vertices[:,b0];length=np.linalg.norm(edge,axis=1)
            dshape=-grad.copy();de=(3*quality*length)[:,None]*edge
            dshape[:,a]+=de;dshape[:,b0]-=de
            parts.append(block(dshape,det-quality*length**3))
    mat=vstack([z[0] for z in parts],format='csr')
    b=np.concatenate([z[1] for z in parts]);slack=np.concatenate([z[2] for z in parts])
    n=mat.shape[1];current=(y[nodes]-initial[nodes]).ravel()
    lo=np.maximum(-1.,(-radius_m-current)/trust_m)
    hi=np.minimum(1.,(radius_m-current)/trust_m)
    if locked is not None:
        lo[2::3]=np.maximum(lo[2::3],-y[nodes,2]/trust_m)
        fixed=locked[nodes].ravel()
        lo[fixed]=0.;hi[fixed]=0.
    if np.any(lo>hi+1e-12):raise ValueError('outside material construction trust region')
    bounds=list(zip(lo,np.maximum(lo,hi)))+[(0,None)]
    # Minimize the largest coordinate shift with strict physical row constraints first.
    band=vstack([eye(n),-eye(n)],format='csr')
    rows=vstack([hstack([mat,csr_matrix((len(b),1))]),
                 hstack([band,-np.ones((2*n,1))])],format='csr')
    obj=np.zeros(n+1);obj[-1]=1.
    sol=linprog(obj,A_ub=rows,b_ub=np.r_[b,np.zeros(2*n)],bounds=bounds,method='highs',
                options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not sol.success:
        if sol.status not in (2,4):raise ValueError(sol.message)
        # Elastic phase I, a solver diagnostic only. A fresh nonlinear check decides the move.
        # Normalize slack to relative volume so the thinnest admissible cells retain a vote.
        # Give each conflicting inequality its own elastic slack; one difficult
        # contact neighborhood must not mask the progress of all other rows.
        from scipy.sparse import diags
        elastic=hstack([mat,-mat,-diags(slack)],format='csr')
        weights=np.r_[np.full(2*n,1e-7),np.ones(len(b))]
        local_bounds=list(zip(np.zeros(n),np.maximum(hi,0)))+list(zip(np.zeros(n),np.maximum(-lo,0)))
        sol=linprog(weights,A_ub=elastic,b_ub=b,bounds=local_bounds+[(0,None)]*len(b),
                    method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        if not sol.success:raise ValueError(sol.message)
        return (sol.x[:n]-sol.x[n:2*n]).reshape(-1,3)*trust_m
    if not sol.success:raise ValueError(sol.message)
    # A minimax optimum can leave unconstrained tangential coordinates at either box
    # edge. Minimize total displacement among equally feasible solutions.
    cap=sol.x[-1]+1e-8
    lo=np.maximum(lo,-cap);hi=np.minimum(hi,cap)
    rhs=b+1e-10
    # Positive/negative parts have exact bounds since zero is always inside the box.
    # The objective prevents both signs from being nonzero for any single coordinate.
    sparse=linprog(np.ones(2*n),A_ub=hstack([mat,-mat],format='csr'),b_ub=rhs,
                   bounds=list(zip(np.zeros(n),np.maximum(hi,0)))+list(zip(np.zeros(n),np.maximum(-lo,0))),
                   method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not sparse.success:raise ValueError(sparse.message)
    return (sparse.x[:n]-sparse.x[n:]).reshape(-1,3)*trust_m


def groups(t,active):
    nodes=np.unique(t[active]);packed=np.full(t.max()+1,-1,dtype=np.int32)
    packed[nodes]=np.arange(len(nodes));idx=packed[t];incident=np.flatnonzero((idx>=0).any(1))
    members=idx[incident];anchor=members.max(1)
    rr=np.broadcast_to(anchor[:,None],members.shape);keep=members>=0
    graph=coo_matrix((np.ones(keep.sum(),dtype=bool),(rr[keep],members[keep])),shape=(len(nodes),len(nodes)))
    count,label=connected_components(graph,directed=False)
    return [(nodes[label==k],incident[label[anchor]==k]) for k in range(count)]


def run(full,donor,seed,out,steps,trust_m,radius_m):
    x,target,owner,t,mat,outer,_,pins=load_geometry(full,donor)
    if json.loads((seed/'result.json').read_text())['source']!=pins:raise ValueError('unmatched source and pose')
    y=x+np.load(seed/'displacement.npy');initial=y.copy()
    boundary=seed/'construction-boundary.npz';locked=None
    if boundary.exists():
        data=np.load(boundary);locked=np.zeros_like(y,dtype=bool)
        locked[data['node']]=data['axis']
        if not np.allclose(y[data['node']][data['axis']],data['value'][data['axis']],atol=2e-15,rtol=0):raise ValueError('unmatched physical boundary')
        y[locked]=data['value'][data['axis']]
        initial=y.copy()
    ref=np.concatenate([differentials(x,t[i:i+32768])[0] for i in range(0,len(t),32768)])
    history=[];carry=np.zeros(len(t),bool)
    for it in range(steps+1):
        J,shape,det,bad=audit(y,t,ref)
        score=severity(J,shape,det,ref)
        entry=dict(iteration=it,inverted=int((det<=0).sum()),failed=int(bad.sum()),worst=float(score.max()))
        history.append(entry);print(entry,flush=True)
        if not bad.any() or it==steps:break
        active=bad|carry
        if bad.sum()<=32:
            # The final witnesses may need tangential motion of the neighboring
            # tissue. Expand the construction freedom one ring, verifying its full star.
            touched=np.zeros(len(x),bool);touched[t[bad].ravel()]=True
            active=active|touched[t].any(axis=1)
        accepted=0;blocked=0;new=np.zeros(len(t),bool)
        for nodes,incident in groups(t,active):
            tt=t[incident];rr=ref[incident]
            if not bad[incident].any():continue
            dy=fit_component(y,initial,tt,rr,nodes,trust_m,radius_m,locked)
            before=y[nodes].copy()
            def key(s):
                return float(np.linalg.norm(s))
            best=key(score[incident]);fraction=0.
            for alpha in (1.,.75,.5,.25,.125,.0625,.03125):
                y[nodes]=before+alpha*dy
                j,h,d,f=audit(y,tt,rr)
                if alpha==1.:
                    new[incident[f]]=True
                candidate=key(severity(j,h,d,rr))
                if candidate<best*(1.-1.e-7):
                    best=candidate;fraction=alpha
            y[nodes]=before+fraction*dy
            if fraction:accepted+=1
            else:
                blocked+=1
                # Add the immediate neighbor cells that actually constrained this failed step.
                near_limit=(J[incident]<.48)|(J[incident]>2.38)|(shape[incident]<1.08e-5)|(det[incident]<6.5e-18)
                new[incident[near_limit]]=True
        entry.update(accepted_groups=accepted,blocked_groups=blocked)
        if not accepted and np.all(~new|active):break
        carry|=new
    if locked is not None:
        if not np.array_equal(y[locked],initial[locked]) or y[:,2].min() < -1.e-14:raise ValueError('lost loaded boundary constraint')
        y[:,2]=np.maximum(0.,y[:,2])
    high=y.astype('f4');low=(y-high.astype(float)).astype('f4')
    out.mkdir(exist_ok=False);np.save(out/'displacement.npy',y-x)
    if boundary.exists():
        import shutil
        shutil.copyfile(boundary,out/boundary.name)
    rec=dict(source=pins,history=history,seed=str(seed),trust_m=trust_m,radius_per_coordinate_m=radius_m,
             quality=measure(y,owner,t,mat),compensated_emission=measure(high.astype(float)+low.astype(float),owner,t,mat),
             float32_world_emission=measure(high.astype(float),owner,t,mat),correction_to_seed=errors(y,initial),
             discrepancy_to_rigid_donors=errors(y,target),outer_error=errors(y[outer],target[outer]),
             scope='Proposed displacement in initial pose. Reference mesh, topology and body masses remain the source inputs. Requires physical attachment, stress, surface and native motion verification.')
    (out/'result.json').write_text(json.dumps(rec,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for v in ('full','donor','seed','out'):p.add_argument(v,type=Path)
    p.add_argument('--steps',type=int,default=12);p.add_argument('--trust-m',type=float,default=.0004)
    p.add_argument('--radius-m',type=float,default=.0015)
    run(**vars(p.parse_args()))
