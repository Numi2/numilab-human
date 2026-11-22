"""Make a continuous volumetric construction guess across rigid donor seams at a measured body pose."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from screen_registered_dense_posture import load_geometry, measure


def fit(x, y, owner, cells, rotations, halo, boundary, floor):
    mixed = np.any(owner[cells]!=owner[cells[:,:1]],axis=1)
    free=np.zeros(len(x),dtype=bool)
    free[cells[mixed].ravel()]=True
    for _ in range(halo):
        touching=free[cells].any(axis=1)
        free[cells[touching].ravel()]=True
    # Anchors outside this morphological band retain their exact donor transform.
    # For a contact construction prescribe only a boundary node's vertical position.
    # Tangential coordinates retain the variational solution at that node.
    constraints=boundary & (y[:,2] < floor) if floor is not None else np.zeros(len(y),bool)
    if constraints.any():
        free[cells[np.any(constraints[cells],axis=1)].ravel()]=True
    touching=np.flatnonzero(free[cells].any(axis=1))
    freeidx=np.flatnonzero(free)
    packed=np.full(len(x),-1,dtype=np.int32);packed[freeidx]=np.arange(len(freeidx))
    ind0=[];ind1=[];data=[]; b=np.zeros((len(freeidx),3))
    disp=y-x
    for start in range(0,len(touching),16384):
        t=cells[touching[start:start+16384]]
        tri=x[t];edges=tri[:,1:]-tri[:,:1]
        det=np.linalg.det(edges)
        if np.any(det<=0):raise ValueError('inverted source assembly')
        V=det/6.
        g=np.empty((len(t),4,3))
        g[:,1:]=np.linalg.inv(edges).transpose(0,2,1);g[:,0]=-g[:,1:].sum(axis=1)
        grad=(rotations[owner[t]].mean(axis=1)-np.eye(3)).transpose(0,2,1)
        local=V[:,None,None]*(g@g.transpose(0,2,1))
        rhs=V[:,None,None]*(g@grad)
        indices=packed[t]
        fixed_disp=disp[t].copy();fixed_disp[indices>=0]=0.
        rhs-=local@fixed_disp
        valid=indices>=0
        np.add.at(b,indices[valid],rhs[valid])
        keep=valid[:,:,None] & valid[:,None,:]
        ind0.append(np.broadcast_to(indices[:,:,None],keep.shape)[keep])
        ind1.append(np.broadcast_to(indices[:,None,:],keep.shape)[keep])
        data.append(local[keep])
    K=coo_matrix((np.concatenate(data),(np.concatenate(ind0),np.concatenate(ind1))),shape=(len(b),len(b))).tocsc()
    scale=1./np.sqrt(K.diagonal())
    from scipy.sparse import diags
    pre=diags(scale)
    K=(pre@K@pre).tocsc()
    b*=scale[:,None]
    print('seam extension',len(freeidx),'free nodes',len(touching),'incident tetrahedra',K.nnz,'nonzeros',flush=True)
    solve=splu(K,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.0,options={'SymmetricMode':True})
    u=solve.solve(b)
    rel=float(np.linalg.norm(K@u-b)/np.linalg.norm(b))
    if rel>1.e-7:raise ValueError(('inaccurate initial-pose extension',rel))
    if constraints.any():
        high=constraints[freeidx]; low=~high
        u[high,2]=(floor-x[freeidx[high],2])/scale[high]
        rhs=b[low,2]-K[low][:,high]@u[high,2]
        solve_z=splu(K[low][:,low].tocsc(),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
        u[low,2]=solve_z.solve(rhs)
    out=y.copy();out[freeidx]=x[freeidx]+u*scale[:,None]
    return out,dict(free_nodes=len(freeidx),incident_tetrahedra=len(touching),intersection_of_donor_banks=int(mixed.sum()),
                    fixed_normal_nodes=int(constraints.sum()),nonzero_matrix_entries=K.nnz,relative_linear_residual=rel)


def run(full,donor,dst,halo,floor):
    start=time.monotonic()
    x,y,owner,cells,mat,outer,rotations,pins=load_geometry(full,donor)
    out,solver=fit(x,y,owner,cells,rotations,halo,outer,floor)
    report=dict(solver=solver,source=pins,halo=halo,construction_floor_m=floor,
                raw=measure(y,owner,cells,mat),continuous=measure(out,owner,cells,mat),
                outer_penetrating=int(np.count_nonzero(out[outer,2]<0)),
                outer_min_gap_m=float(np.min(out[outer,2])),
                moved_from_rigid_guess_rms_m=float(np.sqrt(np.mean(np.sum((out-y)**2,axis=1)))),
                wall_s=time.monotonic()-start,scope='One-time compatible-gradient construction across the bonded volume. Native material/contact acceptance is required before use.')
    dst.mkdir(exist_ok=False);(dst/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    np.save(dst/'displacement.npy',out-x)
    print(json.dumps(report['continuous']),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('full','donor','dst'):p.add_argument(name,type=Path)
    p.add_argument('--halo',type=int,default=0);p.add_argument('--floor',type=float)
    a=p.parse_args();run(a.full,a.donor,a.dst,a.halo,a.floor)
