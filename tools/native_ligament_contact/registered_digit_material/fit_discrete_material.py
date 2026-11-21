"""Fit a connected registration in the tetrahedral basis actually consumed by Matter."""
from pathlib import Path
import sys, json
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from transfer_connected_volume import geometry

def solve(material, target, out, length):
    data=np.load(material);x=data['x'];tet=data['tets'];tags=data['tags']
    ys=np.load(target);delta=ys.astype(float)-x
    free=np.flatnonzero(np.any(delta!=0,axis=1))
    rows=np.full(len(x),-1,np.int32);rows[free]=np.arange(len(free))
    affected=np.flatnonzero(np.any(rows[tet]>=0,axis=1))
    t=tet[affected];vol,quality=geometry(x,t)
    if vol.min()<=1e-18 or quality.min()<=1e-5:raise ValueError('Invalid reference')
    # Lump the L2 matching term by material volume. Use the full reference
    # Dirichlet energy of displacement on each actual linear tetrahedron.
    # Fixed exterior displacements are zero, preserving the patch boundary.
    mass=np.bincount(t.ravel(),weights=np.repeat(vol/4,4),minlength=len(x))[free]
    mat=diags(mass,format='csc')
    for start in range(0,len(t),32000):
        p=x[t[start:start+32000]].astype(float)
        G=np.linalg.inv(p[:,1:]-p[:,:1])
        grad=np.concatenate((-G.sum(axis=2,keepdims=True),G),axis=2)
        K=grad.transpose(0,2,1)@grad
        K*=((length**2)*vol[start:start+len(p)])[:,None,None]
        ids=rows[t[start:start+len(p)]]
        rr=np.broadcast_to(ids[:,:,None],K.shape);cc=np.broadcast_to(ids[:,None,:],K.shape)
        keep=(rr>=0)&(cc>=0)
        mat+=coo_matrix((K[keep],(rr[keep],cc[keep])),shape=mat.shape).tocsc()
    scale=1/np.sqrt(mat.diagonal())
    A=diags(scale)@mat@diags(scale)
    factor=splu(A.tocsc(),permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0,options={'SymmetricMode':True})
    rhs=mass[:,None]*delta[free]
    u=scale[:,None]*factor.solve(scale[:,None]*rhs)
    err=np.linalg.norm(mat@u-rhs)/np.linalg.norm(rhs)
    if err>2e-6:raise ValueError(f'Incomplete fit solve: {err}')
    y=x.copy();y[free]=(x[free].astype(float)+u).astype('f4')
    v,q=geometry(y,t)
    r=dict(regularization_length_m=length,free_nodes=len(free),affected_cells=len(t),residual=err,
           invalid_volume=int((v<=1e-18).sum()),invalid_quality=int((q<=1e-5).sum()),
           failing_cells=int(((v<=1e-18)|(q<=1e-5)).sum()),inversions=int((v<=0).sum()),
           min_quality=float(q.min()),min_volume_m3=float(v.min()),
           maximum_target_miss_mm=1000*float(np.linalg.norm(u-delta[free],axis=1).max()),
           rms_target_miss_mm=1000*float(np.sqrt(np.mean(np.linalg.norm(u-delta[free],axis=1)**2))),
           min_local_volume_ratio=float((v/vol).min()),max_local_volume_ratio=float((v/vol).max()))
    np.save(out.with_suffix('.npy'),y)
    out.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
if __name__=='__main__':solve(*map(Path,sys.argv[1:4]),float(sys.argv[4]))
