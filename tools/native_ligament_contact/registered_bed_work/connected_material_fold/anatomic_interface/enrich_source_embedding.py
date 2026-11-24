"""Keep independent material coordinates wherever an embedded patch loses measured strain."""
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix,diags,save_npz
from supported_material_admission import geom
from compile_connected_joint import digest


def strain_gradient(edges,du):
    H=np.linalg.solve(edges,du[:,1:]-du[:,:1])
    return .5*(H+H.transpose(0,2,1)+H @ H.transpose(0,2,1))


def deficient_nodes(matrix,fit,initial,mass):
    """Return fine rows which depend on unmatched or numerically dependent controls."""
    from scipy.sparse.csgraph import maximum_bipartite_matching
    from scipy.sparse.linalg import splu
    nodes=np.flatnonzero(fit)
    active=matrix[fit[initial]]
    used=np.unique(active.indices)
    if not len(used):
        return nodes
    A=active[:,used].tocsr()
    squared=A.multiply(A).multiply(mass[nodes,None]).tocsr()
    diagonal=np.asarray(squared.sum(axis=0)).ravel()
    scaled=squared.multiply(1/diagonal).tocsr()
    scaled.data=(scaled.data>1e-12).astype(float);scaled.eliminate_zeros()
    matching=maximum_bipartite_matching(scaled,perm_type='row')
    bad=np.flatnonzero(matching<0)
    if not len(bad):
        D=diags(1/np.sqrt(diagonal))
        gram=(D @ (A.T @ A.multiply(mass[nodes,None])) @ D).tocsc()
        try:
            lu=splu(gram)
            if abs(lu.U.diagonal()).min()>1e-10:
                return np.empty(0,dtype='i4')
        except RuntimeError:
            pass
        # A tiny shift locates troublesome pivots. The admission test above
        # always factors the unshifted mass and the physics receives no shift.
        inspect=gram.copy();inspect.setdiag(inspect.diagonal()+1e-13)
        lu=splu(inspect,diag_pivot_thresh=0.,permc_spec='MMD_AT_PLUS_A')
        bad=np.argsort(lu.perm_c)[abs(lu.U.diagonal())<1e-9]
        if not len(bad):
            raise ValueError('Could not identify independent projected mass rows')
    return nodes[np.asarray(A[:,bad].getnnz(axis=1)>0)]


def enrich(stage,x,u,proposed,fit,matrix,q,mass,tet,material,regions,audit):
    initial=fit.copy()
    start=u.copy()
    start[fit]=proposed[fit]
    # The fidelity requirement is regional. Selecting an identical relative
    # error on every near-stationary tetrahedron can add more coordinates
    # than the original body; score cells by their actual volume contribution.
    basevol,baseshape=geom(x+u,tet)
    material_region=regions+material*(1+regions.max())
    expected=np.zeros(len(tet));volume=np.empty(len(tet));difference=np.zeros(len(tet))
    for a in range(0,len(tet),16384):
        ids=slice(a,a+16384);t=tet[ids];e=x[t[:,1:]]-x[t[:,:1]]
        volume[ids]=abs(np.linalg.det(e))/6
        exact=strain_gradient(e,u[t]);expected[ids]=volume[ids]*np.sum(exact*exact,axis=(1,2))
    limit=.05**2*np.bincount(material_region,weights=expected)
    proposed_coordinates=x+start
    active=np.flatnonzero(fit[tet].any(axis=1))
    batches=[]
    for sweep in range(64):
        violating=[]
        for a in range(0,len(active),16384):
            ids=active[a:a+16384];t=tet[ids]
            e=x[t[:,1:]]-x[t[:,:1]]
            err=strain_gradient(e,u[t])-strain_gradient(e,start[t])
            difference[ids]=volume[ids]*np.sum(err*err,axis=(1,2))
            v,shape=geom(proposed_coordinates,t)
            quality=(v<=0)|((shape<1e-5)&((baseshape[ids]>=1e-5)|(shape<baseshape[ids]-1e-12)))
            if np.any(quality):
                violating.append(ids[quality])
        totals=np.bincount(material_region,weights=difference,minlength=len(limit))
        for region in np.flatnonzero(totals>limit):
            ids=np.flatnonzero((material_region==region)&(difference>0))
            order=ids[np.argsort(-difference[ids],kind='stable')]
            need=totals[region]-.7*limit[region]
            end=int(np.searchsorted(np.cumsum(difference[order]),need))+1
            violating.append(order[:end])
        if violating:
            reason='strain or material validity'
            promote=np.unique(tet[np.unique(np.concatenate(violating))])
            promote=promote[fit[promote]]
        else:
            reason='projected mass rank'
            promote=deficient_nodes(matrix,fit,initial,mass)
            if not len(promote):
                break
        if not len(promote):
            raise ValueError('Unrepairable approximation outside the embedded nodes')
        fit[promote]=False;start[promote]=u[promote]
        proposed_coordinates[promote]=x[promote]+u[promote]
        changed=np.zeros(len(x),bool);changed[promote]=True
        active=np.flatnonzero(changed[tet].any(1))
        batches.append(dict(sweep=sweep+1,reason=reason,promoted_nodes=len(promote),remaining_embedded=int(fit.sum())))
    else:
        raise ValueError('Batch refinement did not meet its regional limit')
    local=matrix[fit[initial]]
    used=np.unique(local.indices)
    remaining=np.flatnonzero(~fit)
    node_ids=np.flatnonzero(fit)
    local=local.tocoo()
    basis=csr_matrix((np.r_[local.data,np.ones(len(remaining))],
                      (np.r_[node_ids[local.row],remaining],np.r_[np.searchsorted(used,local.col),len(used)+np.arange(len(remaining))])),
                     shape=(len(x),len(remaining)+len(used)))
    coords=np.concatenate((q[used],x[remaining]))
    error=float(np.linalg.norm(basis @ coords-x,axis=1).max())
    if error>1e-9 or np.any(abs(basis @ np.ones(basis.shape[1])-1)>1e-11):
        raise ValueError('Loss of reference geometry or partition unity')
    # The projected mass is consistent, including its off-diagonal entries.
    # Check translation, rotation and their coupling as complete quadratic forms.
    mass_q=basis.T @ basis.multiply(mass[:,None])
    from scipy.sparse.linalg import splu
    block=mass_q[:len(used),:len(used)].tocsc()
    scales=diags(1/np.sqrt(block.diagonal()))
    gram=(scales @ block @ scales).tocsc()
    try:
        lu=splu(gram)
        min_pivot=float(np.min(np.abs(lu.U.diagonal())))
    except RuntimeError:
        min_pivot=0.
    def rigid(p):
        V=np.zeros((len(p),3,6))
        V[:,:,:3]=np.eye(3)
        for j,e in enumerate(np.eye(3)):
            V[:,:,3+j]=np.cross(e,p)
        return V
    full=rigid(x);coarse=rigid(coords)
    Mfull=np.einsum('nij,nik,n->jk',full,full,mass)
    Msmall=sum(coarse[:,i].T @ (mass_q @ coarse[:,i]) for i in range(3))
    inertia_error=float(np.linalg.norm(Msmall-Mfull)/np.linalg.norm(Mfull))
    if inertia_error>1e-10:raise ValueError('Projected rigid inertia differs')
    path=stage/'source-embedded-physical-rows.npz'
    save_npz(path,basis)
    shape=audit(x,u,start,tet,material,regions)
    if any(d['inverted'] for d in shape.values()):
        raise ValueError('Projected displacement inverts material')
    return dict(promotions=batches,minimum_equilibrated_mass_LU_pivot=min_pivot,passes_mass_pivot_check=min_pivot>1e-10,embedded_nodes=len(node_ids),independent_nodes=basis.shape[1],reduced_coordinate_fraction=1-basis.shape[1]/len(x),projected_mass_relative_rigid_inertia_error=inertia_error,maximum_reference_reconstruction_error_m=error,projected_mass_kg=float(mass_q.sum()),source_mass_kg=float(mass.sum()),sparse_matrix=digest(path),control_order='Referenced source-parent nodes in ascending order, followed by retained current fine nodes in ascending order.',incremental_strain_and_validity=shape,scope='Same-frame adaptive fit. Its displacement and strain checks are in-sample. No nonlinear, force-response or sustained timing qualification.')
