"""Certify sparse three-dimensional material responses to normal bed loads at release."""
from pathlib import Path
import json,sys,time
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.spatial import cKDTree
from material_extension import Factor
from registered_bed_reference import read,sha,mass_ledger,native_plane

def patch_stiffness(x,t,props):
    # P = mu(F-F^-T) + lambda log(J) F^-T at the installed reference.
    # dP[I;dF] = mu(dF+dF^T) + lambda trace(dF) I.
    op=sparse.csr_matrix((3*len(x),3*len(x)))
    eye=np.eye(3)
    for start in range(0,len(t),6000):
        c=t[start:start+6000];mu,lam=props[start:start+len(c)].T
        edge=x[c[:,1:]]-x[c[:,:1]];V=np.linalg.det(edge)/6
        if np.any(V<=0):raise ValueError('Invalid registered physical reference')
        grad=np.linalg.inv(edge).transpose(0,2,1);grad=np.concatenate((-grad.sum(1,keepdims=True),grad),axis=1)
        dofs=(3*c[:,:,None]+np.arange(3)).reshape(len(c),12)
        A=(np.einsum('eac,ebc->eab',grad,grad)[:,:,None,:,None]*eye[None,None,:,None,:]
           + np.einsum('ead,ebc->eacbd',grad,grad))*mu[:,None,None,None,None]
        A+=lam[:,None,None,None,None]*np.einsum('eac,ebd->eacbd',grad,grad)
        A=(V[:,None,None,None,None]*A).reshape(len(c),12,12)
        i=np.broadcast_to(dofs[:,:,None],A.shape);j=np.broadcast_to(dofs[:,None,:],A.shape)
        op+=sparse.coo_matrix((A.ravel(),(i.ravel(),j.ravel())),shape=op.shape).tocsr()
    op.eliminate_zeros()
    return op

def finite_difference(x,t,props,K):
    rng=np.random.default_rng(17)
    H=rng.normal(size=(3,3));u=x@H.T
    modulus=float(u.ravel()@(K@u.ravel()))
    volume=np.linalg.det(x[t[:,1:]]-x[t[:,:1]])/6
    mu,lam=props.T
    def energy(d):
        F=np.eye(3)+d*H;L=np.log(np.linalg.det(F))
        return volume@(.5*mu*(np.sum(F*F)-3-2*L)+.5*lam*L*L)
    h=2e-5;oracle=(energy(h)+energy(-h)-2*energy(0))/h**2
    err=abs(modulus-oracle)/abs(oracle)
    if not np.isfinite(err) or err>3e-6:raise ValueError(('Incorrect physical tangent',err))
    return float(err)

def evaluate(parent,radii,steps):
    from trial import gather
    z,x,t,role,donor,origin,params,frames,anchors,props,pins,Q,shift=gather(parent)
    native=parent/'native-terminal-16';j=read(native/'launch.json');env=j['environment']
    bed=next(Path(s) for s in j['argv'] if s.endswith('.nhcnt'))
    if sha(bed)!=j['pins'][str(bed)]:raise ValueError('Unmatched physical support')
    outer=z['node_O'];height=native_plane(bed)
    sites=outer[(x[outer,2]-height)<=.001]
    m,totals,mhash=mass_ledger(env,t,role,len(x))
    normal=np.array([0.,0.,1.])
    distance=cKDTree(x[sites]).query(x)[0]
    # Include every cell that can respond to an interior displacement, including
    # its exterior neighbour nodes. The unassembled rows then have exactly zero residual.
    touched=distance<=max(radii)
    kept=t[touched[t].any(1)];rows=np.flatnonzero(touched[t].any(1))
    ids,cells=np.unique(kept,return_inverse=True);cells=cells.reshape(-1,4)
    K=patch_stiffness(x[ids],cells,props[role[rows]-15])
    patch_test=finite_difference(x[ids],cells,props[role[rows]-15],K)
    M=np.repeat(m[ids],3)
    it=np.searchsorted(ids,sites);n=len(sites);idx=np.arange(n)
    anchors_in=np.isin(ids,anchors[:,1])
    if np.any(anchors_in[it]) or not np.all(M>0):raise ValueError('Invalid normal load port')
    load=np.zeros((len(M),n));load[3*it+2,idx]=1
    result=[];best={}
    for dt in steps:
        A=(K+sparse.diags(M/dt**2)).tocsr()
        for r in radii:
            active=(distance[ids]<=r)&~anchors_in
            free=np.flatnonzero(np.repeat(active,3));Aii=A[free][:,free].tocsr()
            start=time.monotonic();solver=Factor(Aii,Path('material_factor.dylib'))
            try:Uf=solver.solve(load[free])
            finally:solver.close()
            out=np.zeros_like(load);out[free]=Uf
            residual=load-A@out;residual[np.repeat(anchors_in,3)]=0
            compliance=out[3*it+2]
            reciprocity=float(np.linalg.norm(compliance-compliance.T)/np.linalg.norm(compliance))
            if reciprocity>1e-8:raise ValueError('Reciprocity failure')
            compliance=(compliance+compliance.T)/2
            bound=residual.T@((dt**2/M[:,None])*residual)
            error=float(max(0.,eigh((bound+bound.T)/2,compliance,eigvals_only=True)[-1]))
            internal=float(np.linalg.norm(residual[free])/np.linalg.norm(load))
            rec=dict(step_s=dt,radius_m=r,active_material_nodes=int(active.sum()),full_material_nodes=len(x),
                largest_relative_compliance_error_bound=error,interior_solve_residual=internal,
                reciprocity=reciprocity,lowest_port_displacement_m_per_N=float(compliance[np.argmin(x[sites,2]),np.argmin(x[sites,2])]),
                factor_and_solve_s=time.monotonic()-start)
            result.append(rec);print('shift',dt,'radius',r,'nodes',rec['active_material_nodes'],'bound',error,flush=True)
            if error<=.01 and dt not in best:
                best[dt]=rec
                np.savez_compressed(f'local-response-{dt:g}.npz',physical_nodes=ids[active],ports=sites,
                    compliance_m_per_N=compliance,displacement_m_per_N=Uf.reshape(-1,3,n),
                    force_direction=normal,compliance_uncertainty_m_per_N=bound)
                break
    Path('local-contact.json').write_text(json.dumps(dict(reference=digest_source(native),mass=mhash,material_mass_kg=totals,
        physical_material=pins,bed=sha(bed),port_ids=sites.tolist(),support_band_m=.001,anchors=anchors[:,0].tolist(),
        maximum_patch_nodes=len(ids),incident_tets=len(kept),material_energy_parity=patch_test,checks=result,accepted=best,
        scope='Error bound for all combinations of loads at the 19 physical ports on the full 3D reference skin/fat mass and tangent with the existing deep attachments held at their native pose. Includes lateral degrees of freedom. Other body operators, contact tangents and subsequent material states require fresh certification.'),indent=2,allow_nan=False)+'\n')
def digest_source(case):return sha(case/'physical-reference.node')
if __name__=='__main__':evaluate(Path(sys.argv[1]),[.002,.005,.012,.025,.05,.1,.16],[.000015625,.000125,.0005,.001,.002])
