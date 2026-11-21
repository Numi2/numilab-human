"""Condense registered cut-muscle elasticity onto distributed anatomical interface tractions."""
import json,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from pelvic_specimen import properties
from material_context import sha

def stiffness(x,t,volume,mu,lam):
    ds=x[t[:,1:]]-x[t[:,:1]]
    gd=np.linalg.inv(ds).transpose(0,2,1)
    grad=np.concatenate((-gd.sum(1)[:,None],gd),axis=1)
    dot=grad@grad.transpose(0,2,1)
    # K[ia,jb] = V(mu grad_i.grad_j delta_ab + mu grad_ib grad_ja + lambda grad_ia grad_jb).
    block=mu*np.einsum('tij,ab->tiajb',dot,np.eye(3))+mu*np.einsum('tib,tja->tiajb',grad,grad)+lam*np.einsum('tia,tjb->tiajb',grad,grad)
    free=(3*t[:,:,None]+np.arange(3)).reshape(-1,12)
    A=sparse.coo_matrix(((block*volume[:,None,None,None,None]).ravel(),(np.broadcast_to(free[:,:,None],(len(t),12,12)).ravel(),np.broadcast_to(free[:,None,:],(len(t),12,12)).ravel())),shape=(3*len(x),3*len(x))).tocsc()
    return A,grad

def pressure_regions(pts,n,area,step):
    # Deterministic groups of similar local surface orientation, bounded in position.
    labels=np.floor((pts-pts.min(0))/step).astype(int)
    dominant=abs(n).argmax(1);polarity=(n[np.arange(len(n)),dominant]>=0).astype(int)
    labels=np.column_stack((labels,2*dominant+polarity))
    return np.unique(labels,axis=0,return_inverse=True)[1]

def prepare(project,old,material,out,width):
    out.mkdir(exist_ok=True);spec=properties(material);summary={}
    for sid in (29,30):
        p=np.load(project/'projected'/f'{sid}.npz');port=np.load(project/'ports'/f'{sid}.npz');cl=np.load(project/'partition-layers'/f'muscle_{sid}.npz')
        entry=np.load(old/'native-release-motion'/f'entrance-{sid}.npz')
        x=p['vertices'].astype(float);t=p['cells'].astype(int);v=p['cell_volumes'];loaded=v>0
        active=np.unique(t[loaded]);A,grad=stiffness(x,t[loaded],v[loaded],spec['mu'],spec['lam'])
        src=entry['faces'];sites=json.loads((old/'posterior/binding-parity.json').read_text())[str(sid)]['actuator_sites_world_m']
        remain=cl['vertices'][np.unique(cl['triangles'][cl['parent_bank']==f'muscle_{sid}'])]
        survives=cKDTree(remain).query(x)[0]<2.e-9
        named={};fix=set()
        for site,point in sites.items():
            seed=cKDTree(x).query(point)[1];ring=np.unique(src[(src==seed).any(1)])
            selected=ring[survives[ring]&np.isin(ring,active)]
            if not len(selected):raise ValueError(('Missing surviving named attachment',site))
            named[site]=selected.tolist();fix.update(selected)
        free=np.setdiff1d(active,list(fix));ids=(3*free[:,None]+np.arange(3)).ravel()
        sub=A[ids][:,ids].tocsc();scale=1/np.sqrt(sub.diagonal());D=sparse.diags(scale)
        factor=splu((D@sub@D).tocsc())
        patch=p['port_vertices'];force=-np.cross(patch[:,1]-patch[:,0],patch[:,2]-patch[:,0])*.5
        area=np.linalg.norm(force,axis=1);valid=area>0
        if not valid.all():raise ValueError('Degenerate integration facet')
        n=force/area[:,None];patch_ids=np.zeros(len(n),int);total=0
        for bank in np.unique(port['target_bank']):
            ix=np.flatnonzero(port['target_bank']==bank);labels=pressure_regions(patch[ix].mean(1),n[ix],area[ix],width)
            patch_ids[ix]=total+labels;total+=max(labels)+1
        group_area=np.bincount(patch_ids,area);node=p['cells'][p['port_owner_cell']].astype(int)
        bary=p['port_shape'].mean(1)
        G=np.zeros((len(x),3,total))
        for j in range(4):np.add.at(G.transpose(0,2,1),(node[:,j],patch_ids),force*(bary[:,j]/group_area[patch_ids])[:,None])
        G=G.reshape(3*len(x),total)
        Phi=np.zeros_like(G)
        Phi[ids]=scale[:,None]*factor.solve(scale[:,None]*G[ids])
        residual=sub@Phi[ids]-G[ids]
        compliance=G.T@Phi
        h=(compliance+compliance.T)*.5
        eigen,rotation=eigh(h)
        # Positive energy normalization; retain all numerically representable pressure responses.
        select=eigen>max(eigen[-1]*1e-11,1e-16)
        modes=(Phi@rotation[:,select])/np.sqrt(eigen[select])
        energy=modes.T@(A@modes);error=float(np.max(abs(energy-np.eye(len(energy)))))
        if max(abs(residual).ravel())>1e-7 or error>1e-5:raise ValueError(('Uncertified response',error,abs(residual).max()))
        # First-order rigid material motion must store zero strain energy before the anatomical attachments are applied.
        rigid=np.cross(np.array([.7,.3,-.52]),x-x.mean(0))+.24
        null=float(np.linalg.norm(A@rigid.ravel(),ord=np.inf))
        if null>1e-6:raise ValueError(('Rigid nullspace',null))
        rng=np.random.default_rng(sid);u=rng.normal(size=x.shape)*1.e-8
        g=np.einsum('tki,tkj->tij',u[t[loaded]],grad);strain=.5*(g+g.transpose(0,2,1))
        exact=v[loaded]@(spec['mu']*(strain*strain).sum((1,2))+.5*spec['lam']*np.trace(strain,axis1=1,axis2=2)**2)
        cooked=.5*u.ravel()@(A@u.ravel())
        np.testing.assert_allclose(cooked,exact,rtol=1e-10,atol=1e-24)
        np.savez_compressed(out/f'{sid}.npz',modes=modes.reshape(len(x),3,-1),fixed=np.array(sorted(fix)),pressure_region=patch_ids,region_areas=group_area,compliance_eigenvalues=eigen,mode_directions=rotation[:,select])
        summary[sid]=dict(material=sha(material),source_sha256=sha(project/'projected'/f'{sid}.npz'),width_m=width,regions=int(total),retained_modes=int(select.sum()),free_nodes=len(free),fixed_nodes=len(fix),named_registered_attachment_nodes=named,abs_force_residual_N=float(abs(residual).max()),energy_orthogonality_error=error,rigid_null_residual_N=null,minimum_eigenvalue=eigen[0],maximum_eigenvalue=eigen[-1],reference_energy_error_J=float(abs(cooked-exact)))
        print(sid,{k:v for k,v in summary[sid].items() if k!='named_registered_attachment_nodes'},flush=True)
    (out/'response.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':prepare(*map(Path,sys.argv[1:5]),float(sys.argv[5]))
