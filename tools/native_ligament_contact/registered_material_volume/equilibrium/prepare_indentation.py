"""Construct a flat-bed starting guess using the registered material's consistent tangent.

The contact active set is solved at preparation time. The running scene receives
only current material positions and the plane, never patch forces or constraints.
"""
import json,re,sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from compile_connected_joint import digest
from import_conditioned_interior import write_nodes
from supported_material_admission import geom

def soft_response(x,y,t,mu,bulk):
    a=x[t[:,1:]]-x[t[:,:1]]
    inv=np.linalg.inv(a)
    v=np.linalg.det(a)/6
    g=np.concatenate([-inv.sum(2,keepdims=True),inv],axis=2).transpose(0,2,1)
    def evaluate(y,derivative=False):
        F=(inv@(y[t[:,1:]]-y[t[:,:1]])).transpose(0,2,1)
        J=np.linalg.det(F)
        if np.min(J)<=0:raise ValueError('Inverted material')
        h=g@np.linalg.inv(F)
        p=g@F.transpose(0,2,1)
        ell=bulk*np.log(J)-mu
        force=v[:,None,None]*(mu[:,None,None]*p+ell[:,None,None]*h)
        if not derivative:return force,J
        dot=g@g.transpose(0,2,1)
        K=(mu*v)[:,None,None,None,None]*dot[:,:,:,None,None]*np.eye(3)
        K+=(bulk*v)[:,None,None,None,None]*np.einsum('eai,ebj->eabij',h,h)
        K-=(ell*v)[:,None,None,None,None]*np.einsum('eaj,ebi->eabij',h,h)
        return force,K
    return evaluate

def run(source,previous,bed,output,nonlinear=False):
    output.mkdir(exist_ok=True);v=source/'volume'
    g=json.loads(bed.read_bytes());n=np.array(g['normal']);z0=g['height_m']
    norm=np.linalg.norm(n);normal=n/norm
    tx=np.cross([0,1,0],normal);tx/=np.linalg.norm(tx)
    frame=np.stack((tx,np.cross(normal,tx),normal)).T
    x,original=[np.loadtxt(v/name,skiprows=1,usecols=(1,2,3)) for name in ('mesh.node','current.node')]
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype=np.int32)
    tet=raw[:,:4]-1
    patches=np.load(v/'material-walls.npz')
    outer=np.unique(patches['O'])
    anchors=[int(s) for s in re.findall(r'^\{\d+u,(\d+)u,\d+u,\{',(source/'binding/app/NumiHumanLoadedSkinPoseData.inc').read_text(),re.M)]
    if len(anchors)!=14:raise ValueError('Unmatched anchors')
    fixed=np.unique(np.r_[anchors,patches['D'].ravel(),patches['C'].ravel()])
    gap=original@normal-z0*normal[2]
    seed=outer[gap[outer]<.005]
    # Two connected rings include both sides of the bonded material interface.
    free=seed
    for i in range(2):free=np.setdiff1d(np.unique(tet[np.isin(tet,free).any(1)]),fixed)
    hit=np.isin(tet,free).any(1)
    ids,inv=np.unique(tet[hit],return_inverse=True);t=inv.reshape(-1,4)
    local=np.searchsorted(ids,free);Kmap=np.full(len(ids),-1);Kmap[local]=np.arange(len(free))
    if np.setdiff1d(seed,free).size:raise ValueError('A source attachment touches the bed')
    record=json.loads(previous.read_bytes())
    materials={}
    for filename,sha in record['source_material'].items():
        file=Path(filename)
        if digest(file)!=sha:raise ValueError('Source material changed')
        key=15 if 'skin_dermis' in file.name else 16
        materials[key]=[1e6*float(re.search(r'parameter '+k+r'\s*:\s*MPa\s*=\s*([0-9.eE+-]+)',file.read_text())[1]) for k in ['shear','bulk']]
    mu,bulk=np.array([materials[int(s)] for s in raw[hit,4]]).T
    y=original[ids].copy()
    # Evaluate material in a common orthonormal bed frame. Rigid rotations
    # transform all tensor indices together and preserve our isotropic law.
    X=(x[ids]-x[ids[0]])@frame
    Y=(y-y[0])@frame
    law=soft_response(X,Y,t,mu,bulk)
    f0,k=law(Y,True)
    dof=Kmap[t][:,:,None]*3+np.arange(3);i=dof[:,:,None,:,None];j=dof[:,None,:,None,:]
    I,J=np.broadcast_arrays(i,j);valid=(I>=0)&(J>=0)
    K=coo_matrix((k[valid],(I[valid],J[valid])),shape=(3*len(free),)*2).tocsc()
    K.sum_duplicates()
    pos=np.flatnonzero(np.isin(free,outer));rows=3*pos+2
    eps=np.finfo('f4').eps
    mass=np.fromfile(v/'node-mass.f64','<f8')
    if len(mass)!=len(x) or (mass<=0).any():raise ValueError('Invalid mass')
    # Verify the assembled tangent against centered differences of the exact force.
    rng=np.random.default_rng(76);probe=rng.normal(size=(len(local),3));probe/=np.max(abs(probe))
    def gather(corners):
        f=np.zeros_like(Y);np.add.at(f,t,corners);return f
    D=np.zeros_like(Y);D[local]=probe
    step=1.e-9
    observed=gather((law(Y+step*D)[0]-law(Y-step*D)[0])/(2*step))[local].ravel()
    derr=np.linalg.norm(observed-K@probe.ravel())/np.linalg.norm(observed)
    if derr>1e-5:raise ValueError('Incorrect tangent '+str(derr))
    meta=dict(control_loaded=digest(v/'current.node'),source_reference=digest(v/'mesh.node'),elements=digest(v/'mesh.ele'),source_mass=digest(v/'node-mass.f64'),source_material=record['source_material'],source_bed=digest(bed),contact_frame_receipt=str(bed),free_nodes=len(free),affected_cells=len(t),tangent_difference=float(derr))
    print('free',len(free),'cells',len(t),'differential',derr,flush=True)
    original_force=gather(f0)[local]
    attempts=[];best=None
    for depth in [.00005,.0001,.00025,.0005,.001,.002,.003,.004]:
        height=float(np.float32(z0+depth))
        P=original[free[pos]]
        scale=np.abs(P)@abs(normal)+abs(height*normal[2])
        target=np.maximum(1e-9,16*eps*scale)+16*eps*scale
        bound=(height-z0)*normal[2]+target-gap[free[pos]]
        U=np.zeros(K.shape[0]);active=bound>0
        for it in range(30):
            block=rows[active];mask=np.ones(len(U),bool);mask[block]=False
            U[block]=bound[active]
            U[mask]=splu(K[mask][:,mask]).solve(-K[mask][:,block]@U[block])
            r=K@U
            wrong=active&(r[rows]<-1.e-7)
            crossed=(~active)&(U[rows]<bound-1e-11)
            if not np.any(wrong|crossed):break
            active=(active&~wrong)|crossed
        else:raise ValueError('Contact set did not converge')
        error=None
        if nonlinear:
            from continue_material import refine
            try:
                U,active,error,iterations=refine(law,gather,Y,t,local,rows,bound,U,mass[free],frame[2])
            except ValueError as exc:
                attempts.append(dict(depth_m=depth,admissible=False,reason=str(exc)));break
        ynew=y.copy();ynew[local]=(y[local]+U.reshape(-1,3)@frame.T).astype('f4')
        # Reproject only the *active initialized positions* after Float32 xy
        # emission. Runtime contacts stay unilateral, unconstrained and unseeded.
        contact=local[pos[active]]
        p=ynew[contact]
        scale=abs(p)@abs(normal)+abs(height*normal[2])
        targets=np.maximum(1e-9,16*eps*scale)+16*eps*scale
        ynew[contact,2]=(height+(targets-p[:,:2]@normal[:2])/normal[2]).astype('f4')
        _,q=geom(ynew,t)
        tests=(ynew-y[0])@frame
        try:forces,J=law(tests)
        except ValueError:
            attempts.append(dict(depth_m=depth,admissible=False,reason='inverted'));break
        residual=gather(forces)[local]
        area_force=float(residual[pos[active],2].sum())
        relief=residual-original_force;eliminated=relief.copy();eliminated[pos[active],2]=0
        a=dict(depth_m=depth,height_m=height,normal_reaction_estimate_N=area_force,tangent_contact_N=float(r[block].sum()),absolute_increment_free_residual_N=float(np.linalg.norm(eliminated)),active=int(active.sum()),q_min=float(q.min()),J_min=float(J.min()),J_max=float(J.max()))
        if nonlinear:
            a['local_equilibrium_before_float32_N']=error
            a['newton_steps']=iterations
            physical=residual+mass[free,None]*9.81*frame[2]
            a['normal_reaction_estimate_N']=float(physical[pos[active],2].sum())
            physical[pos[active],2]=np.minimum(physical[pos[active],2],0)
            a['free_residual_after_float32_N']=float(np.linalg.norm(physical))
        a['admissible']=a['q_min']>=1e-5 and a['J_min']>=.4 and a['J_max']<=2.5
        print(a,flush=True);attempts.append(a)
        if not a['admissible']:break
        best=(ynew,height,a)
    (output/'continuation.json').write_text(json.dumps(meta|dict(trials=attempts),indent=2)+'\n')
    if best is None:raise ValueError('No admissible initialization')
    y_fit,height,chosen=best
    changed=np.flatnonzero(np.any(y_fit!=y,axis=1))
    write_nodes(v/'current.node',output/'current.node',{ids[k]:y_fit[k] for k in changed})
    allnew=original.copy();allnew[ids]=y_fit
    gaps=allnew[:,0]*n[0]+allnew[:,1]*n[1]+(allnew[:,2]-height)*n[2]
    if gaps.min()<=0 or np.intersect1d(ids[changed],fixed).size:raise ValueError('Invalid prepared loading')
    meta.update(registered=digest(output/'current.node'),changed_nodes=len(changed),bed_height_m=height,bed_normal=n.tolist(),minimum_gap_m=float(gaps.min()),contact_nodes=np.sort(outer[gaps[outer]<float(np.float32(1e-5))]).tolist(),selected=chosen)
    (output/'fit.json').write_text(json.dumps(meta,indent=2)+'\n')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for arg in ('source','previous','bed','output'):p.add_argument(arg,type=lambda s:Path(s).resolve())
    p.add_argument('--nonlinear',action='store_true')
    run(**vars(p.parse_args()))
