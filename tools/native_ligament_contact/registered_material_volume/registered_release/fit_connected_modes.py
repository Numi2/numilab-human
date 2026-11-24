"""Balance connected tissue volume accelerations inferred from the accepted full-body release."""
from pathlib import Path
import json,sys,re
import numpy as np
from scipy.sparse import coo_matrix, vstack, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsmr
from compile_connected_joint import digest
from import_conditioned_interior import write_nodes
from prepare_indentation import soft_response
from supported_material_admission import geom
from measure_mode import jac_volume

def build(h,v,material,ports,bindings,out):
    data=lambda n:json.loads(n.read_bytes())
    read=lambda n:np.loadtxt(n,skiprows=1,usecols=(1,2,3))
    x,y,z=map(read,(v/'mesh.node',h/'guarded-fit/current.node',h/'guarded-motion/physical-accepted.node'));delta=z-y
    prior=data(h/'guarded-fit/fit.json');clock=data(h/'motion-check/motion.json')
    if digest(h/'guarded-motion/physical-accepted.node')!=clock['physical_frame'] or digest(h/'guarded-fit/current.node')!=prior['registered']:raise ValueError('Different control')
    raw=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=raw[:,:4]-1;role=raw[:,4];dt=clock['accepted_s']
    mass=np.fromfile(v/'node-mass.f64','<f8');coef=data(material)['materials']
    mu,bulk=np.array([[coef[str(m)]['registered'][k]for k in ('source_mu_pa','source_logJ_coefficient_pa')] for m in (15,16)]).T
    walls=np.load(v/'material-walls.npz');port=np.load(ports/'material-contact.npz')
    protected=np.unique(np.r_[walls['O'].ravel(),walls['D'].ravel(),walls['C'].ravel(),port['facets'][:,4:7].ravel(),port['samples'][:,0].astype('i4')])
    attachments=np.asarray([int(n) for n in re.findall(r'^\{\d+u,(\d+)u,\d+u,\{',bindings.read_text(),re.M)],dtype='i4')
    if not len(attachments):raise ValueError('Missing source attachment records')
    protected=np.union1d(protected,attachments)
    targets=[]
    for lo in range(0,len(t),32768):
        c=t[lo:lo+32768]
        det=lambda p:np.linalg.det(p[c[:,1:]]-p[c[:,:1]])
        vol=det(x);start=det(y)/vol;last=det(z)/vol;pred=det(z+delta)/vol
        # The release certificate also bounds volume against its registered starting pose.
        bad=(pred<np.maximum(.35,.35*start))|(pred>2.5)
        targets.extend((np.flatnonzero(bad)+lo).tolist())
    targets=np.asarray(targets);target_nodes=np.unique(t[targets])
    incident=np.flatnonzero(np.isin(t,target_nodes).any(1))
    free=np.setdiff1d(np.unique(t[incident]),protected)
    active=np.union1d(incident,np.flatnonzero(np.isin(t,free).any(1)))
    ids=np.unique(t[active]);T=np.searchsorted(ids,t[active]);X,Y,Z=x[ids],y[ids],z[ids];dY=Z-Y
    loc=np.searchsorted(ids,free);index=np.full(len(ids),-1,'i4');index[loc]=np.arange(len(loc))
    law=soft_response(X,Y,T,mu[role[active]-15],bulk[role[active]-15]);ff,K=law(Y,True)
    internal=np.zeros_like(X);np.add.at(internal,T.ravel(),-ff.reshape(-1,3))
    comp=np.searchsorted(active,targets);target=T[comp]
    eps=1e-6;Jparts=[];P=[];G0=[];mode_cells=[];dof_edges=[]
    for row,t0 in zip(targets,target):
        nodemass=mass[ids[t0]];g=jac_volume(X,Y,t0);ng=g/nodemass[:,None];a0=float(dt*dt*np.sum(ng*internal[t0]))
        area=abs(np.linalg.det((X[t0[1:]]-X[t0[0]])))
        E=lambda p:np.linalg.det(p[t0[1:]]-p[t0[0]])/area
        observed=E(Z)-E(Y)
        if a0*observed<=0 or abs(a0)<.02:continue
        parts=np.argwhere(np.isin(T,t0));grad=np.zeros((len(ids),3))
        for i,k in parts:
            ii=int(np.flatnonzero(t0==T[i,k])[0])
            np.add.at(grad,T[i],-np.einsum('a,jab->jb',ng[ii],K[i,k])*dt**2)
        # The determinant gradient is affine in each individual corner coordinate.
        # A central difference is exact in real arithmetic for this derivative.
        local=Y[t0].copy()
        for i,node in enumerate(t0):
            for axis in range(3):
                a=local.copy();b=local.copy();a[i,axis]+=eps;b[i,axis]-=eps
                dg=(jac_volume(X[t0],a,np.arange(4))-jac_volume(X[t0],b,np.arange(4)))/(2*eps)
                grad[node,axis]+=dt**2*np.sum(dg*internal[t0]/nodemass[:,None])
        available=np.flatnonzero(np.linalg.norm(grad[loc],axis=1)>0)
        if not len(available):continue
        P.append(a0);G0.append(observed);Jparts.append((grad[loc]*1e-6).ravel());mode_cells.append(int(row));dof_edges.append(available)
    if not Jparts:raise ValueError('No mechanically addressable modes')
    # Partition disconnected free-node groups. Include every cross-cell derivative in the graph.
    active_indices=index[T];membership=np.nonzero(active_indices>=0)
    graph=coo_matrix((np.ones(len(membership[0])),(membership[0],active_indices[membership])),shape=(len(T)+len(dof_edges),len(free))).tocsr()
    er=[];ec=[]
    for i,vs in enumerate(dof_edges):er.extend([len(T)+i]*len(vs));ec.extend(vs)
    graph+=coo_matrix((np.ones(len(er)),(er,ec)),shape=graph.shape).tocsr()
    _,label=connected_components(graph.T@graph,directed=False)
    E=lambda P,T:P[T[:,1:]]-P[T[:,:1]]
    inv=np.linalg.inv(E(X,T)); grads=np.concatenate((-inv.sum(2,keepdims=True),inv),axis=2).transpose(0,2,1)
    candidates=Y.copy();logs=[]
    for component in np.unique(label):
        fidx=np.flatnonzero(label==component);sel=[i for i,vs in enumerate(dof_edges) if label[vs[0]]==component]
        if not sel:continue
        f=loc[fidx];cids=np.flatnonzero(np.isin(T,f).any(1));allc=np.union1d(cids,np.flatnonzero(np.isin(T,target[np.searchsorted(targets,np.asarray(mode_cells)[sel])]).any(1)))
        pts=np.unique(T[allc]);tc=np.searchsorted(pts,T[allc]);lf=np.searchsorted(pts,f);fixedX=X[pts];base=Y[pts];past=Z[pts];dd=past-base
        crit=np.asarray(mode_cells)[sel];ct=np.searchsorted(pts,T[np.searchsorted(active,crit)])
        cphase=np.searchsorted(allc,cids);reference_volume=np.linalg.det(E(fixedX,tc))/6
        penalty_idx=np.arange(len(fidx)*3).reshape(-1,3)
        newindex=np.full(len(ids),-1,'i4');newindex[f]=np.arange(len(f))
        free_slots=newindex[T[cids]];ii,kk=np.nonzero(free_slots>=0);node=free_slots[ii,kk]
        # Penalize deformation-gradient increments on thin nearby cells to preserve material shape.
        ax=np.arange(3);rr=(9*ii[:,None,None]+3*ax[None,:,None]+ax[None,None,:]);cc=np.broadcast_to(3*node[:,None,None]+ax[None,:,None],rr.shape)
        val=np.broadcast_to(grads[cids[ii],kk,None,:]*1.e-6,rr.shape)
        L=coo_matrix((val.ravel(),(rr.ravel(),cc.ravel())),shape=(9*len(cids),3*len(f))).tocsr()
        S=np.asarray(Jparts)[np.ix_(sel,(3*fidx[:,None]+ax).ravel())];old=np.asarray(P)[sel]
        weights=1/np.maximum(abs(old),.25);s=coo_matrix(S*weights[:,None]).tocsr()
        trial_law=soft_response(fixedX,base,tc,mu[role[active[allc]]-15],bulk[role[active[allc]]-15])
        iv=np.linalg.inv(E(fixedX,tc));qref=geom(past,tc)[1]
        sv0=[np.linalg.svd(iv@E(w,tc),compute_uv=False) for w in (base,past)]
        def modes(pt):
            force,_=trial_law(pt);ff=np.zeros_like(pt);np.add.at(ff,tc.ravel(),-force.reshape(-1,3))
            return dt**2*np.array([np.sum(jac_volume(fixedX,pt,t0)*ff[t0]/mass[ids[pts[t0]],None])for t0 in ct])
        if not np.allclose(old,modes(base),rtol=2e-6,atol=1e-7):raise ValueError('Incomplete force incidence')
        best=(np.linalg.norm(old*weights),base.copy(),old,None)
        for stiffness in (.25,1.,4.):
            A=vstack([s,stiffness*L,eye(s.shape[1])*.02]).tocsr();b=np.r_[-old*weights,np.zeros(A.shape[0]-len(old))]
            step=lsmr(A,b,atol=1e-9,btol=1e-9,maxiter=320)[0].reshape(-1,3)
            scale=min(1.,200./max(np.linalg.norm(step,axis=1).max(),1e-12))
            for back in range(12):
                pt=base.copy();pt[lf]=(base[lf]+step*scale*2.**(-back)*1e-6).astype('f4').astype(float)
                if not np.any(pt[lf]!=base[lf]):break
                ok=True
                for k,pos in enumerate((pt,pt+dd)):
                    # Unchanged cells can sit on an existing bound and retain their measured margin.
                    J,q=geom(pos,tc);J/=reference_volume;sv=np.linalg.svd(iv[cphase]@E(pos,tc[cphase]),compute_uv=False)
                    min_j=.4 if not k else .35
                    if np.any((J[cphase]<min_j)|(J[cphase]>2.5)|(q[cphase]<(np.minimum(qref[cphase],1e-5) if k else 1e-5))) or np.any(sv[:,0]>1.0048*np.maximum(1.1,sv0[k][cphase,0])) or np.any(sv[:,-1]<np.minimum(.9,sv0[k][cphase,-1])/1.0048):
                        ok=False;break
                if not ok:continue
                curr=modes(pt);score=np.linalg.norm(weights*curr)
                if score<best[0] and np.all(abs(curr)<=abs(old)+.01):best=(score,pt,curr,[stiffness,scale,back])
                if best[0]<.1*np.linalg.norm(weights*old):break
        p=best[1];candidates[f]=p[lf]
        logs.append(dict(source_rows=crit.tolist(),initial_modal_J_change=old.tolist(),fitted_modal_J_change=best[2].tolist(),free=len(f),weights=best[3]))
    out.mkdir(exist_ok=True);used=np.flatnonzero((candidates!=Y).any(1));changes={int(i):p for i,p in zip(ids[used],candidates[used])}
    if not changes:raise ValueError('No admissible connected mode corrections')
    write_nodes(h/'guarded-fit/current.node',out/'current.node',changes)
    affected=sorted(set(prior['nodes_0_based'])|set(changes))
    original=read(v/'current.node');final=read(out/'current.node');actual=np.flatnonzero((final!=original).any(1))
    if sorted(actual)!=affected:raise ValueError('Unexpected source displacement')
    meta={k:prior[k]for k in ['source_reference','control_loaded','elements','source_mass']}
    meta.update(registered=digest(out/'current.node'),nodes_0_based=affected,changed_nodes=len(affected),added_or_updated_nodes=len(changes),eligible_targets=len(mode_cells),predicted_targets=len(targets),max_added_shift_um=float(np.linalg.norm(candidates[used]-Y[used],axis=1).max()*1e6),components=logs)
    (out/'fit.json').write_text(json.dumps(meta,indent=2)+'\n');print(len(targets),len(mode_cells),len(logs),len(changes),meta['max_added_shift_um'],flush=True)
if __name__=='__main__':build(*(Path(s).resolve() for s in sys.argv[1:]))
