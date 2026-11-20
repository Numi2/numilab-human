"""Fit a source-space displacement on the same two typed, native mesh captures."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

here=Path(__file__).resolve().parent
w=here.parent
sys.path[:0]=[str(w/'resting-clearance-and-atrium-006'),
              str(w/'resting-surface-clearance-001/src')]
from register_tissue_bellies import clear,exact,exact_skin_pair,install_for_float32,quality,volume

def pair_rows(skin, points, faces, jacobian, pairs, vertex_group):
    """Build the existing inward-facing separator choice on each exact witness."""
    ids, js=np.array(sorted(pairs),dtype=int).T
    S=skin[0][skin[1][ids]].astype(float)*1000.
    target=faces[js]
    T=points[target].astype(float)*1000.
    origin=S[:,:1]
    def normals(tris):
        v=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0])
        return v/np.linalg.norm(v,axis=1)[:,None]
    ns,nt=normals(S),normals(T)
    a=np.roll(S,-1,axis=1)-S
    b=np.roll(T,-1,axis=1)-T
    edges=np.cross(a[:,:,None],b[:,None,:]).reshape(-1,9,3)
    directions=np.concatenate((ns[:,None],nt[:,None],-nt[:,None],edges,-edges),axis=1)
    norm=np.linalg.norm(directions,axis=2)
    directions/=np.maximum(norm[:,:,None],1.e-18)
    alignment=np.einsum('pkj,pj->pk',directions,ns)
    support=np.einsum('pvj,pkj->pvk',S-origin,directions).min(axis=1)
    projected=np.einsum('pvj,pkj->pvk',T-origin,directions)-support[:,None,:]
    peak=np.max(projected,axis=1)
    cost=np.where((norm>1.e-10)&(alignment>.65),peak/(alignment**2+.001),np.inf)
    idx=cost.argmin(axis=1)
    normal=directions[np.arange(len(idx)),idx]
    offset=projected[np.arange(len(idx)),:,idx].ravel()+.03
    coefficients=np.einsum('pqi,pqij->pqj',normal[:,None,:].repeat(3,axis=1),
                           jacobian[target]).reshape(-1,3)
    return coefficients,offset,vertex_group[target].ravel()

def smooth_edges(points, faces, vertex_group, columns):
    # Decode a unique source seam only once. Weight across the original
    # edge lengths to suppress steep deformation in highly refined triangles.
    edge_pairs=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    unique=np.unique(np.sort(vertex_group[edge_pairs],axis=1),axis=0)
    unique=unique[unique[:,0]!=unique[:,1]]
    choice=np.any(columns[unique]>=0,axis=1)
    chosen=unique[choice]
    edges=(points[chosen[:,1]]-points[chosen[:,0]])*1000.
    f=np.clip(1./np.linalg.norm(edges,axis=1),.15,2.)
    rows=np.repeat(np.arange(len(chosen)),2)
    target=columns[chosen].ravel()
    valid=target>=0
    return sparse.coo_matrix((np.c_[f,-f].ravel()[valid],(rows[valid],target[valid])),
                             shape=(len(chosen),np.count_nonzero(columns>=0))).tocsr()

def vol_gradient(points,faces,group):
    p=(points-points.mean(0))*1000.
    p0,p1,p2=np.moveaxis(p[faces],1,0)
    g=np.zeros_like(p)
    for i,c in enumerate((np.cross(p1,p2),np.cross(p2,p0),np.cross(p0,p1))):
        np.add.at(g,faces[:,i],c/6.)
    gv=np.zeros((group.max()+1,3))
    np.add.at(gv,group,g)
    return gv

def fit(A, b, E, g, prior, volume_penalty):
    def fun(u_flat):
        u=u_flat.reshape(-1,3)
        H=np.maximum(A@u_flat+b,0.)
        Eu=E@u
        v=np.sum(g*u)
        grad=A.T@H + (.20*u + 3.0*(E.T@Eu) + volume_penalty*g*v).ravel()
        y=.5*(H@H + .20*(u_flat@u_flat) + 3.0*np.sum(Eu*Eu) + volume_penalty*v*v)
        return y,grad
    r=minimize(fun,prior.ravel(),jac=True,method='L-BFGS-B',bounds=[(-2.5,2.5)]*A.shape[1],
               options=dict(maxiter=130,ftol=2.e-10,gtol=1.e-7))
    assert np.isfinite(r.x).all()
    return r.x.reshape(-1,3)

def trial_metrics(views, offset, pts, faces, force_audit=False):
    q0,vol=quality(pts,(pts+offset*.001).astype(np.float32),faces)
    if q0<.60 or vol>.005:
        return None
    output=[]
    for skin,index,floats,p,j,before,witness_ring in views:
        world=(p+.001*np.einsum('vij,vj->vi',j,offset)).astype(np.float32)
        q,pose_volume=quality(p,world,faces)
        if q<.60 or pose_volume>.005:
            return None
        pairs=exact_skin_pair(index,floats,world,faces)
        new=pairs-before
        beyond={pair for pair in new if not witness_ring.intersection(skin[1][pair[0]])}
        output.append(dict(before=len(before),after=len(pairs),new=len(new),
            outside_ring=len(beyond),area=q,volume_delta=pose_volume,pairs=pairs))
    return dict(source_area=q0,source_volume_delta=vol,poses=output)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('ids',type=int,nargs='+')
    args=parser.parse_args()
    data=np.load(here/'paired-geometry.npz')
    obs=json.loads((here/'registration-observation.json').read_text())
    result={}
    chosen={}
    install_for_float32()
    poses=[]
    for i in range(2):
        skin=data[f'pose{i}_skin_v'],data[f'pose{i}_skin_f']
        r=clear._exact_surface_records(*skin)
        index=exact._prepare_surface_aabb(r)
        floats=clear._float32_skin_aabb_index(r)
        poses.append((skin,index,floats))

    for sid in args.ids:
        if max(p['banks'][str(sid)]['max_registration_error_m'] for p in obs['poses'])>.00012:
            print('unmatched',sid,flush=True)
            continue
        pts=data[f'b{sid}_v']
        faces=data[f'b{sid}_f']
        views=[]
        targets=[]
        for i,(skin,index,floats) in enumerate(poses):
            p=data[f'pose{i}_b{sid}_v']
            j=data[f'pose{i}_b{sid}_j']
            baseline=data[f'pose{i}_b{sid}_hits']
            hits=set(map(tuple,baseline))
            ring=set(skin[1][baseline[:,0]].ravel())
            views.append((skin,index,floats,p,j,hits,ring))
            targets.extend(baseline[:,1].tolist())
        if not targets:
            continue
        unique,groups=np.unique(pts,axis=0,return_inverse=True)
        if groups.ndim!=1:
            groups=groups.ravel()
        extent=unique.astype(float)
        extent-=extent.mean(0)
        basis=np.linalg.eigh(extent.T@extent)[1][:,-1]
        along=extent@basis
        lo,hi=np.quantile(along,(.06,.94))
        fixed= (along<=lo)|(along>=hi)
        mesh_groups=groups[faces]
        active=np.zeros(len(unique),dtype=bool)
        active[mesh_groups[np.unique(targets)].ravel()]=True
        for _ in range(4):
            active[mesh_groups[np.any(active[mesh_groups],axis=1)].ravel()]=True
        active[fixed]=False
        if not np.any(active):
            continue
        col=np.full(len(active),-1,dtype=int)
        col[active]=np.arange(active.sum())
        regularizer=smooth_edges(unique,faces,groups,col)
        gv=vol_gradient(pts.astype(float),faces,groups)[active]
        sv=abs(volume(pts.astype(float)*1000.,faces))
        gv/=max(sv,1.e-9)
        solution=np.zeros((active.sum(),3))
        best_score=None
        best=None
        raw=[]
        witness_pools=[set(item[5]) for item in views]
        for iteration in range(3):
            pieces=[pair_rows(v[0],v[3],faces,v[4],pairs,groups)
                    for v,pairs in zip(views,witness_pools,strict=True) if pairs]
            coeff,rhs,vertex=map(np.concatenate,zip(*pieces))
            k=col[vertex]
            keep=k>=0
            rhs=rhs[keep]
            j=(k[keep,None]*3+np.arange(3)).ravel()
            i=np.repeat(np.arange(len(rhs)),3)
            A=sparse.coo_matrix((coeff[keep].ravel(),(i,j)),shape=(len(rhs),3*active.sum())).tocsr()
            if not len(rhs):
                break
            solution=fit(A,rhs,regularizer,gv,solution,1.e7)
            fresh=0
            for scale in (1.,.5,.25,.125):
                group_delta=np.zeros((len(unique),3))
                group_delta[active]=solution*scale
                move=group_delta[groups]
                m=trial_metrics(views,move,pts,faces)
                if m is None:
                    continue
                compact=[{k:v for k,v in p.items() if k!='pairs'} for p in m['poses']]
                row=dict(iteration=iteration,scale=scale,maximum_shift_mm=float(np.linalg.norm(move,axis=1).max()),
                         source_area=m['source_area'],source_volume_delta=m['source_volume_delta'],poses=compact)
                raw.append(row)
                for pool,entry in zip(witness_pools,m['poses'],strict=True):
                    added=entry['pairs']-pool
                    pool.update(added)
                    fresh+=len(added)
                if any(p['after']>p['before'] or p['outside_ring'] for p in compact):
                    continue
                improvement=sum(p['before']-p['after'] for p in compact)
                score=(improvement,-row['maximum_shift_mm'])
                if improvement>0 and (best_score is None or score>best_score):
                    best_score=score
                    best=(move,row)
            print(json.dumps(dict(bank=sid,iteration=iteration,active_vertices=int(active.sum()),
                                  active_constraints=len(rhs),candidate_count=len(raw),fresh=fresh,
                                  best=best[1] if best is not None else None)),flush=True)
            if not fresh:
                break
        result[sid]=dict(active_vertices=int(active.sum()),pinned_vertices=int(fixed.sum()),
                         best=best[1] if best is not None else None,trials=raw)
        if best is not None:
            chosen[f'bank{sid}']=best[0]*.001
    (here/f'{args.name}.json').write_text(json.dumps(result,indent=2)+'\n')
    if chosen:
        np.savez_compressed(here/f'{args.name}.npz',**chosen)

if __name__=='__main__':
    main()
