"""Condition coupled material tetrahedra with sparse signed-volume constraints in world micrometres."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,hstack,vstack,eye
from scipy.sparse.csgraph import connected_components
from audit_registered_skeleton import digest
from fit_carpal_clearance import geom


EDGE=np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])


def constraints(x,t):
    p=x[t];a,b,c=(p[:,i]-p[:,0] for i in (1,2,3))
    g=np.stack((np.cross(b,c),np.cross(c,a),np.cross(a,b)),axis=1)
    det=np.einsum('ij,ij->i',g[:,0],a)
    J=np.concatenate((-g.sum(1)[:,None],g),axis=1)
    edges=p[:,EDGE[:,0]]-p[:,EDGE[:,1]];d=(edges*edges).sum(2);ii=np.arange(len(t));j=d.argmax(1)
    long=np.sqrt(d[ii,j]);needed=np.maximum(6.005e-18,1.003e-5*long**3)
    use=needed>6.005e-18
    adj=(3.009e-5*long*use)[:,None]*edges[ii,j]
    J[ii,EDGE[j,0]]-=adj;J[ii,EDGE[j,1]]+=adj
    return det-needed,J,needed


def score(x,t):
    d,J,k=constraints(x,t);ratio=d/k
    return (int((ratio<=-1).sum()),int((ratio<0).sum()),float(np.maximum(-ratio,0).sum()),float(-ratio.min()))


def run(source,out,reference=None,expand_neighbors=False):
    z=np.load(source);base=z['x'].astype(float);x=base.copy();t=z['t'];f=z['facets'];mk=z['markers'];roles=z['role']
    if reference is not None:
        base=np.load(reference)['x'].astype(float)
        if base.shape!=x.shape:raise ValueError('Mismatched source coordinate domain')
    locked=z['locked'].copy();locked[np.unique(f[mk==15])]=True
    # Positive tetrahedra carry their own oriented, outward boundary faces.
    # Every cortical wall may slide tangentially or move away from the cortex.
    wall=f[np.isin(mk,[115,116])]; pts=x[wall];normal=np.cross(pts[:,1]-pts[:,0],pts[:,2]-pts[:,0])
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    byface={tuple(sorted(row)):i for i,row in enumerate(wall)}
    templates=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    for row in t:
        for face in row[templates]:
            j=byface.get(tuple(sorted(face)))
            if j is not None:
                xyz=base[face];n=np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0])
                if n@normal[j]<0:normal[j]*=-1
    history=[];bestscore=score(x,t);before=bestscore
    out.mkdir()
    for iteration in range(64):
        d,J,k=constraints(x,t);bad=d<0
        if not np.any(bad):break
        nodes=np.unique(t[bad])
        if expand_neighbors:
            touched=np.isin(t,nodes).any(1)
            nodes=np.unique(t[touched])
        nodes=nodes[~locked[nodes]]
        key=np.full(len(x),-1,'i4');key[nodes]=np.arange(len(nodes));g=key[t]
        eligible=(g>=0).any(1);cells=np.flatnonzero(eligible);g=g[eligible];J=J[eligible]
        var=len(nodes)*3; n=len(cells);badrows=np.flatnonzero(bad[cells])
        # Independent contact neighborhoods can accept different fractions of
        # the Newton correction. Two neighborhoods share no incident cell.
        i,j=np.where(g>=0)
        incidence=coo_matrix((np.ones(len(i)),(i,g[i,j])),shape=(len(g),len(nodes))).tocsr()
        nc,groups=connected_components(incidence.T@incidence,directed=False)
        spans=[]
        for label in range(nc):
            selected=np.flatnonzero(groups==label)
            zone=cells[np.unique(incidence[:,selected].nonzero()[0])]
            spans.append((selected,zone,score(x,t[zone])))
        ok=False
        for trust in (2.,8.,30.,80.):
            # Bound the source-coordinate displacement of every changed point.
            lo=np.maximum(-trust,-100.-1e6*(x[nodes]-base[nodes])).ravel()
            hi=np.minimum(trust,100.-1e6*(x[nodes]-base[nodes])).ravel()
            scale=np.maximum(np.abs(J).sum((1,2))*1e-6,1e-24)
            ii,vv,kk=np.nonzero(np.broadcast_to((g>=0)[:,:,None],J.shape))
            L=coo_matrix((-J[ii,vv,kk]*1e-6/scale[ii],(ii,3*g[ii,vv]+kk)),shape=(n,var)).tocsr()
            # Only original failures receive slack. Keep a margin on currently
            # valid neighboring cells and include Float32 rounding in line search.
            err=d[cells]/scale; slack=coo_matrix((-np.ones(len(badrows)),(badrows,np.arange(len(badrows)))),shape=(n,len(badrows)))
            size=var+len(badrows)+var; A=[hstack((L,slack,coo_matrix((n,var))))];rhs=[err]
            # Keep positive signed volume for every failing cell even when the
            # quality constraint needs slack during conditioning.
            safety=(d[cells][badrows]+k[cells][badrows])*.60/scale[badrows]
            A.append(hstack((L[badrows],coo_matrix((len(badrows),size-var)))))
            rhs.append(safety)
            # Baseline cortical support half-spaces at each registered contact vertex.
            ids,corner=np.where(key[wall]>=0);cols=key[wall[ids,corner]]
            if len(ids):
                nr=len(ids);coeff=normal[ids];m=coo_matrix((coeff.ravel(),(np.repeat(np.arange(nr),3),(3*cols[:,None]+np.arange(3)).ravel())),shape=(nr,var))
                margin=-np.einsum('ij,ij->i',coeff,(x-base)[wall[ids,corner]])*1.e6
                A.append(hstack((m,coo_matrix((nr,size-var)))));rhs.append(margin+.007)
            I=eye(var,format='csr');zero=coo_matrix((var,len(badrows)))
            A.extend([hstack((I,zero,-I)),hstack((-I,zero,-I))]);rhs.extend([np.zeros(var),np.zeros(var)])
            cost=np.r_[np.zeros(var),np.ones(len(badrows)),np.full(var,.0001)]
            res=linprog(cost,A_ub=vstack(A).tocsr(),b_ub=np.concatenate(rhs),
                        bounds=list(zip(lo,hi))+[(0,None)]*(size-var),method='highs')
            if not res.success:continue
            delta=res.x[:var].reshape(-1,3)*1e-6
            trial=x.copy();accepted=0
            for selected,zone,initial in spans:
                ids=nodes[selected];best=None
                for frac in (1.,.8,.6,.4,.25,.125,.0625,.03125,.015625,.0078125,.00390625,.001953125,.0009765625):
                    trial[ids]=(x[ids]+frac*delta[selected]).astype('f4')
                    metric=score(trial,t[zone])
                    if metric<initial and (best is None or metric<best[0]):
                        best=metric,trial[ids].copy()
                if best is not None:
                    trial[ids]=best[1];accepted+=1
                else:trial[ids]=x[ids]
            if accepted:
                sc=score(trial,t)
                if sc>=bestscore:raise ValueError('Local descent lost global feasibility')
                bestscore,x=sc,trial;ok=True;break
        if not ok:break
        history.append(dict(iteration=iteration,inverted=bestscore[0],bad=bestscore[1],deficit=bestscore[2],trust_um=trust))
        print(iteration,bestscore,flush=True)
    result={name:z[name] for name in z.files};result['x']=x.astype('f4');np.savez_compressed(out/'domain.npz',**result)
    v,q=geom(x[t]);record=dict(source=digest(source),initial=before,iterations=history,bad_cells=int(((v<=1e-18)|(q<1e-5)).sum()),inverted=int((v<=0).sum()),
                      minimum_shape=float(q.min()),minimum_volume_m3=float(v.min()),maximum_displacement_m=float(np.linalg.norm(x-base,axis=1).max()),changed_exterior_skin_vertices=int(np.any(x[np.unique(f[mk==15])]!=base[np.unique(f[mk==15])],axis=1).sum()))
    (out/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='iterations'}))


if __name__=='__main__':run(*map(Path,sys.argv[1:]))
