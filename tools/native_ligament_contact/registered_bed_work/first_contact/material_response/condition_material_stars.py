"""Solve coupled, overlapping material stars for explicit tetrahedron shape margins."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,hstack,vstack,eye
from condition_volume_quality import constraints
from fit_carpal_clearance import geom
from audit_registered_skeleton import digest

def run(source,reference,plc,witness,out,*,allow_surface_slide=False):
    z=np.load(source);x=z['x'].astype(float);base=np.load(reference)['x'].astype(float);t=z['t'];f=z['facets'];mk=z['markers']
    fixed=z['locked'].copy();fixed[np.unique(f[np.isin(mk,(15,16,115,116))])]=True;fixed[np.load(witness)]=True
    interface=f[mk==1516];sliding=np.zeros(len(x),bool);sliding[np.unique(interface)]=True
    normals=np.cross(x[interface[:,1]]-x[interface[:,0]],x[interface[:,2]]-x[interface[:,0]])
    normals/=np.linalg.norm(normals,axis=1)[:,None];anchor=x.copy()
    retained=set(v.tobytes() for v in np.load(plc)['vertex_world_m'].astype('f4'));free=np.array([v.tobytes() not in retained for v in base.astype('f4')]);out.mkdir()
    def measure(tt):
        v,q=geom(x[tt]);return np.minimum(v/1e-18,q/1e-5)
    def score(q):return int((q<1).sum()),float(np.maximum(1.-q,0).sum())
    old=score(measure(t));hist=[]
    for iteration in range(18):
        values=measure(t);bad=np.flatnonzero(values<1);count=0
        for seed in bad[np.argsort(values[bad])]:
            if measure(t[[seed]])[0]>=1:continue
            for ring in (1,2):
                nodes=t[seed]
                for k in range(ring):nodes=np.unique(t[np.isin(t,nodes).any(1)])
                nodes=nodes[~fixed[nodes]]
                if not len(nodes):continue
                key=np.full(len(x),-1,int);key[nodes]=np.arange(len(nodes));cells=np.flatnonzero(np.any(key[t]>=0,axis=1));tt=t[cells];g=key[tt];nv=3*len(nodes)
                d,J,K=constraints(x,tt);dq=J*1e-6/K[:,None,None];q=d/K+1.;idx=int(np.searchsorted(cells,seed));before=score(measure(tt))
                ii,jj,kk=np.where(np.broadcast_to((g>=0)[:,:,None],J.shape))
                L=coo_matrix((-dq[ii,jj,kk],(ii,3*g[ii,jj]+kk)),shape=(len(cells),nv)).tocsr()
                # Linearize quality, retain immediate neighbors and maximize gain on the seed.
                target=np.minimum(1.0,q);target=np.maximum(np.minimum(q,0.1),target-.003)
                weight=np.zeros(len(cells));weight[idx]=max(0.,1.012-q[idx])
                if weight[idx]==0:break
                mat=hstack((L,coo_matrix(weight[:,None])));rhs=q-target
                span=np.where(free[nodes],3200.,150.)[:,None];offset=(x[nodes]-base[nodes])*1e6
                trust=np.where(free[nodes],350.,35.)[:,None]
                if allow_surface_slide:
                    span=np.where(sliding[nodes,None],1450.,span);trust=np.where(sliding[nodes,None],200.,trust)
                    face,corner=np.where(key[interface]>=0);cc=key[interface[face,corner]]
                    limit=85.;current=np.einsum('ij,ij->i',normals[face],(x-anchor)[interface[face,corner]])*1e6
                    lv=coo_matrix((normals[face].ravel(),(np.repeat(np.arange(len(cc)),3),(3*cc[:,None]+np.arange(3)).ravel())),shape=(len(cc),nv)).tocsr()
                    L=vstack((L,lv,-lv)).tocsr();rhs=np.r_[rhs,limit-current,limit+current];weight=np.r_[weight,np.zeros(2*len(cc))]
                    mat=hstack((L,coo_matrix(weight[:,None])))
                lo=np.maximum(-trust,-span-offset).ravel();hi=np.minimum(trust,span-offset).ravel()
                if (lo>hi).any():continue
                bounds=list(zip(lo,hi));a=linprog(np.r_[np.zeros(nv),-1],A_ub=mat,b_ub=rhs,bounds=bounds+[(0,1.)],method='highs')
                if not a.success or a.x[-1]<1e-6:continue
                I=eye(nv,format='csr');A=vstack((hstack((L,coo_matrix((L.shape[0],nv)))),hstack((I,-I)),hstack((-I,-I))))
                b=linprog(np.r_[np.zeros(nv),np.ones(nv)],A_ub=A,b_ub=np.r_[rhs-.995*a.x[-1]*weight,np.zeros(nv*2)],bounds=bounds+[(0,None)]*nv,method='highs')
                step=(b.x if b.success else a.x)[:nv].reshape(-1,3)*1e-6
                initial=x[nodes].copy();best=None
                for frac in (1.,.85,.65,.5,.35,.2,.1,.05,.025,.01):
                    x[nodes]=(initial+frac*step).astype('f4');v=measure(tt);s=score(v)
                    if v.min()>0 and s<before and (best is None or s<best[0]):best=(s,x[nodes].copy())
                x[nodes]=best[1] if best else initial
                if best:count+=1;break
        sc=score(measure(t));hist.append((iteration,count,sc));print(hist[-1],flush=True)
        if not count or not sc[0]:break
    res={k:z[k] for k in z.files};res['x']=x.astype('f4');np.savez_compressed(out/'domain.npz',**res)
    (out/'result.json').write_text(json.dumps(dict(source=digest(source),initial=old,end=score(measure(t)),history=hist),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
