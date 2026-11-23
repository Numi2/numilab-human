"""Solve coupled shape constraints with temporary slack at every shared active material star."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.sparse import coo_matrix,vstack,hstack,eye,diags
from scipy.optimize import linprog
from condition_volume_quality import constraints
from condition_loaded_patch import ratios
from compile_connected_joint import digest


def run(source,planes,witness,out):
    z=np.load(source);x=z['x'].astype(float);ref=x.copy();t=z['t']
    fixed=z['locked'].copy()
    fixed[np.unique(z['facets'][z['markers']==15])]=True
    fixed[np.load(witness)]=True
    d=np.load(planes);ids=d['node'];N=d['normal'];origin=d['on_plane']
    gap0=np.einsum('ij,ij->i',N,x[ids]-origin)*1.e6
    initial=ratios(x,t)
    neighborhood=t[initial<1].ravel()
    # A fully joined interior patch usually needs tangential motion in both layers.
    # Keep the cortical wall fixed here. The preceding exact-star step handled wall vertices.
    nodes=np.unique(t[np.isin(t,neighborhood).any(1)])
    fixed[ids]=True
    nodes=nodes[~fixed[nodes]]
    key=np.full(len(x),-1,int);key[nodes]=np.arange(len(nodes))
    group=key[t];cc=np.flatnonzero((group>=0).any(1));g=group[cc];nv=3*len(nodes)
    shape=(len(cc),nv);I=eye(nv,format='csr')
    i,j,k=np.where(np.broadcast_to((g>=0)[:,:,None],(*g.shape,3)))
    out.mkdir()
    trace=[];goal=1.02
    def merit(v):return np.maximum(goal-v,0.).sum()
    for it in range(60):
        value=ratios(x,t[cc]);before=merit(value)
        if (value>=1.).all():break
        e,J,K=constraints(x,t[cc]);q=e/K+1
        A=coo_matrix((-J[i,j,k]*1.e-6/K[i],(i,3*g[i,j]+k)),shape=shape).tocsr()
        best=None
        for trust in (5.,35.,200.):
            # With this trust box, omit constraints that cannot become active even under the full linearized displacement.
            near=(q-trust*np.ravel(abs(A).sum(axis=1)))<1.1
            B=A[near];n=B.shape[0]
            # L1 penalty on displacement in micrometres picks the smallest coordinated deformation.
            penalties=np.r_[np.zeros(nv),np.ones(n),np.full(nv,1.e-5)]
            zero=coo_matrix((nv,n));P=vstack((hstack((B,-eye(n),coo_matrix((n,nv)))),hstack((I,zero,-I)),hstack((-I,zero,-I)))).tocsr()
            off=(x[nodes]-ref[nodes]).ravel()*1.e6
            bounds=list(zip(np.maximum(-trust,-650.-off),np.minimum(trust,650.-off)))+[(0,None)]*(n+nv)
            r=linprog(penalties,A_ub=P,b_ub=np.r_[q[near]-goal,np.zeros(2*nv)],bounds=bounds,method='highs')
            if not r.success:continue
            step=r.x[:nv].reshape(-1,3)*1.e-6
            for f in (1,.8,.55,.3,.15,.07,.02,.005):
                y=x.copy();y[nodes]=(x[nodes]+f*step).astype('f4')
                v=ratios(y,t[cc]);m=merit(v)
                if v.min()>0 and m < before-1e-7 and (best is None or m<best[0]):best=m,y
        if best is None:break
        x=best[1];v=ratios(x,t);trace.append(dict(iteration=it,bad=int((v<1).sum()),minimum=float(v.min()),deficit=float(best[0])))
        print(trace[-1],flush=True)
    v=ratios(x,t);delta=np.linalg.norm(x-ref,axis=1)
    np.savez_compressed(out/'domain.npz',**{k:(x.astype('f4') if k=='x' else z[k]) for k in z.files})
    (out/'result.json').write_text(json.dumps(dict(source=digest(source),output=digest(out/'domain.npz'),planes=digest(planes),witness=digest(witness),after=int((v<1).sum()),before=int((initial<1).sum()),max_change_m=float(delta.max()),trace=trace),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
