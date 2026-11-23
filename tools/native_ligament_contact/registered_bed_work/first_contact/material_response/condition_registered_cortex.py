"""Condition tetrahedra against the original cortical parent planes after Float32 emission."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,eye,hstack,vstack
from condition_volume_quality import constraints
from fit_carpal_clearance import geom
from audit_registered_skeleton import digest

def run(path,planes,initial,out):
    z=np.load(path);cp=np.load(planes);base=np.load(initial)['x'].astype(float);x=z['x'].astype(float);t=z['t'];role=z['role']
    n=cp['normal'];on=cp['on_plane'];wall=cp['node'];f=z['facets'];frozen=z['locked'].copy();frozen[np.unique(f[z['markers']==15])]=True
    if str(cp['source'])!=digest(path) or str(cp['initial'])!=digest(initial):raise ValueError('Unmatched cortical ancestry')
    def stats(p):
        v,q=geom(p[t]);g=np.einsum('ij,ij->i',n,p[wall]-on)*1e6
        return (int((v<=0).sum()),int(((v<=1e-18)|(q<1e-5)).sum()),int((g<0).sum()),float(np.maximum(0,.10-g).sum()),float(g.min()))
    def empty(rows,cols):return coo_matrix((rows,cols))
    out.mkdir();hist=[];start=stats(x)
    for iteration in range(24):
        gap=np.einsum('ij,ij->i',n,x[wall]-on)*1e6
        d,J,k=constraints(x,t);wanted=.24
        nodes=np.unique(np.r_[wall[gap<wanted],t[d<0].ravel()]);touch=np.isin(t,nodes).any(1);nodes=np.unique(t[touch]);nodes=nodes[~frozen[nodes]]
        key=np.full(len(x),-1,int);key[nodes]=np.arange(len(nodes));cell=np.flatnonzero(np.any(key[t]>=0,axis=1));g=key[t[cell]]
        bad=np.flatnonzero(d[cell]<0);nc=len(cell);nv=3*len(nodes)
        wr=np.flatnonzero(key[wall]>=0);nw=len(wr);wc=3*key[wall[wr],None]+np.arange(3);wmat=coo_matrix((-n[wr].ravel(),(np.repeat(np.arange(nw),3),wc.ravel())),shape=(nw,nv)).tocsr()
        # Normals are from the exact cortical source rather than a rounded material triangle.
        scale=np.maximum(abs(J[cell]).sum((1,2))*1e-6,1e-24);ii,jj,kk=np.where(np.broadcast_to((g>=0)[:,:,None],J[cell].shape))
        deriv=coo_matrix((-J[cell][ii,jj,kk]*1e-6/scale[ii],(ii,3*g[ii,jj]+kk)),shape=(nc,nv)).tocsr()
        sizes=(nv,len(bad),nw,nv);kq=coo_matrix((-np.ones(len(bad)),(bad,np.arange(len(bad)))),shape=(nc,len(bad)));I=eye(nv,format='csr')
        A=[hstack((deriv,kq,empty(nc,nw+nv))),hstack((wmat,empty(nw,len(bad)),-eye(nw),empty(nw,nv))),
           hstack((deriv[bad],empty(len(bad),sum(sizes)-nv))),
           hstack((I,empty(nv,len(bad)+nw),-I)),hstack((-I,empty(nv,len(bad)+nw),-I))]
        rhs=[d[cell]/scale,gap[wr]-wanted,.6*(d[cell][bad]+k[cell][bad])/scale[bad],np.zeros(nv),np.zeros(nv)]
        # Linprog variables and cortical slack use micrometres.
        bounds=list(zip(np.maximum(-20,-105-(x[nodes]-base[nodes])*1e6).ravel(),np.minimum(20,105-(x[nodes]-base[nodes])*1e6).ravel()))+[(0,None)]*(sum(sizes)-nv)
        opt=linprog(np.r_[np.zeros(nv),np.full(len(bad),1.),np.full(nw,12.),np.full(nv,.0001)],A_ub=vstack(A).tocsr(),b_ub=np.concatenate(rhs),bounds=bounds,method='highs')
        if not opt.success:break
        delta=opt.x[:nv].reshape(-1,3)*1e-6;best=None;old=stats(x)
        def value(s):return (s[0],s[1]>start[1],s[3]+.0001*s[1],s[1])
        for frac in (1.,.8,.6,.4,.25,.125,.0625,.03125,.015625,.0078125):
            trial=x.copy();trial[nodes]=(x[nodes]+frac*delta).astype('f4');s=stats(trial)
            if value(s)<value(old) and (best is None or value(s)<value(best[0])):best=s,trial
        if best is None:break
        s,x=best;hist.append(s);print(iteration,s,flush=True)
    if np.any(x[frozen]!=base[frozen]):raise ValueError('Protected skin position changed')
    result={k:z[k] for k in z.files};result['x']=x.astype('f4');np.savez_compressed(out/'domain.npz',**result)
    (out/'result.json').write_text(json.dumps(dict(source=digest(path),ancestry=digest(planes),start=start,final=stats(x),history=hist,largest_coordinate_change_m=float(np.max(abs(x-base)))),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
