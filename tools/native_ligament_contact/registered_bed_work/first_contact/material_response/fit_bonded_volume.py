"""Jointly reduce excess initial strain in connected bonded tissue, retaining measured boundary sites."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,hstack,vstack,eye
from scipy.sparse.csgraph import connected_components
from scipy.optimize import linprog
from compile_connected_joint import digest
from certify_transported_material import measure

def run(job,volume,out):
    out.mkdir();read=lambda f:np.loadtxt(f,skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    x,y=read(volume/'mesh.node'),read(volume/'current.node');start=y.copy()
    t=np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    invert=lambda q:np.linalg.inv(x[q[:,1:]]-x[q[:,:1]])
    def features(q,iv):
        e=y[q[:,1:]]-y[q[:,:1]];F=iv@e;U,s,V=np.linalg.svd(F);return F,U,s,V,np.linalg.det(F)
    J=np.concatenate([features(q,invert(q))[-1] for q in np.array_split(t,64)])
    bad=np.flatnonzero((J>1.65)|(J<.65))
    b=np.load(job/'registered-boundary/registered-boundary.npz')
    a=json.loads((job.parent/'ligament-equilibrium-070/scene-registration/registration.json').read_bytes())
    protected=np.zeros(len(x),bool)
    protected[np.r_[b['O_node_joint'],b['D_node_joint'],np.load(job/'verified-connected-material/contact/material-contact.npz')['samples'][:,0],a['first_contact_new'],[v[2] for v in a['anchors']]].astype(int)]=True
    focus=np.unique(t[bad]);focus=focus[~protected[focus]]
    # One ring provides coupled corrections on both sides of the shared skin/fat interface.
    ring=np.unique(t[np.isin(t,focus).any(1)]);free=ring[~protected[ring]]
    rows=np.flatnonzero(np.isin(t,free).any(1));q=t[rows];local=np.full(len(x),-1,'i4');local[free]=np.arange(len(free))
    ii,jj=np.nonzero(local[q]>=0);inc=coo_matrix((np.ones(len(ii),bool),(ii,local[q[ii,jj]])),shape=(len(q),len(free))).tocsr()
    n,groups=connected_components(inc.T@inc,directed=False)
    history=[];micron=1e-6
    def summary(coords):
        F=invert(q)@(coords[q[:,1:]]-coords[q[:,:1]]);s=np.linalg.svd(F,compute_uv=False);j=np.linalg.det(F)
        return dict(severe=int(np.sum((s[:,0]>2)|(s[:,2]<.5))),low=int(np.sum(j<.65)),high=int(np.sum(j>1.65)),J_range=[float(min(j)),float(max(j))])
    before=summary(y)
    for gi in range(n):
        vs=free[groups==gi];ix=np.nonzero(inc[:,groups==gi].getnnz(axis=1))[0];g=q[ix];orig=J[rows[ix]]
        if not np.any((orig>1.65)|(orig<.65)):continue
        iv=invert(g);grad=np.concatenate([-iv.sum(2)[:,:,None],iv],2)
        loc=np.full(len(x),-1,'i4');loc[vs]=np.arange(len(vs));r,k=np.nonzero(loc[g]>=0)
        ndof=3*len(vs);N=len(g);ci=loc[g[r,k]][:,None]*3+np.arange(3)
        def matrix(A):return coo_matrix((A[r,k].ravel(),(np.repeat(r,3),ci.ravel())),shape=(N,ndof)).tocsr()*micron
        # Rows are unitless deformation derivatives per micrometre of nodal displacement.
        def jac(B):return matrix(np.einsum('ndk,ndj->nkj',grad,B))
        def hinge(j,s):
            return np.r_[np.maximum(j-1.65,0)+np.maximum(.65-j,0),
                         .25*(np.maximum(s[:,0]-1.95,0)+np.maximum(.52-s[:,2],0))]
        trust=24.;steps=0;traces=[]
        for it in range(9):
            F,U,s,V,j=features(g,iv);before_merit=hinge(j,s).sum()
            B=jac(j[:,None,None]*np.linalg.inv(F).transpose(0,2,1))
            H=jac(U[:,:,0,None]*V[:,None,0,:]);L=jac(U[:,:,2,None]*V[:,None,2,:])
            Q=vstack([B,-B,H,-L],format='csr');bound=np.r_[1.65-j,j-.65,1.95-s[:,0],s[:,2]-.52]
            active=(bound<.15)
            Q=Q[active];bb=bound[active];ns=len(bb)
            cost=np.r_[np.full(2*ndof,2e-5),np.tile([1.,1.,.25,.25],(N,1)).T.ravel()[active]]
            def bare(A):return hstack([A,-A,csr_matrix((A.shape[0],ns))],format='csr')
            M=hstack([Q,-Q,-eye(ns)],format='csr')
            hardlo=np.maximum(.400001,np.minimum(.62,orig)-2e-6)
            hardhi=np.minimum(2.49999,np.maximum(1.70,orig)+2e-6)
            # Linearize the actual quality floor, including the derivative of the longest edge.
            shape,scale=measure(y,g);quality=shape/scale
            near=np.flatnonzero(quality<8e-5)
            extra=[M,bare(B),bare(-B)];rhs=[bb,hardhi-j,j-hardlo]
            if len(near):
                locedge=np.array([[i,k] for i in range(4) for k in range(i)])
                ee=y[g[:,locedge[:,0]]]-y[g[:,locedge[:,1]]]
                eidx=np.argmax((ee*ee).sum(2),axis=1);el=ee[np.arange(N),eidx];p0,p1=locedge[eidx].T
                dlog=np.zeros((N,4,3));dlog[np.arange(N),p0]=3*el/(el*el).sum(1)[:,None];dlog[np.arange(N),p1]=-dlog[np.arange(N),p0]
                D=matrix(dlog)-B.multiply((1/j)[:,None])
                extra.append(bare(D.tocsr()[near]));rhs.append(np.log(quality[near]/1.000001e-5))
            res=linprog(cost,A_ub=vstack(extra),b_ub=np.concatenate(rhs),bounds=[(0,trust)]*(2*ndof)+[(0,None)]*ns,method='highs')
            if not res.success:break
            delta=(res.x[:ndof]-res.x[ndof:2*ndof]).reshape(-1,3)*micron;original=y[vs].copy();accepted=False
            for back in range(13):
                y[vs]=(original+delta*2**(-back)).astype('f4')
                vol,scale=measure(y,g);f2,u2,s2,v2,j2=features(g,iv)
                if np.min(vol)<=6e-18 or np.any(vol<1e-5*scale) or np.any(j2<hardlo-1e-7) or np.any(j2>hardhi+1e-7):continue
                if np.sum((s2[:,0]>2)|(s2[:,2]<.5))>np.sum((s[:,0]>2)|(s[:,2]<.5)):continue
                improvement=before_merit-hinge(j2,s2).sum()
                if improvement>1e-7:
                    traces.append([float(improvement),back]);accepted=True;steps+=1;break
            if not accepted:
                y[vs]=original;trust*=.35
                if trust<1.:break
        history.append(dict(nodes=len(vs),cells=N,updates=steps,trials=traces))
    after=summary(y);changed=np.flatnonzero(np.any(start!=y,axis=1))
    if np.any(protected[changed]):raise ValueError('Boundary changed')
    which=set(changed.tolist());np.savez_compressed(out/'candidate.npz',nodes=changed,positions=y[changed].astype('f4'))
    with (volume/'current.node').open() as f,(out/'current.node').open('x') as o:
        o.write(next(f))
        for i,line in enumerate(f):o.write(str(i+1)+' '+' '.join(format(v,'.17g') for v in y[i])+'\n' if i in which else line)
    record=dict(source=digest(volume/'current.node'),material=digest(volume/'mesh.node'),elements=digest(volume/'mesh.ele'),result=digest(out/'current.node'),initial=before,after=after,connected_components=n,patch_cells=len(q),changed=len(changed),largest_change_m=float(np.linalg.norm(y-start,axis=1).max()),history=history)
    (out/'result.json').write_text(json.dumps(record,indent=2)+'\n');print(before,after,len(changed),flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
