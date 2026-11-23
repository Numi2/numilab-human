"""Choose representable neighbor positions with exact linear determinant constraints along each axis."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix,eye,hstack,vstack
from scipy.sparse.csgraph import connected_components
from scipy.optimize import Bounds,milp,LinearConstraint
from certify_transported_material import measure
from compile_connected_joint import digest
def run(h,field,out):
    root=h/'closed-native/native';x=np.loadtxt(root/'mesh.node',skiprows=1,usecols=(1,2,3))
    data=np.loadtxt(root/'mesh.ele',skiprows=1,dtype='i4');t=data[:,1:5]-1;d=np.load(field/'material-transport.npz');y=x.copy();y[d['nodes']]=d['positions'];orig=y.copy()
    out.mkdir();protected=np.zeros(len(x),bool);protected[np.unique(np.load(h/'cortical-registration/registered-cortical-facets.npz')['cell_side_material_bank_nodes'][:,-3:])]=True
    N=len(t);n=len(x);inc=coo_matrix((np.ones(N*4,'i1'),(t.ravel(),np.repeat(np.arange(N,dtype='i4'),4))),shape=(n,N)).tocsr()
    reference,_=measure(x,t)
    def bad(det,l,ref):return (det<=6e-18)|(det<l*1e-5)|(det<.4*ref)|(det>2.5*ref)
    log=[]
    for it in range(4):
        det,L=measure(y,t);failed=np.flatnonzero(bad(det,L,reference))
        if not len(failed):break
        active=np.unique(t[failed]);active=active[~protected[active]];st=inc[active]
        _,labels=connected_components(st@st.T,directed=False)
        choices=[]
        for component in np.unique(labels):
            nodes=active[labels==component];rows=np.unique(inc[nodes].indices);tet=t[rows];mask=np.isin(tet,nodes);u=len(nodes)
            target=failed[np.isin(failed,rows)];was=len(target)
            corners=y[tet];e=corners[:,1:]-corners[:,:1];g=np.stack([np.cross(e[:,1],e[:,2]),np.cross(e[:,2],e[:,0]),np.cross(e[:,0],e[:,1])],axis=1);G=np.concatenate([-g.sum(1)[:,None],g],axis=1)
            ir,ic=np.where(mask);jj=np.searchsorted(nodes,tet[ir,ic])
            # For a fixed axis, determinant is exactly affine in every selected
            # coordinate, including simultaneous changes to multiple tetrahedral corners.
            best=was;bestp=None;axis_used=None
            for budget in (12,80,600,4000):
                for ax in np.argsort(-np.max(abs(G[np.isin(rows,target),:, :]),axis=(0,1))):
                    step=np.abs(np.spacing(y[nodes,ax].astype('f4')).astype(float));C=coo_matrix((G[ir,ic,ax]*step[jj],(ir,jj)),shape=(len(rows),u)).tocsr()
                    k0=np.maximum(np.maximum(6.00001e-18,1.00001e-5*L[rows]),.400001*reference[rows])
                    high=2.49999*reference[rows]
                    margin=np.where(np.isin(rows,target),.001,0)
                    rhs=np.r_[k0*(1+margin)-det[rows],det[rows]-high/(1+margin)]
                    CC=vstack([C,-C],format='csr')
                    scale=np.maximum(np.asarray(abs(CC).max(1).toarray())[:,0],1e-34);idx=np.flatnonzero(rhs>-(budget+1)*np.asarray(abs(CC).sum(1))[:,0])
                    if not len(idx):continue
                    A=CC[idx].multiply(1/scale[idx,None]);z=csr_matrix(A.shape);I=eye(u,format='csr')
                    full=vstack([hstack([A,z]),hstack([-I,I]),hstack([I,I])],format='csr');lo=np.r_[rhs[idx]/scale[idx],np.zeros(2*u)];ub=np.full(len(lo),np.inf)
                    res=milp(np.r_[np.zeros(u),np.ones(u)],integrality=np.r_[np.ones(u),np.zeros(u)],bounds=Bounds(np.r_[np.full(u,-budget),np.zeros(u)],np.full(2*u,budget)),constraints=LinearConstraint(full,lo,ub),options={'time_limit':2.,'mip_rel_gap':.05})
                    if res.x is None:continue
                    candidate=y[nodes].copy();candidate[:,ax]=(candidate[:,ax]+np.rint(res.x[:u])*step).astype('f4')
                    table=np.unique(tet);ti=np.searchsorted(table,tet);a=y[table].copy();a[np.searchsorted(table,nodes)]=candidate
                    v,q=measure(a,ti);miss=bad(v,q,reference[rows]);new=int(miss.sum())
                    if new<best and not np.any(miss&~np.isin(rows,target)):
                        best=new;bestp=candidate;axis_used=int(ax)
                    if best==0:break
                if best==0:break
            if bestp is not None:y[nodes]=bestp
            choices.append(dict(component=int(component),before=was,after=best,axis=axis_used))
        log.append(dict(before=len(failed),regions=choices));print(it,len(failed),'->',sum(i['after'] for i in choices),flush=True)
        if sum(i['after'] for i in choices)==len(failed):break
    det,L=measure(y,t);id=np.flatnonzero(np.any(x!=y,axis=1));fix=np.linalg.norm(y-orig,axis=1)
    np.savez_compressed(out/'material-transport.npz',nodes=id,positions=y[id].astype('f4'))
    res=dict(input=digest(field/'material-transport.npz'),history=log,inverted=int((det<=0).sum()),volume=int((det<=6e-18).sum()),shape=int((det<L*1e-5).sum()),too_compressed=int((det/reference<.4).sum()),too_expanded=int((det/reference>2.5).sum()),changed_rounding_nodes=int((fix>0).sum()),maximum_rounding_motion_m=float(fix.max()))
    (out/'result.json').write_text(json.dumps(res,indent=2)+'\n');print({k:v for k,v in res.items() if k!='history'},flush=True)
if __name__=='__main__':run(*(Path(x) for x in sys.argv[1:]))
