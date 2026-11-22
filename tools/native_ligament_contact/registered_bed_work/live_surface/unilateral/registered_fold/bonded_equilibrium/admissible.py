"""Constrained descent in the bonded tissue's material and mass coordinates."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

def material_direction(x,pos,cells,free,force,moment_rows,trust_um=40.):
    R=x[cells[:,1:]]-x[cells[:,:1]]
    Y=pos[cells[:,1:]]-pos[cells[:,:1]]
    def gradients(e):
        rows=np.linalg.inv(e).transpose(0,2,1)
        return np.concatenate((-rows.sum(1,keepdims=True),rows),axis=1)
    G=gradients(R); H=gradients(Y)
    F=np.linalg.solve(R,Y)
    U,s,Vt=np.linalg.svd(F)
    J=np.linalg.det(F); det=np.linalg.det(Y)
    index=np.full(len(x),-1,dtype=int);index[free]=np.arange(len(free))
    ids=index[cells]; good=ids>=0
    ii=np.broadcast_to(np.arange(len(cells))[:,None,None],(*cells.shape,3))
    jj=ids[:,:,None]*3+np.arange(3)
    valid=np.broadcast_to(good[:,:,None],ii.shape)
    cols=jj[valid];rows=ii[valid]; lhs=[]; rhs=[]
    def upper(gradient,bound):
        bound=np.broadcast_to(bound,(len(cells),))
        mat=coo_matrix((gradient[valid]*1e-6,(rows,cols)),shape=(len(cells),3*len(free))).tocsr()
        scale=np.maximum(np.asarray(abs(mat).max(axis=1).toarray()).ravel(),1e-16)
        if np.any(bound< -1e-8):raise ValueError('Inadmissible linearization')
        lhs.append(mat.multiply((1./scale)[:,None]).tocsr())
        rhs.append(np.maximum(0,bound)/scale)
    for k in range(3):
        g=(G@U[:,:,k,None])*Vt[:,k,None,:]
        upper(g,1.5-s[:,k]); upper(-g,s[:,k]-2/3)
    upper(-H,1-np.maximum(.4/J,6e-18/det))
    upper(H,2.5/J-1)
    # The shape floor must hold for every edge, including tied longest edges.
    for i in range(4):
        for j in range(i):
            edge=pos[cells[:,i]]-pos[cells[:,j]]
            length=np.linalg.norm(edge,axis=1)
            q=det/(length**3)
            grad=-H.copy()
            v=3*edge/length[:,None]**2
            grad[:,i]+=v;grad[:,j]-=v
            upper(grad,1-1e-5/q)
    for i in range(3):
        for j in range(3):
            grad=np.zeros_like(G);grad[:,:,j]=G[:,:,i]
            upper(grad,.04);upper(-grad,.04)
    c=-force.ravel()
    result=linprog(c/max(abs(c).max(),1e-30),A_ub=vstack(lhs),b_ub=np.concatenate(rhs),
                   A_eq=moment_rows,b_eq=np.zeros(len(moment_rows)),bounds=(-trust_um,trust_um),method='highs')
    if not result.success:raise RuntimeError(result.message)
    return result.x.reshape(-1,3)*1e-6

def emit_conservative(pos,base,free,W,tol=7.5e-11):
    out=pos.astype('f4')
    # Correct only rounding error. Each update is an actual adjacent Float32.
    # Evaluate all groups at once so an overlapping anatomical group is preserved.
    for axis in range(3):
        v=out[free,axis].copy()
        residual=W@(v.astype(float)-base[free,axis])
        for _ in range(300):
            if np.max(abs(residual))<=tol:break
            up=np.nextafter(v,np.float32(np.inf));dn=np.nextafter(v,np.float32(-np.inf))
            choices=np.r_[up.astype(float)-v,dn.astype(float)-v]
            candidate=residual[:,None]+np.tile(W,(1,2))*choices
            err=np.sum((np.maximum(0,abs(candidate)-tol)/tol)**2,axis=0)
            score=np.sum((np.maximum(0,abs(residual)-tol)/tol)**2)
            j=err.argmin()
            if err[j]>=score:break
            residual=candidate[:,j]
            if j<len(v):v[j]=up[j]
            else:v[j-len(v)]=dn[j-len(v)]
        out[free,axis]=v
    return out.astype(float)
