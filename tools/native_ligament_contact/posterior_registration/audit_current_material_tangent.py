"""Measure the tangent of the registered finite-strain law at a captured native state."""
from pathlib import Path
import csv,json,sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh,splu,LinearOperator
from audit_complete_pelvic_forces import material
from material_context import sha
assets,run,out=map(Path,sys.argv[1:]);capture=run/'accepted-pelvic-state.csv'
rows=list(csv.DictReader(capture.open()));disp={int(r['index']):[float(r[k]) for k in 'xyz'] for r in rows if r['field']=='material_delta'}
x=np.array([disp[i] for i in range(len(disp))],'f4').astype(float)
cells=np.fromfile(assets/'material-cells.bin','u4').reshape(-1,4).astype(int)
inv=np.fromfile(assets/'pelvic-reference-inverse.bin','f4').reshape(-1,3,4)[:,:,:3].astype(float)
vol=np.fromfile(assets/'pelvic-cell-volume.bin','f4').astype(float)
free=np.fromfile(assets/'free-node-rows.bin','u4').reshape(-1,4)[:,2]!=0
doc=json.loads((assets/'manifest.json').read_text());law=doc['specimen'];mu,lam=law['mu'],law['lam']
result={'accepted_step':int(rows[0]['accepted_step']),'inputs':sha(capture),'analysis':'anchored material subspace; contact has separate one-sided tangent','sides':{}}
for part in doc['contact_flexibility']:
    begin=part['first_node'];n=part['nodes'];sel=(cells[:,0]>=begin)&(cells[:,0]<begin+n)
    t=cells[sel]-begin;I=inv[sel];v=vol[sel];u=x[begin:begin+n]
    a=t[:,[0,0,0,1,1,2]].ravel();b=t[:,[1,2,3,2,3,3]].ravel()
    _,cl=connected_components(coo_matrix((np.ones(len(a)),(a,b)),(n,n)),directed=False)
    fr=free[begin:begin+n];held=set(cl[~fr]);linked=np.array([k in held for k in cl])
    valid=fr&linked;idx=np.flatnonzero(np.repeat(valid,3))
    G=I.transpose(0,2,1);G=np.concatenate((-G.sum(1)[:,None],G),axis=1)
    dof=(3*t[:,:,None]+np.arange(3)).reshape(-1,12)
    rr=np.broadcast_to(dof[:,:,None],(len(t),12,12)).ravel();cc=np.broadcast_to(dof[:,None,:],(len(t),12,12)).ravel()
    def tangent(u):
        F=(I@(u[t[:,1:]]-u[t[:,:1]])).transpose(0,2,1)+np.eye(3)
        det=np.linalg.det(F);T=G@np.linalg.inv(F)
        K=mu*np.einsum('eai,ebi,jk->eajbk',G,G,np.eye(3)) + lam*np.einsum('eaj,ebk->eajbk',T,T) +np.einsum('e,ebj,eak->eajbk',mu-lam*np.log(det),T,T)
        return coo_matrix(((K*v[:,None,None,None,None]).ravel(),(rr,cc)),(3*n,3*n)).tocsr()
    A=tangent(u);M=tangent(np.zeros_like(u))
    # Differentiate the independent Float64 stress integral directly.
    rng=np.random.default_rng(9921);test=rng.normal(size=u.shape);test[~valid]=0
    eps=1e-10;plus=material(u+eps*test,t,I,v,mu,lam,True)[0];minus=material(u-eps*test,t,I,v,mu,lam,True)[0]
    fd=(plus-minus).ravel()/(2*eps);analytic=A@test.ravel();deriv=np.linalg.norm(fd-analytic)/np.linalg.norm(analytic)
    if deriv>1e-5:raise ValueError(('tangent',deriv))
    A=A[idx,:][:,idx].tocsc();M=M[idx,:][:,idx].tocsc()
    # Symmetric diagonal normalization before factoring the reference operator.
    d=1/np.sqrt(M.diagonal());A=A.multiply(d[:,None]).multiply(d[None,:]).tocsc();M=M.multiply(d[:,None]).multiply(d[None,:]).tocsc()
    lu=splu(M);mi=LinearOperator(M.shape,matvec=lu.solve);w0=np.random.default_rng(17).normal(size=len(idx))
    hi,V=eigsh(A,2,M=M,Minv=mi,which='LA',tol=1e-6,v0=w0)
    lo,L=eigsh(A,2,M=M,Minv=mi,which='SA',tol=1e-6,v0=w0)
    def report(z,v):
        dz=A@v;rz=M@v
        # Report generalized eigenpair residual on the diagonal-scaled equations.
        rel=np.linalg.norm(dz-z*rz)/max(np.linalg.norm(dz),abs(z)*np.linalg.norm(rz),1e-30)
        direction=np.zeros(3*n);direction[idx]=v*d;direction=direction.reshape(-1,3);node=int(np.argmax(np.linalg.norm(direction,axis=1)))
        return dict(ratio=z,residual=rel,largest_displacement_node=node+begin)
    result['sides'][str(part['muscle'])]=dict(connected_free_dofs=len(idx),omitted_floating_free_nodes=int(sum(fr&~linked)),
        directional_derivative_relative_error=deriv,minimum_ratio=report(lo[0],L[:,0]),maximum_ratio=report(hi[-1],V[:,-1]))
out.write_text(json.dumps(result,indent=2)+'\n')
