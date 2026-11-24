"""Equilibrate a local gravity-loaded bed indentation using the original hyperelastic energy."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

def refine(law,gather,Y,t,local,rows,bound,U,mass,vertical):
    # Only the connected free coordinates participate in preparation. The
    # holding reactions at the patch boundary are measured separately.
    dof=np.full((len(Y),3),-3,dtype=int)
    dof[local]=np.arange(3*len(local)).reshape(-1,3)
    r=dof[t][:,:,None,:,None];c=dof[t][:,None,:,None,:]
    rr,cc=np.broadcast_arrays(r,c);used=(rr>=0)&(cc>=0)
    body=mass[:,None]*9.81*vertical
    def state(u,tangent=False):
        p=Y.copy();p[local]+=u.reshape(-1,3)
        f,m=law(p,tangent)
        g=(gather(f)[local]+body).ravel()
        if not tangent:return g
        A=coo_matrix((m[used],(rr[used],cc[used])),shape=(len(u),len(u))).tocsc()
        return g,A
    for iteration in range(32):
        g,A=state(U,True);active=U[rows]<=bound+1e-12
        lower=bound-U[rows]
        for _ in range(32):
            fix=rows[active];keep=np.ones(len(U),bool);keep[fix]=False
            step=np.zeros_like(U);step[fix]=lower[active]
            step[keep]=splu(A[keep][:,keep]).solve(-g[keep]-A[keep][:,fix]@step[fix])
            dual=g+A@step
            release=active&(dual[rows]<-1.e-7)
            add=~active&(step[rows]<lower-1.e-11)
            if not (release|add).any():break
            active=(active&~release)|add
        else:raise ValueError('Unresolved nonlinear contact set')
        projected=g.copy();projected[rows[active]]=np.minimum(0,projected[rows[active]])
        error=float(np.linalg.norm(projected))
        if error<1.e-6 and np.max(abs(step))<1e-8:return U,active,error,iteration
        slope=g@step
        scale=1.
        while scale>=2**-20:
            trial=U+scale*step
            try:
                # Integrated work along a Newton line. Each sample uses the
                # full source law, including gravity. Keep an energy decrease.
                a=0.5-0.5/np.sqrt(3);b=1.-a
                work=.5*scale*(state(U+a*scale*step)+state(U+b*scale*step))@step
                state(trial)
                if np.isfinite(work) and work<=min(1.e-13,1e-4*scale*slope+1e-15):break
            except (ValueError,np.linalg.LinAlgError):pass
            scale*=.5
        else:raise ValueError('No admissible nonlinear work decrease')
        U=trial
    raise ValueError('Nonlinear patch did not reach force balance')
