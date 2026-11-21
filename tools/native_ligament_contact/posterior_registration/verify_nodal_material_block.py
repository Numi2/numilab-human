"""Check the analytic nodal material block against finite differences in the captured scene."""
from pathlib import Path
import json,sys
import numpy as np
from audit_complete_pelvic_forces import material
assets,run,out=map(Path,sys.argv[1:]);bound=json.loads((run/'nodal-pressure-bound.json').read_text())
nodes=np.fromfile(run/'candidate-force-nodes.bin','<f4').reshape(-1,2,4).astype(float)
u=nodes[:,1,:3].copy();c=np.fromfile(assets/'material-cells.bin','<u4').reshape(-1,4)
I=np.fromfile(assets/'pelvic-reference-inverse.bin','<f4').reshape(-1,3,4)[:,:,:3].astype(float)
V=np.fromfile(assets/'pelvic-cell-volume.bin','<f4').astype(float)
laws=json.loads((assets/'manifest.json').read_text())['specimen'];mu,la=laws['mu'],laws['lam']
result={}
for side,part in bound['sides'].items():
    n=part['maximum_node'];mask=(c==n).any(1);i=I[mask];t=c[mask];v=V[mask]
    local=(t==n).argmax(1)
    def grad(x):return material(x,t,i,v,mu,la,True)[0][n]
    g=i.transpose(0,2,1);g=np.concatenate((-g.sum(1)[:,None],g),axis=1)[np.arange(len(t)),local]
    F=(i@(u[t[:,1:]]-u[t[:,:1]])).transpose(0,2,1)+np.eye(3)
    J=np.linalg.det(F);ng=np.einsum('eji,ej->ei',np.linalg.inv(F),g)
    H=(v[:,None,None]*(mu*np.sum(g*g,axis=1)[:,None,None]*np.eye(3)+(mu+la-la*np.log(J))[:,None,None]*ng[:,:,None]*ng[:,None,:])).sum(0)
    fd=np.empty((3,3));eps=1e-9
    for k in range(3):
        a=u.copy();b=u.copy();a[n,k]+=eps;b[n,k]-=eps
        fd[:,k]=(grad(a)-grad(b))/(2*eps)
    error=np.linalg.norm(fd-H)/np.linalg.norm(H)
    if error>2e-6:raise ValueError(('directional Hessian',n,error))
    old=grad(u);h=-np.linalg.solve(H,old);new=u.copy();new[n]+=h;after=grad(new)
    energyBefore=material(u,t,i,v,mu,la,True)[1];energyAfter=material(new,t,i,v,mu,la,True)[1]
    if energyAfter>=energyBefore or np.linalg.norm(after)>=np.linalg.norm(old):raise ValueError('nodal correction is not descent')
    result[side]=dict(node=n,cells=int(mask.sum()),block_relative_error=error,correction_m=h.tolist(),correction_norm_m=float(np.linalg.norm(h)),current_residual_N=float(np.linalg.norm(old)),fixed_traction_residual_after_one_update_N=float(np.linalg.norm(after)),energy_change_J=float(energyAfter-energyBefore))
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
