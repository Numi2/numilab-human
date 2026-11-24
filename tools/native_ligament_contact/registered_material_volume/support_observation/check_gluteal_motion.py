"""Verify the new reference tangent against the adipose law on recorded full-scene motion."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.transform import Rotation
from scipy.sparse.linalg import LinearOperator,cg
from assemble_gluteal_material import compliance,reference_stiffness
from bind_gluteal_receivers import digest,require


class Material:
    def __init__(self,g,volume,mu,bulk):
        self.g,self.volume,self.mu,self.bulk=g,volume,mu,bulk

    def gradients(self,displacements):
        return (self.g @ displacements.reshape(-1,3)).reshape(-1,3,3).transpose(0,2,1)

    def gather(self,P):
        weighted=(self.volume[:,None,None]*P).transpose(0,2,1).reshape(-1,3)
        return (self.g.T @ weighted).ravel()

    def state(self,u):
        A=self.gradients(u)
        F=np.eye(3)+A
        J=np.linalg.det(F)
        require(J.min()>.35 and J.max()<2.5,'Adipose deformation exceeds retained material domain')
        log=np.log(J)
        N=np.linalg.inv(F).transpose(0,2,1)
        c=self.bulk*log-self.mu
        # The difference form avoids subtraction of residual stresses of order mu.
        P=self.mu*(A+np.eye(3)-N)+self.bulk*log[:,None,None]*N
        E=self.volume @ (self.mu*(np.trace(A,axis1=1,axis2=2)-log+.5*np.sum(A*A,axis=(1,2)))+.5*self.bulk*log*log)
        return J,N,c,self.gather(P),float(E)

    def action(self,state,du):
        _,N,c,_,_=state
        D=self.gradients(du)
        dp=self.mu*D + self.bulk*np.sum(N*D,axis=(1,2))[:,None,None]*N - c[:,None,None]*(N @ D.transpose(0,2,1) @ N)
        return self.gather(dp)


def run(path,assets,case,out,scale=1.0):
    p=dict(np.load(path/'parents.npz'))
    r=json.loads((path/'reference-tangent.json').read_bytes())
    mu,bulk=np.array(r['material_parameters_Pa'])*scale
    g=sparse.load_npz(path/'gradient.npz')
    material=Material(g,p['volume'],mu,bulk)
    K=reference_stiffness(g,p['volume'],mu,bulk)
    H,_=compliance(assets,p)
    fat=3*len(p['fat'])
    chol=cho_factor(np.eye(H.shape[1])+H.T@K[fat:,fat:]@H)
    def stretch(u0):
        y=np.zeros(H.shape[1]);iterations=[]
        for iteration in range(15):
            u=u0.copy(); u[fat:]+=H@y
            st=material.state(u)
            f=y+H.T@st[3][fat:]
            iterations.append(float(np.linalg.norm(f)))
            if iterations[-1]<1e-12+1e-9*iterations[0]:
                break
            def apply(v):
                du=np.zeros_like(u);du[fat:]=H@v
                return v+H.T@material.action(st,du)[fat:]
            op=LinearOperator((len(y),len(y)),matvec=apply)
            M=LinearOperator(op.shape,matvec=lambda v:cho_solve(chol,v))
            dy,status=cg(op,-f,M=M,rtol=1e-10,atol=1e-15,maxiter=60)
            require(status==0,'Interior Newton correction failed')
            y+=dy
        else:raise ValueError('Interior residual failed')
        return u,st,iterations
    points=p['reference']
    rng=np.random.default_rng(78)
    test=rng.standard_normal(len(points)*3)
    gtest=material.gradients(test)
    test/=max(1.,np.max(abs(gtest)))
    zero=np.zeros_like(test)
    action=material.action(material.state(zero),test)
    reference_error=float(np.linalg.norm(action-K@test)/np.linalg.norm(action))
    require(reference_error<1e-8,'Constitutive reference tangent mismatch')
    # The physiology and rigid components both come from the same accepted frame.
    rec=case/'native-traces/accepted-geometry/step-1.receipt.json'
    a=json.loads(rec.read_bytes())
    pose={k['body_index']:k for k in a['accepted_body_poses']}
    initial={k['body_index']:k for k in a['initial_anatomical_registration']['body_poses']}
    accepted_file=case/'physical-accepted.node'
    current=np.loadtxt(accepted_file,skiprows=1,usecols=(1,2,3))[p['fat']]
    reference=points[:fat//3]
    # Evaluate in the moving pelvis frame, removing its exact finite rigid motion.
    base=Rotation.from_quat(initial[128]['quaternion_xyzw'])
    live=Rotation.from_quat(pose[128]['quaternion_xyzw'])
    to_ref=lambda x:base.apply(live.inv().apply(x-pose[128]['position_m']))+initial[128]['position_m']
    dm=np.zeros((len(points)-len(reference),3))
    for j,body in enumerate(p['donors']):
        X=points[len(reference):]
        old,new=initial[body],pose[body]
        loc=Rotation.from_quat(old['quaternion_xyzw']).inv().apply(X-old['position_m'])
        candidate=Rotation.from_quat(new['quaternion_xyzw']).apply(loc)+new['position_m']
        dm+=(to_ref(candidate)-X)*p['body_weights'][:,j,None]
    observed=np.vstack((to_ref(current)-reference,dm)).ravel()
    u0=np.zeros_like(observed);u0[:fat]=np.tile(p['bed_normal']*1e-4,len(reference))
    samples={}
    for name,base_u in [('recorded_first_step',observed),('virtual_compression_0p1_mm',u0)]:
        u,st,it=stretch(base_u)
        eps=1e-7
        fd=(material.state(u+eps*test)[3]-material.state(u-eps*test)[3])/(2*eps)
        tangent=material.action(st,test)
        err=float(np.linalg.norm(fd-tangent)/np.linalg.norm(tangent))
        require(err<5e-6,'Directional finite-strain tangent failed')
        f=st[3].reshape(-1,3)
        force_error=float(np.linalg.norm(f.sum(0))/max(1e-30,np.linalg.norm(f)))
        moment_error=float(np.linalg.norm(np.cross(points+u.reshape(-1,3)-points.mean(0),f).sum(0))/max(1e-30,np.linalg.norm(f)))
        require(max(force_error,moment_error)<1e-8,'Internal forces do not balance')
        samples[name]=dict(min_J=float(st[0].min()),max_J=float(st[0].max()),energy_J=st[4],
            iterative_residual=it,force_balance_relative=force_error,moment_balance_m=moment_error,
            computed_fat_energy_gradient_N=f[:len(reference)].sum(0).tolist(),directional_tangent_error=err)
    out.write_text(json.dumps(dict(source_reference_gradient_error=reference_error,
        endpoint=digest(rec),skin=digest(accepted_file),material=r['material'],
        adipose_modulus_multiplier=scale,observations=samples,scope='Offline evaluation on actual accepted geometry and a virtual indentation. Native trajectory did not receive these forces.'),
        indent=2,allow_nan=False)+'\n')
    print({k:(v['min_J'],v['max_J'],v['computed_fat_energy_gradient_N']) for k,v in samples.items()})
if __name__=='__main__':
    run(*(Path(v) for v in sys.argv[1:5]),scale=float(sys.argv[5]) if len(sys.argv)>5 else 1.0)
