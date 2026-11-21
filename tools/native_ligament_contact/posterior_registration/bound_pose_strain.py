"""Fit a common neutral material field with bounds on the actual tet deformation gradient.

Each supporting plane is a valid necessary inequality on the convex Frobenius norm.
No force, stiffness, body transform, topology or runtime equation is fitted.
"""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,csr_matrix,eye,hstack,vstack
from scipy.spatial.transform import Rotation
from pose_glutes import parts,states,model,plane
def run(config,trace,receipt,volume,initial,out):
    out.mkdir(exist_ok=True);args=json.loads(config.read_text())['argv'];asset=Path(args[args.index('--soft-tissue-payload')+1]);pose=states(trace,receipt)
    keys=sorted(pose);train=sorted(set(keys[:17]+keys[1::16]+keys[-1:]));o,axis=plane(pose,0);report={}
    for sid in (29,30):
        data=parts(asset,sid);_,_,donors,_,aff=data;seed=np.load(initial/(str(sid)+'-band8.npz'));src=seed['vertices'].astype(float)
        b=seed['secondary_binding'];af=[aff[0],(Rotation.from_quat(b[3:7]).as_matrix()*b[7],b[:3])]
        material=[src@a.T+d for a,d in af];w0=seed['weights'][:,1].astype(float);n=len(src)
        x=model(data,pose[0],1)[0].astype('f4').astype(float)
        tet=np.loadtxt(volume/str(sid)/'source.1.ele',skiprows=1,dtype=int)[:,1:5];pts=x[tet]
        inv=np.linalg.inv((pts[:,1:]-pts[:,:1]).transpose(0,2,1))
        grad=np.concatenate((-inv.sum(1)[:,None,:],inv),axis=1)
        side=(-1 if sid==29 else 1);fixed=np.union1d(np.union1d(seed['origins'],seed['insertions']),np.flatnonzero(side*(x-o)@axis<.008))
        rot0=Rotation.from_quat(pose[0][donors[0]][3:])
        def operators(k):
            ends=[Rotation.from_quat(pose[k][j][3:]).apply(q)+pose[k][j][:3] for j,q in zip(donors,material)]
            R=(Rotation.from_quat(pose[k][donors[0]][3:])*rot0.inv()).as_matrix()
            F0=(ends[0][tet[:,1:]]-ends[0][tet[:,:1]]).transpose(0,2,1)@inv
            return F0-R,ends[1]-ends[0]
        ops={k:operators(k) for k in train}
        target=.62
        I=eye(n,format='csr');zero=csr_matrix((n,1))
        A=[hstack((I,-I,zero)),hstack((-I,-I,zero)),hstack((csr_matrix((n,n)),I,-np.ones((n,1))))]
        rhs=[w0,-w0,np.zeros(n)];bounds=[(0.,1.)]*n+[(0.,None)]*(n+1)
        for j in fixed:bounds[j]=(w0[j],w0[j])
        w=w0.copy();log=[];radius=target
        def strain(B,d,w):
            return B+(d[tet]*w[tet,None]).transpose(0,2,1)@grad
        for iteration in range(120):
            norms=np.array([np.linalg.norm(strain(B,d,w),axis=(1,2)) for B,d in ops.values()])
            which=norms.argmax(0);worst=norms.max(0);bad=np.flatnonzero(worst>radius+.005)
            log.append([iteration,len(bad),float(worst.max()),float(radius)])
            if not len(bad):break
            for a,(k,(B,d)) in enumerate(ops.items()):
                ids=bad[which[bad]==a]
                if not len(ids):continue
                normal=strain(B,d,w)[ids]/worst[ids,None,None]
                co=np.einsum('tij,tvi,tvj->tv',normal,d[tet[ids]],grad[ids])
                block=coo_matrix((co.ravel(),(np.repeat(np.arange(len(ids)),4),tet[ids].ravel())),shape=(len(ids),n)).tocsr()
                A.append(hstack((block,csr_matrix((len(ids),n+1)))));rhs.append(-np.einsum('tij,tij->t',normal,B[ids]))
            # The minimum possible radius is measured if the initial engineering target is infeasible.
            blocks=vstack(A,format='csr')
            epigraph=np.r_[np.zeros(3*n),-np.ones(blocks.shape[0]-3*n)]
            opt=linprog(np.r_[np.zeros(n),np.full(n,.005/n),.025,1.],A_ub=hstack((blocks,epigraph[:,None]),format='csr'),b_ub=np.concatenate(rhs),bounds=bounds+[(target,None)],method='highs')
            if not opt.success:raise ValueError((sid,log[-1],opt.message))
            w=opt.x[:n];radius=opt.x[-1]
        if log[-1][1]:raise ValueError(('Did not certify training gradient',sid,log[-1]))
        w=w.astype('f4');assert np.array_equal(w[fixed],w0[fixed])
        all_norm=[float(np.max(np.linalg.norm(strain(*operators(k),w),axis=(1,2)))) for k in keys]
        report[sid]=dict(training_pose_steps=train,target_gradient_radius=target,attained_radius=radius,maximum_held_out_and_training=max(all_norm),initial_max=log[0][2],iterations=log,maximum_weight_adjustment=float(np.abs(w-w0).max()),pinned=len(fixed))
        np.savez_compressed(out/(str(sid)+'-band8.npz'),**{k:(np.column_stack((1.-w,w)).astype('f4') if k=='weights' else seed[k]) for k in seed.files})
    (out/'strain-gradient.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
