"""Constrain the source material weights against true signed volumes along the recorded hip trajectory."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix,eye,hstack,vstack,csr_matrix
from scipy.spatial.transform import Rotation
from pose_glutes import parts,states,plane
from preserve_muscle_cells import det
from compare_hand_interfaces import query
from material_context import sha

def quality_denom(x,t):
    v=x[t]
    return np.maximum.reduce([np.linalg.norm(v[:,i]-v[:,j],axis=1) for i in range(4) for j in range(i)])**3
def signed_gradient(x,t,d,ref):
    a,b,c,e=x[t].transpose(1,0,2)
    g=np.stack((np.cross(c-a,e-a),np.cross(e-a,b-a),np.cross(b-a,c-a)),axis=1)
    return np.einsum('tij,tij->ti',np.concatenate((-g.sum(1)[:,None,:],g),axis=1),d[t])/ref[:,None]
def run(cfg,trace,receipt,volume,model,exe,out,material=None):
    import re
    ceiling=None if material is None else .96*float(re.search(r'\bmaximum_J\s*=\s*([\d.]+)',material.read_text())[1])
    out.mkdir(exist_ok=True)
    args=json.loads(cfg.read_text())['argv'];payload=Path(args[args.index('--soft-tissue-payload')+1])
    fr=states(trace,receipt);keys=sorted(fr);train=np.r_[np.arange(0,len(keys)-1,4),len(keys)-1]
    stats={};bank={};raw={}
    for sid in (29,30):
        _,_,bs,f,aff=parts(payload,sid);draft=np.load(model/(str(sid)+'-band8.npz'))
        src=draft['vertices'].astype(float);w0=draft['weights'][:,1].astype(float);bind=draft['secondary_binding']
        aff=[aff[0],(Rotation.from_quat(bind[3:7]).as_matrix()*bind[7],bind[:3])]
        local=[src@a.T+b for a,b in aff];donor=[]
        for body,q in zip(bs,local):
            donor.append(np.array([Rotation.from_quat(fr[k][body][3:]).apply(q)+fr[k][body][:3] for k in keys]))
        X=donor[0];D=donor[1]-X;w=w0.copy()
        tet=np.loadtxt(volume/str(sid)/'source.1.ele',skiprows=1,dtype=int)[:,1:5]
        entrance=(X[0]+D[0]*w[:,None]).astype('f4').astype(float)
        reference=det(entrance,tet)
        if reference.min()<=0:raise ValueError('Invalid registered start')
        origin,lateral=plane(fr,0);depth=(1 if sid==30 else -1)*(entrance-origin)@lateral
        fixed=np.union1d(np.union1d(draft['origins'],draft['insertions']),np.flatnonzero(depth<.008))
        # All held-in states are checked after each LP. Only cells anywhere near
        # their determinant/shape guard enter the local solve.
        log=[];ever=set();floors=.28;shape_floor=2.e-5
        def measure(ids,w):
            result=[];quality=[]
            for k in ids:
                xx=(X[k]+D[k]*w[:,None]).astype('f4').astype(float)
                dv=det(xx,tet);result.append(dv/reference);quality.append(dv/quality_denom(xx,tet))
            return np.array(result),np.array(quality)
        for generation in range(16):
            ratios,qualities=measure(train,w)
            bad=(ratios<floors)|(qualities<shape_floor)
            if ceiling is not None:bad|=(ratios>ceiling)
            log.append(dict(iteration=generation,bad_cells=int(bad.any(0).sum()),min_ratio=ratios.min(),min_shape=qualities.min()))
            if not bad.any():break
            ever.update(np.flatnonzero(bad.any(0)))
            free=np.setdiff1d(np.unique(tet[sorted(ever)]),fixed);n=len(free)
            neighborhood=np.flatnonzero(np.any(np.isin(tet,free),axis=1))
            free=np.setdiff1d(np.unique(tet[neighborhood]),fixed);n=len(free)
            near=np.flatnonzero(np.any(np.isin(tet,free),axis=1))
            index=np.full(len(w),-1,int);index[free]=np.arange(n)
            A=[];beta=[]
            for z,k in enumerate(train):
                x=X[k]+D[k]*w[:,None];g=signed_gradient(x,tet[near],D[k],reference[near])
                q=quality_denom(x,tet[near])/reference[near]
                floor=np.maximum(floors+.025,shape_floor*1.15*q)
                positions=index[tet[near]];v=positions>=0
                mat=coo_matrix((-g[v],(np.broadcast_to(np.arange(len(near))[:,None],v.shape)[v],positions[v])),shape=(len(near),n)).tocsr()
                keep=np.flatnonzero(mat.getnnz(1))
                A.append(mat[keep]);beta.append((ratios[z,near]-floor)[keep])
                if ceiling is not None:
                    A.append(-mat[keep]);beta.append((ceiling-.1-ratios[z,near])[keep])
            mat=vstack(A,format='csr');rhs=np.concatenate(beta)
            identity=eye(n,format='csr')
            # Minimize the maximum change in donor weight, then total change.
            uv=vstack((hstack((mat,csr_matrix((mat.shape[0],n+1)))),hstack((identity,-identity,csr_matrix((n,1)))),hstack((-identity,-identity,csr_matrix((n,1)))),hstack((csr_matrix((n,n)),identity,-np.ones((n,1))))),format='csr')
            dx=w[free]-w0[free]
            bounds=list(zip(-w[free],1.-w[free]))+[(0.,None)]*(n+1)
            opt=linprog(np.r_[np.zeros(n),np.full(n,.02/n),1.],A_ub=uv,b_ub=np.r_[rhs,-dx,dx,np.zeros(n)],bounds=bounds,method='highs')
            if not opt.success:raise ValueError(('No feasible anatomical weight correction',sid,log[-1],opt.message))
            w[free]=np.clip(w[free]+opt.x[:n],0.,1.)
        w=w.astype('f4').astype(float);assert np.array_equal(w[fixed],w0[fixed])
        ratio,q=measure(range(len(keys)),w)
        shapes=(X+D*w[None,:,None]).astype('f4');bank[sid]=(shapes,f);raw[sid]=src
        stats[sid]=dict(initial=log[0],solver=log,changed_from_anatomical_field=int(np.count_nonzero(w!=w0)),max_adjustment=float(np.max(abs(w-w0))),fixed_donor_vertices=len(fixed),
            minimum_volume_ratio=float(ratio.min()),minimum_normalized_determinant=float(q.min()),inverted_cell_samples=int((ratio<=0).sum()),low_quality_cell_samples=int((q<1e-5).sum()),cells=len(tet),registered_reference_volume_m3=float(reference.sum()/6))
        np.savez_compressed(out/(str(sid)+'.npz'),vertices=src.astype('f4'),weights=np.column_stack((1.-w,w)).astype('f4'),secondary_binding=bind,origins=draft['origins'],insertions=draft['insertions'])
    x,f=bank[29];y,g=bank[30];fg=np.vstack((f,g+len(x[0])));N=len(f);mask=np.ones(len(fg),bool)
    hits=[]
    for idx,k in enumerate(keys):
        pp=query(exe,np.vstack((x[idx],y[idx])),fg,mask)
        hits.append([k,sum(a<N<=b for a,b in pp),sum(b<N for a,b in pp),sum(a>=N for a,b in pp)])
    result=dict(parent=sha(payload),upper_specimen_volume_guard=ceiling,material_sha256=sha(material) if material else None,held_in_steps=[keys[i] for i in train],held_out_samples=len(keys)-len(train),by_side=stats,all_pose_intersections=hits)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(stats,flush=True)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
