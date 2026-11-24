"""Extend measured boundary motion through each tetrahedral reference by its P1 harmonic displacement."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from assemble_material import frames
from registration_stages import parts
from repair_registered_surface import state
from prepare_infraspinatus import dets,exterior
from prepare_shoulder import MOVING,read
from compile_connected_joint import digest


def certify(case,proj,out):
    out.mkdir(exist_ok=True)
    a=read(case/'launch.json')['argv'];src=Path(a[a.index('--soft-tissue-payload')+1]);p=frames(case)
    result={}
    for sid in MOVING:
        z=np.load(proj/f'bank-{sid}.npz')
        vol=proj/f'cells-{sid}/material.npz';material=np.load(vol)
        Y=material['x'];t=material['t'];surf=z['x'];n=len(surf)
        _,_,owners,_,aff=parts(src,sid)
        targets=[surf]+[state(surf,z['w'],aff,owners,q).astype('f4').astype(float) for q in p]
        if np.max(abs(Y[:n]-targets[-1]))>1e-12 or np.any(dets(Y,t)<=0):raise ValueError('Frame and solid mismatch')
        if set(map(tuple,exterior(t)))!=set(map(tuple,np.sort(z['f'],axis=1))):raise ValueError('Changed surface ancestry')
        d=dets(Y,t)
        g=np.linalg.inv(Y[t[:,1:]]-Y[t[:,:1]])
        g=np.concatenate((-g.sum(2,keepdims=True),g),axis=2).transpose(0,2,1)
        elem=(g@g.transpose(0,2,1))*(d/6)[:,None,None]
        ii,jj=np.broadcast_arrays(t[:,:,None],t[:,None,:])
        A=coo_matrix((elem.ravel(),(ii.ravel(),jj.ravel())),shape=(len(Y),len(Y))).tocsc()
        solve=splu(A[n:,n:]) if len(Y)>n else None
        points=[]
        for target in targets:
            dx=target-Y[:n];X=Y.copy();X[:n]=target
            if solve is not None:X[n:]+=solve.solve(-(A[n:,:n]@dx))
            points.append(X)
        # The native prefix is the final segment (opening->accepted). Source->rest
        # is a coordinate registration, so interpolating their global rotations is meaningless.
        vals=np.array([dets(points[2]*(1-s)+points[3]*s,t)/d for s in [0,1/3,2/3,1]])
        b=np.array([[(1-s)**3,3*s*(1-s)**2,3*s*s*(1-s),s**3] for s in [0,1/3,2/3,1]])
        controls=np.linalg.solve(b,vals);rows={}
        for k,x in zip(['atlas','rig_reference','opening','accepted'],points):
            det=dets(x,t)
            rows[k]=dict(nonpositive=int(np.sum(det<=0)),volume_mm3=float(det.sum()/6*1e9),minimum_J_against_accepted=float((det/d).min()))
        result[sid]=dict(poses=rows,registered_cells=len(t),opening_to_accepted_bernstein_lower=float(controls.min()),cells=digest(vol),peak_native_motion_m=float(np.linalg.norm(points[2]-points[3],axis=1).max()))
        np.savez_compressed(out/f'bank-{sid}.npz',t=t,reference=points[0],rig_reference=points[1],opening=points[2],accepted=points[3])
        print(sid,[r['nonpositive']for r in rows.values()],controls.min(),flush=True)
    (out/'result.json').write_text(json.dumps(dict(rows=result,source=digest(src),scene=digest(case/'launch.json'),stage='registered boundary replay'),indent=2,allow_nan=False)+'\n')
if __name__=='__main__':certify(*(Path(a).resolve()for a in sys.argv[1:]))
