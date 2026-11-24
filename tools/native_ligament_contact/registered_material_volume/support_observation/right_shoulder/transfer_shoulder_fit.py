"""Transport a joint shoulder fit into its source bindings and verify four material poses after emission."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from igl.copyleft import cgal
from assemble_material import frames
from repair_registered_surface import state,volume
from registration_stages import parts
from prepare_infraspinatus import dets
from prepare_shoulder import read,MOVING
from compile_connected_joint import digest


def transfer(case,cur,cage,fit,out):
    out.mkdir(exist_ok=True)
    poses=frames(case);cfg=read(case/'launch.json');a=cfg['argv'];payload=Path(a[a.index('--soft-tissue-payload')+1])
    results={};pre=[]
    for sid in MOVING:
        z=np.load(cur/f'bank-{sid}.npz');p=np.load(cage/f'bank-{sid}.npz');adj=np.load(fit/f'{sid}.npz')
        _,_,owners,_,aff=parts(payload,sid);w=z['w'].astype('f4').astype(float);ref=z['x'];f=z['f'];t=p['t'];Y=p['accepted']
        if np.max(abs(w.sum(1)-1))>2e-6:raise ValueError('Invalid emitted weights')
        if not np.array_equal(adj['t'],t):raise ValueError('Different material topology')
        basis=[Rotation.from_quat(poses[-1][b]['quaternion_xyzw']).as_matrix()@A for b,(A,_)in zip(owners,aff)]
        J=np.einsum('vb,bij->vij',w,basis);dx=adj['displacement_m']
        if np.linalg.norm(dx[adj['fixed_nodes']],axis=1).max()>1e-13:raise ValueError('Attachment moved')
        dv=np.linalg.solve(J,dx[:len(ref),:,None])[:,:,0]
        poses0=[p[k]for k in ['reference','rig_reference','opening','accepted']]
        E=Y[t[:,1:]]-Y[t[:,:1]];dy=dets(Y,t)
        # A nodal push-forward of the material displacement provides an interior
        # registration. The boundary always uses the original skeletal blend.
        maps=[]
        for X in poses0:
            H=np.linalg.solve(E,X[t[:,1:]]-X[t[:,:1]]).transpose(0,2,1)
            avg=np.zeros((len(Y),3,3));weights=np.zeros(len(Y))
            for ids in t.T:
                np.add.at(avg,ids,H*dy[:,None,None]);np.add.at(weights,ids,dy)
            maps.append(avg/weights[:,None,None])
        pre.append((sid,w,ref,f,t,dx,dv,owners,aff,poses0,maps,adj['fixed_nodes']))
    trials=[]
    for halvings in range(14):
        rows={};ready=[];alpha=2.**-halvings
        for sid,w,ref,f,t,dx,dv,owners,aff,poses0,maps,pinned in pre:
            source=(ref+alpha*dv).astype('f4').astype(float)
            boundaries=[source]+[state(source,w,aff,owners,p).astype('f4').astype(float) for p in poses]
            xyz=[];metrics={}
            for key,X,M,b in zip(['reference','rig_reference','opening','accepted'],poses0,maps,boundaries):
                x=X+alpha*np.einsum('nij,nj->ni',M,dx);x[:len(ref)]=b
                J=dets(x,t)/dets(X,t);hits=cgal.remesh_self_intersections(b,np.ascontiguousarray(f),detect_only=True)[2]
                metrics[key]=dict(min_relative_J=float(J.min()),maximum_relative_J=float(J.max()),nonpositive=int((J<=0).sum()),surface_crossing_pairs=len(hits),signed_mm3=volume(b,f)*1e9)
                xyz.append(x)
            samples=[0,1/3,2/3,1.]
            controls=np.linalg.solve(np.array([[(1-a)**3,3*a*(1-a)**2,3*a*a*(1-a),a**3] for a in samples]),np.array([dets((1-a)*xyz[2]+a*xyz[3],t)/dets(xyz[0],t)for a in samples]))
            quant_error=np.linalg.norm(boundaries[-1]-(poses0[-1]+alpha*dx)[:len(ref)],axis=1).max()
            rows[sid]=dict(stages=metrics,minimum_motion_J=float(controls.min()),maximum_compiled_discrepancy_m=float(quant_error),pinned_source_delta_m=float(np.linalg.norm(source[pinned]-ref[pinned],axis=1).max()))
            ready.append((sid,source,w,f,t,xyz,owners))
        trials.append(dict(alpha=alpha,issues={str(k):{kk:[sv['surface_crossing_pairs'],sv['nonpositive']]for kk,sv in row['stages'].items() if sv['surface_crossing_pairs'] or sv['nonpositive']}for k,row in rows.items()}))
        if all(r['minimum_motion_J']>0 and all(s['surface_crossing_pairs']==s['nonpositive']==0 for s in r['stages'].values())for r in rows.values()):break
    else:raise ValueError('No common motion-valid fit')
    for sid,source,w,f,t,xyz,owners in ready:
        np.savez_compressed(out/f'bank-{sid}.npz',x=source,w=w,f=f,native=xyz[-1][:len(source)],owners=owners)
        np.savez_compressed(out/f'cells-{sid}.npz',t=t,reference=xyz[0],rig_reference=xyz[1],opening=xyz[2],accepted=xyz[3])
    result=dict(native=digest(case/'launch.json'),source=digest(cur/'results.json'),fit=digest(fit/'result.json'),fraction=alpha,rows=rows,trials=trials)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('common fit fraction',alpha,'least motion J',min(r['minimum_motion_J']for r in rows.values()),flush=True)


if __name__=='__main__':transfer(*(Path(s).resolve()for s in sys.argv[1:]))
