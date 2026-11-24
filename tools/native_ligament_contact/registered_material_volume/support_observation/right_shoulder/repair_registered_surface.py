"""Repair intersecting source patches in registered shoulder surfaces across matched body poses."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from igl.copyleft import cgal
from registration_stages import parts
from prepare_shoulder import read, LABELS
from compile_connected_joint import digest
from inspect_shoulder_source import nearest


def volume(x,f):
    p=x[f]-x.mean(0)
    return np.einsum('ij,ij->',p[:,0],np.cross(p[:,1],p[:,2]))/6


def signed_volume_gradient(x,f):
    grad=np.zeros_like(x)
    a,b,c=x[f].transpose(1,0,2)
    n=np.cross(b-a,c-a)/6
    for ids in f.T:
        np.add.at(grad,ids,n)
    return grad


def state(x,weights,bindings,owners,poses):
    mapped=np.zeros_like(x)
    for i,(b,(M,t)) in enumerate(zip(owners,bindings)):
        R=Rotation.from_quat(poses[b]['quaternion_xyzw']).as_matrix()
        mapped+=weights[:,i,None]*(x@(R@M).T+R@t+poses[b]['position_m'])
    return mapped


def per_pose(v,w,faces,aff,owners,poses):
    # Quantize both source and world coordinates as the emitted mesh does.
    src=v.astype('f4').astype('f8')
    frames=[src]+[state(src,w,aff,owners,p).astype('f4').astype('f8') for p in poses]
    return [cgal.remesh_self_intersections(np.ascontiguousarray(x),faces,detect_only=True)[2] for x in frames]


def repair(v,w,f,bindings,owners,poses,pins):
    ix=np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    adj=sparse.csr_matrix((np.ones(len(ix)*2),(np.r_[ix[:,0],ix[:,1]],np.r_[ix[:,1],ix[:,0]])),shape=(len(v),len(v)))
    adj.data.fill(1)
    average=sparse.diags(1/np.asarray(adj.sum(1)).ravel())@adj
    ref=v.copy();goal=volume(ref,f);counts=[];seen=np.zeros(len(v),bool)
    for step in range(64):
        hits=per_pose(v,w,f,bindings,owners,poses)
        counts.append([len(h) for h in hits])
        if not any(counts[-1]):break
        active=np.zeros(len(v),bool)
        for h in hits:active[f[h].ravel()]=True
        halo=np.asarray(adj@active.astype(float)>0)
        seen|=active|halo
        alpha=np.where(active,.45,np.where(halo,.12,0))
        alpha[pins]=0
        if not np.any(alpha):break
        cand=v+alpha[:,None]*(average@v-v)
        # Balance signed volume within the moving patch only; keep attachment sites fixed.
        for _ in range(5):
            g=signed_volume_gradient(cand,f)*alpha[:,None]
            d=np.sum(g*signed_volume_gradient(cand,f))
            if d==0:break
            delta=(goal-volume(cand,f))/d
            cand+=g*delta
            if abs(volume(cand,f)-goal)<1.e-14:break
        v=cand.astype('f4').astype('f8')
    return v,counts,seen


def main(case,srcdir,lib,out):
    out.mkdir(exist_ok=True)
    a=read(case/'launch.json')['argv'];p=Path(a[a.index('--soft-tissue-payload')+1])
    tendon=Path(a[a.index('--tendon-payload')+1]);endpoints=read(tendon.with_suffix('.manifest.json'))['endpoints']
    opening=read(case/'native-initial-bodies.json')['all_bodies']
    rc=next((case/'native-traces/accepted-geometry').glob('*.receipt.json'))
    poses=[{r['body_index']:dict(quaternion_xyzw=r[k]['orientation_world_xyzw'],position_m=r[k]['position_world_m']) for r in opening} for k in ['rest','current']]
    accepted={r['body_index']:r for r in read(rc)['accepted_body_poses']};poses.append(accepted)
    rows={}
    for sid in (75,85,87):
        v,w,owners,f,aff=parts(p,sid);used=np.unique(f)
        u,idx,inv=np.unique(v[used],axis=0,return_index=True,return_inverse=True)
        cur=(u,w[used[idx]],inv[np.searchsorted(used,f)])
        atlas=np.load(srcdir/f'{sid}.npz')
        if list(atlas['body_ids'])!=owners:raise ValueError('Incompatible donors')
        for name,(x,wt,fc) in [('atlas',(atlas['x'],atlas['weights'],atlas['faces'])),('current',cur)]:
            fc=np.ascontiguousarray(fc,np.int64)
            xf=state(x,wt,aff,owners,accepted)
            pins=set(); anchors=[]
            for e in endpoints:
                if e.get('muscle')!=LABELS[sid]:continue
                b=e['body_index'];R=Rotation.from_quat(accepted[b]['quaternion_xyzw'])
                surface=e.get('surface') or {}
                target=R.apply(np.array([e['resolved_local_point_m']]+list(surface.get('node_local_points_m',[]))))+accepted[b]['position_m']
                _,dist,ids=nearest((np.ascontiguousarray(xf),np.ascontiguousarray(fc,'i4')),target,lib)
                pins.update(fc[ids].ravel().tolist());anchors.append((e['endpoint'],b))
            if {r[0] for r in anchors}!={'origin','insertion'}:raise ValueError('Missing donor attachment')
            y,hist,seen=repair(x.astype('f4').astype('f8'),wt,fc,aff,owners,poses,sorted(pins))
            after=per_pose(y,wt,fc,aff,owners,poses)
            if not np.array_equal(y[sorted(pins)].astype('f4'),x[sorted(pins)].astype('f4')):raise ValueError('Changed pinned source')
            key=f'{sid}-{name}'
            dx=np.linalg.norm(x-y,axis=1)
            rows[key]=dict(before=hist[0],after=[len(r) for r in after],iterations=len(hist),peak_mm=float(dx.max()*1000),
                moved=int((dx>1e-7).sum()),source_volume_mm3=volume(x,fc)*1e9,delta_volume_mm3=(volume(y,fc)-volume(x,fc))*1e9,
                pins=len(pins),anchors=anchors,history=hist,body_ids=owners)
            np.savez_compressed(out/(key+'.npz'),source=x,vertices=y,weights=wt,faces=fc,owners=owners,pins=sorted(pins),patch=seen,
                accepted=state(y,wt,aff,owners,accepted).astype('f4'))
            print(key,hist[0],rows[key]['after'],round(rows[key]['peak_mm'],4),round(rows[key]['delta_volume_mm3'],3),flush=True)
            (out/'results.json').write_text(json.dumps(dict(rows=rows,stages=['source','rest','opening','accepted'],payload=digest(p),tendon=digest(tendon),frame=digest(rc)),indent=2)+'\n')

if __name__=='__main__':
    import sys
    main(*(Path(a).resolve() for a in sys.argv[1:]))
