"""Fit one pair of shoulder material attachments in both recorded joint postures."""
import json
from pathlib import Path

import numpy as np

import os
import sys
from pathlib import Path
work=Path.cwd()
prior=work.parent/"resting-clearance-and-atrium-010"
sys.path.insert(0,str(prior))
os.chdir(prior)
from transport_shoulder_attachments import ref, norm, actual, normal, seeds, stable, crossings, ag
os.chdir(work)

weights=np.asarray([(a,b,9-a-b) for a in range(1,8) for b in range(1,9-a)])/9.

def sample(sid, reference_normals, moved_normals, seed, room):
    vertices,faces=ref[sid]
    center=np.asarray(ag.project(tuple(seed),ref[sid])[0])
    triangles=vertices[faces]
    ids=np.flatnonzero(np.all(triangles.min(axis=1)<=center+room,axis=1)&
                       np.all(triangles.max(axis=1)>=center-room,axis=1))
    if not len(ids):
        raise ValueError(f'no source landmark triangles in patch {sid}')
    flat=np.einsum('ba,fad->fbd',weights,triangles[ids]).reshape(-1,3)
    radius=np.linalg.norm(flat-center,axis=1)
    rows=np.flatnonzero(radius<=room)
    rows=rows[np.argsort(radius[rows],kind='stable')]
    _,keep=np.unique(np.floor(flat[rows]/0.00050).astype(int),axis=0,return_index=True)
    rows=rows[np.sort(keep)]
    f=ids[rows//len(weights)]
    w=weights[rows%len(weights)]
    def interpolate(values):
        return np.einsum('ra,rad->rd',w,values[faces[f]])
    def normals(values):
        v=interpolate(values)
        return v/np.linalg.norm(v,axis=1)[:,None]
    return flat[rows],normals(reference_normals),interpolate(actual[sid][0]),normals(moved_normals),f,w,radius[rows]

def metrics(a,b,na,nb):
    vector=b[None]-a[:,None]
    length=np.linalg.norm(vector,axis=2)
    vector/=np.maximum(length,1e-12)[:,:,None]
    return length,np.einsum('ik,ijk->ij',na,vector),-np.einsum('jk,ijk->ij',nb,vector)

def cleared(first,second,meshes):
    lo=np.minimum(first.min(axis=0),second.min(axis=0))-1e-9
    hi=np.maximum(first.max(axis=0),second.max(axis=0))+1e-9
    relevant=[]
    for vertex,faces in meshes.values():
        if np.any(vertex.max(axis=0)<lo) or np.any(vertex.min(axis=0)>hi):
            continue
        tri=vertex[faces]
        relevant.append(tri[(np.all(tri.max(axis=1)>=lo,axis=1))&(np.all(tri.min(axis=1)<=hi,axis=1))])
    tri=np.concatenate(relevant,axis=0)
    a,b,c=np.moveaxis(tri,1,0)
    e1=b-a
    e2=c-a
    scale0=np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)
    ok=np.ones(len(first),dtype=bool)
    for start in range(0,len(first),128):
        sl=slice(start,start+128)
        diff=second[sl]-first[sl]
        h=np.cross(diff[:,None],e2[None])
        det=np.einsum('ij,kij->ki',e1,h)
        scale=np.linalg.norm(diff,axis=1)[:,None]*scale0[None]
        admitted=np.abs(det)>32*np.finfo(np.float64).eps*scale
        inv=np.divide(1.,det,out=np.zeros_like(det),where=admitted)
        s=first[sl,None]-a[None]
        u=inv*np.sum(s*h,axis=2)
        q=np.cross(s,e1[None])
        v=inv*np.einsum('kj,kij->ki',diff,q)
        fraction=inv*np.einsum('ij,kij->ki',e2,q)
        reached=admitted&(u>=-1e-8)&(u<=1+1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(fraction>1e-5)&(fraction<1-1e-5)
        ok[sl]=~np.any(reached,axis=1)
    return ok

candidate=None
stages=[]
for room in (.005,.007,.009):
    a,na,aa,naa,f,fa,da=sample(stable[0],norm[0],normal[0],seeds[0],room)
    b,nb,bb,nbb,g,gb,db=sample(stable[1],norm[1],normal[1],seeds[1],room)
    l,x,y=metrics(a,b,na,nb)
    ll,xx,yy=metrics(aa,bb,naa,nbb)
    take=np.argwhere((x>.015)&(y>.015)&(xx>.015)&(yy>.015)&(ll>0.005)&(ll<0.030))
    cost=da[take[:,0]]**2+db[take[:,1]]**2
    stages.append(dict(radius_m=room,pairs_with_outgoing_endpoints=len(take),origin_samples=len(a),insertion_samples=len(b)))
    ordered=take[np.argsort(cost,kind='stable')]
    if len(ordered):
        i,j=ordered.T
        through_reference=cleared(a[i],b[j],ref)
        stages[-1]['reference_clear']=int(through_reference.sum())
        subset=ordered[through_reference]
        i,j=subset.T
        if len(i):
            subset=subset[cleared(aa[i],bb[j],actual)]
        stages[-1]['clear_in_both']=len(subset)
    else:
        subset=ordered
    for i,j in subset:
        if crossings(a[i],b[j],ref) or crossings(aa[i],bb[j],actual):
            continue
        if [ag.project(tuple(p),actual[sid])[2] for p,sid in zip((aa[i],bb[j]),stable)]!=[f[i],g[j]]:
            continue
        candidate=dict(name='superior_glenohumeral_capsuloligament',side=0,stable_ids=stable,
                       seeds_m=[aa[i].tolist(),bb[j].tolist()],
                       current_source_face_rows=[int(f[i]),int(g[j])],
                       same_material_point_barycentric=[fa[i].tolist(),gb[j].tolist()],
                       reference_offsets_m=[float(da[i]),float(db[j])],
                       endpoint_facing_cosine=[float(xx[i,j]),float(yy[i,j])],
                       earlier_facing_cosine=[float(x[i,j]),float(y[i,j])],
                       candidate_length_m=float(ll[i,j]),earlier_length_m=float(l[i,j]))
        break
    if candidate:
        break
final=dict(stages=stages,candidate=candidate,qualified_in_both_poses=candidate is not None,
           qualified_through_motion=False)
Path('shoulder-region-search.json').write_text(json.dumps(final,indent=2)+'\n')
print(json.dumps(final))
