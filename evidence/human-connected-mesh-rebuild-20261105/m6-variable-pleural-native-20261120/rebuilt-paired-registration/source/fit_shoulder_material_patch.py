"""Fit one pair of shoulder material attachments in both recorded joint postures."""
import json
from pathlib import Path

import numpy as np

from transport_shoulder_attachments import ref, norm, actual, normal, seeds, stable, crossings, ag

weights=np.asarray([(a,b,15-a-b) for a in range(1,14) for b in range(1,15-a)])/15.

def sample(sid, reference_normals, moved_normals, seed, room):
    vertices,faces=ref[sid]
    center=np.asarray(ag.project(tuple(seed),ref[sid])[0])
    triangles=vertices[faces]
    ids=np.flatnonzero(np.all(triangles.min(axis=1)<=center+room,axis=1)&
                       np.all(triangles.max(axis=1)>=center-room,axis=1))
    if not len(ids):
        raise ValueError(f'no source landmark triangles in patch {sid}')
    points=np.einsum('ij,j->i',triangles[ids,:,0],weights[0])
    flat=np.einsum('ba,fad->fbd',weights,triangles[ids]).reshape(-1,3)
    radius=np.linalg.norm(flat-center,axis=1)
    rows=np.flatnonzero(radius<=room)
    rows=rows[np.argsort(radius[rows],kind='stable')]
    _,keep=np.unique(np.floor(flat[rows]/0.0003).astype(int),axis=0,return_index=True)
    rows=rows[np.sort(keep)[:950]]
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

candidate=None
stages=[]
for room in (.003,.005,.007):
    a,na,aa,naa,f,fa,da=sample(stable[0],norm[0],normal[0],seeds[0],room)
    b,nb,bb,nbb,g,gb,db=sample(stable[1],norm[1],normal[1],seeds[1],room)
    l,x,y=metrics(a,b,na,nb)
    ll,xx,yy=metrics(aa,bb,naa,nbb)
    take=np.argwhere((x>.035)&(y>.035)&(xx>.035)&(yy>.035)&(ll>0.005)&(ll<0.030))
    cost=da[take[:,0]]**2+db[take[:,1]]**2
    stages.append(dict(radius_m=room,pairs_with_outgoing_endpoints=len(take)))
    for i,j in take[np.argsort(cost,kind='stable')[:1400]]:
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
Path('settled-shoulder-material-patch.json').write_text(json.dumps(final,indent=2)+'\n')
print(json.dumps(final))
