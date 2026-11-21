"""Compare source triangle material sites in the atlas and the native body frames."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile

import numpy as np

from audit_solids import winding
from shoulder_inputs import load, transforms


def closest_distance(tri,p):
    a,b,c=tri[:,0],tri[:,1],tri[:,2]
    u,v=b-a,c-a
    normal=np.cross(u,v)
    square=np.einsum("ij,ij->i",normal,normal)
    h=np.einsum("ij,ij->i",p-a,normal)/square
    proj=p-h[:,None]*normal
    inner=np.ones(len(tri),bool)
    edges=np.full(len(tri),np.inf)
    for i in range(3):
        base=tri[:,i]; e=tri[:,(i+1)%3]-base
        inner &= np.einsum("ij,ij->i",np.cross(e,proj-base),normal)>=0
        t=np.einsum("ij,ij->i",p-base,e)/np.einsum("ij,ij->i",e,e)
        edges=np.minimum(edges,np.linalg.norm(p-base-np.clip(t,0,1)[:,None]*e,axis=1))
    best=np.minimum(edges,np.where(inner,np.abs(h)*np.sqrt(square),np.inf))
    j=int(np.argmin(best))
    return float(best[j]),j


def vertices_and_faces(raw):
    v=[]; f=[]
    for line in raw.splitlines():
        fields=line.split()
        if not fields:continue
        if fields[0]==b'v':v.append([float(x) for x in fields[1:4]])
        if fields[0]==b'f':
            ids=[int(c.split(b'/')[0])-1 for c in fields[1:]]
            for k in range(1,len(ids)-1): f.append([ids[0],ids[k],ids[k+1]])
    return np.asarray(v)*.001,np.asarray(f)


def geometry_audit(launch,registration,motion,archive):
    cfg,catalog,meshes,anchors,poses=load(launch,registration,motion)
    manifest=Path(cfg['argv'][3]).with_suffix('.manifest.json')
    doc=json.loads(manifest.read_text())
    members={a['core_body_index']:a for a in doc['source']['anchors'] if a['member_id'] in ('FJ3384','FJ3368','FJ3279','FJ3262')}
    assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==doc['source']['bodyparts']['archives'][0]['sha256']
    result=[]
    with zipfile.ZipFile(archive) as z:
        for row in anchors['candidates']:
            site=[]; tri=[]; rest=[]; names=[]; err=[]
            bodies=[catalog[s]['body'] for s in row['stable_ids']]
            for s,b,face,w in zip(row['stable_ids'],bodies,row['triangle_rows'],row['material_barycentric']):
                a=members[b]; n=a['member_id']; raw=z.read(next(p for p in z.namelist() if p.endswith('/'+n+'.obj')))
                assert hashlib.sha256(raw).hexdigest()==a['member_sha256']
                v,f=vertices_and_faces(raw); local,fs,*_=meshes[s]
                assert np.array_equal(f,fs),(n,f.shape,fs.shape)
                rest.append(v[f]);tri.append(local[fs]);site.append(np.asarray(w)@local[fs[face]])
                names.append(n)
                homogeneous=np.column_stack([v,np.ones(len(v))])
                fit=np.linalg.lstsq(homogeneous,local,rcond=None)[0]
                err.append(float(np.max(np.linalg.norm(homogeneous@fit-local,axis=1))))
            origin=np.array(row['material_barycentric'][0])@rest[0][row['triangle_rows'][0]]
            atlas_depth,atlas_face=closest_distance(rest[1],origin)
            values=[]
            for step in poses:
                tr,rot=transforms(poses[step],bodies[1],bodies[0]); point=tr+rot.apply(site[0])
                depth,nearest=closest_distance(tri[1],point)
                values.append(dict(step=step,winding=winding(tri[1],point),distance_m=depth,nearest_humeral_face=nearest))
            result.append(dict(side=row['side'],members=names,atlas_native_fit_max_residual_m=err,
                 source_origin_in_source_humerus=winding(rest[1],origin),source_origin_to_humerus_m=atlas_depth,
                 source_nearest_humeral_face=atlas_face,poses=values))
    return result


if __name__=='__main__':
    *inputs,out=map(Path,sys.argv[1:])
    out.write_text(json.dumps(geometry_audit(*inputs),indent=2,allow_nan=False)+'\n')
