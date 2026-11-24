"""Check the registered fat volume against the accepted anatomy and existing physical tissue."""
import argparse
import json
import mmap
import runpy
from collections import Counter
from pathlib import Path

import manifold3d as md
import numpy as np

from bind_gluteal_receivers import digest, require


def solid(vertices, faces):
    return md.Manifold(md.Mesh64(np.ascontiguousarray(vertices, dtype=np.float64),
                                 np.ascontiguousarray(faces, dtype=np.uint64)))


def patch_mesh(c):
    uv = c['plane_xy']
    xy = uv.reshape(-1,2)
    unique, first, inverse = np.unique(np.rint(xy * 1e11).astype('i8'), axis=0,
                                       return_index=True, return_inverse=True)
    ids = inverse.reshape(-1,3)
    # Numerical duplicates share their original coordinates. No coordinates are
    # snapped to the grouping lattice.
    lo, hi = (np.c_[xy, c[field].ravel()] for field in ('fat_z','muscle_z'))
    anchor = np.r_[lo[first],hi[first]]
    count = len(first)
    require(max(np.max(abs(lo-anchor[inverse])),np.max(abs(hi-anchor[count+inverse]))) < 1e-10,
            'Discontinuous material height')
    signed = np.cross(np.c_[uv[:,1]-uv[:,0],np.zeros(len(uv))],
                      np.c_[uv[:,2]-uv[:,0],np.zeros(len(uv))])[:,2]
    ids[signed < 0] = ids[signed < 0, ::-1]
    directed = np.concatenate((ids[:,[0,1]],ids[:,[1,2]],ids[:,[2,0]]))
    _, edges, counts=np.unique(np.sort(directed,axis=1),axis=0,return_index=True,return_counts=True)
    require(np.all(counts<=2), 'Overlapping projected columns')
    a,b=directed[edges[counts==1]].T
    faces=np.vstack([ids[:,::-1],ids+count,np.column_stack((a,b,b+count)),np.column_stack((a,b+count,a+count))])
    xyz=anchor @ c['frame'].T+c['origin']
    item=solid(xyz,faces)
    require(item.status()==md.Error.NoError and item.volume()>0, 'Patch boundary is not a closed material solid')
    return item,xyz,faces


def clipped_area(points, c, columns):
    """Positive surface area strictly within a material column, for open banks."""
    tri = (points-c['origin']) @ c['frame']
    area=0.
    for i in columns:
        p=tri
        for normal,offset in c['planes'][i]:
            if len(p)<3:
                break
            d=p @ normal-offset
            if np.min(d)>=0:
                continue
            if np.max(d)<=0:
                p=[]
                break
            kept=[]
            for j,v in enumerate(p):
                if d[j]*d[j-1] < 0:
                    kept.append(p[j-1]+d[j-1]/(d[j-1]-d[j])*(v-p[j-1]))
                if d[j]>=0:
                    kept.append(v)
            p=np.asarray(kept)
        if len(p)>=3:
            area+=np.linalg.norm(np.cross(p[1:-1]-p[0],p[2:]-p[0]),axis=1).sum()/2
    return float(area)


def columns_geometry(c):
    uv=c['plane_xy']
    low=np.dstack([uv,c['fat_z']]); high=np.dstack([uv,c['muscle_z']])
    corners=np.concatenate((low,high),axis=1)
    normal=[];offset=[]
    for a,b,d in ((0,1,2),(3,4,5),(0,1,4),(1,2,5),(2,0,3)):
        center=corners.mean(1)
        n=np.cross(corners[:,b]-corners[:,a],corners[:,d]-corners[:,a])
        n/=np.linalg.norm(n,axis=1)[:,None]
        n*=np.sign(np.einsum('ij,ij->i',n,center-corners[:,a]))[:,None]
        normal.append(n);offset.append(np.einsum('ij,ij->i',n,corners[:,a]))
    world=corners @ c['frame'].T+c['origin']
    return list(zip(np.stack(normal,1),np.stack(offset,1))), world.min(1), world.max(1)


def intersect_banks(case, audit, patch, bounds, c):
    receipt=json.loads((case/'native-traces/accepted-geometry/step-1.receipt.json').read_bytes())
    pack=Path(receipt['accepted_pack_path'])
    require(digest(pack)==receipt['pack_file_sha256'], 'Pack identity changed')
    poses={s['body_index']:s for s in receipt['accepted_body_poses']}
    cols,lower,upper=columns_geometry(c)
    c['planes']=[list(zip(a,b)) for a,b in cols]
    rows=[]
    totals=Counter()
    with pack.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as buf:
        for (typ,oid),_,v,tri in runpy.run_path(str(audit))['primitives'](buf,poses):
            totals[int(typ)]+=1
            if np.any(v.max(0)<bounds[0]) or np.any(v.min(0)>bounds[1]):
                continue
            pts=v[tri]
            selected=((pts.max(1)>=bounds[0])&(pts.min(1)<=bounds[1])).all(1)
            unique,ids=np.unique(v,axis=0,return_inverse=True)
            f=ids[tri].reshape(-1,3)
            b=solid(unique,f)
            valid=b.status()==md.Error.NoError and b.volume()>0
            overlap=None
            area=None
            if valid:
                result=patch^b
                require(result.status()==md.Error.NoError, 'Failed Boolean on closed anatomical bank')
                overlap=float(result.volume()*1e9)
            elif selected.any():
                area=0.
                for pt in pts[selected]:
                    candidates=np.flatnonzero(((upper>=pt.min(0))&(lower<=pt.max(0))).all(1))
                    area+=clipped_area(pt,c,candidates)*1e6
            rows.append({'class':int(typ),'id':int(oid),'closed_positive':valid,
                         'aabb_faces':int(selected.sum()),'overlap_mm3':overlap,
                         'open_surface_inside_mm2':area})
    return dict(source=digest(pack),total_by_class=totals,overlapping_bounds=rows)


def existing_tissue(case,volume,patch,bounds):
    x=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    cells=np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype=int)
    found=[]
    for start in range(0,len(cells),32768):
        tet=x[cells[start:start+32768,:4]-1]
        found.extend((start+np.flatnonzero(((tet.max(1)>=bounds[0])&(tet.min(1)<=bounds[1])).all(1))).tolist())
    out={}
    for role in np.unique(cells[found,4]):
        indices=np.array(found)[cells[found,4]==role]
        t=cells[indices,:4]-1
        boundary=t[:,[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]].reshape(-1,3)
        _,ix,counts=np.unique(np.sort(boundary,axis=1),axis=0,return_index=True,return_counts=True)
        require(counts.max()<=2,'Nonmanifold bulk face')
        verts, f=np.unique(boundary[ix[counts==1]],return_inverse=True)
        b=solid(x[verts],f.reshape(-1,3))
        if b.status()==md.Error.NoError and b.volume()>0:
            result=patch^b
            require(result.status()==md.Error.NoError,'Invalid existing volume Boolean')
            measured=result.volume()
        else:
            measured=0.
            for row in t:
                tet=solid(x[row],np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]]))
                require(tet.status()==md.Error.NoError and tet.volume()>0,'Invalid existing FEM cell')
                inter=tet^patch
                require(inter.status()==md.Error.NoError,'Invalid cell intersection')
                measured+=inter.volume()
        out[str(role)]={'cells_in_bounds':len(t),'overlap_mm3':measured*1e9}
    return out


def run(inputs,case,volume,audit):
    info=json.loads((inputs/'registration.json').read_bytes())
    require(info['maximum_relative_area_error']<1e-8 and
            digest(inputs/'registered-columns.npz')==info['columns_sha256'],'Unqualified projected coverage')
    c=dict(np.load(inputs/'registered-columns.npz'))
    patch,x,f=patch_mesh(c)
    require(abs(patch.volume()/info['volume_m3']-1)<1e-8,'Column quadrature and closed volume disagree')
    bounds=(x.min(0),x.max(0))
    report=dict(registration=digest(inputs/'registration.json'),volume_m3=patch.volume(),
                components=len(patch.decompose()),vertices=len(x),boundary_faces=len(f))
    report['skin_fat']=existing_tissue(case,volume,patch,bounds)
    report['banks']=intersect_banks(case,audit,patch,bounds,c)
    (inputs/'anatomical-intersections.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('existing FEM',report['skin_fat'])
    for r in report['banks']['overlapping_bounds']:
        if (r['overlap_mm3'] or 0)>1e-5 or (r['open_surface_inside_mm2'] or 0)>1e-5:
            print(r)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('inputs','case','volume','audit'):
        parser.add_argument(k,type=Path)
    run(**vars(parser.parse_args()))
