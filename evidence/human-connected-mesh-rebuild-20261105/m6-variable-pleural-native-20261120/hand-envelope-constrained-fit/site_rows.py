"""Pick the local outward-facing triangle under each exposed physical route site."""
import json
from pathlib import Path
import sys

import numpy as np

from screen import clearance, skin, skin_faces

def extract(sample_path):
    query=json.loads(Path(sample_path).read_text())["exact_local_ray_parity"]
    outside=[row for row in query if row["location"]=="outside"]
    triangles=skin[skin_faces].astype(float)
    normal=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    norm=np.linalg.norm(normal,axis=1)
    assert (norm>0).all()
    normal*=np.sign(clearance._signed_surface_integral(skin,skin_faces))/norm[:,None]
    rows=[]
    for site in outside:
        p=np.array(site["world_float32_m"])
        nearby=np.flatnonzero(np.all((triangles.min(axis=1)<p+.014)&
                                    (triangles.max(axis=1)>p-.014),axis=1))
        a,b,c=triangles[nearby].transpose(1,0,2)
        for axis in (0,2,1):
            n=normal[nearby,axis]
            mask=np.abs(n)>.35
            locations=nearby[mask]
            face=triangles[locations]
            tangential=[d for d in range(3) if d!=axis]
            ref=face[:,0,tangential]
            v=face[:,1,tangential]-ref
            w=face[:,2,tangential]-ref
            d=p[tangential]-ref
            det=v[:,0]*w[:,1]-v[:,1]*w[:,0]
            u=(d[:,0]*w[:,1]-d[:,1]*w[:,0])/det
            v2=(v[:,0]*d[:,1]-v[:,1]*d[:,0])/det
            bary=np.c_[1-u-v2,u,v2]
            axial=np.sum(bary*face[:,:,axis],axis=1)
            signed_dist=(p[axis]-axial)*np.sign(n[mask])
            good=np.flatnonzero((bary>=0).all(axis=1)&(signed_dist>0)&(signed_dist<.012))
            by_direction=[]
            for direction in (-1,1):
                side=good[np.sign(n[mask][good])==direction]
                if len(side)==0:
                    continue
                best=side[np.argmin(signed_dist[side])]
                idx=locations[best]
                by_direction.append(dict(axis=axis,side=direction,face=int(idx),
                    bary=bary[best].tolist(),point_m=p.tolist(),
                    nodal_rows=skin_faces[idx].tolist(),
                    gap_m=float(signed_dist[best]),normal_alignment=float(abs(n[mask][best])),
                    skin_intercept_m=float(axial[best])))
            rows.append(dict(route=site["route"],site=site["site_order"],axis=axis,
                candidate_faces=by_direction,point_m=p.tolist()))
    return rows

if __name__=="__main__":
    print(json.dumps(extract(sys.argv[2]),indent=2))
