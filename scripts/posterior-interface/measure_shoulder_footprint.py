"""Clip the registered bed-facing triangles to the real plane and integrate material parent weights."""
import json
from pathlib import Path
import sys

import numpy as np
from compile_connected_joint import digest


def clipped_face(x, z, n):
    rows=[]
    for i in range(3):
        prev=(i+2)%3
        if (z[prev]<0)!=(z[i]<0):
            a=z[prev]/(z[prev]-z[i])
            rows.append((1-a)*np.eye(3)[prev]+a*np.eye(3)[i])
        if z[i]<=0:rows.append(np.eye(3)[i])
    if len(rows)<3:return 0.,np.zeros(3),np.zeros(3)
    poly=np.array(rows)
    bary=np.zeros(3);moment=np.zeros(3);area=0.
    for j in range(1,len(rows)-1):
        weights=poly[[0,j,j+1]]
        tri=weights@x
        a=.5*abs(np.dot(n,np.cross(tri[1]-tri[0],tri[2]-tri[0])))
        area+=a
        bary+=a*weights.mean(0)
        moment+=a*tri.mean(0)
    if abs(bary.sum()-area)>1e-12 or np.linalg.norm(bary@x-moment)>1e-12:
        raise ValueError('Area or first moment lost during clipping')
    return area,bary,moment


def measure(case,material,shoulder,out):
    sup=json.loads((case/'support-check.json').read_bytes())
    ref=json.loads(shoulder.read_bytes());n=np.array(sup['bed_normal']);offset=sup['source_plane_offset_m']
    x=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    walls=np.load(material/'material-walls.npz')
    nd=ref['dominant_skin_node'];f=walls['O'][np.any(walls['O']==nd,axis=1)]
    q=x[f]; gaps=q@n-offset
    near=np.linalg.norm(x[np.setdiff1d(f,nd)]-x[nd],axis=1)
    areas=.5*np.abs(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])@n)
    table=[]
    for advance in (0.,.0001,.00025,.0005,.001,.002,.005):
        covered=0.;moment=np.zeros(3);nodal={}
        for vertices,tri,gap in zip(f,q,gaps):
            a,b,m=clipped_face(tri,gap-advance,n);covered+=a;moment+=m
            for v,w in zip(vertices,b):
                nodal[int(v)]=nodal.get(int(v),0.)+w
        table.append(dict(plane_advance_m=advance,projected_contact_area_m2=covered,
            projected_face_fraction=covered/areas.sum(),center_world_m=(moment/covered).tolist() if covered else None,
            integrated_basis_m2=nodal))
    out.write_text(json.dumps(dict(accepted_frame=digest(case/'physical-accepted.node'),load_history=ref['history'],
        source_plane=digest(case/'support-check.json'),node=nd,one_ring_cells=len(f),edge_length_m=near.tolist(),
        one_ring_face_nodes=f.tolist(),native_gap_m=float(x[nd]@n-offset),one_ring_projected_area_m2=areas.sum(),
        samples=table,scope='Exact flat-plane clipping in the frozen physical frame. Positive advances are geometric probes. They do not apply an external force.'),indent=2)+'\n')

if __name__=='__main__':measure(*(Path(s).resolve() for s in sys.argv[1:]))
