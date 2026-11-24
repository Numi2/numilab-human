import csv,json
from pathlib import Path
from itertools import combinations
import numpy as np
from bind_gluteal_receivers import digest
import sys


def inspect(v, native, previous, out):
    topology=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype=int)
    t=topology[:,:4]-1
    ident=json.loads((previous/'force-observer/identity.json').read_text())
    search=json.loads((previous/'search-measurement.json').read_text())
    # Use the actual four largest changes in the measured star.
    deltas=np.array([np.subtract(r['stages'][0],r['stages'][1]) for r in search['native_directions_m_per_s']])
    chosen=np.array([r['global_node'] for r in search['native_directions_m_per_s']])[np.argsort(np.linalg.norm(deltas,axis=1))[-4:]]-ident['offset']
    cells=np.flatnonzero(np.isin(t,chosen).all(1))
    if len(cells)!=1:raise ValueError('No unique common tetrahedron')
    i=cells.item()
    near=np.flatnonzero(np.isin(t,chosen).any(1))
    verts=np.unique(t[near]);ring=np.flatnonzero(np.isin(t,verts).any(1))
    ids=np.unique(t[ring]); faces=np.load(v/'material-walls.npz')
    points=[np.loadtxt(p,skiprows=1,usecols=(1,2,3))[ids] for p in (v/'mesh.node',v/'current.node',native/'physical-accepted.node')]
    labels=np.loadtxt(v/'current-child.labels.csv',skiprows=1,delimiter=',',dtype=np.int64)[ring]
    wall={key:faces[key][np.isin(faces[key],verts).any(1)] for key in 'OIDC'}
    mass=np.fromfile(v/'cell-mass.f64','<f8')[ring]
    np.savez_compressed(out/'cavity.npz',tet=t[ring],rows=ring,local=np.searchsorted(ids,t[ring]),ids=ids,
        mass=mass,xyz=points,labels=labels,selected=i,neighbors=near,**wall)
    edge=lambda x,c:x[c[1:]]-x[c[0]]
    tet=t[i]; c=np.searchsorted(ids,tet);m=labels[np.searchsorted(ring,i)]
    around=[]
    for f in combinations(tet,3):
        together=np.flatnonzero(np.isin(t[ring],f).sum(1)==3)
        around.append(dict(face=list(map(int,f)),wall=[k for k,v in wall.items()if np.isin(v,f).all(1).any()],
                cells=ring[together].tolist(),labels=labels[together].tolist()))
    data=dict(registered_row=i,source_tet_1_based=topology[i,:4].tolist(),labels=m.tolist(),
        vertices_wall_memberships={str(int(n)):[k for k,v in wall.items()if (v==n).any()] for n in tet},
        faces=around,source_reference=digest(v/'mesh.node'),topology=digest(v/'mesh.ele'),
        source_mass_kg=mass[np.searchsorted(ring,i)],poses=[])
    for name,x in zip(('reference','loaded','first_accepted'),points):
        e=edge(x,c);s=np.linalg.svd(e,compute_uv=False);grad=np.linalg.inv(e)
        grad=np.c_[-grad.sum(1),grad];vol=abs(np.linalg.det(e))/6
        data['poses'].append(dict(pose=name,volume_m3=vol,singular_values_m=s.tolist(),
               min_altitude_m=1/np.linalg.norm(grad,axis=0).max(),J=np.linalg.det(e)/np.linalg.det(edge(points[0],c))))
    (out/'cavity.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':inspect(*(Path(s) for s in sys.argv[1:]))
