"""Measure pre-existing support tissue in the entire proposed reference compartment."""
import json
import sys
from pathlib import Path
import numpy as np
import manifold3d as md
from bind_gluteal_receivers import digest, require
from certify_gluteal_volume import patch_mesh, solid
from reconstruct_gluteal_fat import glutes


def run(fitted,prior,assets,out):
    c=dict(np.load(fitted/'registered-columns.npz'))
    patch,x,_=patch_mesh(c)
    cap=json.loads((prior/'coupled/native-traces/accepted-geometry/step-1.receipt.json').read_bytes())
    cfg=json.loads((prior/'coupled/launch.json').read_bytes())
    atlas,owners=glutes(assets,cap,json.loads((prior/'registered-receivers.json').read_bytes()),c['frame'],c['origin'])
    atlas=atlas @ c['frame'].T+c['origin']
    faces_file=assets/'closed-material-triangles.bin'
    surface_faces=np.fromfile(faces_file,'<u4').reshape(-1,3,4)[:,0,:3]
    muscle={}
    for sid in (29,30):
        tri=surface_faces[owners==sid]
        ids, first, ind=np.unique(tri.ravel(),return_index=True,return_inverse=True)
        repeated=atlas[owners==sid].reshape(-1,3)
        points=repeated[first]
        require(np.max(abs(points[ind]-repeated))<1e-14,'Inconsistent material vertex')
        mesh=solid(points,ind.reshape(-1,3))
        intersection=patch^mesh
        require(mesh.status()==md.Error.NoError and mesh.volume()>0 and intersection.status()==md.Error.NoError,'Invalid material solid')
        muscle[sid]=intersection.volume()*1e9
    sources={}
    for k in ('NODES','TETS'):
        key='NUMI_H706_FRACTIONAL_SOURCE330_'+k
        f=Path(cfg['environment'][key])
        require(digest(f)==cfg['environment'][key+'_SHA256'],'Changed baseline tissue source')
        sources[k]=f
    nodes=np.loadtxt(sources['NODES'],skiprows=1,delimiter=',',usecols=(2,3,4,5,6,7))
    table=np.loadtxt(sources['TETS'],skiprows=1,delimiter=',',usecols=(1,3,4,5,6),dtype='i4')
    faces=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
    lo,hi=x.min(0),x.max(0)
    overlap={}
    for key,positions in [('current',nodes[:,:3]),('rest',nodes[:,3:])]:
        geom=positions[table[:,1:]]
        sel=((geom.max(1)>=lo)&(geom.min(1)<=hi)).all(1)
        by_role={}
        for role in np.unique(table[:,0]):
            ids=np.flatnonzero(sel & (table[:,0]==role))
            volumes=[]
            for row in ids:
                v=geom[row]
                direction=np.linalg.det(v[1:]-v[0])
                mesh=solid(v,faces if direction>0 else faces[:,::-1])
                result=patch^mesh
                require(mesh.status()==md.Error.NoError and result.status()==md.Error.NoError,'Bad source tetrahedron')
                volumes.append(result.volume()*1e9)
            by_role[str(role)]={'broadphase_cells':len(ids),'summed_mm3':float(sum(volumes)),'max_mm3':float(max(volumes,default=0.0))}
        overlap[key]=by_role
    entry={'inputs':{k:digest(f) for k,f in sources.items()},'registration':digest(fitted/'registration.json'),
           'source330_mm3':overlap,'physical_gluteal_boundary_mm3':muscle}
    out.write_text(json.dumps(entry,indent=2)+'\n')
    print(overlap,muscle)
if __name__=='__main__':run(*(Path(v)for v in sys.argv[1:]))
