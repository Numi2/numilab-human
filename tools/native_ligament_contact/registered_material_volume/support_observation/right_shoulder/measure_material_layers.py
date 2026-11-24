"""Sum physical skin/fat cell intersections; split edge-touching boundary chunks if needed."""
from pathlib import Path
import json,sys
import numpy as np
from manifold3d import Mesh64,Manifold,Error
from build_carpal_cavity import boundary
from measure_assembly_contacts import surface,canonical
from compile_connected_joint import digest
from prepare_shoulder import MOVING,read


def intersect_cells(points,tet,ids,anatomy,counts):
    if not len(ids):return 0.
    cells=tet[ids];f=boundary(cells);ix,r=np.unique(f,return_inverse=True)
    # Topology uses material-node identities. Coordinate-only welding could collapse independent fans.
    shape=Manifold(Mesh64(np.ascontiguousarray(points[ix]),np.ascontiguousarray(r.reshape(-1,3),dtype=np.uint64)))
    v=points[cells];total=np.linalg.det(v[:,1:]-v[:,:1]).sum()/6.
    if total<=0:raise ValueError('Unoriented material')
    if shape.status()==Error.NoError:
        if shape.volume()<0:shape=Manifold(Mesh64(np.ascontiguousarray(points[ix]),np.ascontiguousarray(r.reshape(-1,3)[:,[0,2,1]],dtype=np.uint64)))
        if abs(shape.volume()-total)<max(1e-8,total*1e-8):
            z=anatomy^shape
            if z.status()==Error.NoError and z.volume()>=-1e-8:
                counts[0]+=1
                return max(z.volume(),0)
    if len(ids)==1:raise ValueError('Unmeasurable physical material cell')
    mid=v.mean(1);axis=np.ptp(mid,axis=0).argmax();order=np.argsort(mid[:,axis],kind='stable')
    counts[1]+=1
    return sum(intersect_cells(points,tet,ids[q],anatomy,counts)for q in np.array_split(order,2))


def layers(j,reg,fix,out):
    vdir=j/'conditioned/witness-material/volume';native=j/'posterior-interface-082/native-pair/physical-accepted.node';org=np.array(read(reg/'registration.json')['origin'])
    pts=(np.loadtxt(native,skiprows=1,usecols=(1,2,3))-org)*1000.
    raw=np.loadtxt(vdir/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=raw[:,:4]-1;role=raw[:,4];lo=[];hi=[]
    for sub in np.array_split(t,64):
        p=pts[sub];lo.append(p.min(1));hi.append(p.max(1))
    low=np.concatenate(lo);high=np.concatenate(hi);rows=[]
    for k in MOVING:
        d=np.load(fix/f'bank-{k}.npz')
        for name,(x,f) in [('before',surface(reg/f'mesh-{k}/source.poly')),('after',((d['native']-org)*1000,d['f']))]:
            muscle,meta=canonical(x,f)
            if muscle is None:raise ValueError('No material volume')
            candidate=np.all(high>=x.min(0),axis=1)&np.all(low<=x.max(0),axis=1)
            for material in (15,16):
                ids=np.flatnonzero(candidate&(role==material));counter=[0,0]
                val=intersect_cells(pts,t,ids,muscle,counter)
                rows.append(dict(muscle=k,stage=name,material=material,tetrahedra=len(ids),solid_partitions=counter[0],subdivisions=counter[1],mm3=val))
        print(k,[(r['material'],round(r['mm3'],4))for r in rows[-4:]],flush=True)
    out.write_text(json.dumps(dict(physical_nodes=digest(native),physical_cells=digest(vdir/'mesh.ele'),rows=rows),indent=2,allow_nan=False)+'\n')
if __name__=='__main__':layers(*(Path(a).resolve() for a in sys.argv[1:]))
