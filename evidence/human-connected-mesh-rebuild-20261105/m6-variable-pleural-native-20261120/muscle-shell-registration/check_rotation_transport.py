"""Compare the existing weighted native registration with rigid quaternion transport."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

here=Path(__file__).resolve().parent
w=here.parent
sys.path[:0]=[str(w/'resting-clearance-and-atrium-006'),
              str(w/'resting-surface-clearance-001/src')]
from register_tissue_bellies import (CASES, source, blend, quality, clear,
                                    exact, exact_skin_pair, install_for_float32)

def product(a,b):
    av,aw=a[...,:3],a[...,3,None]
    bv,bw=b[...,:3],b[...,3,None]
    return np.concatenate((aw*bv+bw*av+np.cross(av,bv),
        aw*bw-np.sum(av*bv,axis=-1,keepdims=True)),axis=-1)

def conjugate(q):
    return q*np.array([-1.,-1.,-1.,1.])

def dqs(bank,bodies):
    real,dual,scale=[],[],[]
    for binding in bank['bindings']:
        body=bodies[binding[0]]
        base=Rotation.from_quat(body['quaternion_xyzw'])
        trans=base.apply(binding[1:4])+body['position_m']
        q=(base*Rotation.from_quat(binding[4:8])).as_quat()
        real.append(q)
        dual.append(.5*product(np.r_[trans,0.],q))
        scale.append(binding[8])
    assert max(scale)-min(scale)<1.e-6
    ids,weights=bank['ids'],bank['weights']
    real,dual=np.asarray(real)[ids],np.asarray(dual)[ids]
    sign=np.where(np.einsum('vki,vi->vk',real,real[np.arange(len(ids)),weights.argmax(axis=1)])<0,-1.,1.)
    signed_weight=sign*weights
    q=np.einsum('vk,vki->vi',signed_weight,real)
    qd=np.einsum('vk,vki->vi',signed_weight,dual)
    norm=np.linalg.norm(q,axis=1)
    assert np.all(np.isfinite(norm)) and np.min(norm)>.10
    q/=norm[:,None]
    qd/=norm[:,None]
    local=bank['points']*scale[0]
    return Rotation.from_quat(q).apply(local)+2.*product(qd,conjugate(q))[:,:3]

data=np.load(here/'paired-geometry.npz')
obs=json.loads((here/'registration-observation.json').read_text())
base=Path(obs['source_path'])
assert hashlib.sha256(base.read_bytes()).hexdigest()==obs['source_sha256']
banks=source(base)
v=w/'resting-clearance-and-atrium-017/floor-native-live-parent-026/native/accepted-geometry/step-16.mrvpack'
pose_paths=(v,CASES[-1][0])
install_for_float32()
out={}
eligible=[str(i) for i in (73,74,76,109,110,120,137,138,150,164)]
for pose,path in enumerate(pose_paths):
    receipt=json.loads(path.with_suffix('.receipt.json').read_text())
    assert receipt['pack_file_sha256']==obs['poses'][pose]['pack_sha256']
    bodies={row['body_index']:row for row in receipt['accepted_body_poses']}
    skin=data[f'pose{pose}_skin_v'],data[f'pose{pose}_skin_f']
    records=clear._exact_surface_records(*skin)
    index=exact._prepare_surface_aabb(records)
    float_index=clear._float32_skin_aabb_index(records)
    for name in eligible:
        if max(p['banks'][name]['max_registration_error_m'] for p in obs['poses'])>.00012:
            continue
        sid=int(name)
        b=banks[sid]
        scales=[a[8] for a in b['bindings']]
        if max(scales)-min(scales)>=1.e-6:
            continue
        world=data[f'pose{pose}_b{sid}_v']
        before=set(map(tuple,data[f'pose{pose}_b{sid}_hits']))
        faces=b['source_faces']
        original=blend(b,bodies)[0]
        displacement=dqs(b,bodies)-original
        source_delta=np.linalg.solve(data[f'pose{pose}_b{sid}_j'].astype(float),
                                     displacement[...,None])[...,0]
        per_fraction=[]
        for fraction in (.25,.5,1.):
            moved=(world+fraction*displacement).astype(np.float32)
            q,volume_delta=quality(world,moved,faces)
            if q<=0:
                per_fraction.append(dict(fraction=fraction,minimum_oriented_area_ratio=q,
                                         volume_delta=volume_delta,rejected_geometry=True))
                continue
            pairs=exact_skin_pair(index,float_index,moved,faces)
            new=pairs-before
            original_skin_face=np.asarray(sorted({a for a,_ in before}),dtype=int)
            if len(original_skin_face):
                neighborhood=set(skin[1][original_skin_face].ravel())
                beyond=sum(not neighborhood.intersection(skin[1][a]) for a,_ in new)
            else:
                beyond=len(new)
            per_fraction.append(dict(fraction=fraction,before=len(before),after=len(pairs),
                new_pairs=len(new),new_pairs_outside_skin_vertex_ring=beyond,
                minimum_oriented_area_ratio=q,volume_delta=volume_delta))
        evidence=dict(pose=pose,step=receipt['accepted_step'],bank=sid,
                      label=obs['poses'][pose]['banks'][name]['label'],
                      maximum_native_displacement_m=float(np.linalg.norm(displacement,axis=1).max()),
                      maximum_source_equivalent_displacement_m=float(np.linalg.norm(source_delta,axis=1).max()),
                      fractions=per_fraction)
        out.setdefault(name,[]).append(evidence)
        print(json.dumps(evidence),flush=True)
(here/'rotation-transport-screen.json').write_text(json.dumps(out,indent=2)+'\n')
