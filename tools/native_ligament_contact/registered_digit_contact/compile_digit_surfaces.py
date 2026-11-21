"""Emit source-registered cortical triangles into the existing compact Metal buffers."""
import json,sys,struct,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_digit_contacts import sha

def compile_surfaces(control,out):
    cfg=json.loads((control/'launch.json').read_text())
    bone=Path(next(a for a in cfg['argv'] if a.endswith('.nhbones')))
    if sha(bone)!=cfg['pins'][str(bone)]:raise ValueError('Changed cortical payload')
    reg=json.loads((out/'registration.json').read_text());base=json.loads((control/'initial-cortical.json').read_text())
    if base['cortical_payload_sha256']!=reg['bone_sha256'] or sha(bone)!=reg['bone_sha256']:raise ValueError('Digit registration differs')
    raw=bone.read_bytes();header=struct.Struct('<8s5I32s');record=struct.Struct('<6I8fI')
    magic,abi,nb,nv,ni,_,_=header.unpack_from(raw);start=header.size+nb*record.size
    if magic!=b'NHBONES1' or abi!=3 or len(raw)!=start+nv*24+ni*4:raise ValueError('Unknown cortical layout')
    verts=np.frombuffer(raw,'<f4',count=nv*6,offset=start).reshape(-1,6)[:,:3]
    indices=np.frombuffer(raw,'<u4',count=ni,offset=start+nv*24)
    targets=reg['rows'];ids={int(i) for r in targets for i in r['stable']}
    banks={};bodies={};data=np.load(control/'initial-cortical.npz')
    poses={p['body_index']:p for p in json.loads((control/'step-0.receipt.json').read_text())['accepted_body_poses']}
    for id_ in sorted(ids):
        body,v0,n,f0,nf,k,*tail=record.unpack_from(raw,header.size+(id_-1)*record.size)
        name,srcbody,count,_=base['member_by_stable_id'][str(id_)]
        if k!=id_ or nf!=count*3 or srcbody!=body:raise ValueError('Source face ownership differs')
        f=indices[f0:f0+nf].reshape(-1,3);used=np.unique(f);fi=np.searchsorted(used,f)
        if not np.array_equal(fi,data[name+'_f']):raise ValueError('Source ancestry changed')
        x=(Rotation.from_quat(tail[3:7]).apply(verts[used].astype(float)*tail[7])+tail[:3]).astype('<f4')
        p=poses[body];y=Rotation.from_quat(p['quaternion_xyzw']).apply(x)+p['position_m']
        if np.linalg.norm(y-data[name+'_v'],axis=1).max()>2e-6:raise ValueError('Source frame disagrees with emitted geometry')
        tri=x[fi]
        if (np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)<1.e-14).any():raise ValueError('Degenerate facet')
        banks[k]=tri;bodies[k]=body
    tris=[];dims=[];meta=[]
    for offset in [0,2]:
        a,b=targets[offset]['stable'];A=banks[a];B=banks[b]
        # Two disjoint source-face partitions retain all triangles and supply independent rows.
        center=A.mean(1);axis=np.argmax(np.ptp(center,axis=0));order=np.argsort(center[:,axis],kind='stable')
        b0=len(tris);tris.extend(B)
        for split in np.array_split(order,2):
            a0=len(tris);tris.extend(A[split]);dims.append((a0,len(split),b0,len(B)))
            meta.append(dict(stable=[a,b],body=[bodies[a],bodies[b]],first_source_face_ids=split.tolist(),second_source_face_count=len(B)))
    tri=np.asarray(tris,dtype='<f4');gpu=np.zeros((len(tri),3,4),dtype='<f4');gpu[:,:,:3]=tri
    file=out/'proximal-surface.bin';file.write_bytes(np.array(dims,dtype='<u4').tobytes()+gpu.tobytes())
    (out/'proximal-surface.json').write_text(json.dumps(dict(bone_sha256=sha(bone),sha256=sha(file),rows=meta),indent=2)+'\n')
if __name__=='__main__':compile_surfaces(*map(Path,sys.argv[1:]))
