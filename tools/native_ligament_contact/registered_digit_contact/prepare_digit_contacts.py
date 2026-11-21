"""Bind measured cortical features to the original NHBONES triangles and body frames."""
import hashlib,json,struct,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(run,out):
    cfg=json.loads((run/'launch.json').read_text());r=json.loads((run/'step-0.receipt.json').read_text())
    bone=Path(next(a for a in cfg['argv'] if a.endswith('.nhbones')));m=json.loads(bone.with_suffix('.manifest.json').read_text())
    if sha(bone)!=cfg['pins'][str(bone)] or sha(bone)!=m['payload']['sha256']:raise ValueError('Cortical identity mismatch')
    raw=bone.read_bytes();h=struct.Struct('<8s5I32s');rec=struct.Struct('<6I8fI')
    magic,abi,nb,nv,ni,*_=h.unpack_from(raw);assert magic==b'NHBONES1' and abi==3
    vo=h.size+nb*rec.size;v=np.frombuffer(raw,'<f4',count=nv*6,offset=vo).reshape(-1,6)[:,:3]
    idx=np.frombuffer(raw,'<u4',count=ni,offset=vo+nv*24)
    poses={row['body_index']:row for row in r['accepted_body_poses']};a=np.load(run/'initial-cortical.npz')
    meshes={};maxerr=0.
    for i,anchor in enumerate(m['source']['anchors']):
        name=anchor['myosim_body']
        if name not in ['4proxph_r','5proxph_r','4proxph_l','5proxph_l']:continue
        body,first,n,f0,nf,stable,*tail=rec.unpack_from(raw,h.size+i*rec.size);assert body==anchor['core_body_index'] and stable==i+1 and tail[-1]==anchor['source_record_index']
        faces=idx[f0:f0+nf].reshape(-1,3);used=np.unique(faces);tri=np.searchsorted(used,faces)
        assert np.array_equal(tri,a[name+'_f']);t,q,scale=tail[:3],tail[3:7],tail[7]
        local=(Rotation.from_quat(q).apply(v[used].astype(float)*scale)+t).astype('<f4')
        pose=poses[body];world=Rotation.from_quat(pose['quaternion_xyzw']).apply(local)+pose['position_m']
        err=float(np.linalg.norm(world-a[name+'_v'],axis=1).max());maxerr=max(maxerr,err)
        if err>2e-6:raise ValueError(('Cortical source pose does not match native',name,err))
        meshes[name]=body,stable,local,world,tri
    records=[];rows=[]
    for side,kind,ia,ib in [('r','EE',[108,110],[9,27]),('r','EE',[110,112],[9,27]),('l','VF',[8],[103,136,101]),('l','VF',[8],[136,103,137])]:
        A=meshes['4proxph_'+side];B=meshes['5proxph_'+side]
        def verify_feature(indices,faces):
            if not np.any(np.all(np.isin(indices,faces),axis=-1)) and len(indices)==1:raise ValueError('Unreferenced vertex')
            if not any(set(indices)<=set(f) for f in faces):raise ValueError('Feature absent from emitted cortex')
        verify_feature(ia,A[4]);verify_feature(ib,B[4])
        if kind=='EE':normal=np.cross(A[3][ia[1]]-A[3][ia[0]],B[3][ib[1]]-B[3][ib[0]])
        else:normal=np.cross(B[3][ib[1]]-B[3][ib[0]],B[3][ib[2]]-B[3][ib[0]])
        n=normal/np.linalg.norm(normal);sign=1 if n@(A[3][ia[0]]-B[3][ib[0]])>0 else -1
        # Use fixed original material coordinates; the GPU supplies each current transform.
        xyz=np.vstack([A[2][ia],B[2][ib]])
        rows.append(dict(side=side,kind=kind,body=[A[0],B[0]],stable=[A[1],B[1]],native_vertex_ids=[ia,ib],sign=sign))
        records.append(struct.pack('<4I',A[0],B[0],int(kind=='VF'),int(sign<0))+np.column_stack((xyz,np.zeros(4))).astype('<f4').tobytes())
    payload=b''.join(records);out.mkdir(exist_ok=True)
    (out/'digit-cortical-features.bin').write_bytes(payload)
    (out/'registration.json').write_text(json.dumps(dict(bone_sha256=sha(bone),payload_sha256=hashlib.sha256(payload).hexdigest(),source_capture=r['pack_file_sha256'],maximum_native_vertex_error_m=maxerr,rows=rows),indent=2)+'\n')
if __name__=='__main__':prepare(*map(Path,sys.argv[1:]))
