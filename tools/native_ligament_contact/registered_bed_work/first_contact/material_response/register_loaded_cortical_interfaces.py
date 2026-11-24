"""Bind each physical cavity facet to its anatomical rigid owner at the prescribed pose."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from rebind_ligament_material import banks
from classify_connected_boundary import pack
from inspect_shoulder_source import nearest
from compare_supported_postures import poses

def run(h,job,lib,out,pose_file=None):
    pose_file=pose_file or job/'posed-app/current.node'
    sw=np.load(h/'cortical-registration/registered-cortical-facets.npz')
    wr=np.load(job/'wrist/cortical-facets.npz');xyz=np.loadtxt(pose_file,skiprows=1,usecols=(1,2,3))
    ct=sw['cell_side_material_bank_nodes'];c2=np.column_stack((wr['child'],wr['side'],wr['material'],np.full(len(wr['side']),int(wr['source_bone'])),wr['nodes']))
    rows=np.vstack((ct,c2)).astype('i4');z=np.load(job/'boundaries/faces.npz');K=pack(z['face']);ii=np.searchsorted(K,pack(rows[:,4:]))
    if len(np.unique(ii))!=len(ii) or not np.array_equal(K[ii],pack(rows[:,4:])) or not np.array_equal(z['cell'][ii],rows[:,0]) or not np.array_equal(z['side'][ii],rows[:,1]):raise ValueError('Lost exact cortical material incidence')
    initial=job/'distal-registered-native/target-pose.json';p=poses(initial);cfg=json.loads((job/'posed-app/scene/launch.json').read_bytes());bone=Path(cfg['argv'][3])
    if digest(bone)!=cfg['pins'][str(bone)]:raise ValueError('Scene skeleton identity changed')
    B=banks(bone);tors={sid:(body,faces,v) for (body,sid),(faces,v,_) in B.items()}
    weights=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    dst=np.empty((len(rows),3));ids=np.empty((len(rows),3),'i4');bc=np.empty((len(rows),3,3));owner=np.empty(len(rows),'i4');summary={}
    for sid in np.unique(rows[:,3]):
        body,F,v=tors[sid];select=rows[:,3]==sid;owner[select]=body
        world=(Rotation.from_quat(p[body,3:]).apply(v)+p[body,:3]).astype('f4').astype(float)
        query=(weights@xyz[rows[select,4:]]).reshape(-1,3);target,dist,index=nearest((world,F),query,lib);tri=world[F[index]];base=tri[:,0];A=(tri[:,1:]-base[:,None]).transpose(0,2,1)
        uv=np.einsum('ijk,ik->ij',np.linalg.pinv(A),target-base);bary=np.column_stack((1-uv.sum(1),uv))
        if np.min(bary)<-2e-6 or np.any(np.linalg.norm(np.einsum('ij,ijk->ik',bary,tri)-target,axis=1)>2e-8):raise ValueError('Cortical barycentric reconstruction failed')
        dst[select]=dist.reshape(-1,3);ids[select]=index.reshape(-1,3);bc[select]=bary.reshape(-1,3,3)
        summary[int(sid)]=dict(body=body,material_faces=int(select.sum()),distance_m=np.quantile(dist,[0,.5,.95,1]).tolist())
    out.mkdir();file=out/'registered-cortical-facets.npz'
    np.savez_compressed(file,cell_side_material_bank_nodes=rows,body=owner,quadrature_barycentric=weights,cortex_faces=ids,cortex_barycentric=bc,loaded_distance_m=dst)
    result=dict(registered=summary,facets=len(rows),samples=dst.size,loaded_geometry=digest(pose_file),body_pose=digest(initial),bone=digest(bone),source_shoulder=digest(h/'cortical-registration/registered-cortical-facets.npz'),source_wrist=digest(job/'wrist/cortical-facets.npz'),mapping=digest(file))
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result['registered'])
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
