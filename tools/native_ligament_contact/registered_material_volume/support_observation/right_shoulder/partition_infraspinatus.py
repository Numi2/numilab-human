"""Partition a registered muscle against its two actual bone solids, retaining source-face parents."""
import json
from pathlib import Path
import sys
import manifold3d as md
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_material_interface import solid
from rebind_ligament_material import banks
from compile_connected_joint import digest


def partition(case,source,report,out):
    out.mkdir(exist_ok=True)
    z=np.load(source);registered=json.loads(report.read_bytes())
    cfg=json.loads((case/'launch.json').read_bytes());packet=next((case/'native-traces/accepted-geometry').glob('*.receipt.json'))
    frame=json.loads(packet.read_bytes());asset=Path(cfg['argv'][3]);bn=banks(asset)
    if digest(asset)!=registered['bone_payload'] or digest(packet)!=registered['native_pose'] or digest(source)!=registered['volume_registration']:
        raise ValueError('Changed material, capture or cortex')
    stages={}
    for name,poses in [('opening',frame['initial_anatomical_registration']['body_poses']),('accepted',frame['accepted_body_poses'])]:
        by_id={r['body_index']:r for r in poses}
        xyz=z[name];origin=xyz.mean(0);s,ident,_=solid((xyz-origin)*1000,z['boundary'],'muscle83')
        bone_ids={ident:0};cortex={};expected=registered['results'][name]
        if s.status()!=md.Error.NoError or abs(s.volume()-expected['muscle_mm3'])>1.e-6:raise ValueError('Incorrect shoulder volume')
        original=s.volume()
        for r in expected['interactions']:
            if r['overlap_mm3']<=1e-9:continue
            k=r['bank'];body=r['body'];pose=by_id[body];f,v,_=bn[body,k]
            points=(Rotation.from_quat(pose['quaternion_xyzw']).apply(v)+pose['position_m']).astype('f4').astype(float)
            b,bid,_=solid((points-origin)*1000,f,f'cortex{k}');bone_ids[bid]=k;cortex[k]=b
            if b.status()!=md.Error.NoError:raise ValueError('Invalid loaded cortical bank')
            s=s-b
        components=[float(q.volume()) for q in s.decompose()]
        print(name,'component volumes mm3',components,flush=True)
        if s.status()!=md.Error.NoError:raise ValueError('Invalid boolean partition')
        m=s.to_mesh64();v=m.vert_properties[:,:3];f=np.asarray(m.tri_verts,dtype=np.int64)
        parent_bank=np.array([bone_ids[k]for k in np.repeat(m.run_original_id,np.diff(m.run_index)//3)])
        parent_face=np.asarray(m.face_id,dtype=int)
        if not set(parent_bank)=={0,10,12}:raise ValueError('Lost cortical interface')
        check,_,_=solid(v,f,'reimport');remainders={k:abs((check^b).volume())for k,b in cortex.items()}
        if max(remainders.values())>1.e-6 or abs(check.volume()-s.volume())>1.e-5:raise ValueError('Reimported partition intersects bone')
        # Cortical interface areas and normals come from the actual shared faces.
        normals=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
        area={k:float(np.linalg.norm(normals[parent_bank==k],axis=1).sum()/2)for k in np.unique(parent_bank)}
        saved=out/(name+'.npz')
        np.savez_compressed(saved,vertices_m=v/1000+origin,triangles=f,parent_bank=parent_bank,parent_face=parent_face)
        stages[name]=dict(source_mm3=original,partition_mm3=s.volume(),displaced_mm3=original-s.volume(),
                          overlap_mm3=remainders,parent_surface_area_mm2={str(k):a for k,a in area.items()},
                          component_volumes_mm3=components,vertices=len(v),faces=len(f),surface=digest(saved))
    out.joinpath('partition.json').write_text(json.dumps(dict(source=digest(source),cortical_witness=digest(report),
        stages=stages,scope='Two independent registered partitions. New cortical faces are interface candidates. Their existence does not determine an attachment traction or qualify continuous contact.'),indent=2)+'\n')
    print(stages,flush=True)

if __name__=='__main__':partition(*(Path(s).resolve()for s in sys.argv[1:]))
