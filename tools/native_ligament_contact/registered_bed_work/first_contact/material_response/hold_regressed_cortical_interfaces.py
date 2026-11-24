"""Protect registered boundary triangles that can affect the measured cortical regressions."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rebind_ligament_material import banks
from compile_connected_joint import digest
def run(job,out):
    mat=job/'interval-certified-scene/material';base=json.loads((job/'interval-certified-scene/bone-audit/result.json').read_bytes())
    trial=json.loads((job/'conditioned-loaded-strain/bone-overlap/result.json').read_bytes())
    if (trial['bone'],trial['capture'])!=(base['bone'],base['capture']):raise ValueError('Mismatched bone registration')
    blocked={int(k) for k,v in trial['results'].items() if any(v[n]>base['results'][k][n]+1e-6 for n in ['skin','fat'])}
    cfg=json.loads((job/'distal-registered-native/launch.json').read_bytes());bone=Path(cfg['argv'][3]);pose=job/'distal-registered-native/target-pose.json'
    if digest(bone)!=base['bone'] or digest(pose)!=base['capture']:raise ValueError('Incorrect cortical frames')
    q=json.loads(pose.read_bytes())['all_bodies'];y=np.loadtxt(mat/'current.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    z=np.load(job/'registered-boundary/registered-boundary.npz');f=np.vstack([z[k+'_node_joint'][z[k+'_face_nodes']] for k in 'ID']);cut=[]
    for (body,bank),(_,v,_) in banks(bone).items():
        if bank not in blocked:continue
        p=q[body]['current'];pts=Rotation.from_quat(p['orientation_world_xyzw']).inv().apply(y-np.array(p['position_world_m']))
        tri=pts[f];hit=np.all((tri.min(1)<=v.max(0)+.002)&(tri.max(1)>=v.min(0)-.002),axis=1)
        cut.append(f[hit].ravel())
    if len(cut)!=len(blocked):raise ValueError('Unmapped bone')
    out.mkdir();np.save(out/'nodes.npy',np.unique(np.concatenate(cut)));(out/'map.json').write_text(json.dumps(dict(banks=sorted(blocked),pose=base['capture'],bones=base['bone'],padded_box_m=.002))+'\n')
if __name__=='__main__':run(*(Path(p).resolve() for p in sys.argv[1:]))
