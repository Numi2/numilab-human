"""Hold interface faces that can reach a cortical solid in the registered trust region."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rebind_ligament_material import banks
from compile_connected_joint import digest
def run(job,trial,first,path):
    rd=lambda f:json.loads(f.read_bytes())
    failed=rd(trial/'quality.json')['increased_bone_overlap_mm3'];blocked={int(b) for b,role,d in failed}
    comp=rd(trial/'bone-overlap/result.json');source=job/'distal-registered-native'
    bone=Path(rd(source/'launch.json')['argv'][3]);pose=source/'target-pose.json'
    if digest(bone)!=comp['bone'] or digest(pose)!=comp['capture']:raise ValueError('Wrong rigid pose')
    y=np.loadtxt(first/'current.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    b=np.load(job/'registered-boundary/registered-boundary.npz');face=b['I_node_joint'][b['I_face_nodes']]
    q=rd(pose)['all_bodies'];fixed=[];found=set()
    for (body,bank),(_,v,_) in banks(bone).items():
        if bank not in blocked:continue
        state=q[body]['current'];local=Rotation.from_quat(state['orientation_world_xyzw']).inv().apply(y-np.array(state['position_world_m']))
        f=local[face];pad=np.sqrt(3)*.0001+2e-7
        hits=(np.all((f.min(1)<=v.max(0)+pad)&(f.max(1)>=v.min(0)-pad),axis=1))
        fixed.extend(face[hits].ravel());found.add(bank)
    if found!=blocked:raise ValueError('Missing bone')
    nodes=np.unique(fixed).astype('i4');np.savez_compressed(path,nodes=nodes,banks=sorted(blocked))
    print('cortical banks',sorted(blocked),'interface nodes',len(nodes))
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
