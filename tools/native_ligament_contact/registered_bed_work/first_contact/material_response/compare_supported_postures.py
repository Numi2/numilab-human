"""Compare registered limb poses and cortical contact on the unchanged physical tissue."""
from pathlib import Path
import json,sys,struct
import numpy as np
from scipy.spatial.transform import Rotation
from audit_registered_skeleton import closed_solid,intersection_mm3
from audit_fiber_load_bounds import read_pose
from rebind_ligament_material import banks
from compile_connected_joint import digest

def poses(s):
    d=json.loads(s.read_text());return np.array([p['current']['position_world_m']+p['current']['orientation_world_xyzw'] for p in d['all_bodies']])
def run(h,job,fit,native,out):
    x=poses(job/'native-initial-bodies.json');y=poses(native);q=json.loads(fit.read_text())['arms']
    pre=np.array(read_pose(h.parent/'native-ligament-material/stderr.log')['bodyPoses']);asset=next((h.parent/'supported-bones').glob('*.nhbones'));bank=banks(asset)
    stats={};shapes={};initial={};bound={}
    for item in q:
        side=item['side'];hand=item['hand'];test=np.array(item['body_poses'][str(hand)]);ref=x[hand]
        orientation=lambda a,b:float(np.rad2deg((Rotation.from_quat(a[3:])*Rotation.from_quat(b[3:]).inv()).magnitude()))
        raw=(h.parent/'supported/source-bound.nhrigid').read_bytes();bodies=[hand]
        for child in range(hand+1,len(y)):
            parent,owner=struct.unpack_from('<2I',raw,224+157*160+144*(child-1))
            if owner!=child:raise ValueError('Native joint tree mismatch')
            if parent in bodies:bodies.append(child)
        errors=np.linalg.norm(y[bodies,:3]-x[bodies,:3],axis=1)
        stats[side]=dict(hand_body=hand,bodies=len(bodies),native_target_error_mm=float(1000*np.linalg.norm(y[hand,:3]-ref[:3])),native_target_orientation_deg=orientation(y[hand],ref),analytical_vs_native_mm=float(1000*np.linalg.norm(y[hand,:3]-test[:3])),analytical_vs_native_deg=orientation(y[hand],test),maximum_hand_subtree_center_error_mm=float(1000*errors.max()),maximum_hand_subtree_orientation_deg=max(orientation(y[b],x[b]) for b in bodies))
    for (b,s),(f,v,_) in bank.items():
        def shape(p):
            w=(Rotation.from_quat(p[b,3:]).apply(v)+p[b,:3]).astype('f4').astype(float)
            return closed_solid(w*1000.,f)
        shapes[s]=shape(y);initial[s]=shape(pre);bound[s]=b
    before={};after={}
    for a in shapes:
        for b in shapes:
            if b<=a or bound[a]==bound[b]:continue
            va=intersection_mm3(shapes[a],shapes[b]);vb=intersection_mm3(initial[a],initial[b]);k=f'{a}/{b}'
            if va>1e-6:after[k]=va
            if vb>1e-6:before[k]=vb
    delta={k:after.get(k,0.)-before.get(k,0.) for k in (after.keys()|before.keys())}
    result=dict(pose=digest(native),fit=digest(fit),skin_fat_geometry='unchanged',bone=digest(asset),wrist=stats,separate_rigid_owner_bones=dict(before=len(before),after=len(after),new=sorted(after.keys()-before.keys()),removed=sorted(before.keys()-after.keys()),largest_increase_mm3=sorted(delta.items(),key=lambda x:-x[1])[:12],before_mm3=before,after_mm3=after))
    out.write_text(json.dumps(result,indent=2)+'\n');print(stats,flush=True)
if __name__=='__main__':run(*(Path(x).resolve() for x in sys.argv[1:]))
