"""Restore the atlas articular frame with one constant rigid joint-base correction per side."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from fast_skin_intersection_audit import install_for_float32
from compare_registration import prepare
from bone_overlap import intersections


def solve(launch,registration,motion,archive):
    _,_,_,anchors,poses,reference,data=prepare(launch,registration,motion,archive)
    install_for_float32(); outputs=[]
    for row in anchors['candidates']:
        s,h=(data[k] for k in row['stable_ids'])
        # Localize the closest source joint surface patch using nearest vertices.
        # Winding and exact face pairs below decide actual clearance.
        tree=cKDTree(s[2]); distance,_=tree.query(h[2])
        patch=distance<=distance.min()+.003
        pivot=h[2][patch].mean(axis=0)
        # Atlas left/right is its X axis. Derive sign from the humeral centroid.
        lateral=np.array([np.sign(h[2][:,0].mean()-s[2][:,0].mean()),0.,0.])
        rs=Rotation.from_quat(reference[s[1]][3:])
        origin=reference[s[1]][:3]
        world_rotation=s[6]*h[6].inv()
        parent_rot=rs.inv()*world_rotation*rs
        hum_p=h[6].apply(pivot)*h[7]+h[8]
        scap_p=s[6].apply(pivot)*s[7]+s[8]
        base=rs.inv().apply(world_rotation.apply(origin-hum_p)+scap_p-origin)
        normal=rs.inv().apply(s[6].apply(lateral))
        frames={}
        for key,pose in [(0,reference),*poses.items()]:
            ps,ph=pose[s[1]],pose[h[1]]
            rot=Rotation.from_quat(ps[3:])
            x=ps[:3]+rot.apply(s[4])
            y=ph[:3]+Rotation.from_quat(ph[3:]).apply(h[4])
            newy=ps[:3]+rot.apply(parent_rot.apply(rot.inv().apply(y-ps[:3]))+base)
            frames[key]=(x,newy,rot.apply(normal))
        sentinels=[0,32,256,8000,-1,-2,-3]
        trials=[]
        for mm in (0,.25,.5,1,2,3,4,5,6,8):
            counts={}
            for k in sentinels:
                x,y,n=frames[k]
                hits=intersections(x,s[3],y+mm*.001*n,h[3]);counts[k]=len(hits)
                if hits:break
            print(row['side'],mm,counts,flush=True)
            trials.append(dict(additional_shift_mm=mm,counts=counts))
            if any(counts.values()):continue
            for k in sorted(poses):
                if k in counts:continue
                x,y,n=frames[k]
                hits=intersections(x,s[3],y+mm*.001*n,h[3]);counts[k]=len(hits)
                if hits:break
            if not any(counts.values()):break
        outputs.append(dict(side=row['side'],patch_count=int(patch.sum()),source_closest_vertex_distance_m=float(distance.min()),
                     reference_pivot_m=pivot.tolist(),parent_rotation_xyzw=parent_rot.as_quat().tolist(),
                     parent_translation_base_m=base.tolist(),parent_lateral_unit=normal.tolist(),trials=trials))
    return outputs
if __name__=='__main__':
    *args,out=map(Path,sys.argv[1:]);out.write_text(json.dumps(solve(*args),indent=2,allow_nan=False)+'\n')
