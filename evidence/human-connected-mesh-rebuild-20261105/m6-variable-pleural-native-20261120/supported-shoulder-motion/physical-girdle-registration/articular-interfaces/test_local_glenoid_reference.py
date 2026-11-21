"""Test a common source frame on the two glenohumeral interfaces with the distal bones fixed."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from compare_registration import prepare
from connected_registration import points
from girdle_ik import contact_site
from shoulder_inputs import old,digest
from bone_overlap import intersections
from audit_solids import winding
from compare_atlas import closest_distance
from fast_skin_intersection_audit import install_for_float32

def run(launch,sites,motion,archive,zero,out):
    _,cat,meshes,attachments,poses,rest,data=prepare(launch,sites,motion,archive)
    rec=json.loads(zero.read_text())
    assert rec['accepted_step']==0 and rec['accepted_time_s']==0
    cap={p['body_index']:np.array(p['position_m']+p['quaternion_xyzw']) for p in rec['accepted_body_poses']}
    install_for_float32();rows=[]
    for att in attachments['candidates']:
        sb,hb=att['stable_ids'];sc,hu=data[sb],data[hb]
        src=(contact_site(sc[2],hu[2])+contact_site(hu[2],sc[2]))*.5
        for ratio in (0.,.25,.5,.75,1.):
            center=(1-ratio)*points(sc,src)+ratio*points(hu,src)
            common=R.from_rotvec((hu[6]*sc[6].inv()).as_rotvec()*ratio)*sc[6]
            scale=(1-ratio)*sc[7]+ratio*hu[7]
            geometry={}
            for k,item in [(sb,sc),(hb,hu)]:
                v=item[2];new=center+common.apply(v-src)*scale;oldx=points(item,v)
                radius=np.linalg.norm(v-src,axis=1)
                # Include the glenoid and humeral head, then fade the rigid
                # source registration into the original rest material.
                t=np.clip((radius-.044)/.075,0,1)
                w=1-((6*t-15)*t+10)*t**3
                world=oldx+w[:,None]*(new-oldx)
                p=rest[item[1]];local=R.from_quat(p[3:]).inv().apply(world-p[:3])
                mask=w>0;local[~mask]=meshes[k][0][~mask]
                geometry[k]=local
            samples=[]
            for step,q in [(0,rest),(-4,cap),(-3,poses[-3]),(32,poses[32]),(8000,poses[8000])]:
                def world(k):
                    p=q[cat[k]['body']];return (p[:3]+R.from_quat(p[3:]).apply(geometry.get(k,meshes[k][0]))).astype('f4')
                v,x=world(sb),world(hb);f,g=sc[3],hu[3]
                p=np.asarray(att['material_barycentric'][0])@v[f[att['triangle_rows'][0]]]
                a=np.asarray(att['material_barycentric'][1])@x[g[att['triangle_rows'][1]]]
                wh=winding(x[g].astype(float),p);dist,_=closest_distance(x[g].astype(float),p)
                pairs=intersections(v,f,x,g)
                # Straight physical strap admission remains an independent check.
                span_hits=sum(len(old.audit.lp.vector_hits(p,a,(vv.astype(float),ff))) for vv,ff in ((v,f),(x,g)))
                samples.append(dict(step=step,pairs=len(pairs),origin_in_humerus=bool(abs(wh)>.5),origin_to_humerus_m=dist,
                    straight_span_hits=span_hits,bed_scapula_m=float(v[:,2].min()) if step==-4 else None))
            shape=[]
            for k in (sb,hb):
                f=meshes[k][1];a=meshes[k][0][f];b=geometry[k][f]
                u=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);v=np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0])
                n=np.linalg.norm(u,axis=1);m=np.linalg.norm(v,axis=1)
                shape.append(dict(bone=k,max_vertex_displacement_m=np.linalg.norm(geometry[k]-meshes[k][0],axis=1).max(),
                    min_normal_cosine=(np.einsum('ij,ij->i',u,v)/n/m).min(),face_area_ratio=[(m/n).min(),(m/n).max()]))
            rows.append(dict(side=att['side'],humeral_frame_fraction=ratio,samples=samples,shape=shape))
    out.write_text(json.dumps(dict(atlas_sha=digest(archive),pose_sha=digest(zero),rows=rows),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
