"""Fit a rigid clavicle between its sternum and the registered scapular articular site."""
from pathlib import Path
import json,sys,hashlib,zipfile
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from compare_registration import prepare
from compare_atlas import vertices_and_faces
from bone_overlap import intersections
from fast_skin_intersection_audit import install_for_float32


def contact_site(bone,neighbor):
    v=np.unique(bone,axis=0)
    distance,_=cKDTree(neighbor).query(v)
    return v[distance<=distance.min()+.003].mean(axis=0)

def orient(a,b):
    u,v=a/np.linalg.norm(a),b/np.linalg.norm(b);w=np.cross(u,v)
    return Rotation.from_rotvec(w/np.linalg.norm(w)*np.arctan2(np.linalg.norm(w),u@v))

def run(launch,ref,motion,archive,frames):
    cfg,cat,meshes,an,poses,default,data=prepare(launch,ref,motion,archive)
    doc=json.loads(Path(cfg['argv'][3]).with_suffix('.manifest.json').read_text())
    member=next(v for v in doc['source']['anchors'] if v['member_id']=='FJ3290')
    with zipfile.ZipFile(archive) as z:
        raw=z.read(next(n for n in z.namelist() if n.endswith('/FJ3290.obj')))
    assert hashlib.sha256(raw).hexdigest()==member['member_sha256']
    verts,faces=vertices_and_faces(raw)
    sternum=next(k for k,c in cat.items() if c['body']==member['core_body_index'] and c['vertexCount']==len(verts) and c['indexCount']==faces.size)
    vl,fl,*_=meshes[sternum]; assert np.array_equal(faces,fl)
    body=member['core_body_index'];pb=default[body]
    x=pb[:3]+Rotation.from_quat(pb[3:]).apply(vl)
    affine=np.linalg.lstsq(np.c_[verts,np.ones(len(verts))],x,rcond=None)[0]
    assert np.max(np.linalg.norm(np.c_[verts,np.ones(len(verts))]@affine-x,axis=1))<1e-6
    frame=json.loads(frames.read_text()); result=[]; moved={}
    for a,cal in zip(an['candidates'],frame):
        sid,hid=a['stable_ids'];clav=9 if a['side']==0 else 8;s,h,c=data[sid],data[hid],data[clav]
        # Match the two sourced clavicular interface patches to their neighbors in common rest.
        s1=contact_site(c[2],s[2]); s0=contact_site(c[2],verts)
        orig0=c[6].apply(s0)*c[7]+c[8]; orig1=c[6].apply(s1)*c[7]+c[8]
        target0=np.r_[s0,1.]@affine
        scap_world=s[6].apply(s1)*s[7]+s[8]
        b=default[s[1]];rs=Rotation.from_quat(b[3:]);rsdelta=Rotation.from_quat(cal['parent_rotation_xyzw']).inv()
        def apply_s(points):
            return rsdelta.apply(points-np.asarray(cal['parent_translation_base_m']))
        target1=b[:3]+rs.apply(apply_s(rs.inv().apply(scap_world-b[:3])))
        # A rigid fit with the sternoclavicular interface as its pivot. Report
        # unrecoverable length error explicitly; there is no stretch allowance.
        rc=orient(orig1-orig0,target1-target0)
        new_c=rc.apply(c[5]-orig0)+target0
        bc=default[c[1]];c_local=Rotation.from_quat(bc[3:]).inv().apply(new_c-bc[:3])
        moved[clav]=c_local; moved[sid]=apply_s(s[4])
        length=np.linalg.norm(orig1-orig0);required=np.linalg.norm(target1-target0)
        sc=(contact_site(c[2],verts)+contact_site(verts,c[2]))/2
        ac=(contact_site(c[2],s[2])+contact_site(s[2],c[2]))/2
        gh=(contact_site(s[2],h[2])+contact_site(h[2],s[2]))/2
        source=[np.linalg.norm(ac-sc),np.linalg.norm(gh-ac)]
        l1=source[0]*c[7]; l2=source[1]*s[7]
        base_sc=np.r_[sc,1.]@affine
        target_gh=h[6].apply(gh)*h[7]+h[8]
        d=np.linalg.norm(target_gh-base_sc)
        srcd=np.linalg.norm(gh-sc)
        reach=dict(clavicular_length_m=float(l1),scapular_length_m=float(l2),
            available_reach_m=float(l1+l2),required_reach_m=float(d),shortfall_m=float(max(0,d-l1-l2)),
            original_atlas_reach_m=float(srcd),original_atlas_maximum_reach_m=float(sum(source)))
        e=(target_gh-base_sc)/d
        along=(l1*l1-l2*l2+d*d)/(2*d)
        ring_radius=np.sqrt(l1*l1-along*along)
        # Keep the acromioclavicular site closest to the current clavicle
        # while satisfying both measured bone lengths and sharing joint centers.
        proposal=c[6].apply(ac)*c[7]+c[8]
        radial=proposal-base_sc-np.dot(proposal-base_sc,e)*e
        radial/=np.linalg.norm(radial)
        target_ac=base_sc+along*e+ring_radius*radial
        rotations=[]
        for bone,k,src0,src1,goal0,goal1,preferred in (
            (c,clav,sc,ac,base_sc,target_ac,c[6]),
            (s,sid,ac,gh,target_ac,target_gh,h[6]),
        ):
            rr=orient(preferred.apply(src1-src0),goal1-goal0)*preferred
            world=goal0+rr.apply(bone[2]-src0)*bone[7]
            bp=default[bone[1]]
            moved[k]=Rotation.from_quat(bp[3:]).inv().apply(world-bp[:3])
            residual=goal0+rr.apply(src1-src0)*bone[7]-goal1
            assert np.linalg.norm(residual)<1e-10
            rotations.append(float((rr*bone[6].inv()).magnitude()*180/np.pi))
        reach.update(achieved_ac_site_m=target_ac.tolist(),joint_fit_rotation_degrees=rotations)
        result.append(dict(side=a['side'],two_link_reach=reach,actual_clavicular_interpatch_span_m=float(length),required_span_m=float(required),
                         residual_at_acromioclavicular_site_m=float(abs(length-required)),rotation_deg=float(rc.magnitude()*180/np.pi)))
    install_for_float32();counts=[]
    for key,pose in [(0,default),*((s,poses[s]) for s in (-3,32,256,8000))]:
        banks={}
        for sid,(v,f,*_) in meshes.items():
            if cat[sid]['body'] not in pose and isinstance(pose,dict):continue
            p=pose[cat[sid]['body']];rot=Rotation.from_quat(p[3:])
            banks[sid]=p[:3]+rot.apply(moved[sid] if sid in moved else v)
        hits=[]
        for sid in moved:
            for k,v in banks.items():
                if k==sid or k in moved and k<sid:continue
                x=banks[sid]
                if np.any(x.min(axis=0)>v.max(axis=0)+1e-7) or np.any(v.min(axis=0)>x.max(axis=0)+1e-7):continue
                pairs=intersections(x,meshes[sid][1],v,meshes[k][1])
                if pairs:hits.append([sid,k,len(pairs),pairs[0]])
        counts.append(dict(step=key,hits=hits))
    return dict(clavicle_fit=result,changed_bone_interfaces=counts)
if __name__=='__main__':
    *args,out=map(Path,sys.argv[1:]);out.write_text(json.dumps(run(*args),indent=2)+'\n')
