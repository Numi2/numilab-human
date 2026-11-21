"""Recover the passive routes already present in the pinned, bilateral MyoSim model."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,mujoco
from myo_sim.build.compose import build_model
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from source_pair_audit import inputs
from shoulder_inputs import old,digest
from audit_solids import winding
from compare_atlas import closest_distance

def run(launch,refs,motion,bones,src,out):
    m=build_model('myofullbody');d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    base,poses,rest,owners=inputs(launch,refs,motion)
    old.audit.ag.BONES=bones;_,raw,_,_,_,_,_,cat,_,vo,io=old.audit.ag.read_payloads()
    ms=old.load_local_meshes(raw,cat,vo,io)
    sph=np.load(src);rows=[]
    for side in ('l','r'):
        for name in ('s_glenohum_ligament','coracohum_ligament'):
            n=name+('_l' if side=='l' else '');i=m.tendon(n).id
            indices=list(range(m.tendon_adr[i],m.tendon_adr[i]+m.tendon_num[i]))
            ty=list(map(int,m.wrap_type[indices]));assert ty==[3,4,3]
            start,g,end=map(int,m.wrap_objid[indices]);side_site=int(m.wrap_prm[indices[1]])
            assert m.geom_type[g]==int(mujoco.mjtGeom.mjGEOM_SPHERE)
            def local(x,b):return (x-d.xipos[b])@d.ximat[b].reshape(3,3)
            def describe(site):
                b=int(m.site_bodyid[site]);key=m.body(b).name;sid,core=owners[key]
                x=local(d.site_xpos[site],b)
                tissues={}
                for title,vs,fs in [('source',sph[key+'_local'],sph[key+'_faces']),('registered_parent',*base[sid][:2]),('registered_candidate',*ms[sid][:2])]:
                    fd,face=closest_distance(vs[fs],x);w=winding(vs[fs],x)
                    tissues[title]=dict(surface_distance_m=fd,face=face,winding=w)
                assert np.linalg.norm(rest[core][:3]+R.from_quat(rest[core][3:]).apply(x)-d.site_xpos[site])<1e-6
                return dict(site=m.site(site).name,body=core,stable_id=sid,point_local_m=x.tolist(),geometry=tissues)
            gh=describe(start);sc=describe(end);core=gh['body'];b=m.geom_bodyid[g]
            c=local(d.geom_xpos[g],b);q=local(d.site_xpos[side_site],m.site_bodyid[side_site]);rad=float(m.geom_size[g,0])
            assert int(m.site_bodyid[start])==b==int(m.site_bodyid[side_site])
            row=dict(name=n,side=side,source_tendon_index=i,sites=[gh,sc],source_stiffness=float(m.tendon_stiffness[i]),
                source_wrap=dict(geom=m.geom(g).name,core_body=core,source_geom_type='sphere',radius_m=rad,center_local_m=c.tolist(),side_site_local_m=q.tolist(),side_site=m.site(side_site).name),
                source_springlength_m=m.tendon_lengthspring[i].tolist(),default_source_path_m=float(d.ten_length[i]),default_source_wrap_points=int(d.ten_wrapnum[i]))
            # The unclamped witness is the chord, before admission at the
            # source sphere. This tests availability of the control route.
            values=[]
            for k,pp in [(0,rest),*poses.items()]:
                p,h=pp[core],pp[sc['body']]
                x=np.array(gh['point_local_m']);y=R.from_quat(p[3:]).inv().apply(h[:3]+R.from_quat(h[3:]).apply(sc['point_local_m'])-p[:3])
                diff=y-x;t=np.clip((c-x)@diff/(diff@diff),0,1)
                clearance=np.linalg.norm(x+t*diff-c)-rad
                values.append([int(k),float(np.linalg.norm(diff)),float(clearance)])
            a=np.array(values);row.update(retained_chord_range_m=[float(a[:,1].min()),float(a[:,1].max())],
                retained_chord_minimum_sphere_clearance_m=float(a[:,2].min()),retained_chords_enter_sphere=int(np.sum(a[:,2]<0)),poses=len(a))
            rows.append(row)
    out.write_text(json.dumps(dict(bone_sha256=digest(bones),source_surfaces_sha256=digest(src),routes=rows),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
