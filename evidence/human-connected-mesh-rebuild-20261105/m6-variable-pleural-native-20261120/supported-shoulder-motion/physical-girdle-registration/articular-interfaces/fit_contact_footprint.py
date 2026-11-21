"""Repair the two inherited medial clavicular intersections in material coordinates."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from shoulder_inputs import digest,old
from connected_registration import Field
from compare_registration import prepare
from girdle_ik import contact_site

def volume(v,f):
    tri=(v-v.mean(0))[f];return np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6

def solve(launch,sites,motion,archive,zero,out):
    cfg,cat,meshes,_,poses,rest,_=prepare(launch,sites,motion,archive)
    parent=Path(cfg['argv'][3])
    old.audit.ag.BONES=parent;_,raw,_,_,_,_,_,_,_,vo,_=old.audit.ag.read_payloads()
    doc=json.loads(parent.with_suffix('.manifest.json').read_text());assert digest(parent)==doc['payload']['sha256']
    receipt=json.loads(zero.read_text());cap={r['body_index']:np.array(r['position_m']+r['quaternion_xyzw']) for r in receipt['accepted_body_poses']}
    assert receipt['accepted_step']==0
    posesets=[rest,cap,*[pose for k,pose in sorted(poses.items()) if k<0 or k%800==0 or k in (32,128,256,512)]]
    field=Field(*meshes[27][:2],h=.00032)
    targets=[];summary=[];payload=bytearray(raw)
    for clav,scap in ((8,10),(9,11)):
        v,f,norm=meshes[clav][:3];v=np.array(v)
        geom=cat[clav];b=geom['body'];ob=cat[27]['body']
        pose=rest[b];p0=pose[:3];rr=R.from_quat(pose[3:])
        # Medial and lateral ends come from the nearest regions of the
        # original manubrium and acromion on the source registration.
        cv=p0+rr.apply(v);sv=rest[ob][:3]+R.from_quat(rest[ob][3:]).apply(meshes[27][0])
        acb=cat[scap]['body'];av=rest[acb][:3]+R.from_quat(rest[acb][3:]).apply(meshes[scap][0])
        sc=rr.inv().apply(contact_site(cv,sv)-p0);ac=rr.inv().apply(contact_site(cv,av)-p0)
        axis=ac-sc;axis/=np.linalg.norm(axis)
        tris=v[f];samples=np.vstack([np.unique(v,axis=0),tris.mean(1),*(.5*(tris[:,k]+tris[:,(k+1)%3]) for k in range(3))])
        samples=samples[(samples-sc)@axis<.04]
        transforms=[]
        for p in posesets:
            a,q=p[b],p[ob];inv=R.from_quat(q[3:]).inv()
            transforms.append((inv*R.from_quat(a[3:]),inv.apply(a[:3]-q[:3])))
        # Locate a material-interior anchor next to the medial end. Restrict
        # the map to source faces near a real opposed surface in some pose.
        from audit_solids import winding
        ax=(v-sc)@axis
        core=v[(ax>.009)&(ax<.023)].mean(0)
        assert abs(winding(v[f],core))>.99
        def worst(x):
            return np.min([field.values(r.apply(x)+t) for r,t in transforms],axis=0)
        sample_gap=worst(samples);sites=samples[sample_gap<.0007]
        required=np.zeros(len(sites))
        crossed=np.zeros(len(sites),bool)
        for frac in np.arange(0,.71,.005):
            passing=worst(sites+frac*(core-sites))>=.0007
            required[passing&~crossed]=frac
            crossed|=passing
            if np.all(crossed):break
        assert crossed.all(),'no local interior radial route to the opposed surface'
        unique,which=np.unique(sites,axis=0,return_index=True);required=required[which];sites=unique
        tree=cKDTree(sites);radius=.011
        def deform(x):
            disp=np.zeros_like(x);use=cKDTree(x).sparse_distance_matrix(tree,radius,output_type='ndarray')
            i,j,d=(use[k] for k in ('i','j','v'))
            w=1-(d/radius)**2
            envelope=np.zeros(len(x));np.maximum.at(envelope,i,required[j]*w**3)
            return x+envelope[:,None]*(core-x)
        new=deform(v);changed=np.any(new!=v,axis=1)
        jac=np.stack([(deform(v+e)-deform(v-e))/(2.e-6) for e in np.eye(3)*1.e-6],axis=2)
        assert np.min(np.linalg.det(jac))>0
        distance=float(np.linalg.norm(new-v,axis=1).max())
        normal=np.linalg.solve(jac.transpose(0,2,1),norm[:,:,None])[:,:,0]
        normal/=np.linalg.norm(normal,axis=1)[:,None]
        inverse=R.from_quat(geom['boneQ']).inv()
        vals=np.frombuffer(payload,dtype='<f4',count=6*geom['vertexCount'],offset=vo+24*geom['firstVertex']).reshape(-1,6)
        vals[changed,:3]=(inverse.apply(new[changed]-geom['translation'])/geom['scale']).astype('f4')
        vals[changed,3:]=inverse.apply(normal[changed]).astype('f4')
        actual=R.from_quat(geom['boneQ']).apply(vals[:,:3].astype(float)*geom['scale'])+geom['translation']
        t0=v[f];t1=actual[f];a=np.cross(t0[:,1]-t0[:,0],t0[:,2]-t0[:,0]);z=np.cross(t1[:,1]-t1[:,0],t1[:,2]-t1[:,0])
        area=np.linalg.norm(z,axis=1)/np.linalg.norm(a,axis=1)
        cosine=np.einsum('ij,ij->i',a,z)/np.linalg.norm(a,axis=1)/np.linalg.norm(z,axis=1)
        assert np.all(cosine>0) and np.all(area>0)
        targets.append(dict(stable_id=clav,member_id=doc['source']['anchors'][clav-1]['member_id'],core_body=b))
        summary.append(dict(stable_id=clav,changed_vertices=int(changed.sum()),vertices=len(v),distance_m=distance,
            sc_local_m=sc.tolist(),axis=axis.tolist(),footprint_radius_m=radius,interior_anchor_local_m=core.tolist(),grid_clearance_before_m=float(sample_gap.min()),
            grid_clearance_after_m=float(worst(deform(samples)).min()),relative_signed_volume_change=volume(actual,f)/volume(v,f)-1,
            max_displacement_m=float(np.linalg.norm(actual-v,axis=1).max()),face_area_ratio=[area.min(),area.max()],
            min_face_normal_cosine=cosine.min(),sampled_material_jacobian_range=[float(np.linalg.det(jac).min()),float(np.linalg.det(jac).max())]))
    out.mkdir(exist_ok=True);bone=out/parent.name;bone.write_bytes(payload)
    doc['source']['connected_girdle_parent']=dict(bone_sha256=digest(parent),changed_anchors=targets)
    info=dict(parent_sha256=digest(parent),atlas_sha256=digest(archive),pose_sha256=digest(zero),
        motion_sha256=digest(motion),source_program_sha256=digest(Path(__file__)),repairs=summary)
    doc['source']['medial_clavicle_reference_repair']=info
    doc['payload']['sha256']=digest(bone);doc['status']='experimental_medial_clavicle_geometry_requires_coupled_admission'
    (out/parent.with_suffix('.manifest.json').name).write_text(json.dumps(doc,indent=2)+'\n')
    (out/'repair.json').write_text(json.dumps(dict(**info,bones_sha256=digest(bone)),indent=2)+'\n')
    print(*summary,sep='\n',flush=True)
if __name__=='__main__':solve(*map(Path,sys.argv[1:]))
