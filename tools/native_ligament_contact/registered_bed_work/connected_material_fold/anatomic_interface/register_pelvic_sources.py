"""Check the authored pelvic cortex against the existing mechanics meshes in their common body frame."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from compile_connected_joint import digest
from rebind_ligament_material import banks
from compare_supported_postures import poses
from inspect_shoulder_source import nearest
from measure_forearm_enclosure import Enclosure
from audit_registered_skeleton import closed_solid
from build_carpal_cavity import boundary
from partition_carpal_material import solid

def sample(x,f,n,seed):
    rng=np.random.default_rng(seed);tri=x[f];a=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    u,v=rng.random((2,n));u=np.sqrt(u)
    return np.einsum('ij,ijk->ik',np.c_[1-u,u*(1-v),u*v],tri[rng.choice(len(tri),n,p=a/a.sum())])
def move(x,H):return x@H[:3,:3].T+H[:3,3]
def rigid(a,b):
    x=a.mean(0);y=b.mean(0);U,_,V=np.linalg.svd((a-x).T@(b-y));R=V.T@U.T
    if np.linalg.det(R)<0:V[-1]*=-1;R=V.T@U.T
    H=np.eye(4);H[:3,:3]=R;H[:3,3]=y-R@x;return H
def run(work,checkout,lib):
    sys.path.insert(0,str(checkout))
    import mujoco
    from myo_sim.build.compose import build_model
    m=build_model('myofullbody');q=work.parent;j=q.parent.parent;d=json.loads((q/'gravity-scene/launch.json').read_bytes());asset=Path(d['argv'][3])
    if digest(asset)!=d['pins'][str(asset)]:raise ValueError('Unverified rigid geometry')
    P=poses(j/'distal-registered-native/target-pose.json');B=banks(asset);coords={};results=[];comparison=[]
    W=json.loads((q/'rigid-penetration/result.json').read_bytes())['banks'];stl={}
    for bank,body,side in [(18,128,'r'),(19,128,'l'),(14,42,'r')]:
        name='pelvis_'+side if bank!=14 else 'ulna_r';g=m.geom(name).id;b=m.geom_bodyid[g];mi=m.geom_dataid[g]
        rot=lambda w:Rotation.from_quat(w[[1,2,3,0]])
        av=m.mesh_vertadr[mi];af=m.mesh_faceadr[mi];v=m.mesh_vert[av:av+m.mesh_vertnum[mi]].astype(float);F=m.mesh_face[af:af+m.mesh_facenum[mi]].copy()
        v=rot(m.body_iquat[b]).inv().apply(rot(m.geom_quat[g]).apply(v)+m.geom_pos[g]-m.body_ipos[b])
        f,x,_=B[body,bank];coords[bank]=(f,x,F,v);w=next(s for s in W if s['bank']==bank);R=Rotation.from_quat(P[body,3:]);world=(R.apply(v)+P[body,:3]).astype('f4').astype('f8');ref=Enclosure(lib,world,F)
        try:inside,dist=ref.query(np.array([w['registered_m']]))
        finally:ref.close()
        # Anatomical shape and mechanics geometry can have different open boundaries.
        und=np.sort(np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1)
        _,n=np.unique(und,axis=0,return_counts=True)
        comparison.append(dict(bank=bank,geometry_name=name,source_body=int(b),core_body=body,mechanics_vertices=len(v),mechanics_edges_not_twice=int((n!=2).sum()),mechanics_inside_classifier=int(inside[0]),distance_to_mechanics_mm=float(abs(dist[0])*1e3),registered_anatomy_inside_depth_mm=w['maximum_inside_node_depth_mm']))
        mname=m.mesh(mi).name;path=checkout/'myo_sim/models/meshes'/(mname+'.stl')
        if path.is_file():stl[mname]=digest(path)
    train=[];test=[]
    for k in (18,19):
        f,x,F,v=coords[k];train.append((sample(x,f,1500,127+k),sample(v,F,1500,213+k)));test.append((sample(x,f,2500,441+k),sample(v,F,2500,225+k)))
    H=np.eye(4);poses_to_test={'authored':H.copy()}
    for step in range(40):
        aa=[];bb=[]
        for (xs,vs),k in zip(train,(18,19)):
            f,x,F,v=coords[k];cur=move(xs,H);mv=move(x,H)
            p,ds,_=nearest((v,F),cur,lib);rev,dr,_=nearest((mv,f),vs,lib)
            for l,r,e in [(cur,p,ds),(rev,vs,dr)]:
                keep=e<=np.quantile(e,.8);aa.append(l[keep]);bb.append(r[keep])
        change=rigid(np.vstack(aa),np.vstack(bb));H=change@H
        if step in (4,14,39):poses_to_test['bilateral_pass_'+str(step+1)]=H.copy()
    # The comparison uses actual occupied material solids and a nearest-surface
    # test held out from the source fit. No anatomical bank is resized.
    vol=q/'qualified-material';y=np.loadtxt(vol/'current.node',skiprows=1,usecols=(1,2,3));cell=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');t=cell[:,:4]-1
    lo=np.column_stack([y[:,i][t].min(1) for i in range(3)]);hi=np.column_stack([y[:,i][t].max(1)for i in range(3)])
    ref=json.loads((vol/'bone-overlap/result.json').read_bytes())['results']
    for name,xf in poses_to_test.items():
        vals={}
        for (xs,vs),k in zip(test,(18,19)):
            f,x,F,v=coords[k];a=move(x,xf);ds=np.r_[nearest((v,F),move(xs,xf),lib)[1],nearest((a,f),vs,lib)[1]]
            world=(Rotation.from_quat(P[128,3:]).apply(a)+P[128,:3]).astype('f4').astype(float);origin=world.mean(0);_,_,bone=closed_solid((world-origin)*1000,f)
            near=((hi>=world.min(0))&(lo<=world.max(0))).all(1);over={}
            for tag,role in [(15,'skin'),(16,'fat')]:
                hit=t[near&(cell[:,4]==tag)];ff=boundary(hit);nd,inv=np.unique(ff,return_inverse=True);tis=solid((y[nd]-origin)*1000,inv.reshape(-1,3));r=tis^bone
                if r.status()!=bone.status():raise ValueError('Failed solid overlap')
                over[role]=max(0.,r.volume())
                if name=='authored' and abs(over[role]-ref[str(k)][role])>1e-5:raise ValueError('Control overlap mismatched')
            vals[str(k)]=dict(held_out_surface_mm=np.quantile(ds*1000,[.50,.9,.99]).tolist(),rigid_tissue_overlap_mm3=over)
        results.append(dict(method=name,transform=xf.tolist(),rotation_deg=float(Rotation.from_matrix(xf[:3,:3]).magnitude()*180/np.pi),translation_mm=float(np.linalg.norm(xf[:3,3])*1000),banks=vals))
    record=dict(bone=digest(asset),native_material=digest(vol/'result.json'),model='myofullbody',source_mesh_sha256=stl,mechanics_comparison=comparison,trials=results);(work/'pelvis-registration.json').write_text(json.dumps(record,indent=2)+'\n');print([(r['method'],r['banks']) for r in results],flush=True)
if __name__=='__main__':run(Path(__file__).resolve().parent,Path(sys.argv[1]),Path(sys.argv[2]))
