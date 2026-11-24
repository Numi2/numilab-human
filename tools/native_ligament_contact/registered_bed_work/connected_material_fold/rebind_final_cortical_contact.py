"""Rebuild the existing native cortical contact program on the compact, validated volume."""
from pathlib import Path
import json,os
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
from material_cavity_flips import faces
from compile_connected_joint import digest
from inspect_shoulder_source import nearest
from compare_supported_postures import poses
from rebind_ligament_material import banks
from compile_cortical_skin_contact import run as compile_contact
from build_cortical_skin_registration import write_contact_include
def run(q,lib):
    p=q.parent;j=p.parent;src=q/'qualified-material';out=q/'registered-material-contacts';out.mkdir(exist_ok=True)
    doc=json.loads((src/'result.json').read_bytes());x=np.loadtxt(src/'mesh.node',skiprows=1,usecols=(1,2,3));y=np.loadtxt(src/'current.node',skiprows=1,usecols=(1,2,3))
    for name in ('mesh.ele','mesh.node','current.node'):
        if digest(src/name)!=doc['inputs'][name] or doc['exit']!=0:raise ValueError('Unverified material body')
    current=np.load(src/'material-walls.npz');C=current['C'];ori=np.load(p/'binding/cortex/registered-cortical-facets.npz')['cell_side_material_bank_nodes'];op=np.load(q/'obstacle-seam/volume-and-facets.npz');keep=np.setdiff1d(np.arange(len(ori)),np.setdiff1d(op['old_cortex_faces'],op['surviving_cortex_source_faces']))
    srcface=keep
    for path in [q/'consolidated-fold',q/'remaining-fold/collar-2/consolidated-fold']:
        z=np.load(path/'surface-ownership.npz');before=z['old_roles_rows'];after=z['new_roles_rows_nodes'];removed=before[before[:,0]==3,1];added=after[after[:,0]==3,1]
        srcface=np.r_[np.delete(srcface,removed),srcface[added]]
    if len(srcface)!=len(C):raise ValueError('Unmatched source wall')
    t=np.loadtxt(src/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4');ele=t[:,:4]-1;sel=np.flatnonzero(np.isin(ele,np.unique(C)).sum(1)>=3)
    keys={tuple(sorted(f)):[]for f in C}
    for i in sel:
        for k,face in enumerate(faces(ele[i:i+1])):
            tkey=tuple(sorted(face))
            if tkey in keys:keys[tkey].append((i,k))
    local=np.array([keys[tuple(sorted(f))][0]for f in C])
    if len(keys)!=len(C) or any(len(row)!=1 for row in keys.values()):raise ValueError('Nonunique cortical surface owner')
    entity=ori[srcface,3];mat=t[local[:,0],4]
    if not np.array_equal(mat,ori[srcface,2]):raise ValueError('Wrong cortical material')
    pose=j/'distal-registered-native/target-pose.json';P=poses(pose);cfg=json.loads((q/'gravity-scene/launch.json').read_bytes());bone=Path(cfg['argv'][3])
    if digest(bone)!=cfg['pins'][str(bone)]:raise ValueError('Changed rigid source')
    topology={sid:(body,f,v)for (body,sid),(f,v,_) in banks(bone).items()}
    quad=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    bodies=np.empty(len(C),'i4');triids=np.empty((len(C),3),'i4');bary=np.empty((len(C),3,3));dist=np.empty((len(C),3))
    for bank in np.unique(entity):
        body,f,v=topology[bank];selection=entity==bank;world=(Rotation.from_quat(P[body,3:]).apply(v)+P[body,:3]).astype('f4').astype('f8');query=(quad@y[C[selection]]).reshape(-1,3)
        target,d,index=nearest((world,f),query,lib);T=world[f[index]];bc=np.einsum('ijk,ik->ij',np.linalg.pinv((T[:,1:]-T[:,:1]).transpose(0,2,1)),target-T[:,0])
        weights=np.c_[1-bc.sum(1),bc]
        if weights.min()<-2e-6 or np.linalg.norm(np.einsum('ij,ijk->ik',weights,T)-target,axis=1).max()>2e-8:raise ValueError('Unresolved cortical landmark')
        triids[selection]=index.reshape(-1,3);bary[selection]=weights.reshape(-1,3,3);dist[selection]=d.reshape(-1,3);bodies[selection]=body
    dest=out/'cortex';dest.mkdir(exist_ok=True);mapping=dest/'registered-cortical-facets.npz'
    np.savez_compressed(mapping,cell_side_material_bank_nodes=np.c_[local,mat,entity,C].astype('i4'),body=bodies,quadrature_barycentric=quad,cortex_faces=triids,cortex_barycentric=bary,loaded_distance_m=dist,source_facet_row=srcface)
    (dest/'result.json').write_text(json.dumps(dict(mapping=digest(mapping),bone=digest(bone),pose=digest(pose),geometry=digest(src/'current.node'),elements=digest(src/'mesh.ele'),facets=len(C),quadrature_points=int(dist.size),nearest_checker=digest(lib)),indent=2)+'\n')
    # Preserve the deep-boundary area-weighted thickness when derived vertices merge.
    # All other boundary thickness fields are sampled from their original rows.
    tables=out/'tables';tables.mkdir(exist_ok=True);job=out/'physical-owner';(job/'registered-boundary').mkdir(parents=True,exist_ok=True)
    if not (job/'posed-app').exists():(job/'posed-app').symlink_to(j/'posed-app',target_is_directory=True)
    original=np.load(j/'registered-boundary/registered-boundary.npz');ids=np.load(q/'seam-volume/local-ancestry.npz')['surviving_source_nodes'];alias=dict(zip(op['retired_vertices'],op['surviving_vertices']))
    oldx=np.loadtxt(p/'relative/volume/mesh.node',skiprows=1,usecols=(1,2,3));files={};bound={};weight_errors={}
    env=cfg['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    for n in 'OID':
        orig=original[n+'_node_joint'];rem=np.array([alias.get(i,i)for i in orig]);index=np.searchsorted(ids,rem)
        if not np.array_equal(ids[index],rem):raise ValueError('Unknown surviving boundary node')
        unique,inv=np.unique(index,return_inverse=True);tri=original[n+'_face_nodes'];pts=oldx[orig[tri]];area=np.linalg.norm(np.cross(pts[:,1]-pts[:,0],pts[:,2]-pts[:,0]),axis=1)/6
        aw=sum(np.bincount(col,weights=area,minlength=len(orig))for col in tri.T)
        if min(aw)<=0:raise ValueError('No contributing boundary area')
        tag='ROLE_'+n+'_CONTACT_THICKNESS';path=Path(env[prefix+tag+'_CSV'])
        if digest(path)!=env[prefix+tag+'_SHA256']:raise ValueError('Unpinned tissue thickness')
        a=np.loadtxt(path,skiprows=1,delimiter=',');weights=np.bincount(inv,weights=aw,minlength=len(unique))
        combined=np.column_stack([np.bincount(inv,weights=aw*col)/weights for col in a[:,1:].T]);weight_errors[n]=float(np.max(abs(combined.T@weights-a[:,1:].T@aw)))
        new=tables/(n+'-thickness.csv')
        with new.open('w')as stream:
            stream.write(path.open().readline());np.savetxt(stream,np.c_[np.arange(len(unique)),combined],fmt=['%d']+['%.17g']*combined.shape[1],delimiter=',')
        bound[n+'_node_joint']=unique;files[tag]={'path':str(new),'sha256':digest(new)}
    np.savez_compressed(job/'registered-boundary/registered-boundary.npz',**bound);(tables/'result.json').write_text(json.dumps(dict(files=files,weighted_thickness_errors=weight_errors),indent=2)+'\n')
    target=out/'compiled';compile_contact(j.parent/'ligament-equilibrium-070',job,tables,target,src,src/'current.node',dest)
    app=out/'app';app.mkdir(exist_ok=True);write_contact_include(target,app)
    counts=Counter(map(int,entity));actual=json.loads((src/'bone-overlap/result.json').read_bytes())['results']
    summary=dict(elements=doc['inputs']['mesh.ele'],material_report=digest(src/'result.json'),contact_program=digest(target/'material-contact.npz'),native_include=digest(app/'NumiHumanCorticalSkinBinding.inc'),mapped_banks=dict(counts),mapped_faces=len(C),quadrature_points=int(dist.size),rigid_banks_with_positive_overlap_and_no_explicit_wall=[int(k)for k,r in actual.items() if max(r['skin'],r['fat'])>1e-6 and int(k) not in counts],phase='Rebuilt explicit cortical contact program on the imported material. The coupled driver and remaining source-credit tables require binding to the same revision.')
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary,flush=True)
if __name__=='__main__':
    import sys
    run(Path(__file__).resolve().parent,Path(sys.argv[1]))
