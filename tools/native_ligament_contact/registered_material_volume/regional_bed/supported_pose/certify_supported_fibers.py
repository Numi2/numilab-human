"""Re-certify existing source-owned cortical fibers in the actual regional-bed release pose."""
from pathlib import Path
import json,re,subprocess,shutil
import numpy as np
from scipy.spatial.transform import Rotation
from audit_fiber_load_bounds import read_pose
from rebind_ligament_material import banks
from certify_cortical_paths import Cortex,verify
import certify_cortical_paths
from prepare_live_cortex import fl
from compile_connected_joint import digest
def registration(out,cfg,actual_pose_log,integrated_reference_log):
    h=Path('active-ligament-header.txt').read_text().strip();header=Path(h);unit=header.with_name('NumiHumanCorticalRegistration.inc')
    source=Path(cfg['argv'][3]);geom=banks(source)
    basepath=Path(certify_cortical_paths.__file__).parent
    old=read_pose(basepath.parent/'native-ligament-material/stderr.log');now=read_pose(actual_pose_log)
    if old['bonePayloadSha256']!=digest(source)or now['bonePayloadSha256']!=digest(source)or len(geom)!=185:raise ValueError('Cortical source changed')
    poses=np.asarray(now['bodyPoses'],dtype='f4');base=np.asarray(old['bodyPoses'],dtype='f4')
    if poses.shape!=base.shape or poses.shape!=(157,7):raise ValueError('Body table changed')
    transform=Rotation.from_quat(poses[0,3:])*Rotation.from_quat(base[0,3:]).inv()
    src_origin=base[0,:3].astype(float)
    dst_origin=poses[0,:3].astype(float)
    rigid_world=lambda point:transform.apply(point-src_origin)+dst_origin
    # The accepted integrated release already has a different forearm pose from the original shoulder source.
    # Check the complete body relative to the integrated release, and each source fiber against its own source.
    joined=read_pose(integrated_reference_log)
    if joined['bonePayloadSha256']!=digest(source):raise ValueError('Joined source identity changed')
    previous=np.array(joined['bodyPoses'],float)
    group=Rotation.from_quat(poses[0,3:])*Rotation.from_quat(previous[0,3:]).inv()
    body_motion_error=float(np.linalg.norm(group.apply(previous[:,:3]-previous[0,:3])+poses[0,:3]-poses[:,:3],axis=1).max())
    body_rotation_error=float(((Rotation.from_quat(poses[:,3:])*Rotation.from_quat(previous[:,3:]).inv())*group.inv()).magnitude().max())
    # A distant moving bone is checked in its actual pose below. Each fiber must still undergo a single rigid transform across all of its material owners.
    ref=unit.read_text();pattern=r'^\{"([^"]+)",(\d+),(\d+)u,([^,]+),\{([^}]+)\},\{([^}]+)\}\},$'
    fibers={ (name,int(side)):(int(tag),float.fromhex(L),[float.fromhex(x)for x in begin.split(',')],[float.fromhex(x)for x in end.split(',')]) for name,side,tag,L,begin,end in re.findall(pattern,ref,re.M)}
    if len(fibers)!=6:raise ValueError('Expected six certified source fibers')
    world={};cache={};lines=[];report=[]
    for key,(faces,v,raw)in geom.items():
        pv=now['bodyPoses'][key[0]];world[key]=Rotation.from_quat(pv[3:]).apply(v)+pv[:3]
    for p in sorted((basepath/'oriented-routes').glob('*.json')):
        row=json.loads(p.read_bytes());section=json.loads((basepath/'section-routes'/p.name).read_bytes());fiber,side=p.stem.rsplit('-',1);tag,L,start,end=fibers[fiber,int(side)]
        if row['registered_bones']!=digest(source) or row['section']!=digest(basepath/'section-routes'/p.name) or row['relaxed_interiors']:raise ValueError('Foreign material path')
        pts=np.asarray(row['world_m']);sites=section['sites'];edges=row['edges']
        if sites[0]['body']!=(84 if int(side)==0 else 34):
            pts=pts[::-1];sites=sites[::-1];edges=edges[::-1]
        ownership=[sites[0]['body']]+[r['body']for r in edges]+[sites[1]['body']]
        if len(ownership)!=len(pts)or np.max(abs(pts[[0,-1]]-[start,end]))>2e-7:raise ValueError('Force endpoints differ from pinned material')
        anchor=ownership[0]
        before=Rotation.from_quat(old['bodyPoses'][anchor][3:])
        after=Rotation.from_quat(now['bodyPoses'][anchor][3:])
        fiber_rotation=after*before.inv()
        moved=[];directions=[]
        for q,b in zip(pts,ownership):
            x=old['bodyPoses'][b];y=now['bodyPoses'][b];U=Rotation.from_quat(x[3:]);V=Rotation.from_quat(y[3:])
            moved.append(V.apply(U.inv().apply(q-x[:3]))+y[:3]);directions.append(((V*U.inv())*fiber_rotation.inv()).magnitude())
        moved=np.asarray(moved);length=float(np.linalg.norm(np.diff(moved,axis=0),axis=1).sum());drift=float(np.max(np.linalg.norm((moved-moved[0])-fiber_rotation.apply(pts-pts[0]),axis=1)))
        if drift>2e-7 or max(directions)>1e-6 or abs(L-length)>2e-7:raise ValueError(('Requires fresh sliding-path fit',p.name,drift,L,length))
        predicted=fiber_rotation.apply(pts-old['bodyPoses'][anchor][:3])+now['bodyPoses'][anchor][:3]
        if np.max(np.linalg.norm(moved-predicted,axis=1))>2e-7:
            raise ValueError(('Fiber material owners moved relative to each other',fiber,side))
        near=[]
        for key,v in world.items():
            if np.any(v.max(0)+1e-6<moved.min(0))or np.any(moved.max(0)+1e-6<v.min(0)):continue
            if key not in cache:cache[key]=Cortex(*key,geom[key],now['bodyPoses'][key[0]])
            near.append(key)
        defects=verify(moved*1000,[cache[k]for k in near])
        if defects:raise ValueError(('New cortical interior',p.name,defects[:2]))
        # Check the material provenance again from source face/barycentric coordinates at both ends.
        offsets=[]
        for site,q in zip(sites,moved[[0,-1]]):
            k=(site['body'],site['bone']);f=geom[k][0];expected=np.array(site['barycentric'])@world[k][f[site['triangle']]]
            offsets.append(float(np.linalg.norm(q-expected)))
        if max(offsets)>2e-7:raise ValueError(('Material site moved',p.name,offsets))
        vec=lambda q:'{'+','.join(float(x).hex()for x in q)+'}'
        lines.append('{"'+fiber+'",'+side+','+str(tag)+'u,'+length.hex()+','+vec(moved[0])+','+vec(moved[-1])+'},')
        report.append(dict(fiber=fiber,side=int(side),edge_count=len(row['edges']),reference_length_m=L,certified_length_m=length,
                        maximum_nonrigid_change_m=drift,material_endpoint_residual_m=offsets,nearby_banks=[k[1]for k in near],crossings=defects))
    if len(lines)!=6:raise ValueError('Missing fibers')
    text=ref.split('inline constexpr RegisteredCorticalFiber registeredCorticalFibers[]=')[0]+'inline constexpr RegisteredCorticalFiber registeredCorticalFibers[]={\n'+'\n'.join(lines)+'\n};\ninline constexpr float registeredCorticalPose[][7]={\n'+',\n'.join('{'+','.join(fl(v)for v in line)+'}'for line in poses)+'\n};\n'
    (out/unit.name).write_text(text);shutil.copy2(header,out/header.name)
    result=dict(source_geometry=digest(source),source_cook=digest(unit),actual_release=digest(actual_pose_log),pose_count=len(poses),bone_count=len(geom),maximum_body_position_rigid_error_m=body_motion_error,maximum_body_rotation_rigid_error_rad=body_rotation_error,paths=report,candidate_inc=digest(out/unit.name))
    result['integrated_reference_pose']=digest(integrated_reference_log)
    (out/'registered-fibers.json').write_text(json.dumps(result,indent=2)+'\n');print([(r['fiber'],r['side'],max(r['material_endpoint_residual_m'])) for r in report],flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('launch','pose_log','reference_log','output'):parser.add_argument(key,type=Path)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=True)
    registration(args.output,json.loads(args.launch.read_bytes()),args.pose_log,args.reference_log)
