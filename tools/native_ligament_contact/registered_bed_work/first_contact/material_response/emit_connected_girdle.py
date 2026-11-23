"""Emit coordinated shoulder registrations and compare the complete skeleton at supported release."""
from pathlib import Path
from itertools import combinations
import json
import struct
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from numilab_human.model import _bodyparts_visual_local_pose
from audit_registered_skeleton import HEADER, RECORD, closed_solid, digest, intersection_mm3
from compile_connected_joint import matrix


def write(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def run(job, fits, compiled, original_registration, control_audit, out):
    cfg=json.loads((job/'launch.json').read_bytes())
    release=json.loads((job/'native-initial-bodies.json').read_bytes())['all_bodies']
    src=Path(cfg['argv'][3]);meta=src.with_suffix('.manifest.json')
    manifest=json.loads(meta.read_bytes());registration=json.loads(original_registration.read_bytes())
    control=json.loads(control_audit.read_bytes())
    raw=src.read_bytes();data=bytearray(raw)
    if digest(src)!=manifest['payload']['sha256'] or digest(src)!=cfg['pins'][str(src)] or digest(original_registration)!=manifest['source']['registration']['sha256']:
        raise ValueError('Registration identity mismatch')
    rigid=compiled/'source-bound.nhrigid'
    info=json.loads((compiled/'result.json').read_bytes())
    if info['candidate']!=digest(rigid) or info['control']!=digest(Path(cfg['argv'][1])):
        raise ValueError('Stale articulated bindings')
    mesh={};registered={};forecasts={};groups={}
    out.mkdir()
    for side,ids,joint in [('l',[9,11],84),('r',[8,10],34)]:
        fitpath=fits/(side+'.json')
        if digest(fitpath)!=info['selected'][side]:raise ValueError('Unmatched candidate')
        c=json.loads(fitpath.read_bytes()); origin=np.asarray(c['origin_m'])
        transform={}
        for key,a in c['girdle_transforms_mm'].items():
            a=np.array(a);a[:3,3]=origin-a[:3,:3]@origin+.001*a[:3,3]
            transform[key]=a
        groups[joint]=transform['arm']
        for sid,k in zip(ids,['cl','sc']):
            anchor=registration['anchors'][sid-1]; b=anchor['target']['core_body_index']
            pose=release[b]['current']
            world=matrix(pose['position_world_m']+pose['orientation_world_xyzw'])
            offset=np.linalg.solve(world,transform[k]@world)
            reg=anchor['registration'];prior=np.array(reg['source_obj_mm_to_core_inertial_body_m'])
            result=offset@prior
            reg['source_obj_mm_to_core_inertial_body_m']=result.tolist()
            t,q,scale=_bodyparts_visual_local_pose(result.tolist(),str(sid))
            struct.pack_into('<8f',data,HEADER.size+(sid-1)*RECORD.size+24,*t,*q,scale)
            registered[sid]=dict(body=b,source=anchor['source']['member_id'],body_local_transform=offset.tolist())
            forecasts[sid]=transform[k]
    magic,version,n,v,i,_,_=HEADER.unpack_from(data)
    assert (magic,version,n)==(b'NHBONES1',3,185)
    vo=HEADER.size+RECORD.size*n;io=vo+v*24
    def owners(body):
        while body:
            if body-1 in groups:return body-1
            parent,child=struct.unpack_from('<2I',rb,224+157*160+(body-1)*144)
            if child!=body or parent>=body:raise ValueError('Disconnected joint tree')
            body=parent
        return None
    rb=rigid.read_bytes()
    poses0=np.frombuffer(rb,'<f4',103*7,len(rb)-103*28).reshape(-1,7)
    pose_ids=np.frombuffer(rb,'<u4',103,len(rb)-103*32)
    oracle=dict(zip(map(int,pose_ids),poses0))
    shapes={};originals={};names={};max_error=0.
    for row in range(n):
        o=HEADER.size+row*RECORD.size
        rec=RECORD.unpack_from(data,o);ctrl=RECORD.unpack_from(raw,o)
        body,start,count,start_index,indices,sid=rec[:6]
        x=np.frombuffer(raw,'<f4',6*count,vo+24*start).reshape(-1,6)[:,:3]
        tri=np.frombuffer(raw,'<u4',indices,io+4*start_index).astype('i8').reshape(-1,3)-start
        if np.any(tri<0) or np.any(tri>=count):raise ValueError('Out-of-bank facet')
        def local(record):
            return Rotation.from_quat(record[9:13]).apply(record[13]*x.astype(float))+record[6:9]
        old,new=local(ctrl),local(rec)
        state=release[body]['current'];W=matrix(state['position_world_m']+state['orientation_world_xyzw'])
        group=owners(body)
        before=old@W[:3,:3].T+W[:3,3]
        live=(groups[group]@W) if group is not None else W
        dest=new@live[:3,:3].T+live[:3,3]
        exact=forecasts.get(sid,groups.get(group,np.eye(4)))
        max_error=max(max_error,float(np.linalg.norm(dest-(before@exact[:3,:3].T+exact[:3,3]),axis=1).max()))
        if body in oracle:rest=matrix(oracle[body])
        else:
            if group is not None:raise ValueError('Missing recalibrated source rest pose')
            p=release[body]['rest'];rest=matrix(p['position_world_m']+p['orientation_world_xyzw'])
        mesh[sid]=new@rest[:3,:3].T+rest[:3,3]
        if sid in registered:
            reg=registration['anchors'][sid-1]['registration']
            centroid=mesh[sid].mean(0)
            reg.update(default_pose_vertex_centroid_world_m=centroid.tolist(),vertex_centroid_to_source_com_residual_m=float(np.linalg.norm(centroid-rest[:3,3])))
        shapes[sid]=closed_solid(dest.astype('f4').astype(float)*1000,tri)
        originals[sid]=closed_solid(before.astype('f4').astype(float)*1000,tri)
        names[sid]=(manifest['source']['anchors'][row]['member_id'],body)
    if max_error>2e-6:raise ValueError('Emitted Float32 cortical geometry differs from requested registration')
    registration['regional_derivation']['connected_shoulder_registration']=dict(parent_sha256=digest(original_registration),rigid_sha256=digest(rigid),changed=registered)
    regpath=out/'connected-source.json';write(regpath,registration)
    struct.pack_into('<I',data,24,int(digest(regpath)[:8],16))
    payload=out/src.name;payload.write_bytes(data)
    manifest['payload'].update(sha256=digest(payload),registration_fingerprint32=digest(regpath)[:8])
    manifest['source']['registration']=dict(file=regpath.name,sha256=digest(regpath))
    manifest['source']['connected_shoulder_registration']=dict(parent_bone_sha256=digest(src),joint_asset_sha256=digest(rigid),joint_compilation_sha256=digest(compiled/'result.json'))
    by_name={name:sid for sid,(name,b) in names.items()}
    gaps={}
    for entry in manifest['upper_limb_continuity']['transitions']:
        a,b=map(by_name.__getitem__,entry['source_member_ids']);gap=float(cKDTree(mesh[a]).query(mesh[b])[0].min())
        entry['minimum_vertex_gap_m']=gap;gaps[entry['name']]=gap<=entry['maximum_allowed_gap_m']
        entry['status']='passed' if gaps[entry['name']] else 'failed'
    manifest['status']='experimental_connected_joint_registration'
    write(out/meta.name,manifest)
    delta=[];original_pairs=set();new_pairs=set()
    for a,b in combinations(shapes,2):
        if names[a][1]==names[b][1]:continue
        old=intersection_mm3(originals[a],originals[b]);new=intersection_mm3(shapes[a],shapes[b])
        if old>1e-9:original_pairs.add((a,b))
        if new>1e-9:new_pairs.add((a,b))
        if old>1e-9 or new>1e-9:
            delta.append(dict(banks=[a,b],original_mm3=old,compiled_mm3=new))
    prior={tuple(sorted(t['stable_ids'])) for t in control['pairs'] if not t['same_rigid_owner'] and t['native_mm3']>1e-9}
    if prior!=original_pairs:raise ValueError(f'Control pair inventory differs: {prior^original_pairs}')
    write(out/'cortical-admission.json',dict(source=digest(src),candidate=digest(payload),max_emission_error_m=max_error,
        original=len(original_pairs),candidate_pair_count=len(new_pairs),removed=sorted(original_pairs-new_pairs),
        introduced=sorted(new_pairs-original_pairs),gaps=gaps,pairs=delta))
    print('source',len(original_pairs),'candidate',len(new_pairs),'removed',len(original_pairs-new_pairs),'introduced',len(new_pairs-original_pairs),'failed gap tests',sum(not v for v in gaps.values()))


if __name__=='__main__':run(*map(Path,sys.argv[1:]))
