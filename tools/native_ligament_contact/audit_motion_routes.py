"""Recheck the source-authored shoulder paths against registered bone solids in motion."""
import csv, hashlib, json, mmap, runpy, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from certify_material_branch import Bone, certify, other_solids, posed, tauten
from numilab_human.capsular_material_registration import read_material_banks, material_bank, emitted_body_vertices

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def pose(t,rot):return np.r_[t,rot.as_quat()]
def inv(q):
    rot=Rotation.from_quat(q[3:]).inv()
    return pose(rot.apply(-q[:3]),rot)
def compose(q,s):
    r=Rotation.from_quat(q[3:])
    return pose(q[:3]+r.apply(s[:3]),r*Rotation.from_quat(s[3:]))
def transform(q,x):
    return q[:3]+Rotation.from_quat(q[3:]).apply(x)
def states(rows):
    return {r['body_index']:np.array(r['position_m']+r['quaternion_xyzw']) for r in rows}
def main(geometry_script,bone,sites,routes,entry_log,pack,motion,output):
    receipts=json.loads(pack.with_suffix('.receipt.json').read_text());step=int(receipts['accepted_step'])
    if digest(pack)!=receipts['pack_file_sha256']:raise ValueError('Accepted geometry hash mismatch')
    endpoint=states(receipts['accepted_body_poses']);rest=states(receipts['initial_anatomical_registration']['body_poses'])
    doc=json.loads(routes.read_text())
    if doc['bones_sha256']!=digest(bone) or doc['sites_sha256']!=digest(sites):raise ValueError('Material route source changed')
    author=next(json.loads(s.partition('=')[2]) for s in entry_log.open() if s.startswith('registered_ligament_actual_pose_json='))
    if author['bonePayloadSha256']!=doc['bones_sha256']:raise ValueError('Wrong ligament entrance')
    entrance=np.array(author['bodyPoses'])
    bystep={}
    for row in csv.DictReader(motion.open()):
        xyz=[float(row[f'root_{a}_m'])+float(row[f'com_relative_root_{a}_m']) for a in 'xyz']
        q=[float(row['orientation_'+a]) for a in 'xyzw']
        bystep.setdefault(int(row['accepted_step']),{})[int(row['body_index'])]=np.r_[xyz,q]
    if set(bystep)!=set(range(1,step+1)) or any(set(s)!=set(endpoint) for s in bystep.values()):
        raise ValueError('Incomplete articulated motion trace')
    discrepancies=[np.linalg.norm(bystep[step][k][:3]-p[:3]) for k,p in endpoint.items()]
    if max(discrepancies)>1e-6:raise ValueError('Terminal motion trace differs from captured bodies')
    bystep[step]=endpoint
    author_sites={row['side']:row['attachments'] for row in json.loads(sites.read_text())['rows']}
    attested={a['stable'] for sides in author_sites.values() for a in sides}
    banks=read_material_banks(bone);reader=runpy.run_path(str(geometry_script))
    meshes={};cat={};frames={};templates={};precision=[];material_mismatch=[]
    ident=np.array([0,0,0,0,0,0,1.])
    with pack.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as handle:
        for (sem,sid),_,world,faces in reader['primitives'](handle,receipts_by_id(receipts)):
            if sem!=51004:continue
            body=banks[2][sid][0];_,rec,attrs,raw_faces=material_bank(banks,sid,body)
            used=np.unique(raw_faces)
            if not np.array_equal(faces,np.searchsorted(used,raw_faces)):raise ValueError(f'Face identity {sid}')
            xyz=emitted_body_vertices(attrs,rec);v=xyz[used];w=world.astype('f8')
            origin=v.mean(0);mean=w.mean(0)
            u,_,vt=np.linalg.svd((w-mean).T@(v-origin))
            mat=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
            fit=pose(mean-mat@origin,Rotation.from_matrix(mat))
            precision.append(float(np.linalg.norm(transform(fit,v)-w,axis=1).max()))
            frames[sid]=compose(inv(endpoint[body]),fit)
            cat[sid]={'body':sid}
            meshes[sid]=(xyz,raw_faces,None,xyz.min(0),xyz.max(0))
            if sid in attested:
                templates[sid]=Bone(meshes[sid],ident,sid)
                frames[sid]=compose(inv(rest[body]),entrance[body])
                calculated=transform(compose(endpoint[body],frames[sid]),v)
                material_mismatch.append((sid,float(np.linalg.norm(calculated-w,axis=1).max())))
    if set(frames)!=set(banks[2]) or max(precision)>1e-6:
        raise ValueError(f'Frozen bone inventory or rigid fit failed: {len(frames)}, {max(precision)}')
    if max(x[1] for x in material_mismatch)>2e-6:
        raise ValueError(f'Entrances disagree with emitted bones: {material_mismatch}')
    reports=[]
    for i,bs in sorted(bystep.items()):
        poses={sid:compose(bs[banks[2][sid][0]],frame) for sid,frame in frames.items()}
        for route in doc['routes']:
            side=route['side'];attach=author_sites[side];s0,s1=[a['stable'] for a in attach]
            bp={s:posed(templates[s],poses[s]) for s in (s0,s1)}
            target=transform(poses[s1],route['target'])
            edges=[(s0,e['edge'],e['t']) for e in route['edges']]
            prefix=transform(poses[s0],np.array(route['prefixPoints']))
            suffix,params=tauten(list(bp.values()),[prefix[-1],target],edges)
            pts=np.vstack((prefix,suffix[1:]))
            full=[*route['prefixEdges'],*edges];t=[a[2] for a in route['prefixEdges']]+list(params)
            c=certify(list(bp.values()),pts,full,t,attach)
            others=other_solids(pts,poses,cat,meshes,bp)
            reports.append(dict(step=i,side=side,length_m=c['length_m'],owner_crossings=c['crossings'],
                                owner_inside=c['inside'],other_bones=others,normal_cone_residual=c['normal_cone_residual_per_unit_tension']))
    report=dict(source_bones_sha256=digest(bone),strip_sha256=digest(routes),capture_sha256=digest(pack),
        trace_sha256=digest(motion),bone_banks=len(frames),captured_terminal_step=step,
        scope='Registered rigid bone solids. Earlier steps use the accepted articulated motion trace. Terminal uses the frozen native body poses. Routes are reconstructed on the emitted source material; nonosseous contacts and branch transitions are outside this check.',
        max_terminal_trace_delta_m=max(discrepancies),max_bone_vertex_fit_error_m=max(precision),shoulder_material_frame_error_m=material_mismatch,rows=reports)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if any(row['owner_crossings'] or row['owner_inside'] or row['other_bones'] or row['normal_cone_residual']>1e-6 for row in reports):
        raise ValueError('Shoulder contact branch failed in sampled motion; see report')
    print('banks',len(frames),'cross',sum(len(i['owner_crossings'])+len(i['owner_inside'])+len(i['other_bones']) for i in reports),
          'cone',max(i['normal_cone_residual'] for i in reports),'cases',len(reports))

def receipts_by_id(receipt):return {r['body_index']:r for r in receipt['accepted_body_poses']}
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
