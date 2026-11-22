"""Compare the actual presented outer skin to registered material and interior anatomy at release."""
import sys,subprocess,hashlib,runpy,json,time
import re
from collections import Counter
from pathlib import Path
import numpy as np
def scan_banks(audit,raw,poses,vertices,faces,result,out,filename,check_skin=False):
    c=audit['clearance'];intersections=audit['intersections'];surface=audit['SKIN']
    records=c._exact_surface_records(vertices,faces)
    idx=intersections._prepare_surface_aabb(records);floatidx=c._float32_skin_aabb_index(records)
    pairs=Counter();banks=Counter();stats=[];start=time.monotonic()
    if check_skin:
        result['skin_self_intersections']=intersections._audit_pair_prepared_first(idx,records,same_surface=True)['count']
    with (out/'target-witnesses.jsonl').open('w') as output:
        for key,i,x,f in audit['primitives'](raw,poses):
            if key==surface:continue
            chosen,broad=c._candidate_target_rows_for_skin_aabb(f,x,floatidx)
            selected=np.asarray(chosen,dtype=int)
            ids=np.unique(f[selected]);local=np.searchsorted(ids,f[selected])
            rec=c._exact_surface_records(x[ids],local)
            rec=[(a,b,c0,int(selected[k]),e) for a,b,c0,k,e in rec]
            hit=intersections._audit_pair_prepared_first(idx,rec)
            if hit['aabb_candidate_pairs']!=broad:raise ValueError('Broadphase ledger mismatch')
            ocular=key[0]==51010 and 381<=key[1]<=397
            label='ocular' if ocular else str(key[0]);banks[label]+=1;pairs[label]+=hit['count']
            if hit['count']:
                row=dict(semantic=key[0],stable_id=key[1],primitive=i,ocular_source_id=ocular,**hit);output.write(json.dumps(row,separators=(',',':'))+'\n');output.flush()
                stats.append([key[0],key[1],i,hit['count']])
    result.update(target_banks_by_semantic=dict(banks),intersection_pairs_by_semantic=dict(pairs),nonocular_pairs=sum(v for k,v in pairs.items() if k!='ocular'),affected_banks=stats,wall_seconds=time.monotonic()-start)
    (out/filename).write_text(json.dumps(result,indent=2)+'\n')

def release_root_error(body_rows, log):
    """Compare the recorded registration or execution frame with the executed starting root."""
    if not log.is_file():return None
    rows=re.findall(r'^resting_initial_support_alignment=applied .*?executed_root_xyz_m=(\[[^\]]+\])',log.read_text(),re.M)
    if not rows:return None
    if any(row!=rows[0] for row in rows):raise ValueError('Ambiguous initial root placement')
    executed=np.asarray(json.loads(rows[0]),dtype=float)
    root=[r['current']['position_world_m'] for r in body_rows if r['body_index']==0]
    if len(root)!=1 or executed.shape!=(3,) or not np.isfinite(executed).all() or not np.isfinite(root).all():
        raise ValueError('Invalid executed or supplied root')
    return float(np.linalg.norm(executed-root[0]))

def at_release():
    impl,pack,identity,pose,reference,current,out=map(Path,sys.argv[1:8])
    sys.path.insert(0,str(impl.parent/'src'))
    audit=runpy.run_path(str(impl));c=audit['clearance'];intersections=audit['intersections'];surface=audit['SKIN']
    one,two=[json.loads(p.read_text()) for p in (reference,current)]
    # Both images start from the same native executable, physiology and anatomical payloads.
    if one['argv']!=two['argv'] or one['pins']!=two['pins']:raise ValueError('Native input identities differ')
    def project(env):return {k:v for k,v in env.items() if k not in ('NUMI_HUMAN_SKIN_REFERENCE_ALL_BODY_POSES','NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP')}
    if project(one['environment'])!=project(two['environment']):raise ValueError('Native environment differs')
    if two['environment']['NUMI_HUMAN_SKIN_REFERENCE_ALL_BODY_POSES']!='1': raise ValueError('Incomplete body-pose capture')
    if Path(two['environment']['NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP']) != pose:
        # The RAM volume is released after the native run. An independently retained
        # reference capture authenticates its relocated byte-identical copy.
        if len(sys.argv)!=9 or hashlib.sha256(pose.read_bytes()).hexdigest()!=json.loads(Path(sys.argv[8]).read_text())['pose_sha256']:
            raise ValueError('Unbound pose capture')
    def sha(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    p=json.loads(pose.read_text());poses={r['body_index']:{'position_m':r['current']['position_world_m'],'quaternion_xyzw':r['current']['orientation_world_xyzw']} for r in p['all_bodies']}
    assert len(poses)==len(p['all_bodies']) and sorted(poses)==list(range(len(poses)))
    raw=subprocess.check_output(['/opt/homebrew/bin/zstd','-q','-dc',str(pack)]);pkhash=hashlib.sha256(raw).hexdigest()
    if pkhash!=json.loads(identity.read_text())['sha256']:raise ValueError('Initial pack SHA mismatch')
    ((_,_,vertices,faces),)=audit['primitives'](raw,poses,skin_only=True)
    pfx='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_O_'
    nodes=Path(one['environment'][pfx+'NODES_CSV']); indices=Path(one['environment'][pfx+'FACES_CSV'])
    if sha(nodes)!=one['environment'][pfx+'NODES_SHA256'] or sha(indices)!=one['environment'][pfx+'FACES_SHA256']: raise ValueError('Surface input drift')
    xyz=np.loadtxt(nodes,skiprows=1,delimiter=',',usecols=(2,3,4)).astype('f4')
    tri=np.loadtxt(indices,skiprows=1,delimiter=',',usecols=(1,2,3),dtype='i4')
    if len(vertices)!=len(xyz) or len(tri)!=len(faces): raise ValueError('Presentation has different physical skin dimensions')
    error=np.linalg.norm(vertices.astype('f8')-xyz.astype('f8'),axis=1)
    root_error=release_root_error(p['all_bodies'],current.parent/'stdout.log')
    result=dict(scope='native opening mesh in supplied anatomical registration coordinates',executed_root_error_m=root_error,pose_count=len(poses),pose_sha256=sha(pose),pack_sha256=pkhash,skin_vertices=len(vertices),skin_faces=len(faces),same_triangle_indices=bool(np.array_equal(tri,faces)),max_skin_position_difference_m=float(error.max()),changed_skin_positions=int((error>0).sum()))
    out.mkdir(exist_ok=True);(out/'displayed-skin.json').write_text(json.dumps(result,indent=2)+'\n')
    if root_error is not None and root_error>2e-6:
        raise ValueError(f'Registration root differs from executed placement by {root_error:.9g} m; use the native executed pose capture')
    if not result['same_triangle_indices'] or error.max()>2e-7:raise ValueError('Rendered-to-material mismatch; retain parity witness before wider comparison')
    scan_banks(audit,raw,poses,vertices,faces,result,out,'initial-envelope.json')

def at_terminal(impl,pack,receipt,admission,launch,out):
    from emit_skin_source_registration import sha
    sys.path.insert(0,str(impl.parent/'src'))
    audit=runpy.run_path(str(impl))
    report=json.loads(admission.read_text());state=json.loads(receipt.read_text());job=json.loads(launch.read_text())
    n=report['accepted_steps']
    identity=state['pack_file_sha256']
    if (report['exit_code']!=0 or report['requested_roots']!=n or n!=state['accepted_step'] or
        state['physical_endpoint']!='accepted' or state['surface_audit_endpoint']!='passed' or
        report['terminal_pack_sha256']!=identity or report['terminal_archive_sha256']!=sha(pack) or
        Path(state['accepted_pack_path'])!=Path(job['argv'][4])/'accepted-geometry'/f'step-{n}.mrvpack'):
        raise ValueError('Endpoint, source launch or physical transaction mismatch')
    raw=subprocess.check_output(['/opt/homebrew/bin/zstd','-q','-dc',str(pack)])
    if hashlib.sha256(raw).hexdigest()!=identity:raise ValueError('Terminal pack digest mismatch')
    poses={r['body_index']:r for r in state['accepted_body_poses']}
    if len(poses)!=state['accepted_body_pose_count'] or sorted(poses)!=list(range(len(poses))):
        raise ValueError('Incomplete accepted articulated pose')
    ((_,_,x,faces),)=audit['primitives'](raw,poses,skin_only=True)
    pfx='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_O_';env=job['environment']
    node=Path(env[pfx+'NODES_CSV']);facet=Path(env[pfx+'FACES_CSV'])
    if sha(node)!=env[pfx+'NODES_SHA256'] or sha(facet)!=env[pfx+'FACES_SHA256']:raise ValueError('Changed physical reference')
    xyz=np.loadtxt(node,skiprows=1,delimiter=',',usecols=(2,3,4),dtype='f4')
    tri=np.loadtxt(facet,skiprows=1,delimiter=',',usecols=(1,2,3),dtype='i4')
    if xyz.shape!=x.shape or not np.array_equal(tri,faces):raise ValueError('Accepted skin topology differs from the registered physical boundary')
    error=np.linalg.norm(x.astype('f8')-xyz.astype('f8'),axis=1)
    result=dict(scope='one accepted physical endpoint',accepted_step=n,accepted_time_s=state['accepted_time_s'],
                pack_sha256=identity,pose_count=len(poses),accepted_root_fingerprint_hex=state['accepted_root_fingerprint_hex'],
                registered_skin_vertices=len(x),changed_from_reference=int(np.count_nonzero(error)),
                maximum_skin_motion_m=float(error.max()),rms_skin_motion_m=float(np.sqrt(np.mean(error**2))))
    out.mkdir(exist_ok=True)
    scan_banks(audit,raw,poses,x,faces,result,out,'accepted-envelope.json',check_skin=True)

if __name__=='__main__':
    if sys.argv[1:2]==['--terminal']:at_terminal(*(Path(a) for a in sys.argv[2:]))
    else:at_release()
