"""Compare the actual presented outer skin to registered material and interior anatomy at release."""
import sys,subprocess,hashlib,runpy,json,time
from collections import Counter
from pathlib import Path
import numpy as np
impl,pack,identity,pose,reference,current,out=map(Path,sys.argv[1:])
sys.path.insert(0,str(impl.parent/'src'))
audit=runpy.run_path(str(impl));c=audit['clearance'];intersections=audit['intersections'];surface=audit['SKIN']
one,two=[json.loads(p.read_text()) for p in (reference,current)]
# Both images start from the same native executable, physiology and anatomical payloads.
if one['argv']!=two['argv'] or one['pins']!=two['pins']:raise ValueError('Native input identities differ')
def project(env):return {k:v for k,v in env.items() if k not in ('NUMI_HUMAN_SKIN_REFERENCE_ALL_BODY_POSES','NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP')}
if project(one['environment'])!=project(two['environment']):raise ValueError('Native environment differs')
if Path(two['environment']['NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP'])!=pose or two['environment']['NUMI_HUMAN_SKIN_REFERENCE_ALL_BODY_POSES']!='1':raise ValueError('Unbound pose capture')
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
result=dict(scope='initial native pack with exact full-volume release-pose body transforms',pose_count=len(poses),pose_sha256=sha(pose),pack_sha256=pkhash,skin_vertices=len(vertices),skin_faces=len(faces),same_triangle_indices=bool(np.array_equal(tri,faces)),max_skin_position_difference_m=float(error.max()),changed_skin_positions=int((error>0).sum()))
out.mkdir(exist_ok=True);(out/'displayed-skin.json').write_text(json.dumps(result,indent=2)+'\n')
if not result['same_triangle_indices'] or error.max()>2e-7:raise ValueError('Rendered-to-material mismatch; retain parity witness before wider comparison')
records=c._exact_surface_records(vertices,faces)
idx=intersections._prepare_surface_aabb(records);floatidx=c._float32_skin_aabb_index(records)
pairs=Counter();banks=Counter();stats=[];start=time.monotonic()
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
(out/'initial-envelope.json').write_text(json.dumps(result,indent=2)+'\n')
