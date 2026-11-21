"""Classify actual crossings and isolate the input surface from physical layer transfer."""
import json,sys,runpy,subprocess,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from emit_skin_source_registration import sha
impl,owner,run,freeze,source_obs,pack,pose,prior,out=map(Path,sys.argv[1:])
sys.path[:0]=[str(impl.parent/'src'),str(owner/'tools')]
from rebind_human_skin_reference import decode
audit=runpy.run_path(str(impl));c=audit['clearance'];ci=audit['intersections']
j=json.loads(run.read_text());receipt=json.loads(freeze.read_text())
skin=Path(j['argv'][j['argv'].index('--skin-payload')+1])
tissue=Path(j['argv'][j['argv'].index('--soft-tissue-payload')+1])
manifest=json.loads(tissue.with_suffix('.manifest.json').read_text())
assert sha(skin)==receipt['current']['sha256'] and sha(tissue)==manifest['payload']['sha256']
assert sha(source_obs)==receipt['current_capture_sha256']
names={r['stable_id']:r for r in manifest['source']['surfaces']}
data=decode(skin.read_bytes());xyz=np.loadtxt(source_obs,skiprows=1,delimiter=',');f=data['inds'].reshape(-1,3)
b=json.loads(pose.read_text())['all_bodies'];ids=xyz[:,1].astype(int)
m=np.array([v['current']['local_to_world_row_major_f32'] for v in b]).reshape(-1,3,3);t=np.array([v['current']['position_world_m'] for v in b])
bound=float(np.linalg.norm(t[ids]+(m[ids]@xyz[:,8:11,None])[:,:,0]-xyz[:,2:5],axis=1).max())
if bound>2e-7 or abs(m[ids].transpose(0,2,1)-xyz[:,11:].reshape(-1,3,3)).max()>2e-7:raise ValueError('Source material poses differ')
poses={v['body_index']:{'position_m':v['current']['position_world_m'],'quaternion_xyzw':v['current']['orientation_world_xyzw']} for v in b}
summary=json.loads((prior/'initial-envelope.json').read_text())
raw=subprocess.check_output(['/opt/homebrew/bin/zstd','-qdc',str(pack)])
assert hashlib.sha256(raw).hexdigest()==summary['pack_sha256'] and sha(pose)==summary['pose_sha256']
((_,_,outer,of),)=audit['primitives'](raw,poses,skin_only=True)
selected=sorted((r for r in summary['affected_banks'] if r[0]==51005),key=lambda r:-r[-1])[:8]
selection={r[1] for r in selected}
witnesses={r['primitive']:r for r in map(json.loads,(prior/'target-witnesses.jsonl').open())}
obs=xyz[:,2:5].astype('f4')
src=c._exact_surface_records(obs,f);idx=ci._prepare_surface_aabb(src);fl=c._float32_skin_aabb_index(src)
def proper(a,b):
    def cuts(x,plane):
        origin=plane[0];norm=ci._cross(ci._sub(plane[1],origin),ci._sub(plane[2],origin))
        values=[ci._dot(norm,ci._sub(p,origin)) for p in x]
        return min(values)<0<max(values)
    return cuts(a,b) and cuts(b,a)
def count_hits(x,fx,y,fy,h):
    tx={};ty={};n=0;targets=set();first=None
    def corners(cache,layout,tris,i):
        if i not in cache:cache[i]=[ci.float32_point_lattice_key(v) for v in layout[tris[i]]]
        return cache[i]
    for i,j in h:
        if proper(corners(tx,x,fx,i),corners(ty,y,fy,j)):
            n+=1;targets.add(j)
            if first is None:first=[i,j]
    return dict(intersection_pairs=len(h),transverse_pairs=n,touch_or_coincident_pairs=len(h)-n,distinct_transverse_target_triangles=len(targets),first_transverse=first)
by={};compare=[]
for key,i,x,g in audit['primitives'](raw,poses):
    if i not in witnesses:continue
    row=witnesses.pop(i);h=row['triangle_pairs'];assert len(h)==row['count']
    item=count_hits(outer,of,x,g,h)
    item.update(semantic=key[0],stable_id=key[1],primitive=i,ocular=row['ocular_source_id'])
    if key[0] in (51005,51006):
        label=names[key[1]];item.update(label=label.get('label',label.get('member_id')),member_id=label.get('member_id'))
    by[i]=item
    if key[0]!=51005 or key[1] not in selection:continue
    chosen,mc=c._candidate_target_rows_for_skin_aabb(g,x,fl);ch=np.asarray(chosen,dtype=int)
    v=np.unique(g[ch]); rec=c._exact_surface_records(x[v],np.searchsorted(v,g[ch]))
    ans=ci._audit_pair_prepared_first(idx,[(a,b,c0,int(ch[k]),e) for a,b,c0,k,e in rec]);assert ans['aabb_candidate_pairs']==mc
    compare.append(dict(stable_id=key[1],label=item['label'],material=item,source=count_hits(obs,f,x,g,ans['triangle_pairs'])))
if witnesses or len(compare)!=len(selection):raise ValueError('Missing crossing bank or input comparison')
results=dict(pack_sha256=summary['pack_sha256'],body_pose_sha256=sha(pose),source_skin_sha256=sha(skin),tissue_sha256=sha(tissue),source_pose_max_frame_error_m=bound,nonocular_transverse=sum(r['transverse_pairs'] for r in by.values() if not r['ocular']),ocular_transverse=sum(r['transverse_pairs'] for r in by.values() if r['ocular']),banks=sorted(by.values(),key=lambda r:-r['transverse_pairs']),source_comparison=sorted(compare,key=lambda r:r['stable_id']))
out.write_text(json.dumps(results,indent=2)+'\n')
