"""Re-audit the replacement facets against every original bank at the identical release pose."""
import json,hashlib,sys,subprocess,runpy,csv
from pathlib import Path
from collections import Counter
import numpy as np
from emit_skin_source_registration import sha
impl,pack,pose,launch,ledger,star,out=map(Path,sys.argv[1:])
sys.path.insert(0,str(impl.parent/'src'));api=runpy.run_path(str(impl));c=api['clearance'];ci=api['intersections']
j=json.load(launch.open());env=j['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
vol=Path(env[prefix+'TETGEN_NODE']);assert sha(vol)==env[prefix+'TETGEN_NODE_SHA256']
with vol.open() as f:
    n=int(next(f).split()[0]);x=np.loadtxt(f,max_rows=n,usecols=(1,2,3)).astype('f4')
p=np.load(star/'local-volume-proposal.npz');s=json.load((star/'edge-stars.json').open());assert sha(star/'local-volume-proposal.npz')==s['proposal_sha256']
xx=np.vstack((x,p['new_points_world_m'].astype('f4')));sf=p['skin_faces'];touched=set(map(int,p['affected_skin_faces']));orig={r['primitive']:r for r in map(json.loads,(ledger/'target-witnesses.jsonl').open())}
base=json.load((ledger/'initial-envelope.json').open());b=json.load(pose.open())['all_bodies'];poses={r['body_index']:{'position_m':r['current']['position_world_m'],'quaternion_xyzw':r['current']['orientation_world_xyzw']} for r in b}
raw=subprocess.check_output(['/opt/homebrew/bin/zstd','-qdc',str(pack)]);assert hashlib.sha256(raw).hexdigest()==base['pack_sha256'] and sha(pose)==base['pose_sha256']
patch_ids=np.unique(sf);remap=np.searchsorted(patch_ids,sf);rec=c._exact_surface_records(xx[patch_ids],remap)
idx=ci._prepare_surface_aabb(rec);fl=c._float32_skin_aabb_index(rec)
lower=xx[patch_ids].min(0);upper=xx[patch_ids].max(0)
rows=[];summ=Counter();unaffected=Counter()
for key,i,v,g in api['primitives'](raw,poses):
    if key==api['SKIN']:continue
    prev=orig.get(i,{}).get('triangle_pairs',[])
    fixed={(a,b) for a,b in prev if a not in touched}
    old=sorted((a,b) for a,b in prev if a in touched)
    new=[]
    if not (np.any(v.max(0)<lower) or np.any(v.min(0)>upper)):
        tr=v[g];mask=np.flatnonzero(np.all(tr.max(1)>=lower,1)&np.all(tr.min(1)<=upper,1))
        if len(mask):
            ch,mc=c._candidate_target_rows_for_skin_aabb(g[mask],v,fl)
            ch=mask[np.asarray(ch,dtype='i4')];u=np.unique(g[ch]);targets=c._exact_surface_records(v[u],np.searchsorted(u,g[ch]))
            hit=ci._audit_pair_prepared_first(idx,[(a,b,vv,int(ch[k]),ee) for a,b,vv,k,ee in targets]);assert hit['aabb_candidate_pairs']==mc
            new=sorted((int(p['skin_parent'][a]),b) for a,b in hit['triangle_pairs'])
    ocular=key[0]==51010 and 381<=key[1]<=397
    if not ocular:
        summ['original_pairs']+=len(prev);summ['proposed_pairs']+=len(fixed)+len(new)
        summ['original_target_faces']+=len({b for a,b in prev});summ['proposed_target_faces']+=len({b for a,b in fixed}|{b for a,b in new})
    if old or new:
        unchanged={b for a,b in fixed};removed={b for a,b in old}-{b for a,b in new}-unchanged;added={b for a,b in new}-{b for a,b in old}-unchanged
        rows.append(dict(semantic=key[0],stable_id=key[1],primitive=i,original_affected_pairs=len(old),proposed_affected_pairs=len(new),removed_target_faces=sorted(removed),new_target_faces=sorted(added)))
out.write_text(json.dumps(dict(scope='offline exact same-pose facet swap; native state unmodified',whole_nonocular=dict(summ),affected_banks=rows,proposed_edge_nodes=len(p['new_points_world_m']),full_baseline_audit_sha256=sha(ledger/'initial-envelope.json'),candidate_sha256=s['proposal_sha256']),indent=2)+'\n')
