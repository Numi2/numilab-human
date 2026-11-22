"""Check real inner-skin and deep-fat interfaces against the four source-clear muscle banks."""
from pathlib import Path
import hashlib,json,runpy,subprocess,sys
import numpy as np
from emit_skin_source_registration import sha
launch,impl,packed,pose,baseline,out=map(Path,sys.argv[1:]);base=json.loads(baseline.read_text())
sys.path.insert(0,str(impl.parent/'src'));audit=runpy.run_path(str(impl));c=audit['clearance'];ci=audit['intersections']
env=json.loads(launch.read_text())['environment'];pfx='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_'
b=json.loads(pose.read_text())['all_bodies'];poses={v['body_index']:{'position_m':v['current']['position_world_m'],'quaternion_xyzw':v['current']['orientation_world_xyzw']} for v in b}
raw=subprocess.check_output(['/opt/homebrew/bin/zstd','-qdc',str(packed)]);assert hashlib.sha256(raw).hexdigest()==base['pack_sha256'] and sha(pose)==base['body_pose_sha256']
selected={r['stable_id']:r['label'] for r in base['source_comparison'] if r['source']['intersection_pairs']==0};assert len(selected)==4
muscles={k[1]:(x,f) for k,i,x,f in audit['primitives'](raw,poses) if k[0]==51005 and k[1] in selected};del raw
result={};surface_rows={}
for layer in ['I','D']:
    prefix=pfx+layer+'_';p=Path(env[prefix+'NODES_CSV']);q=Path(env[prefix+'FACES_CSV'])
    assert sha(p)==env[prefix+'NODES_SHA256'] and sha(q)==env[prefix+'FACES_SHA256']
    x=np.loadtxt(p,delimiter=',',skiprows=1,usecols=(2,3,4)).astype('f4');f=np.loadtxt(q,delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
    rec=c._exact_surface_records(x,f);idx=ci._prepare_surface_aabb(rec);fl=c._float32_skin_aabb_index(rec)
    for sid,(v,g) in muscles.items():
        chosen,mc=c._candidate_target_rows_for_skin_aabb(g,v,fl);ch=np.asarray(chosen,dtype=int);ids=np.unique(g[ch])
        tr=c._exact_surface_records(v[ids],np.searchsorted(ids,g[ch]))
        hits=ci._audit_pair_prepared_first(idx,[(a,b,h,int(ch[k]),e) for a,b,h,k,e in tr])
        assert hits['aabb_candidate_pairs']==mc
        surface_rows[f'{layer}_{sid}']=np.unique([i for i,j in hits['triangle_pairs']]).astype('i4')
        result.setdefault(str(sid),dict(label=selected[sid]))[layer]=dict(pairs=hits['count'],muscle_faces=len({j for _,j in hits['triangle_pairs']}),first_pairs=hits['triangle_pairs'][:5])
np.savez_compressed(out.with_suffix('.npz'),**surface_rows)
out.write_text(json.dumps(dict(scope='reference-pose muscle intersections with the physical inner-skin I and deep-fat D surfaces',muscles=result,pack_sha256=base['pack_sha256']),indent=2)+'\n')
