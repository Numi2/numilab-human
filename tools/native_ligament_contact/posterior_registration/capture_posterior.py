"""Read actual native pack geometry with the frame furnished by its receipt."""
import json,mmap,runpy,sys
from pathlib import Path
import numpy as np
from bilateral_bindings import decode
from material_context import sha
from prepare_posterior_muscles import interface_audit
from fit_bilateral_gluteal_bindings import pair_count
from preserve_muscle_cells import det

def audit_native(w,cap,launch,source,exe,baseline,run,out):
    cfg=json.loads(launch.read_text())['argv'];payload=Path(cfg[cfg.index('--soft-tissue-payload')+1]);out.mkdir(exist_ok=True)
    reader=runpy.run_path(str(source/'audit.py'))
    # Include all surface layers, including both closed inner material boundaries.
    reader['BANK_CLASSES'].add(51007)
    file=next((run/'native/accepted-geometry').glob('step-8.receipt.json'))
    rc=json.loads(file.read_text());pack=run/'native/resting-human.mrvpack';end=Path(rc['accepted_pack_path'])
    initial=np.array(next(json.loads(line.partition('=')[2]) for line in (cap/'emitted-native/stderr.log').open() if line.startswith('registered_ligament_actual_pose_json='))['bodyPoses'])
    result={}
    for name,mesh,pin,poses in [('entrance',pack,rc['base_pack_file_sha256'],rc['initial_anatomical_registration']['body_poses']),
                                 ('accepted-8',end,rc['pack_file_sha256'],rc['accepted_body_poses'])]:
        if sha(mesh)!=pin:raise ValueError('Native pack identity mismatch')
        frame={int(r['body_index']):r for r in poses}
        for index,p in enumerate(initial):
            if index not in frame and name=='entrance':frame[index]=dict(position_m=p[:3],quaternion_xyzw=p[3:])
        with mesh.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as data:
            # In the accepted pack, mrvpack may reference a small number of rigid instances
            # which the receipt omits. Use the full accepted body pose list for those only.
            if name!='entrance':
                for row in rc['accepted_body_poses']:
                    if 'body_index' in row:frame.setdefault(row['body_index'],row)
            full={key:(v.copy(),f.copy()) for key,i,v,f in reader['primitives'](data,frame)}
        row={'pack':pin,'sides':{}}
        for sid in (29,30):
            v,f=full[51005,sid];other=59-sid
            hits=interface_audit((51005,sid),full,exe)
            t=np.loadtxt(baseline/str(sid)/'source.1.ele',skiprows=1,dtype=int)[:,1:5]
            x0=np.loadtxt(baseline/str(sid)/'source.1.node',skiprows=1)[:,1:4]
            rel=det(v.astype(float),t)/det(x0,t)
            row['sides'][sid]=dict(self_pairs=len(hits.get('51005/%d'%sid,[])),bilateral_pairs=len(hits.get('51005/%d'%other,[])),inverted=int((rel<=0).sum()),minimum_determinant_ratio=rel.min(),counts={k:len(x) for k,x in hits.items()})
            (out/(name+'-'+str(sid)+'-hits.json')).write_text(json.dumps(hits,separators=(',',':'))+'\n')
            np.savez_compressed(out/(name+'-'+str(sid)+'.npz'),xyz=v,faces=f)
        result[name]=row
    result['accepted_step']=rc['accepted_step'];result['accepted_time_s']=rc['accepted_time_s'];result['payload_sha256']=sha(payload)
    (out/'actual.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':audit_native(*map(Path,sys.argv[1:]))
