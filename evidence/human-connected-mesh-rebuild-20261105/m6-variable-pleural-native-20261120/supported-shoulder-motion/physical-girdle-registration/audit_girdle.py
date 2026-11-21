"""Recheck every changed bone interface against the 185 registered banks."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from source_pair_audit import inputs,world
from bone_overlap import intersections
from fast_skin_intersection_audit import install_for_float32

def audit(launch,ref,motion,correction,out):
    ms,poses,default,_=inputs(launch,ref,motion)
    from shoulder_inputs import load
    _,cat,*_=load(launch,ref,motion)
    changes={r['sid']:r for r in json.loads(correction.read_text()) if r['bone'].split('_')[0] in ('scapula','clavicle')}
    adjusted={s:Rotation.from_quat(r['rotation_xyzw']).apply(ms[s][0])*r['scale']+r['translation_m'] for s,r in changes.items()}
    report=[]
    install_for_float32()
    for step,p in [(0,default),*((k,poses[k]) for k in [-3,32,256,8000])]:
        banks={s:world(p[cat[s]['body']],v) for s,(v,*_) in ms.items() if not isinstance(p,dict) or cat[s]['body'] in p}
        new={s:world(p[cat[s]['body']],v) for s,v in adjusted.items()}
        rows=[]
        for s in sorted(new):
            for t,old in banks.items():
                if t==s or t in new and t<s:continue
                vals=[]
                for a,b in ((banks[s],old),(new[s],new.get(t,old))):
                    a=a.astype(np.float32);b=b.astype(np.float32)
                    if np.any(a.min(axis=0)>b.max(axis=0)) or np.any(a.max(axis=0)<b.min(axis=0)):v=[]
                    else:v=intersections(a,ms[s][1],b,ms[t][1])
                    vals.append(v)
                if any(vals):
                    rows.append(dict(bones=[s,t],before=len(vals[0]),after=len(vals[1]),first_after=vals[1][:1]))
        report.append(dict(step=step,rows=rows))
        print(step,sum(r['before'] for r in rows),sum(r['after'] for r in rows),[(r['bones'],r['after']) for r in rows if r['after']],flush=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':audit(*map(Path,sys.argv[1:]))
