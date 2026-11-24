"""Locate self-intersections before or after registration, retaining exact TetGen witnesses."""
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_shoulder import MOVING, read
from registration_stages import parts
from compile_connected_joint import digest


def check(x,f,tool,folder):
    folder.mkdir(parents=True,exist_ok=True)
    lines=[f'{len(x)} 3 0 0']
    lines.extend(f'{i+1} '+' '.join(f'{v:.17g}' for v in r) for i,r in enumerate(x))
    lines.append(f'{len(f)} 1')
    for i,tri in enumerate(f):lines.extend([f'1 0 {i+1}','3 '+' '.join(str(q+1)for q in tri)])
    (folder/'surface.poly').write_text('\n'.join(lines+['0','0'])+'\n')
    p=subprocess.run([tool,'-dV','surface.poly'],cwd=folder,capture_output=True,text=True)
    s=p.stdout+p.stderr;(folder/'predicates.log').write_text(s)
    tags=[int(v)-1 for v in re.findall(r'facet triangle:.*?tag\((\d+)\)',s)]
    rows=re.findall(r'Warning:\s+([^\n]+intersect[^.\n]*\.)',s,re.I)
    nontarget=[q for q in s.splitlines()if 'Warning:' in q and 'intersection' not in q and 'intersect' not in q]
    return dict(exit=p.returncode,witnesses=len(rows),unique_facets=len(set(tags)),reported_tags=sorted(set(tags)),
                other_warnings=nontarget[:4],log=digest(folder/'predicates.log'))


def run(j,source,tool,out):
    case=j/'posterior-interface-082/native-pair';cfg=read(case/'launch.json');a=cfg['argv']
    payload=Path(a[a.index('--soft-tissue-payload')+1])
    frames=read(case/'native-initial-bodies.json')['all_bodies']
    rc=read(next((case/'native-traces/accepted-geometry').glob('*.receipt.json')))
    poses={r['body_index']:r for r in rc['accepted_body_poses']}
    result={}
    for sid in MOVING:
        v,w,b,f,aff=parts(payload,sid);used=np.unique(f);v,w,f=v[used],w[used],np.searchsorted(used,f)
        u,sel,inv=np.unique(v,axis=0,return_index=True,return_inverse=True)
        local=inv[f];basis=w[sel]
        if np.max(abs(basis[inv]-w))>1e-6:raise ValueError('Conflicting seam weights')
        x=1000*(u-u.mean(0));stages={'stored':x}
        for name in ['neutral','supported']:
            declared={r['body_index']:dict(position_m=r['rest']['position_world_m'],quaternion_xyzw=r['rest']['orientation_world_xyzw']) for r in frames} if name=='neutral' else poses
            y=np.zeros_like(u)
            maps=[]
            for i,owner in enumerate(b):
                R=Rotation.from_quat(declared[owner]['quaternion_xyzw']).as_matrix()
                p=u@(R@aff[i][0]).T+declared[owner]['position_m']+R@aff[i][1]
                maps.append(p);y+=p*basis[:,i,None]
            stages[name]=(y-y.mean(0))*1000
        stages['emitted']=np.loadtxt(source/f'mesh-{sid}/source.poly',skiprows=1,max_rows=len(u),usecols=(1,2,3))
        result[sid]={k:check(p,local,tool,out/f'{sid}-{k}')for k,p in stages.items()}
        print(sid,[(k,r['witnesses'],r['exit'])for k,r in result[sid].items()],flush=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve()for s in sys.argv[1:]))
