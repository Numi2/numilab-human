"""Evaluate the actual retained source-parent motion field at both matched body poses."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.spatial.transform import Rotation
from audit_current_tissue_parents import records
from compile_connected_joint import digest
def run(historical,base,active,target,out):
    rd=lambda f:json.loads(f.read_bytes());ids=set(range(153,159));report={}
    for mode in ['rest','current']:
        states=[];torso=[]
        for scene,which in [(historical,historical/'native-initial-bodies.json'),(base,target),(active,target)]:
            argv=rd(scene/'launch.json')['argv'];r=rd(Path(argv[argv.index('--resting-anatomy-receipt')+1]))['passive_tissue_parent_bindings'];p=[b[mode] for b in rd(which)['all_bodies']];torso.append(p[20]);pool={}
            for i in [2,3]:
                s=r['source_payloads'][i];f=Path(s['path'])
                if digest(f)!=s['sha256']:raise ValueError('Source identity differs')
                pool[i]=records(f,p,ids)
            x={}
            for s in r['surfaces']:
                sid=s['stable_id']
                if sid not in ids:continue
                n=len(s['vertices']);source=np.zeros((n,3));rows=s['vertices'];q=np.repeat(np.arange(n),list(map(len,rows)));a=np.array([term for row in rows for term in row]);p0,b,v=a[:,:3].astype(int).T
                for i,j in np.unique(np.column_stack([p0,b]),axis=0):
                    mask=(p0==i)&(b==j);np.add.at(source,q[mask],a[mask,3,None]*pool[i][j][0][v[mask]])
                x[sid]=source
            states.append(x)
        rots=[Rotation.from_quat(p['orientation_world_xyzw']).inv() for p in torso];rr={}
        for sid in sorted(ids):
            x=[r.apply(g[sid]-t['position_world_m']) for r,g,t in zip(rots,states,torso)];before=np.linalg.norm(x[1]-x[0],axis=1);after=np.linalg.norm(x[2]-x[0],axis=1)
            rr[sid]=dict(vertices=len(after),previous_max_mm=float(before.max()*1000),current_max_mm=float(after.max()*1000),current_rms_mm=float(np.sqrt((after**2).mean())*1000))
        report[mode]=rr
    out.write_text(json.dumps(dict(reference_launch=digest(historical/'launch.json'),current_launch=digest(active/'launch.json'),current_body_capture=digest(target),rule='Registered material parents evaluated at the two measured configurations relative to their thorax body.',poses=report),indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
