"""Use a single immutable fiber table for all poses and the original full native body indexing."""
import json,sys
from pathlib import Path
import numpy as np
from audit_fiber_load_bounds import read_pose

def run(probes,bodylog,out):
    q=json.loads((probes/'reference.json').read_bytes());records=q['records']
    poses=read_pose(bodylog)['bodyPoses'];table=np.array(poses,'f4')
    fibers=np.fromfile(probes/'fibers.bin','u1').reshape(-1,80);ranges=np.fromfile(probes/'ranges.bin','<u4').reshape(-1,4)
    motion=np.fromfile(probes/'poses.bin','<f4').reshape(-1,4,4);unique=list(dict.fromkeys(r['fiber'] for r in records));cases=list(dict.fromkeys(r['probe'] for r in records))
    F=[];R=[];bodies=np.zeros((len(cases),len(poses),2,4),'f4');bodies[:,:,0,:3]=table[None,:,:3];bodies[:,:,1]=table[None,:,3:]
    order=[];owners={}
    for i,name in enumerate(unique):
        group=[j for j,r in enumerate(records) if r['fiber']==name]
        if np.any(fibers[group]!=fibers[group[0]]) or np.any(ranges[group]!=ranges[group[0]]):raise ValueError('Physical model changes between poses')
        F.append(fibers[group[0]]);R.append(ranges[group[0]])
        a,b=F[-1].view('<u4')[:2]
        if a in owners or b in owners:raise ValueError('Incompatible paired shoulder sweep')
        owners[a]=owners[b]=name
        for j in group:
            env=cases.index(records[j]['probe']);bodies[env,a]=motion[j,:2];bodies[env,b]=motion[j,2:];order.append((env,i,j))
    if len(set((e,i) for e,i,j in order))!=len(cases)*len(unique):raise ValueError('Missing native case')
    out.mkdir();np.array(F).tofile(out/'fibers.bin');np.array(R).tofile(out/'ranges.bin');bodies.tofile(out/'poses.bin');(out/'edges.bin').hardlink_to(probes/'edges.bin')
    np.array([len(cases),len(poses)],'<u4').tofile(out/'dimensions.bin')
    q['records']=[records[j] for e,i,j in sorted(order)];(out/'reference.json').write_text(json.dumps(q,indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
