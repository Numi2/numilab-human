"""Screen complete native posture donors against the target articulation's own limits."""
from pathlib import Path
import json, struct, sys
import numpy as np
from material_context import sha
config, output, *roots=map(Path,sys.argv[1:])
cfg=json.loads(config.read_text());rigid=Path(cfg['argv'][1]);digest=sha(rigid)
assert cfg['pins'][str(rigid)]==digest
data=rigid.read_bytes();v,abi,_,nb,nj,nq,nv,*_=struct.unpack_from('<10I',data,8)
assert data[:8]==b'NHRIGID2' and (v,abi)==(1,5)
start=224+160*nb+144*nj
limits=[]
for i in range(nv):
    _,jid,q,v,_,flags,_,_,lo,hi=struct.unpack_from('<8I2f',data,start+64*i)
    if v!=i:raise ValueError('incompatible DoF packing')
    if flags&4 and q!=0xffffffff: limits.append((q,jid,lo,hi))
runs=[]
for parent in roots:
    for launch in parent.glob('*/launch.json'):
        log=launch.parent/'stdout.log'
        if not log.is_file():continue
        doc=json.loads(launch.read_text());a=doc.get('argv',[]);pins=doc.get('pins',{})
        if len(a)<4 or pins.get(a[1])!=digest:continue
        terminal=None;completed=False
        with log.open(errors='replace') as f:
            for l in f:
                if l.startswith('stand_terminal_state='):terminal=json.loads(l.split('=',1)[1])
                if l.startswith('resting_integrated_body=completed ') and 'root_assistance=false' in l:completed=True
        if not completed or not terminal or terminal['root_assistance']:continue
        x=np.asarray(terminal['q']);vel=np.asarray(terminal['v'])
        if x.shape!=(nq,) or vel.shape!=(nv,):raise ValueError('wrong q/v dimension')
        failures=[]
        for q,j,lo,hi in limits:
            distance=max(lo-x[q],x[q]-hi,0)
            if distance>1e-7:failures.append(dict(q=q,joint=j,position=x[q],lo=lo,hi=hi,error=distance))
        runs.append(dict(source=str(log),seconds=terminal['step_count']*terminal['timestep_seconds'],source_sha256=sha(log),violations=failures,peak_velocity=float(abs(vel).max()),root_xyz=x[:3].tolist()))
output.write_text(json.dumps(dict(rigid_sha256=digest,source_position_tolerance=1e-7,candidates=runs),indent=2)+'\n')
for r in sorted(runs,key=lambda r: -r['seconds'])[:18]:
    print(r['seconds'],len(r['violations']),max((v['error'] for v in r['violations']),default=0),Path(r['source']).parent.name)
