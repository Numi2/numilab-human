"""Resolve the real registered admissibility floor of the final geometry-limiting cells."""
import json,re,sys
from pathlib import Path
import numpy as np
from compile_connected_joint import digest

def run(job, volume, fit, out):
    scene=fit/'scene'
    records=(scene/'stderr.log').read_text().splitlines()
    spans=[l for l in records if l.startswith('registered_material_span ')]
    if len(set(spans))!=1:raise ValueError('Missing unique native material span')
    match=re.search(r'element_offset=(\d+) element_count=(\d+)',spans[0])
    offset,n=map(int,match.groups())
    tet=np.loadtxt(volume/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    if len(tet)!=n:raise ValueError('Unexpected topology')
    cfg=json.loads((scene/'launch.json').read_bytes())
    mode=cfg['environment']['NM_FEM_REGISTERED_RELEASE_BOUNDS']
    path=Path(cfg['environment']['NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE'])
    if mode!='2' or digest(path)!=cfg['pins'][str(path)]:raise ValueError('Unmatched loaded-state floor')
    read=lambda p:np.loadtxt(p,skiprows=1,usecols=(1,2,3)).astype('f4').astype(float)
    ref,pose=read(volume/'mesh.node'),read(path)
    bd=np.load(job/'registered-boundary/registered-boundary.npz')
    pin={k:set(map(int,bd[k+'_node_joint'])) for k in 'OID'}
    held=set(map(int,np.load(fit.parent/'held-cortex.npz')['nodes']))
    d=next(json.loads(l.split('=',1)[1]) for l in records if l.startswith('fem_backtracking_trials='))
    if not all(t.get('tile_witnesses_complete') for t in d['trials'] if t['invalid_tiles']):raise ValueError('Incomplete witnesses')
    tg=sorted({v for t in d['trials'] for v in t.get('tile_witness_tets',[])})
    bad=[]
    for g in tg:
        if not offset<=g<offset+n:raise ValueError('Witness is outside of registered volume')
        idx=tet[g-offset];a=ref[idx];b=pose[idx];J=np.linalg.det(b[1:]-b[0])/np.linalg.det(a[1:]-a[0])
        first=[r for r in d['trials'] if r['first_bad_tet']==g]
        entry=dict(global_id=g,local_id=g-offset,nodes=idx.tolist(),loaded_J=J,
            registered_floor=float(np.nextafter(np.float32(.35*J),np.float32(np.inf))),
            layer_nodes={k:sorted(v.intersection(idx)) for k,v in pin.items()},
            held_cortical_interface=sorted(held.intersection(idx)),last_rejected_alpha=min(r['alpha'] for r in d['trials'] if g in r.get('tile_witness_tets',[])))
        entry['last_observed_rejection']=dict(first[-1]) if first else None
        bad.append(entry)
    out.write_text(json.dumps(dict(launch_sha256=digest(scene/'launch.json'),current_node_sha256=digest(path),offset=offset,
        release_mode=mode,observed_selected_alpha=d['selected_alpha'],cells=bad),indent=2,default=lambda x:x.item())+'\n')

if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
