"""Measure where registered arm muscles extend past the accepted outer material surface."""
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
from cortical_distance import CorticalSurface
from digital_clearance import complete_remote_loops
from material_context import sha

def measure(run,implementation,library,out):
    sys.path.insert(0,str(implementation.parent/'src'))
    api=runpy.run_path(str(implementation))
    rc=json.loads((run/'terminal.receipt.json').read_text());job=json.loads((run/'launch.json').read_text())
    adm=json.loads((run/'admission.json').read_text());surface=json.loads((run/'envelope/accepted-envelope.json').read_text())
    pack=run/'accepted-terminal.mrvpack.zst'
    if (rc['physical_endpoint']!='accepted' or adm['terminal_archive_sha256']!=sha(pack)
        or rc['accepted_step']!=adm['accepted_steps'] or surface['skin_self_intersections']!=0):raise ValueError('Unknown or intersecting skin')
    raw=subprocess.check_output(['zstd','-qdc',str(pack)])
    if hashlib.sha256(raw).hexdigest()!=rc['pack_file_sha256'] or surface['pack_sha256']!=rc['pack_file_sha256']:raise ValueError('Unmatched pose')
    bodies={r['body_index']:r for r in rc['accepted_body_poses']}
    ((_,_,v,f),)=api['primitives'](raw,bodies,skin_only=True)
    prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_O_'
    e=job['environment'];table=Path(e[prefix+'FACES_CSV'])
    if sha(table)!=e[prefix+'FACES_SHA256'] or not np.array_equal(f,np.loadtxt(table,skiprows=1,usecols=(1,2,3),delimiter=',',dtype='i4')):raise ValueError('Outer skin is not the bound material boundary')
    V,inverse=np.unique(v,axis=0,return_inverse=True);F=inverse[f]
    edge=np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]))
    un,counts=np.unique(np.sort(edge,axis=1),axis=0,return_counts=True)
    if np.any(counts>2):raise ValueError('Nonmanifold skin')
    rim=V[un[counts==1].ravel()]
    tissue=Path(job['argv'][job['argv'].index('--soft-tissue-payload')+1])
    if sha(tissue)!=job['pins'][str(tissue)]:raise ValueError('Unknown muscle asset')
    meta=json.loads(tissue.with_suffix('.manifest.json').read_text())
    if meta['payload']['sha256']!=sha(tissue):raise ValueError('Unknown bank labels')
    names={r['stable_id']:r.get('label',r['member_id']) for r in meta['source']['surfaces']}
    landmarks={137,138,109,110,120,119}
    witness={r['primitive']:r for r in map(json.loads,(run/'envelope/target-witnesses.jsonl').read_text().splitlines())}
    arm=[]
    for (sem,sid),idx,x,t in api['primitives'](raw,bodies):
        if sem!=51005 or sid not in landmarks:continue
        query=np.arange(len(x))
        if len(rim):
            query=query[(x[:,1]<rim[:,1].min()-.501)|(x[:,1]>rim[:,1].max()+.501)]
        if len(query):arm.append((sid,idx,x,query,t))
    if not arm:raise ValueError('No eligible distal arm queries')
    capV,capF,gap=complete_remote_loops(V,F,np.vstack([x[q] for _,_,x,q,_ in arm]))
    tissue_boundary=CorticalSurface(library,capV,capF)
    rows=[]
    for sid,idx,x,query,t in arm:
        depth,normals,nearest,face=tissue_boundary.query(x[query])
        if not np.isfinite(depth).all() or np.any(face>=len(f)):raise ValueError('Nearest surface is not physical skin')
        worst=np.argmax(depth);p=int(query[worst])
        if not np.isclose(abs(depth[worst]),np.linalg.norm(x[p]-nearest[worst]),rtol=2e-6,atol=1e-9):raise ValueError('Inconsistent closest point')
        rows.append(dict(stable_id=int(sid),label=names[sid],retained_triangle_pairs=int(witness.get(idx,{}).get('count',0)),
                         queried_vertices=len(query),total_vertices=len(x),outside_vertices=int(np.sum(depth>0)),outside_over_1mm=int(np.sum(depth>.001)),
                         outside_over_5mm=int(np.sum(depth>.005)),maximum_outside_mm=max(0.,1000*depth[worst]),
                         worst_bank_vertex=p,nearest_physical_skin_face=int(face[worst]),world_m=x[p].tolist(),nearest_skin_m=nearest[worst].tolist()))
    out.write_text(json.dumps(dict(pack_sha256=rc['pack_file_sha256'],accepted_step=rc['accepted_step'],asset_sha256=sha(tissue),predicate_sha256=sha(library),rim_separation_m=gap,
             scope='Signed closest-feature distances at accepted muscle vertices more than 0.5 m along the body from open head edges. Temporary head caps only define the interior. Every reported closest point is on an original physical triangle. Triangle interiors and dynamic contact forces are not sampled.',banks=rows),indent=2,allow_nan=False)+'\n')
if __name__=='__main__':measure(*(Path(a).resolve() for a in sys.argv[1:]))
