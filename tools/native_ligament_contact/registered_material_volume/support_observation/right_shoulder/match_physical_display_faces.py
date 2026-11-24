"""Check exact material/display triangle identity and every target-pair witness at an accepted pose."""
import json
import mmap
from pathlib import Path
import runpy
import sys

import numpy as np
from compile_connected_joint import digest


def read(p):
    return json.loads(p.read_bytes())


def canonical(f):
    s=np.sort(f,axis=1)
    perm=np.lexsort((s[:,2],s[:,1],s[:,0]))
    if np.any(np.all(s[perm[1:]]==s[perm[:-1]],axis=1)):
        raise ValueError('Ambiguous duplicate spatial triangles')
    order=np.empty_like(perm)
    order[perm]=np.arange(len(f))
    return s[perm],order


def prove(registry,case,physical,display,out):
    comp=read(physical/'result.json')
    pose=next((case/'native-traces/accepted-geometry').glob('*.mrvpack'))
    if digest(pose)!=comp['pose'] or digest(case/'physical-accepted.node')!=comp['physical_nodes']:
        raise ValueError('Mismatched material snapshot')
    api=runpy.run_path(str(registry))
    receipt=read(pose.with_suffix('.receipt.json'))
    with pose.open('rb')as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)as src:
        [(key,idx,v,fa)]=list(api['primitives'](src,{r['body_index']:r for r in receipt['accepted_body_poses']},skin_only=True))
        v=np.array(v,dtype=np.float32);fa=fa.copy()
    e=read(case/'launch.json')['environment']
    p=Path(e['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE']).parent/'material-walls.npz'
    if digest(p)!=comp['material_walls']: raise ValueError('Unmatched physical boundary')
    with np.load(p)as z:ff=z['O']
    idx,inv=np.unique(ff,return_inverse=True);fb=inv.reshape(-1,3)
    w=np.loadtxt(case/'physical-accepted.node',skiprows=1,usecols=(1,2,3))[idx].astype('f4')
    spatial,ids=np.unique(np.vstack((v,w)),axis=0,return_inverse=True)
    a,ma=canonical(ids[:len(v)][fa]);b,mb=canonical(ids[len(v):][fb])
    eq=a.shape==b.shape and np.array_equal(a,b)
    def pairs(report,order):
        d=read(report)
        return {(r['semantic'],r['stable_id'],r['primitive'],int(order[i]),j)
                for r in d['surfaces'] for i,j in r['triangle_pairs']}
    if not eq:raise ValueError('Material and display faces diverge in world space')
    pa=pairs(display/'step-2.json',ma)
    pb=pairs(physical/'material-wall-report.json',mb)
    if pa!=pb:raise ValueError('Material and display have different target witnesses')
    out.write_text(json.dumps(dict(step=receipt['accepted_step'],pack=digest(pose),physical_node_sha256=comp['physical_nodes'],
                      unique_world_vertices=len(spatial),identical_unoriented_triangles=len(a),identical_skin_target_witnesses=len(pa),
                      scope='World-space Float32 vertex coordinates, triangle connectivity up to winding and order, and the full target witness identities match. This test addresses the recorded two-step physical endpoint.'),indent=2)+'\n')


if __name__=='__main__':
    prove(*(Path(p).resolve()for p in sys.argv[1:]))
