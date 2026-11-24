"""Audit a saved external material boundary against the same anatomy as its emitted native skin."""
import json
import mmap
from pathlib import Path
import runpy
import sys

import igl
import numpy as np
from compile_connected_joint import digest


def read(p):
    return json.loads(p.read_bytes())


def metrics(points, surface_nodes, faces):
    d = np.sqrt(igl.point_mesh_squared_distance(np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(surface_nodes,dtype=np.float64),np.ascontiguousarray(faces,dtype=np.int64))[0])
    if not np.isfinite(d).all():
        raise ValueError('Nonfinite material surface distance')
    return dict(points=len(d),rms_m=float(np.sqrt(np.mean(d*d))),p50_m=float(np.quantile(d,.5)),
                p95_m=float(np.quantile(d,.95)),p99_m=float(np.quantile(d,.99)),max_m=float(d.max()),above_1mm=int(np.sum(d>.001)))


def compare(source, adapter, case, control_material, display, out):
    out.mkdir()
    cfg=read(case/'launch.json')
    a=runpy.run_path(str(source))
    runpy.run_path(str(adapter))['install_for_float32']()
    vol=Path(cfg['environment']['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE']).parent
    pack=next((case/'native-traces/accepted-geometry').glob('*.mrvpack'))
    receipt=read(pack.with_suffix('.receipt.json'))
    display_frame=read(display/'result.json')['frames'][0]
    physical=case/'physical-accepted.node'
    check=read(control_material/'material-check.json')
    if read(case/'execution.json')['exit']!=0 or digest(pack)!=display_frame['pack'] or receipt['pack_file_sha256']!=display_frame['pack'] or digest(physical)!=check['physical_frame'] or receipt['accepted_step']!=check['accepted_step']:
        raise ValueError('Display and material do not share their physical endpoint')
    with np.load(vol/'material-walls.npz')as archive: faces=archive['O']
    node=np.loadtxt(physical,skiprows=1,usecols=(1,2,3))
    ids,inv=np.unique(faces,return_inverse=True)
    ff=np.ascontiguousarray(inv.reshape(-1,3))
    xx=np.ascontiguousarray(node[ids].astype('f4'))
    with pack.open('rb')as src,mmap.mmap(src.fileno(),0,access=mmap.ACCESS_READ)as m:
        poses={r['body_index']:r for r in receipt['accepted_body_poses']}
        [(key,idx,v,f)]=list(a['primitives'](m,poses,skin_only=True))
        v=v.copy();f=f.copy()
    distances=dict(material_to_display=metrics(xx,v,f),display_to_material=metrics(v,xx,ff))
    original=a['primitives']
    def substituted(m,poses,*,skin_only=False):
        if skin_only:
            yield key,idx,xx,ff
        else:
            yield from original(m,poses,skin_only=False)
    fn=a['run']
    fn.__globals__['primitives']=substituted
    raw=out/'material-wall-report.json'
    fn(pack,raw)
    result=read(raw)
    result['method']='Physical accepted outer material nodes in world coordinates rounded to Float32. Native target anatomy from the very same accepted step and pack. Existing exact integer triangle and same-surface adjacency predicates.'
    raw.write_text(json.dumps(result,indent=2)+'\n')
    summary=dict(launch=digest(case/'launch.json'),pose=display_frame['pack'],step=receipt['accepted_step'],accepted_time_s=receipt['accepted_time_s'],
                 physical_nodes=digest(physical),material_walls=digest(vol/'material-walls.npz'),faces=[len(f),len(ff)],vertices=[len(v),len(xx)],
                 material_report=digest(raw),native_report=display_frame['report'],distances=distances,
                 displayed_pairs_by_semantic=display_frame['by_semantic'],physical_pairs_by_semantic=result['crossing_pairs_by_semantic'],
                 displayed_self=display_frame['skin_self'],physical_self=result['self_intersections']['count'],
                 scope='Comparison of two recorded representations at a common physical time. Distances measure sampled vertices against the other full triangle surface, not volumetric penetration depths. Triangle counts depend on tessellation.')
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    compare(*(Path(p).resolve()for p in sys.argv[1:]))
