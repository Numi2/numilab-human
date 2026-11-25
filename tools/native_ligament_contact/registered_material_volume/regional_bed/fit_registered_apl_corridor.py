"""Register a connected radial tendon corridor on three measured native body frames."""
import itertools
import json
import runpy
import sys
from pathlib import Path

import igl
import numpy as np
from scipy.spatial.transform import Rotation
from audit_guide_corridor import geometry, path_metrics
from compile_connected_joint import digest
from fit_physical_guides import world_linear
from fit_route_binding import read
from install_physical_routes import check_intervals
from measure_route_motion import segments_in_solid


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def applied(frame, sid, fit):
    p, matrices, _ = frame['routes'][sid]
    p = p.copy()
    p[1:-1] += np.einsum('nij,nj->ni', matrices[1:-1], fit['source_delta_mm'])
    bodies = (43, 52) if sid == 147 else (93, 102)
    for i, body, local in zip([0, 7], bodies, fit['endpoint_local_m']):
        pose = frame['poses'][body]
        p[i] = 1000 * (Rotation.from_quat(pose['quaternion_xyzw']).apply(local) + pose['position_m'])
    return p


def proposals(baseline, alternate, frame):
    a, b = [np.asarray(x['source_delta_mm']) for x in (baseline, alternate)]
    for u, v in sorted(itertools.product(np.linspace(0,1,9),repeat=2),key=lambda t:sum(t)):
        d = a.copy()
        d[3] += u * (b[3]-a[3])
        d[4] += v * (b[4]-a[4])
        yield dict(baseline,source_delta_mm=d.tolist()), dict(method='interpolate',fractions=[u,v])
    points = applied(frame,147,baseline)
    axis = points[5]-points[4]
    axis /= np.linalg.norm(axis)
    normal = np.cross(axis,np.eye(3)[np.argmin(abs(axis))])
    normal /= np.linalg.norm(normal)
    matrices = frame['routes'][147][1]
    for radius in [.025,.05,.1,.2,.4,.8,1.6]:
        for angle in np.arange(24)*2*np.pi/24:
            shift = radius*(normal*np.cos(angle)+np.cross(axis,normal)*np.sin(angle))
            d = a.copy()
            for i in (4,5): d[i-1] += np.linalg.solve(matrices[i],shift)
            yield dict(baseline,source_delta_mm=d.tolist()),dict(method='guide_translation',offset_in_first_frame_mm=shift.tolist())


def metrics(frame, points):
    result = {}
    lo, hi = points.min(0),points.max(0)
    for key,(v,f) in frame['banks'].items():
        if key!='skin' and (np.any(v.max(0)<lo-6) or np.any(v.min(0)>hi+6)):
            continue
        m = path_metrics(points,v,f,key=='skin')
        if key=='skin' or m['sample_min_mm']<3 or m['surface_hits'] or m['invalid_segments']:
            result[key] = m
    return result


def main(reader, control, accepted8, phys, refit, out):
    out.mkdir()
    reader = runpy.run_path(str(reader))
    frames = []
    for case,initial in [(control,True),(accepted8,False),(control,False)]:
        rc, = (case/'native-traces/accepted-geometry').glob('*.receipt.json')
        poses,banks,pack = geometry(reader,case,rc,initial,False)
        cfg = read(case/'launch.json')
        routes = {sid:world_linear(cfg,poses,sid,mi) for sid,mi in [(147,259),(148,322)]}
        shortlist = []
        for k in ('16','77'):
            v,f = banks[k]
            tree = igl.AABB()
            tree.init(v,f)
            shortlist.append((tree,v,f))
        frames.append(dict(poses=poses,banks=banks,pack=pack,routes=routes,trees=shortlist,
                           step=0 if initial else read(rc)['accepted_step'],case=case.name,initial=initial))
    rows = []
    for sid in (147,148):
        a,b = (read(path/f'{sid}.json') for path in (phys,refit))
        options = proposals(a,b,frames[0]) if sid==147 else [(a,dict(method='physical_native_16')),(b,dict(method='refitted_16'))]
        for attempt,(fit,choice) in enumerate(options,1):
            pts=[applied(frame,sid,fit) for frame in frames]
            if sid==147 and any(segments_in_solid(tree,v,f,p[4:6]) for frame,p in zip(frames,pts) for tree,v,f in frame['trees']):
                continue
            records=[]
            try:
                for frame,p in zip(frames,pts):
                    m=metrics(frame,p)
                    check_intervals(sid,m)
                    records.append(dict(side=sid,run=frame['case'],initial=frame['initial'],step=frame['step'],pack=frame['pack'],metrics=dict(candidate=m)))
            except ValueError as exc:
                print('Failed complete corridor',sid,attempt,str(exc),flush=True)
                continue
            fit.update(packs=[frame['pack'] for frame in frames],sites=frames[0]['routes'][sid][2],launch=digest(control/'launch.json'),choice=choice,
                       parents=[digest(path/f'{sid}.json') for path in (phys,refit)])
            save(out/f'{sid}.json',fit)
            rows.extend(records)
            print('Passed all three poses',sid,attempt,choice,flush=True)
            break
        else:
            raise ValueError(f'No registered corridor for {sid}')
    save(out/'audit.json',dict(rows=rows))

if __name__=='__main__': main(*(Path(p).resolve() for p in sys.argv[1:]))
