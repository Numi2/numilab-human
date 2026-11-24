"""Compare the complete native state and reused physical qualification against the matched control."""
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np
from check_full_motion import run as measure_material
from measure_supported_rest import measure
from compile_connected_joint import digest


def run(j, case):
    parent=j/'compensated-start-two'
    assert json.loads((case/'launch.json').read_bytes())['control_launch']==digest(parent/'launch.json')
    v=case/'audit/witness-material/volume'
    v.mkdir(parents=True,exist_ok=True)
    for source in (parent/'audit/witness-material/volume').iterdir():
        if not (v/source.name).exists():
            (v/source.name).symlink_to(source.resolve())
    measure_material(case,case/'audit',case/'material-check.json')
    measure(case,v,case/'support-check.json')
    for d in (case,parent):
        if json.loads((d/'execution.json').read_bytes())['exit']!=0:raise ValueError('Incomplete prefix')
    f=case/'native-traces/resting-coupled.csv'
    with f.open() as s, (parent/'native-traces/resting-coupled.csv').open() as t:
        a,b=list(csv.DictReader(s)),list(csv.DictReader(t))
    assert len(a)==len(b)>0 and all(x['time_s']==y['time_s'] for x,y in zip(a,b))
    deltas={}
    for key in a[0]:
        try: delta=max(abs(float(x[key])-float(y[key])) for x,y in zip(a,b))
        except ValueError:continue
        if delta:deltas[key]=delta
    points=lambda k: np.loadtxt(k/'physical-accepted.node',skiprows=1,usecols=(1,2,3))
    delta=np.linalg.norm(points(case)-points(parent),axis=1)
    obs=np.fromfile(case/'force-observer.f32','<f4').reshape(4,8,4).astype(float)
    active=np.flatnonzero(obs[:,7,3])
    if list(active)!=[0,1]:raise ValueError('Wrong observation steps')
    norm=[]
    for i in active:
        forces=obs[i,:6:2,:3];t=obs[i,1:6:2,:3];skin=obs[i,6,:3]
        norm.append(dict(source_step=int(i),skeletal_force_N=forces.tolist(),torque_Nm=t.tolist(),skin_total_N=skin.tolist(),
            imbalance_N=float(np.linalg.norm(forces.sum(0)+skin)),energy_J=float(obs[i,7,0])))
    if max(r['imbalance_N']for r in norm)>1e-8:raise ValueError('Native action and reaction mismatch')
    term=next(l for l in (case/'stdout.log').read_text().splitlines() if l.startswith('resting_integrated_body=completed'))
    result=dict(terminal=term,control_terminal=next(l for l in (parent/'stdout.log').read_text().splitlines() if l.startswith('resting_integrated_body=completed')),
                force_owner=norm,accepted_frame_max_change_m=float(delta.max()),time_series_max_absolute_deltas=deltas,
                control_launch=digest(parent/'launch.json'),launch=digest(case/'launch.json'),
                precision='Observer stores float32 applied terms. Total external residuals use the existing whole-body solver.')
    (case/'matched.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    run(*(Path(k).resolve() for k in sys.argv[1:]))
