"""Measure actual stored material and terminal bed loading at the accepted full-scene endpoint."""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from supported_material_admission import certify, installed_topology, stress_limits


def measure(run, out):
    cfg = json.loads((run/'launch.json').read_text())
    e = cfg['environment']
    x,y,original,roles,_ = inputs(run)
    cells, role = installed_topology(original,roles,e)
    capture=run/'physical-accepted.node'
    begin=run/'physical-reference.node'
    def read_pose(p):
        a=np.loadtxt(p,skiprows=1)
        if a.shape!=(len(x),4) or not np.array_equal(a[:,0],np.arange(1,len(x)+1)) or not np.isfinite(a[:,1:]).all():
            raise ValueError('Incomplete accepted-node capture')
        return a[:,1:]
    if not np.array_equal(read_pose(begin),y):
        raise ValueError('Native start differs from preparation')
    end=read_pose(capture)
    props,files=materials(Path(cfg.get('compiled_material_owner',Path(cfg['argv'][0]).parent)))
    result=certify(x,end,cells,role,props,stress_limits(files))
    stdout=(run/'stdout.log').read_text()
    line=next(l for l in stdout.splitlines() if l.startswith('resting_fem_skin_bed_accepted_root_reaction '))
    def number(line, key):
        return float(re.search(r'(?:^| )'+key+r'=([^ ]+)',line)[1])
    clock=list(csv.DictReader((run/'native-traces/resting-coupled.csv').open()))
    done=next(l for l in stdout.splitlines() if l.startswith('resting_integrated_body=completed '))
    dt=float(cfg['argv'][cfg['argv'].index('--muscle-step-seconds')+1]);n=int(cfg['argv'][cfg['argv'].index('--muscle-step-count')+1])
    if not np.isclose(n*dt,float(clock[-1]['time_s']),rtol=2e-7,atol=1e-14) or 'physiology_body_clock=matched' not in done:
        raise ValueError('No completed matched physiology and body clock')
    components=[json.loads(l.split('initial_gravity_component=',1)[1]) for l in stdout.splitlines() if 'initial_gravity_component={' in l]
    if [v['object'] for v in components]!=list(range(-1,22)):raise ValueError('Incomplete body mass')
    weight=sum(v['mass_kg'] for v in components)*9.81
    if 'root_assistance=false' not in done or not components:
        raise ValueError('Missing native root or mass evidence')
    wall=json.loads((run/'execution.json').read_text())['wall_s']
    witness=np.array([131581,130206,130605,130659])
    if not np.array_equal(cells[870989],witness):raise ValueError('Wrong material witness')
    def vol(p):
        a=p[witness].astype(float)
        return np.linalg.det(a[1:]-a[:1])
    summary=dict(launch=sha(run/'launch.json'),binary=sha(Path(cfg['argv'][0])),physical_nodes=sha(capture),
        source_materials=files,cells=len(cells),accepted_steps=n,accepted_seconds=float(clock[-1]['time_s']),
        end_to_end_cold_wall_s=wall,cold_factor=float(clock[-1]['time_s'])/wall,
        native_body_window_wall_s=number(done,'wall_s'),native_body_window_factor=number(done,'real_time_factor'),
        material=result,world_weight_N=weight,last_bed_reaction_N=number(line,'mean_normal_force_N'),
        terminal_active_bed_rows=int(number(line,'valid_contact_sample_count')),
        last_bed_fraction_of_weight=number(line,'mean_normal_force_N')/weight,
        peak_motion_m=float(np.linalg.norm(end-y,axis=1).max()),witness_J_start=vol(y)/vol(x),
        witness_J_end=vol(end)/vol(x),variable_pleural='resting_vascular_external_pressure=accepted_absolute_pleural_gauge' in stdout,
        root_assistance='root_assistance=true' in done,pleural_Pa_first_last=[float(clock[i]['pleural_pa']) for i in (0,-1)],
        complete_breaths=int(clock[-1]['breaths']),complete_filling_ejection_cycles=int(clock[-1]['complete_filling_ejection_cycles']))
    (out).write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    measure(*(Path(a).resolve() for a in sys.argv[1:]))
