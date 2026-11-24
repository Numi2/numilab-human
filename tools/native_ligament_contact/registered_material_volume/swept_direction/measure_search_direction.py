"""Check how the native sparse material and bed direction operators act on a failing node star."""
import json
from pathlib import Path
import re
import sys
import numpy as np
from bind_gluteal_receivers import digest


def measure(run,control,identity,out):
    identity=json.loads(identity.read_bytes())
    phases=[{} for _ in range(3)]
    for line in (run/'stderr.log').read_text().splitlines():
        if not line.startswith('search_row '):continue
        data=dict(kv.split('=') for kv in line.split()[1:])
        phase=phases[int(data['phase'])]; node=int(data['global'])
        if node in phase:raise ValueError('Repeated native direction sample')
        phase[node]=list(map(float,data['delta'].split(',')))
    ordered=identity['global_nodes']
    if not all(set(p)==set(ordered) for p in phases):raise ValueError('Incomplete matched star')
    values=np.array([[p[i] for i in ordered]for p in phases])
    if not np.isfinite(values).all():raise ValueError('Nonfinite direction')
    i=ordered.index(identity['global_peak'])
    same={}
    for q in ['native-traces/resting-coupled.csv','native-traces/resting-com-support-impulses.csv','native-traces/resting-com-momentum-diagnostic.csv']:
        h=digest(run/q)
        if h!=digest(control/q):raise ValueError('Observer altered committed state')
        same[q]=h
    for k in ('segment_start','matter_code','matter_diagnostics'):
        if json.loads((run/'failure.json').read_bytes())[k]!=json.loads((control/'failure.json').read_bytes())[k]:
            raise ValueError('Observer changed the terminal certificate')
    rms=lambda a:float(np.sqrt((a*a).sum()/len(ordered)))
    summary=dict(launch=digest(run/'launch.json'),matched_control=same,global_peak=identity['global_peak'],
                 cells=len(identity['local_cells']),degrees_of_freedom=values[0].size,
                 inputs=['solved Newton direction','after volume feasibility','after unilateral bed feasibility'],
                 peak_m_per_s=values[:,i,:].tolist(),
                 peak_differences_m_per_s=np.linalg.norm(np.diff(values[:,i,:],axis=0),axis=1).tolist(),
                 star_rms_m_per_s=[rms(p) for p in values],
                 star_rms_change_m_per_s=[rms(p) for p in np.diff(values,axis=0)],
                 native_directions_m_per_s=[dict(global_node=n,stages=values[:,j,:].tolist())for j,n in enumerate(ordered)])
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print('native peak',summary['peak_m_per_s'],'star RMS changes',summary['star_rms_change_m_per_s'])


if __name__=='__main__':measure(*(Path(s) for s in sys.argv[1:]))
