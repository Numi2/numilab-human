"""Summarize the accepted time, determinant and contact bounds from the same native scene."""
import csv
import json
from pathlib import Path
import sys
from bind_gluteal_receivers import digest


def summarize(run):
    lines=(run/'stderr.log').read_text().splitlines()
    def records(prefix):
        return [json.loads(l.partition('=')[2]) for l in lines if l.startswith(prefix+'=')]
    steps=[]
    for r in records('fem_newton_iterate'):
        if r['root']!=1 or r['stage']!='final_line_search':continue
        steps.append(dict(iteration=r['iteration'],fraction=r['alpha'][0],
                          reassembled_residual=r['fgmres'][0]['nonlinear'][1],
                          linear_metric=r['fgmres'][0]['diagnostics'][0]))
    work=records('matter_fgmres_active_work')
    terminal=records('fem_backtracking_trials')
    outputs={}
    for name in ('resting-coupled.csv','resting-com-momentum-diagnostic.csv','resting-com-support-impulses.csv'):
        path=run/'native-traces'/name
        if path.is_file():outputs[name]=digest(path)
    with (run/'native-traces/resting-coupled.csv').open() as f:
        times=[float(r['time_s']) for r in csv.DictReader(f)]
    result=dict(launch=digest(run/'launch.json'),execution=json.loads((run/'execution.json').read_bytes()),
                roots=work,accepted_s=times,accepted_trace_hashes=outputs,second_root_iterations=steps,
                last_geometric_witness=[dict(iteration=r['iteration'],object=r['object'],selected_alpha=r['selected_alpha'],
                     trials=[{k:t[k] for k in ('alpha','first_bad_tet','trial_J','current_J','reason')} for t in r['trials']])for r in terminal],
                contact_minima=records('fem_deformable_limit_minimum'))
    if (run/'failure.json').exists():
        failure=json.loads((run/'failure.json').read_bytes())
        result['failed_after_steps']=failure['segment_start']
        result['rejected']=failure['matter_diagnostics']
    return result


if __name__=='__main__':
    path=Path(sys.argv[1])
    path.write_text(json.dumps({n:summarize(Path(n))for n in sys.argv[2:]},indent=2)+'\n')
