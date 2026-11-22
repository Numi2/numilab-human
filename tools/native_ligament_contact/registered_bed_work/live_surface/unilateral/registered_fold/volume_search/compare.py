"""Compare physical clocks, fresh residuals and deterministic material witnesses at identical settings."""
import csv
import json
from pathlib import Path
import re
import sys
from material_context import sha
from verify_bonded_native import matched_horizon

def read(p):return json.loads(p.read_text())
def records(s,key):return [json.loads(l[len(key)+1:]) for l in s.splitlines() if l.startswith(key+'=')]
def verify(root,extra):
    baseline=root/'fold-repair/three';measured=root/'material-direction'
    def configuration(p):
        job=read(p/'launch.json');cwd,args,pins,env=matched_horizon(job)
        for k in ('NUMI_MATTER_METALLIB_OVERRIDE','DYLD_LIBRARY_PATH'):
            path=Path(env.pop(k));path=path/'libmetalrobo.dylib' if k=='DYLD_LIBRARY_PATH' else path
            if pins.pop(str(path))!=sha(path):raise ValueError('Stale native library')
        env.pop('NM_FEM_REGISTERED_VOLUME_DIRECTION',None)
        env.pop('NM_FEM_REGISTERED_RELEASE_BOUNDS',None)
        return cwd,args,pins,env
    sources=[baseline,measured/'three',measured/'quantized/three',*extra]
    common=configuration(baseline);states=[]
    for case in sources:
        if configuration(case)!=common:raise ValueError('Unmatched physics, observer or solver options')
        launch=read(case/'launch.json');out=(case/'stdout.log').read_text();log=(case/'stderr.log').read_text()
        loads=[l for l in out.splitlines() if l.startswith('accepted_fem_body_load_capture=accepted ')]
        with (case/'native-traces/resting-coupled.csv').open() as f:trace=list(csv.DictReader(f))
        status=read(case/'execution.json');stop=case/'failure.json'
        clocks=[float(m.group(1)) for m in re.finditer(r'^human_standing_progress=accepted step=\d+ simulated_seconds=([\d.eE+-]+)',out,re.M)]
        if len(loads)!=len(clocks):raise ValueError('Inconsistent accepted force and progress counts')
        terminal=records(log,'fem_newton_terminal_certificate')
        row=dict(run=str(case.relative_to(root)),launch=sha(case/'launch.json'),source=sha(case/'stderr.log'),
            accepted_steps=len(clocks),simulated_s=clocks[-1] if clocks else 0.,wall_s=status['wall_s'],
            physical_prefix_loads_equal_to_control=loads[:len(controls)]==controls if states else True,
            physical_prefix_trace_equal_to_control=trace[:len(controltrace)]==controltrace if states else True,
            terminal_certificates=[dict(root=r['root'],status=r['status'],fgmres=r['fgmres']) for r in terminal],
            work=records(log,'matter_fgmres_microtick_work'),failed=status['exit']!=0)
        if not states:controls=loads;controltrace=trace
        for name in ('NM_FEM_REGISTERED_BED_DUAL','NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ'):
            if name not in launch['environment']:raise ValueError('Missing active physics')
        if stop.exists():
            failure=read(stop)
            if failure['segment_start']!=len(clocks):raise ValueError('Rejected time mismatch')
            row.update(fresh=terminal[-1]['objects'][15]['nonlinear'][0],failing_native_index=failure['matter_failing_index'],matter_diagnostics=failure['matter_diagnostics'],matter_code=failure['matter_code'],
                rollback_common=failure['candidate_common_coordinates']==failure['accepted_common_coordinates'],
                rollback_respiratory=failure['respiratory_candidate_status']==failure['respiratory_accepted_status'])
            if failure['matter_diagnostics'][3]==2 and not any(terminal[-1]['objects'][15]['nonlinear']) and not any(terminal[-1]['objects'][15]['contact']):
                row.pop('fresh')
                row['residual_status']='geometry rejected during line search before final force assembly'
        elif status['exit']!=0:
            terminal_error=re.search(r'resting_fem_terminal_reject object=(.+?) global_tet=(\d+) local_tet=(\d+) J=([^ ]+) rest_det_m3=([^ ]+) current_det_m3=([^ ]+)',log)
            if not terminal_error:raise ValueError('Unclassified failed native capture')
            row.update(terminal_rejection=dict(object=terminal_error[1],element=int(terminal_error[2]),
                loaded_relative_J=float(terminal_error[4]),loaded_determinant_m3=float(terminal_error[5]),
                candidate_determinant_m3=float(terminal_error[6])),
                fresh=terminal[-1]['objects'][15]['nonlinear'][0],qualified_endpoint=False)
        else:
            r=read(case/'terminal.receipt.json')
            if not (r['accepted_step']==len(clocks) and r['physical_endpoint']=='accepted'):
                raise ValueError('Missing complete native state')
        for logrow in records(log,'fem_backtracking_trials'):
            if not logrow['trials']:
                continue
            row['last_tissue_line_search']=dict(root=logrow['root'],iteration=logrow['iteration'],
                material_alpha=logrow['selected_alpha'],full_trial=logrow['trials'][0])
        states.append(row)
    return states
if __name__=='__main__':print(json.dumps(verify(Path(sys.argv[1]).resolve(),[Path(v).resolve() for v in sys.argv[2:]]),indent=2,allow_nan=False))
