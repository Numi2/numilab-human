"""Record accepted work, actual bed reaction and first-failure ownership without treating a rejected trial as motion."""
import csv,json,sys,re
from pathlib import Path
p=Path(sys.argv[1]);a=json.loads((p/'admission.json').read_text()); s=(p/'stdout.log').read_text();err=(p/'stderr.log').read_text()
l=json.loads((p/'launch.json').read_text());dt=float(l['argv'][l['argv'].index('--muscle-step-seconds')+1]);requested=a['requested_roots']
failure=p/'failure.json';j=json.loads(failure.read_text())['first_failure_latch'] if failure.exists() else {}
rows=list(csv.DictReader((p/'native/resting-coupled.csv').open())) if (p/'native/resting-coupled.csv').exists() else []
receipt=p/'native/accepted-geometry'/f'step-{requested}.receipt.json'
if a['exit_code']==0:
    r=json.loads(receipt.read_text());steps=r['accepted_step'];t=r['accepted_time_s'];assert steps==requested and abs(t-steps*dt)<dt/5
else:steps=j.get('step');t=None if steps is None else steps*dt
def numbers(prefix):
    ls=[s0 for s0 in s.splitlines() if s0.startswith(prefix)]
    return [dict(re.findall(r'(\w+)=([^ ]+)',v)) for v in ls]
result=dict(accepted_steps=steps,requested_steps=requested,accepted_seconds=t,native_launch_wall_seconds=a['wall_seconds'],end_to_end_factor=(t/a['wall_seconds'] if t is not None else None),
            device='Apple M6',dt_s=dt,exit_code=a['exit_code'],watchdog=a['watchdog'],bed_reaction=numbers('resting_fem_skin_bed_accepted_root_reaction'),bed_history=numbers('resting_fem_skin_bed_terminal_history'),
            initial_install=numbers('registered_loaded_material_install='),material_endpoints=numbers('registered_cell_shape '),
            failure=dict(matter_status=j.get('matter_status'),code=j.get('matter_code'),diagnostics=j.get('matter_diagnostics'),accepted_brain=j.get('brain_accepted_metadata'),accepted_respiration=j.get('respiration_accepted_status')) if j else None,
            pleural_samples_pa=[r['pleural_pa'] for r in rows if 'pleural_pa' in r],variable_pleural_configured=l['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']=='1')
(p/'native-outcome.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('accepted_steps','accepted_seconds','end_to_end_factor','bed_reaction','failure','watchdog')})[:1500])
