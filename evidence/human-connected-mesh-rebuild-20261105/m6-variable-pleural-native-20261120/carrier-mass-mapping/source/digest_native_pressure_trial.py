"""Extract accepted physical time, pressure, bed reactions and rollback evidence."""
from pathlib import Path
import csv
import json
import re
import sys

run = Path(sys.argv[1])
output = json.loads((run/'execution.json').read_text())
with (run/'native/resting-coupled.csv').open() as stream:
    rows = list(csv.reader(stream))
names = set(rows[0])
assert 'pleural_pressure_pa' in names or 'pleural_pa' in names
pressure = rows[0].index('pleural_pressure_pa' if 'pleural_pressure_pa' in names else 'pleural_pa')
rows = rows[1:]
logs = [line for line in (run/'stdout.log').read_text().splitlines()
        if line.startswith('human_standing_progress=accepted ')]
assert len(logs) == len(rows)
def column(key):
    return [float(re.search(r'\b'+key+r'=([^ ]+)', line).group(1)) for line in logs]
output.update(
    accepted_steps=len(rows), simulated_s=column('simulated_seconds')[-1] if logs else 0.,
    pleural_pa_accepted=[float(row[pressure]) for row in rows],
    maximum_support_force_n=max(column('support_force_n'),default=0.),
    maximum_penetration_m=max(column('penetration_m'),default=0.),
    maximum_root_assistance_force_n=max(column('root_assistance_force_n'),default=0.),
    maximum_root_assistance_torque_nm=max(column('root_assistance_torque_nm'),default=0.),
)
output['wall_real_time_factor'] = output['simulated_s']/output['wall_s']
failure = run/'failure.json'
if failure.is_file():
    data = json.loads(failure.read_text())
    output['first_failure'] = {key:data[key] for key in ('matter_code','matter_object_index','matter_microsteps','matter_diagnostics')}
    output['accepted_brain_time_us'] = data['brain_input']['source_time_us']
    output['rejected_brain_target_time_us'] = data['brain_candidate']['timestamp_us']
    assert data['first_failure_latch']['brain_accepted_metadata'][1] == output['accepted_brain_time_us']
print(json.dumps(output,indent=2))
(run/'accepted-digest.json').write_text(json.dumps(output,indent=2)+'\n')
