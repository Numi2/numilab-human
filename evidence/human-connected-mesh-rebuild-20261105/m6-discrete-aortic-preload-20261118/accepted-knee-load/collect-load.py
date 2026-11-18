from pathlib import Path
import gzip
import json
import re

out = Path(__file__).resolve().parent
with gzip.open(out / 'accepted-load.stdout.log.gz', mode='rt') as lines:
    report = list(lines)

def single(prefix):
    row, = [line.strip() for line in report if line.startswith(prefix)]
    return row

def vector(row, label):
    return [float(value) for value in re.search(rf' {label}=\[([^\]]+)\]', row)[1].split(',')]

load_row = single('accepted_fem_body_load_capture=accepted')
assert ' source_step=0 ' in load_row
load = vector(load_row, 'generalized_load')
q = vector(load_row, 'source_q')
assert len(load) == 128 and len(q) == 129
profile = single('resting_native_profile step=1 ')
bed = single('resting_fem_skin_bed_accepted_root_reaction ')
output = {
    'scope': 'Accepted first root release diagnostic at 15.625 us, no resting settlement claim',
    'source_step': 0,
    'accepted_count': 1,
    'accepted_pre_dynamics_source_q': q,
    'accepted_pre_dynamics_fem_generalized_force_si': load,
    'root_translation_force_N': load[:3],
    'root_rotation_torque_Nm': load[3:6],
    'knee_flexion_torque_Nm': {'right': load[105], 'left': load[119]},
    'ankle_flexion_torque_Nm': {'right': load[108], 'left': load[122]},
    'source_joint_load_audits': [single(f'resting_initial_joint_load dof={dof} ') for dof in (105, 119, 108, 122)],
    'native_profile': profile,
    'bed_reaction': bed,
}
(out / 'load-and-source-pose.json').write_text(json.dumps(output, indent=2) + '\n')
