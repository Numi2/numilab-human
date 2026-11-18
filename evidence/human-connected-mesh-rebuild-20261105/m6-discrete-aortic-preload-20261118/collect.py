from pathlib import Path
import gzip
import hashlib
import json
import re
import shutil

out = Path(__file__).resolve().parent
work = out.parent
trials = {
    'recruitment-025-pressure-ramp': 'resume-load-continuation-knees-001',
    'recruitment-025-four-support': 'resume-contact-four-support-001',
    'recruitment-040-full-pressure': 'resume-load-balance-observer-001',
    'recruitment-025-discrete-wall': 'resume-aortic-discrete-native-002',
}

def prefix_records(log, prefix):
    records = []
    for line in log.splitlines():
        if not line.startswith(prefix):
            continue
        fields = dict(field.split('=', 1) for field in line.split()[1:] if '=' in field)
        records.append({key: float(value) for key, value in fields.items()})
    return records

def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

result = {}
for trial, directory in trials.items():
    folder = work/directory
    raw = (folder/'install-motion.stdout.log.gz')
    stderr = folder/'install-motion.stderr.log'
    plan = json.loads((folder/'install-motion.plan.json').read_text())
    execution = json.loads((folder/'install-motion.execution.json').read_text())
    assert digest(raw) == execution['stdout_sha256']
    assert digest(stderr) == execution['stderr_sha256']
    reports = gzip.decompress(raw.read_bytes()).decode()
    errors = stderr.read_text().splitlines()
    profiles = prefix_records(reports, 'resting_native_profile ')
    observations = prefix_records(reports, 'supported_prelude_observation ')
    startup = prefix_records(reports, 'resting_startup_load ')
    for row in startup:
        del row['aorta_mmhg']  # an early observer mislabeled total circulation volume
    work_rows = [json.loads(line.split('=',1)[1]) for line in errors
                 if line.startswith('matter_fgmres_active_work=')]
    assert len(profiles)==len(observations)==len(work_rows)
    command_sec = sum(row['command_ms'] for row in profiles)/1e3
    gpu_sec = sum(row['gpu_ms'] for row in profiles)/1e3
    simulated_sec = len(profiles)*15.625e-6
    stage=plan['stages'][0]
    stage['environment'] = dict(sorted(stage['environment'].items()))
    (out/(trial+'.launch.json')).write_text(json.dumps({
        'launch': stage, 'binary': plan['native_binary'],
        'runtime_dependencies': plan['runtime_dependencies'],
        'inputs': plan['required_transitive_input_pins'],
        'runner': execution['plan_path'],
        'plan_sha256': execution['plan_sha256'],
    }, indent=2)+'\n')
    result[trial] = {
        'initialization': next(line for line in reports.splitlines()
                               if line.startswith('resting_initialization=dynamic_release ')),
        'accepted_steps': len(profiles), 'accepted_physical_seconds': simulated_sec,
        'native_command_seconds': command_sec, 'gpu_seconds': gpu_sec,
        'full_process_seconds': execution['elapsed_s'],
        'command_scope_real_time_factor': simulated_sec/command_sec,
        'process_scope_real_time_factor': simulated_sec/execution['elapsed_s'],
        'corrected_accepted_velocity_settlement_predicate':
            trial in ('recruitment-040-full-pressure', 'recruitment-025-discrete-wall'),
        'work': work_rows, 'profiles': profiles, 'settlement_observations': observations,
        'startup_motion': startup, 'failure': errors[-1],
        'initial_joint_moments': [line for line in reports.splitlines()
            if line.startswith('resting_initial_joint_load ')],
        'discrete_preload': [line for line in reports.splitlines()
            if line.startswith('resting_aortic_discrete_preload=compiled ')],
        'raw_evidence_sha256': {path.name: digest(path) for path in (raw,stderr)},
        'raw_replay_sha256': digest(folder/'install-motion/resting-human.mrvpack'),
        'settled_and_motion_qualified': False
    }
ref = result['recruitment-025-pressure-ramp']
candidate = result['recruitment-025-four-support']
result['matched_contact_direction_comparison'] = {
    'same_observed_motion': ref['startup_motion']==candidate['startup_motion'],
    'same_settlement_observations': ref['settlement_observations']==candidate['settlement_observations'],
    'same_solver_counters_and_residuals': ref['work']==candidate['work'],
    'command_seconds_change_fraction': candidate['native_command_seconds']/ref['native_command_seconds']-1,
    'gpu_seconds_change_fraction': candidate['gpu_seconds']/ref['gpu_seconds']-1,
    'qualified_for_solver_promotion': False
}
wall = result['recruitment-025-discrete-wall']
prefix_ref = ref['profiles'][:8]
prefix_ref_work = ref['work'][:8]
prefix_reference_milliseconds = sum(row['command_ms'] for row in prefix_ref)
prefix_reference_columns = sum(row['used_columns_sum'] for row in prefix_ref_work)
wall_columns = sum(row['used_columns_sum'] for row in wall['work'])
result['wall_eight_root_comparison'] = {
    'scope': 'Accepted prefix; source recruitment cap matches. Pressure is full at root one in the fitted wall, and the reference uses the two-step pressure ramp.',
    'reference_command_seconds': prefix_reference_milliseconds/1e3,
    'wall_command_seconds': wall['native_command_seconds'],
    'command_seconds_change_fraction': wall['native_command_seconds']*1e3/prefix_reference_milliseconds-1,
    'reference_columns': prefix_reference_columns,
    'wall_columns': wall_columns,
    'column_count_change_fraction': wall_columns/prefix_reference_columns-1,
    'reference_peak_fem_speed_m_s':max(row['max_fem_speed'] for row in ref['settlement_observations'][:8]),
    'wall_peak_fem_speed_m_s':max(row['max_fem_speed'] for row in wall['settlement_observations']),
}
(out/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')
for src, dst in (
    ('resume-load-continuation-knees-001/regional-startup-audit.patch','regional-startup-audit.patch'),
    ('resume-load-balance-observer-001/accepted-balance-observer.patch','accepted-balance-observer.patch'),
    ('resume-contact-four-support-001/four-support.patch','four-support.patch'),
    ('resume-contact-four-support-001/exact-pack-clones.log','exact-pack-clones.log'),
):
    shutil.copyfile(work/src,out/dst)
