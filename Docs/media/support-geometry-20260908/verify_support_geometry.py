#!/usr/bin/env python3
"""Recompute the bounded support result from retained native logs and states."""
from pathlib import Path
import hashlib
import json
import math
import re

root = Path(__file__).resolve().parent

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read(name):
    return json.loads((root / name).read_text())

def metrics(row):
    raw = (root / (row['label'] + '.log')).read_text()
    result = dict(re.findall(r'([a-zA-Z_0-9]+)=([^\s]+)', raw))
    for key, value in row['metrics'].items():
        require(result.get(key) == value, f"log/summary disagreement: {row['label']} {key}")
    terminal = next((json.loads(line.split('=', 1)[1]) for line in raw.splitlines()
                     if line.startswith('stand_terminal_state=')), None)
    if 'terminal_state' in row:
        require(terminal == row['terminal_state'], f"terminal state mismatch: {row['label']}")
    return result, terminal

receipt_path = root / 'receipt.json'
if receipt_path.exists():
    for name, expected in read('receipt.json')['files'].items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
                f'artifact hash drift: {name}')

before = read('before.json')
rows = read('qualified.json')
require(len(before) == 5 and len(rows) == 7, 'missing baseline or qualification horizon')
old_metrics = [metrics(row)[0] for row in before]
new = [metrics(row) for row in rows]
require(all(row['exit_code'] == 0 for row in before + rows), 'a native horizon failed')
require(len({row['binary_sha256'] for row in rows}) == 1, 'mixed qualification executable')
require(all(float(m['compiled_support_max_separated_force_n']) == 0.0 for m, _ in new),
        'static force across a gap')
require(all(m['compiled_support_geometry'] == 'admissible' and
            float(m['compiled_support_min_gap_m']) >= -float(m['compiled_support_gap_tolerance_m'])
            for m, _ in new), 'penetrating static geometry admitted')
require(all(int(m['compiled_support_rejected_pose_candidates']) > 0 for m, _ in new),
        'the anatomical penetrating-pose regression did not execute')
require(all(m['stand_deterministic_replay'] == 'bitwise' for m, _ in new), 'native replay failed')
require(all(m['compiled_stand_balanced'] == 'false' for m, _ in new), 'balance claim changed')
for row, (_, state) in zip(rows, new):
    require(state is not None and not state['root_assistance'], 'missing or assisted terminal state')
    require(state['schema'] == 'numi.human.legacy-stand-terminal.v1', 'unknown terminal schema')
    command = row['command']
    require(state['step_count'] == int(command[command.index('--muscle-step-count') + 1]) and
            state['timestep_seconds'] == float(command[command.index('--muscle-step-seconds') + 1]),
            'horizon command/state disagreement')
    for key, count in [('q', 129), ('initial_q', 129), ('v', 128), ('initial_v', 128)]:
        require(len(state[key]) == count and all(math.isfinite(v) for v in state[key]),
                f'incomplete/nonfinite {key}')
    require(state['initial_q'] == new[0][1]['initial_q'] and
            state['initial_v'] == new[0][1]['initial_v'], 'initial state changed with timestep')
    require(float(dict(re.findall(r'([a-zA-Z_0-9]+)=([^\s]+)',
                                 (root / (row['label'] + '.log')).read_text()))[
        'persistent_max_root_assistance_force_n']) == 0.0, 'hidden root assistance force')

# Command equivalence keeps source payloads, flags, timestep and horizon fixed.
old_command = list(before[0]['command']); new_command = list(rows[0]['command'])
old_command[4] = new_command[4] = '<output>'
require(old_command == new_command, 'one-step before/after commands differ')
old_jump = float(old_metrics[0]['muscle_step_max_velocity_delta'])
new_jump = float(new[0][0]['muscle_step_max_velocity_delta'])
require(new_jump < old_jump * 0.01, 'initial kick was not removed')
require(float(new[0][0]['source_support_min_plane_gap_m']) >= -1e-6,
        'initial support penetration persists')

refinements = [state for _, state in new[1:5]]
require(all(abs(s['step_count'] * s['timestep_seconds'] - 0.0006) < 1e-12
            for s in refinements), 'timestep comparisons have unequal physical duration')
def maximum_difference(a, b, key):
    return max(abs(x - y) for x, y in zip(a[key], b[key]))
deltas = [{key: maximum_difference(a, b, key) for key in ['q', 'v']}
          for a, b in zip(refinements, refinements[1:])]
refinement_trend = all(all(a[key] > b[key] for a, b in zip(deltas, deltas[1:]))
                       for key in ['q', 'v'])
require(refinement_trend, 'short refinement differences do not decrease')
regressions = {r['label']: r for r in read('regressions.json')}
for label in ['analytic-correct', 'analytic-restored', 'muscle-reference',
              'human-io-validated']:
    require(regressions[label]['exit_code'] == 0, f'regression failed: {label}')
require(regressions['analytic-mutant']['exit_code'] != 0,
        'unsupported-force negative control did not fail')
require('airborne' in (root/'analytic-mutant.log').read_text() or
        'separated witness' in (root/'analytic-mutant.log').read_text(),
        'negative control failed for an unrelated reason')
require('argument motorReadyGate' in (root/'human-io-corrected.log').read_text(),
        'original Metal binding failure is missing')
require('Metal API Validation Enabled' in (root/'human-io-validated.log').read_text() and
        'numanx_human_io_probe passed' in (root/'human-io-validated.log').read_text(),
        'corrected IO was not validated by Metal')
costal_attempts = [r for name, r in regressions.items() if name.startswith('costal-')]
require(costal_attempts, 'missing costal regression attempt')
for attempt in costal_attempts:
    require((root / (attempt['label'] + '.log')).is_file(), 'missing costal attempt log')
costal_passed = costal_attempts[-1]['exit_code'] == 0
pre_io = read('final.json')
require(all(a['terminal_state'] == b['terminal_state'] for a, b in zip(pre_io, rows))
        and len(pre_io) == len(rows), 'IO binding fix changed native standing states')
report = {
    'format': 'numi-human-support-geometry-verification-v1',
    'source_support_geometry': True,
    'static_airborne_force_zero': True,
    'negative_control_detected': True,
    'replay_exact': True,
    'initial_velocity_jump_before': old_jump,
    'initial_velocity_jump_after': new_jump,
    'initial_velocity_jump_reduction_factor': old_jump / new_jump,
    'equal_duration_seconds': 0.0006,
    'refinement_maximum_differences': deltas,
    'short_refinement_trend': refinement_trend,
    'longest_horizon_seconds': max(s['step_count'] * s['timestep_seconds'] for _, s in new),
    'native_standing_states_unchanged_after_io_binding_fix': True,
    'maximum_static_root_force_residual_n': float(new[0][0]['compiled_stand_max_root_force_residual']),
    'metal_io_validation': True,
    'costal_transaction_regression': costal_passed,
    'costal_attempt_exit_codes': [r['exit_code'] for r in costal_attempts],
    'static_balance': False,
    'source_compliant_sustained_behavior': False,
    'experimental_calibration': False,
    'performance_qualification': False,
    'full_release': False,
}
print(json.dumps(report, indent=2))
