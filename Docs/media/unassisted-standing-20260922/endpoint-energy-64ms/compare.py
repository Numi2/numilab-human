#!/usr/bin/env python3
"""Inspect the three retained native prefixes; never claim energy closure."""
import gzip
import json
import math
from pathlib import Path
import struct
import sys

def check(condition, message):
    if not condition:
        raise ValueError(message)

def fields(line):
    parsed = {}
    for part in line.split():
        key, separator, value = part.partition('=')
        check(bool(separator) and key not in parsed, f'bad or duplicate energy field: {part}')
        parsed[key] = value
    return parsed

def number(record, key):
    value = float(record[key])
    check(math.isfinite(value), f'nonfinite energy field: {key}')
    return value

def close(actual, expected, label):
    check(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-14),
          f'{label}: {actual} != {expected}')

work_keys = (
    'gravity_potential_work_j', 'passive_potential_work_j',
    'muscle_midpoint_work_j', 'joint_damping_midpoint_work_j',
    'body_damping_midpoint_work_j', 'contact_normal_endpoint_work_j',
    'contact_tangential_endpoint_work_j', 'equality_endpoint_work_j',
)
unavailable_keys = (
    'source_limit_endpoint_work', 'exact_projection_work',
    'musculotendon_internal_energy',
)

root = Path(sys.argv[1])
check((root / 'inputs-before.sha256').read_bytes() ==
      (root / 'inputs-after.sha256').read_bytes(), 'input bytes changed')
names = ('feedback-energy', 'feedback-plain', 'feedback-disabled-energy')
states = {}
for name in names:
    folder = root / name
    check((folder / 'exit-code.txt').read_text().strip() == '0', f'{name} did not exit zero')
    raw = folder / 'stdout.txt'
    lines = (raw.read_text() if raw.exists() else
             gzip.decompress((folder / 'stdout.txt.gz').read_bytes()).decode()).splitlines()
    terminal = [line.partition('=')[2] for line in lines if line.startswith('stand_terminal_state=')]
    check(len(terminal) == 1, f'{name} has no unique terminal state')
    state = states[name] = json.loads(terminal[0])
    check(state['step_count'] == 64 and state['timestep_seconds'] == 0.001,
          f'{name} did not accept the 64 ms prefix')
    check(state['root_assistance'] is False, f'{name} used root assistance')
    check(len(state['q']) == 129 and len(state['v']) == 128 and
          len(state['initial_q']) == 129 and len(state['initial_v']) == 128,
          f'{name} has incomplete terminal state')
    check(all(math.isfinite(x) for key in ('q', 'v', 'initial_q', 'initial_v')
              for x in state[key]), f'{name} has nonfinite state')
    steps = [line for line in lines if line.startswith('human_endpoint_energy=accepted ')]
    totals = [line for line in lines if line.startswith('human_endpoint_energy_total=observed ')]
    check(len(steps) == (0 if name == 'feedback-plain' else 64),
          f'{name} has incomplete accepted energy rows')
    check(len(totals) == (0 if name == 'feedback-plain' else 1),
          f'{name} has no unique energy total')
    previous_kinetic = None
    accepted = []
    for index, line in enumerate(steps, 1):
        step = fields(line)
        check(int(step['step']) == index, 'accepted endpoint step index has a gap')
        check(step['closure'] == 'not_established', 'step falsely claims energy closure')
        check(all(step[key] == 'unavailable' for key in unavailable_keys),
              'step omits an unavailable physical energy term')
        before = number(step, 'kinetic_before_j')
        after = number(step, 'kinetic_after_j')
        check(min(before, after) >= 0, 'negative endpoint kinetic energy')
        close(number(step, 'delta_kinetic_j'), after - before, 'step kinetic delta')
        for key in (*work_keys, 'equality_solver_iterate_work_j'):
            number(step, key)
        if previous_kinetic is not None:
            check(before == previous_kinetic, 'accepted endpoint chain has a gap')
        previous_kinetic = after
        accepted.append(step)
    print(f'{name}: accepted=64 duration=0.064s root_assistance=false')
    for line in totals:
        total = fields(line)
        check(int(total['accepted_steps']) == len(accepted), 'energy total step count mismatch')
        check(total['closure'] == 'not_established', 'total falsely claims energy closure')
        check(all(total[key] == 'unavailable' for key in unavailable_keys) and
              total['integration_bias_and_quadrature_remainder'] == 'unresolved',
              'total omits an unavailable physical energy term')
        close(number(total, 'kinetic_before_j'), number(accepted[0], 'kinetic_before_j'),
              'total initial kinetic')
        close(number(total, 'kinetic_after_j'), number(accepted[-1], 'kinetic_after_j'),
              'total final kinetic')
        close(number(total, 'delta_kinetic_j'),
              number(total, 'kinetic_after_j') - number(total, 'kinetic_before_j'),
              'total kinetic delta')
        for key in (*work_keys, 'equality_solver_iterate_work_j'):
            close(number(total, key), math.fsum(number(step, key) for step in accepted),
                  f'total {key}')
        # Solver-iterate correction work is diagnostic, not accepted endpoint work.
        available = math.fsum(number(total, key) for key in work_keys)
        close(number(total, 'unclosed_available_terms_residual_j'),
              number(total, 'delta_kinetic_j') - available, 'available-terms residual')
        print(line)

def fp32(values):
    return struct.pack(f'<{len(values)}f', *values)

reference = states['feedback-plain']
observed = states['feedback-energy']
for name, state in states.items():
    for key in ('initial_q', 'initial_v'):
        check(fp32(reference[key]) == fp32(state[key]),
              f'{name} changed initial {key}')
for key in ('q', 'v'):
    check(fp32(reference[key]) == fp32(observed[key]),
          f'accounting changed terminal {key}')
print('accounting_on_off_terminal_q_v=bitwise_equal same_binary_and_inputs=true')
disabled = states['feedback-disabled-energy']
for key in ('q', 'v'):
    difference = [abs(a - b) for a, b in zip(observed[key], disabled[key])]
    print(f'feedback_enabled_disabled_max_{key}_delta_mixed_units={max(difference):.17g} index={difference.index(max(difference))}')
print('scope=short_prefix_accounting_only energy_closure=not_established ten_second_energy_run=not_performed')
