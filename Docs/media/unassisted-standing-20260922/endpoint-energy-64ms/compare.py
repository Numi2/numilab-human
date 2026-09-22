#!/usr/bin/env python3
"""Inspect the three retained native prefixes; never claim energy closure."""
import gzip
import json
import math
from pathlib import Path
import struct
import sys

root = Path(sys.argv[1])
assert (root / 'inputs-before.sha256').read_bytes() == (root / 'inputs-after.sha256').read_bytes()
names = ('feedback-energy', 'feedback-plain', 'feedback-disabled-energy')
states = {}
for name in names:
    folder = root / name
    assert (folder / 'exit-code.txt').read_text().strip() == '0', name
    raw = folder / 'stdout.txt'
    lines = (raw.read_text() if raw.exists() else
             gzip.decompress((folder / 'stdout.txt.gz').read_bytes()).decode()).splitlines()
    terminal = [line.partition('=')[2] for line in lines if line.startswith('stand_terminal_state=')]
    assert len(terminal) == 1, name
    state = states[name] = json.loads(terminal[0])
    assert state['step_count'] == 64 and state['timestep_seconds'] == 0.001
    assert state['root_assistance'] is False
    assert len(state['q']) == 129 and len(state['v']) == 128
    assert all(math.isfinite(x) for key in ('q', 'v', 'initial_q', 'initial_v') for x in state[key])
    steps = [line for line in lines if line.startswith('human_endpoint_energy=accepted ')]
    totals = [line for line in lines if line.startswith('human_endpoint_energy_total=observed ')]
    assert len(steps) == (0 if name == 'feedback-plain' else 64), name
    assert len(totals) == (0 if name == 'feedback-plain' else 1), name
    previous_kinetic = None
    for index, line in enumerate(steps, 1):
        fields = dict(part.split('=', 1) for part in line.split())
        assert int(fields['step']) == index
        before = float(fields['kinetic_before_j'])
        after = float(fields['kinetic_after_j'])
        assert math.isfinite(before) and math.isfinite(after) and min(before, after) >= 0
        if previous_kinetic is not None:
            assert before == previous_kinetic, 'accepted endpoint chain has a gap'
        previous_kinetic = after
    print(f'{name}: accepted=64 duration=0.064s root_assistance=false')
    for line in totals:
        assert 'closure=not_established' in line
        print(line)

def fp32(values):
    return struct.pack(f'<{len(values)}f', *values)

reference = states['feedback-plain']
observed = states['feedback-energy']
for name, state in states.items():
    for key in ('initial_q', 'initial_v'):
        assert fp32(reference[key]) == fp32(state[key]), (name, key)
for key in ('q', 'v'):
    assert fp32(reference[key]) == fp32(observed[key]), f'accounting changed terminal {key}'
print('accounting_on_off_terminal_q_v=bitwise_equal same_binary_and_inputs=true')
disabled = states['feedback-disabled-energy']
for key in ('q', 'v'):
    difference = [abs(a - b) for a, b in zip(observed[key], disabled[key])]
    print(f'feedback_enabled_disabled_max_{key}_delta_mixed_units={max(difference):.17g} index={difference.index(max(difference))}')
print('scope=short_prefix_accounting_only energy_closure=not_established ten_second_energy_run=not_performed')
