#!/usr/bin/env python3
"""Compare four source-bound native traces; this is not a qualification gate.

Usage: python compare_traces.py CASE_DIR_100 CASE_DIR_50 CASE_DIR_25 CASE_DIR_12P5
Each CASE_DIR contains execution.json and NAME/stdout.txt, stderr.txt as emitted
by human-bounded-release.yml. Printed metrics compare captured projected q/v,
not all authoritative compensated coordinates, muscle states or force owners.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import sys
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def strict_json(text: str) -> Any:
    def invalid(value: str) -> None:
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def load_case(folder: Path) -> dict[str, Any]:
    manifest_bytes = (folder / 'execution.json').read_bytes()
    manifest = strict_json(manifest_bytes.decode())
    require(manifest['schema'] == 'numi.human.hosted-bounded-release.v1', 'unsupported execution manifest')
    for key in ['native_commit', 'input_commit']:
        require(re.fullmatch('[0-9a-f]{40}', manifest[key]) is not None, 'invalid source revision')
    require(manifest['initialization'] == 'canonical_numi_human_stand_authored_support_stance', 'initialization mismatch')
    require(manifest['rendering_requested'] is False, 'not a mechanics-only run')
    require(len(manifest['cases']) == 1, 'expected exactly one matrix case')
    case = manifest['cases'][0]
    name, ns, steps = case['name'], case['timestep_nanoseconds'], case['steps']
    require((name, ns, steps) in [('100us', 100000, 64), ('50us', 50000, 128),
                                ('25us', 25000, 256), ('12p5us', 12500, 512)], 'wrong common-duration grid')
    require(type(ns) is int and type(steps) is int and ns * steps == 6400000, 'wrong clock')
    require(type(case['exit_code']) is int and case['exit_code'] == 0, 'native process incomplete')
    require(case['status'] == 'process_completed' and case['source_artifacts_unchanged'] is True,
            'source changed or process incomplete')
    text = ''
    for kind in ['stdout', 'stderr']:
        path = folder / name / (kind + '.txt')
        require(path.is_file() and not path.is_symlink(), 'missing or redirected native log')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == case[kind + '_sha256'], 'native log hash mismatch')
        if kind == 'stdout':
            text = raw.decode('utf-8')
    summary = [line for line in text.splitlines() if line.startswith('myosim_articulated_mechanics=')]
    require(len(summary) == 1, 'missing or ambiguous native summary')
    metrics = unique([tuple(token.split('=', 1)) for token in shlex.split(summary[0]) if '=' in token])
    require(metrics == case['metrics'], 'manifest metrics disagree with actual native output')
    for key, expected in {
        'myosim_articulated_mechanics': 'ok', 'rendering_performed': 'false',
        'persistent_root_assistance': 'none', 'core_bodies': '157',
        'visual_coverage_qualified': 'false', 'persistent_metal_horizon': 'true',
        'compiled_stand_recruited_muscles': '416',
        'compiled_stand_balanced': 'true', 'persistent_completed_steps': str(steps),
        'muscle_step_count': str(steps), 'stand_deterministic_replay': 'bitwise',
        'persistent_passive_joint_law': 'current_state_linear_backward_euler',
    }.items():
        require(metrics.get(key) == expected, 'native execution contract: ' + key)
    require(float(metrics['muscle_step_seconds']) == ns * 1e-9 or
            abs(float(metrics['muscle_step_seconds']) - ns * 1e-9) < 1e-15, 'summary clock differs')
    rows = [line.split('=', 1)[1] for line in text.splitlines() if line.startswith('persistent_stand_trace=')]
    require(len(rows) == 1, 'missing or ambiguous native trace')
    trace = strict_json(rows[0])
    require(trace['schema'] == 'numi.human.persistent-stand-trace.v4', 'unsupported trace schema')
    require(trace['endpoint_equivalent'] == 'bitwise' and trace['endpoint_max_q_delta'] == 0
            and trace['endpoint_max_v_delta'] == 0, 'trace replay differs')
    samples = trace['samples']
    require(len(samples) == steps + 1, 'trace is incomplete')
    for i, sample in enumerate(samples):
        require(type(sample['step']) is int and sample['step'] == i, 'trace ordering is invalid')
        require(finite(sample['time_seconds']) and abs(sample['time_seconds'] - i * ns * 1e-9) < 1e-12,
                'trace sample has the wrong clock')
        for key, count in [('q', 129), ('v', 128)]:
            require(len(sample[key]) == count and all(finite(x) for x in sample[key]), 'invalid trace vector')
        require(abs(math.sqrt(sum(x*x for x in sample['q'][3:7])) - 1.0) < 2e-5, 'invalid root quaternion')
        require(finite(sample['normal_impulse']) and sample['normal_impulse'] >= 0, 'invalid normal impulse')
    return {'manifest': manifest, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'case': case, 'samples': samples, 'folder': str(folder)}


def maximum_delta(a: list[float], b: list[float]) -> float:
    return max(abs(x-y) for x, y in zip(a, b, strict=True))


def orientation_delta(a: list[float], b: list[float]) -> float:
    a = [x/math.sqrt(sum(v*v for v in a)) for x in a]
    b = [x/math.sqrt(sum(v*v for v in b)) for x in b]
    sign = 1 if sum(x*y for x, y in zip(a, b)) >= 0 else -1
    # Sign-invariant relative angle, stable close to zero.
    return 4 * math.atan2(math.sqrt(sum((x-sign*y)**2 for x, y in zip(a, b))),
                          math.sqrt(sum((x+sign*y)**2 for x, y in zip(a, b))))


def compare(folders: list[Path]) -> dict[str, Any]:
    require(len(folders) == 4, 'all four grids are required')
    runs = sorted([load_case(folder) for folder in folders], key=lambda r: r['case']['timestep_nanoseconds'], reverse=True)
    require([r['case']['timestep_nanoseconds'] for r in runs] == [100000, 50000, 25000, 12500], 'duplicate or missing grid')
    for key in ['native_commit', 'input_commit']:
        require(len({r['manifest'][key] for r in runs}) == 1, 'mixed source revisions')
    payloads = ['rigid', 'muscle', 'tendon', 'support_contact', 'joint_equalities', 'launcher']
    for key in payloads:
        require(len({r['manifest']['artifacts'][key]['sha256'] for r in runs}) == 1, 'mixed runtime input: ' + key)
    binaries = {r['manifest']['artifacts']['binary']['sha256'] for r in runs}
    finest = runs[-1]
    comparisons = []
    for run in runs[:-1]:
        q_translation, q_orientation, q_scalar, velocity, impulse_force = [], [], [], [], []
        stride = 100000 // run['case']['timestep_nanoseconds']
        fine_stride = 8
        for i in range(65):
            a, b = run['samples'][i*stride], finest['samples'][i*fine_stride]
            q_translation.append(maximum_delta(a['q'][:3], b['q'][:3]))
            q_orientation.append(orientation_delta(a['q'][3:7], b['q'][3:7]))
            q_scalar.append(maximum_delta(a['q'][7:], b['q'][7:]))
            velocity.append(maximum_delta(a['v'], b['v']))
            if i:
                coarse = sum(s['normal_impulse'] for s in run['samples'][(i-1)*stride+1:i*stride+1])
                fine = sum(s['normal_impulse'] for s in finest['samples'][(i-1)*fine_stride+1:i*fine_stride+1])
                impulse_force.append((coarse - fine) / 0.0001)
        comparisons.append({'case': run['case']['name'], 'reference': finest['case']['name'], 'common_times': 65,
            'root_translation_max_m': max(q_translation), 'root_orientation_max_rad': max(q_orientation),
            'scalar_configuration_max_mixed_units': max(q_scalar), 'generalized_velocity_max_mixed_units': max(velocity),
            'normal_reaction_common_interval_max_delta_n': max(abs(x) for x in impulse_force),
            'normal_reaction_common_interval_rms_delta_n': math.sqrt(sum(x*x for x in impulse_force)/len(impulse_force))})
    initial = finest['samples'][0]
    rows = []
    for run in runs:
        c, m = run['case'], run['case']['metrics']
        rows.append({'name': c['name'], 'steps': c['steps'], 'timestep_nanoseconds': c['timestep_nanoseconds'],
            'manifest_sha256': run['manifest_sha256'], 'stdout_sha256': c['stdout_sha256'],
            'binary_sha256': run['manifest']['artifacts']['binary']['sha256'],
            'device': m['source_support_metal_device'],
            'static_residual_rms': float(m['compiled_stand_normalized_residual_rms']),
            'static_support_n': float(m['compiled_stand_total_support_force_n']),
            'peak_generalized_acceleration_mixed_units': float(m['persistent_max_acceleration']),
            'peak_penetration_m': float(m['persistent_max_penetration_m']),
            'mean_normal_reaction_n': sum(s['normal_impulse'] for s in run['samples'][1:])/0.0064,
            'initial_projected_q_delta': maximum_delta(run['samples'][0]['q'], initial['q']),
            'initial_v_delta': maximum_delta(run['samples'][0]['v'], initial['v'])})
    return {'schema': 'numi.human.canonical-trace-comparison.v1', 'native_commit': runs[0]['manifest']['native_commit'],
        'input_commit': runs[0]['manifest']['input_commit'], 'common_duration_nanoseconds': 6400000,
        'same_binary_hash': len(binaries) == 1, 'cases': rows, 'comparisons': comparisons,
        'force_convergence': False, 'sustained_standing': False,
        'boundary': 'Offline comparison of captured projected q/v and interval-integrated normal impulses. '
        'Not a full-state or full-force certificate; compensated low components, muscle states, complete '
        'constraint reactions, force residuals and impulsive work are not jointly compared. Hosted '
        'Paravirtual execution is not a physical M4 qualification. No tolerance has been fitted to these results.'}


if __name__ == '__main__':
    try:
        print(json.dumps(compare([Path(x) for x in sys.argv[1:]]), indent=2, allow_nan=False))
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise SystemExit('comparison rejected: ' + str(error))
