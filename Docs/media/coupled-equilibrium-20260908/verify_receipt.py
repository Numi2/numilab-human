#!/usr/bin/env python3
"""Verify bounded offline equilibrium evidence; never promote loaded behavior."""
import gzip
import hashlib
import importlib.util
import json
import math
import re
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(value, detail):
    if not value:
        raise ValueError(detail)


def module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def unique(log, prefix):
    rows = [line[len(prefix):] for line in log.splitlines() if line.startswith(prefix)]
    require(len(rows) == 1, 'missing/duplicate ' + prefix)
    return rows[0]


def audit_state(log):
    summary = dict(word.split('=', 1) for word in unique(
        log, 'numi_human_whole_body_support_wrench=').split() if '=' in word)
    require(summary['joint_manifold'] == 'NHEQ1' and summary['support_payload'] == 'NHCNT2', 'scope drift')
    require(summary['passive_joint_tissue'] == 'none', 'unqualified passive tissue')
    residual = float(summary['internal_normalized_residual_rms'])
    require(math.isfinite(residual) and 0 <= residual <= 0.05 and summary['internal_balanced'] == 'true',
            'internal balance gate failed')
    require(summary['replay'] == 'bitwise', 'native replay failed')
    trace = json.loads(unique(log, 'compiled_equilibrium_search_trace='))
    poses = int(summary['accepted_pose_steps'])
    require(len(trace) == poses + 2 and trace[0]['kind'] == 0 and trace[-1]['kind'] == 2,
            'search trace incomplete')
    for index, row in enumerate(trace):
        require(type(row['coupled_pose_proposal']) is bool, 'invalid proposal flag')
        require(all(type(row[k]) in (int, float) and math.isfinite(row[k]) and row[k] >= 0
                    for k in ('normalized_residual_rms', 'objective')), 'invalid trace metric')
        require(row['accepted_pose_steps'] == min(index, poses), 'accepted pose sequence changed')
        require(type(row['rejected_constraint_candidates']) is int and row['rejected_constraint_candidates'] >= 0,
                'invalid rejection count')
        if 0 < index < len(trace) - 1:
            require(row['kind'] == 1, 'nonphysical search kind changed')
        if index:
            require(row['objective'] <= trace[index-1]['objective'] + 1e-12, 'accepted objective increased')
            require(row['rejected_constraint_candidates'] >= trace[index-1]['rejected_constraint_candidates'],
                    'numerical failures omitted')
    require(abs(trace[-1]['normalized_residual_rms'] - residual) <= 1e-11, 'final residual mismatch')
    require(sum(r['coupled_pose_proposal'] for r in trace[1:-1]) == int(summary['accepted_coupled_pose_steps']),
            'coupled search count mismatch')
    require(trace[-1]['rejected_constraint_candidates'] == int(summary['rejected_constraint_candidates']),
            'rejected evaluation count mismatch')
    muscles = json.loads(unique(log, 'compiled_equilibrium_muscles='))
    require(muscles['schema'] == 'numi.human.offline-muscle-state.v1', 'muscle state schema')
    keys = ('activation_fp64', 'activation_fp32', 'reference_fiber_length_m',
            'actuator_force_n', 'passive_actuator_force_n')
    for key in keys:
        values = muscles[key]
        require(len(values) == 416 and all(type(x) in (int, float) and math.isfinite(x) for x in values),
                'invalid muscle vector ' + key)
    for a, transported, fiber, force, passive in zip(*(muscles[k] for k in keys)):
        require(0 <= a <= 1 and struct.unpack('<f', struct.pack('<f', a))[0] == transported,
                'activation transport changed')
        require(fiber > 0 and force <= 0 and passive <= 0, 'invalid fibre or source force sign')
    reactions = module(HERE.parent / 'limit-reactions-20260908/verify_reactions.py')
    q, forces, _ = reactions.parse(log)
    manifest_bytes = reactions.MANIFEST.read_bytes()
    require(hashlib.sha256(manifest_bytes).hexdigest() == reactions.MANIFEST_SHA, 'source map drift')
    audit = reactions.audit(q, forces, json.loads(manifest_bytes))
    require(audit['status'] == 'offline_reactions_passed', str(audit['violations']))
    return {'residual': residual, 'accepted_pose_steps': poses,
            'rejected_constraint_candidates': trace[-1]['rejected_constraint_candidates'],
            'loaded_source_stops': audit['loaded_source_stops']}


def main():
    receipt = json.loads((HERE / 'receipt.json').read_text())
    require(receipt['schema'] == 'numi.human.coupled-equilibrium-receipt.v1', 'receipt schema')
    base = HERE.parents[2]
    for item in receipt['dependencies']:
        path = (base / item['path']).resolve()
        require(path.is_relative_to(base) and path.is_file(), 'dependency escaped repository')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'dependency drift: ' + item['path'])
    for item in receipt['artifacts']:
        path = (HERE / item['path']).resolve()
        require(path.is_relative_to(HERE) and path.is_file(), 'artifact escaped bundle')
        raw = path.read_bytes()
        require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], item['path'])
    log_bytes = gzip.decompress((HERE / 'clean-stdout.log.gz').read_bytes())
    digest = hashlib.sha256(log_bytes).hexdigest()
    require(digest == receipt['certificate_sha256'], 'certificate drift')
    result = audit_state(log_bytes.decode())
    launch = json.loads((HERE / 'clean-completion.json').read_text())
    require(launch['native_status'] == '' and launch['native_head'] == receipt['native_commit'] and
            launch['returncode'] == 0, 'native clean execution failed')
    require(launch['patch_sha256'] == hashlib.sha256(b'').hexdigest(), 'dirty source patch')
    require(launch['authority'] == 'offline_native_initialization_only', 'authority drift')
    require(launch['binary_sha256'] == launch['artifacts_sha256'][launch['command'][0]], 'binary identity drift')
    require(launch['command'][-2:] == ['--whole-body-pose-sweeps', '48'], 'pose budget drift')
    ctest = (HERE / 'native-clean-tests.log').read_text()
    require('100% tests passed out of 4' in ctest and len(re.findall(r'Test\s+#\d+:.*Passed', ctest)) == 4,
            'native regressions failed')
    source = json.loads((HERE / 'source-mass.json').read_text())
    geometry = json.loads((HERE / 'source-mass.geometry.json').read_text())
    for record in (source, geometry):
        require(record['certificate_sha256'] == digest and record['mujoco_version'] == '3.12.0', 'source oracle identity')
        require(record['source_archive_sha256'] == receipt['source_archive_sha256'], 'source archive identity')
    require(geometry['source_files_verified'] == 57 and all(geometry[k] is True for k in
            ('primitive_geometry_passed', 'lowering_passed', 'wrench_passed')), 'source geometry/wrench failed')
    mass = module(HERE / 'verify_source_mass.py')
    mass.audit_vectors(source['native_force_residual'], source['source_mass_times_native_acceleration'],
                       source['native_gravity'], source['source_gravity'])
    forces = json.loads(unique(log_bytes.decode(), 'compiled_equilibrium_reactions='))
    require(source['native_force_residual'] == forces['force_residual'] and
            source['native_gravity'] == forces['gravity_target'], 'mass oracle native inputs drift')
    for key in ('fp32_equilibrium', 'prepared_pose_loaded_limits', 'registered_tissue', 'standing',
                'walking', 'calibration', 'costal_timeout_resolved', 'performance', 'full_release'):
        require(receipt['qualification'][key] is False, 'unsupported promotion: ' + key)
    require(receipt['qualification']['offline_fp64_equilibrium'] is True, 'offline gate missing')
    exploration = json.loads(gzip.decompress((HERE / 'exploration.json.gz').read_bytes()))
    require(len(exploration) == 18 and sum(r['metadata']['returncode'] != 0 for r in exploration) == 3,
            'historical failed/exploratory runs omitted')
    print(json.dumps({'status': 'offline_fp64_equilibrium_passed', **result, 'full_release': False}))


if __name__ == '__main__':
    main()
