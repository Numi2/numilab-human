#!/usr/bin/env python3
"""Audit NHINIT1 transport and bounded accepted-root receipts independently."""
import gzip
import base64
import hashlib
import json
import math
import re
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fnv(data, value=0xcbf29ce484222325):
    for byte in data:
        value = ((value ^ byte) * 0x100000001b3) & ((1 << 64) - 1)
    return value


def audit_state(payload, certificate, fixture):
    require(len(payload) == 7780, 'NHINIT1 byte count')
    header = struct.unpack_from('<8s8I3Q32s', payload)
    require(header[:9] == (b'NHINIT1\0', 1, 96, 129, 128, 416, 4, 0, 0), 'NHINIT1 ABI')
    human, world, clock, source = header[9:]
    require(human == int(fixture['composed_human_source_fp'], 16) and
            world == int(fixture['world_fp'], 16) and clock == 100, 'state source/world/clock drift')
    require(source.hex() == '280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975', 'source archive')
    require(fnv(payload) == int(fixture['initial_state_fp'], 16), 'state fingerprint drift')
    def record(prefix):
        rows = [json.loads(l[len(prefix):]) for l in certificate.splitlines() if l.startswith(prefix)]
        require(len(rows) == 1, 'missing/duplicate native state')
        return rows[0]
    q = record('compiled_equilibrium_q=')
    muscle = record('compiled_equilibrium_muscles=')
    expected = [*q, *([0] * 128)]
    for activation, fiber in zip(muscle['activation_fp32'], muscle['reference_fiber_length_m']):
        expected += [activation, activation, fiber, 0]
    require(payload[96:] == struct.pack('<1921f', *expected), 'prepared state differs from compiled stance')
    return {'q': list(struct.unpack_from('<129f', payload, 96)), 'muscles': 416}


def audit_trajectory(log):
    require('Executed 2 tests, with 0 failures' in log, 'prepared admission/publication test failed')
    tokens = re.findall(r'GateB physical-token words=([^\n]+)', log)
    require(len(tokens) == 9 and [int(t.split(',')[4], 16) for t in tokens] == [1,2,3,4,4,5,6,7,8],
            'physical acceptance/rejection sequence changed')
    require(tokens[3] == tokens[4], 'rejected candidate changed accepted physical identity')
    states = [json.loads(l.split('=', 1)[1]) for l in log.splitlines() if l.startswith('prepared_accepted_state=')]
    require(len(states) == 8, 'accepted state history incomplete')
    for generation, state in enumerate(states, 1):
        require(state['schema'] == 'numi.human.accepted-prepared-state.v1' and
                state['physics_generation'] == generation and state['elapsed_microseconds'] == generation * 100,
                'accepted clock/generation drift')
        require(state['nq'] == 129 and state['nv'] == 128, 'accepted dimensions drift')
        packed = base64.b64decode(state['q_v_fp32_le_base64'], validate=True)
        require(len(packed) == 4 * 257, 'accepted state packing invalid')
        values = struct.unpack('<257f', packed)
        state['q'], state['v'] = list(values[:129]), list(values[129:])
        for key, count in [('q', 129), ('v', 128)]:
            require(len(state[key]) == count and all(type(x) in (int, float) and math.isfinite(x) for x in state[key]),
                    'invalid accepted generalized state')
    return tokens, states


def main():
    receipt = json.loads((HERE / 'receipt.json').read_text())
    require(receipt['schema'] == 'numi.human.prepared-state-receipt.v1', 'receipt schema')
    for item in receipt['artifacts']:
        path = (HERE / item['path']).resolve()
        require(path.is_relative_to(HERE) and path.is_file(), 'artifact outside bundle')
        raw = path.read_bytes()
        require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'], item['path'])
    certificate_raw = gzip.decompress((HERE.parent / 'coupled-equilibrium-20260908/clean-stdout.log.gz').read_bytes())
    require(hashlib.sha256(certificate_raw).hexdigest() == receipt['offline_certificate_sha256'], 'offline source certificate drift')
    fixture = json.loads((HERE / 'fixture/prepared.json').read_text())
    initial = audit_state((HERE / 'fixture/prepared.nhinit').read_bytes(), certificate_raw.decode(), fixture)
    runs = []
    for label in ['prepared-clean', 'prepared-replay']:
        launch = json.loads((HERE / (label + '-launch.json')).read_text())
        require(launch['returncode'] == 0, 'joint transaction failed')
        for name in ['native', 'brain']:
            require(launch['source_state'][name]['status'] == '' and
                    launch['source_state'][name]['revision'] == receipt[name + '_commit'], 'source revision/status drift')
        runs.append(audit_trajectory(gzip.decompress((HERE / (label + '.log.gz')).read_bytes()).decode()))
    require(runs[0] == runs[1], 'prepared physical/state replay differs')
    require(json.loads((HERE / 'accepted-states.json').read_text()) == runs[0][1], 'decoded states drift')
    replay = json.loads((HERE / 'fixture-replay.json').read_text())
    require(replay['status'] == 'bitwise' and replay['native_commit'] == receipt['native_commit'], 'fixture replay failed')
    for name, digest in replay['files'].items():
        require(hashlib.sha256((HERE / 'fixture' / name).read_bytes()).hexdigest() == digest, 'cooked fixture drift')
    ctest = (HERE / 'native-clean-tests.log').read_text()
    require('100% tests passed out of 6' in ctest, 'native regression coverage')
    detail = (HERE / 'native-clean-test-details.log').read_text()
    require('rows=18 receptors=10 impulse=171 centre_of_pressure=conserved replay=bitwise malformed_rows=4' in detail,
            'GPU aggregation conformance missing')
    legacy = gzip.decompress((HERE / 'legacy-clean.log.gz').read_bytes()).decode()
    require('Executed 2 tests, with 0 failures' in legacy, 'legacy source runtime regression')
    geometry = json.loads((HERE / 'terminal-source-geometry.json').read_text())
    require(geometry['primitive_geometry_passed'] is False and geometry['lowering_passed'] is True,
            'terminal geometry failure omitted or promoted')
    require(geometry['native_state_sha256'] == hashlib.sha256((HERE / 'terminal-state.log').read_bytes()).hexdigest(),
            'terminal geometry state mismatch')
    terminal = json.loads((HERE / 'terminal-state.log').read_text().split('=', 1)[1])
    require(terminal['q'] == runs[0][1][-1]['q'] and terminal['v'] == runs[0][1][-1]['v'], 'terminal state mismatch')
    for key in ['loaded_equilibrium', 'registered_anatomical_tissue', 'standing', 'walking',
                'calibration', 'costal_timeout_resolved', 'performance', 'full_release']:
        require(receipt['qualification'][key] is False, 'unsupported promotion: ' + key)
    print(json.dumps({'status': 'bounded_prepared_state_passed', 'accepted_roots': 8, 'elapsed_seconds': 0.0008,
                      'muscles': initial['muscles'], 'replay': 'exact', 'full_release': False}))


if __name__ == '__main__':
    main()
