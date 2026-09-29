"""Actual full-weight native decoding and additional articulated source views."""
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from numilab_human.skin_surface_audit import audit_skin_surface


@pytest.fixture(scope='module')
def full_root():
    path = os.environ.get('NUMILAB_HUMAN_NATIVE_SKIN_AUDIT_ROOT')
    if not path:
        pytest.skip('full skin checks require exact native corpus')
    return Path(path).resolve()


def source_inputs(root, case):
    repo = Path(__file__).resolve().parents[1]
    capture = root / ('native-' + case)
    assert (capture / 'exit.code').read_text().strip() == '0'
    pose = json.loads((capture / 'pose.json').read_text())
    return (repo / 'Sources', repo / 'Build/myosim-fullbody',
            repo / 'Build/knee-parity-registration-20260929/candidate.v6.registration.json',
            root / 'payload/bodyparts3d-myosim-skinned-shell.nhskin',
            next(capture.glob('*.mrvpack')), next(capture.glob('*.skin-poses.json')),
            None if pose is None else tuple(map(tuple, pose)))


@pytest.mark.parametrize('case', ['shoulder-elevation', 'hip-flexion',
                                  'unilateral-reach', 'asymmetric-knee'])
def test_full_field_reaches_additional_independent_native_poses(full_root, case):
    inputs = source_inputs(full_root, case)
    report = audit_skin_surface(*inputs)
    assert report['passed'] and report['runtime_influences_per_vertex'] == 86
    assert report['runtime_discarded_weight_mass'] == 0
    inputs[4].with_name('source-skin-audit.json').write_text(json.dumps(report, indent=2) + '\n')


@pytest.mark.parametrize('corruption', ['negative', 'nan', 'partition', 'truncated',
                                       'trailing', 'false_legacy', 'unknown_abi'])
def test_native_full_weight_reader_rejects_malformed_input_before_render(full_root, tmp_path, corruption):
    payload = full_root / 'payload/bodyparts3d-myosim-skinned-shell.nhskin'
    raw = bytearray(payload.read_bytes())
    _, abi, nb, nv, ni, *_ = struct.unpack_from('<8s5I32s', raw)
    assert abi == 5
    weights_offset = 60 + 36*nb + 56*nv + 4*ni
    if corruption in {'negative', 'nan', 'partition'}:
        row = np.frombuffer(raw, '<f4', count=nb, offset=weights_offset)
        if corruption == 'negative': row[0] = -.1
        elif corruption == 'nan': row[0] = float('nan')
        else: row[int(np.argmin(row))] += .02
        message = 'full weight partition is malformed' if corruption == 'partition' else 'full weight is malformed'
    elif corruption == 'truncated':
        raw = raw[:-4]
        message = 'full-weight skinned-shell byte count differs'
    elif corruption == 'trailing':
        raw += b'\0\0\0\0'
        message = 'full-weight skinned-shell byte count differs'
    else:
        struct.pack_into('<I', raw, 8, 4 if corruption == 'false_legacy' else 6)
        message = 'payload has trailing bytes' if corruption == 'false_legacy' else 'payload/header disagreement'
    changed = tmp_path / 'invalid.nhskin'
    changed.write_bytes(raw)
    command = json.loads((full_root / 'native-neutral/command.json').read_text())
    command[4] = str(tmp_path / 'views')
    command[command.index('--skin-payload') + 1] = str(changed)
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    (tmp_path / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    (tmp_path / 'stdout').write_text(result.stdout)
    (tmp_path / 'stderr').write_text(result.stderr)
    assert result.returncode != 0 and message in result.stderr
    assert not list((tmp_path / 'views').glob('*.mrvpack'))


@pytest.mark.parametrize('case', ['raw-source-rest', 'neutral', 'coupled-torso',
                                  'coupled-reach', 'knee-flexion'])
def test_archived_four_weight_source_evidence_remains_verifiable(case):
    path = os.environ.get('NUMILAB_HUMAN_LEGACY_SKIN_AUDIT_ROOT')
    if not path:
        pytest.skip('legacy source evidence requires retained corpus')
    report = audit_skin_surface(*source_inputs(Path(path).resolve(), case))
    assert report['passed'] and report['payload_abi'] == 4
    assert report['runtime_influences_per_vertex'] == 4
