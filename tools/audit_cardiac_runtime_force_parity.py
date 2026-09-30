"""Audit the initial native ventricular stress assembly against source force.

The native Metal capture is an opt-in copy from the first Newton assembly in
the accepted one-microsecond fixture. Active-minus-zero removes the passive
stress at the same candidate; neither field is a physiological heartbeat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from numilab_human import cardiac_split_active_force_gate as source_gate


ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path('/Users/home/numi-lab-cardiac-native-20260930')
BUILD = NATIVE / 'Build/cardiac-active-native'
EVIDENCE = ROOT / 'Build/cardiac-runtime-force-parity-20260930'
OUT = ROOT / 'Docs/media/cardiac-runtime-force-parity-20260930'
COOK = ROOT / 'Docs/media/cardiac-point-contact-split-20260930'
PACKAGE = ROOT / 'Build/cardiac-point-contact-split-20260930/ventricular.nmpkg'
TENSION = ROOT / 'Build/cardiac-point-contact-split-20260930/cooked-tension.f32le'
NODES = 218080
CELLS = 1097534
POINTS = (17565, 170947, 235754)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError('cardiac runtime force parity: ' + message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def native_run(directory: Path, name: str, zero: bool) -> dict:
    command = [str(BUILD / 'numi-matter-ventricular-source-step'),
               str(PACKAGE), str(TENSION), str(directory / f'{name}-nodes.bin')]
    if zero:
        command.append('--zero')
    command.extend(('--capture-elements', str(directory / f'{name}-elements.bin')))
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    require(not result.stderr and len(result.stdout.strip().splitlines()) == 1,
            f'{name} native receipt shape')
    return json.loads(result.stdout)


def audit(*, execute: bool = False) -> dict:
    tracked = subprocess.check_output(
        ['git', '-C', str(NATIVE), 'status', '--porcelain=v1',
         '--untracked-files=no'], text=True)
    require(not tracked, 'native owner has uncommitted tracked changes')
    source = source_gate.audit(source_gate.ASSET, source_gate.ACTIVATION,
                               source_gate.PRIOR, source_gate.COOKED,
                               source_gate.CANDIDATE)
    require(source['status'] == 'passed_full_source_split_mechanical_force_reference'
            and source['corrected_mechanical_nodes'] == NODES,
            'corrected full-source reference')
    cook = json.loads((COOK / 'receipt.json').read_text())
    require(sha(PACKAGE) == cook['cooked_package_sha256'] and
            sha(TENSION) == cook['files_sha256']['cooked-tension.f32le'],
            'exact accepted native package and source tension')
    paths = [EVIDENCE / f'{name}-{kind}.bin'
             for name in ('active', 'zero', 'replay')
             for kind in ('nodes', 'elements')]
    for path in paths:
        require(path.is_file(), f'missing {path.name}')
        if path.name.endswith('-elements.bin'):
            require(path.stat().st_size == CELLS*64,
                    f'element count in {path.name}')
    hashes = {path.name: sha(path) for path in paths}
    require(hashes['active-elements.bin'] == hashes['replay-elements.bin'] and
            hashes['active-nodes.bin'] == hashes['replay-nodes.bin'] and
            hashes['active-nodes.bin'] == cook['accepted_state_sha256']['active'] and
            hashes['zero-nodes.bin'] == cook['accepted_state_sha256']['zero'],
            'exact native force replay and unchanged accepted state')

    if execute:
        with tempfile.TemporaryDirectory(prefix='cardiac-runtime-force-') as folder:
            temp = Path(folder)
            receipts = [native_run(temp, name, name == 'zero')
                        for name in ('active', 'zero', 'replay')]
            require(all(row['status_code'] == 0 and
                        row['accepted_native_steps'] == 1 and
                        row['initial_element_force_capture'] is True and
                        row['heartbeat_qualified'] is False
                        for row in receipts), 'native capture runs accepted')
            require(all(sha(temp / name) == digest
                        for name, digest in hashes.items()),
                    'native rerun differs from retained force or accepted state')

    asset = source_gate.ASSET
    labels = np.fromfile(asset / 'labels.u32le', '<u4')
    source_cells = np.flatnonzero((labels == 1) | (labels == 2))
    require(len(source_cells) == CELLS, 'full ventricular cell count')
    tets = np.fromfile(asset / 'tetrahedra.u32le', '<u4').reshape(-1, 4)[source_cells]
    mapping = np.fromfile(COOK / 'cooked-source-nodes.u32le', '<u4')
    require(len(mapping) == NODES and tuple(mapping[-3:]) == POINTS,
            'split mechanical node map')
    lut = np.full(300965, -1, np.int32)
    lut[mapping[:NODES-3]] = np.arange(NODES-3, dtype=np.int32)
    cooked = lut[tets]
    rv = labels[source_cells] == 2
    for index, point in enumerate(POINTS):
        cooked[rv[:, None] & (tets == point)] = NODES-3+index
    require(np.all(cooked >= 0), 'complete source-to-cooked ownership')

    active = np.memmap(EVIDENCE / 'active-elements.bin', '<f4', 'r').reshape(CELLS, 4, 4)
    zero = np.memmap(EVIDENCE / 'zero-elements.bin', '<f4', 'r').reshape(CELLS, 4, 4)
    require(np.isfinite(active).all() and np.isfinite(zero).all() and
            np.all(active[:, :, 3] == 0) and np.all(zero[:, :, 3] == 0),
            'finite native element forces and zero padding')
    # Metal's element vector is internal force; the offline reference uses
    # positive weak-form internal residual, hence the minus sign.
    element_residual = zero[:, :, :3].astype(np.float64) - \
        active[:, :, :3].astype(np.float64)
    runtime = np.zeros((NODES, 3), np.float64)
    for corner in range(4):
        np.add.at(runtime, cooked[:, corner], element_residual[:, corner])
    reference_path = source_gate.CANDIDATE / 'split-residual-100ms.f64le'
    require(sha(reference_path) == source['frames'][0]['source_to_cooked_residual_sha256'],
            'pinned 100 ms offline residual')
    reference = np.fromfile(reference_path, '<f8').reshape(NODES, 3)
    difference = runtime - reference
    maximum = float(np.max(np.abs(difference)))
    relative = float(np.linalg.norm(difference) / np.linalg.norm(reference))
    require(maximum < 1e-6 and relative < 1e-5,
            'full-source native/internal residual mismatch')
    point_rows = []
    for index, point in enumerate(POINTS):
        lv = int(np.flatnonzero(mapping[:NODES-3] == point)[0])
        rv_node = NODES-3+index
        require(float(np.max(np.abs(difference[[lv, rv_node]]))) < 1e-6,
                'split point-contact native force mismatch')
        point_rows.append({'source_node': point, 'lv_node': lv, 'rv_node': rv_node,
                           'max_component_error_n': float(np.max(
                               np.abs(difference[[lv, rv_node]])))})
    OUT.mkdir(parents=True, exist_ok=True)
    field = OUT / 'native-internal-residual-100ms.f64le'
    runtime.astype('<f8').tofile(field)
    return {
        'schema': 'numi.human.cardiac-runtime-force-parity.v1',
        'status': 'passed_full_source_native_initial_force_parity',
        'source_force_gate_sha256': sha(source_gate.CANDIDATE / 'independent-gate.json'),
        'source_reference_residual_sha256': sha(reference_path),
        'point_contact_cook_receipt_sha256': sha(COOK / 'receipt.json'),
        'package_sha256': sha(PACKAGE), 'tension_sha256': sha(TENSION),
        'native_revision': subprocess.check_output(
            ['git', '-C', str(NATIVE), 'rev-parse', 'HEAD'], text=True).strip(),
        'native_runtime_source_sha256': sha(NATIVE / 'matter/src/runtime.mm'),
        'native_header_source_sha256': sha(NATIVE / 'matter/include/numi/matter/matter.hpp'),
        'native_shader_source_sha256': sha(NATIVE / 'matter/src/metal/fem.metalinc'),
        'native_step_source_sha256': sha(NATIVE / 'matter/tools/ventricular_source_step.mm'),
        'native_step_binary_sha256': sha(BUILD / 'numi-matter-ventricular-source-step'),
        'native_metallib_sha256': sha(BUILD / 'shaders/NumiMatter.metallib'),
        'auditor_source_sha256': sha(Path(__file__)),
        'retained_local_capture_sha256': hashes,
        'native_internal_residual_sha256': sha(field),
        'native_and_replay_byte_identical': True,
        'accepted_active_and_zero_state_unchanged': True,
        'source_ventricular_tetrahedra': CELLS,
        'cooked_mechanical_nodes': NODES,
        'max_component_error_n': maximum,
        'relative_l2_error': relative,
        'point_only_contacts': point_rows,
        'native_net_internal_residual_n': runtime.sum(axis=0).tolist(),
        'native_accepted_fixture_steps_per_run': 1,
        'source_activation_model_reproduced': False,
        'physiological_heartbeat_qualified': False,
        'boundary': 'First Newton assembly of one synthetic-support 1 us native FEM fixture; active-minus-zero internal stress at supplied CT geometry. No native electrical evolution, chamber loading, anatomical support, physiological motion or heartbeat.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='rerun active, zero and replay native captures')
    parser.add_argument('--output', type=Path, default=OUT / 'receipt.json')
    args = parser.parse_args()
    result = audit(execute=args.execute)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': result['status'],
                      'max_component_error_n': result['max_component_error_n'],
                      'relative_l2_error': result['relative_l2_error']}))


if __name__ == '__main__':
    main()
