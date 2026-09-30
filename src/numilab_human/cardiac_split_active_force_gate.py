"""Audit the corrected ventricular active-force reference in cooked node order.

This transfers only the three point-contact RV force contributions from the
pinned full-source reference. It is an offline force check, not a heartbeat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from . import cardiac_active_force_gate as prior_gate


ROOT = prior_gate.ROOT
ASSET = ROOT / 'Build/cardiac-electrical-source-20260930/asset'
ACTIVATION = ROOT / 'Docs/media/cardiac-source-activation-20260930'
PRIOR = ROOT / 'Docs/media/cardiac-active-force-reference-20260930'
COOKED = ROOT / 'Docs/media/cardiac-point-contact-split-20260930'
CANDIDATE = ROOT / 'Docs/media/cardiac-split-active-force-20260930'
PRODUCER = ROOT / 'tools/cardiac_split_active_force_reference.cpp'
POINT_ONLY = (17565, 170947, 235754)


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError('split active force gate: ' + message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _point_contribution(arrays: dict, cell: int, source_node: int,
                        time_ms: int, parameters: dict) -> np.ndarray:
    ids = arrays['tetrahedra.u32le'][cell]
    positions = arrays['nodes.f64le'][ids]
    corner = np.flatnonzero(ids == source_node)
    require(len(corner) == 1, 'point-only corner multiplicity')
    index = int(corner[0])
    e1, e2, e3 = positions[1:] - positions[0]
    det = float(np.dot(e1, np.cross(e2, e3)))
    require(math.isfinite(det) and det > 0, 'positive RV source tetrahedron')
    fibre = arrays['fibres.f64le'][cell]
    norm = float(np.linalg.norm(fibre))
    require(math.isfinite(norm) and norm > 0, 'source fibre')
    fibre = fibre / norm
    source_nodes = arrays['ventricular-source-nodes.u32le']
    base = np.searchsorted(source_nodes[:218077], ids)
    require(np.array_equal(source_nodes[base], ids), 'point-cell electrical IDs')
    base[index] = 218077 + POINT_ONLY.index(source_node)
    time = 1000*arrays['refined-arrival.f64le'][base]
    qa = (1+3/math.sqrt(5))/4
    qb = (1-qa)/3
    arrival = qb*time.sum()+(qa-qb)*time
    elapsed = time_ms-arrival-parameters['electromechanical_delay_ms']
    active = (elapsed > 0) & (elapsed < parameters['transient_duration_ms'])
    tension = np.zeros(4, np.float64)
    rise = np.tanh(elapsed[active]/parameters['contraction_time_constant_ms'])
    fall = np.tanh((parameters['transient_duration_ms']-elapsed[active]) /
                   parameters['relaxation_time_constant_ms'])
    tension[active] = parameters['peak_isometric_tension_pa']*rise**2*fall**2
    stress = float(tension.mean())
    g1 = np.cross(e2,e3)/det
    g2 = np.cross(e3,e1)/det
    g3 = np.cross(e1,e2)/det
    gradients = np.stack((-(g1+g2+g3), g1, g2, g3))
    return det/6*stress*float(np.dot(fibre, gradients[index]))*fibre


def audit(asset: Path, activation: Path, prior: Path, cooked: Path,
          candidate: Path, *, execute: bool = False) -> dict:
    asset, activation, prior, cooked, candidate = (
        Path(path).resolve() for path in
        (asset, activation, prior, cooked, candidate))
    previous = prior_gate.audit(asset, activation, prior)
    require(previous['status'] == 'passed_full_source_reference_residual_and_work_gate'
            and previous['mechanical_source_nodes_checked_per_frame'] == 218077
            and previous['heartbeat_qualified'] is False,
            'full-source prior force reference')
    config, manifest, activation_summary, prior_summary, arrays = (
        prior_gate.source_inputs(asset, activation, prior))
    point_receipt = json.loads((cooked / 'receipt.json').read_text())
    mapping_path = cooked / 'cooked-source-nodes.u32le'
    mapping = np.fromfile(mapping_path, '<u4')
    source_nodes = arrays['ventricular-source-nodes.u32le']
    selected = ((arrays['labels.u32le'] == 1) |
                (arrays['labels.u32le'] == 2))
    unique, first = np.unique(
        arrays['tetrahedra.u32le'][selected].ravel(), return_index=True)
    require(point_receipt['status'] ==
            'full_source_native_point_contacts_split_and_step_accepted'
            and point_receipt['native_bitwise_replay'] is True
            and point_receipt['heartbeat_qualified'] is False
            and point_receipt['files_sha256'][mapping_path.name] == sha(mapping_path)
            and len(mapping) == 218080
            and np.array_equal(mapping[:218077], unique[np.argsort(first)])
            and np.array_equal(mapping[-3:], POINT_ONLY)
            and np.array_equal(source_nodes[:218077], unique),
            'native cooked node order and corrected source identities')
    source_to_cooked = np.full(len(arrays['nodes.f64le']), -1, np.int32)
    source_to_cooked[mapping[:218077]] = np.arange(218077, dtype=np.int32)
    parameters = config['source_parameters']
    binary = candidate / 'split-active-force'
    require(binary.is_file() and not binary.is_symlink(), 'native producer binary')
    frame_results = []
    for frame in prior_summary['frames']:
        time_ms = frame['time_ms']
        old_path = prior / frame['file']
        require(sha(old_path) == frame['sha256'], 'old complete-source force hash')
        old = np.fromfile(old_path, '<f8').reshape(218077, 3)
        expected = old[np.searchsorted(source_nodes[:218077], mapping[:218077])]
        require(np.array_equal(source_nodes[np.searchsorted(
            source_nodes[:218077], mapping[:218077])], mapping[:218077]),
            'source-ordered reference remap')
        expected = np.concatenate([expected, np.zeros((3,3), np.float64)])
        transfers = []
        labels = arrays['labels.u32le']
        tets = arrays['tetrahedra.u32le']
        rv_cells = tets[labels == 2]
        rv_indices = np.flatnonzero(labels == 2)
        for offset, source_node in enumerate(POINT_ONLY):
            incident = np.flatnonzero((rv_cells == source_node).any(axis=1))
            require(len(incident) == 1, 'exact one-cell RV point-only incidence')
            contribution = _point_contribution(
                arrays, int(rv_indices[incident[0]]), source_node,
                time_ms, parameters)
            left = source_to_cooked[source_node]
            right = 218077+offset
            require(left >= 0, 'LV point-only source owner')
            expected[left] -= contribution
            expected[right] += contribution
            transfers.append(float(np.linalg.norm(contribution)))
        path = candidate / f'split-residual-{time_ms}ms.f64le'
        replay_path = candidate / f'split-residual-{time_ms}ms-replay.f64le'
        native = np.fromfile(path, '<f8').reshape(218080, 3)
        native_receipt_path = candidate / f'native-{time_ms}ms.json'
        replay_receipt_path = candidate / f'native-{time_ms}ms-replay.json'
        native_receipt = json.loads(native_receipt_path.read_text())
        require(sha(path) == sha(replay_path) and
                native_receipt_path.read_bytes() == replay_receipt_path.read_bytes()
                and native_receipt['status'] ==
                'source_active_force_point_contact_split'
                and native_receipt['frame_ms'] == time_ms
                and native_receipt['ventricular_nodes'] == 218080
                and native_receipt['rv_point_only_cells'] == 3
                and native_receipt['native_accepted_steps'] == 0
                and native_receipt['heartbeat_qualified'] is False,
                'native frame and byte-identical replay')
        max_error = float(np.max(np.abs(native-expected)))
        norms_error = float(np.max(np.abs(
            np.asarray(native_receipt['rv_transfer_force_norm_n'])-transfers)))
        positions = arrays['nodes.f64le'][mapping]
        old_positions = arrays['nodes.f64le'][source_nodes[:218077]]
        net = native.sum(axis=0)
        torque = np.cross(positions, native).sum(axis=0)
        old_net = old.sum(axis=0)
        old_torque = np.cross(old_positions, old).sum(axis=0)
        new_work = float(np.sum(native*(positions*prior_gate.H)))
        old_work = float(np.sum(old*(old_positions*prior_gate.H)))
        require(max_error < 1e-12 and norms_error < 1e-12
                and all(value > 0 for value in transfers)
                and np.linalg.norm(net-old_net) < 1e-10
                and np.linalg.norm(torque-old_torque) < 1e-10
                and abs(new_work-old_work) < 1e-10,
                'independent point transfer, force, torque and work closure')
        if execute:
            with tempfile.TemporaryDirectory(prefix='split-active-force-') as tmp:
                target = Path(tmp) / 'rerun.f64le'
                command = [str(binary), str(asset), str(activation), str(old_path),
                           str(mapping_path), str(target), str(time_ms),
                           *(str(parameters[name]) for name in (
                               'peak_isometric_tension_pa',
                               'electromechanical_delay_ms',
                               'contraction_time_constant_ms',
                               'relaxation_time_constant_ms',
                               'transient_duration_ms'))]
                rerun = subprocess.run(command, check=True, text=True,
                                       capture_output=True)
                require(not rerun.stderr and
                        rerun.stdout == native_receipt_path.read_text() and
                        sha(target) == sha(path),
                        'exact native force reproduction')
        frame_results.append({
            'time_ms': time_ms,
            'source_to_cooked_residual_sha256': sha(path),
            'native_receipt_sha256': sha(native_receipt_path),
            'max_independent_component_error_n': max_error,
            'rv_point_transfer_force_norm_n': transfers,
            'net_force_n': net.tolist(),
            'net_torque_nm': torque.tolist(),
            'affine_virtual_work_j': new_work,
            'old_merged_virtual_work_j': old_work,
            'replay_bitwise': True,
        })
    return {
        'schema': 'numi.human.cardiac-split-active-force-gate.v1',
        'status': 'passed_full_source_split_mechanical_force_reference',
        'asset_manifest_sha256': sha(asset/'manifest.json'),
        'activation_summary_sha256': sha(activation/'summary.json'),
        'prior_force_summary_sha256': sha(prior/'summary.json'),
        'prior_force_gate_sha256': sha(prior/'independent-gate.json'),
        'corrected_cook_receipt_sha256': sha(cooked/'receipt.json'),
        'corrected_cooked_source_map_sha256': sha(mapping_path),
        'producer_source_sha256': sha(PRODUCER),
        'producer_binary_sha256': sha(binary),
        'gate_source_sha256': sha(Path(__file__)),
        'ventricular_tetrahedra': 1097534,
        'corrected_mechanical_nodes': 218080,
        'frames': frame_results,
        'native_accepted_steps_with_this_residual': 0,
        'source_activation_model_reproduced': False,
        'heartbeat_qualified': False,
        'boundary': ('Offline source F=I active-stress internal residual in corrected '
                     'cooked node order. The 100-ms source tension was separately '
                     'used by the bounded native mechanics fixture. This does not '
                     'prove runtime force parity, physiological motion, or heartbeat.'),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, default=ASSET)
    parser.add_argument('--activation', type=Path, default=ACTIVATION)
    parser.add_argument('--prior', type=Path, default=PRIOR)
    parser.add_argument('--cooked', type=Path, default=COOKED)
    parser.add_argument('--candidate', type=Path, default=CANDIDATE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = audit(args.asset, args.activation, args.prior, args.cooked,
                   args.candidate, execute=args.execute)
    encoded = json.dumps(result, sort_keys=True, indent=2)+'\n'
    if args.output.exists():
        require(args.output.read_text() == encoded, 'retained gate drift')
    else:
        args.output.write_text(encoded)
    print(json.dumps({'status': result['status'],
                      'corrected_mechanical_nodes': result['corrected_mechanical_nodes'],
                      'max_component_error_n': max(row[
                          'max_independent_component_error_n'] for row in result['frames'])}))


if __name__ == '__main__':
    main()
