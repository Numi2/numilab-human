"""Check source-ordered active tension against the full FEM residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import cardiac_active_force_gate as reference
from . import cardiac_active_tension_ingress as ingress


def audit(asset: Path, activation: Path, candidate: Path) -> dict:
    asset, activation, candidate = (Path(p).resolve() for p in
                                    (asset, activation, candidate))
    config, _, _, force_summary, arrays = reference.source_inputs(
        asset, activation, ingress.REFERENCE)
    summary = reference.human.read_json(candidate / 'summary.json')
    reference.require(
        summary['schema'] == 'numi.human.cardiac-active-tension-ingress.v1'
        and summary['status'] == 'source_ordered_prescribed_active_tension_candidate'
        and summary['producer_source_sha256'] == reference.sha(Path(ingress.__file__))
        and summary['source_asset_manifest_sha256'] == reference.sha(asset / 'manifest.json')
        and summary['candidate_activation_summary_sha256'] == reference.sha(activation / 'summary.json')
        and summary['fixed_reference_summary_sha256'] == reference.sha(
            ingress.REFERENCE / 'summary.json')
        and summary['source_cell_count'] == 1470083
        and summary['accepted_native_anatomical_steps'] == 0
        and summary['heartbeat_qualified'] is False
        and [f['time_ms'] for f in summary['frames']] == [100, 250],
        'source-order ingress provenance and boundary')
    nodes = arrays['nodes.f64le']
    tetrahedra = arrays['tetrahedra.u32le']
    labels = arrays['labels.u32le']
    fibres = arrays['fibres.f64le']
    ventricular = (labels == 1) | (labels == 2)
    selected = np.flatnonzero(ventricular)
    reference.require(len(selected) == 1097534, 'ventricular source-cell count')
    source_nodes = arrays['ventricular-source-nodes.u32le'][:218077]
    output = []
    for frame, prior in zip(summary['frames'], force_summary['frames']):
        reference.require(frame['time_ms'] == prior['time_ms'] and
                          frame['reference_internal_residual_sha256'] == prior['sha256'],
                          'source force-frame identity')
        path = candidate / frame['file']
        reference.checked_file(path, frame['sha256'], 1470083 * 4)
        tension = np.fromfile(path, '<f4').astype(np.float64)
        reference.require(np.isfinite(tension).all() and
                          np.all((tension >= 0) &
                                 (tension <= config['source_parameters']['peak_isometric_tension_pa']))
                          and np.all(tension[~ventricular] == 0)
                          and int(np.count_nonzero(tension[ventricular])) ==
                          frame['active_ventricular_cells']
                          and float(tension.max()) == frame['maximum_f32_tension_pa'],
                          f'finite ventricular-only active tension at {frame["time_ms"]} ms')
        assembled = np.zeros_like(nodes)
        stress_volume = 0.0
        for start in range(0, len(selected), 25000):
            index = selected[start:start + 25000]
            ids = tetrahedra[index]
            p = nodes[ids]
            e1, e2, e3 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
            det = np.einsum('ij,ij->i', e1, np.cross(e2, e3))
            reference.require(np.isfinite(det).all() and np.all(det > 0),
                              'positive ventricular cell volume')
            volume = det / 6.0
            f = fibres[index]
            length = np.linalg.norm(f, axis=1)
            reference.require(np.isfinite(length).all() and np.all(length > 0),
                              'source fibre norm')
            f = f / length[:, None]
            g1 = np.cross(e2, e3) / det[:, None]
            g2 = np.cross(e3, e1) / det[:, None]
            g3 = np.cross(e1, e2) / det[:, None]
            grads = np.stack((-(g1 + g2 + g3), g1, g2, g3), axis=1)
            directional = np.einsum('nij,nj->ni', grads, f)
            contributions = ((volume * tension[index])[:, None, None] *
                             directional[:, :, None] * f[:, None, :])
            np.add.at(assembled, ids.ravel(), contributions.reshape(-1, 3))
            stress_volume += float(np.dot(volume, tension[index]))
        reference_path = ingress.REFERENCE / prior['file']
        reference.checked_file(reference_path, prior['sha256'], 218077 * 3 * 8)
        expected = np.fromfile(reference_path, '<f8').reshape(218077, 3)
        actual = assembled[source_nodes]
        error = actual - expected
        maximum = float(np.max(np.abs(error)))
        relative = float(np.linalg.norm(error) / np.linalg.norm(expected))
        net = actual.sum(axis=0)
        reference.require(maximum < 1e-8 and relative < 1e-6
                          and np.linalg.norm(net) < 1e-9
                          and abs(stress_volume -
                                  prior['native_metrics']['stress_volume_integral_j']) < 1e-5,
                          f'full-source active force parity at {frame["time_ms"]} ms')
        output.append({'time_ms': frame['time_ms'], 'tension_sha256': frame['sha256'],
                       'reference_residual_sha256': prior['sha256'],
                       'maximum_component_difference_n': maximum,
                       'relative_l2_difference': relative,
                       'net_internal_residual_norm_n': float(np.linalg.norm(net)),
                       'stress_volume_integral_j': stress_volume})
    return {'schema': 'numi.human.cardiac-active-tension-ingress-gate.v1',
            'status': 'passed_full_source_float32_tension_to_residual',
            'candidate_summary_sha256': reference.sha(candidate / 'summary.json'),
            'gate_source_sha256': reference.sha(Path(__file__)),
            'source_cells_checked_per_frame': 1470083,
            'ventricular_cells_assembled_per_frame': 1097534,
            'frames': output,
            'accepted_native_anatomical_steps': 0,
            'heartbeat_qualified': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--activation', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.asset, args.activation, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'frames': result['frames']}))


if __name__ == '__main__':
    main()
