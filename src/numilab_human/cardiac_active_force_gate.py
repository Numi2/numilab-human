"""Independent full-source check of ventricular active-stress internal residuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import model as human


ROOT = human.REPOSITORY_ROOT
SOURCE_CPP = ROOT / 'tools/cardiac_active_force_reference.cpp'
CONFIG = ROOT / 'config/cardiac-rodero18-active-force-reference.v1.json'
SOURCE_CONFIG = ROOT / 'config/cardiac-wall-rodero18.v1.json'
SUPPLEMENT = ROOT / 'Docs/media/cardiac-source-activation-20260930/rodero-s4-source.pdf'
H = np.array([.01, -.004, -.006], np.float64)


def require(value: bool, message: str) -> None:
    if not value:
        raise human.ImportError('cardiac active-force independent gate: ' + message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path: Path, expected: str, size: int | None = None) -> None:
    require(path.is_file() and not path.is_symlink()
            and (size is None or path.stat().st_size == size)
            and sha(path) == expected, f'hash/size mismatch for {path.name}')


def source_inputs(asset: Path, activation: Path, candidate: Path):
    config = human.read_json(CONFIG)
    manifest_path, activation_path = asset/'manifest.json', activation/'summary.json'
    summary_path = candidate/'summary.json'
    manifest, earlier, summary = (human.read_json(p) for p in
                                  (manifest_path, activation_path, summary_path))
    require(config['schema'] == 'HumanPack.cardiac-rodero18-active-force-reference.v1'
            and config['anatomy_manifest_sha256'] == sha(manifest_path)
            and config['candidate_activation_summary_sha256'] == sha(activation_path)
            and summary['asset_manifest_sha256'] == sha(manifest_path)
            and summary['candidate_activation_summary_sha256'] == sha(activation_path)
            and summary['reference_config_sha256'] == sha(CONFIG)
            and summary['source_config_sha256'] == sha(SOURCE_CONFIG)
            and summary['source_supplement_sha256'] == sha(SUPPLEMENT)
            and summary['source_cpp_sha256'] == sha(SOURCE_CPP)
            and summary['native_binary_sha256'] == sha(candidate/'cardiac-active-internal-residual')
            and summary['status'] == 'candidate_fixed_reference_active_internal_residual'
            and summary['production_native_electromechanical_steps'] == 0
            and summary['heartbeat_qualified'] is False,
            'source-to-candidate provenance and claim boundary')
    arrays = {}
    source_fields = {
        'nodes.f64le': ('<f8', (300965, 3)),
        'tetrahedra.u32le': ('<u4', (1470083, 4)),
        'labels.u32le': ('<u4', (1470083,)),
        'fibres.f64le': ('<f8', (1470083, 3)),
    }
    for name, (kind, shape) in source_fields.items():
        row = manifest['buffers'][name]
        path = asset/name
        checked_file(path, row['sha256'], row['bytes'])
        arrays[name] = np.fromfile(path, kind).reshape(shape)
    activation_fields = {
        'ventricular-source-nodes.u32le': ('<u4', (218080,)),
        'ventricular-dof-regions.u32le': ('<u4', (218080,)),
        'refined-arrival.f64le': ('<f8', (218080,)),
    }
    for name, (kind, shape) in activation_fields.items():
        path = activation/name
        checked_file(path, earlier['output_sha256'][name])
        arrays[name] = np.fromfile(path, kind).reshape(shape)
    source_nodes = arrays['ventricular-source-nodes.u32le']
    regions = arrays['ventricular-dof-regions.u32le']
    require(np.all(regions[:218077] == 0) and np.all(regions[218077:] == 2)
            and np.all(np.diff(source_nodes[:218077].astype(np.int64)) > 0)
            and source_nodes[218077:].tolist() == [17565, 170947, 235754]
            and np.isfinite(arrays['refined-arrival.f64le']).all()
            and np.all(arrays['refined-arrival.f64le'] >= 0)
            and earlier['lv_rv_point_only_shared_source_nodes'] == [17565, 170947, 235754],
            'mechanical source IDs and electrically split point contacts')
    return config, manifest, earlier, summary, arrays


def independently_assemble(arrays: dict, frame_ms: int, parameters: dict,
                           chunk_size: int = 25000) -> tuple[np.ndarray, dict]:
    nodes = arrays['nodes.f64le']
    tets = arrays['tetrahedra.u32le']
    labels = arrays['labels.u32le']
    fibres = arrays['fibres.f64le']
    source_nodes = arrays['ventricular-source-nodes.u32le']
    arrivals_ms = arrays['refined-arrival.f64le'] * 1000
    lut = np.full(len(nodes), -1, np.int32)
    lut[source_nodes[:218077]] = np.arange(218077, dtype=np.int32)
    rv_lut = lut.copy()
    rv_lut[source_nodes[218077:]] = np.arange(218077, 218080, dtype=np.int32)
    selected = np.flatnonzero((labels == 1) | (labels == 2))
    require(len(selected) == 1097534, 'ventricular source-cell count')
    residual = np.zeros_like(nodes)
    volume_sum = 0.0
    stress_volume = 0.0
    analytic_work = 0.0
    maximum_tension = 0.0
    active_cells = 0
    qa = (1 + 3 / math.sqrt(5)) / 4
    qb = (1 - qa) / 3
    peak = parameters['peak_isometric_tension_pa']
    delay = parameters['electromechanical_delay_ms']
    tc = parameters['contraction_time_constant_ms']
    tr = parameters['relaxation_time_constant_ms']
    duration = parameters['transient_duration_ms']
    for start in range(0, len(selected), chunk_size):
        index = selected[start:start+chunk_size]
        ids = tets[index]
        require(np.all(ids < len(nodes)), 'source cell node range')
        cell_lut = lut[ids].copy()
        rv = labels[index] == 2
        cell_lut[rv] = rv_lut[ids[rv]]
        require(np.all(cell_lut >= 0), 'ventricular node activation coverage')
        p = nodes[ids]
        e1, e2, e3 = p[:, 1]-p[:, 0], p[:, 2]-p[:, 0], p[:, 3]-p[:, 0]
        det = np.einsum('ij,ij->i', e1, np.cross(e2, e3))
        require(np.isfinite(det).all() and np.all(det > 0), 'positive ventricular volume')
        volume = det / 6
        f = fibres[index]
        f_norm = np.linalg.norm(f, axis=1)
        require(np.isfinite(f_norm).all() and np.all(f_norm > 0), 'valid source fibres')
        f = f/f_norm[:, None]
        activation = arrivals_ms[cell_lut]
        # Four barycentric points: each corner receives the major weight once.
        quad_time = qb*activation.sum(axis=1)[:, None] + (qa-qb)*activation
        elapsed = frame_ms - quad_time - delay
        active = (elapsed > 0) & (elapsed < duration)
        tension = np.zeros_like(elapsed)
        rise = np.tanh(elapsed[active]/tc)
        fall = np.tanh((duration-elapsed[active])/tr)
        tension[active] = peak * rise**2 * fall**2
        stress = tension.mean(axis=1)
        g1 = np.cross(e2, e3)/det[:, None]
        g2 = np.cross(e3, e1)/det[:, None]
        g3 = np.cross(e1, e2)/det[:, None]
        grads = np.stack((-(g1+g2+g3), g1, g2, g3), axis=1)
        directional = np.einsum('nij,nj->ni', grads, f)
        contributions = (volume*stress)[:, None, None] * directional[:, :, None] * f[:, None, :]
        np.add.at(residual, ids.ravel(), contributions.reshape(-1, 3))
        volume_sum += float(volume.sum())
        stress_volume += float(np.dot(volume, stress))
        analytic_work += float(np.dot(volume*stress, (f*f) @ H))
        maximum_tension = max(maximum_tension, float(stress.max()))
        active_cells += int(np.count_nonzero(stress > 0))
    measured = {'ventricular_cells': len(selected), 'active_cells': active_cells,
                'volume_m3': volume_sum, 'stress_volume_integral_j': stress_volume,
                'max_tension_pa': maximum_tension,
                'analytic_affine_virtual_work_j': analytic_work}
    return residual[source_nodes[:218077]], measured


def audit(asset: Path, activation: Path, candidate: Path) -> dict:
    asset, activation, candidate = (Path(p).resolve() for p in (asset, activation, candidate))
    config, manifest, earlier, summary, arrays = source_inputs(asset, activation, candidate)
    require(config['virtual_work_affine_gradient_diagonal'] == H.tolist()
            and [row['time_ms'] for row in summary['frames']] == [100, 250]
            and all(row['replay_bitwise'] is True for row in summary['frames']),
            'frame/virtual-work contract')
    source_nodes = arrays['ventricular-source-nodes.u32le'][:218077]
    positions = arrays['nodes.f64le'][source_nodes]
    output_rows = []
    for frame in summary['frames']:
        path = candidate / frame['file']
        checked_file(path, frame['sha256'], 218077*3*8)
        native = np.fromfile(path, '<f8').reshape(218077, 3)
        require(np.isfinite(native).all(), 'finite native residual')
        independent, measured = independently_assemble(
            arrays, frame['time_ms'], config['source_parameters'])
        delta = native-independent
        max_error = float(np.max(np.abs(delta)))
        relative_l2 = float(np.linalg.norm(delta)/np.linalg.norm(independent))
        net_force = native.sum(axis=0)
        net_torque = np.cross(positions, native).sum(axis=0)
        nodal_work = float(np.sum(native * (positions*H)))
        native_metrics = frame['native_metrics']
        require(max_error < 1e-9 and relative_l2 < 1e-10
                and np.linalg.norm(net_force) < 1e-9
                and np.linalg.norm(net_torque) < 1e-9
                and abs(nodal_work-measured['analytic_affine_virtual_work_j']) < 1e-9
                and abs(native_metrics['volume_m3']-measured['volume_m3']) < 1e-12
                and abs(native_metrics['stress_volume_integral_j']-
                        measured['stress_volume_integral_j']) < 1e-9
                and abs(native_metrics['max_tension_pa']-measured['max_tension_pa']) < 1e-8
                and native_metrics['active_cells'] == measured['active_cells'],
                f'full-source force, balance, stress or virtual work at {frame["time_ms"]} ms')
        output_rows.append({
            'time_ms': frame['time_ms'], 'native_file_sha256': frame['sha256'],
            'independent': measured, 'max_component_discrepancy_n': max_error,
            'relative_l2_discrepancy': relative_l2,
            'net_internal_residual_n': net_force.tolist(),
            'net_internal_residual_norm_n': float(np.linalg.norm(net_force)),
            'net_internal_torque_nm': net_torque.tolist(),
            'net_internal_torque_norm_nm': float(np.linalg.norm(net_torque)),
            'nodal_affine_virtual_work_j': nodal_work,
            'virtual_work_discrepancy_j': nodal_work-measured['analytic_affine_virtual_work_j'],
            'sum_nodal_residual_magnitudes_n': float(np.linalg.norm(native, axis=1).sum()),
            'largest_nodal_residual_n': float(np.linalg.norm(native, axis=1).max()),
        })
    return {
        'schema': 'numi.human.cardiac-active-force-independent-gate.v1',
        'status': 'passed_full_source_reference_residual_and_work_gate',
        'candidate_summary_sha256': sha(candidate/'summary.json'),
        'asset_manifest_sha256': sha(asset/'manifest.json'),
        'candidate_activation_summary_sha256': sha(activation/'summary.json'),
        'gate_source_sha256': sha(Path(__file__)),
        'ventricular_cells_checked_per_frame': 1097534,
        'mechanical_source_nodes_checked_per_frame': 218077,
        'frames': output_rows,
        'accepted_native_matter_steps': 0,
        'heartbeat_qualified': False,
        'boundary': ('Independent NumPy full-source fixed-reference internal-residual check. '
                     'This does not supply stress-free reference, deformation, accepted-state '
                     'coupling, pressure/flow, or physiological cardiac motion.'),
    }


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
