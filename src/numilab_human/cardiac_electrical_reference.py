"""Source-bound passive electrical geometry reference on the real heart mesh.

Electrical degrees of freedom are keyed by (myocardial region, source node).
Unit-direction diffusion and a dimensionless probe verify conservation and
energy dissipation. This is not an ionic model, activation map or heartbeat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import model as human
from .whole_body_embeddedness import atomic_json


ROOT = human.REPOSITORY_ROOT
SCHEMA = 'numi.human.cardiac-passive-electrical-reference.v1'
MYOCARDIUM = (1, 2, 3, 4)
FIELDS = {
    'nodes.f64le': ('<f8', (-1, 3)),
    'tetrahedra.u32le': ('<u4', (-1, 4)),
    'labels.u32le': ('<u4', (-1,)),
    'fibres.f64le': ('<f8', (-1, 3)),
    'sheets.f64le': ('<f8', (-1, 3)),
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise human.ImportError('cardiac electrical reference: '+message)


def _immutable(path: Path, data: bytes) -> None:
    require(not path.is_symlink(), f'output symlink {path.name}')
    if path.exists():
        require(path.read_bytes() == data, f'output identity changed {path.name}')
        return
    temp = path.with_name(path.name+f'.{os.getpid()}.pending')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _load(asset: Path):
    manifest_path = asset/'manifest.json'
    manifest = human.read_json(manifest_path)
    config = ROOT/'config/cardiac-wall-rodero18.v1.json'
    require(manifest['schema'] == 'HumanPack.cardiac-wall-source-asset.v1'
            and manifest['source_config_sha256'] == human.sha256(config)
            and {row['id'] for row in manifest['source_config']['labels']
                 if row['role'] == 'myocardium'} == set(MYOCARDIUM)
            and manifest['qualification']['physical_steps'] == 0,
            'pinned anatomical source and myocardial roles')
    arrays = {}
    for name, (kind, shape) in FIELDS.items():
        path = asset/name
        entry = manifest['buffers'][name]
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == entry['bytes']
                and human.sha256(path) == entry['sha256'],
                f'source buffer identity {name}')
        arrays[name] = np.fromfile(path, dtype=kind).reshape(shape)
    nodes = arrays['nodes.f64le']
    tetrahedra = arrays['tetrahedra.u32le']
    labels = arrays['labels.u32le']
    require(nodes.shape == (300965, 3)
            and tetrahedra.shape == (1470083, 4)
            and labels.shape == (len(tetrahedra),)
            and arrays['fibres.f64le'].shape == (len(tetrahedra), 3)
            and arrays['sheets.f64le'].shape == (len(tetrahedra), 3)
            and bool(np.isfinite(nodes).all())
            and int(tetrahedra.max()) < len(nodes),
            'anatomical source array shape')
    return manifest_path, manifest, arrays


def _gradients(points: np.ndarray, cells: np.ndarray):
    xyz = points[cells]
    edges = xyz[:, 1:] - xyz[:, 0, None, :]
    determinant = np.linalg.det(edges)
    require(bool(np.isfinite(determinant).all() and (determinant > 0).all()),
            'positive source tetrahedra')
    inverse = np.linalg.inv(edges)
    gradients = np.empty((len(cells), 4, 3), dtype=np.float64)
    gradients[:, 1:] = inverse.transpose(0, 2, 1)
    gradients[:, 0] = -gradients[:, 1:].sum(axis=1)
    scale = np.max(np.abs(gradients), axis=(1, 2))
    closure = np.max(np.abs(gradients.sum(axis=1)), axis=1)
    require(bool(np.all(closure <= 1e-10*scale)),
            'tetrahedral shape-function partition gradient')
    return gradients, determinant/6, float(np.max(closure/scale))


def _frame(fibres: np.ndarray, sheets: np.ndarray):
    fnorm = np.linalg.norm(fibres, axis=1)
    snorm = np.linalg.norm(sheets, axis=1)
    require(bool(np.isfinite(fnorm).all() and np.isfinite(snorm).all()
                 and (fnorm > 0).all() and (snorm > 0).all()),
            'finite nonzero source fibre/sheet fields')
    f = fibres/fnorm[:, None]
    s_raw = sheets - np.sum(sheets*f, axis=1)[:, None]*f
    s_norm = np.linalg.norm(s_raw, axis=1)
    require(bool((s_norm > 1e-8).all()), 'parallel source fibre/sheet fields')
    s = s_raw/s_norm[:, None]
    n = np.cross(f, s)
    return f, s, n, {
        'maximum_source_fibre_norm_error': float(np.max(np.abs(fnorm-1))),
        'maximum_source_sheet_norm_error': float(np.max(np.abs(snorm-1))),
        'maximum_source_normalized_fibre_sheet_dot':
            float(np.max(np.abs(np.sum(f*(sheets/snorm[:, None]), axis=1)))),
        'derived_orthonormal_frame': True,
        'source_vectors_modified': False,
    }


def _operator(gradients: np.ndarray, volumes: np.ndarray,
              cells: np.ndarray, dof_count: int, field: np.ndarray):
    field_gradient = np.einsum('mij,mi->mj', gradients, field[cells])
    local = volumes[:, None]*np.einsum('mij,mj->mi', gradients, field_gradient)
    residual = np.zeros(dof_count, dtype=np.float64)
    np.add.at(residual, cells.ravel(), local.ravel())
    energy = float(np.sum(volumes*np.sum(field_gradient**2, axis=1))/2)
    return residual, energy, field_gradient


def _region(region: int, nodes: np.ndarray, all_cells: np.ndarray,
            all_labels: np.ndarray, fibres: np.ndarray, sheets: np.ndarray,
            expected_count: int, expected_volume: float):
    selected = all_labels == region
    source_cells = all_cells[selected]
    source_nodes = np.unique(source_cells)
    cells = np.searchsorted(source_nodes, source_cells)
    points = nodes[source_nodes]
    require(len(source_cells) == expected_count
            and np.array_equal(source_nodes[cells], source_cells),
            f'region {region} complete source node mapping')
    gradients, volumes, closure = _gradients(points, cells)
    capacity = np.zeros(len(source_nodes), dtype=np.float64)
    np.add.at(capacity, cells.ravel(), np.repeat(volumes/4, 4))
    require(bool((capacity > 0).all())
            and abs(float(capacity.sum())-expected_volume)
            <= 1e-10*expected_volume,
            f'region {region} geometric capacity conservation')
    f, s, n, frame_quality = _frame(fibres[selected], sheets[selected])

    # The source has no electrical conductivity or voltage measurement. A
    # positive, affine source-coordinate probe tests only the unit operator.
    extent = float(np.ptp(points[:, 0]))
    require(extent > 0, f'region {region} x extent')
    initial = 1+0.2*(points[:, 0]-points[:, 0].min())/extent
    initial_bytes = initial.tobytes()
    stiffness = volumes[:, None, None]*np.einsum(
        'mik,mjk->mij', gradients, gradients)
    absolute_row_bound = np.zeros(len(source_nodes), dtype=np.float64)
    np.add.at(absolute_row_bound, cells.ravel(),
              np.abs(stiffness).sum(axis=2).ravel())
    require(bool((absolute_row_bound > 0).all()),
            f'region {region} positive operator bound')
    tau = float(0.5*np.min(capacity/absolute_row_bound))
    require(tau > 0 and np.isfinite(tau), f'region {region} finite safe probe step')
    del stiffness, absolute_row_bound

    before_residual, energy_before, gradient_before = _operator(
        gradients, volumes, cells, len(source_nodes), initial)
    directed_energy = {
        'fibre': float(np.sum(volumes*np.einsum('mi,mi->m',
                        gradient_before, f)**2)/2),
        'sheet': float(np.sum(volumes*np.einsum('mi,mi->m',
                        gradient_before, s)**2)/2),
        'normal': float(np.sum(volumes*np.einsum('mi,mi->m',
                        gradient_before, n)**2)/2),
    }
    require(abs(sum(directed_energy.values())-energy_before)
            <= 1e-10*energy_before,
            f'region {region} source-aligned energy decomposition')
    candidate = initial-tau*before_residual/capacity
    replay = initial-tau*_operator(gradients, volumes, cells,
                                   len(source_nodes), initial)[0]/capacity
    require(np.array_equal(candidate, replay)
            and initial.tobytes() == initial_bytes,
            f'region {region} pure replay')
    after_residual, energy_after, _ = _operator(
        gradients, volumes, cells, len(source_nodes), candidate)
    charge_before = float(np.dot(capacity, initial))
    charge_after = float(np.dot(capacity, candidate))
    relative_charge_error = abs(charge_after-charge_before)/charge_before
    require(energy_after < energy_before
            and relative_charge_error <= 1e-12
            and abs(float(before_residual.sum()))
                <= 1e-10*float(np.abs(before_residual).sum())
            and abs(float(after_residual.sum()))
                <= 1e-10*float(np.abs(after_residual).sum()),
            f'region {region} passive energy/charge certificate')
    return source_nodes.astype('<u4'), {
        'region_label': region,
        'source_cell_count': len(source_cells),
        'electrical_dof_count': len(source_nodes),
        'geometric_capacity_sum_m3': float(capacity.sum()),
        'shape_gradient_relative_closure': closure,
        'source_frame': frame_quality,
        'unit_direction_energy_basis': directed_energy,
        'unit_isotropic_probe_step_m2': tau,
        'dimensionless_probe_energy_before': energy_before,
        'dimensionless_probe_energy_after': energy_after,
        'dimensionless_probe_charge_before_m3': charge_before,
        'dimensionless_probe_charge_after_m3': charge_after,
        'relative_charge_error': relative_charge_error,
        'initial_field_sha256': hashlib.sha256(initial_bytes).hexdigest(),
        'candidate_field_sha256': hashlib.sha256(candidate.tobytes()).hexdigest(),
        'bitwise_replay': True,
    }


def audit(asset: Path, output_dir: Path) -> dict:
    asset, output_dir = Path(asset).resolve(), Path(output_dir).resolve()
    manifest_path, manifest, arrays = _load(asset)
    nodes = arrays['nodes.f64le']
    cells = arrays['tetrahedra.u32le']
    labels = arrays['labels.u32le']
    source_nodes_by_region = {}
    rows = []
    for label in MYOCARDIUM:
        source_nodes, row = _region(
            label, nodes, cells, labels,
            arrays['fibres.f64le'], arrays['sheets.f64le'],
            manifest['topology']['regional_cell_counts'][str(label)],
            manifest['topology']['regional_geometric_volume_m3'][str(label)])
        source_nodes_by_region[label] = source_nodes
        rows.append(row)
        print(json.dumps({'region': label,
                          'cells': row['source_cell_count'],
                          'dofs': row['electrical_dof_count'],
                          'relative_charge_error': row['relative_charge_error'],
                          'energy_after': row['dimensionless_probe_energy_after']}),
              flush=True)
    shared = {}
    for i, first in enumerate(MYOCARDIUM):
        for second in MYOCARDIUM[i+1:]:
            shared[f'{first}:{second}'] = int(np.intersect1d(
                source_nodes_by_region[first],
                source_nodes_by_region[second]).size)
    active_nodes = np.unique(np.concatenate(list(source_nodes_by_region.values())))
    dof_nodes = np.concatenate(list(source_nodes_by_region.values()))
    dof_labels = np.concatenate([
        np.full(len(source_nodes_by_region[label]), label, dtype='<u4')
        for label in MYOCARDIUM])
    require(len(dof_nodes) == 281704 and len(active_nodes) == 274315
            and len(dof_nodes)-len(active_nodes) == 7389
            and shared == {'1:2': 2631, '1:3': 1550, '1:4': 253,
                           '2:3': 0, '2:4': 2290, '3:4': 701},
            'cross-region source-node sharing changed')
    output_dir.mkdir(parents=True, exist_ok=True)
    _immutable(output_dir/'electrical-dof-source-nodes.u32le',
               dof_nodes.astype('<u4').tobytes())
    _immutable(output_dir/'electrical-dof-region-labels.u32le',
               dof_labels.tobytes())
    result = {
        'schema': SCHEMA,
        'status': 'passed_anatomical_passive_operator_reference_only',
        'asset_manifest_sha256': human.sha256(manifest_path),
        'source_buffers_sha256': {name: manifest['buffers'][name]['sha256']
                                  for name in FIELDS},
        'source_config_sha256': manifest['source_config_sha256'],
        'predicate_source_sha256': human.sha256(Path(__file__)),
        'source_myocardial_cell_count': sum(r['source_cell_count'] for r in rows),
        'electrical_dof_count': len(dof_nodes),
        'unique_source_node_count': len(active_nodes),
        'duplicated_cross_region_dof_count': len(dof_nodes)-len(active_nodes),
        'cross_region_shared_source_node_counts': shared,
        'dof_source_nodes_sha256': human.sha256(
            output_dir/'electrical-dof-source-nodes.u32le'),
        'dof_region_labels_sha256': human.sha256(
            output_dir/'electrical-dof-region-labels.u32le'),
        'region_rows': rows,
        'explicit_inter_region_conduction_edges': 0,
        'unit_conductivity_only': True,
        'diagnostic_operator_steps': 1,
        'physical_steps': 0,
        'source_voltage_or_activation_map': False,
        'ionic_model': False,
        'native_electrical_runtime': False,
        'heartbeat_qualified': False,
        'clinical_electrophysiology': False,
        'boundary': ('This isolates electrical DOFs at source myocardial label '
                     'interfaces and checks only a passive unit finite-element '
                     'operator. No physiological conductivity, voltage, stimulus, '
                     'AV/Purkinje link, activation law, electromechanical feedback, '
                     'native transaction or calibrated heartbeat is supplied.'),
    }
    atomic_json(output_dir/'summary.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path,
                        default=ROOT/'Build/cardiac-electrical-source-20260930/asset')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.asset, args.output_dir)
    print(json.dumps({key: result[key] for key in
                      ('status', 'source_myocardial_cell_count',
                       'electrical_dof_count',
                       'duplicated_cross_region_dof_count')}), flush=True)


if __name__ == '__main__':
    main()
