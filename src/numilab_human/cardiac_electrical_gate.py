"""Independent source/DOF identity gate for the passive cardiac reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import model as human
from .whole_body_embeddedness import atomic_json


ROOT = human.REPOSITORY_ROOT
SCHEMA = 'numi.human.cardiac-passive-electrical-reference-gate.v1'
SOURCE_FILES = ('nodes.f64le', 'tetrahedra.u32le', 'labels.u32le',
                'fibres.f64le', 'sheets.f64le')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise human.ImportError('cardiac electrical gate: '+message)


def audit(asset: Path, candidate: Path, output: Path) -> dict:
    asset, candidate, output = (Path(p).resolve()
                                for p in (asset, candidate, output))
    source_manifest_path = asset/'manifest.json'
    source = human.read_json(source_manifest_path)
    summary_path = candidate/'summary.json'
    summary = human.read_json(summary_path)
    require(summary['schema'] == 'numi.human.cardiac-passive-electrical-reference.v1'
            and summary['status'] == 'passed_anatomical_passive_operator_reference_only'
            and summary['asset_manifest_sha256'] == human.sha256(source_manifest_path)
            and summary['source_config_sha256'] == human.sha256(
                ROOT/'config/cardiac-wall-rodero18.v1.json')
            and summary['predicate_source_sha256'] == human.sha256(
                Path(__file__).with_name('cardiac_electrical_reference.py'))
            and summary['source_myocardial_cell_count'] == 1337558
            and summary['electrical_dof_count'] == 281704
            and summary['unique_source_node_count'] == 274315
            and summary['duplicated_cross_region_dof_count'] == 7389
            and summary['explicit_inter_region_conduction_edges'] == 0
            and summary['unit_conductivity_only'] is True
            and summary['diagnostic_operator_steps'] == 1
            and summary['physical_steps'] == 0
            and summary['source_voltage_or_activation_map'] is False
            and summary['ionic_model'] is False
            and summary['native_electrical_runtime'] is False
            and summary['heartbeat_qualified'] is False,
            'candidate claim/source identity')
    for name in SOURCE_FILES:
        require(summary['source_buffers_sha256'][name] ==
                source['buffers'][name]['sha256']
                and human.sha256(asset/name) == source['buffers'][name]['sha256'],
                f'source buffer {name}')
    nodes = np.fromfile(asset/'nodes.f64le', '<f8').reshape(-1, 3)
    tets = np.fromfile(asset/'tetrahedra.u32le', '<u4').reshape(-1, 4)
    labels = np.fromfile(asset/'labels.u32le', '<u4')
    require(nodes.shape == (300965, 3) and tets.shape == (1470083, 4)
            and labels.shape == (len(tets),), 'source dimensions')
    node_file = candidate/'electrical-dof-source-nodes.u32le'
    label_file = candidate/'electrical-dof-region-labels.u32le'
    require(human.sha256(node_file) == summary['dof_source_nodes_sha256']
            and human.sha256(label_file) == summary['dof_region_labels_sha256'],
            'DOF artifact hashes')
    recorded_nodes = np.fromfile(node_file, '<u4')
    recorded_labels = np.fromfile(label_file, '<u4')
    expected_nodes, expected_labels = [], []
    checks = []
    nodes_by_region = {}
    for label in (1, 2, 3, 4):
        selected = tets[labels == label]
        unique = np.unique(selected)
        expected_nodes.append(unique)
        expected_labels.append(np.full(len(unique), label, dtype='<u4'))
        nodes_by_region[label] = unique
        row = next(row for row in summary['region_rows']
                   if row['region_label'] == label)
        tetra = nodes[selected]
        edges = tetra[:, 1:] - tetra[:, 0, None, :]
        volume = float(np.sum(np.einsum(
            'ij,ij->i', edges[:, 0],
            np.cross(edges[:, 1], edges[:, 2])))/6)
        source_volume = source['topology']['regional_geometric_volume_m3'][str(label)]
        require(len(selected) == row['source_cell_count']
                == source['topology']['regional_cell_counts'][str(label)]
                and len(unique) == row['electrical_dof_count']
                and abs(volume-source_volume) <= 1e-10*source_volume
                and abs(volume-row['geometric_capacity_sum_m3'])
                    <= 1e-10*volume
                and row['shape_gradient_relative_closure'] < 1e-10
                and row['bitwise_replay'] is True
                and row['relative_charge_error'] <= 1e-12
                and row['dimensionless_probe_energy_after']
                    < row['dimensionless_probe_energy_before']
                and abs(sum(row['unit_direction_energy_basis'].values())
                        -row['dimensionless_probe_energy_before'])
                    <= 1e-10*row['dimensionless_probe_energy_before'],
                f'region {label} source/electrical certificate')
        checks.append({'region_label': label,
                       'source_cell_count': len(selected),
                       'source_node_count': len(unique),
                       'independent_geometric_capacity_m3': volume})
    require(np.array_equal(recorded_nodes, np.concatenate(expected_nodes))
            and np.array_equal(recorded_labels, np.concatenate(expected_labels)),
            'exact (region, source-node) DOF mapping')
    overlap = {
        f'{a}:{b}': int(np.intersect1d(nodes_by_region[a],
                                      nodes_by_region[b]).size)
        for a in (1, 2, 3, 4) for b in range(a+1, 5)}
    require(overlap == summary['cross_region_shared_source_node_counts']
            and len(np.unique(recorded_nodes)) ==
                summary['unique_source_node_count'],
            'cross-region node overlap')
    result = {
        'schema': SCHEMA, 'status': 'passed_independent_source_and_dof_gate',
        'candidate_summary_sha256': human.sha256(summary_path),
        'source_manifest_sha256': human.sha256(source_manifest_path),
        'predicate_source_sha256': human.sha256(Path(__file__)),
        'region_checks': checks,
        'cross_region_shared_source_node_counts': overlap,
        'no_implicit_inter_region_conduction': True,
        'native_or_physiological_qualification': False,
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path,
                        default=ROOT/'Build/cardiac-electrical-source-20260930/asset')
    parser.add_argument('--candidate', type=Path,
                        default=ROOT/'Build/cardiac-electrical-reference-20260930')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.asset, args.candidate, args.output)
    print(json.dumps({'status': result['status'],
                      'region_count': len(result['region_checks'])}), flush=True)


if __name__ == '__main__':
    main()
