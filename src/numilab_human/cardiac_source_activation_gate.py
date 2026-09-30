"""Independent source/topology/timing gate for the cardiac activation candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from . import model as human
from .whole_body_embeddedness import atomic_json


ROOT = human.REPOSITORY_ROOT
SOURCE_OUTPUT = ROOT/'Docs/media/cardiac-wall-anatomy-20260912/ct-simulation_output.csv'
CONFIG = ROOT/'config/cardiac-rodero18-electrical-s4.v1.json'
PRODUCER = ROOT/'src/numilab_human/cardiac_source_activation.py'
REFINER = ROOT/'tools/cardiac_activation_refine.cpp'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise human.ImportError('cardiac source activation gate: '+message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _faces(cells: np.ndarray) -> np.ndarray:
    # Independent lexicographic face sorting, not the producer's V12 helper.
    views = [np.sort(cells[:, pair], axis=1) for pair in
             ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))]
    return np.unique(np.concatenate(views), axis=0)


def audit(asset: Path, supplement: Path, candidate: Path, output: Path) -> dict:
    asset, supplement, candidate, output = map(
        lambda p: Path(p).resolve(), (asset, supplement, candidate, output))
    manifest_path = asset/'manifest.json'
    manifest = human.read_json(manifest_path)
    config = human.read_json(CONFIG)
    summary_path = candidate/'summary.json'
    summary = human.read_json(summary_path)
    require(summary['schema'] == 'numi.human.cardiac-source-activation-reconstruction.v1'
            and summary['status'] == 'full_source_reconstruction_with_source_output_mismatch'
            and summary['asset_manifest_sha256'] == sha(manifest_path)
            and summary['electrical_config_sha256'] == sha(CONFIG)
            and config['anatomy_manifest_sha256'] == sha(manifest_path)
            and summary['source_supplement_sha256'] == sha(supplement)
            == config['source']['supplement_sha256']
            and summary['source_output_sha256'] == sha(SOURCE_OUTPUT)
            == config['source']['source_output_sha256']
            and summary['producer_source_sha256'] == sha(PRODUCER)
            and summary['native_refiner_source_sha256'] == sha(REFINER)
            and summary['replay_bitwise'] is True
            and summary['atrial_tetrahedra_in_electrical_solve'] == 0
            and summary['source_voltage_or_ionic_model'] is False
            and summary['production_native_electrical_steps'] == 0
            and summary['heartbeat_qualified'] is False,
            'source/claim identity')
    fields = {'nodes.f64le': ('<f8', (-1, 3)),
              'tetrahedra.u32le': ('<u4', (-1, 4)),
              'labels.u32le': ('<u4', (-1,)),
              'fibres.f64le': ('<f8', (-1, 3)),
              'uvc_rho.f64le': ('<f8', (-1,)),
              'uvc_z.f64le': ('<f8', (-1,))}
    arrays = {}
    for name, (kind, shape) in fields.items():
        path = asset/name
        require(path.stat().st_size == manifest['buffers'][name]['bytes']
                and sha(path) == manifest['buffers'][name]['sha256'],
                f'source buffer {name}')
        arrays[name] = np.fromfile(path, kind).reshape(shape)
    positions = arrays['nodes.f64le']
    tets = arrays['tetrahedra.u32le']
    labels = arrays['labels.u32le']
    fibre = arrays['fibres.f64le']
    rho = arrays['uvc_rho.f64le']
    z = arrays['uvc_z.f64le']
    require(positions.shape == (300965, 3)
            and tets.shape == (1470083, 4)
            and labels.shape == (len(tets),), 'source arrays')
    left, right = tets[labels == 1], tets[labels == 2]
    left_faces, right_faces = _faces(left), _faces(right)
    # Pair rows by their exact three source node IDs.
    face_type = np.dtype([('a', '<u4'), ('b', '<u4'), ('c', '<u4')])
    joined = np.intersect1d(left_faces.view(face_type).reshape(-1),
                            right_faces.view(face_type).reshape(-1))
    face_nodes = np.unique(joined.view('<u4').reshape(-1, 3))
    shared = np.intersect1d(np.unique(left), np.unique(right))
    point_only = np.setdiff1d(shared, face_nodes)
    require(len(joined) == 4494 and len(face_nodes) == 2628
            and np.array_equal(point_only,
                               np.array([17565, 170947, 235754], '<u4'))
            and summary['lv_rv_complete_shared_faces'] == len(joined)
            and summary['lv_rv_face_connected_shared_nodes'] == len(face_nodes)
            and summary['lv_rv_point_only_shared_source_nodes'] == point_only.tolist(),
            'complete-face versus point-only LV/RV contacts')
    active = np.unique(np.concatenate([left.ravel(), right.ravel()]))
    expected_sources = np.concatenate([active, point_only])
    expected_regions = np.concatenate([np.zeros(len(active), '<u4'),
                                       np.full(len(point_only), 2, '<u4')])
    source_file = candidate/'ventricular-source-nodes.u32le'
    region_file = candidate/'ventricular-dof-regions.u32le'
    graph_file = candidate/'graph-arrival.f64le'
    refined_file = candidate/'refined-arrival.f64le'
    for path in (source_file, region_file, graph_file, refined_file,
                 candidate/'refiner.stdout', candidate/'refiner.stderr'):
        require(sha(path) == summary['output_sha256'][path.name],
                f'candidate artifact hash {path.name}')
    actual_sources = np.fromfile(source_file, '<u4')
    actual_regions = np.fromfile(region_file, '<u4')
    require(np.array_equal(actual_sources, expected_sources)
            and np.array_equal(actual_regions, expected_regions)
            and len(actual_sources) == summary['ventricular_dofs'] == 218080,
            'exact source/region DOF identities')
    graph = np.fromfile(graph_file, '<f8')
    refined = np.fromfile(refined_file, '<f8')
    require(len(graph) == len(refined) == len(expected_sources)
            and bool(np.isfinite(graph).all())
            and bool(np.isfinite(refined).all())
            and bool((refined >= 0).all())
            and bool((refined <= graph).all()),
            'finite monotone activation candidate')
    lut = np.full(len(positions), -1, np.int32)
    lut[active] = np.arange(len(active), dtype=np.int32)
    rv_lut = lut.copy()
    rv_lut[point_only] = np.arange(len(active), len(expected_sources), dtype=np.int32)
    ventricular = (labels == 1) | (labels == 2)
    cells = tets[ventricular]
    selected_labels = labels[ventricular]
    mapped = lut[cells]
    mapped[selected_labels == 2] = rv_lut[cells[selected_labels == 2]]
    seeds = np.unique(mapped[(rho[cells] == 0)
                             & (z[cells] >= 0)
                             & (z[cells] <= config['stimulus_max_uvc_z'])])
    fec = ((rho[cells] == 0).any(axis=1)
           & (z[cells].max(axis=1) <= config['fast_layer_max_uvc_z']))
    require(len(seeds) == summary['stimulus_dof_count'] == 3927
            and int(fec.sum()) == summary['fast_layer_tetrahedra'] == 101712
            and bool((refined[seeds] == 0).all()),
            'source UVC stimulus and one-cell fast layer')
    fibre = fibre[ventricular]
    fibre = fibre/np.linalg.norm(fibre, axis=1)[:, None]
    largest_violation = 0.0
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        delta = positions[cells[:, i]]-positions[cells[:, j]]
        projected = np.einsum('ij,ij->i', delta, fibre)
        length2 = np.einsum('ij,ij->i', delta, delta)
        time = np.sqrt(projected**2/config['fibre_conduction_m_per_s']**2
                       + np.maximum(0, length2-projected**2)
                       /config['transverse_conduction_m_per_s']**2)
        time[fec] = np.sqrt(length2[fec])/config['fast_endocardial_conduction_m_per_s']
        largest_violation = max(largest_violation, float(np.max(
            np.abs(refined[mapped[:, i]]-refined[mapped[:, j]])-time)))
    require(largest_violation <= 1e-12,
            'all ventricular source edges satisfy local travel-time bound')
    left_times = refined[lut[np.unique(left)]]*1000
    left_span = float(np.ptp(left_times))
    left_1090 = float(np.diff(np.percentile(left_times, [10, 90]))[0])
    with SOURCE_OUTPUT.open(newline='') as stream:
        row = next(item for item in csv.DictReader(stream)
                   if item['Mesh_ID'] == '18')
    require(abs(left_span-summary['lv']['span_ms']) <= 1e-10
            and abs(left_1090-summary['lv']['p10_to_p90_ms']) <= 1e-10
            and abs(left_span-float(row['AT_LV'])
                    -summary['source_model_comparison']['reconstructed_AT_LV_minus_source_ms'])
                <= 1e-10
            and abs(left_1090-float(row['AT1090_LV'])
                    -summary['source_model_comparison']['reconstructed_AT1090_LV_minus_source_ms'])
                <= 1e-10
            and summary['source_model_comparison']['exact_source_model_reproduced'] is False,
            'source-model comparison and mismatch retained')
    result = {
        'schema': 'numi.human.cardiac-source-activation-gate.v1',
        'status': 'passed_source_topology_and_timing_bounds_with_model_mismatch',
        'candidate_summary_sha256': sha(summary_path),
        'predicate_source_sha256': sha(Path(__file__)),
        'asset_manifest_sha256': sha(manifest_path),
        'source_supplement_sha256': sha(supplement),
        'source_output_sha256': sha(SOURCE_OUTPUT),
        'ventricular_tetrahedra_checked': len(cells),
        'exact_face_couplings': len(joined),
        'point_only_contacts_kept_separate': point_only.tolist(),
        'stimulus_dofs': len(seeds),
        'fast_layer_tetrahedra': int(fec.sum()),
        'maximum_local_edge_travel_time_violation_seconds': largest_violation,
        'lv_span_ms': left_span,
        'lv_10_to_90_ms': left_1090,
        'case18_source_simulation_span_ms': float(row['AT_LV']),
        'case18_source_simulation_10_to_90_ms': float(row['AT1090_LV']),
        'native_accepted_electrical_steps': 0,
        'source_model_reproduction_qualified': False,
        'clinical_electrophysiology': False,
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path,
                        default=ROOT/'Build/cardiac-electrical-source-20260930/asset')
    parser.add_argument('--supplement', type=Path,
                        default=ROOT/'Build/rodero-s4-source-20260930.pdf')
    parser.add_argument('--candidate', type=Path,
                        default=ROOT/'Build/cardiac-source-activation-20260930')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.asset, args.supplement, args.candidate, args.output)
    print(json.dumps({'status': result['status'],
                      'exact_face_couplings': result['exact_face_couplings'],
                      'lv_span_ms': result['lv_span_ms']}))


if __name__ == '__main__':
    main()
