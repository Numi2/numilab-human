"""Full-source ventricular activation reconstruction with an honest source comparison.

This is an offline Apple CPU reconstruction of the Rodero S4 ventricular
eikonal setup. It is not the source CARP solver, a voltage/ionic model, a
native Human accepted-state transaction or a physiological heartbeat.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from . import model as human
from .whole_body_embeddedness import atomic_json


ROOT = human.REPOSITORY_ROOT
CONFIG = ROOT/'config/cardiac-rodero18-electrical-s4.v1.json'
SOURCE_OUTPUT = ROOT/'Docs/media/cardiac-wall-anatomy-20260912/ct-simulation_output.csv'
REFINER = ROOT/'tools/cardiac_activation_refine.cpp'
SOURCE_FIELDS = {
    'nodes.f64le': ('<f8', (-1, 3)),
    'tetrahedra.u32le': ('<u4', (-1, 4)),
    'labels.u32le': ('<u4', (-1,)),
    'fibres.f64le': ('<f8', (-1, 3)),
    'uvc_rho.f64le': ('<f8', (-1,)),
    'uvc_z.f64le': ('<f8', (-1,)),
}
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise human.ImportError('cardiac source activation: '+reason)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def immutable_array(path: Path, data: np.ndarray, kind: str) -> None:
    payload = data.astype(kind, copy=False).tobytes()
    require(not path.is_symlink(), f'output symlink {path.name}')
    if path.exists():
        require(path.read_bytes() == payload, f'changed output {path.name}')
        return
    temporary = path.with_name(path.name+f'.{os.getpid()}.pending')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _unique_faces(cells: np.ndarray) -> np.ndarray:
    faces = np.concatenate([cells[:, [0, 1, 2]], cells[:, [0, 1, 3]],
                            cells[:, [0, 2, 3]], cells[:, [1, 2, 3]]])
    faces.sort(axis=1)
    return np.unique(np.ascontiguousarray(faces).view('V12').reshape(-1))


def _source(asset: Path, supplement: Path):
    config = human.read_json(CONFIG)
    manifest_path = asset/'manifest.json'
    manifest = human.read_json(manifest_path)
    require(config['schema'] == 'HumanPack.cardiac-rodero18-electrical-s4.v1'
            and config['anatomy_manifest_sha256'] == sha(manifest_path)
            and config['source']['supplement_sha256'] == sha(supplement)
            and config['source']['source_output_sha256'] == sha(SOURCE_OUTPUT)
            and config['ventricular_region_labels'] == [1, 2]
            and config['fibre_conduction_m_per_s'] == 0.8
            and config['transverse_conduction_m_per_s'] == 0.23
            and config['fast_endocardial_conduction_m_per_s'] == 5.6
            and config['endocardium_rho'] == 0
            and config['stimulus_max_uvc_z'] == 0.33
            and config['fast_layer_max_uvc_z'] == 0.7
            and manifest['schema'] == 'HumanPack.cardiac-wall-source-asset.v1',
            'pinned source and supplement identity')
    arrays = {}
    for name, (kind, shape) in SOURCE_FIELDS.items():
        path = asset/name
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == manifest['buffers'][name]['bytes']
                and sha(path) == manifest['buffers'][name]['sha256'],
                f'source buffer {name}')
        arrays[name] = np.fromfile(path, kind).reshape(shape)
    require(arrays['nodes.f64le'].shape == (300965, 3)
            and arrays['tetrahedra.u32le'].shape == (1470083, 4)
            and arrays['labels.u32le'].shape == (1470083,)
            and bool(np.isfinite(arrays['nodes.f64le']).all())
            and bool(np.isfinite(arrays['fibres.f64le']).all()),
            'source dimensions/finite fields')
    with SOURCE_OUTPUT.open(newline='') as stream:
        comparator = next(row for row in csv.DictReader(stream)
                          if row['Mesh_ID'] == '18')
    require(float(comparator['QRS_LV']) == config['target_comparison']['QRS_LV_ms']
            and float(comparator['AT1090_LV']) == config['target_comparison']['AT1090_LV_ms']
            and float(comparator['AT_LV']) == config['target_comparison']['AT_LV_ms'],
            'case18 source-model comparison rows')
    return config, manifest, arrays, comparator


def _domains(tets: np.ndarray, labels: np.ndarray):
    lv = tets[labels == 1]
    rv = tets[labels == 2]
    shared_faces = np.intersect1d(_unique_faces(lv), _unique_faces(rv))
    face_nodes = np.unique(shared_faces.view('<u4').reshape(-1, 3))
    shared_nodes = np.intersect1d(np.unique(lv), np.unique(rv))
    point_only = np.setdiff1d(shared_nodes, face_nodes)
    require(len(shared_faces) == 4494 and len(face_nodes) == 2628
            and np.array_equal(point_only,
                               np.array([17565, 170947, 235754], '<u4')),
            'complete-face ventricular interface versus point-only contacts')
    active = np.unique(np.concatenate([lv.reshape(-1), rv.reshape(-1)]))
    dof_source = np.concatenate([active, point_only]).astype('<u4')
    dof_regions = np.concatenate([np.zeros(len(active), '<u4'),
                                  np.full(len(point_only), 2, '<u4')])
    lut = np.full(300965, -1, np.int32)
    lut[active] = np.arange(len(active), dtype=np.int32)
    rv_lut = lut.copy()
    rv_lut[point_only] = np.arange(len(active), len(dof_source), dtype=np.int32)
    require(len(dof_source) == 218080 and len(active) == 218077,
            'full ventricular DOF coverage')
    return dof_source, dof_regions, lut, rv_lut, shared_faces, face_nodes, point_only


def _graph(nodes: np.ndarray, source_tets: np.ndarray, cells: np.ndarray,
           fibres: np.ndarray, fec: np.ndarray, seeds: np.ndarray,
           count: int, config: dict):
    normalized = fibres/np.linalg.norm(fibres, axis=1)[:, None]
    require(bool(np.isfinite(normalized).all()), 'nonzero source fibre field')
    keys, weights = [], []
    for ia, ib in PAIRS:
        a, b = cells[:, ia], cells[:, ib]
        lo = np.minimum(a, b).astype(np.uint64)
        hi = np.maximum(a, b).astype(np.uint64)
        delta = nodes[source_tets[:, ia]]-nodes[source_tets[:, ib]]
        parallel = np.einsum('ij,ij->i', delta, normalized)
        length2 = np.einsum('ij,ij->i', delta, delta)
        transverse2 = np.maximum(0, length2-parallel*parallel)
        weight = np.sqrt(
            parallel**2/config['fibre_conduction_m_per_s']**2
            + transverse2/config['transverse_conduction_m_per_s']**2)
        weight[fec] = np.sqrt(length2[fec])/config['fast_endocardial_conduction_m_per_s']
        keys.append(lo*np.uint64(count)+hi)
        weights.append(weight)
    keys, weights = np.concatenate(keys), np.concatenate(weights)
    order = np.argsort(keys)
    keys, weights = keys[order], weights[order]
    first = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1]
    unique_keys, unique_weights = keys[first], np.minimum.reduceat(weights, first)
    a = (unique_keys//count).astype(np.int32)
    b = (unique_keys%count).astype(np.int32)
    require(bool((unique_weights > 0).all()) and not bool((a == b).any()),
            'positive distinct source-edge travel times')
    graph = csr_matrix((np.concatenate([unique_weights, unique_weights,
                                        np.full(len(seeds), 1e-15)]),
                        (np.concatenate([a, b, np.full(len(seeds), count, np.int32)]),
                         np.concatenate([b, a, seeds.astype(np.int32)]))),
                       shape=(count+1, count+1))
    components, _ = connected_components(graph[:count, :count], directed=False)
    require(components == 1, 'source-connected ventricular domain')
    arrival = dijkstra(graph, directed=True, indices=count)[:count]
    require(bool(np.isfinite(arrival).all()) and len(seeds) == 3927,
            'all ventricles reached from source stimulus')
    return arrival, len(a), components


def _region_stats(values: np.ndarray, ids: np.ndarray) -> dict:
    times = values[ids]*1000
    return {'source_node_count': len(ids),
            'minimum_ms': float(times.min()),
            'maximum_ms': float(times.max()),
            'span_ms': float(np.ptp(times)),
            'p10_ms': float(np.percentile(times, 10)),
            'p90_ms': float(np.percentile(times, 90)),
            'p10_to_p90_ms': float(np.percentile(times, 90)-np.percentile(times, 10))}


def audit(asset: Path, supplement: Path, output: Path) -> dict:
    asset, supplement, output = map(lambda p: Path(p).resolve(),
                                    (asset, supplement, output))
    config, manifest, arrays, comparator = _source(asset, supplement)
    nodes = arrays['nodes.f64le']
    tets = arrays['tetrahedra.u32le']
    labels = arrays['labels.u32le']
    fibre = arrays['fibres.f64le']
    rho = arrays['uvc_rho.f64le']
    z = arrays['uvc_z.f64le']
    dof_source, dof_regions, lut, rv_lut, shared_faces, face_nodes, point_only = (
        _domains(tets, labels))
    selected = np.isin(labels, [1, 2])
    source_tets = tets[selected]
    selected_labels = labels[selected]
    cells = lut[source_tets]
    cells[selected_labels == 2] = rv_lut[source_tets[selected_labels == 2]]
    require(bool((cells >= 0).all()), 'complete source tetrahedral DOF map')
    stimulus = ((rho[source_tets] == 0)
                & (z[source_tets] >= 0)
                & (z[source_tets] <= config['stimulus_max_uvc_z']))
    seeds = np.unique(cells[stimulus])
    fec = ((rho[source_tets] == 0).any(axis=1)
           & (z[source_tets].max(axis=1) <= config['fast_layer_max_uvc_z']))
    require(len(seeds) == 3927 and int(fec.sum()) == 101712,
            'source stimulus/FEC reconstruction coverage')
    output.mkdir(parents=True, exist_ok=True)
    immutable_array(output/'ventricular-source-nodes.u32le', dof_source, '<u4')
    immutable_array(output/'ventricular-dof-regions.u32le', dof_regions, '<u4')
    graph_time, edge_count, components = _graph(
        nodes, source_tets, cells, fibre[selected], fec, seeds,
        len(dof_source), config)
    immutable_array(output/'graph-arrival.f64le', graph_time, '<f8')
    binary = output/'cardiac-activation-refine'
    compile_command = ['clang++', '-std=c++20', '-O3', '-march=native',
                       str(REFINER), '-o', str(binary)]
    subprocess.run(compile_command, check=True, capture_output=True, text=True)
    run_command = [str(binary), str(asset), str(output),
                   str(config['fibre_conduction_m_per_s']),
                   str(config['transverse_conduction_m_per_s']),
                   str(config['fast_endocardial_conduction_m_per_s']),
                   str(config['stimulus_max_uvc_z']),
                   str(config['fast_layer_max_uvc_z'])]
    run = subprocess.run(run_command, check=True, capture_output=True, text=True)
    (output/'refiner.stdout').write_text(run.stdout)
    (output/'refiner.stderr').write_text(run.stderr)
    refined_path = output/'refined-arrival.f64le'
    first_hash = sha(refined_path)
    replay = subprocess.run(run_command, check=True, capture_output=True, text=True)
    require(sha(refined_path) == first_hash and replay.stdout == run.stdout
            and replay.stderr == run.stderr, 'native solver bitwise replay')
    refined = np.fromfile(refined_path, '<f8')
    require(len(refined) == len(dof_source)
            and bool(np.isfinite(refined).all())
            and bool((refined >= 0).all())
            and bool((refined <= graph_time).all())
            and bool((refined[seeds] == 0).all()),
            'native full-mesh activation output bounds')
    lv_nodes = np.unique(tets[labels == 1])
    rv_nodes = np.unique(tets[labels == 2])
    lv = _region_stats(refined, lut[lv_nodes])
    rv = _region_stats(refined, rv_lut[rv_nodes])
    source_lv_span = float(comparator['AT_LV'])
    source_lv_1090 = float(comparator['AT1090_LV'])
    summary = {
        'schema': 'numi.human.cardiac-source-activation-reconstruction.v1',
        'status': 'full_source_reconstruction_with_source_output_mismatch',
        'asset_manifest_sha256': sha(asset/'manifest.json'),
        'electrical_config_sha256': sha(CONFIG),
        'source_supplement_sha256': sha(supplement),
        'source_output_sha256': sha(SOURCE_OUTPUT),
        'producer_source_sha256': sha(Path(__file__)),
        'native_refiner_source_sha256': sha(REFINER),
        'native_refiner_binary_sha256': sha(binary),
        'compile_command': compile_command,
        'platform': platform.platform(),
        'ventricular_tetrahedra': len(source_tets),
        'ventricular_unique_source_nodes': len(dof_source)-len(point_only),
        'ventricular_dofs': len(dof_source),
        'lv_rv_complete_shared_faces': len(shared_faces),
        'lv_rv_face_connected_shared_nodes': len(face_nodes),
        'lv_rv_point_only_shared_source_nodes': point_only.tolist(),
        'atrial_tetrahedra_in_electrical_solve': 0,
        'fast_layer_tetrahedra': int(fec.sum()),
        'stimulus_dof_count': len(seeds),
        'graph_undirected_edge_count': edge_count,
        'ventricular_graph_components': components,
        'replay_bitwise': True,
        'output_sha256': {
            name: sha(output/name) for name in
            ('ventricular-source-nodes.u32le', 'ventricular-dof-regions.u32le',
             'graph-arrival.f64le', 'refined-arrival.f64le', 'refiner.stdout',
             'refiner.stderr')},
        'lv': lv, 'rv': rv,
        'source_model_comparison': {
            'case18_QRS_LV_ms': float(comparator['QRS_LV']),
            'case18_AT_LV_ms': source_lv_span,
            'case18_AT1090_LV_ms': source_lv_1090,
            'reconstructed_AT_LV_minus_source_ms': lv['span_ms']-source_lv_span,
            'reconstructed_AT1090_LV_minus_source_ms':
                lv['p10_to_p90_ms']-source_lv_1090,
            'comparison_is_simulation_output_not_measured_human_data': True,
            'exact_source_model_reproduced': False,
        },
        'source_voltage_or_ionic_model': False,
        'production_native_electrical_steps': 0,
        'heartbeat_qualified': False,
        'boundary': ('A source-parameter, full-mesh offline ventricular timing '
                     'reconstruction with explicit complete-face LV/RV coupling. '
                     'Its output differs from the published case18 CARP simulation. '
                     'The one-cell FEC reconstruction is a declared variant; no '
                     'voltage, atrial/AV system, native accepted step, ECG, '
                     'mechanical contraction or clinical qualification follows.'),
    }
    atomic_json(output/'summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path,
                        default=ROOT/'Build/cardiac-electrical-source-20260930/asset')
    parser.add_argument('--supplement', type=Path,
                        default=ROOT/'Build/rodero-s4-source-20260930.pdf')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.asset, args.supplement, args.output)
    print(json.dumps({'status': result['status'], 'lv': result['lv'],
                      'source_model_comparison': result['source_model_comparison']}))


if __name__ == '__main__':
    main()
