"""Compare the source ventricular FEC vertex ring with exact boundary faces.

The source archive lacks the case18 fast-endocardial cell tag. This produces a
geometric sensitivity candidate, not a replacement for that missing tag or a
qualification of the published CARP simulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from numilab_human import cardiac_source_activation as activation
from numilab_human import cardiac_source_activation_gate as prior_gate


ROOT = activation.ROOT
ASSET = ROOT / 'Build/cardiac-electrical-source-20260930/asset'
SUPPLEMENT = ROOT / 'Docs/media/cardiac-source-activation-20260930/rodero-s4-source.pdf'
PRIOR = ROOT / 'Docs/media/cardiac-source-activation-20260930'
OUTPUT = ROOT / 'Docs/media/cardiac-fec-layer-topology-20260930'
BUILD = ROOT / 'Build/cardiac-fec-layer-topology-20260930'
FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError('cardiac FEC topology: ' + message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def masks(tets: np.ndarray, labels: np.ndarray, rho: np.ndarray,
          z: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    selected = (labels == 1) | (labels == 2)
    cells = tets[selected]
    rz = rho[cells] == 0.0
    below = np.max(z[cells], axis=1) <= 0.7
    count = np.sum(rz, axis=1)
    vertex = (count > 0) & below
    # Exact quotient over complete source faces. A face is an exterior
    # ventricular face only if it occurs in one selected tetrahedron.
    faces = np.concatenate([cells[:, corner] for corner in FACES])
    faces.sort(axis=1)
    keys = np.ascontiguousarray(faces).view('V12').ravel()
    _, inverse, incidence = np.unique(
        keys, return_inverse=True, return_counts=True)
    boundary = (incidence[inverse] == 1).reshape(4, len(cells))
    face = np.zeros(len(cells), bool)
    boundary_endo_faces = 0
    for face_index, corner in enumerate(FACES):
        admitted = boundary[face_index] & below & \
            np.all(rho[cells[:, corner]] == 0.0, axis=1)
        face |= admitted
        boundary_endo_faces += int(admitted.sum())
    require(np.all(face <= vertex), 'boundary-face cells outside vertex ring')
    measurements = {
        'ventricular_cells': int(len(cells)),
        'vertex_ring_fec_cells': int(vertex.sum()),
        'boundary_face_fec_cells': int(face.sum()),
        'boundary_endocardial_faces': boundary_endo_faces,
        'vertex_only_fec_cells': int(np.sum(vertex & ~face)),
        'point_contact_only_cells': int(np.sum((count == 1) & below)),
        'edge_contact_only_cells': int(np.sum((count == 2) & below)),
        'three_or_four_endo_corner_cells_without_boundary_face':
            int(np.sum((count >= 3) & below & ~face)),
        'ventricular_unique_faces': int(len(incidence)),
        'ventricular_exterior_faces': int(np.sum(incidence == 1)),
        'ventricular_nonmanifold_faces': int(np.sum(incidence > 2)),
    }
    require(measurements['ventricular_cells'] == 1097534 and
            measurements['vertex_ring_fec_cells'] == 101712 and
            measurements['boundary_face_fec_cells'] == 35081 and
            measurements['boundary_endocardial_faces'] == 35082 and
            measurements['point_contact_only_cells'] == 32792 and
            measurements['edge_contact_only_cells'] == 33837 and
            measurements['three_or_four_endo_corner_cells_without_boundary_face'] == 2 and
            measurements['ventricular_nonmanifold_faces'] == 0,
            'pinned source FEC contact dimensions changed')
    full = np.zeros(len(labels), np.uint8)
    full[selected] = face
    return vertex, full, measurements


def masked_refiner_source() -> str:
    original = activation.REFINER.read_text()
    changes = (
        ('auto arrival=read<double>(scratch+"/graph-arrival.f64le");',
         'auto arrival=read<double>(scratch+"/graph-arrival.f64le");\n'
         '    auto fec_mask=read<uint8_t>(scratch+"/fec-mask.u8le");'),
        ('rho.size()!=positions.size() || z.size()!=positions.size()) return 3;',
         'rho.size()!=positions.size() || z.size()!=positions.size() || '
         'fec_mask.size()!=labels.size()) return 3;'),
        ('bool fec=endo && maxz<=fec_z;',
         'bool fec=fec_mask[i]!=0;'),
    )
    for old, new in changes:
        require(original.count(old) == 1, 'native refiner transformation anchor')
        original = original.replace(old, new)
    return original


def run_variant(arrays: dict, full_mask: np.ndarray,
                config: dict) -> tuple[np.ndarray, dict]:
    BUILD.mkdir(parents=True, exist_ok=True)
    tets = arrays['tetrahedra.u32le']
    labels = arrays['labels.u32le']
    selected = (labels == 1) | (labels == 2)
    source_tets = tets[selected]
    source_labels = labels[selected]
    dof_source, dof_regions, lut, rv_lut, _, _, _ = activation._domains(
        tets, labels)
    original_nodes = np.fromfile(PRIOR / 'ventricular-source-nodes.u32le', '<u4')
    original_regions = np.fromfile(PRIOR / 'ventricular-dof-regions.u32le', '<u4')
    require(np.array_equal(dof_source, original_nodes) and
            np.array_equal(dof_regions, original_regions),
            'source electrical quotient changed')
    cells = lut[source_tets]
    cells[source_labels == 2] = rv_lut[source_tets[source_labels == 2]]
    rho = arrays['uvc_rho.f64le']
    z = arrays['uvc_z.f64le']
    stimulus = (rho[source_tets] == 0.0) & (z[source_tets] >= 0.0) & \
        (z[source_tets] <= 0.33)
    seeds = np.unique(cells[stimulus])
    require(len(seeds) == 3927, 'source stimulus coverage')
    graph, edges, components = activation._graph(
        arrays['nodes.f64le'], source_tets, cells,
        arrays['fibres.f64le'][selected], full_mask[selected].astype(bool),
        seeds, len(dof_source), config)
    require(components == 1 and len(graph) == 218080,
            'boundary-face candidate graph reachability')
    dof_source.astype('<u4').tofile(BUILD / 'ventricular-source-nodes.u32le')
    dof_regions.astype('<u4').tofile(BUILD / 'ventricular-dof-regions.u32le')
    graph.astype('<f8').tofile(BUILD / 'graph-arrival.f64le')
    full_mask.tofile(BUILD / 'fec-mask.u8le')
    generated = BUILD / 'cardiac_activation_refine_masked.cpp'
    generated.write_text(masked_refiner_source())
    binary = BUILD / 'cardiac-activation-refine-masked'
    subprocess.run(['clang++', '-std=c++20', '-O3', '-march=native',
                    str(generated), '-o', str(binary)],
                   check=True, text=True, capture_output=True)
    command = [str(binary), str(ASSET), str(BUILD),
               str(config['fibre_conduction_m_per_s']),
               str(config['transverse_conduction_m_per_s']),
               str(config['fast_endocardial_conduction_m_per_s']),
               str(config['stimulus_max_uvc_z']),
               str(config['fast_layer_max_uvc_z'])]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    first_hash = sha(BUILD / 'refined-arrival.f64le')
    replay = subprocess.run(command, check=True, capture_output=True, text=True)
    require(sha(BUILD / 'refined-arrival.f64le') == first_hash and
            first.stdout == replay.stdout and first.stderr == replay.stderr,
            'masked native refiner replay')
    result = np.fromfile(BUILD / 'refined-arrival.f64le', '<f8')
    require(len(result) == len(dof_source) and np.isfinite(result).all() and
            (result >= 0).all() and (result <= graph + 1e-12).all() and
            np.all(result[seeds] == 0), 'masked native arrival bounds')
    lv = np.unique(tets[labels == 1])
    rv = np.unique(tets[labels == 2])
    return result, {
        'lv': activation._region_stats(result, lut[lv]),
        'rv': activation._region_stats(result, rv_lut[rv]),
        'graph_edges': edges,
        'graph_arrival_sha256': sha(BUILD / 'graph-arrival.f64le'),
        'generated_refiner_sha256': sha(generated),
        'refiner_binary_sha256': sha(binary),
        'native_replay_bitwise': True,
        'refiner_stdout_sha256': hashlib.sha256(first.stdout.encode()).hexdigest(),
    }


def audit(execute: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix='cardiac-fec-prior-') as folder:
        prior = prior_gate.audit(ASSET, SUPPLEMENT, PRIOR,
                                 Path(folder) / 'prior-gate.json')
    require(prior['status'] == 'passed_source_topology_and_timing_bounds_with_model_mismatch',
            'pinned full-source activation reference')
    config, manifest, arrays, _ = activation._source(ASSET, SUPPLEMENT)
    vertex, face_full, measured = masks(
        arrays['tetrahedra.u32le'], arrays['labels.u32le'],
        arrays['uvc_rho.f64le'], arrays['uvc_z.f64le'])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mask_path = OUTPUT / 'boundary-face-fec-candidate.u8le'
    activation.immutable_array(mask_path, face_full, '<u1')
    if execute:
        arrival, variant = run_variant(arrays, face_full, config)
        activation.immutable_array(
            OUTPUT / 'boundary-face-arrival.f64le', arrival, '<f8')
    else:
        arrival_path = OUTPUT / 'boundary-face-arrival.f64le'
        receipt_path = OUTPUT / 'receipt.json'
        require(arrival_path.is_file() and receipt_path.is_file(),
                'candidate arrival requires --execute')
        retained = json.loads(receipt_path.read_text())
        require(sha(arrival_path) == retained['boundary_face_arrival_sha256'] and
                sha(mask_path) == retained['boundary_face_mask_sha256'],
                'retained FEC mask or arrival changed')
        arrival = np.fromfile(arrival_path, '<f8')
        require(len(arrival) == 218080 and np.isfinite(arrival).all(),
                'retained boundary-face arrival')
        variant = retained['variant']
    prior_summary = json.loads((PRIOR / 'summary.json').read_text())
    require(len(vertex) == measured['ventricular_cells'] and
            prior_summary['fast_layer_tetrahedra'] == measured['vertex_ring_fec_cells'],
            'existing vertex-ring reconstruction coverage')
    require(variant['lv']['span_ms'] > prior_summary['lv']['span_ms'],
            'boundary-face candidate did not slow LV activation span')
    return {
        'schema': 'numi.human.cardiac-fec-layer-topology.v1',
        'status': 'boundary_face_candidate_measured_source_mismatch_retained',
        'source_manifest_sha256': sha(ASSET / 'manifest.json'),
        'source_activation_gate_sha256': sha(PRIOR / 'independent-gate.json'),
        'source_activation_summary_sha256': sha(PRIOR / 'summary.json'),
        'source_refiner_sha256': sha(activation.REFINER),
        'auditor_source_sha256': sha(Path(__file__)),
        'boundary_face_mask_sha256': sha(mask_path),
        'boundary_face_arrival_sha256': sha(OUTPUT / 'boundary-face-arrival.f64le'),
        'measurements': measured,
        'vertex_ring_lv': prior_summary['lv'],
        'variant': variant,
        'published_case18_lv_span_ms': config['target_comparison']['AT_LV_ms'],
        'published_case18_lv_10_to_90_ms': config['target_comparison']['AT1090_LV_ms'],
        'source_model_reproduced': False,
        'native_accepted_electrical_steps': 0,
        'heartbeat_qualified': False,
        'boundary': 'Geometric fast-layer sensitivity on the pinned case18 mesh. The source FEC cell tag is absent; neither vertex-ring nor boundary-face mask is asserted to be the published CARP tag. No voltage, ionic state, native accepted electrical step or heartbeat.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    result = audit(args.execute)
    path = OUTPUT / 'receipt.json'
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'status': result['status'],
                      'vertex_ring_cells': result['measurements']['vertex_ring_fec_cells'],
                      'boundary_face_cells': result['measurements']['boundary_face_fec_cells'],
                      'vertex_ring_lv_span_ms': result['vertex_ring_lv']['span_ms'],
                      'boundary_face_lv_span_ms': result['variant']['lv']['span_ms']}))


if __name__ == '__main__':
    main()
