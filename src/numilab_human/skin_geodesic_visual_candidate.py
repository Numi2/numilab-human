"""Compile a source-bound, derived visual skin-weight candidate.

Only visual weights change. The source skin, binding table, face indices and
normal/position records stay exact. This does not qualify skin mechanics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from . import model as human
from .skin_crossing_attribution import _source_graph_distances
from .skin_embeddedness_gate import _source_mesh
from .whole_body_embeddedness import atomic_json


SCHEMA = 'numi.human.derived-geodesic-skin-visual-candidate.v1'
BLEND = 0.05
RADIUS_M = 0.12
CODE_FILES = ('skin_geodesic_visual_candidate.py',
              'skin_crossing_attribution.py', 'skin_embeddedness_gate.py',
              'skin_crossing_support.py')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError('geodesic skin visual candidate: ' + message)


def _write_immutable(path: Path, data: bytes) -> None:
    require(not path.is_symlink(), f'output symlink {path.name}')
    if path.exists():
        require(path.read_bytes() == data, f'output identity differs {path.name}')
        return
    temp = path.with_name(path.name+f'.{os.getpid()}.pending')
    with temp.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def compile_candidate(payload: Path, manifest_path: Path, solution_path: Path,
                      attribution_path: Path, output_dir: Path, *,
                      blend: float = BLEND, radius_m: float = RADIUS_M) -> dict:
    payload, manifest_path, solution_path, attribution_path, output_dir = (
        Path(p).resolve() for p in (payload, manifest_path, solution_path,
                                   attribution_path, output_dir))
    require(all(p.is_file() and not p.is_symlink() for p in
                (payload, manifest_path, solution_path, attribution_path)),
            'retained source input missing')
    require(np.isfinite(blend) and 0 < blend <= 1
            and np.isfinite(radius_m) and radius_m > 0,
            'candidate blend/radius range')
    manifest = human.read_json(manifest_path)
    nb, nv, ni, fingerprint, vertices, faces = _source_mesh(payload, manifest)
    require(manifest['coverage']['binding_solution']['sha256'] == human.sha256(solution_path),
            'source full-field solution identity')
    with np.load(solution_path) as archive:
        full = archive['full_weights']
        targets = archive['seed_targets']
    require(full.shape == (nv, nb) and targets.shape[1] == 5
            and bool(np.isfinite(full).all()), 'source solution layout')
    raw = payload.read_bytes()
    vertex_start = 60+36*nb
    full_start = vertex_start+56*nv+4*ni
    original_f4 = np.frombuffer(raw, '<f4', count=nv*nb,
                                 offset=full_start).reshape(nv, nb)
    require(np.array_equal(full.astype('<f4'), original_f4),
            'source full-field byte parity')
    body_ids = np.frombuffer(raw, '<u4', count=nb*9, offset=60).reshape(nb, 9)[:, 0]
    envelopes = {b['core_body_index']: b
                 for b in manifest['source']['registered_bone_envelopes']}
    require(set(map(int, body_ids)) == set(envelopes), 'bone envelope coverage')
    diameters = np.asarray([
        np.linalg.norm(np.asarray(envelopes[int(i)]['maximum_world_m'])
                       -np.asarray(envelopes[int(i)]['minimum_world_m']))
        for i in body_ids])
    require(bool(np.isfinite(diameters).all() and (diameters > 0).all()),
            'source bone envelope diameter')
    inverse, distances = _source_graph_distances(vertices, faces, targets, nb)
    screened = full*np.exp(-distances[inverse]/diameters)
    screened /= screened.sum(axis=1, keepdims=True)

    attribution = human.read_json(attribution_path)
    cases = attribution.get('cases', [])
    case_ids = [c.get('case') for c in cases]
    pairs_by_case = [c.get('exact_intersection_pairs') for c in cases]
    require(attribution.get('schema') == 'numi.human.native-skin-crossing-attribution.v1'
            and attribution.get('source', {}).get('payload_sha256') == human.sha256(payload)
            and attribution.get('source', {}).get('manifest_sha256') == human.sha256(manifest_path)
            and attribution.get('source', {}).get('binding_solution_sha256') == human.sha256(solution_path)
            and bool(cases) and all(isinstance(case_id, str) and case_id for case_id in case_ids)
            and len(set(case_ids)) == len(case_ids)
            and all(isinstance(pairs, list)
                    and c.get('exact_intersection_pair_count') == len(pairs)
                    for c, pairs in zip(cases, pairs_by_case, strict=True))
            and attribution.get('total_exact_intersection_pairs') ==
                sum(len(pairs) for pairs in pairs_by_case),
            'exact crossing attribution identity and complete pair counts')
    support_faces = sorted({int(face) for case in attribution['cases']
                            for pair in case['exact_intersection_pairs'] for face in pair})
    require(len(support_faces) > 0 and support_faces[0] >= 0
            and support_faces[-1] < len(faces)
            and all(len(pair) == 2 and all(type(face) is int and 0 <= face < len(faces)
                                           for face in pair)
                    for pairs in pairs_by_case for pair in pairs),
            'source crossing witness faces')
    points, check_inverse = np.unique(vertices, axis=0, return_inverse=True)
    require(np.array_equal(inverse, check_inverse), 'source quotient replay')
    quotient_faces = inverse[faces]
    edges = np.unique(np.sort(np.concatenate((quotient_faces[:, [0, 1]],
                                               quotient_faces[:, [1, 2]],
                                               quotient_faces[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(points[edges[:, 0]]-points[edges[:, 1]], axis=1)
    graph = coo_matrix((np.r_[lengths, lengths],
                        (np.r_[edges[:, 0], edges[:, 1]],
                         np.r_[edges[:, 1], edges[:, 0]])),
                       shape=(len(points), len(points))).tocsr()
    support = np.unique(quotient_faces[support_faces])
    distance_to_crossing = dijkstra(graph, directed=False,
                                    indices=support, min_only=True)[inverse]
    require(bool(np.isfinite(distance_to_crossing).all()),
            'crossing support does not cover the connected skin')
    fraction = np.clip(distance_to_crossing/radius_m, 0, 1)
    locality = 1-(3*fraction*fraction-2*fraction*fraction*fraction)
    candidate = full+blend*locality[:, None]*(screened-full)
    require(bool(np.isfinite(candidate).all()) and candidate.min() >= 0
            and float(np.max(np.abs(candidate.sum(axis=1)-1))) < 1e-12,
            'derived candidate partition and positivity')
    first = np.unique(vertices, axis=0, return_index=True)[1]
    require(np.array_equal(candidate, candidate[first][inverse]),
            'exact source seam full-field equality')

    candidate_f4 = candidate.astype('<f4')
    bytes_out = bytearray(raw)
    changed_quartets = 0
    for vertex in range(nv):
        quartet = np.argsort(-candidate[vertex], kind='stable')[:4]
        selected = candidate[vertex, quartet]
        selected /= selected.sum()
        previous = struct.unpack_from('<4I4f', raw, vertex_start+56*vertex+24)
        if list(quartet) != list(previous[:4]):
            changed_quartets += 1
        struct.pack_into('<4I4f', bytes_out, vertex_start+56*vertex+24,
                         *map(int, quartet), *map(float, selected))
    bytes_out[full_start:full_start+4*nv*nb] = candidate_f4.tobytes()
    require(len(bytes_out) == len(raw)
            and bytes_out[:vertex_start] == raw[:vertex_start]
            and bytes_out[vertex_start+56*nv:full_start] == raw[vertex_start+56*nv:full_start]
            and all(bytes_out[vertex_start+56*i:vertex_start+56*i+24]
                    == raw[vertex_start+56*i:vertex_start+56*i+24]
                    for i in range(nv)),
            'source positions/normals/bindings/faces changed')
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir/'bodyparts3d-myosim-skinned-shell-geodesic-candidate.nhskin'
    _write_immutable(candidate_path, bytes(bytes_out))
    candidate_manifest = {
        'schema': SCHEMA,
        'payload': {
            'file': candidate_path.name, 'sha256': human.sha256(candidate_path),
            'payload_abi': 5, 'registration_fingerprint32': fingerprint,
            'binding_count': nb, 'vertex_count': nv,
            'triangle_count': ni//3, 'index_count': ni,
        },
        'source': {'skin': manifest['source']['skin']},
        'base_payload_sha256': human.sha256(payload),
        'base_manifest_sha256': human.sha256(manifest_path),
        'binding_solution_sha256': human.sha256(solution_path),
        'crossing_attribution_sha256': human.sha256(attribution_path),
        'method': 'source_graph_geodesic_localized_envelope_attenuation_blend',
        'blend_fraction': blend, 'source_geodesic_radius_m': radius_m,
        'support_face_ids': support_faces,
        'support_face_count': len(support_faces),
        'changed_vertex_count': int((locality > 0).sum()),
        'full_strength_vertex_count': int((locality == 1).sum()),
        'changed_diagnostic_quartet_count': changed_quartets,
        'maximum_full_weight_l1_change': float(np.max(np.abs(candidate-full).sum(axis=1))),
        'maximum_full_weight_float32_partition_error':
            float(np.max(np.abs(candidate_f4.sum(axis=1)-1))),
        'source_positions_normals_bindings_and_faces_byte_identical': True,
        'native_execution': False, 'physical_skin': False, 'clinical_anatomy': False,
        'boundary': ('Derived visual-only candidate. Offline exact sampled-pose checks '
                     'are separate from a native pack audit, continuous motion, skin '
                     'material/contact and clinical anatomy.'),
        'source_code_sha256': {name: human.sha256(Path(__file__).with_name(name))
                               for name in CODE_FILES},
    }
    manifest_file = output_dir/'candidate.manifest.json'
    _write_immutable(manifest_file,
                     (json.dumps(candidate_manifest, indent=2, sort_keys=True)+'\n').encode())
    return candidate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = human.REPOSITORY_ROOT
    parser.add_argument('--payload', type=Path, default=root/'Build/skin-seam-continuity-20260929/production/payload/bodyparts3d-myosim-skinned-shell.nhskin')
    parser.add_argument('--manifest', type=Path, default=root/'Build/skin-seam-continuity-20260929/production/payload/bodyparts3d-myosim-skinned-shell.manifest.json')
    parser.add_argument('--solution', type=Path, default=root/'Build/skin-seam-continuity-20260929/production/payload/bodyparts3d-skin-binding-solution.npz')
    parser.add_argument('--attribution', type=Path, default=root/'Build/skin-embeddedness-20260930/crossing-attribution.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--blend-fraction', type=float, default=BLEND)
    parser.add_argument('--source-geodesic-radius-m', type=float, default=RADIUS_M)
    a = parser.parse_args()
    result = compile_candidate(a.payload, a.manifest, a.solution,
                               a.attribution, a.output_dir,
                               blend=a.blend_fraction,
                               radius_m=a.source_geodesic_radius_m)
    print(json.dumps({'schema': result['schema'], 'payload_sha256': result['payload']['sha256'],
                      'changed_vertex_count': result['changed_vertex_count'],
                      'source_unchanged': result['source_positions_normals_bindings_and_faces_byte_identical']}))


if __name__ == '__main__':
    main()
