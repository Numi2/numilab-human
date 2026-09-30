"""Retain and independently audit the complete FJ2810 source skin solid.

The visual NHSKIN payload intentionally contains only the outer sheet. This
offline geometry artifact retains both sheets and every source connector; it
does not assign skin mass, material, collision, deformation or clinical status.
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
from scipy.sparse.csgraph import connected_components

from . import model as human
from .skin_embeddedness_gate import _source_mesh
from .surface_topology_audit import exact_embedding
from .whole_body_embeddedness import atomic_json


SCHEMA = 'numi.human.full-source-skin-solid-geometry.v1'
MAGIC = b'NHSOLID1'
HEADER = struct.Struct('<8sIII32s32s')


def require(ok: bool, message: str) -> None:
    if not ok:
        raise human.ImportError('full source skin solid: ' + message)


def _immutable(path: Path, payload: bytes) -> None:
    require(not path.is_symlink(), f'output symlink {path}')
    if path.exists():
        require(path.read_bytes() == payload, f'output identity changed {path}')
        return
    temporary = path.with_name(path.name + f'.{os.getpid()}.pending')
    with temporary.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _source(sources: Path, visual_manifest: dict, registration: Path):
    archive, member, obj = human._bodyparts_obj_member(sources, 'is_a', 'FJ2810')
    pinned = visual_manifest['source']['skin']
    require(member == pinned['member']
            and human.sha256(archive) == pinned['archive_sha256']
            and hashlib.sha256(obj).hexdigest() == pinned['member_sha256'],
            'BodyParts3D source identity')
    vertices_mm, faces = human._bodyparts_obj_triangles(obj, member)
    vertices_mm = np.asarray(vertices_mm, dtype='<f8')
    faces = np.asarray(faces, dtype='<u4')
    require(vertices_mm.shape == (102467, 3) and faces.shape == (203382, 3),
            'complete source layout')
    frame = visual_manifest['coverage']['source_common_frame'][
        'global_source_mm_to_myosim_world_m']
    authored = human.read_json(registration)
    authored_frame = authored['coordinate_system']['global_source_mm_to_myosim_world_m']
    frame_gap = float(np.max(np.abs(np.asarray(authored_frame)-np.asarray(frame))))
    require(human.sha256(registration) ==
            visual_manifest['source']['registration']['sha256']
            and frame_gap <= 1e-9,
            'registered common source frame')
    return vertices_mm, faces, pinned, frame


def _signed_volume(vertices_m: np.ndarray, faces: np.ndarray) -> float:
    return float(np.einsum('ij,ij->', vertices_m[faces[:, 0]],
                          np.cross(vertices_m[faces[:, 1]],
                                   vertices_m[faces[:, 2]])) / 6)


def _index_components(vertex_count: int, faces: np.ndarray) -> int:
    a = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    b = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
    graph = coo_matrix((np.ones(len(a), dtype=np.uint8), (a, b)),
                       shape=(vertex_count, vertex_count)).tocsr()
    return int(connected_components(graph, directed=False)[0])


def build(sources: Path, visual_payload: Path, visual_manifest_path: Path,
          registration: Path, output_dir: Path) -> dict:
    sources, visual_payload, visual_manifest_path, registration, output_dir = (
        Path(p).resolve() for p in (sources, visual_payload,
                                    visual_manifest_path, registration,
                                    output_dir))
    manifest = human.read_json(visual_manifest_path)
    vertices_mm, faces, pinned, frame = _source(sources, manifest, registration)
    require(manifest['payload']['sha256'] == human.sha256(visual_payload),
            'visual shell identity')
    _, _, _, _, outer, outer_faces = _source_mesh(visual_payload, manifest)
    full = (vertices_mm * 0.001).astype('<f4')
    quotient, inverse = np.unique(full, axis=0, return_inverse=True)
    quotient_faces = inverse[faces].astype('<u4')
    require(quotient.shape == (101691, 3)
            and quotient_faces.shape == (203382, 3),
            'exact compiled-coordinate quotient')
    lookup = {tuple(point): i for i, point in enumerate(quotient)}
    outer_map = np.fromiter((lookup[tuple(point)] for point in outer),
                            dtype=np.uint32, count=len(outer))
    selected = {tuple(face) for face in quotient_faces.tolist()}
    require(all(tuple(face) in selected for face in outer_map[outer_faces]),
            'visual outer triangles are not a subset of the complete source')
    index_components = _index_components(len(vertices_mm), faces)
    require(index_components == 100, 'source-index component structure')
    source_member_hash = bytes.fromhex(pinned['member_sha256'])
    registration_hash = bytes.fromhex(human.sha256(registration))
    payload = (HEADER.pack(MAGIC, 1, len(quotient), len(quotient_faces),
                           source_member_hash, registration_hash)
               + quotient.tobytes() + quotient_faces.tobytes())
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir/'bodyparts3d-full-source-skin-solid.nhsolid'
    _immutable(artifact, payload)
    result = {
        'schema': SCHEMA, 'status': 'unqualified_source_geometry_candidate',
        'artifact_file': artifact.name, 'artifact_sha256': human.sha256(artifact),
        'artifact_abi': 1, 'source_member_id': 'FJ2810',
        'source_member_sha256': pinned['member_sha256'],
        'source_archive_sha256': pinned['archive_sha256'],
        'registration_sha256': human.sha256(registration),
        'visual_payload_sha256': human.sha256(visual_payload),
        'visual_manifest_sha256': human.sha256(visual_manifest_path),
        'source_vertex_count': len(vertices_mm),
        'source_triangle_count': len(faces),
        'source_index_component_count': index_components,
        'compiled_quotient_vertex_count': len(quotient),
        'compiled_triangle_count': len(quotient_faces),
        'visual_outer_triangle_subset_count': len(outer_faces),
        'source_local_units': 'm',
        'source_mm_to_myosim_world_m': frame,
        'candidate_geometry_modified_from_source': False,
        'source_skin_mass_material_contact_or_motion': False,
        'clinical_anatomy': False,
        'producer_source_sha256': human.sha256(Path(__file__)),
        'boundary': ('Exact complete FJ2810 source surface, coordinate-identified '
                     'and Float32 quantized. This is offline source geometry; the '
                     'NHSKIN visual shell and native/physical runtime are unchanged.'),
    }
    _immutable(output_dir/'manifest.json',
               (json.dumps(result, indent=2, sort_keys=True)+'\n').encode())
    return result


def audit(sources: Path, visual_payload: Path, visual_manifest_path: Path,
          registration: Path, output_dir: Path) -> dict:
    sources, visual_payload, visual_manifest_path, registration, output_dir = (
        Path(p).resolve() for p in (sources, visual_payload,
                                    visual_manifest_path, registration,
                                    output_dir))
    manifest_path = output_dir/'manifest.json'
    manifest = human.read_json(manifest_path)
    visual_manifest = human.read_json(visual_manifest_path)
    vertices_mm, source_faces, pinned, frame = _source(sources, visual_manifest,
                                                      registration)
    artifact = output_dir/manifest['artifact_file']
    raw = artifact.read_bytes()
    magic, abi, nv, nf, member_hash, registration_hash = HEADER.unpack_from(raw)
    require(magic == MAGIC and abi == 1 and nv == 101691 and nf == 203382
            and member_hash.hex() == pinned['member_sha256']
            and registration_hash.hex() == human.sha256(registration)
            and human.sha256(artifact) == manifest['artifact_sha256']
            and manifest['producer_source_sha256'] == human.sha256(Path(__file__))
            and manifest['visual_payload_sha256'] == human.sha256(visual_payload)
            and manifest['visual_manifest_sha256'] == human.sha256(visual_manifest_path)
            and manifest['source_mm_to_myosim_world_m'] == frame
            and len(raw) == HEADER.size+12*nv+12*nf,
            'artifact and source identities')
    compiled = np.frombuffer(raw, dtype='<f4', count=nv*3,
                             offset=HEADER.size).reshape(-1, 3)
    compiled_faces = np.frombuffer(raw, dtype='<u4', count=nf*3,
                                   offset=HEADER.size+12*nv).reshape(-1, 3)
    expected, inverse = np.unique((vertices_mm*0.001).astype('<f4'),
                                  axis=0, return_inverse=True)
    require(np.array_equal(compiled, expected)
            and np.array_equal(compiled_faces, inverse[source_faces]),
            'source-only exact face and vertex ancestry')
    raw_points, raw_inverse = np.unique(vertices_mm, axis=0,
                                        return_inverse=True)
    raw_faces = raw_inverse[source_faces]
    source_check = exact_embedding(raw_points.tolist(), raw_faces.tolist())
    compiled_check = exact_embedding(compiled.tolist(), compiled_faces.tolist())
    require(source_check['count'] == compiled_check['count'] == 0
            and source_check['topology']['closed_oriented_manifold_candidate']
            and compiled_check['topology']['closed_oriented_manifold_candidate']
            and source_check['topology']['face_component_count'] == 1
            and compiled_check['topology']['face_component_count'] == 1,
            'closed exact embedding of both source and Float32 candidate')
    local_volume = _signed_volume(compiled.astype(np.float64), compiled_faces)
    source_volume = _signed_volume(raw_points*0.001, raw_faces)
    frame_det = float(np.linalg.det(np.asarray(frame)[:3, :3]))
    world_volume = source_volume * frame_det / 1e-9
    require(source_volume > 0 and local_volume > 0 and frame_det > 0
            and abs(local_volume-source_volume) < 1e-7,
            'oriented finite source skin solid volume')
    result = {
        'schema': SCHEMA+'-audit',
        'status': 'passed_exact_source_and_float32_geometric_skin_solid',
        'artifact_sha256': human.sha256(artifact),
        'source_member_sha256': pinned['member_sha256'],
        'visual_payload_sha256': human.sha256(visual_payload),
        'source_exact': {
            'vertex_count': len(raw_points), 'triangle_count': len(raw_faces),
            'self_intersection_pair_count': source_check['count'],
            'aabb_candidate_pair_count': source_check['aabb_candidate_pairs'],
            'topology': source_check['topology'],
            'signed_local_volume_m3': source_volume,
        },
        'compiled_float32_exact': {
            'vertex_count': len(compiled), 'triangle_count': len(compiled_faces),
            'self_intersection_pair_count': compiled_check['count'],
            'aabb_candidate_pair_count': compiled_check['aabb_candidate_pairs'],
            'topology': compiled_check['topology'],
            'signed_local_volume_m3': local_volume,
        },
        'registration_determinant_m3_per_mm3': frame_det,
        'registered_rest_volume_m3': world_volume,
        'physical_skin_volume': False, 'skin_material_or_contact': False,
        'deformation_or_native_execution': False, 'clinical_anatomy': False,
        'predicate_source_sha256': {
            name: human.sha256(Path(__file__).with_name(name))
            for name in ('skin_full_source_solid.py', 'surface_topology_audit.py',
                         'cardiac_cavity_intersections.py',
                         'cardiac_cavity_geometry.py')},
        'boundary': ('Closed exact source geometry only. No native full-solid '
                     'deformation, material, mass, collision, self-contact, '
                     'calibration, or clinical qualification.'),
    }
    atomic_json(output_dir/'audit.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = human.REPOSITORY_ROOT
    payload_root = root/'Build/skin-seam-continuity-20260929/production/payload'
    parser.add_argument('--sources', type=Path, default=root/'Sources')
    parser.add_argument('--visual-payload', type=Path,
                        default=payload_root/'bodyparts3d-myosim-skinned-shell.nhskin')
    parser.add_argument('--visual-manifest', type=Path,
                        default=payload_root/'bodyparts3d-myosim-skinned-shell.manifest.json')
    parser.add_argument('--registration', type=Path,
                        default=root/'Build/knee-parity-registration-20260929/candidate.v6.registration.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    built = build(args.sources, args.visual_payload, args.visual_manifest,
                  args.registration, args.output_dir)
    checked = audit(args.sources, args.visual_payload, args.visual_manifest,
                    args.registration, args.output_dir)
    print(json.dumps({'status': checked['status'],
                      'artifact_sha256': built['artifact_sha256'],
                      'source_volume_m3': checked['source_exact']['signed_local_volume_m3'],
                      'registered_rest_volume_m3': checked['registered_rest_volume_m3']}),
          flush=True)


if __name__ == '__main__':
    main()
