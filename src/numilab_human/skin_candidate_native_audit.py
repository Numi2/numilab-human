"""Exact-coordinate-quotient self-intersection audit for a native skin pack."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from . import model as human
from .compiled_quotient_embeddedness import coordinate_quotient
from .skin_embeddedness_gate import _source_mesh
from .surface_topology_audit import exact_embedding
from .torso_anatomy_audit import _pack_sections


SCHEMA = 'numi.human.skin-candidate-native-embeddedness.v1'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError('skin candidate native audit: ' + message)


def audit(payload: Path, manifest_path: Path, native_pack: Path,
          condition: str, output_path: Path, base_payload: Path | None = None) -> dict:
    payload, manifest_path, native_pack, output_path = (
        Path(item).resolve() for item in
        (payload, manifest_path, native_pack, output_path))
    require(all(path.is_file() and not path.is_symlink() for path in
                (payload, manifest_path, native_pack)), 'retained input missing')
    manifest = human.read_json(manifest_path)
    source_bytes_identical = None
    if base_payload is not None:
        base_payload = Path(base_payload).resolve()
        require(base_payload.is_file() and not base_payload.is_symlink()
                and manifest.get('base_payload_sha256') == human.sha256(base_payload),
                'candidate base payload identity')
        base_raw, candidate_raw = base_payload.read_bytes(), payload.read_bytes()
        _, _, base_nb, base_nv, base_ni, _, _ = struct.unpack_from(
            '<8s5I32s', base_raw)
        _, _, cand_nb, cand_nv, cand_ni, _, _ = struct.unpack_from(
            '<8s5I32s', candidate_raw)
        require((base_nb, base_nv, base_ni) == (cand_nb, cand_nv, cand_ni),
                'candidate payload source layout')
        vertex_start = 60+36*base_nb
        full_start = vertex_start+56*base_nv+4*base_ni
        source_bytes_identical = (
            candidate_raw[:vertex_start] == base_raw[:vertex_start]
            and candidate_raw[vertex_start+56*base_nv:full_start]
                == base_raw[vertex_start+56*base_nv:full_start]
            and all(candidate_raw[vertex_start+56*i:vertex_start+56*i+24]
                    == base_raw[vertex_start+56*i:vertex_start+56*i+24]
                    for i in range(base_nv)))
        require(source_bytes_identical,
                'candidate changed registered positions, normals, bindings, or faces')
    _, nv, ni, _, _, source_faces = _source_mesh(payload, manifest)
    sections = _pack_sections(native_pack)
    require(all(kind in sections and sections[kind][2] == stride
                and len(sections[kind][0]) == sections[kind][1]*stride
                for kind, stride in ((2, 80), (3, 4), (4, 64), (5, 80))),
            'native pack layout')
    vertices = np.frombuffer(sections[2][0], '<f4').reshape(-1, 20)
    indices = np.frombuffer(sections[3][0], '<u4')
    primitives = np.frombuffer(sections[4][0], '<u4').reshape(-1, 16)
    selected = primitives[primitives[:, 4] == 51007]
    require(len(selected) == 1, 'one native skin primitive')
    first, count, _, _ = map(int, selected[0, :4])
    require(count == ni and first+count <= len(indices)
            and int(selected[0, 5]) == 1
            and int(selected[0, 6]) == 0xffffffff,
            'skin source primitive identity')
    emitted = indices[first:first+count]
    base = int(emitted.min())
    require(base+nv <= len(vertices)
            and np.array_equal(emitted-base, source_faces.ravel()),
            'skin source face-order parity')
    positions = vertices[base:base+nv, :3].astype(np.float64)
    qvertices, qfaces, mapping = coordinate_quotient(
        positions.tolist(), source_faces.tolist())
    checked = exact_embedding(qvertices, qfaces)
    geometry_sha = hashlib.sha256(
        positions.astype('<f4').tobytes()+source_faces.astype('<u4').tobytes()
    ).hexdigest()
    result = {
        'schema': SCHEMA, 'condition': condition,
        'payload_sha256': human.sha256(payload),
        'payload_manifest_sha256': human.sha256(manifest_path),
        'native_pack_sha256': human.sha256(native_pack),
        'geometry_sha256': geometry_sha,
        'source_vertex_count': nv, 'triangle_count': ni//3,
        'quotient_vertex_count': len(qvertices),
        'source_face_order_preserved': True,
        'exact_intersection_pairs': checked['count'],
        'triangle_pairs': checked['triangle_pairs'],
        'aabb_candidate_pairs': checked['aabb_candidate_pairs'],
        'topology': checked['topology'],
        'payload_source_positions_normals_bindings_and_faces_byte_identical':
            source_bytes_identical,
        'predicate_source_sha256': {
            name: human.sha256(Path(__file__).with_name(name))
            for name in ('skin_candidate_native_audit.py',
                         'compiled_quotient_embeddedness.py',
                         'surface_topology_audit.py',
                         'skin_embeddedness_gate.py')},
        'physical_skin_volume': False, 'skin_mechanics': False,
        'clinical_anatomy': False,
        'boundary': ('Exact self-intersection predicate after quotienting only '
                     'bit-identical coordinates, for this retained native snapshot. '
                     'No continuous-motion, mechanics, or clinical qualification.'),
    }
    # Keep the coordinate-quotient ancestry computation live as a layout check.
    require(len(mapping) == nv and checked['topology']['face_count'] == ni//3,
            'complete quotient and source face coverage')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, indent=2, sort_keys=True)+'\n').encode()
    require(not output_path.is_symlink(), 'output symlink')
    if output_path.exists():
        require(output_path.read_bytes() == encoded, 'output identity differs')
    else:
        temporary = output_path.with_name(output_path.name+f'.{os.getpid()}.pending')
        with temporary.open('xb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--native-pack', type=Path, required=True)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-payload', type=Path)
    args = parser.parse_args()
    result = audit(args.payload, args.manifest, args.native_pack,
                   args.condition, args.output, args.base_payload)
    print(json.dumps({'condition': result['condition'],
                      'exact_intersection_pairs': result['exact_intersection_pairs'],
                      'geometry_sha256': result['geometry_sha256']}))


if __name__ == '__main__':
    main()
