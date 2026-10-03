"""Build complete source-face support from retained ABI5 native skin packs.

The exact pairs are sampled-geometry evidence, not a skin mechanics or anatomy
qualification. Duplicate bilateral captures are both bound to the receipt even
when their byte-identical packed geometry lets the exact predicate be reused.
"""
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


SCHEMA = 'numi.human.native-skin-crossing-attribution.v1'
CONDITIONS = ('skin-flexion-right', 'skin-flexion-left',
              'skin-rest-right', 'skin-rest-left')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError('skin crossing support: ' + message)


def _write_immutable(path: Path, content: bytes) -> None:
    require(not path.is_symlink(), f'output symlink {path.name}')
    if path.exists():
        require(path.read_bytes() == content, f'output identity differs {path.name}')
        return
    temporary = path.with_name(path.name + f'.{os.getpid()}.pending')
    with temporary.open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_support(payload: Path, manifest_path: Path, solution_path: Path,
                  plan_path: Path, output_path: Path) -> dict:
    payload, manifest_path, solution_path, plan_path, output_path = (
        Path(item).resolve() for item in
        (payload, manifest_path, solution_path, plan_path, output_path))
    require(all(path.is_file() and not path.is_symlink() for path in
                (payload, manifest_path, solution_path, plan_path)),
            'retained input missing')
    manifest = human.read_json(manifest_path)
    plan = human.read_json(plan_path)
    require(plan.get('schema') ==
            'numi.human.skin-abi5-native-packed-surface-intersection-plan.v1'
            and plan.get('status') == 'preregistered_amendment',
            'registered ABI5 native pack plan')
    fixed = plan['fixed_inputs']
    require(fixed['payload']['sha256'] == human.sha256(payload)
            and fixed['manifest']['sha256'] == human.sha256(manifest_path),
            'plan/source identity')
    _, nv, ni, _, _, source_faces = _source_mesh(payload, manifest)
    conditions = {item['id']: item for item in plan['conditions']}
    require(set(conditions) == set(CONDITIONS), 'four planned rest/flexion conditions')
    rows_root = plan_path.parent/'embeddedness'/'rows'
    rows = {}
    geometry_cache = {}
    for condition_id in CONDITIONS:
        condition = conditions[condition_id]
        pack_ref = condition['native_pack']
        pack = (plan_path.parent/pack_ref['path'].split(
            'Docs/media/skin-abi5-flexion-visual-20261003/', 1)[-1]
                if pack_ref['path'].startswith('Docs/media/skin-abi5-flexion-visual-20261003/')
                else Path(pack_ref['path']))
        pack = pack.resolve()
        require(pack.is_file() and not pack.is_symlink()
                and human.sha256(pack) == pack_ref['sha256'],
                f'planned native pack identity: {condition_id}')
        sections = _pack_sections(pack)
        require(all(kind in sections and sections[kind][2] == stride
                    and len(sections[kind][0]) == sections[kind][1]*stride
                    for kind, stride in ((2, 80), (3, 4), (4, 64), (5, 80))),
                f'native pack section layout: {condition_id}')
        packed_vertices = np.frombuffer(sections[2][0], '<f4').reshape(-1, 20)
        packed_indices = np.frombuffer(sections[3][0], '<u4')
        primitives = np.frombuffer(sections[4][0], '<u4').reshape(-1, 16)
        selected = primitives[primitives[:, 4] == 51007]
        require(len(selected) == 1, f'one skin primitive: {condition_id}')
        first, count, _, _ = map(int, selected[0, :4])
        require(count == ni and first+count <= len(packed_indices)
                and int(selected[0, 5]) == 1
                and int(selected[0, 6]) == 0xffffffff,
                f'skin primitive range/owner: {condition_id}')
        emitted_faces = packed_indices[first:first+count]
        base = int(emitted_faces.min())
        require(base+nv <= len(packed_vertices)
                and np.array_equal(emitted_faces-base, source_faces.ravel()),
                f'source face-order parity: {condition_id}')
        positions = packed_vertices[base:base+nv, :3].astype(np.float64)
        geometry_sha = hashlib.sha256(
            positions.astype('<f4').tobytes()+source_faces.astype('<u4').tobytes()
        ).hexdigest()
        reused = geometry_sha in geometry_cache
        if not reused:
            quotient_vertices, quotient_faces, _ = coordinate_quotient(
                positions.tolist(), source_faces.tolist())
            geometry_cache[geometry_sha] = exact_embedding(
                quotient_vertices, quotient_faces)['triangle_pairs']
        pairs = geometry_cache[geometry_sha]
        row_path = rows_root/(condition_id+'.json')
        row = human.read_json(row_path)
        pose_q = condition['pose_q']
        require(row['condition'] == condition_id
                and row['pose_q'] == pose_q
                and row['payload_sha256'] == human.sha256(payload)
                and row['native_pack_sha256'] == pack_ref['sha256']
                and row['geometry_sha256'] == geometry_sha
                and row['exact_intersection_pairs'] == len(pairs),
                f'prior exact audit parity: {condition_id}')
        rows[condition_id] = {
            'case': condition_id, 'pose_q': pose_q,
            'native_pack_sha256': pack_ref['sha256'],
            'geometry_sha256': geometry_sha,
            'exact_intersection_pair_count': len(pairs),
            'exact_intersection_pairs': pairs,
            'audit_reused_for_identical_geometry': reused
        }
    result = {
        'schema': SCHEMA,
        'total_exact_intersection_pairs': sum(
            row['exact_intersection_pair_count'] for row in rows.values()),
        'source': {
            'payload_sha256': human.sha256(payload),
            'manifest_sha256': human.sha256(manifest_path),
            'binding_solution_sha256': human.sha256(solution_path),
        },
        'plan_sha256': human.sha256(plan_path),
        'exact_predicate_source_sha256': human.sha256(Path(__file__).with_name(
            'surface_topology_audit.py')),
        'conditions': len(rows),
        'cases': list(rows.values()),
        'boundary': ('Complete exact triangle pairs from the retained ABI5 outer-shell '
                     'snapshots. No continuous-motion, skin-mechanics, clinical-anatomy, '
                     'or whole-Human qualification.'),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output_path,
                     (json.dumps(result, indent=2, sort_keys=True)+'\n').encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--solution', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build_support(args.payload, args.manifest, args.solution,
                           args.plan, args.output)
    print(json.dumps({'schema': result['schema'],
                      'total_exact_intersection_pairs':
                          result['total_exact_intersection_pairs'],
                      'conditions': {row['case']: row['exact_intersection_pair_count']
                                     for row in result['cases']}}))


if __name__ == '__main__':
    main()
