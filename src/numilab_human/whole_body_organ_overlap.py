"""Exact same-owner overlap diagnostics for selected whole-body organ surfaces.

This audits surface intersections between individually embedded source atlas
surfaces in their shared native owner frame. It does not decide whether two
source representations are independent organs, component/aggregate views, or
clinical tissue boundaries.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import itertools
import json
import os
from pathlib import Path

from . import abdominal_organ_separation as exact_audit
from . import lung_envelope as lung
from . import model as human
from . import surface_topology_repair as repair
from . import whole_body_embeddedness as indexed
from . import whole_body_surface_gate as selected_gate


SCHEMA = 'numi.human.whole-body-organ-overlap-diagnostic.v1'
ROOT = human.REPOSITORY_ROOT
CONTROL_RECEIPT_SHA256 = '7900a0341974317d90ef44c06a969b1f96d7ce2b57283bdfe89aa4d08db90990'
SOURCE_FILES = (
    'partof_element_parts.txt',
    'partof_inclusion_relation_list.txt',
    'isa_element_parts.txt',
    'isa_inclusion_relation_list.txt',
)
CODE_FILES = (
    'whole_body_organ_overlap.py',
    'abdominal_organ_separation.py',
    'cardiac_cavity_intersections.py',
    'compiled_quotient_embeddedness.py',
    'whole_body_embeddedness.py',
    'whole_body_surface_gate.py',
    'surface_topology_repair.py',
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise human.ImportError('whole-body organ overlap: ' + message)


def _progress(stage: str, **fields) -> None:
    print(json.dumps({'stage': stage, **fields}, sort_keys=True), flush=True)


class SourceHierarchy:
    """Pinned BodyParts3D concept memberships and parent-to-child graphs."""

    def __init__(self, mappings: dict[str, dict[str, dict[str, str]]],
                 parents: dict[str, dict[str, set[str]]],
                 concept_names: dict[str, dict[str, str]], hashes: dict[str, str]):
        self.mappings = mappings
        self.parents = parents
        self.concept_names = concept_names
        self.hashes = hashes
        self._ancestor_cache: dict[tuple[str, str], frozenset[str]] = {}

    @staticmethod
    def _rows(path: Path, columns: int, header: tuple[str, ...]):
        lines = path.read_text(encoding='utf-8').splitlines()
        require(bool(lines), f'empty hierarchy source {path.name}')
        require(tuple(lines[0].split('\t')) == header,
                f'hierarchy header drift in {path.name}')
        result = []
        for line_number, line in enumerate(lines[1:], 2):
            if not line:
                continue
            values = line.split('\t')
            require(len(values) == columns and all(values),
                    f'malformed hierarchy row {path.name}:{line_number}')
            result.append(values)
        return result

    @classmethod
    def load(cls, sources: Path, lock_path: Path) -> SourceHierarchy:
        sources, lock_path = sources.resolve(), lock_path.resolve()
        lock = human.read_json(lock_path)
        expected = lock['sources']['bodyparts3d_4']['files']
        mappings: dict[str, dict[str, dict[str, str]]] = {}
        parents: dict[str, dict[str, set[str]]] = {}
        names: dict[str, dict[str, str]] = {}
        hashes = {}
        definitions = {
            'part_of': ('partof_element_parts.txt', 'partof_inclusion_relation_list.txt'),
            'is_a': ('isa_element_parts.txt', 'isa_inclusion_relation_list.txt'),
        }
        for hierarchy, (mapping_name, edge_name) in definitions.items():
            for filename in (mapping_name, edge_name):
                path = sources / filename
                actual = human.sha256(path)
                require(actual == expected[filename]['sha256'],
                        f'pinned hierarchy source hash mismatch: {filename}')
                hashes[filename] = actual
            mapping_rows = cls._rows(
                sources / mapping_name, 3,
                ('concept id', 'name', 'element file id'),
            )
            edge_rows = cls._rows(
                sources / edge_name, 4,
                ('parent id', 'parent name', 'child id', 'child name'),
            )
            member_terms: dict[str, dict[str, str]] = defaultdict(dict)
            concept_names: dict[str, str] = {}
            for concept_id, name, member_id in mapping_rows:
                require(concept_id.startswith('FMA') and member_id.startswith('FJ'),
                        f'invalid concept/member identifier in {mapping_name}')
                require(concept_id not in concept_names or concept_names[concept_id] == name,
                        f'conflicting concept label for {concept_id} in {mapping_name}')
                concept_names[concept_id] = name
                member_terms[member_id][concept_id] = name
            parent_edges: dict[str, set[str]] = defaultdict(set)
            for parent_id, parent_name, child_id, child_name in edge_rows:
                require(parent_id.startswith('FMA') and child_id.startswith('FMA'),
                        f'invalid concept edge in {edge_name}')
                for concept_id, name in ((parent_id, parent_name), (child_id, child_name)):
                    require(concept_id not in concept_names or concept_names[concept_id] == name,
                            f'conflicting concept label for {concept_id} in {edge_name}')
                    concept_names[concept_id] = name
                parent_edges[child_id].add(parent_id)
            mappings[hierarchy] = dict(member_terms)
            parents[hierarchy] = dict(parent_edges)
            names[hierarchy] = concept_names
        return cls(mappings, parents, names, hashes)

    def ancestors(self, hierarchy: str, concept_id: str) -> frozenset[str]:
        key = (hierarchy, concept_id)
        if key not in self._ancestor_cache:
            visited: set[str] = set()
            pending = list(self.parents[hierarchy].get(concept_id, ()))
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(self.parents[hierarchy].get(current, ()))
            self._ancestor_cache[key] = frozenset(visited)
        return self._ancestor_cache[key]

    def relation_annotation(self, first_member: str, second_member: str) -> dict:
        result = {'same_source_member_id': first_member == second_member,
                  'shared_direct_concepts': {}, 'direct_concept_ancestry': []}
        ancestry = []
        for hierarchy in ('part_of', 'is_a'):
            first = self.mappings[hierarchy].get(first_member, {})
            second = self.mappings[hierarchy].get(second_member, {})
            shared = sorted(set(first) & set(second))
            result['shared_direct_concepts'][hierarchy] = [
                {'concept_id': concept, 'name': self.concept_names[hierarchy][concept]}
                for concept in shared
            ]
            for left in sorted(first):
                right_ancestors = self.ancestors(hierarchy, left)
                for right in sorted(second):
                    if right in right_ancestors:
                        ancestry.append({'hierarchy': hierarchy, 'ancestor_member': 'second',
                                         'ancestor_concept_id': right,
                                         'ancestor_name': self.concept_names[hierarchy][right],
                                         'descendant_member': 'first',
                                         'descendant_concept_id': left,
                                         'descendant_name': self.concept_names[hierarchy][left]})
                    if left in self.ancestors(hierarchy, right):
                        ancestry.append({'hierarchy': hierarchy, 'ancestor_member': 'first',
                                         'ancestor_concept_id': left,
                                         'ancestor_name': self.concept_names[hierarchy][left],
                                         'descendant_member': 'second',
                                         'descendant_concept_id': right,
                                         'descendant_name': self.concept_names[hierarchy][right]})
        result['direct_concept_ancestry'] = ancestry
        return result


def _surface_reference(row: dict, owner_names: list[str]) -> dict:
    body_index = row['source_body_index']
    source_id = row['source_stable_id'] if 'source_stable_id' in row else row['stable_id']
    return {
        'source_stable_id': source_id,
        'visible_stable_id': row.get('visible_stable_id', source_id),
        'source_member_id': row['source_member_id'],
        'body_index': body_index,
        'body_name': owner_names[body_index],
    }


def _bounds_disjoint(first_vertices, second_vertices) -> bool:
    import numpy as np

    first = np.asarray(first_vertices, dtype=np.float64)
    second = np.asarray(second_vertices, dtype=np.float64)
    return bool(np.any(first.max(axis=0) < second.min(axis=0))
                or np.any(second.max(axis=0) < first.min(axis=0)))


def _write_immutable(path: Path, value: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    if path.exists():
        require(path.read_bytes() == encoded,
                'diagnostic receipt is immutable; choose a new output path')
        return
    pending = path.with_name(path.name + f'.{os.getpid()}.pending')
    try:
        with pending.open('xb') as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def audit(candidate_payload: Path, selected_report: Path, indexed_census: Path,
          quotient_census: Path,
          candidate_audit: Path, selection: Path, sources: Path,
          source_lock: Path, body_manifest: Path, control_receipt: Path) -> dict:
    import numpy as np

    paths = [Path(path).resolve() for path in (
        candidate_payload, selected_report, indexed_census, quotient_census, candidate_audit,
        selection, sources, source_lock, body_manifest, control_receipt,
    )]
    (candidate_payload, selected_report, indexed_census, quotient_census, candidate_audit,
     selection, sources, source_lock, body_manifest, control_receipt) = paths
    gate = human.read_json(selected_report)
    require(gate['schema'] == selected_gate.SCHEMA
            and gate['baseline_payload_sha256'] == repair.BASE_SHA
            and gate['source_surface_count'] == 579
            and len(gate['rows']) == 579,
            'selected source view schema or completeness')
    require(human.sha256(candidate_payload) == gate['candidate_payload_sha256'],
            'selected compiled payload identity')
    require(human.sha256(candidate_audit) == gate['candidate_audit_sha256']
            and human.sha256(selection) == gate['selection_sha256'],
            'candidate audit or inspection selection receipt identity')
    candidate_proof = human.read_json(candidate_audit)
    choice = human.read_json(selection)
    require(candidate_proof['schema'] == 'numi.human.source-topology-repair-audit.v1'
            and candidate_proof['passed'] is True
            and candidate_proof['raw_579_geometry_byte_identical'] is True
            and candidate_proof['payload_sha256'] == gate['candidate_payload_sha256']
            and candidate_proof['candidate_count'] == 19
            and choice['source_payload_sha256'] == gate['candidate_payload_sha256']
            and choice['source_audit_sha256'] == gate['candidate_audit_sha256'],
            'source topology candidate chain identity')

    indexed_summary, indexed_identity, indexed_rows, indexed_summary_sha = selected_gate.census_rows(
        indexed_census, repair.BASE_SHA,
    )
    quotient_summary, quotient_identity, census, quotient_summary_sha = selected_gate.census_rows(
        quotient_census, repair.BASE_SHA,
    )
    require(indexed_summary_sha == gate['indexed_census_sha256']
            and quotient_summary_sha == gate['quotient_census_sha256']
            and indexed_identity['source_manifest_hashes']
                == quotient_identity['source_manifest_hashes']
            and all(a['geometry_sha256'] == b['geometry_sha256']
                    and a['source_body_index'] == b['source_body_index']
                    and a['source_layer_code'] == b['source_layer_code']
                    and a['source_member_id'] == b['source_member_id']
                    for a, b in zip(indexed_rows, census, strict=True)),
            'selected gate/indexed census summary identity')
    replayed = selected_gate.selected_surface_rows(census, candidate_proof['rows'], choice)
    require(replayed == gate['rows'], 'selected surface gate does not replay')

    _, header, records, vertices, indices = lung.decode(candidate_payload)
    require(header[0] == 5 and header[1] == 598 and len(records) == 598,
            'selected compiled anatomy ABI/count')
    core = human.read_json(body_manifest)['core_tree']
    owner_names = core['body_order']
    require(len(owner_names) == core['engine_body_count'], 'native owner index/name table')
    hierarchy = SourceHierarchy.load(sources, source_lock)

    candidate_by_source = {row['source_stable_id']: row for row in candidate_proof['rows']}
    meshes: dict[str, tuple[object, object]] = {}
    refs: dict[int, dict] = {}
    exclusions = []
    organ_rows = []
    for row, census_row in zip(gate['rows'], census, strict=True):
        sid = row['source_stable_id']
        visible_id = row['visible_stable_id']
        require(row['source_layer_code'] == census_row['source_layer_code']
                and row['source_member_id'] == census_row['source_member_id']
                and census_row['source_body_index'] < len(owner_names),
                f'source owner/layer identity at stable ID {sid}')
        if row['source_layer_code'] != 1:
            continue
        if not row['selected_closed_embedded_surface_candidate']:
            exclusions.append({**_surface_reference(census_row, owner_names),
                               'selected_status': row['selected_status']})
            continue
        record = list(map(int, records[visible_id-1]))
        owner, fv, nv, fi, ni, stable, layer, reserved = record
        require(stable == visible_id and owner == census_row['source_body_index']
                and layer == 1 and reserved == 0 and ni % 3 == 0
                and 0 <= fv < fv+nv <= len(vertices)
                and 0 <= fi < fi+ni <= len(indices),
                f'selected organ record identity at stable ID {sid}')
        geometry_hash = indexed.geometry_hash(vertices[fv:fv+nv], indices[fi:fi+ni])
        if visible_id == sid:
            require(geometry_hash == census_row['geometry_sha256'],
                    f'compiled source geometry differs from exact census at stable ID {sid}')
        else:
            candidate = candidate_by_source[sid]
            proof = candidate['executed_FP32_embedding']
            require(candidate['candidate_stable_id'] == visible_id
                    and candidate['member_id'] == row['source_member_id']
                    and candidate['passed_source_chain_and_topology'] is True
                    and proof['self_intersection_free'] is True
                    and proof['topology']['vertex_count'] == nv
                    and proof['topology']['face_count'] == ni//3,
                    f'repaired candidate topology identity at stable ID {sid}')
        faces = (indices[fi:fi+ni].reshape(-1, 3).astype(np.int64)-fv)
        require(bool(((faces >= 0) & (faces < nv)).all()),
                f'selected organ face escapes its mesh at stable ID {sid}')
        reference = _surface_reference(census_row, owner_names)
        reference.update({'triangle_count': ni//3,
                          'compiled_geometry_sha256': geometry_hash,
                          'source_hierarchy_terms': {
                              hierarchy_name: [
                                  {'concept_id': concept, 'name': name}
                                  for concept, name in sorted(
                                      hierarchy.mappings[hierarchy_name].get(
                                          row['source_member_id'], {}).items())
                              ]
                              for hierarchy_name in ('part_of', 'is_a')
                          }})
        refs[sid] = reference
        meshes[str(sid)] = (vertices[fv:fv+nv, :3].astype(np.float64), faces)
        organ_rows.append(reference)

    require(len([r for r in gate['rows'] if r['source_layer_code'] == 1]) == 120
            and len(organ_rows) == 112 and len(exclusions) == 8,
            'selected organ surface cohort or excluded defect count')
    _progress('source_surfaces_bound', selected=112, excluded_unqualified=8,
              triangles=sum(row['triangle_count'] for row in organ_rows))
    denominator, exact_meshes = exact_audit.exact_integer_meshes(meshes)
    _progress('exact_float32_meshes_prepared', denominator_bits=denominator.bit_length())
    by_owner: dict[int, list[int]] = defaultdict(list)
    for sid, ref in refs.items():
        by_owner[ref['body_index']].append(sid)

    disjoint_pairs = []
    disjoint_control_members = set()
    exact_pairs = []
    owner_counts = {}
    broad_pair_count = 0
    for owner, stable_ids in sorted(by_owner.items()):
        disjoint = 0
        tested = 0
        statuses: Counter[str] = Counter()
        for first_id, second_id in itertools.combinations(sorted(stable_ids), 2):
            broad_pair_count += 1
            first = refs[first_id]
            second = refs[second_id]
            a = meshes[str(first_id)][0]
            b = meshes[str(second_id)][0]
            if _bounds_disjoint(a, b):
                disjoint += 1
                disjoint_pairs.append({'body_index': owner,
                                       'first_stable_id': first_id,
                                       'second_stable_id': second_id,
                                       'first_member_id': first['source_member_id'],
                                       'second_member_id': second['source_member_id']})
                if owner == 7:
                    disjoint_control_members.add(frozenset((first['source_member_id'],
                                                            second['source_member_id'])))
                continue
            tested += 1
            result = exact_audit.audit_pair(exact_meshes[str(first_id)],
                                            exact_meshes[str(second_id)], denominator)
            result_row = {
                'body_index': owner,
                'first': first,
                'second': second,
                **result,
                'source_hierarchy_annotation': hierarchy.relation_annotation(
                    first['source_member_id'], second['source_member_id']),
            }
            statuses[result['status']] += 1
            exact_pairs.append(result_row)
            if len(exact_pairs) % 50 == 0:
                _progress('exact_pair_progress', tested=len(exact_pairs), total=481,
                          crossings=sum(v['status'] == 'surface_crossing' for v in exact_pairs))
        owner_counts[str(owner)] = {
            'body_name': owner_names[owner],
            'selected_individually_embedded_organ_surfaces': len(stable_ids),
            'possible_same_owner_pairs': len(stable_ids)*(len(stable_ids)-1)//2,
            'aabb_separated_pairs': disjoint,
            'exact_pair_tests': tested,
            'exact_pair_status_counts': dict(sorted(statuses.items())),
        }

    require(sum(value['possible_same_owner_pairs'] for value in owner_counts.values()) == 3178
            and len(disjoint_pairs) == 2697 and len(exact_pairs) == 481,
            'same-owner pair broad phase coverage changed')
    status_counts = Counter(row['status'] for row in exact_pairs)
    require(status_counts == {'surface_crossing': 198, 'separate_closed_domains': 283},
            'exact same-owner pair outcome counts changed')

    control = human.read_json(control_receipt)
    require(human.sha256(control_receipt) == CONTROL_RECEIPT_SHA256
            and control['schema'] == 'numi.human.compiled-abdominal-organ-separation.v2'
            and control['pair_count'] == 28,
            'published eight-organ control receipt identity')
    control_members = {row['label']: row['member_id'] for row in control['organs']}
    by_member_pair = {
        frozenset((row['first']['source_member_id'], row['second']['source_member_id'])): row
        for row in exact_pairs if row['body_index'] == 7
    }
    controls = []
    for expected in control['pairs']:
        key = frozenset((control_members[expected['first']], control_members[expected['second']]))
        actual = by_member_pair.get(key)
        if actual is None:
            require(key in disjoint_control_members,
                    'eight-organ control pair missing from whole-body census')
            actual_status = 'separate_closed_domains'
            actual_candidate_pairs = 0
            actual_intersections = 0
            actual_crossings = 0
            actual_point_contacts = 0
            actual_triangle_pairs = []
        else:
            actual_status = actual['status']
            actual_candidate_pairs = actual['aabb_candidate_triangle_pairs']
            actual_intersections = actual['exact_intersecting_triangle_pairs']
            actual_crossings = actual['exact_segment_or_polygon_crossing_pairs']
            actual_point_contacts = actual['exact_point_contact_pairs']
            actual_triangle_pairs = actual['first_intersecting_triangle_pairs']
        matched = (
            actual_status == expected['status']
            and actual_candidate_pairs == expected['aabb_candidate_triangle_pairs']
            and actual_intersections == expected['exact_intersecting_triangle_pairs']
            and actual_crossings == expected['exact_segment_or_polygon_crossing_pairs']
            and actual_point_contacts == expected['exact_point_contact_pairs']
            and actual_triangle_pairs == expected['first_intersecting_triangle_pairs']
        )
        require(matched, 'published eight-organ exact control pair changed')
        controls.append({'first': expected['first'], 'second': expected['second'],
                         'member_ids': sorted(key), 'matched': True,
                         'exact_intersecting_triangle_pairs': actual_intersections})

    crossing_rows = [row for row in exact_pairs if row['status'] == 'surface_crossing']
    maximum_segment = max(row['maximum_intersection_segment_m'] for row in crossing_rows)
    require(broad_pair_count == 3178, 'same-owner broad-phase enumeration incomplete')
    _progress('exact_pair_audit_complete', tested=len(exact_pairs),
              crossings=len(crossing_rows), disjoint=len(disjoint_pairs))
    source_hashes = {filename: human.sha256(sources/filename) for filename in SOURCE_FILES}
    return {
        'schema': SCHEMA,
        'status': 'completed_exact_same_owner_cross_surface_diagnostic',
        'source_payload_sha256': human.sha256(candidate_payload),
        'selection_report_sha256': human.sha256(selected_report),
        'selection_report_schema': gate['schema'],
        'candidate_audit_sha256': human.sha256(candidate_audit),
        'inspection_selection_sha256': human.sha256(selection),
        'indexed_census_summary_sha256': indexed_summary_sha,
        'indexed_census_identity_sha256': indexed_summary['identity_sha256'],
        'quotient_census_summary_sha256': quotient_summary_sha,
        'quotient_census_identity_sha256': quotient_summary['identity_sha256'],
        'source_hierarchy_sha256': source_hashes,
        'source_lock_sha256': human.sha256(source_lock),
        'native_body_manifest_sha256': human.sha256(body_manifest),
        'coordinate_semantics': 'exact_rational_values_of_selected_compiled_Float32_owner_local_metres',
        'same_owner_frame_basis': ('Only surfaces attached to the same native rigid owner are '
                                   'compared. A shared rigid body pose cannot alter their '
                                   'intersection relation.'),
        'selected_organ_surface_count': len(organ_rows),
        'selected_organ_triangle_count': sum(row['triangle_count'] for row in organ_rows),
        'excluded_individually_unqualified_organ_surfaces': exclusions,
        'owner_summary': owner_counts,
        'all_same_owner_surface_pairs': sum(v['possible_same_owner_pairs']
                                            for v in owner_counts.values()),
        'aabb_disjoint_pairs': len(disjoint_pairs),
        'aabb_disjoint_pair_rows': disjoint_pairs,
        'exact_pair_tests': len(exact_pairs),
        'exact_pair_status_counts': dict(sorted(status_counts.items())),
        'exact_pairs': exact_pairs,
        'surface_crossing_pair_count': len(crossing_rows),
        'exact_intersecting_triangle_pair_count': sum(
            row['exact_intersecting_triangle_pairs'] for row in crossing_rows),
        'maximum_intersection_segment_m': maximum_segment,
        'published_eight_organ_control': {
            'receipt_sha256': human.sha256(control_receipt),
            'matched_pair_count': len(controls), 'pair_count': len(control['pairs']),
            'pairs': controls,
        },
        'executed_source_sha256': {
            name: human.sha256(Path(__file__).with_name(name)) for name in CODE_FILES
        },
        'clinical_anatomy': False,
        'physical_volume': False,
        'organ_mechanics': False,
        'hierarchy_resolves_overlap_intent': False,
        'boundary': ('Exact crossings describe the selected atlas triangle surfaces only. '
                     'Source hierarchy terms are annotations, not proof that a crossing is '
                     'an aggregate/component overlap or an anatomical defect. The eight '
                     'individually unqualified organ surfaces are excluded. No surfaces were '
                     'moved, clipped, or capped; this does not establish clinical registration, '
                     'tissue boundaries, physical organ volumes, or mechanics.'),
    }


def command(arguments: argparse.Namespace) -> int:
    result = audit(arguments.candidate_payload, arguments.selected_report,
                   arguments.indexed_census, arguments.quotient_census,
                   arguments.candidate_audit,
                   arguments.selection, arguments.sources, arguments.source_lock,
                   arguments.body_manifest, arguments.control_receipt)
    _write_immutable(arguments.output, result)
    print(json.dumps({
        'status': result['status'],
        'selected_organ_surface_count': result['selected_organ_surface_count'],
        'all_same_owner_surface_pairs': result['all_same_owner_surface_pairs'],
        'aabb_disjoint_pairs': result['aabb_disjoint_pairs'],
        'exact_pair_tests': result['exact_pair_tests'],
        'exact_pair_status_counts': result['exact_pair_status_counts'],
        'control_pairs_matched': result['published_eight_organ_control']['matched_pair_count'],
        'output': str(arguments.output.resolve()),
    }, sort_keys=True), flush=True)
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    media = ROOT / 'Docs/media'
    topology = media / 'source-topology-repair-20260929'
    whole_body = media / 'whole-body-embeddedness-20260929'
    parser.add_argument('--candidate-payload', type=Path,
                        default=topology/'payload/source-topology-repair-candidates.nhanatomy')
    parser.add_argument('--selected-report', type=Path,
                        default=whole_body/'selected-surface-gate.json')
    parser.add_argument('--indexed-census', type=Path,
                        default=whole_body/'indexed-census.tar.gz')
    parser.add_argument('--quotient-census', type=Path,
                        default=whole_body/'quotient-census.tar.gz')
    parser.add_argument('--candidate-audit', type=Path, default=topology/'source-audit.json')
    parser.add_argument('--selection', type=Path,
                        default=topology/'inspection-selection.final.json')
    parser.add_argument('--sources', type=Path, default=ROOT/'Sources')
    parser.add_argument('--source-lock', type=Path, default=ROOT/'sources.lock.json')
    parser.add_argument('--body-manifest', type=Path,
                        default=ROOT/'Build/myosim-fullbody/myosim-fullbody-reference.manifest.json')
    parser.add_argument('--control-receipt', type=Path,
                        default=media/'abdominal-organ-separation-20260929/receipt-v2.json')
    parser.add_argument('--output', type=Path, required=True,
                        help='new immutable exact overlap diagnostic receipt')
    parser.set_defaults(handler=command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        return command(arguments)
    except (human.ImportError, OSError, KeyError, ValueError) as error:
        parser.exit(2, f'whole-body-organ-overlap: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
