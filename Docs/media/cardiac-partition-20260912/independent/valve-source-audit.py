#!/usr/bin/env python3
"""Reproduce the pinned atlas tricuspid leaflet/interface inventory.

This is static source inspection, not physical stepping or valve calibration.
Source-byte verification and exact coincident-coordinate topology precede the
inventory. Missing atlas definitions do not establish absence of anatomy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
from numilab_human.cardiac_cavity_geometry import (
    ARCHIVE, ARCHIVE_SHA256, ARCHIVE_BYTES, TABLE_HASHES,
    analyze_topology, exact_coordinate_quotient, extract_cavity_surfaces, parse_obj,
)
from numilab_human.model import ImportError as HumanImportError, _bodyparts_obj_member
from numilab_human.physiology import canonical, load_anatomy, read_json

LABELS = {
    'part_of': ('partof_parts_list_e.txt', '9224080557053e6f1322f1e13ab27f0ecde0db19bb3b505f0631afad230eeebd'),
    'is_a': ('isa_parts_list_e.txt', 'ab7796deedd49205e77f3609a1cb8c53e2bbee14ecb5c9a6ca05227469780513'),
}
LEAFLETS = (
    ('FMA7238', 'anterior leaflet of tricuspid valve', 'FJ2421', 'ab4d65a190910effcffa60cb5522359756504edb45b9dd312971083cb70c3af7'),
    ('FMA7239', 'posterior leaflet of tricuspid valve', 'FJ2433', '736c0689a5489394df568e2788d27e1a547d8fcbc464490e7bef94ba9c66e8e8'),
    ('FMA7240', 'septal leaflet of tricuspid valve', 'FJ2436', 'd2d152ec63544d37077ede2c72f2e8ebd80eb98537386ab917c6a19788f8ed98'),
)
QUERIES = (
    ('tricuspid_and_right_av', r'tricuspid|right (?:atrioventricular|atrio-ventricular)'),
    ('annulus_orifice_and_ostium', r'annul|anulus|orifice|ostium'),
    ('fibrous_skeleton_and_rings', r'fibrous (?:skeleton of heart|ring)'),
)


def require(ok, message):
    if not ok:
        raise HumanImportError('valve source audit: ' + message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def summary(topology):
    keys = ('vertex_count', 'face_count', 'edge_count', 'boundary_edge_count',
            'boundary_loop_count', 'face_component_count', 'euler_characteristic',
            'closed_oriented_manifold_candidate', 'self_intersection_status',
            'interdomain_overlap_status')
    out = {key: topology[key] for key in keys}
    for key in ('unused_vertex_ids', 'boundary_branch_vertex_ids', 'nonmanifold_edges',
                'orientation_defect_edges', 'degenerate_face_ids', 'repeated_vertex_face_ids',
                'duplicate_face_ids', 'vertex_manifold_defect_ids'):
        out[key + '_count'] = len(topology[key])
    out['boundary_loop_vertex_counts'] = sorted(len(loop) for loop in topology['boundary_loops'])
    out['component_face_counts'] = sorted(len(component) for component in topology['face_components'])
    return out


def audit(*, sources=ROOT / 'Sources', source_lock=ROOT / 'sources.lock.json'):
    # The permanent extractor verifies the pinned 4.0 archive and both complete
    # membership tables before any additional valve interpretation is attempted.
    cavities = extract_cavity_surfaces(sources=sources, source_lock=source_lock)
    anatomy = load_anatomy(sources, source_lock)
    lock = read_json(source_lock)['sources']['bodyparts3d_4']
    expected_files = {ARCHIVE: ARCHIVE_SHA256, **TABLE_HASHES,
                      **{filename: expected for filename, expected in LABELS.values()}}
    before = {}
    for filename, expected in sorted(expected_files.items()):
        path = sources / filename
        require(path.is_file() and not path.is_symlink(), 'missing or redirected source file: ' + filename)
        require(lock['files'][filename]['sha256'] == expected, 'source lock differs: ' + filename)
        before[filename] = sha(path)
        require(before[filename] == expected, 'source bytes differ: ' + filename)
    label_rows = {}
    for hierarchy, (filename, _) in LABELS.items():
        lines = (sources / filename).read_text(encoding='utf-8').splitlines()
        require(lines[0] == 'concept id\trepresentation id\ten', 'label header differs')
        rows = []
        for line in lines[1:]:
            fields = line.split('\t')
            require(len(fields) == 3 and re.fullmatch(r'FMA[0-9]+', fields[0])
                    and re.fullmatch(r'BP[0-9]+', fields[1]) and fields[2], 'malformed label row')
            rows.append(dict(zip(('concept_id', 'representation_id', 'source_name'), fields)))
        label_rows[hierarchy] = rows
    expected_members = {row[2] for row in LEAFLETS}
    compound_queries = (('part_of', 'FMA7234', 'tricuspid valve'),
                        ('is_a', 'FMA7237', 'leaflet of tricuspid valve'))
    compounds = []
    for hierarchy, concept, name in compound_queries:
        members = anatomy['tables'][hierarchy].get((concept, name))
        require(members == expected_members, 'tricuspid compound membership differs')
        compounds.append({'hierarchy': hierarchy, 'concept_id': concept, 'source_name': name,
                          'member_ids': sorted(members)})
    leaflets = []
    for concept, name, member_id, expected_sha in LEAFLETS:
        memberships = []
        for hierarchy in LABELS:
            require(anatomy['tables'][hierarchy].get((concept, name)) == {member_id},
                    'leaflet membership differs')
            memberships.append({'hierarchy': hierarchy, 'concept_id': concept,
                                'source_name': name, 'member_ids': [member_id]})
        archive, member, data = _bodyparts_obj_member(sources, 'part_of', member_id)
        require(archive.name == ARCHIVE and member == f'partof_BP3D_4.0_obj_99/{member_id}.obj',
                'unexpected leaflet member path')
        require(hashlib.sha256(data).hexdigest() == expected_sha, 'leaflet member bytes differ')
        parsed = parse_obj(data, member)
        require(any(line.startswith('# Bounds(mm):') for line in parsed['comments']), 'missing source millimetre convention')
        raw = analyze_topology(parsed['vertices_mm'], parsed['triangles'])
        quotient = exact_coordinate_quotient(parsed)
        vertices = parsed['vertices_mm']
        leaflets.append({'concept_id': concept, 'semantic_id': 'FMA:' + concept[3:],
                        'source_name': name, 'member_id': member_id, 'memberships': memberships,
                        'source': {'archive': ARCHIVE, 'member': member, 'sha256': expected_sha,
                                   'bytes': len(data), 'header_comments': parsed['comments']},
                        'bounds_source_mm': {'minimum': [min(p[k] for p in vertices) for k in range(3)],
                                             'maximum': [max(p[k] for p in vertices) for k in range(3)]},
                        'raw_topology': summary(raw),
                        'exact_coordinate_quotient': {'method': quotient['method'],
                            'identified_vertex_count': quotient['identified_vertex_count'],
                            'topology': summary(quotient['topology'])},
                        'source_annular_curve': None, 'source_partition_surface': None})
    queries = [{'id': key, 'case_insensitive_regular_expression': pattern,
                'matching_labels': {hierarchy: sorted((row for row in rows if re.search(pattern, row['source_name'], re.I)),
                                                      key=lambda row: (row['concept_id'], row['source_name']))
                                    for hierarchy, rows in label_rows.items()}}
               for key, pattern in QUERIES]
    interface_candidates = []
    for hierarchy, rows in label_rows.items():
        for row in rows:
            name = row['source_name'].lower()
            is_right_av = 'tricuspid' in name or bool(re.search(r'right (?:atrioventricular|atrio-ventricular)', name))
            if is_right_av and re.search(r'annul|anulus|orifice|ostium|ring', name):
                interface_candidates.append({'hierarchy': hierarchy, **row})
    fibrous = []
    for concept, name in (('FMA9496', 'fibrous skeleton of heart'), ('FMA9498', 'fibrous ring of mitral valve')):
        members = anatomy['tables']['part_of'].get((concept, name))
        require(members is not None, 'fibrous source query is unresolved')
        fibrous.append({'hierarchy': 'part_of', 'concept_id': concept, 'source_name': name,
                        'member_ids': sorted(members)})
    after = {filename: sha(sources / filename) for filename in before}
    require(before == after, 'source bytes changed during audit')
    owners = ('src/numilab_human/cardiac_cavity_geometry.py', 'src/numilab_human/physiology.py',
              'src/numilab_human/model.py')
    return {'schema': 'HumanPack.tricuspid-source-interface-audit.v1',
            'anatomy_source': anatomy['source'], 'source_coordinate_frame': 'BodyParts3D 4.0 source millimetres',
            'archive': {'file': ARCHIVE, 'sha256': ARCHIVE_SHA256, 'bytes': ARCHIVE_BYTES},
            'source_files_sha256_before': before, 'source_files_sha256_after': after,
            'source_bytes_unchanged': True, 'source_lock_sha256': sha(source_lock),
            'audit_script_sha256': sha(Path(__file__)), 'owner_source_sha256': {path: sha(ROOT / path) for path in owners},
            'tricuspid_compounds': compounds, 'leaflets': leaflets, 'source_label_queries': queries,
            'right_av_annulus_orifice_candidate_labels': interface_candidates,
            'fibrous_membership_queries': fibrous,
            'fibrous_skeleton_matches_mitral_ring_membership': fibrous[0]['member_ids'] == fibrous[1]['member_ids'],
            'source_cavity_member_ids': {row['source_id']: row['member_id'] for row in cavities['chambers']},
            'findings': {'explicit_right_av_annulus_orifice_definition_found': bool(interface_candidates),
                         'all_leaflet_quotients_have_no_boundary_loops': all(row['exact_coordinate_quotient']['topology']['boundary_loop_count'] == 0 for row in leaflets),
                         'source_defined_partition_surface_available': False,
                         'missing_source_definition_is_not_absent_anatomy': True},
            'source_geometry_modified': False, 'added_faces': 0, 'physical_stepping': False,
            'biological_interface_selected': False, 'mechanical_mass_assigned': False,
            'boundary': 'The pinned source resolves three leaflet surface models. Exact seam identification exposes no annular boundary loop. Queried source labels and memberships do not define a right atrioventricular partition surface; this is missing source support, not evidence that the anatomical annulus or orifice is absent. Leaflet self-intersection, valve phase, attachment/contact mechanics and biological calibration are not established.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=ROOT / 'Sources')
    parser.add_argument('--source-lock', type=Path, default=ROOT / 'sources.lock.json')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args(argv)
    try:
        result = audit(sources=args.sources, source_lock=args.source_lock)
        encoded = canonical(result) + b'\n'
        require(not args.output.is_symlink() and (not args.output.exists() or args.output.read_bytes() == encoded),
                'output is immutable; choose a new path')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not args.output.exists():
            with args.output.open('xb') as stream:
                stream.write(encoded)
        print(json.dumps({'status': 'verified_source_leaflet_inventory', 'output': str(args.output),
                          'sha256': hashlib.sha256(encoded).hexdigest(), 'leaflets': len(result['leaflets']),
                          'source_partition_surface_available': result['findings']['source_defined_partition_surface_available'],
                          'source_bytes_unchanged': result['source_bytes_unchanged']}))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({'status': 'rejected', 'error': str(error)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
