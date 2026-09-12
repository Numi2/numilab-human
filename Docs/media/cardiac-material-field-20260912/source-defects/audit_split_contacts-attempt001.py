#!/usr/bin/env python3
"""Exact witnesses that node-identity normalization is not geometry repair."""
import argparse
from array import array
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--asset', type=Path, required=True)
p.add_argument('--report', type=Path, required=True)
p.add_argument('--owners-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
start = time.monotonic()
sha = lambda b: hashlib.sha256(b).hexdigest()
rb = a.report.read_bytes(); r = json.loads(rb)
mb = (a.asset/'manifest.json').read_bytes(); m = json.loads(mb)
assert sha(mb) == r['asset_manifest_sha256']
owner_path = a.owners_root/'src/numilab_human/cardiac_cavity_intersections.py'
owner_sha = sha(owner_path.read_bytes())
assert owner_sha == '8b138882161c78312cb9b5aa46f49795092eb9b67ed9f705dc354787cb0a635b'
sys.path.insert(0, str(a.owners_root/'src'))
from numilab_human import cardiac_cavity_intersections as exact
def read(name, code):
    raw = (a.asset/name).read_bytes()
    assert sha(raw) == m['buffers'][name]['sha256']
    values = array(code); values.frombytes(raw)
    if sys.byteorder != 'little': values.byteswap()
    return values
nodes = read('nodes.f64le', 'd'); boundary = read('boundary.u32le', 'I')
owners = read('boundary_owners.u32le', 'I'); labels = read('labels.u32le', 'I')
tets = read('tetrahedra.u32le', 'I'); comps = read('boundary_components.u32le', 'I')
clone = {}; clone_source = {}; stars = {}
for row in r['vertex_star_reports']:
    node = row['source_node']; stars[node] = []
    for part in row['components']:
        identity = part['node_identity']; clone_source[identity] = node
        for cell in part['source_cells']:
            clone[node, cell] = identity; stars[node].append(cell)
def mapped_node(n, cell): return clone.get((n, cell), n)
def mapped_tet(cell): return tuple(mapped_node(n, cell) for n in tets[4*cell:4*cell+4])
preserved = 0
for node, cells in stars.items():
    original_faces = {}
    for c in cells:
        others = [n for n in tets[4*c:4*c+4] if n != node]
        for pair in itertools.combinations(others, 2):
            face = tuple(sorted((node, *pair)))
            original_faces.setdefault(face, []).append(c)
    for face, ids in original_faces.items():
        if len(ids) == 2:
            assert sorted(mapped_node(n, ids[0]) for n in face) == sorted(mapped_node(n, ids[1]) for n in face)
            preserved += 1
for (node, cell), identity in clone.items():
    assert clone_source[identity] == node and node in tets[4*cell:4*cell+4]
    assert len(set(mapped_tet(cell))) == 4

edge_faces = {tuple(e['source_edge']): [] for e in r['edge_link_reports']}
for i, face in enumerate(zip(*[iter(boundary)]*3)):
    for pair in itertools.combinations(face, 2):
        edge = tuple(sorted(pair))
        if edge in edge_faces: edge_faces[edge].append(i)
den = max(v.as_integer_ratio()[1] for v in nodes)
def point(node):
    return tuple(n*(den//d) for n,d in (v.as_integer_ratio() for v in nodes[3*node:3*node+3]))
def encode(q):
    q = Fraction(q, den)
    return {'numerator_hex': hex(q.numerator), 'denominator_hex': hex(q.denominator)}
witnesses = []
for edge, ids in sorted(edge_faces.items()):
    assert len(ids) == 4
    for i, j in itertools.combinations(ids, 2):
        source_first = tuple(boundary[3*i:3*i+3]); source_second = tuple(boundary[3*j:3*j+3])
        first = tuple(mapped_node(n, owners[i]) for n in source_first)
        second = tuple(mapped_node(n, owners[j]) for n in source_second)
        tri = tuple(point(n) for n in source_first); other = tuple(point(n) for n in source_second)
        hits = exact.triangle_intersection_points(tri, other)
        assert hits
        shared = set(first) & set(second)
        common = {tri[first.index(n)] for n in shared}
        forbidden = [q for q in hits if not exact._allowed_shared_point(q, common)]
        if forbidden:
            witnesses.append({'source_edge': edge, 'boundary_faces': [i,j],
                'source_cells': [owners[i],owners[j]], 'source_labels': [labels[owners[i]],labels[owners[j]] ,
                'source_boundary_components': [comps[i],comps[j]],
                'source_node_ids': [source_first, source_second],
                'variant_node_ids': [first, second], 'shared_variant_nodes': sorted(shared),
                'intersection_points_m': [[encode(v) for v in q] for q in sorted(hits)],
                'forbidden_point_count': len(forbidden),
                'classification': 'exact geometric contact extends beyond allowed shared topological simplex'})
report = {'schema': 'NumiHuman.Rodero18SplitContactWitnesses.v1',
    'asset_manifest_sha256': sha(mb), 'star_audit_sha256': sha(rb),
    'predicate_owner_path': str(owner_path), 'predicate_owner_sha256': owner_sha,
    'script_sha256': sha(Path(__file__).read_bytes()), 'command': sys.argv,
    'existing_face_constraints_independently_checked': preserved,
    'all_source_cell_coordinates_and_volumes_unchanged': True,
    'forbidden_geometric_contact_pair_count': len(witnesses),
    'distinct_original_edge_count': len({tuple(w['source_edge']) for w in witnesses}),
    'pairs_by_source_boundary_component': {str(k): sum(w['source_boundary_components']==[k,k] for w in witnesses) for k in sorted(set(comps))},
    'witnesses': witnesses, 'maximal_split_embedded': False,
    'audit_scope': 'all incident boundary face pairs of original 31 bad edges only; sufficient counterexamples, not a full global intersection inventory',
    'source_geometry_edits': False, 'physical_stepping': False,
    'elapsed_seconds': time.monotonic()-start}
a.output.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
print('forbidden contact pairs', len(witnesses), 'edges', report['distinct_original_edge_count'],
      'by component', report['pairs_by_source_boundary_component'],
      'face constraints', preserved, 'seconds', report['elapsed_seconds'], flush=True)
