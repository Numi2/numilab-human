#!/usr/bin/env python3
"""Read-only tetrahedral-star audit and scratch node-identity split experiment.

Coordinates are exact emitted binary64 metres; no source cell, face, coordinate,
label, parameter or physical state is changed. A vertex may split only between
face-connected components of its incident tetrahedral star. This computes the
maximal node duplication consistent with preserving every existing shared
tetrahedral face. It is an experiment, not production repair or calibration.
"""
import argparse
from array import array
from collections import Counter, defaultdict, deque
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

PIN_MANIFEST = '7eb93af573b328c77f3960f3f9fb40e337356d64fa6fa3ff59e258d1959b0dee'
PIN_AUDIT = 'ffea96141ab67089f71ed04de8adf4f7132a59e74192edb17eaf8c3555475d32'
PIN_CSV = '3f59ea2e6388aa370efe9046ecd5b4fd8bee87037aa904f45a80356f3932edcd'
p = argparse.ArgumentParser()
p.add_argument('--asset', type=Path, required=True)
p.add_argument('--boundary-audit', type=Path, required=True)
p.add_argument('--source-csv', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
start = time.monotonic()
sha = lambda b: hashlib.sha256(b).hexdigest()
mb = (a.asset / 'manifest.json').read_bytes()
assert sha(mb) == PIN_MANIFEST
m = json.loads(mb)
ab = a.boundary_audit.read_bytes()
assert sha(ab) == PIN_AUDIT
audit = json.loads(ab)
assert audit['asset_manifest_sha256'] == PIN_MANIFEST
cb = a.source_csv.read_bytes()
assert sha(cb) == PIN_CSV
source_row = next(r for r in csv.DictReader(cb.decode().splitlines()) if int(r['Mesh_ID']) == 18)
inputs = {}
def read(name, code):
    raw = (a.asset / name).read_bytes()
    expected = m['buffers'][name]
    assert sha(raw) == expected['sha256'] and len(raw) == expected['bytes']
    inputs[name] = expected
    result = array(code); result.frombytes(raw)
    if sys.byteorder != 'little': result.byteswap()
    return result

nodes = read('nodes.f64le', 'd')
tets = read('tetrahedra.u32le', 'I')
labels = read('labels.u32le', 'I')
boundary = read('boundary.u32le', 'I')
owners = read('boundary_owners.u32le', 'I')
components = read('boundary_components.u32le', 'I')
uvc_v = read('uvc_v.f64le', 'd')
assert len(tets) == 4 * len(labels) and len(nodes) == 3 * len(uvc_v)
bad_vertices = {x['node'] for x in audit['global_vertex_link_defects']}
bad_edges = {tuple(d['nodes']) for c in audit['components'] for d in c['edge_defects']}
assert len(bad_vertices) == 47 and len(bad_edges) == 31
stars = defaultdict(dict)
for cell, tet in enumerate(zip(*[iter(tets)] * 4)):
    for node in bad_vertices.intersection(tet): stars[node][cell] = tet
print('read pinned asset; gathered', len(stars), 'vertex stars', flush=True)

def graph_components(graph):
    unseen = set(graph); result = []
    while unseen:
        todo = [min(unseen)]; part = []
        while todo:
            x = todo.pop()
            if x not in unseen: continue
            unseen.remove(x); part.append(x); todo.extend(graph[x].keys() & unseen)
        result.append(sorted(part))
    return sorted(result)

star_reports = []
star_graphs = {}
cell_part = {}
clones = {}
next_node = len(nodes) // 3
for node in sorted(stars):
    cells = stars[node]
    adjacency = {cell: {} for cell in cells}
    face_incidence = defaultdict(list)
    for cell, tet in cells.items():
        others = [v for v in tet if v != node]
        for pair in itertools.combinations(others, 2):
            face_incidence[tuple(sorted((node, *pair)))].append(cell)
    assert all(len(ids) <= 2 for ids in face_incidence.values())
    for face, ids in face_incidence.items():
        if len(ids) == 2:
            c, d = ids
            adjacency[c][d] = face; adjacency[d][c] = face
    parts = graph_components(adjacency)
    star_graphs[node] = adjacency
    cell_part[node] = {cell: i for i, part in enumerate(parts) for cell in part}
    clone_rows = []
    for i, part in enumerate(parts):
        clone = node if i == 0 else next_node
        if i: next_node += 1
        for cell in part: clones[node, cell] = clone
        clone_rows.append({'node_identity': clone, 'source_cells': part,
            'source_label_counts': dict(sorted(Counter(str(labels[c]) for c in part).items()))})
    star_reports.append({'source_node': node, 'incident_tetrahedra': len(cells),
        'face_connected_star_components': len(parts), 'components': clone_rows})

def path_witness(node, start_cell, end_cell):
    queue = deque([start_cell]); prev = {start_cell: None}
    graph = star_graphs[node]
    while queue and end_cell not in prev:
        x = queue.popleft()
        for y in sorted(graph[x]):
            if y not in prev: prev[y] = x; queue.append(y)
    if end_cell not in prev: return None
    cells = [end_cell]
    while cells[-1] != start_cell: cells.append(prev[cells[-1]])
    cells.reverse()
    return {'cells': cells, 'shared_faces': [list(graph[x][y]) for x, y in zip(cells, cells[1:])]}

# One common dyadic denominator makes all local determinant signs exact.
den = max(v.as_integer_ratio()[1] for v in nodes)
points = [tuple(n * (den // d) for n, d in (v.as_integer_ratio() for v in nodes[i:i+3]))
          for i in range(0, len(nodes), 3)]
def sub(x, y): return tuple(a-b for a, b in zip(x, y))
def add(x, y): return tuple(a+b for a, b in zip(x, y))
def dot(x, y): return sum(a*b for a, b in zip(x, y))
def cross(x, y): return (x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0])
def determinant(tet):
    o, x, y, z = (points[v] for v in tet)
    return dot(sub(x, o), cross(sub(y, o), sub(z, o)))

edge_reports = []
for edge in sorted(bad_edges):
    u, v = edge
    cells = {c: t for c, t in stars[u].items() if v in t}
    link = defaultdict(dict); cell_rays = {}
    for cell, tet in cells.items():
        x, y = (n for n in tet if n not in edge)
        assert y not in link[x], 'duplicate link edge'
        link[x][y] = cell; link[y][x] = cell; cell_rays[cell] = (x, y)
    parts = graph_components(link)
    fan_cells = [sorted({c for n in part for c in link[n].values()}) for part in parts]
    fan_rows = [{'link_nodes': part, 'cells': ids,
        'end_nodes': [n for n in part if len(link[n]) == 1],
        'link_degrees': {str(n): len(link[n]) for n in part},
        'endpoint_star_components': {str(n): sorted({cell_part[n][c] for c in ids}) for n in edge},
        'source_label_counts': dict(sorted(Counter(str(labels[c]) for c in ids).items()))}
        for part, ids in zip(parts, fan_cells)]
    # Exact cross-sectional positive-interior overlap test for all incident cells.
    axis = sub(points[v], points[u])
    orient = lambda x, y: dot(axis, cross(x, y))
    cones = {}
    for cell, (x, y) in cell_rays.items():
        x, y = sub(points[x], points[u]), sub(points[y], points[u])
        turn = orient(x, y)
        assert turn != 0
        if turn < 0: x, y = y, x
        cones[cell] = (x, y)
    interior_overlaps = []
    for c, d in itertools.combinations(sorted(cells), 2):
        x, y = cones[c]; z, w = cones[d]
        inside = lambda q, a, b: orient(a, q) > 0 and orient(q, b) > 0
        if (any(inside(q, z, w) for q in (x, y, add(x, y))) or
            any(inside(q, x, y) for q in (z, w, add(z, w)))):
            interior_overlaps.append([c, d])
    relations = []
    for i, j in itertools.combinations(range(len(parts)), 2):
        c, d = fan_cells[i][0], fan_cells[j][0]
        paths = {str(n): path_witness(n, c, d) for n in edge}
        relations.append({'fans': [i, j], 'same_edge_after_maximal_split':
            all(paths[str(n)] is not None for n in edge), 'endpoint_continuity_paths': paths})
    edge_reports.append({'source_edge': list(edge), 'link_component_count': len(parts),
        'incident_cell_count': len(cells), 'fans': fan_rows, 'fan_relations': relations,
        'exact_positive_angle_overlap_cell_pairs': interior_overlaps})
print('edge fan and face-continuity analysis complete;', next_node-len(nodes)//3, 'possible clones', flush=True)

# Boundary-only evaluation of the maximal face-preserving coincident-node split.
# Tet owners fix each clone choice; every source facet remains present and oriented.
faces = []
for i, face in enumerate(zip(*[iter(boundary)] * 3)):
    faces.append(tuple(clones.get((n, owners[i]), n) for n in face))
edges = defaultdict(list); incident = defaultdict(list)
parent = list(range(len(faces)))
def find(n):
    while parent[n] != n:
        parent[n] = parent[parent[n]]; n = parent[n]
    return n
for i, face in enumerate(faces):
    for n in face: incident[n].append(i)
    for x, y in zip(face, face[1:] + face[:1]): edges[min(x, y), max(x, y)].append((i, x < y))
for rows in edges.values():
    for j, _ in rows[1:]: parent[find(j)] = find(rows[0][0])
groups = defaultdict(list)
for i in range(len(faces)): groups[find(i)].append(i)
edge_defects = [{'nodes': list(edge), 'faces': [i for i, _ in rows]} for edge, rows in edges.items()
                if len(rows) != 2 or rows[0][1] == rows[1][1]]
vertex_defects = []
for n, ids in incident.items():
    graph = defaultdict(dict); counts = Counter()
    for i in ids:
        x, y = (v for v in faces[i] if v != n)
        graph[x][y] = i; graph[y][x] = i; counts[min(x, y), max(x, y)] += 1
    parts = graph_components(graph)
    if len(parts) != 1 or any(len(v) != 2 for v in graph.values()) or any(c != 1 for c in counts.values()):
        vertex_defects.append({'node': n, 'link_components': len(parts),
            'noncycle_degrees': {str(v): len(x) for v, x in graph.items() if len(x) != 2},
            'source_boundary_components': sorted({components[i] for i in ids})})
variant_components = []
for ids in sorted(groups.values(), key=lambda v: min(v)):
    local_vertices = {n for i in ids for n in faces[i]}
    local_edges = {(min(x, y), max(x, y)) for i in ids for x, y in zip(faces[i], faces[i][1:]+faces[i][:1])}
    bad_e = [d for d in edge_defects if tuple(d['nodes']) in local_edges]
    bad_v = [d for d in vertex_defects if d['node'] in local_vertices]
    variant_components.append({'first_boundary_face': min(ids), 'face_count': len(ids),
        'source_boundary_components': sorted({components[i] for i in ids}),
        'node_count': len(local_vertices), 'edge_count': len(local_edges),
        'euler': len(local_vertices)-len(local_edges)+len(ids),
        'edge_defect_count': len(bad_e), 'vertex_link_defect_count': len(bad_v),
        'closed_oriented_vertex_manifold': not bad_e and not bad_v})
print('scratch split has', len(edge_defects), 'edge defects and', len(vertex_defects), 'vertex defects', flush=True)

# Exact determinant sums by source label and UVC V signatures. No relabeling.
sums = Counter(); counts = Counter(); uvc_groups = defaultdict(Counter)
uvc_sums = defaultdict(Counter)
for cell, tet in enumerate(zip(*[iter(tets)] * 4)):
    value = determinant(tet)
    assert value > 0
    label = labels[cell]
    sums[label] += value; counts[label] += 1
    signature = ','.join(format(v, '.17g') for v in sorted({uvc_v[n] for n in tet}))
    uvc_groups[label][signature] += 1; uvc_sums[label][signature] += value
volumes = {str(label): sums[label] / (6*den**3) * 1e6 for label in sorted(sums)}
comparisons = {}
for chamber, label in (('LV', 1), ('RV', 2)):
    source = float(source_row['Myo_vol_'+chamber]); volume = volumes[str(label)]
    comparisons[chamber] = {'source_label': label, 'geometry_volume_mL': volume,
        'source_csv_volume_mL': source, 'difference_mL': volume-source,
        'relative_difference': (volume-source)/source,
        'exact_volume_numerator_hex': hex(sums[label]), 'exact_volume_denominator_hex': hex(6*den**3)}
alternatives = {}
for name, tags in (('RV_tissue_only', (2,)), ('RV_plus_tricuspid_layer', (2,8)),
                   ('RV_plus_pulmonary_layer', (2,10)), ('RV_plus_both_valve_layers', (2,8,10))):
    volume = sum(sums[tag] for tag in tags)/(6*den**3)*1e6
    alternatives[name] = {'source_labels': tags, 'geometry_volume_mL': volume,
        'difference_from_source_RV_mL': volume-float(source_row['Myo_vol_RV']),
        'interpretation': 'diagnostic only; artificial valve tissue is not source RV myocardium'}

report = {'schema': 'NumiHuman.Rodero18SourceDefectAudit.v1',
    'qualification': 'offline_source_topology_analysis_only',
    'source_geometry_changed': False, 'physical_stepping': False,
    'asset_manifest_sha256': PIN_MANIFEST, 'boundary_audit_sha256': PIN_AUDIT,
    'source_archive_sha256': m['source_config']['source']['archive']['sha256'],
    'source_config_sha256': m['source_config_sha256'], 'input_buffers': inputs,
    'script_sha256': sha(Path(__file__).read_bytes()), 'command': sys.argv,
    'host': platform.node(), 'python': sys.version,
    'original_boundary_edge_defects': len(bad_edges), 'original_boundary_vertex_link_defects': len(bad_vertices),
    'vertex_star_reports': star_reports, 'edge_link_reports': edge_reports,
    'maximal_face_preserving_split': {'source_node_count': len(nodes)//3,
        'cloned_node_count': next_node-len(nodes)//3,
        'existing_shared_faces_preserved_by_construction': True,
        'every_cell_geometry_label_frame_and_volume_identical': True,
        'boundary_faces_added_or_removed': False,
        'edge_defect_count': len(edge_defects), 'vertex_link_defect_count': len(vertex_defects),
        'edge_defects': edge_defects, 'vertex_link_defects': vertex_defects,
        'boundary_components': variant_components,
        'embeddedness_checked': False, 'native_cavity_admitted': False},
    'source_volume_comparison': {'source_csv_sha256': PIN_CSV, 'source_csv_case18': source_row,
        'comparisons': comparisons, 'regional_tetrahedron_counts': dict(sorted(counts.items())),
        'regional_volume_mL': volumes, 'adjacent_label_diagnostics': alternatives,
        'uvc_v_by_label': {str(k): {'counts': dict(sorted(uvc_groups[k].items())),
            'volume_mL': {s: n/(6*den**3)*1e6 for s,n in sorted(uvc_sums[k].items())}}
            for k in sorted(uvc_groups)},
        'rv_discrepancy_resolved': False,
        'node_identity_split_can_change_source_label_volume': False},
    'elapsed_seconds': time.monotonic()-start}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
print('elapsed seconds', report['elapsed_seconds'], 'RV', comparisons['RV'], flush=True)
