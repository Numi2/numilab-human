"""Conform resting thorax triangles to one piecewise-affine basal field.

Asset preparation only. The native owner still advances the same two mechanical
volume coordinates on Metal. This discretizes its registered basal shape field;
it introduces neither tissue elements nor another time integrator.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from .resting_anatomy_interface_patch import basal_field

# A dyadic 15.625 mm grid avoids decimal-grid ambiguity in Float32 source space.
GRID_SPACING_M = 1.0 / 64.0
PLANE_FAMILIES = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (1, -1, 0), (1, 0, -1), (0, 1, -1))


def kuhn_basis(points, spacing=GRID_SPACING_M):
    """Value and constant cell gradient of the continuous Kuhn interpolant.

Within each cube, order fractional coordinates largest first. The four cell
vertices follow those axes from the low corner to the high corner. In this
triangulation the Y derivative is a difference along an original Y grid edge;
the source basal field is nonincreasing in Y, so the interpolant is also.
"""
    points = np.asarray(points, dtype=np.float64)
    u = points / spacing
    low = np.floor(u).astype(np.int64)
    f = u - low
    order = np.argsort(-f, axis=1, kind="stable")
    corners = np.repeat(low[:, None, :], 4, axis=1)
    for j in range(3):
        corners[:, j + 1] = corners[:, j]
        corners[np.arange(len(points)), j + 1, order[:, j]] += 1
    values = basal_field(corners.reshape(-1, 3) * spacing).reshape(-1, 4)
    gradient = np.empty_like(points)
    for j in range(3):
        gradient[np.arange(len(points)), order[:, j]] = (values[:, j + 1] - values[:, j]) / spacing
    value = values[:, 0] + np.sum(gradient * (points - low * spacing), axis=1)
    return value, gradient


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _normal(a, b, c):
    x = tuple(b[i] - a[i] for i in range(3))
    y = tuple(c[i] - a[i] for i in range(3))
    return (x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0])


def _split(polygon, normal, offset):
    distances = [_dot(p, normal) - offset for p in polygon]
    if min(distances) >= 0 or max(distances) <= 0:
        return [polygon]
    sides = [[], []]
    for i, point in enumerate(polygon):
        following = polygon[(i + 1) % len(polygon)]
        d, e = distances[i], distances[(i + 1) % len(polygon)]
        if d <= 0:
            sides[0].append(point)
        if d >= 0:
            sides[1].append(point)
        if d * e < 0:
            t = d / (d - e)
            intersection = tuple(point[k] + t * (following[k] - point[k]) for k in range(3))
            for side in sides:
                side.append(intersection)
    if any(len(p) < 3 for p in sides):
        raise ValueError("cell clipping made a polygon with fewer than three vertices")
    return sides


def _triangulate(polygon, reference_normal):
    polygon = list(polygon)
    result = []
    while len(polygon) > 3:
        ears = []
        for i, p in enumerate(polygon):
            tri = (polygon[i-1], p, polygon[(i+1) % len(polygon)])
            if _dot(_normal(*tri), reference_normal) <= 0:
                continue
            remaining = polygon[:i] + polygon[i+1:]
            if len(remaining) == 3 and _dot(_normal(*remaining), reference_normal) <= 0:
                continue
            ears.append((p, i, tri))
        if not ears:
            raise ValueError("cell polygon has no positive-area ear")
        _, i, tri = min(ears)
        result.append(tri)
        polygon.pop(i)
    if _dot(_normal(*polygon), reference_normal) <= 0:
        raise ValueError("cell clipping made a degenerate or reversed triangle")
    result.append(tuple(polygon))
    return result


def conform_surface(vertices6, faces, spacing=GRID_SPACING_M, progress=None, coordinate_resolution_m=0):
    """Exact rational clipping; round once at existing NHANAT Float32 boundary.

    All six plane families partition space into a shared tetrahedral complex.
    Canonical ear selection makes opposite-wound reciprocal faces choose the
    same diagonals. Exact coordinate keys share all generated edge points.
    """
    source = np.asarray(vertices6, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if coordinate_resolution_m not in (0, 1e-6):
        raise ValueError('unsupported source coordinate resolution')
    if not np.isfinite(source).all() or not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("invalid source or grid spacing")
    h = Fraction.from_float(float(spacing))
    points = [tuple(Fraction.from_float(float(x)) for x in row[:3]) for row in source]
    vertex_lookup = {}
    output_points, output_faces, mapping = [], [], {}
    maximum_rounding = 0.0
    quantum = Fraction(str(coordinate_resolution_m)) if coordinate_resolution_m else None
    serialization_merges = 0
    collapsed_generated_faces = 0
    @lru_cache(maxsize=None)
    def vertex(point):
        nonlocal maximum_rounding, serialization_merges
        float64 = np.asarray([float(x) for x in point], dtype=np.float64)
        stored = tuple(math.floor(x/quantum + Fraction(1, 2))*quantum for x in point) if quantum else point
        rounded = np.asarray([float(x) for x in stored], dtype=np.float32)
        maximum_rounding = max(maximum_rounding, float(np.linalg.norm(rounded.astype(float) - float64)))
        key = tuple(float(x) for x in rounded)
        # NHANAT's vertices are Float32. Quotient only exactly identical
        # serialized coordinates, recording each rational-point merge.
        if key in vertex_lookup:
            previous, index = vertex_lookup[key]
            if previous != point:
                serialization_merges += 1
            return index
        index = len(output_points)
        output_points.append(rounded)
        vertex_lookup[key] = (point, index)
        maximum_rounding = max(maximum_rounding, float(np.linalg.norm(rounded.astype(float) - float64)))
        return index
    for fi, ids in enumerate(faces):
        triangle = tuple(points[int(i)] for i in ids)
        reference_normal = _normal(*triangle)
        if not any(reference_normal):
            raise ValueError(f"source triangle {fi} is exactly degenerate")
        polygons = [triangle]
        for normal in PLANE_FAMILIES:
            distances = [_dot(p, normal) / h for p in triangle]
            for k in range(math.floor(min(distances)) + 1, math.ceil(max(distances))):
                polygons = [child for polygon in polygons for child in _split(polygon, normal, k*h)]
        begin = len(output_faces)
        for polygon in polygons:
            for tri in _triangulate(polygon, reference_normal):
                output_face = tuple(vertex(p) for p in tri)
                serialized = [tuple(float(x) for x in output_points[i]) for i in output_face]
                if len(set(output_face)) < 3 or not any(_normal(*serialized)):
                    collapsed_generated_faces += 1
                else:
                    output_faces.append(output_face)
        mapping[fi] = list(range(begin, len(output_faces)))
        if progress and fi % 10000 == 0:
            progress(fi, len(faces), len(output_faces))
    v = np.asarray(output_points, dtype=np.float32)
    f = np.asarray(output_faces, dtype=np.int64)
    tri = v[f].astype(np.float64)
    normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    if np.any(np.linalg.norm(normals, axis=1) == 0):
        raise ValueError("cell triangles degenerate after Float32 serialization")
    vertex_normals = np.zeros_like(v, dtype=np.float64)
    for i in range(3):
        np.add.at(vertex_normals, f[:, i], normals)
    lengths = np.linalg.norm(vertex_normals, axis=1)
    if np.any(lengths == 0):
        raise ValueError("conforming surface has a zero vertex normal")
    vertex_normals /= lengths[:, None]
    return np.column_stack((v, vertex_normals)).astype(np.float32), f, mapping, {
        "input_vertex_count": len(source), "input_triangle_count": len(faces),
        "output_vertex_count": len(v), "output_triangle_count": len(f),
        "maximum_float32_rounding_distance_m": maximum_rounding,
        "rational_vertices_merged_at_identical_float32_position": serialization_merges,
        "generated_faces_collapsed_by_serialization_quotient": collapsed_generated_faces,
        "spacing_m": spacing,
        "coordinate_resolution_m": coordinate_resolution_m,
    }


def resolve_short_edges(prepared, resolution_m, seam_surface_id=None):
    """Contract only short mesh edges using one shared coordinate quotient.

    Components choose an existing minimax representative, not a rounded point.
    No point may move farther than the declared resolution. Opposite faces and
    duplicated coordinates therefore receive exactly the same operation. The
    optional seam is restricted to paired boundary points within two nanometres;
    the caller must first establish the source identity of that numerical seam.
    """
    if resolution_m != 1.25e-7:
        raise ValueError('unsupported short-edge resolution')
    from . import resting_anatomy_interface_patch as base
    points, lookup, local = [], {}, {}
    for sid, (vertices, _, _, _) in prepared.items():
        indices = []
        for row in vertices[:, :3]:
            key = tuple(float(x) for x in row)
            if key not in lookup:
                lookup[key] = len(points)
                points.append(key)
            indices.append(lookup[key])
        local[sid] = np.asarray(indices, dtype=np.int64)
    points = np.asarray(points, dtype=np.float64)
    parent = np.arange(len(points))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    def union(a, b):
        a, b = find(int(a)), find(int(b))
        if a != b:
            if tuple(points[a]) > tuple(points[b]):
                a, b = b, a
            parent[b] = a

    short_counts, seam_pairs = {}, []
    for sid, (_, faces, _, _) in prepared.items():
        f = local[sid][faces]
        edges = np.unique(np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1), axis=0)
        distances = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
        short = edges[distances < resolution_m]
        short_counts[str(sid)] = len(short)
        for a, b in short:
            union(a, b)
    if seam_surface_id is not None:
        topology = base.topology_report(prepared[seam_surface_id][1])
        vertices = sorted({int(v) for edge in topology['boundary_edges'] for v in edge})
        for k, a in enumerate(vertices):
            for b in vertices[k + 1:]:
                ga, gb = local[seam_surface_id][a], local[seam_surface_id][b]
                distance = float(np.linalg.norm(points[ga] - points[gb]))
                if 0 < distance <= 2e-9:
                    union(ga, gb)
                    seam_pairs.append([a, b, distance])
        if not seam_pairs:
            raise ValueError('qualified numerical seam has no paired boundary points')
    groups = {}
    for i in range(len(points)):
        groups.setdefault(find(i), []).append(i)
    for group in groups.values():
        if len(group) < 2:
            continue
        coordinates = points[group]
        distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
        chosen = min(range(len(group)), key=lambda i: (float(distances[i].max()), float(distances[i].sum()), tuple(coordinates[i])))
        representative = group[chosen]
        parent[group] = representative
        parent[representative] = representative
    remap = np.asarray([find(i) for i in range(len(points))])
    shifts = np.linalg.norm(points[remap] - points, axis=1)
    if float(shifts.max(initial=0)) > resolution_m:
        raise ValueError('transitive short-edge component exceeds the displacement bound')
    report = {'resolution_m': resolution_m, 'maximum_displacement_m': float(shifts.max(initial=0)),
              'changed_unique_points': int(np.count_nonzero(shifts)),
              'short_edge_counts': short_counts, 'qualified_seam_pairs': seam_pairs}
    for sid, (old_vertices, old_faces, parent_faces, detail) in list(prepared.items()):
        coordinates = points[remap[local[sid]]].astype(np.float32)
        vertices, inverse = np.unique(coordinates, axis=0, return_inverse=True)
        faces = inverse[old_faces]
        tri = vertices[faces].astype(np.float64)
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        keep = np.any(normals != 0, axis=1)
        old_tri = old_vertices[old_faces, :3].astype(np.float64)
        old_normals = np.cross(old_tri[:, 1] - old_tri[:, 0], old_tri[:, 2] - old_tri[:, 0])
        if np.any(keep & (np.sum(normals * old_normals, axis=1) <= 0)):
            raise ValueError(f'short-edge contraction reverses a source face in {sid}')
        old_to_new = np.cumsum(keep) - 1
        mapping = {parent: [int(old_to_new[i]) for i in children if keep[i]] for parent, children in parent_faces.items()}
        faces = faces[keep]
        used = np.unique(faces)
        trim = np.zeros(len(vertices), dtype=np.int64)
        trim[used] = np.arange(len(used))
        vertices, faces = vertices[used], trim[faces]
        vertex_normals = np.zeros_like(vertices, dtype=np.float64)
        normals = normals[keep]
        for j in range(3):
            np.add.at(vertex_normals, faces[:, j], normals)
        lengths = np.linalg.norm(vertex_normals, axis=1)
        if np.any(lengths == 0):
            raise ValueError('short-edge contraction leaves a zero vertex normal')
        vertex_normals /= lengths[:, None]
        detail.update({'output_vertex_count': len(vertices), 'output_triangle_count': len(faces),
                       'short_edge_removed_triangle_count': int((~keep).sum()),
                       'short_edge_maximum_displacement_m': float(shifts[local[sid]].max(initial=0))})
        prepared[sid] = (np.column_stack((vertices, vertex_normals)).astype(np.float32), faces, mapping, detail)
    return report


def build_candidate(payload, receipt_path, config_path, output, coordinate_resolution_m=0, short_edge_resolution_m=0):
    from . import resting_anatomy_interface_patch as base
    from . import resting_anatomy_conforming_refinement as refine
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    header, rows = base.parse_payload(payload)
    receipt = json.loads(receipt_path.read_text())
    if receipt['payload']['sha256'] != sha(payload):
        raise ValueError('input receipt does not bind the input anatomy')
    if output.exists():
        raise ValueError('refusing to overwrite a retained candidate')
    output.mkdir(parents=True)
    respiratory = receipt['functional_bindings']['respiratory_geometry_binding']
    expected = {'basal_blend_start_m': -.052405, 'basal_blend_span_m': .06,
                'footprint_rim_ellipse_m': [.0545, .00055, .0978, .1375],
                'footprint_rim_transition': [.85, .20],
                'footprint_crural_ellipses_m': [[.005, .027, .020, .015], [.005, -.027, .020, .015]],
                'footprint_crural_transition': [.65, .70]}
    if any(respiratory.get(k) != v for k, v in expected.items()):
        raise ValueError('source smooth respiratory field differs from the declared reference')
    modified = (305, 306, 307, 308, 309, 311)
    unaffected = {str(sid): refine._surface_identity_sha256(sid, row) for sid, row in rows.items() if sid not in modified}
    if coordinate_resolution_m and short_edge_resolution_m:
        raise ValueError('coordinate quantization and short-edge resolution are distinct alternatives')
    # Establish the source seam identity before authorizing its contraction.
    seam_paths = [[7095, 7113, 7222, 7402, 7575, 7615], [7095, 7114, 7223, 7403, 7574, 7615]]
    seam_edges = {tuple(sorted(edge)) for path in seam_paths for edge in zip(path, path[1:])}
    if short_edge_resolution_m:
        seam_topology = base.topology_report(rows[311]['faces'])
        seam_points = rows[311]['vertices6'][:, :3].astype(float)
        if set(seam_topology['boundary_edges']) != seam_edges or np.linalg.norm(seam_points[seam_paths[0]] - seam_points[seam_paths[1]], axis=1).max() > 2e-9:
            raise ValueError('short-edge source does not contain the qualified numerical seam')
    prepared = {sid: conform_surface(rows[sid]['vertices6'], rows[sid]['faces'],
        progress=lambda *x, sid=sid: print('surface', sid, *x, flush=True), coordinate_resolution_m=coordinate_resolution_m) for sid in modified}
    short_edge_report = resolve_short_edges(prepared, short_edge_resolution_m, 311) if short_edge_resolution_m else None
    details, mappings = {}, {}
    for sid in modified:
        row = rows[sid]
        before_topology = base.topology_report(row['faces'])
        before_volume = base.signed_volume(row['vertices6'][:, :3].astype(float), row['faces'])
        v, f, mapping, detail = prepared[sid]
        topology = base.topology_report(f)
        if any(topology[k] for k in ('nonmanifold_edge_count', 'orientation_error_edge_count', 'boundary_branch_vertex_count')):
            raise ValueError(f'conforming cells invalidated topology of {sid}: {topology}')
        seam_closure = None
        if topology['boundary_loop_count'] != before_topology['boundary_loop_count']:
            # The retained source boundary is two paths only 1.86 nm apart.
            # Validate this specific numerical seam; never close an arbitrary
            # source opening merely because a mesh quantizer eliminated it.
            paths = [[7095, 7113, 7222, 7402, 7575, 7615],
                     [7095, 7114, 7223, 7403, 7574, 7615]]
            expected_edges = {tuple(sorted(edge)) for path in paths for edge in zip(path, path[1:])}
            actual_edges = set(before_topology['boundary_edges'])
            source_points = row['vertices6'][:, :3].astype(float)
            if sid != 311 or not (coordinate_resolution_m == 1e-6 or short_edge_resolution_m == 1.25e-7) or topology['boundary_loop_count'] != 0 or actual_edges != expected_edges:
                raise ValueError(f'conforming cells changed an unqualified source opening of {sid}')
            widths = np.linalg.norm(source_points[paths[0]] - source_points[paths[1]], axis=1)
            if widths.max() > 2e-9:
                raise ValueError('declared diaphragm seam exceeds its measured 2 nm resolution bound')
            seam_closure = {'source_boundary_paths': paths, 'corresponding_path_distances_m': widths.tolist(),
                'maximum_path_distance_m': float(widths.max()),
                'interpretation': 'closure of a sub-resolution mesh seam, not an identified anatomical hiatus',
                'source_payload_sha256': sha(payload), 'source_boundary_edges': len(actual_edges),
                'candidate_boundary_edges': topology['boundary_edge_count']}
        if sid != 311 and topology['boundary_edge_count']:
            raise ValueError(f'conforming cells opened source lobe {sid}')
        after_volume = base.signed_volume(v[:, :3].astype(float), f)
        volume_tolerance = 1e-8 if coordinate_resolution_m else 1e-11
        if abs(after_volume - before_volume) > volume_tolerance:
            raise ValueError(f'cell clipping changed source volume of {sid} beyond its declared numerical bound')
        detail.update({'source_signed_volume_m3': before_volume,
                       'candidate_signed_volume_m3': after_volume,
                       'volume_change_m3': after_volume - before_volume,
                       'topology': {k: val for k, val in topology.items() if k != 'boundary_edges'},
                       'source_boundary_edges': before_topology['boundary_edge_count']})
        if seam_closure:
            detail['numerical_seam_closure'] = seam_closure
        row['vertices6'], row['faces'] = v, f
        details[str(sid)], mappings[sid] = detail, mapping
        np.savez(output/f'surface-{sid}.npz', vertices6=v, faces=f)
        (output/f'surface-{sid}-parent-faces.json').write_text(json.dumps(mapping)+'\n')
    interface = receipt['provenance']['diaphragm_lung_interface']
    for entry in interface['interface_rows']:
        sid = int(entry['lung_stable_id'])
        lobe_ids = refine.remap_ids(refine._patch_face_ids(entry, 'registered_lung_face_index_ranges'), mappings[sid])
        diaphragm_ids = refine.remap_ids(refine._patch_face_ids(entry, 'diaphragm_patch_face_start'), mappings[311])
        if not diaphragm_ids or diaphragm_ids != list(range(diaphragm_ids[0], diaphragm_ids[-1]+1)):
            raise ValueError('conforming diaphragm patch is not contiguous')
        entry['registered_lung_face_index_ranges'] = base.index_ranges(lobe_ids)
        entry['registered_lung_face_count'] = len(lobe_ids)
        entry['diaphragm_patch_face_start'] = diaphragm_ids[0]
        entry['diaphragm_patch_face_count'] = len(diaphragm_ids)
        entry['patch_area_m2'] = refine._area(rows[sid]['vertices6'], rows[sid]['faces'], lobe_ids)
        edges = refine.face_set_boundary_edges(rows[sid]['faces'], lobe_ids)
        entry['shared_boundary_edge_count'] = len(edges)
        entry['shared_boundary_vertex_count'] = len({v for e in edges for v in e})
    # This exact key check also verifies reciprocal patch face count/geometry.
    mates, _ = refine.interface_face_mates(rows, receipt)
    for (sid, face), (_, opposite) in mates.items():
        a = rows[sid]['vertices6'][rows[sid]['faces'][face], :3].astype(float)
        b = rows[311]['vertices6'][rows[311]['faces'][opposite], :3].astype(float)
        if np.dot(np.cross(a[1]-a[0], a[2]-a[0]), np.cross(b[1]-b[0], b[2]-b[0])) >= 0:
            raise ValueError('reciprocal patch lost opposite winding')
    interface['diaphragm_topology_after_patch'] = details['311']['topology']
    if details['311'].get('numerical_seam_closure'):
        interface['numerical_seam_closure'] = details['311']['numerical_seam_closure']
        interface['source_opening_identity'] = 'previous unassigned boundary measured as a 1.86 nm double-path numerical seam; closed at the explicitly declared source resolution'
    interface['added_reversed_interface_face_count'] = len(mates)
    interface['added_reversed_interface_surface_area_m2'] = sum(x['patch_area_m2'] for x in interface['interface_rows'])
    area_rows, volumes = [], []
    for sid in modified[:5]:
        v = rows[sid]['vertices6'][:, :3].astype(float)
        f = rows[sid]['faces']
        value, _ = kuhn_basis(v)
        volume = base.signed_volume(v, f)
        moved = v.copy(); moved[:, 1] -= .01 * value
        area = (base.signed_volume(moved, f)-volume)/.01
        if area <= 0:
            raise ValueError('conforming diaphragm field does not expand a lobe')
        area_rows.append({'lung_stable_id': sid, 'effective_area_m2': area})
        volumes.append(volume)
    area = sum(x['effective_area_m2'] for x in area_rows)
    previous_area = respiratory['diaphragm_effective_area_m2']
    respiratory.update({'basal_weight_interpolation': 'conforming_kuhn_grid_v1',
                        'conforming_grid_spacing_m': GRID_SPACING_M,
                        'diaphragm_effective_area_m2': area,
                        'per_lobe_effective_area_m2': area_rows})
    respiratory.pop('finite_displacement_sensitivity_m2', None)
    respiratory['source_refinement_area_update'] = {'previous_area_m2': previous_area, 'refined_area_m2': area,
        'basis': 'closed lobe volume derivative of the shared piecewise-affine basal interpolation'}
    receipt['thorax_source_volume_m3'].update({'five_lung_envelopes': volumes, 'sum': sum(volumes)})
    if unaffected != {str(sid): refine._surface_identity_sha256(sid, rows[sid]) for sid in rows if sid not in modified}:
        raise ValueError('respiratory remeshing changed an unrelated source surface')
    data, nv, ni = refine._pack(header, rows)
    target = output/'resting-thorax.nhanatomy'; target.write_bytes(data)
    receipt['payload'].update({'path': str(target), 'sha256': sha(target), 'surface_count': len(rows), 'vertex_count': nv, 'index_count': ni})
    receipt['functional_bindings']['anatomy_payload_sha256'] = sha(target)
    cardiac = receipt['provenance']['cardiac_geometry_binding']
    cardiac['output_anatomy_payload_sha256'] = sha(target)
    cardiac['downstream_respiratory_cell_refinement'] = {'input_payload_sha256': sha(payload), 'output_payload_sha256': sha(target),
        'modified_stable_ids': list(modified), 'unmodified_source_surface_sha256': unaffected}
    receipt['provenance']['conforming_respiratory_cells'] = {
        'algorithm': 'exact_rational_shared_kuhn_cell_clipping_v1',
        'source_payload_path': str(payload), 'source_payload_sha256': sha(payload),
        'source_receipt_path': str(receipt_path), 'source_receipt_sha256': sha(receipt_path),
        'source_config_path': str(config_path), 'source_config_sha256': sha(config_path),
        'implementation_sha256': sha(Path(__file__)), 'grid_spacing_m': GRID_SPACING_M,
        'coordinate_resolution_m': coordinate_resolution_m,
        'short_edge_resolution': short_edge_report,
        'source_geometry_status': ('inferred reference geometry with bounded short-edge contraction' if short_edge_resolution_m else
                                  'inferred reference geometry with declared source-space numerical quantization' if coordinate_resolution_m else
                                  'source coordinates retained within Float32 serialization precision'),
        'surface_details': details, 'reciprocal_patch_triangle_count': len(mates),
        'functional_owner': 'existing GPU respiratory mechanics; no new physical state',
        'qualification': 'static topology and reciprocal interfaces checked; native complete-cycle audit pending'}
    receipt['qualification']['diaphragm_lung_interface'] = 'conforming respiratory field candidate; native full-cycle audit pending'
    result_receipt = output/'resting-anatomy-receipt.json'
    result_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    config = json.loads(config_path.read_text()); config['diaphragm_area_m2'] = area
    config.setdefault('parameter_scope', {})['conforming_basal_field'] = 'Inferred reference piecewise-affine interpolation, area recomputed from registered lobe boundaries; not measured person data.'
    result_config = output/'resting-reference-respiration.json'
    result_config.write_text(json.dumps(config, indent=2, sort_keys=True)+'\n')
    result = {'payload_sha256': sha(target), 'receipt_sha256': sha(result_receipt), 'config_sha256': sha(result_config),
              'area_m2': area, 'previous_area_m2': previous_area, 'surface_details': details}
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coordinate-resolution-m', type=float, default=0)
    parser.add_argument('--short-edge-resolution-m', type=float, default=0)
    args = parser.parse_args()
    if args.coordinate_resolution_m not in (0, 1e-6):
        parser.error('the validated candidate numerical resolutions are zero or one micrometre')
    if args.short_edge_resolution_m not in (0, 1.25e-7):
        parser.error('the short-edge resolutions are zero or 0.125 micrometres')
    print(json.dumps(build_candidate(args.input, args.receipt, args.config, args.output, args.coordinate_resolution_m, args.short_edge_resolution_m), sort_keys=True))
