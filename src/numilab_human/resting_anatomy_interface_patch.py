from __future__ import annotations
import argparse, hashlib, json, math, struct
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

HEADER = struct.Struct('<8s5I32s')
RECORD = struct.Struct('<8I')
BASE = Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/passive-viscera-candidate-002')
EVIDENCE = Path('/Users/n/numi-human-resting-evidence-20261005/zanatomy-diaphragm-source-check-001')
BASE_SHA = '10c847decea932fd4e1fb486d5dbc40abbe33141604046573f834edc1dd54902'
NPZ_SHA = 'df1a4305b8cd3e0cc8482a4c269d9e5a07001cd9fcf86826bb615f745d6a42f7'
SEAM_AUDIT_SHA = '0730d2ec5a25f44db6eb89b7731928d7267bd01ce8d22b76d2a8505ac6fa2c58'
CLIP_PROOF_SHA = 'bb433233c465ca90174e8439abf7f53eb339e3516c268b487abdaf61171a8145'
Z_EXPORT_SHA = '70c22373d7e0ffa4bd431976fcf84a30f7d7763c58202d64c564c783120fce5b'
REGISTRATION_SHA = 'b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e'
Z_ARCHIVE_SHA = 'e029688545627bd0214b269e1063143abb580aad72b2c2445d6d8a9a0d9da736'
Z_BLEND_SHA = '9f08a17ea0115fed80b2a73ecdf0a1bc2ab2f6956f37c593ce23d513ea35afcd'
Z_REVISION = '9a2ef22cc8443d14f9aa18ba91246416b1c6ab7a'
Z_EXPORTER_SHA = 'fcc69755777aaffe1a4b6fd0409d913e98b6aa86fae94bd4065ce590022005d7'
LOBE_IDS = (305, 306, 307, 308, 309)
LOBE_NAMES = {
    305: 'Inferior lobe of left lung',
    306: 'Inferior lobe of right lung',
    307: 'Middle lobe of right lung',
    308: 'Superior lobe of left lung',
    309: 'Superior lobe of right lung',
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces].astype(np.float64)
    return float(np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def index_ranges(indices):
    """Losslessly encode sorted face IDs as half-open integer ranges."""
    ids = sorted(map(int, indices))
    if not ids:
        return []
    out = []
    first = previous = ids[0]
    for value in ids[1:]:
        if value == previous + 1:
            previous = value
            continue
        out.append([first, previous + 1])
        first = previous = value
    out.append([first, previous + 1])
    return out


def normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = np.zeros(vertices.shape, dtype=np.float64)
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    lengths = np.linalg.norm(n, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-15):
        raise ValueError('cannot form finite normals for interface geometry')
    return n / lengths[:, None]


def oriented_edges(faces: np.ndarray):
    rows = defaultdict(list)
    for fi, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            aa, bb = int(a), int(b)
            rows[(min(aa, bb), max(aa, bb))].append((fi, 1 if aa < bb else -1))
    return rows


def connected_components_excluding_edges(faces: np.ndarray, cut_edges: np.ndarray):
    cut = {tuple(sorted(map(int, e))) for e in np.asarray(cut_edges).reshape(-1, 2)}
    edge_faces = defaultdict(list)
    for fi, tri in enumerate(faces):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_faces[tuple(sorted((int(a), int(b))))].append(fi)
    adj = [[] for _ in range(len(faces))]
    for edge, incident in edge_faces.items():
        if edge not in cut and len(incident) == 2:
            a, b = incident
            adj[a].append(b)
            adj[b].append(a)
    unseen = set(range(len(faces)))
    out = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        queue = [seed]
        comp = []
        while queue:
            fi = queue.pop()
            comp.append(fi)
            for other in adj[fi]:
                if other in unseen:
                    unseen.remove(other)
                    queue.append(other)
        out.append(np.asarray(sorted(comp), dtype=np.int64))
    return out, edge_faces, cut


def topology_report(faces: np.ndarray):
    edges = oriented_edges(faces)
    boundary = {e: rows for e, rows in edges.items() if len(rows) == 1}
    nonmanifold = {e: rows for e, rows in edges.items() if len(rows) > 2}
    orientation_errors = {e: rows for e, rows in edges.items() if len(rows) == 2 and rows[0][1] + rows[1][1] != 0}
    adjacency = defaultdict(set)
    for (a, b), rows in boundary.items():
        adjacency[a].add(b)
        adjacency[b].add(a)
    branches = sum(len(v) != 2 for v in adjacency.values())
    unseen = set(adjacency)
    loops = 0
    while unseen:
        loops += 1
        stack = [unseen.pop()]
        while stack:
            for v in adjacency[stack.pop()] & unseen:
                unseen.remove(v)
                stack.append(v)
    return {
        'edge_count': len(edges), 'boundary_edge_count': len(boundary),
        'boundary_loop_count': loops, 'boundary_branch_vertex_count': branches,
        'nonmanifold_edge_count': len(nonmanifold),
        'orientation_error_edge_count': len(orientation_errors),
        'boundary_edges': boundary,
    }


def parse_payload(path: Path):
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError('truncated NHANAT payload')
    h = HEADER.unpack_from(raw)
    magic, abi, count, vertex_count, index_count, registration_fp, source_sha = h
    if magic != b'NHANAT1\0' or abi != 5:
        raise ValueError('expected existing NHANAT1 ABI5 payload')
    if len(raw) != HEADER.size + count * RECORD.size + vertex_count * 24 + index_count * 4:
        raise ValueError('NHANAT exact byte length mismatch')
    records = [RECORD.unpack_from(raw, HEADER.size + i * RECORD.size) for i in range(count)]
    vo = HEADER.size + count * RECORD.size
    packed = np.frombuffer(raw, dtype='<f4', count=vertex_count * 6, offset=vo).reshape(-1, 6)
    io = vo + vertex_count * 24
    indices = np.frombuffer(raw, dtype='<u4', count=index_count, offset=io)
    rows = {}
    for rec in records:
        body, sv, nv, si, ni, sid, layer, flags = map(int, rec)
        if sid in rows:
            raise ValueError(f'duplicate source stable ID {sid}')
        if sv + nv > vertex_count or si + ni > index_count or ni % 3:
            raise ValueError(f'invalid record bounds for stable ID {sid}')
        tri_global = indices[si:si + ni].reshape(-1, 3).astype(np.int64)
        if tri_global.size and (tri_global.min() < sv or tri_global.max() >= sv + nv):
            raise ValueError(f'non-local source index in stable ID {sid}')
        rows[sid] = {
            'body_index': body, 'layer': layer, 'flags': flags,
            'vertices6': packed[sv:sv + nv].copy(),
            'faces': (tri_global - sv).astype(np.int64),
        }
    return h, rows


def decode_source_receipt(path: Path):
    d = json.loads(path.read_text())
    if d.get('payload', {}).get('sha256') != BASE_SHA:
        raise ValueError('base receipt is not bound to the exact anatomy candidate')
    if d.get('payload', {}).get('surface_count') != 463:
        raise ValueError('unexpected source surface count')
    return d


def make_patches(npz):
    d_vertices = np.asarray(npz['diaphragm_vertices'], dtype=np.float32).copy()
    d_faces = np.asarray(npz['diaphragm_triangles'], dtype=np.int64).copy()
    d_seam_edges = np.asarray(npz['diaphragm_cut_edges'], dtype=np.int64).reshape(-1, 2)
    coord_to_diaphragm = {}
    for i, v in enumerate(d_vertices):
        key = tuple(map(float, v))
        if key in coord_to_diaphragm and coord_to_diaphragm[key] != i:
            # Duplicate coordinates remain distinct unless this exact seam weld
            # requires one. The seam audit will fail closed if ambiguous.
            continue
        coord_to_diaphragm[key] = i
    added_vertices = []
    interface_rows = []
    added_face_groups = []
    for sid in LOBE_IDS[:-1]:
        lv = np.asarray(npz[f'lobe_{sid}_vertices'], dtype=np.float32)
        lf = np.asarray(npz[f'lobe_{sid}_triangles'], dtype=np.int64)
        lcut = np.asarray(npz[f'lobe_{sid}_cut_edges'], dtype=np.int64).reshape(-1, 2)
        components, edge_faces, cut = connected_components_excluding_edges(lf, lcut)
        if len(components) != 2:
            raise ValueError(f'lobe {sid} seam did not partition into two components')
        # The basal membrane patch is the lower connected component in torso +Y.
        patch_ids = min(components, key=lambda ids: float(lv[lf[ids]].mean(axis=(0, 1))[1]))
        patch_faces_source = lf[patch_ids]
        patch_area = float(.5 * np.linalg.norm(np.cross(
            lv[patch_faces_source][:, 1] - lv[patch_faces_source][:, 0],
            lv[patch_faces_source][:, 2] - lv[patch_faces_source][:, 0]), axis=1).sum())
        reversed_source = patch_faces_source[:, [0, 2, 1]]
        coordinate_map = {}
        for old_index in np.unique(reversed_source):
            point = lv[int(old_index)]
            key = tuple(map(float, point))
            match = coord_to_diaphragm.get(key)
            if match is not None:
                new_index = match
            else:
                new_index = len(d_vertices) + len(added_vertices)
                added_vertices.append(point.copy())
                coord_to_diaphragm[key] = new_index
            coordinate_map[int(old_index)] = new_index
        patch_faces = np.vectorize(coordinate_map.__getitem__, otypes=[np.int64])(reversed_source)
        if not np.array_equal(d_vertices[patch_faces] if patch_faces.max() < len(d_vertices) else
                              np.vstack([d_vertices, np.asarray(added_vertices, dtype=np.float32)])[patch_faces],
                              lv[patch_faces_source][:, [0, 2, 1]]):
            raise ValueError(f'lobe {sid} interface patch did not preserve exact shared coordinates')
        patch_boundary_edges = {e for e, incident in edge_faces.items()
                                if e in cut and sum(int(fi) in set(patch_ids.tolist()) for fi in incident) == 1}
        if patch_boundary_edges != cut:
            raise ValueError(f'lobe {sid} patch boundary differs from exact diaphragm seam')
        start = len(d_faces) + sum(len(x) for x in added_face_groups)
        added_face_groups.append(patch_faces)
        interface_rows.append({
            'diaphragm_stable_id': 311, 'lung_stable_id': sid,
            'diaphragm_patch_face_start': int(start), 'diaphragm_patch_face_count': int(len(patch_faces)),
            'registered_lung_face_index_ranges': index_ranges(patch_ids),
            'registered_lung_face_count': int(len(patch_ids)),
            'patch_area_m2': patch_area, 'shared_boundary_edge_count': len(cut),
            'shared_boundary_vertex_count': len({v for e in cut for v in e}),
            'orientation_relation': 'reversed triangle winding across exact coincident material-interface faces',
            'volume_owner_stable_id': sid,
        })
    if added_vertices:
        d_vertices = np.vstack([d_vertices, np.asarray(added_vertices, dtype=np.float32)])
    if added_face_groups:
        d_faces = np.vstack([d_faces, *added_face_groups])
    topo = topology_report(d_faces)
    if topo['nonmanifold_edge_count'] or topo['orientation_error_edge_count'] or topo['boundary_branch_vertex_count']:
        raise ValueError('diaphragm patch topology is nonmanifold, inconsistent, or branched')
    if topo['boundary_edge_count'] != 10 or topo['boundary_loop_count'] != 1:
        raise ValueError(f'expected one retained source opening boundary (10 edges), got {topo["boundary_edge_count"]} edges / {topo["boundary_loop_count"]} loops')
    boundary = topo.pop('boundary_edges')
    if any(tuple(sorted(map(int, e))) in {tuple(sorted(map(int, x))) for x in d_seam_edges} for e in boundary):
        raise ValueError('a lobe interface seam remains open after patch assembly')
    exact_d_seam = {tuple(sorted(map(int, e))) for e in d_seam_edges}
    d_edges = oriented_edges(d_faces)
    paired_seams = 0
    for e in exact_d_seam:
        rows = d_edges.get(e, [])
        if len(rows) != 2 or rows[0][1] + rows[1][1] != 0:
            raise ValueError('diaphragm seam edge is not paired with opposite orientation')
        paired_seams += 1
    topo['retained_opening_edge_count'] = len(boundary)
    topo['all_source_lobe_seam_edges_paired'] = paired_seams
    topo['original_registered_diaphragm_boundary_edge_count'] = 12
    topo['source_opening_identity'] = 'source-authored diaphragm aperture; source package does not assign a specific hiatus label'
    return d_vertices, d_faces, interface_rows, topo


def smoothstep(t):
    t = np.clip(t, 0., 1.)
    return t * t * (3. - 2. * t)


def basal_field(vertices):
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    xr, zr, rx, rz = .0545, .00055, .0978, .1375
    rr = np.sqrt(((x - xr) / rx) ** 2 + ((z - zr) / rz) ** 2)
    w_rim = 1. - smoothstep((rr - .85) / .20)
    crura = []
    for cz in (.027, -.027):
        rc = np.sqrt(((x - .005) / .020) ** 2 + ((z - cz) / .015) ** 2)
        crura.append(smoothstep((rc - .65) / .70))
    w = w_rim * crura[0] * crura[1]
    g = 1. - smoothstep((y + .052405) / .06)
    return g * w


def source_area_basis(lobe_rows):
    per_lobe = []
    steps = (.005, .01, .02)
    by_step = {d: [] for d in steps}
    for sid in LOBE_IDS:
        row = lobe_rows[sid]
        v = np.asarray(row['vertices6'][:, :3], dtype=np.float64)
        f = np.asarray(row['faces'], dtype=np.int64)
        b = basal_field(v)
        v0 = abs(signed_volume(v, f))
        area_rows = {}
        for d in steps:
            moved = v.copy()
            moved[:, 1] -= d * b
            vd = abs(signed_volume(moved, f))
            area_rows[str(d)] = (vd - v0) / d
            by_step[d].append((sid, (vd - v0) / d))
        per_lobe.append({'lung_stable_id': sid, 'effective_area_m2': area_rows['0.01']})
    total = sum(x['effective_area_m2'] for x in per_lobe)
    if total <= 0 or not math.isfinite(total):
        raise ValueError('derived diaphragm effective area is not positive')
    total_steps = {str(d): sum(a for _, a in vals) for d, vals in by_step.items()}
    return {
        'model': 'basal_superior_sweep_v1',
        'calibration_displacement_m': .01,
        'effective_area_m2': total,
        'per_lobe_effective_area_m2': per_lobe,
        'finite_displacement_sensitivity_m2': total_steps,
    }


def build_candidate(base_payload: Path, base_receipt_path: Path, output: Path):
    if sha(base_payload) != BASE_SHA:
        raise ValueError('base NHANAT SHA does not match retained partition proof')
    npz_path = EVIDENCE / 'diaphragm-lobe-shared-seam-prototype-v1.npz'
    seam_audit_path = EVIDENCE / 'diaphragm-lung-shared-seam-audit-v1.json'
    clip_proof_path = EVIDENCE / 'diaphragm-exact-clip-prototype-v1.json'
    z_export_path = EVIDENCE / 'zanatomy-thorax-plus-diaphragm.json'
    expected = ((npz_path, NPZ_SHA), (seam_audit_path, SEAM_AUDIT_SHA),
                (clip_proof_path, CLIP_PROOF_SHA), (z_export_path, Z_EXPORT_SHA))
    for path, digest in expected:
        if sha(path) != digest:
            raise ValueError(f'input identity changed: {path}')
    proof = json.loads(clip_proof_path.read_text())
    seam_audit = json.loads(seam_audit_path.read_text())
    if (proof.get('source_nha_sha256') != BASE_SHA or proof.get('registration_sha256') != REGISTRATION_SHA
            or proof.get('source_z_export_sha256') != Z_EXPORT_SHA
            or seam_audit.get('prototype_npz_sha256') != NPZ_SHA
            or seam_audit.get('source_nha_sha256') != BASE_SHA):
        raise ValueError('clipping proof does not bind exact base/source registration')
    z_export = json.loads(z_export_path.read_text())
    if (z_export.get('source', {}).get('blend_sha256') != Z_BLEND_SHA
            or z_export.get('source', {}).get('registration_sha256') != REGISTRATION_SHA
            or z_export.get('source', {}).get('exporter_sha256') != Z_EXPORTER_SHA
            or [x.get('object_name') for x in z_export.get('objects', [])][-1:] != ['Diaphragm']):
        raise ValueError('Z-Anatomy extended export source identity mismatch')
    receipt = decode_source_receipt(base_receipt_path)
    header, rows = parse_payload(base_payload)
    if set(rows) != set(map(int, receipt['provenance']['source_id_map'].keys())):
        raise ValueError('source receipt stable IDs do not match base payload')
    npz = np.load(npz_path)
    # Replace only the five registered lobe meshes and diaphragm stable ID.
    for sid in LOBE_IDS:
        v = np.asarray(npz[f'lobe_{sid}_vertices'], dtype=np.float32)
        f = np.asarray(npz[f'lobe_{sid}_triangles'], dtype=np.int64)
        current = rows[sid]
        old_v = current['vertices6'][:, :3].astype(np.float64)
        new_set = {tuple(map(float, p)) for p in v}
        if any(tuple(map(float, p.astype(np.float32))) not in new_set for p in old_v):
            raise ValueError(f'lobe {sid} refined interface changed a non-seam source vertex')
        old_volume = abs(signed_volume(old_v, current['faces']))
        new_volume = abs(signed_volume(v.astype(np.float64), f))
        if abs(new_volume - old_volume) > 1e-12:
            raise ValueError(f'lobe {sid} refinement changed source volume beyond 1e-12 m3')
        if not f.size or f.min() < 0 or f.max() >= len(v):
            raise ValueError(f'lobe {sid} candidate faces are out of bounds')
        n = normals(v.astype(np.float64), f).astype(np.float32)
        current['vertices6'] = np.column_stack([v, n]).astype('<f4')
        current['faces'] = f
    d_vertices, d_faces, interfaces, d_topology = make_patches(npz)
    rows[311]['vertices6'] = np.column_stack([d_vertices, normals(d_vertices.astype(np.float64), d_faces).astype(np.float32)]).astype('<f4')
    rows[311]['faces'] = d_faces
    rows[311]['layer'] = 1
    # Recalculate exact source-bound respiratory basis on seam-refined lung meshes.
    area = source_area_basis(rows)
    # Pack existing NHANAT1 ABI5, preserving all records' decoded geometry except IDs 305..311.
    surfaces = []
    vertex_blocks = []
    index_blocks = []
    nv = ni = 0
    for sid in sorted(rows):
        r = rows[sid]
        v6 = np.asarray(r['vertices6'], dtype='<f4')
        faces = np.asarray(r['faces'], dtype=np.uint32)
        if not np.isfinite(v6).all() or (faces.size and (faces.min() < 0 or faces.max() >= len(v6))):
            raise ValueError(f'candidate stable ID {sid} has invalid geometry')
        surfaces.append(RECORD.pack(r['body_index'], nv, len(v6), ni, faces.size, sid, r['layer'], r['flags']))
        vertex_blocks.append(v6.tobytes())
        index_blocks.append((faces.ravel() + nv).astype('<u4').tobytes())
        nv += len(v6)
        ni += faces.size
    payload = HEADER.pack(header[0], header[1], len(surfaces), nv, ni, header[5], header[6])
    payload += b''.join(surfaces) + b''.join(vertex_blocks) + b''.join(index_blocks)
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / 'resting-thorax.nhanatomy'
    payload_path.write_bytes(payload)
    payload_sha = sha(payload_path)
    # Bind source identities, reciprocal material interfaces, and exact derived areas.
    receipt['payload'].update({'path': str(payload_path), 'sha256': payload_sha,
                               'surface_count': len(surfaces), 'vertex_count': nv,
                               'index_count': ni, 'registration_fingerprint32': f'{header[5]:08x}'})
    receipt['functional_bindings']['anatomy_payload_sha256'] = payload_sha
    receipt['functional_bindings']['respiratory_geometry_binding'] = {
        'lung_motion_model': 'basal_superior_sweep_v1',
        'basal_blend_start_m': -.052405, 'basal_blend_span_m': .06,
        'diaphragm_effective_area_m2': area['effective_area_m2'],
        'per_lobe_effective_area_m2': area['per_lobe_effective_area_m2'],
        'footprint_rim_ellipse_m': [.0545, .00055, .0978, .1375],
        'footprint_rim_transition': [.85, .20],
        'footprint_crural_ellipses_m': [[.005, .027, .020, .015], [.005, -.027, .020, .015]],
        'footprint_crural_transition': [.65, .70],
        'parameter_status': 'inferred_reference_registration_not_measured_subject_geometry',
        'finite_displacement_sensitivity_m2': area['finite_displacement_sensitivity_m2'],
        'field_definition': 'g(y)=1-smoothstep((y+0.052405)/0.06); w=w_rim*w_crura; b=g*w; positive qD displaces toward -Y',
        'volume_derivation': 'per-lobe exact signed volume difference at 10 mm caudal field displacement, divided by displacement; common anatomy field; radial qR term owned by native accepted-state deformation',
    }
    receipt['provenance']['zanatomy_diaphragm_source'] = {
        'archive_sha256': Z_ARCHIVE_SHA, 'archive_path': 'Z-Anatomy.zip',
        'license': 'CC-BY-SA-4.0', 'repository': 'https://github.com/Z-Anatomy/Models-of-human-anatomy',
        'revision': Z_REVISION, 'blend_sha256': Z_BLEND_SHA,
        'export_json_path': str(z_export_path), 'export_json_sha256': Z_EXPORT_SHA,
        'exporter_sha256': Z_EXPORTER_SHA, 'registration_sha256': REGISTRATION_SHA,
        'surface_object_name': 'Diaphragm', 'source_vertex_count': 19437,
        'source_triangle_count': 38868, 'declared_modifier': 'Subdivision levels=1 render_levels=2 from source scene',
    }
    receipt['provenance']['diaphragm_lung_interface'] = {
        'npz_path': str(npz_path), 'npz_sha256': NPZ_SHA,
        'exact_partition_proof_path': str(clip_proof_path), 'exact_partition_proof_sha256': sha(clip_proof_path),
        'shared_seam_audit_path': str(seam_audit_path), 'shared_seam_audit_sha256': SEAM_AUDIT_SHA,
        'source_anatomy_payload_sha256': BASE_SHA, 'source_registration_sha256': REGISTRATION_SHA,
        'source_diaph_face_count_before_clip': 38868,
        'retained_clipped_diaph_face_count_before_interface_patches': 33197,
        'source_surface_area_m2': proof.get('source_diaph_surface_area_m2'),
        'retained_clipped_surface_area_m2': proof.get('retained_surface_area_m2'),
        'removed_overlap_sheet_area_m2': proof.get('removed_surface_area_m2_exact_grid'),
        'added_reversed_interface_face_count': int(sum(x['diaphragm_patch_face_count'] for x in interfaces)),
        'added_reversed_interface_surface_area_m2': float(sum(x['patch_area_m2'] for x in interfaces)),
        'interface_rows': interfaces, 'diaphragm_topology_after_patch': d_topology,
        'ownership': 'diaphragm and each lobe retain a reciprocal, exact, coincident source-derived face patch with opposite winding; lobe stable ID owns gas-envelope volume; these are declared tissue interfaces, not nonseam crossings',
        'qualification': 'static Float32 source-bound material interface topology passed; dynamic breathing-cycle clearance remains a separate native geometry gate',
    }
    for sid in LOBE_IDS:
        row = receipt['provenance']['source_id_map'][str(sid)]
        introw = next(x for x in interfaces if x['lung_stable_id'] == sid) if sid != 309 else None
        if introw is not None:
            row.setdefault('repair', {})['diaphragm_interface_partition'] = {
                'method': 'exact_triangle_arrangement_shared_seam_with_reciprocal_patch',
                'interface_face_count': introw['diaphragm_patch_face_count'],
                'shared_boundary_edge_count': introw['shared_boundary_edge_count'],
                'source_displacement_m': 0.0,
            }
        row['registered_mesh_sha256'] = NPZ_SHA
    receipt['provenance']['source_id_map']['311'] = {
        'body_index': int(rows[311]['body_index']), 'head_family': None, 'layer': 1,
        'name': 'Diaphragm', 'provider': 'Z-Anatomy source atlas, exact registered thoracic export',
        'source_member': 'Diaphragm', 'source_sha256': Z_EXPORT_SHA, 'source_owner_metadata': None,
        'repair': {
            'method': 'exact source triangle arrangement; retain cells outside lobe interiors; add reciprocal reversed lobe-base surface fragments',
            'source_coordinates_modified': False,
            'source_face_count': 38868, 'retained_clipped_surface_faces': 33197,
            'reciprocal_interface_faces': int(sum(x['diaphragm_patch_face_count'] for x in interfaces)),
            'final_boundary_edges': d_topology['retained_opening_edge_count'],
            'final_boundary_loops': d_topology['boundary_loop_count'],
            'natural_source_aperture_identity': 'not assigned by source file',
        },
    }
    receipt['thorax_source_volume_m3']['five_lung_envelopes'] = [
        abs(signed_volume(rows[sid]['vertices6'][:, :3].astype(np.float64), rows[sid]['faces'])) for sid in LOBE_IDS]
    receipt['thorax_source_volume_m3']['sum'] = sum(receipt['thorax_source_volume_m3']['five_lung_envelopes'])
    receipt['thorax_source_volume_m3']['interpretation'] = 'registered geometric atlas envelopes include tissue and gas; this geometry is not FRC or lung gas volume'
    receipt['qualification']['diaphragm_lung_interface'] = 'static Float32 material-interface boundary topology passed; exact reciprocal patches; dynamic cycle audit pending'
    receipt['qualification']['self_intersection'] = 'lung lobe self-intersection candidates pass retained source audit; full rest of anatomy not assessed here'
    receipt['qualification']['inter_lobe_intersection'] = 'source lobe interfaces remain on the registered closed-envelope candidates; thorax-wide dynamic interfaces require accepted-state native geometry audit'
    receipt['qualification']['organ interface audit'] = 'diaphragm to five lung lobes has exact declared reciprocal face patches; other organ interfaces not assessed'
    receipt['qualification']['clinical_validation'] = False
    # Source map and payload share the same geometry hash and source receipts.
    receipt_path = output / 'resting-anatomy-receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    manifest = json.loads((base_receipt_path.parent / 'resting-anatomy-manifest.json').read_text())
    manifest.update({
        'payload': receipt['payload'],
        'receipt': {'path': str(receipt_path), 'sha256': sha(receipt_path)},
        'functional_bindings': receipt['functional_bindings'],
        'qualification': receipt['qualification'],
        'native_muscle_surfaces': receipt['provenance']['native_muscle_surfaces'],
        'thorax_source_volume_m3': receipt['thorax_source_volume_m3'],
        'source_surfaces': receipt['provenance']['source_id_map'],
        'mass_geometry_accounting': receipt['mass_geometry_accounting'],
    })
    (output / 'resting-anatomy-manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    for nhtiss in ('bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue',
                   'bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json'):
        source = base_payload.parent / nhtiss
        if source.exists():
            dest = output / nhtiss
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(source.resolve())
    # Independent decode and preservation check for every untouched source surface.
    h2, out_rows = parse_payload(payload_path)
    if h2[2] != 463 or set(out_rows) != set(rows):
        raise ValueError('output surface inventory changed unexpectedly')
    for sid in rows:
        if sid in set(LOBE_IDS) | {311}:
            continue
        if not np.array_equal(out_rows[sid]['vertices6'], rows[sid]['vertices6']) or not np.array_equal(out_rows[sid]['faces'], rows[sid]['faces']):
            raise ValueError(f'unmodified source surface {sid} changed during rebuild')
    if sha(payload_path) != receipt['payload']['sha256'] or sha(receipt_path) != manifest['receipt']['sha256']:
        raise ValueError('payload/receipt hashes diverged')
    return {
        'payload': receipt['payload'], 'receipt_path': str(receipt_path),
        'manifest_path': str(output / 'resting-anatomy-manifest.json'),
        'respiratory_geometry_binding': {
            'model': area['model'], 'effective_area_m2': area['effective_area_m2'],
            'per_lobe_effective_area_m2': area['per_lobe_effective_area_m2'],
            'finite_displacement_sensitivity_m2': area['finite_displacement_sensitivity_m2'],
        },
        'diaphragm_lung_interface': {
            'topology': d_topology,
            'patches': [{k: x[k] for k in ('lung_stable_id','diaphragm_patch_face_start',
                'diaphragm_patch_face_count','patch_area_m2','shared_boundary_edge_count',
                'registered_lung_face_count')} for x in interfaces],
            'added_reversed_interface_face_count': int(sum(x['diaphragm_patch_face_count'] for x in interfaces)),
        },
        'qualification': receipt['qualification'],
        'preserved_non_target_surface_count': 463 - len(LOBE_IDS) - 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-payload', type=Path, default=BASE / 'resting-thorax.nhanatomy')
    ap.add_argument('--base-receipt', type=Path, default=BASE / 'resting-anatomy-receipt.json')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(build_candidate(args.base_payload, args.base_receipt, args.output), indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
