#!/usr/bin/env python3
"""Compile reference forearm bindings into bounded existing NHTISS4 sources.

Reuse the frozen original source; split added source tuples into ABI5 payloads.
New PQ and barycentric-parent carrier normals follow the corrected child surface.
Kind 2 uses the existing parent material point and its declared parent offset.
No native or anatomical admission is provided by this load-time compiler.
"""
from pathlib import Path
import argparse, hashlib, json, pickle, struct
from collections import defaultdict
import numpy as np
from numilab_human import passive_attachment_composition as pc

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_parent_point(point, oldpoint, deltas, child_gid, parent_gid, kind):
    """Check the single-add Float32 offset of the owning material parent."""
    offset_gid = parent_gid if kind == 2 else child_gid
    delta = deltas.get(offset_gid)
    expected = oldpoint if delta is None else np.asarray(oldpoint + delta, dtype='<f4')
    require(np.max(np.abs(np.asarray(point, dtype='<f4').astype(float) -
                          expected.astype(float))) < 2e-6,
            'source offset differs from declared material parent correction')

def carrier_normal(kind, old, child_normal):
    return child_normal.tobytes() if kind in (1, 2) else old[12:24]

def validate_lineage(lineage, grouped):
    """Require explicit source-face ancestry for every new material child."""
    records = lineage.get('new_vertices', {})
    children = {g for g, terms in grouped.items() if any(t[2] == 2 for t in terms)}
    require(bool(children), 'lineage supplied without barycentric material children')
    for gid in children:
        r = records.get(str(gid))
        require(isinstance(r, dict), 'missing new child source-face lineage')
        face = r.get('nearest_source_face', {})
        bary = np.asarray(face.get('barycentric', []), dtype=float)
        require(bary.shape == (3,) and np.isfinite(bary).all() and
                np.all(bary >= -1e-12) and abs(float(bary.sum()) - 1) <= 1e-12,
                'invalid source-face barycentric ancestry')
        require(int(face.get('stable_id', -1)) in (132, 134) and
                int(face.get('face_index', -1)) >= 0 and
                len(face.get('vertex_gids', [])) == 3, 'invalid source face identity')
        offset = np.asarray(face.get('inferred_source_offset_from_face_base_m', []), dtype=float)
        require(offset.shape == (3,) and np.isfinite(offset).all(),
                'missing inferred source displacement')
        parents = r.get('material_parents', [])
        require(0 < len(parents) <= 4, 'invalid material parent support')
        weights = [float(p['coefficient']) for p in parents]
        require(all(np.isfinite(w) and w > 0 for w in weights) and
                abs(sum(weights)-1) <= 1e-12, 'invalid material parent partition')
        require({int(p['global_gid']) for p in parents} ==
                {t[0] for t in grouped[gid] if t[2] == 2},
                'field parent identity differs from lineage')

def main():
    ap = argparse.ArgumentParser()
    for key in ('candidate', 'field', 'carriers', 'overlay', 'output'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--lineage', type=Path)
    a = ap.parse_args()
    require(not a.output.exists(), 'preserve prior output')
    pins = {str(p.resolve()): sha(p) for p in
            (Path(__file__), a.candidate, a.field, a.carriers, a.overlay, Path(pc.__file__))}
    if a.lineage is not None:
        pins[str(a.lineage.resolve())] = sha(a.lineage)
    c = pickle.loads(a.candidate.read_bytes())
    z = np.load(a.field, allow_pickle=False)
    overlay = json.loads(a.overlay.read_text())
    if 'candidate_sha256' in overlay:
        require(overlay['candidate_sha256'] == sha(a.candidate), 'offset candidate differs')
    if 'field_carrier_sha256' in overlay:
        require(overlay['field_carrier_sha256'] == sha(a.carriers), 'offset carrier differs')
    deltas = {int(g): np.asarray(d, dtype='<f4') for g, d in
              overlay['source_offsets_by_global_gid'].items()}
    require(all(d.shape == (3,) and np.isfinite(d).all() and
                np.linalg.norm(d.astype(float)) <= .003 + 1e-9
                for d in deltas.values()), 'invalid reference offset')
    t = pc._read_nhtiss4(a.carriers)
    rows = {int(r[6]): pc._row_slices(t, r) for r in t['records']}
    require(set(rows) == {132, 134}, 'unexpected carrier inventory')
    # Every frozen097 source vertex has all21 positive one-body carriers.
    lookup, signatures = {}, {}
    original = set(map(int,z['term_parent_gid'][np.isin(z['term_kind_old_parent_or_pq'], (0, 2))]))
    for sid, surface in c['surfaces'].items():
        for gid, terms in zip(surface['global_vertex_ids'], surface['parent_binding_terms']):
            if int(gid) not in original:
                continue
            for source, rs, vi, coefficient in terms:
                rs, vi = int(rs), int(vi)
                require(source == 0 and rs in rows, 'unsupported carrier ancestry')
                row = rows[rs]
                vb = row['vertex_bytes'][vi*56:(vi+1)*56]
                lanes = struct.unpack_from('<4I', vb, 24)
                require(struct.unpack_from('<4f', vb, 40) == (1., 0., 0., 0.),
                        'carrier must have one body owner')
                binding = row['binding_bytes'][lanes[0]*36:(lanes[0]+1)*36]
                core = struct.unpack_from('<I', binding)[0]
                key = (int(gid), core)
                signature = (vb[:12], binding)
                require(signatures.setdefault(key, signature) == signature,
                        'coincident source or binding differs')
                lookup.setdefault(key, (rs, vi))
    require(z['canonical_binding_bytes'].item() == rows[132]['binding_bytes'] ==
            rows[134]['binding_bytes'], 'canonical binding table differs')
    term_keys = ('term_child_gid', 'term_parent_gid', 'term_core_body_index',
                 'term_kind_old_parent_or_pq', 'term_coefficient_binary64',
                 'term_carrier_source_position_f32')
    require(len({len(z[key]) for key in term_keys}) == 1,
            'field term array lengths differ')
    ids = z['canonical_output_gids']
    require(ids.ndim == 1 and np.all(np.diff(ids) > 0),
            'canonical output identities must be sorted and unique')
    grouped = defaultdict(list)
    templates = {}
    for (pg,core), (rs,vi) in lookup.items():
        templates.setdefault(core,(rs,vi))
    for g,pg,core,kind,coefficient,point in zip(
            z['term_child_gid'], z['term_parent_gid'], z['term_core_body_index'],
            z['term_kind_old_parent_or_pq'],z['term_coefficient_binary64'],
            z['term_carrier_source_position_f32']):
        g,pg,core,kind = int(g),int(pg),int(core),int(kind)
        coefficient = float(coefficient)
        require(kind in (0,1,2) and np.isfinite(coefficient) and coefficient > 0,
                'invalid hybrid parent term')
        point = np.asarray(point,dtype='<f4')
        require(point.shape == (3,) and np.isfinite(point).all(),'invalid source point')
        grouped[g].append((pg,core,kind,coefficient,point))
    has_barycentric = any(t[2] == 2 for ts in grouped.values() for t in ts)
    require(not has_barycentric or a.lineage is not None, 'kind 2 requires source-face lineage')
    if a.lineage is not None:
        validate_lineage(json.loads(a.lineage.read_text()), grouped)
    extras = []
    seed = rows[132]
    require(rows[134]['binding_bytes'] == seed['binding_bytes'], 'different carrier registration tables')
    capacity = 1_000_000 - seed['vertex_count']
    require(capacity > 0, 'no bounded source capacity')
    cache, surfaces = {}, []
    maximum_error = 0.
    for sid in (132, 134):
        s = c['surfaces'][sid]
        require(np.array_equal(s['global_vertex_ids'], z[f'output_gids_{sid}']),
                'output vertex order differs')
        require(np.array_equal(s['faces'], z[f'output_faces_{sid}']), 'topology differs')
        expected = np.asarray(s['vertices_float32_m'], dtype='<f4').copy()
        for local_gid, gid0 in enumerate(s['global_vertex_ids']):
            if int(gid0) in deltas:
                expected[local_gid] += deltas[int(gid0)]
        require(np.array_equal(expected, z['canonical_corrected_source_positions_f32'][
            np.searchsorted(z['canonical_output_gids'], s['global_vertex_ids'])]),
            'actual corrected child source differs from geometry and declared offsets')
        faces = np.asarray(s['faces'], dtype=np.int64)
        area_vectors = np.cross(expected[faces[:,1]].astype(float)-expected[faces[:,0]],
                                expected[faces[:,2]].astype(float)-expected[faces[:,0]])
        require(np.isfinite(area_vectors).all() and
                np.all(np.linalg.norm(area_vectors,axis=1) > 0),
                'degenerate corrected child surface')
        normals = np.zeros(expected.shape,dtype=np.float64)
        for lane in range(3):
            np.add.at(normals,faces[:,lane],area_vectors)
        lengths = np.linalg.norm(normals,axis=1)
        require(np.isfinite(lengths).all() and np.all(lengths > 0),
                'undefined corrected child vertex normal')
        normals = np.asarray(normals/lengths[:,None],dtype='<f4')
        output = []
        for local_vertex, raw_gid in enumerate(s['global_vertex_ids']):
            gid = int(raw_gid)
            terms = []
            for pg,core,kind,coefficient,point in grouped[gid]:
                if kind in (0, 2):
                    require((pg,core) in lookup,'positive original carrier missing')
                    rs,vi = lookup[pg,core]
                else:
                    require(core in (92,93) and core in templates,'invalid PQ owner')
                    rs,vi = templates[core]
                row = rows[rs]
                old = row['vertex_bytes'][vi*56:(vi+1)*56]
                oldpoint = np.frombuffer(old,dtype='<f4',count=3)
                if kind in (0, 2):
                    validate_parent_point(point, oldpoint, deltas, gid, pg, kind)
                else:
                    ids = z['canonical_output_gids']
                    k = int(np.searchsorted(ids,gid))
                    require(k < len(ids) and int(ids[k]) == gid and
                            np.array_equal(point,z['canonical_corrected_source_positions_f32'][k]),
                            'PQ target differs from corrected anatomical child point')
                normal = carrier_normal(kind, old, normals[local_vertex])
                if point.tobytes() == oldpoint.tobytes() and normal == old[12:24]:
                    # Source zero is the existing, unmodified frozen carrier.
                    term = [0,rs,vi,coefficient]
                else:
                    key = (rs,vi,point.tobytes(),normal)
                    if key not in cache:
                        new = bytearray(old); new[:12] = point.tobytes()
                        new[12:24] = normal
                        extra = len(extras)
                        extras.append(bytes(new))
                        cache[key] = [1 + extra//capacity,132,
                                      seed['vertex_count'] + extra%capacity]
                        require(new[24:] == old[24:],
                                'registered binding or one-body ownership changed')
                        require(kind in (1, 2) or new[12:24] == old[12:24],
                                'original material carrier normal changed')
                    term = cache[key] + [coefficient]
                terms.append(term)
            error = abs(sum(t[3] for t in terms)-1)
            maximum_error = max(maximum_error, error)
            require(0 < len(terms) <= 128 and error <= 1e-12, 'invalid partition')
            output.append(terms)
        surfaces.append({'stable_id': sid, 'vertices': output})
    a.output.mkdir()
    descriptors = [{'path':str(a.carriers.resolve()),'sha256':sha(a.carriers)}]
    chunks = []
    # Each additional source retains the original complete row132 mesh and
    # registered binding table. Extra tuples are unrendered transform carriers,
    # with no added mass, physical force, or independently advancing state.
    require((len(extras)+capacity-1)//capacity <= 3, 'existing source budget exceeded')
    for ordinal, begin in enumerate(range(0,len(extras),capacity),1):
        chunk = extras[begin:begin+capacity]
        vb = seed['vertex_bytes'] + b''.join(chunk)
        count = len(vb)//56
        require(count <= 1_000_000, 'bounded source vertex inventory exceeded')
        record = struct.pack('<8I',0,seed['binding_count'],0,count,0,
                             seed['index_count'],132,seed['layer'])
        header = struct.pack('<8s6I32s',b'NHTISS4\0',5,1,seed['binding_count'],
                             count,seed['index_count'],t['fingerprint'],t['source_digest'])
        payload = header+record+seed['binding_bytes']+vb+np.asarray(
            seed['local_faces'],dtype='<u4').tobytes()
        target = a.output/f'reference-field-added-carriers-{ordinal}.nhtissue'
        target.write_bytes(payload)
        decoded = pc._read_nhtiss4(target)
        newrow = pc._row_slices(decoded,decoded['records'][0])
        require(newrow['binding_bytes'] == seed['binding_bytes'] and
                np.array_equal(newrow['local_faces'],seed['local_faces']) and
                newrow['vertex_bytes'][:len(seed['vertex_bytes'])] == seed['vertex_bytes'],
                'frozen seed geometry or registration changed')
        descriptors.append({'path':target.name,'sha256':sha(target)})
        chunks.append({'path':target.name,'extra_tuples':len(chunk),
                       'vertices':count,'bytes':target.stat().st_size})
    parent = a.output/'parent-surfaces.json'
    parent.write_text(json.dumps(surfaces,separators=(',',':'))+'\n')
    (a.output/'parent-source-payloads.json').write_text(
        json.dumps(descriptors,separators=(',',':'))+'\n')
    require(all(sha(p) == digest for p, digest in pins.items()), 'input changed')
    report = {
        'inputs_sha256': pins, 'source_payloads':descriptors, 'additional_chunks':chunks,
        'parent_surfaces_sha256': sha(parent),
        'original_carrier_vertices': t['vertex_count'], 'added_offset_carriers': len(extras),
        'maximum_partition_error': maximum_error,
        'runtime_owner': 'existing NHTISS4 ABI5 and passive_tissue_parent_bindings',
        'source_interpretation': 'Declared passive reference geometry and material attachment field. Ordered positive coefficients and carrier coordinates are copied exactly from the pinned field; coefficient-reference provenance belongs to that field construction. Declared source offsets are bounded to 3mm per canonical GID; new material-child source-face ancestry and inferred offsets are pinned separately. No measured-person or original097 covariance claim.',
        'new_carrier_normal_owner': 'Area-weighted corrected child row source surface; private interface rows retain their own orientation. Positions, ordered positive coefficients, registration and physical state are unchanged by normal reconstruction.',
        'source_face_lineage_sha256': sha(a.lineage) if a.lineage else None, 'barycentric_parent_material_children': sum(any(t[2] == 2 for t in ts) for ts in grouped.values()), 'original_carrier_bytes_preserved': True, 'maximum_source_offset_m':max((float(np.linalg.norm(d.astype(float))) for d in deltas.values()),default=0.), 'physical_routes_changed': False,
        'native_admission': False,
    }
    (a.output/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'inputs_sha256'}))

if __name__ == '__main__':
    main()
