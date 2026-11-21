"""Compare the source registration with the same retained accepted audit banks."""
import hashlib
import json
from pathlib import Path
import runpy
import struct
import subprocess
import sys

import numpy as np

run = Path(sys.argv[1]).resolve()
work = run.parent.parent
audit = work / 'resting-surface-clearance-001'
sys.path[:0] = [str(audit / 'src'), str(work / 'resting-clearance-and-atrium-005')]
from fast_skin_intersection_audit import install_for_float32

install_for_float32()
a = runpy.run_path(str(audit / 'audit.py'))
clearance = a['clearance']
intersections = a['intersections']
receipt = json.loads((run / 'native/accepted-geometry/step-16.receipt.json').read_text())
terminal = json.loads((run / 'exact-skin-surface.json').read_text())
assert terminal['accepted_pack_sha256'] == receipt['pack_file_sha256']
pose = {p['body_index']: p for p in receipt['initial_anatomical_registration']['body_poses']}
source = run / 'native/resting-human.mrvpack.zst'
data = subprocess.check_output(['zstd', '-dc', str(source.resolve())])
assert hashlib.sha256(data).hexdigest() == receipt['base_pack_file_sha256']

vertices, faces = [(v, f) for _, _, v, f in a['primitives'](data, pose, skin_only=True)][0]
records = clearance._exact_surface_records(vertices, faces)
indexed = intersections._prepare_surface_aabb(records)
broad = clearance._float32_skin_aabb_index(records)
top = sorted((r for r in terminal['surfaces'] if r['semantic'] == 51005),
             key=lambda r: -r['count'])

header = clearance._PACK_HEADER
directory = clearance._PACK_DIRECTORY
magic, version, count, *_ = header.unpack_from(data)
assert (magic, version) == (b'MRVPACK2', 2)
sections = {s[0]: s for i in range(count)
            if (s := directory.unpack_from(data, header.size + i * directory.size))}
vertex, index, primitive, instance = (sections[x] for x in range(2, 6))
assert tuple(row[5] for row in (vertex, index, primitive, instance)) == (80, 4, 64, 80)
world = np.ndarray((vertex[4],20), dtype='<f4', buffer=data, offset=vertex[2])[:, :3]
indices = np.ndarray(index[4], dtype='<u4', buffer=data, offset=index[2])

def bed_clearance(capture, points):
    gap = points[:, 2]
    assert np.isfinite(gap).all()
    return {'capture': capture, 'rows': len(gap), 'minimum_gap_m': float(gap.min()),
            'nonpositive_gap_rows': int(np.count_nonzero(gap <= 0)),
            'within_1um_rows': int(np.count_nonzero(gap <= 1e-6)),
            'within_1mm_rows': int(np.count_nonzero(gap <= 1e-3))}

native_log = (run / 'stdout.log').read_text()
assert 'resting_fem_skin_bed_source=enabled object=15 outer_nodes=76153 proxy_count=1 normal=0,0,1 plane_offset_m=0 ' in native_log
accepted_source = run / 'native/accepted-geometry/step-16.mrvpack.zst'
accepted_pack = subprocess.check_output(['zstd', '-dc', str(accepted_source.resolve())])
assert hashlib.sha256(accepted_pack).hexdigest() == receipt['pack_file_sha256']
accepted_vertices, accepted_faces = [(v,f) for _,_,v,f in a['primitives'](
    accepted_pack, {}, skin_only=True)][0]
del accepted_pack
assert len(vertices) == len(accepted_vertices) == 76153 and np.array_equal(faces, accepted_faces)
print(json.dumps({'bed_geometry': [bed_clearance('initial_source_pack', vertices),
                                  bed_clearance('accepted_step_16', accepted_vertices)],
                  'maximum_surface_displacement_m': float(np.linalg.norm(
                      accepted_vertices.astype(np.float64)-vertices, axis=1).max())}), flush=True)

print(json.dumps({'base_pack_sha256': receipt['base_pack_file_sha256'],
                  'accepted_pack_sha256': receipt['pack_file_sha256'],
                  'accepted_time_s': receipt['accepted_time_s'],
                  'scope': 'authored source registration with captured initial owner transforms versus accepted rendered endpoint; exact Float32 triangle pairs'}), flush=True)
for row in top:
    bank = row['primitive']
    first, index_count, _, instance_id, semantic, _, _, stable = struct.unpack_from(
        '<8I', data, primitive[2] + bank * 64)
    assert (semantic, stable, index_count // 3) == (row['semantic'], row['stable_id'], row['target_faces'])
    transform = struct.unpack_from('<8f12I', data, instance[2] + instance_id * 80)
    assert transform[10] in (0, 3)
    summary = {'stable_id': stable, 'primitive': bank, 'binding': transform[10],
               'body': transform[9], 'accepted_pairs': row['count']}
    if transform[10] == 3 and transform[9] not in pose:
        summary['source_pairs'] = None
        print(json.dumps(summary), flush=True)
        continue
    bank_faces = indices[first:first+index_count].reshape(-1, 3)
    used = np.unique(bank_faces)
    bank_points = a['world_positions'](world[used], transform)
    if transform[10] == 3:
        body = pose[transform[9]]
        frozen = (*body['position_m'], 1.0, *body['quaternion_xyzw'])
        bank_points = a['world_positions'](bank_points, frozen)
    bank_faces = np.searchsorted(used, bank_faces)
    chosen, broad_count = clearance._candidate_target_rows_for_skin_aabb(bank_faces, bank_points, broad)
    used = np.unique(bank_faces[chosen])
    bank_records = clearance._exact_surface_records(bank_points[used], np.searchsorted(used, bank_faces[chosen]))
    bank_records = [(p, q, b, int(chosen[d]), e) for p,q,b,d,e in bank_records]
    tested = intersections._audit_pair_prepared_first(indexed, bank_records)
    assert tested['aabb_candidate_pairs'] == broad_count
    source_pairs = {tuple(pair) for pair in tested['triangle_pairs']}
    accepted_pairs = {tuple(pair) for pair in row['triangle_pairs']}
    summary.update(source_pairs=len(source_pairs), same_pairs=len(source_pairs & accepted_pairs),
                   initial_only_pairs=len(source_pairs - accepted_pairs), accepted_only_pairs=len(accepted_pairs - source_pairs),
                   source_skin_triangle_count=len({pair[0] for pair in source_pairs}),
                   smallest_source_witness=min(source_pairs) if source_pairs else None)
    print(json.dumps(summary), flush=True)
