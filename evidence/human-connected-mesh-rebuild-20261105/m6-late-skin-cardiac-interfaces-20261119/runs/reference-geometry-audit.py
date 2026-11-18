"""Check intersections in a frozen native MRVPACK2 using retained Float32 predicates."""
import argparse
import collections
import hashlib
import json
import mmap
from pathlib import Path
import struct

import numpy as np
from numilab_human import cardiac_cavity_intersections as intersections
from numilab_human import common_atlas_skin_clearance as clearance

SKIN = (51007, 1)
BANK_CLASSES = {51004, 51005, 51006, 51010, 51011, 51012, *range(51020, 51027)}

def world_positions(positions, instance):
    translation = np.array(instance[:3], dtype=np.float64)
    scale = instance[3]
    vector = np.array(instance[4:7], dtype=np.float64)
    real = instance[7]
    assert abs(np.dot(vector, vector) + real * real - 1) < 1e-4
    assert scale > 0
    if scale == 1 and real == 1 and not np.any(translation) and not np.any(vector):
        return positions
    local = np.asarray(positions, dtype=np.float64) * scale
    cross = np.cross(vector, local)
    return np.ascontiguousarray(local + 2 * (real * cross + np.cross(vector, cross)) + translation,
                                dtype=np.float32)

def primitives(mapped, body_poses, *, skin_only=False):
    header = clearance._PACK_HEADER
    directory = clearance._PACK_DIRECTORY
    magic, version, count, *_ = header.unpack_from(mapped)
    assert magic == b'MRVPACK2' and version == 2
    sections = {}
    for i in range(count):
        row = directory.unpack_from(mapped, header.size + i * directory.size)
        sections[row[0]] = row
    vertex, index, primitive, instance = (sections[x] for x in range(2, 6))
    assert tuple(row[5] for row in (vertex, index, primitive, instance)) == (80, 4, 64, 80)
    vertices = np.ndarray((vertex[4],20), dtype='<f4', buffer=mapped, offset=vertex[2])[:, :3]
    indices = np.ndarray(index[4], dtype='<u4', buffer=mapped, offset=index[2])
    for i in range(primitive[4]):
        first, count, _, instance_id, semantic, _, _, stable = struct.unpack_from(
            '<8I', mapped, primitive[2] + i * 64)
        key = semantic, stable
        if skin_only and key != SKIN:
            continue
        if key != SKIN and semantic not in BANK_CLASSES:
            continue
        assert first+count <= len(indices) and count % 3 == 0 and instance_id < instance[4]
        transform = struct.unpack_from('<8f12I', mapped, instance[2] + instance_id * 80)
        assert transform[10] in (0, 3), (i, semantic, stable, transform)
        faces = indices[first:first+count].reshape(-1,3)
        ids = np.unique(faces)
        assert ids[-1] < len(vertices)
        positions = world_positions(vertices[ids], transform)
        if transform[10] == 3:
            body = body_poses[transform[9]]
            frozen = (*body['position_m'], 1.0, *body['quaternion_xyzw'])
            positions = world_positions(positions,frozen)
        assert np.isfinite(positions).all()
        yield (semantic, stable), i, positions, np.searchsorted(ids, faces)

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def run(pack, output):
    receipt = json.loads(pack.with_suffix('.receipt.json').read_text())
    assert receipt['accepted_pack_path'] == str(pack)
    assert receipt['pack_file_sha256'] == sha(pack)
    assert receipt['physical_endpoint'] == 'accepted'
    assert receipt['surface_audit_endpoint'] == 'passed'
    body_poses = {b['body_index']:b for b in receipt['accepted_body_poses']}
    assert len(body_poses)==receipt['accepted_body_pose_count']
    with pack.open('rb') as stream, mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapped:
        skin = [(v,f) for key,_,v,f in primitives(mapped,body_poses,skin_only=True)]
        assert len(skin) == 1
        vertices, faces = skin[0]
        skin_records = clearance._exact_surface_records(vertices,faces)
        skin_index = intersections._prepare_surface_aabb(skin_records)
        float_index = clearance._float32_skin_aabb_index(skin_records)
        result = {
            'accepted_step':receipt['accepted_step'],
            'accepted_time_s':receipt['accepted_time_s'],
            'accepted_pack_sha256':receipt['pack_file_sha256'],
            'accepted_root_fingerprint_hex':receipt['accepted_root_fingerprint_hex'],
            'method':'Captured positions with frozen rigid instances composed once and rounded to Float32 world coordinates. Exact retained integer triangle predicates. No extra seam exemptions.',
            'limit':'Skin surface intersections only. No through-volume containment test, pairwise interior audit, or dense motion coverage. Ocular interfaces are reported separately from other interfaces.',
            'self_intersections':intersections._audit_pair_prepared_first(skin_index,skin_records,same_surface=True),
            'surfaces':[],
        }
        counts = collections.Counter()
        pairs = collections.Counter()
        with output.open('x') as out:
            for key, primitive_id, vertices, full_faces in primitives(mapped,body_poses):
                if key == SKIN:
                    continue
                chosen, broad_count = clearance._candidate_target_rows_for_skin_aabb(full_faces,vertices,float_index)
                ids = np.unique(full_faces[chosen])
                selected_faces = np.searchsorted(ids,full_faces[chosen])
                recs = clearance._exact_surface_records(vertices[ids],selected_faces)
                recs = [(a,b,c,int(chosen[d]),e) for a,b,c,d,e in recs]
                audit = intersections._audit_pair_prepared_first(skin_index,recs)
                assert audit['aabb_candidate_pairs'] == broad_count
                ocular = key[0] == 51010 and 381 <= key[1] <= 397
                row = {'semantic':key[0],'stable_id':key[1],'primitive':primitive_id,
                       'ocular_source_id':ocular,'target_faces':len(full_faces),**audit}
                counts[key[0]] += 1
                pairs[key[0]] += audit['count']
                if audit['count']:
                    print(key, primitive_id, audit['count'], flush=True)
                result['surfaces'].append(row)
            result['target_banks_by_semantic'] = dict(counts)
            result['crossing_pairs_by_semantic'] = dict(pairs)
            json.dump(result,out,indent=2)
            out.write('\n')
    print(json.dumps({'source':str(pack),'crossing_pairs':dict(pairs),
                      'skin_self_pairs':result['self_intersections']['count']}),flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('pack',type=Path)
    p.add_argument('report',type=Path)
    args = p.parse_args()
    run(args.pack.resolve(), args.report.resolve())
