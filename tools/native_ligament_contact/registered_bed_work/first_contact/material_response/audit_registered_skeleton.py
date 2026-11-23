"""Audit source, body-local rest, and rendered release bone volumes on matched native evidence."""
import argparse
import hashlib
import json
import mmap
import runpy
import struct
import zipfile
from itertools import combinations
from pathlib import Path

import manifold3d as md
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


BONE_SEMANTIC = 51004
HEADER = struct.Struct('<8s5I32s')
RECORD = struct.Struct('<6I8fI')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_geometry(data):
    points = []
    triangles = []
    for line in data.splitlines():
        fields = line.split()
        if fields[:1] == [b'v']:
            points.append([float(s) for s in fields[1:4]])
        elif fields[:1] == [b'f']:
            ring = [int(s.split(b'/')[0]) - 1 for s in fields[1:]]
            if len(ring) < 3 or min(ring) < 0:
                raise ValueError('Invalid source polygon')
            triangles.extend((ring[0], ring[i], ring[i+1]) for i in range(1, len(ring)-1))
    return np.asarray(points), np.asarray(triangles)


def closed_solid(points_mm, triangles):
    unique, group = np.unique(points_mm, axis=0, return_inverse=True)
    faces = group[triangles]
    edges = np.sort(faces[:, [[0,1], [1,2], [2,0]]].reshape(-1, 2), axis=1)
    if not np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2):
        raise ValueError('Signed volume requires a closed, welded source')
    centered = unique - unique.mean(axis=0)
    vol6 = np.einsum('ij,ij->', centered[faces[:,0]],
                     np.cross(centered[faces[:,1]], centered[faces[:,2]]))
    if vol6 < 0:
        faces = faces[:, [0,2,1]]
    shape = md.Manifold(md.Mesh64(np.ascontiguousarray(unique, dtype='f8'),
                                  np.ascontiguousarray(faces, dtype='u8')))
    if shape.status() != md.Error.NoError or shape.volume() <= 0:
        raise ValueError('Invalid bone solid')
    return unique.min(axis=0), unique.max(axis=0), shape


def intersection_mm3(a, b):
    if np.any(a[1] < b[0]) or np.any(b[1] < a[0]):
        return 0.
    overlap = a[2] ^ b[2]
    if overlap.status() != md.Error.NoError:
        raise ValueError('Indeterminate cortical intersection')
    return max(0., float(overlap.volume()))


def at_pose(local, state):
    return (Rotation.from_quat(state['orientation_world_xyzw']).apply(local)
            + state['position_world_m']).astype('f4').astype(float)


def verify(job, reader, atlas):
    launch_file = job/'launch.json'
    launch = json.loads(launch_file.read_bytes())
    asset = Path(launch['argv'][3])
    manifest_file = asset.with_suffix('.manifest.json')
    manifest = json.loads(manifest_file.read_bytes())
    if digest(asset) != launch['pins'][str(asset)]:
        raise ValueError('Launch used a different bone payload')
    if digest(atlas) != manifest['source']['bodyparts']['archives'][0]['sha256']:
        raise ValueError('Source archive differs from the declared registration')
    snapshot = job/'native-initial-bodies.json'
    bodies = {b['body_index']: b for b in json.loads(snapshot.read_bytes())['all_bodies']}
    pack = job/'native-traces/resting-human.mrvpack'
    render_pose = {i: dict(position_m=b['current']['position_world_m'],
                           quaternion_xyzw=b['current']['orientation_world_xyzw'])
                   for i,b in bodies.items()}
    decode = runpy.run_path(str(reader))['primitives']
    rendered = {}
    with pack.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as raw:
        for (semantic, stable), _, vertices, faces in decode(raw, render_pose):
            if semantic == BONE_SEMANTIC:
                if stable in rendered:
                    raise ValueError(f'Duplicate native bone bank {stable}')
                rendered[stable] = vertices.copy(), faces.copy()
    payload = asset.read_bytes()
    magic, abi, count, nvert, nindex, _, _ = HEADER.unpack_from(payload)
    first_vert = HEADER.size + RECORD.size * count
    first_index = first_vert + nvert * 24
    anchors = manifest['source']['anchors']
    if (magic != b'NHBONES1' or abi != 3 or len(anchors) != count
            or len(payload) != first_index + 4*nindex or len(rendered) != count):
        raise ValueError('Incomplete bone state or unmatched payload layout')
    names, stages, parity = {}, {}, {}
    with zipfile.ZipFile(atlas) as archive:
        members = {Path(n).stem:n for n in archive.namelist() if n.endswith('.obj')}
        for i, anchor in enumerate(anchors):
            row = RECORD.unpack_from(payload, HEADER.size + i*RECORD.size)
            body, start, nv, offset, ni, stable = row[:6]
            if body != anchor['core_body_index'] or stable in names:
                raise ValueError('Duplicate or mismatched rigid owner')
            original = archive.read(members[anchor['member_id']])
            if hashlib.sha256(original).hexdigest() != anchor['member_sha256']:
                raise ValueError('Source member no longer matches registration')
            xyz, src_faces = source_geometry(original)
            source = closed_solid(xyz, src_faces)  # Source OBJ coordinates are millimetres.
            if start+nv > nvert or offset+ni > nindex or ni%3:
                raise ValueError('Payload geometry outside its bank')
            points = np.frombuffer(payload,'<f4',6*nv,first_vert+24*start).reshape(-1,6)[:,:3]
            faces = np.frombuffer(payload,'<u4',ni,first_index+4*offset).astype('i8').reshape(-1,3)-start
            if np.any(faces<0) or np.any(faces>=nv):
                raise ValueError('Payload facet references another bank')
            local = Rotation.from_quat(row[9:13]).apply(row[13]*points.astype(float))+row[6:9]
            rest = at_pose(local, bodies[body]['rest'])
            live = at_pose(local, bodies[body]['current'])[np.unique(faces)]
            vertices, tri = rendered[stable]
            used = vertices[np.unique(tri)]
            error = max(cKDTree(used).query(live)[0].max(), cKDTree(live).query(used)[0].max())
            if error > 3e-6:
                raise ValueError(f'Native pose reconstruction mismatch on bone {stable}: {error} m')
            names[stable] = (anchor['myosim_body'],body)
            parity[stable] = float(error)
            stages[stable] = (source,closed_solid(1000*rest,faces),closed_solid(1000*vertices.astype(float),tri))
    pairs = []
    for i,j in combinations(sorted(stages),2):
        vals = [intersection_mm3(x,y) for x,y in zip(stages[i],stages[j])]
        # Retain source-only and rest-only intersections as well as release hits.
        if max(vals) <= 1e-9:
            continue
        pairs.append(dict(stable_ids=[i,j], bodies=[names[i][1],names[j][1]],
                          names=[names[i][0],names[j][0]], same_rigid_owner=names[i][1]==names[j][1],
                          source_mm3=vals[0],rest_mm3=vals[1],native_mm3=vals[2]))
    pairs.sort(key=lambda p:-p['native_mm3'])
    live = [p for p in pairs if p['native_mm3']>1e-9]
    return dict(native_launch=digest(launch_file),native_pose=digest(snapshot),native_pack=digest(pack),
                bone_asset=digest(asset),manifest=digest(manifest_file),source_archive=digest(atlas),
                pack_reader=digest(reader),closed_bone_banks=len(stages),all_pairs=count*(count-1)//2,
                positive_release_pairs=len(live),separate_rigid_owner_release_pairs=sum(not p['same_rigid_owner'] for p in live),
                max_reconstruction_error_m=max(parity.values()),reporting_threshold_mm3=1e-9,pairs=pairs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('job','reader','atlas','output'):
        parser.add_argument(arg,type=Path)
    parser.add_argument('--require-clear-separate-owners',action='store_true')
    args = parser.parse_args()
    report = verify(args.job,args.reader,args.atlas)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if args.require_clear_separate_owners and report['separate_rigid_owner_release_pairs']:
        parser.exit(1, f"{report['separate_rigid_owner_release_pairs']} separate-owner bone pairs overlap at release\n")
