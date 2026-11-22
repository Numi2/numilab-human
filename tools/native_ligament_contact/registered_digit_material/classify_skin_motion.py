"""Distinguish transverse physical crossings from touching at the measured endpoints."""
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

from functools import lru_cache


def check_binding(case, stable, world, faces, poses, api):
    """Replay the installed source-to-skeleton mapping, then check both surfaces."""
    import struct
    import numpy as np
    from scipy.spatial.transform import Rotation
    from extensor_transfer import transfer

    launch = json.loads((case/'launch.json').read_text())
    argv = launch['argv']
    source = Path(argv[argv.index('--soft-tissue-payload')+1])
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    manifest = json.loads(source.with_suffix('.manifest.json').read_text())
    if digest!=launch['pins'][str(source)] or digest!=manifest['payload']['sha256']:
        raise ValueError('Tissue ownership differs from the native launch')
    magic,version,nrec,nbind,nvertex,nindex,_ = struct.unpack_from('<8s6I',data)
    if magic!=b'NHTISS4\0' or version!=5:
        raise ValueError('Unsupported tissue binding format')
    records = np.frombuffer(data,'<u4',8*nrec,64).reshape(-1,8)
    (record,) = records[records[:,6]==stable]
    first_binding,count,first_vertex,n,first_index,ni,*_ = map(int,record)
    vertex_offset = 64+32*nrec+36*nbind
    indices = np.frombuffer(data,'<u4',ni,vertex_offset+56*nvertex+4*first_index).reshape(-1,3)
    used = np.unique(indices)
    if used[0]<first_vertex or used[-1]>=first_vertex+n or not np.array_equal(np.searchsorted(used,indices),faces):
        raise ValueError('Registered tissue does not have source topology')
    vertices = np.frombuffer(data,'<f4',14*nvertex,vertex_offset).reshape(-1,14)[used]
    source_x = vertices[:,:3]
    slots = vertices[:,6:10].copy().view('<u4')
    rows,lanes = np.nonzero(vertices[:,10:14]>0)
    weights = np.zeros((len(used),count))
    np.add.at(weights,(rows,slots[rows,lanes]),vertices[rows,10+lanes])
    if np.abs(weights.sum(1)-1).max()>1e-5:
        raise ValueError('Unnormalized sparse influence')
    rotations, shifts, scales = [], [], []
    for k in range(count):
        body,*local = struct.unpack_from('<I8f',data,64+32*nrec+36*(first_binding+k))
        pose = poses[body]
        rotation = Rotation.from_quat(pose['quaternion_xyzw'])
        rotations.append(rotation * Rotation.from_quat(local[3:7]))
        shifts.append(rotation.apply(local[:3])+pose['position_m'])
        scales.append(local[7])
    predicted = transfer(source_x.astype(float),weights,rotations,shifts,scales)
    error = float(np.linalg.norm(predicted-world,axis=1).max())
    if error>2e-6:
        raise ValueError(f'Replayed tissue registration differs by {error} metres')
    def self_pairs(points):
        welded,remap = np.unique(points,axis=0,return_inverse=True)
        records = api['clearance']._exact_surface_records(welded,remap[faces])
        idx = api['intersections']._prepare_surface_aabb(records)
        return api['intersections']._audit_pair_prepared_first(idx,records,same_surface=True)['count']
    row = next(x for x in manifest['source']['surfaces'] if x['stable_id']==stable)
    return dict(name=row.get('label',row.get('member_id')),source_sha256=digest,source_member_sha256=row['member_sha256'],
        source_self_pairs=self_pairs(source_x),native_self_pairs=self_pairs(world),max_reconstruction_error_m=error)

def classify(case, reader, initial=False, muscle_ids=()):
    rec = json.loads((case/'terminal.receipt.json').read_text())
    folder = case/('initial-envelope' if initial else 'envelope')
    summary = json.loads((folder/'physical-banks.json').read_text())
    identity = rec['base_pack_file_sha256' if initial else 'pack_file_sha256']
    raw = subprocess.check_output(['zstd','-qdc',case/('initial.mrvpack.zst' if initial else 'accepted.mrvpack.zst')])
    if hashlib.sha256(raw).hexdigest() != identity or identity != summary['pack_sha256']:
        raise ValueError('Geometry differs from the audited native state')
    poses = rec['initial_anatomical_registration']['body_poses'] if initial else rec['accepted_body_poses']
    poses = {r['body_index']:r for r in poses}
    sys.path.insert(0,str(reader.parent/'src'))
    api = runpy.run_path(str(reader))
    exact = api['intersections']
    witness = {w['primitive']:w for w in map(json.loads,(folder/'target-witnesses.jsonl').read_text().splitlines())}
    ((_, _, skin, faces),) = api['primitives'](raw,poses,skin_only=True)

    def corners(points, triangles):
        @lru_cache(maxsize=None)
        def get(index):
            return [exact.float32_point_lattice_key(p) for p in points[triangles[index]]]
        return get

    def straddles(points, tri):
        a,b,c=tri
        n=exact._cross(exact._sub(b,a),exact._sub(c,a))
        signed=[exact._dot(n,exact._sub(p,a)) for p in points]
        return min(signed)<0<max(signed)

    get_skin = corners(skin,faces)
    result = []
    for key,primitive,points,triangles in api['primitives'](raw,poses):
        w=witness.pop(primitive,None)
        if w is None:
            continue
        if key!=(w['semantic'],w['stable_id']) or w['count']!=len(w['triangle_pairs']):
            raise ValueError('Mismatched triangle witnesses')
        if key[0]==51010:
            raise ValueError('Separate ocular/display witnesses before promotion')
        get_target = corners(points,triangles)
        proper = 0
        for i,j in w['triangle_pairs']:
            a,b=get_skin(i),get_target(j)
            proper += straddles(a,b) and straddles(b,a)
        item=dict(semantic=key[0],stable_id=key[1],primitive=primitive,pairs=w['count'],transverse=proper)
        if key[0]==51005 and key[1] in muscle_ids:
            item['registered_binding']=check_binding(case,key[1],points,triangles,poses,api)
        result.append(item)
    total=sum(r['pairs'] for r in result)
    if witness or total!=summary['physical_nonocular_pairs']:
        raise ValueError('Incomplete or unexpected set of physical witnesses')
    return dict(frame_sha256=identity,pose_count=len(poses),physical_pairs=total,
                transverse=sum(r['transverse'] for r in result),banks=result)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('reader', type=Path)
    parser.add_argument('--initial', action='store_true')
    parser.add_argument('--muscle-ids', type=int, nargs='+', default=[])
    args = parser.parse_args()
    print(json.dumps(classify(args.case.resolve(),args.reader.resolve(),args.initial,args.muscle_ids),indent=2))
