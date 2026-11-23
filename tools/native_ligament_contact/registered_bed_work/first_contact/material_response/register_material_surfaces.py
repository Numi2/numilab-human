"""Rebind a repaired TetGen skin/fat boundary to its original registered donor frames."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import numpy as np
from bind_registered_shoulder import sha


PREFIX = 'NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'


def pinned(env, name, hash_key):
    path = Path(env[PREFIX + name])
    if sha(path) != env[PREFIX + hash_key]:
        raise ValueError(f'Stale registration source {name}')
    return path


def tetgen_positions(path):
    with path.open() as f:
        count, dimension, attrs, markers = map(int, f.readline().split())
        rows = np.loadtxt(f)
    if (dimension, attrs, markers) != (3, 0, 0) or rows.shape != (count, 4):
        raise ValueError('Invalid TetGen node domain')
    if not np.array_equal(rows[:,0], np.arange(1, count + 1)) or not np.isfinite(rows).all():
        raise ValueError('Invalid TetGen node identity or coordinate')
    return rows[:,1:]


def area_normals(x, faces):
    tri = x[faces]
    area = np.cross(tri[:,1] - tri[:,0], tri[:,2] - tri[:,0])
    norm = np.zeros_like(x)
    for column in faces.T:
        np.add.at(norm, column, area)
    return norm


def vector(row, prefix, suffix=''):
    return np.array([float(row[prefix+a+suffix]) for a in 'xyz'])


def write_vector(row, prefix, values, suffix=''):
    for axis, value in zip('xyz', values):
        row[prefix+axis+suffix] = format(value, '.17g')


def write_layer(old, new, joint, faces, source, dest, frames):
    delta = new - old
    moved = np.any(delta != 0, axis=1)
    affected = np.zeros(len(old), bool)
    affected[faces[np.any(moved[faces], axis=1)]] = True
    prior_normals, normals = (area_normals(points, faces) for points in (old, new))
    with source.open() as f, dest.open('x') as out:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(out, reader.fieldnames, lineterminator='\n')
        writer.writeheader()
        count = 0
        for i, r in enumerate(reader):
            if int(r['vertex']) != i or not np.array_equal(vector(r,'w'), old[i]):
                raise ValueError(f'Inconsistent source vertex {i}')
            if moved[i]:
                position, R = frames[int(r['donor'])]
                previous_local = vector(r,'l')
                if np.linalg.norm(position + R @ previous_local - old[i]) > 1e-6:
                    raise ValueError(f'Source donor frame mismatch at {i}')
                write_vector(r,'l',np.linalg.solve(R,new[i]-position))
                write_vector(r,'w',new[i])
                source_face = int(r.get('source_face_row',r.get('d37_parent_surface_face_row',-1)))
                if source_face >= 0:
                    write_vector(r,'source_residual_',vector(r,'source_residual_','_m')+delta[i],'_m')
            if affected[i]:
                length = np.linalg.norm(normals[i])
                direction = np.dot(prior_normals[i],vector(r,'n'))
                if length < 1e-20 or direction == 0:
                    raise ValueError(f'Unresolved area normal at {i}')
                write_vector(r,'n',np.sign(direction)*normals[i]/length)
            writer.writerow(r)
            count += 1
        if count != len(joint):
            raise ValueError('Incorrect registered row count')
    return dict(rows=count, changed_world_coordinates=int(moved.sum()), recomputed_normals=int(affected.sum()),
                max_displacement_m=float(np.linalg.norm(delta,axis=1).max()))


def validate_layer(old_source, emitted, points, frames):
    allowed = {'w'+a for a in 'xyz'} | {'l'+a for a in 'xyz'} | {'n'+a for a in 'xyz'} | {'source_residual_'+a+'_m' for a in 'xyz'}
    maximum = 0.
    with old_source.open() as a, emitted.open() as b:
        prior, now = csv.DictReader(a), csv.DictReader(b)
        if prior.fieldnames != now.fieldnames:
            raise ValueError('Changed native surface row layout')
        fixed = set(prior.fieldnames) - allowed
        for i, (previous, row) in enumerate(zip(prior, now, strict=True)):
            if any(previous[k] != row[k] for k in fixed) or int(row['vertex']) != i:
                raise ValueError('Source identity, topology or binding changed')
            position, R = frames[int(row['donor'])]
            if not np.array_equal(vector(row,'w'),points[i]):
                raise ValueError('Emitted world-space position differs')
            error = np.linalg.norm(position + R @ vector(row,'l') - points[i])
            if not np.isfinite(error) or error > 1e-6:
                raise ValueError(f'Emitted donor-local coordinate differs at row {i}: {error}')
            maximum = max(maximum, error)
    return float(maximum)


def register(launch, material, destination):
    env = json.loads(launch.read_bytes())['environment']
    original = pinned(env,'TETGEN_NODE','TETGEN_NODE_SHA256')
    node_map = pinned(env,'COMBINED_NODE_MAP_CSV','COMBINED_NODE_MAP_SHA256')
    donor_frames = pinned(env,'DONOR_FRAMES_CSV','DONOR_FRAMES_SHA256')
    receipt = json.loads((material/'native-import.json').read_bytes())
    node = material/'mesh.node'
    if not receipt['native_imported'] or sha(node) != receipt['node_sha256']:
        raise ValueError('Missing certified physical reference')
    old, new = tetgen_positions(original), tetgen_positions(node)
    frames = {}
    with donor_frames.open() as f:
        for r in csv.DictReader(f):
            R = np.asarray([float(r['current_R'+str(i)+str(j)]) for i in range(3) for j in range(3)]).reshape(3,3)
            if np.max(np.abs(R.T@R-np.eye(3))) > 1e-5 or np.linalg.det(R) <= 0:
                raise ValueError('Invalid rigid donor transform')
            frames[int(r['donor'])] = (vector(r,'current_'),R)
    regions = {k:{} for k in 'OID'}
    with node_map.open() as f:
        for r in csv.DictReader(f):
            k = r['registered_surface_layer']
            if k in regions:
                row, joint = int(r['registered_surface_vertex_row']),int(r['joint_node_id'])-1
                # Bonded skin/fat interface rows occur in both material partitions.
                if regions[k].get(row, joint) != joint or not 0 <= joint < min(len(old),len(new)):
                    raise ValueError('Ambiguous or missing registered surface node')
                regions[k][row] = joint
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        stage = Path(tmp)/'surfaces'
        stage.mkdir()
        result = dict(physical_nodes=sha(node), launch=sha(launch), donor_frames=sha(donor_frames),
                      original_nodes=sha(original), original_combined_map=sha(node_map), layers={})
        for k, mapping in regions.items():
            name = 'SURFACE_'+k
            table = pinned(env,name+'_NODES_CSV',name+'_NODES_SHA256')
            face_file = pinned(env,name+'_FACES_CSV',name+'_FACES_SHA256')
            raw = np.loadtxt(face_file, delimiter=',', skiprows=1, usecols=(0,1,2,3),dtype=np.int32)
            joint = np.asarray([mapping[i] for i in range(len(mapping))])
            if (not np.array_equal(raw[:,0],np.arange(len(raw))) or raw[:,1:].min()<0 or
                raw[:,1:].max() >= len(joint) or len(np.unique(joint)) != len(joint)):
                raise ValueError('Invalid registered face or layer domain')
            target = stage/table.name
            row = write_layer(old[joint],new[joint],joint,raw[:,1:],table,target,frames)
            row['maximum_donor_round_trip_m'] = validate_layer(table,target,new[joint],frames)
            row.update(source=sha(table), emitted=sha(target), faces=sha(face_file))
            result['layers'][k] = row
        (stage/'registration.json').write_text(json.dumps(result,indent=2)+'\n')
        stage.rename(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('launch','material','output'):
        parser.add_argument(key,type=Path)
    p=parser.parse_args()
    register(p.launch, p.material, p.output)
