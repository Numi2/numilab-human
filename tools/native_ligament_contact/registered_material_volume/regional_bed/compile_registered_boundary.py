"""Compile the outer material faces and donor-frame kinematics into existing NHSKIN v5."""
import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from rebind_human_skin_reference import decode


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def compile_boundary(launch, template, output):
    case = json.loads(launch.read_bytes())
    env = case['environment']
    prefix = 'NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
    sources = {}

    def pinned(name, suffix='_CSV'):
        path = Path(env[prefix + name + suffix])
        actual = sha(path)
        if actual != env[prefix + name + '_SHA256']:
            raise ValueError(f'Unmatched material input {name}')
        sources[name] = actual
        return path

    def table(name, cols=None):
        return np.loadtxt(pinned(name), delimiter=',', skiprows=1, usecols=cols, ndmin=2)

    def ordered(rows, name):
        if not np.array_equal(rows[:, 0], np.arange(len(rows))):
            raise ValueError(f'Unordered or missing {name} identities')
        return rows[:, 1:]

    original = decode(template.read_bytes())
    if len(original['binds']) != 86 or sha(template) != case['pins'][str(template)]:
        raise ValueError('Unmatched NHSKIN donor registry')
    rows = table('SURFACE_O_NODES', range(8))
    geometry = ordered(rows, 'outer node')[:, 1:]
    xyz = geometry[:, :3].astype('<f4')
    normal = geometry[:, 3:]
    weights = ordered(table('ROLE_O_FULL86'), 'weight')
    sections = ordered(table('ROLE_O_CONTACT_THICKNESS'), 'column')
    triangles = ordered(table('SURFACE_O_FACES', range(4)), 'face')
    if np.any(triangles != triangles.astype(np.int64)) or np.any(triangles < 0) or np.any(triangles >= len(xyz)):
        raise ValueError('Outer face index outside material domain')
    triangles = triangles.astype('<u4')
    if weights.shape != (len(xyz),86) or sections.shape != (len(xyz),2):
        raise ValueError('Role arrays disagree on the material domain')
    if not np.isfinite(sections).all() or np.any(sections <= 0) or not np.isfinite(geometry).all():
        raise ValueError('Invalid material column or surface')
    frames = ordered(table('DONOR_FRAMES'), 'binding')
    if not np.array_equal(frames[:, 0], original['binds']['body']):
        raise ValueError('Different body ownership in skin and material')
    frame_path = launch.parent/'native-initial-bodies.json'
    actual = json.loads(frame_path.read_bytes())
    body_rows = actual['bodies']
    if not np.array_equal([r['body_index'] for r in body_rows],frames[:,0]):
        raise ValueError('Loaded body order differs from the source registry')
    def loaded_frames(which):
        position = np.array([r[which]['position_world_m'] for r in body_rows])
        rotation = Rotation.from_quat([r[which]['orientation_world_xyzw'] for r in body_rows]).as_matrix()
        return position,rotation
    translation, loaded_rotations = loaded_frames('current')
    _, rest = loaded_frames('rest')
    evidence = launch.parent/'physical-reference.node'
    if not evidence.is_file():
        raise ValueError('Paired initial material coordinates required')
    rotations = frames[:, 4:13].reshape(-1,3,3)
    if np.max(abs(rotations.transpose(0,2,1)@rotations-np.eye(3))) > 3e-6 or np.any(np.linalg.det(rotations)<0):
        raise ValueError('Donor frames must be proper rotations')

    # Match the wall to its actual volume nodes and oriented faces, not a nearest surface.
    volume = np.full(len(xyz), -1, dtype=np.int64)
    with pinned('COMBINED_NODE_MAP').open() as source:
        for r in csv.DictReader(source):
            if r['registered_surface_layer'] != 'O':
                continue
            i = int(r['registered_surface_vertex_row'])
            if not 0 <= i < len(volume) or volume[i] >= 0:
                raise ValueError('Duplicate or invalid material row')
            volume[i] = int(r['joint_node_id']) - 1
    mesh = pinned('TETGEN_NODE', '').parent
    if np.any(volume < 0) or len(np.unique(volume)) != len(volume):
        raise ValueError('Incomplete outer material ancestry')
    wall_path = mesh/'material-walls.npz'
    with np.load(wall_path) as walls:
        wall = walls['O']
    def oriented(faces):
        order = np.argmin(faces,axis=1)
        a = np.take_along_axis(faces,(order[:,None]+np.arange(3))%3,axis=1)
        return a[np.lexsort((a[:,2],a[:,1],a[:,0]))]
    if not np.array_equal(oriented(volume[triangles]), oriented(wall)):
        raise ValueError('Registered topology differs from the physical wall')
    current = mesh/'current.node'
    loaded = np.loadtxt(current,skiprows=1,usecols=(0,1,2,3))
    if not np.array_equal(loaded[:,0],np.arange(1,len(loaded)+1)) or volume.max() >= len(loaded):
        raise ValueError('Invalid material node IDs')
    if sha(evidence)!=sha(current):
        runtime_xyz=np.loadtxt(evidence,skiprows=1,usecols=(1,2,3))
        if not np.array_equal(loaded[:,1:],runtime_xyz):
            raise ValueError('Physical opening differs from the referenced loaded state')
    gap = np.linalg.norm(loaded[volume,1:] - xyz,axis=1)
    # Typed O provides ancestry, faces and weights. Geometry comes from the loaded
    # material; its supported state can differ from the authoring surface.
    xyz = loaded[volume,1:].astype('<f4')
    edge_a = xyz[triangles[:,1]].astype(float)-xyz[triangles[:,0]]
    edge_b = xyz[triangles[:,2]].astype(float)-xyz[triangles[:,0]]
    face_normals = np.cross(edge_a,edge_b)
    normal = np.zeros((len(xyz),3))
    for corner in triangles.T:
        np.add.at(normal,corner,face_normals)
    if np.any(np.linalg.norm(face_normals,axis=1)==0):
        raise ValueError('Degenerate physical boundary triangle')

    # Each donor receives its own inverse reference transform. The native weighted
    # position and its force transpose therefore share the same per-body lever arms.
    bind = original['binds'].copy()
    source_rotation_gap = (Rotation.from_matrix(rotations).inv()*Rotation.from_matrix(loaded_rotations)).magnitude()
    rotations = loaded_rotations
    inv = Rotation.from_matrix(rotations).inv()
    bind['v'][:, :3] = inv.apply(-translation)
    bind['v'][:,3:7] = inv.as_quat()
    bind['v'][:,7] = 1
    vertices = np.zeros((len(xyz),14), dtype='<f4')
    vertices[:, :3] = xyz
    norm = np.linalg.norm(normal,axis=1)
    if np.min(norm) <= np.finfo(float).tiny:
        raise ValueError(f'Undefined normal at physical outer rows {np.flatnonzero(norm==0)[:16]}')
    # ABI 5 normals are in skeleton rest-world axes. Map current normals back
    # through the weighted bind-to-current rotation before encoding them.
    normal_map = rotations @ rest.transpose(0,2,1)
    averaged = (weights @ normal_map.reshape(86,9)).reshape(-1,3,3)
    encoded = np.linalg.solve(averaged, (normal/norm[:,None])[:,:,None])[:,:,0]
    nlength = np.linalg.norm(encoded,axis=1)
    if not np.isfinite(nlength).all() or np.any(nlength<=0):
        raise ValueError('Unrepresentable normal in the native rest frame')
    vertices[:,3:6] = encoded / nlength[:,None]
    top = np.argsort(-weights,axis=1,kind='stable')[:, :4]
    vertices[:,6:10].view('<u4')[:] = top
    subset = np.take_along_axis(weights,top,axis=1)
    vertices[:,10:] = subset / subset.sum(axis=1)[:,None]
    header = struct.pack('<8s5I',b'NHSKIN1\0',5,86,len(xyz),triangles.size,original['fingerprint'])
    data = b''.join((header,original['source_sha256'],bind.tobytes(),vertices.tobytes(),
                     triangles.tobytes(),weights.astype('<f4').tobytes()))
    check = decode(data)
    if not np.array_equal(check['weights'],weights.astype('f')) or not np.array_equal(check['inds'],triangles):
        raise ValueError('Weight or face identity lost in emission')
    # Perform the emitted forward calculation in native storage precision.
    posed = np.zeros_like(xyz)
    for i, row in enumerate(bind['v']):
        local = xyz @ Rotation.from_quat(row[3:7]).as_matrix().astype('f').T + row[:3]
        world = local @ rotations[i].astype('f').T + translation[i].astype('f')
        posed += check['weights'][:,i,None] * world
    err = np.linalg.norm(posed-xyz,axis=1)
    if err.max() > 3e-6:
        raise ValueError(f'Float32 donor roundtrip exceeds 3 micrometres: {err.max()}')
    output.mkdir(parents=True,exist_ok=True)
    dst = output/'registered-boundary.nhskin'
    dst.write_bytes(data)
    report = dict(initial_bodies_path=str(frame_path.resolve()),contact_columns_path=env[prefix+'ROLE_O_CONTACT_THICKNESS_CSV'],source_launch=sha(launch),initial_frame_sha256=sha(frame_path),runtime_opening_sha256=sha(evidence),authoring_to_opening_frame_max_deg=float(np.degrees(source_rotation_gap).max()),material_inputs=sources,source_skin_sha256=sha(template),
                  nhskin_sha256=sha(dst),material_wall_sha256=sha(wall_path),loaded_nodes_sha256=sha(current),
                  outer_nodes=len(xyz),outer_faces=len(triangles),body_lanes=len(bind),
                  authoring_to_loaded_max_m=float(gap.max()),authoring_to_loaded_rms_m=float(np.sqrt(np.mean(gap**2))),loaded_material_error_max_m=float(np.linalg.norm(xyz-loaded[volume,1:],axis=1).max()),float32_roundtrip_max_m=float(err.max()),
                  float32_roundtrip_rms_m=float(np.sqrt(np.mean(err**2))))
    (output/'material-boundary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('launch','template','output'):
        p.add_argument(key,type=Path)
    compile_boundary(**vars(p.parse_args()))
