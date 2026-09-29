"""Complete declared organ source families without relabeling spaces as tissue.

All geometry is source-authored. Shared family incidences and overlapping
aggregate/descendant representations are retained, never summed as tissue.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

from . import lung_envelope as lung
from . import model as human
from .physiology import load_anatomy
from .torso_anatomy_coverage import source_family_topology, source_organ_coverage

ROOT = human.REPOSITORY_ROOT
CONFIG = ROOT / 'config/source-organ-family-composite.v1.json'
PAYLOAD_NAME = 'source-organ-family-anatomy.nhanatomy'
MANIFEST_NAME = 'source-organ-family-anatomy.manifest.json'


def require(condition, message):
    if not condition:
        raise human.ImportError('organ family geometry: ' + message)


def configuration():
    return human.read_json(CONFIG)


def source_selection(sources: Path, config=None):
    """Resolve every family independently from both pinned hierarchy tables."""
    config = configuration() if config is None else config
    require(human.sha256(ROOT / 'sources.lock.json') == config['source_lock_sha256'], 'source lock identity')
    lock = human.read_json(ROOT / 'sources.lock.json')['sources']['bodyparts3d_4']['files']
    for filename in ['isa_BP3D_4.0_obj_99.zip', 'partof_BP3D_4.0_obj_99.zip']:
        path = sources / filename
        require(path.is_file() and path.stat().st_size == lock[filename]['bytes']
                and human.sha256(path) == lock[filename]['sha256'], 'source archive identity: ' + filename)
    graph_path = ROOT / config['template']['file']
    map_path = ROOT / config['baseline_map']['file']
    require(human.sha256(graph_path) == config['template']['sha256'], 'family template identity')
    require(human.sha256(map_path) == config['baseline_map']['sha256'], 'baseline map identity')
    graph = human.read_json(graph_path)
    baseline = human.read_json(map_path)['entries']
    require(len(baseline) == config['base_bodyparts_surface_count'], 'baseline surface count')
    anatomy = load_anatomy(sources)
    regions = graph['regions']
    require(len(regions) == len(config['family_bindings']) and
            {r['id'] for r in regions} == set(config['family_bindings']), 'complete family identities')
    by_member = {}
    family_rows = []
    for region in regions:
        binding = config['family_bindings'][region['id']]
        require(all(region[k] == binding[k] for k in ['concept_id', 'source_name', 'hierarchy']), 'family source identity')
        expected = anatomy['tables'][region['hierarchy']].get((region['concept_id'], region['source_name']))
        require(expected and region['selection'] == 'complete_membership' and
                len(set(region['member_ids'])) == len(region['member_ids']) and
                set(region['member_ids']) == expected, 'complete source family membership')
        family_rows.append({**binding, 'id': region['id'], 'expected_members': sorted(expected)})
        for member in sorted(expected):
            owner = config['member_owner_overrides'].get(member, binding['myosim_body'])
            row = by_member.setdefault(member, {'member_id': member, 'myosim_body': owner,
                                                 'families': [], 'source_regions': []})
            require(row['myosim_body'] == owner, 'shared family member has conflicting body owners')
            row['families'].append(region['id'])
            row['source_regions'].append({k: region[k] for k in ['concept_id', 'source_name', 'hierarchy']})
    require(set(config['member_owner_overrides']) <= set(by_member), 'unused member owner override')
    by_type = {}
    for (concept, label), members in anatomy['tables']['is_a'].items():
        for member in members:
            by_type.setdefault(member, []).append({'concept_id': concept, 'label': label})
    baseline_by_member = {e['member_id']: e for e in baseline}
    require(len(baseline_by_member) == len(baseline), 'unique baseline member identity')
    added = []
    for member, row in sorted(by_member.items()):
        types = sorted(by_type[member], key=lambda t: (t['concept_id'], t['label']))
        if member in baseline_by_member:
            require(baseline_by_member[member]['myosim_body'] == row['myosim_body'], 'existing source family body drift')
            continue
        matches = []
        for name, rule in config['layer_registry'].items():
            if name == 'organ_component':
                continue
            if rule.get('required_family') and rule['required_family'] not in row['families']:
                continue
            if rule.get('required_families') and not set(rule['required_families']) & set(row['families']):
                continue
            if any({'concept_id': c, 'label': n} in types for c, n in rule.get('excluded_source_types', [])):
                continue
            if any({'concept_id': c, 'label': n} in types for c, n in rule['source_types']):
                matches.append(name)
        require(len(matches) <= 1, 'ambiguous source tissue/vessel/duct/space type')
        layer = matches[0] if matches else 'organ_component'
        primary = row['source_regions'][0]
        if layer == 'organ_component':
            typed = source_organ_coverage(primary, types)
        else:
            typed = {'source_structure_kind': layer, 'organ_coverage': 'source_' + layer + '_representation'}
        _, archive_member, obj = human._bodyparts_obj_member(sources, primary['hierarchy'], member)
        added.append({**row, **primary, **typed, 'layer': layer,
                      **{k: config['layer_registry'][layer][k] for k in ['layer_code', 'semantic']},
                      'source_is_a_types': types, 'source_member': archive_member,
                      'source_member_sha256': hashlib.sha256(obj).hexdigest(),
                      'source_topology': source_family_topology(obj, archive_member)})
    for i, row in enumerate(added, config['base_surface_count'] + 1):
        row['stable_id'] = i
    require(config['base_surface_count'] + len(added) <= 1024, 'native surface capacity')
    return {'families': family_rows, 'added': added,
            'required_member_count': len(by_member),
            'shared_family_members': {m: r['families'] for m, r in sorted(by_member.items()) if len(r['families']) > 1},
            'baseline_source_members': baseline_by_member, 'anatomy_source': anatomy['source']}


def context(sources, artifact, registration_path, base_payload, config=None):
    config = configuration() if config is None else config
    require(human.sha256(base_payload) == config['base_payload_sha256'], 'base composite payload identity')
    raw, header, records, vertices, indices = lung.decode(base_payload)
    require(header[:2] == (config.get('base_payload_abi', 3), config['base_surface_count']), 'base composite ABI/count')
    model, rest, _, _, reg, checks = lung.oracle_model(sources, artifact, registration_path)
    require(human.sha256(registration_path) == config['registration_sha256'], 'source registration identity')
    _, owners = human._bodyparts_runtime_bindings(reg, artifact)
    return raw, header, records, vertices, indices, model, rest, reg, owners, checks


def oracle_triangles(obj, member):
    """Parse source faces independently of the compiler's OBJ lowering."""
    import numpy as np
    vertices, triangles = [], []
    for line in obj.decode('utf-8').splitlines():
        fields = line.split()
        if not fields or fields[0].startswith('#'):
            continue
        if fields[0] == 'v':
            require(len(fields) == 4, 'oracle source vertex arity: ' + member)
            vertices.append([float(x) for x in fields[1:]])
        elif fields[0] == 'f':
            require(len(fields) >= 4, 'oracle source face arity: ' + member)
            face = [int(x.split('/')[0]) for x in fields[1:]]
            require(all(x != 0 for x in face), 'oracle zero source index')
            face = [x-1 if x > 0 else len(vertices)+x for x in face]
            triangles.extend([[face[0], face[i], face[i+1]] for i in range(1, len(face)-1)])
    v, f = np.asarray(vertices, float), np.asarray(triangles, int)
    require(v.ndim == 2 and v.shape[1] == 3 and len(v) > 0 and np.isfinite(v).all()
            and f.ndim == 2 and f.shape[1] == 3 and len(f) > 0 and f.min() >= 0 and f.max() < len(v),
            'oracle source geometry ranges: ' + member)
    return v, f


def source_surface(sources, spec, model, rest, matrix, owners):
    """Independent source-face, normal and NumPy/MuJoCo coordinate oracle."""
    import mujoco
    import numpy as np
    _, member, obj = human._bodyparts_obj_member(sources, spec['hierarchy'], spec['member_id'])
    require(hashlib.sha256(obj).hexdigest() == spec['source_member_sha256'] and member == spec['source_member'],
            'source OBJ identity')
    v, f = oracle_triangles(obj, member)
    owner, source_body = owners[spec['myosim_body']]
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec['myosim_body'])
    require(sid == source_body['source_body_id'], 'named source body identity')
    world = v @ matrix[:3, :3].T + matrix[:3, 3]
    local = (world - rest.xipos[sid]) @ rest.ximat[sid].reshape(3, 3)
    normals = np.zeros_like(v)
    face_normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for corner in range(3):
        np.add.at(normals, f[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    normals[lengths <= 1e-12] = v[lengths <= 1e-12] - v.mean(axis=0)
    require(bool((np.linalg.norm(normals, axis=1) > 1e-12).all()), 'source normal degeneracy')
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normals = normals @ matrix[:3, :3].T
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normals = normals @ rest.ximat[sid].reshape(3, 3)
    return owner, sid, local, normals, f


def compiled_surface(sources, spec, registration, bodies):
    """Use the existing Human lowering and consumed source/body catalog."""
    import numpy as np
    _, member, obj = human._bodyparts_obj_member(sources, spec['hierarchy'], spec['member_id'])
    require(hashlib.sha256(obj).hexdigest() == spec['source_member_sha256'], 'compiler source member identity')
    vertices, triangles = human._bodyparts_obj_triangles(obj, member)
    normals = human._bodyparts_vertex_normals(vertices, triangles, member)
    translation, quaternion, scale = human._bodyparts_visual_local_pose(
        registration['coordinate_system']['global_source_mm_to_myosim_world_m'], 'organ family global frame')
    global_rotation = human._myosim_matrix_from_quaternion_xyzw(quaternion)
    body = bodies[spec['myosim_body']]
    position = body['default_com_position_world_m']
    orientation = body['default_inertial_quaternion_world_xyzw']
    world = human._bodyparts_source_mm_to_body_world(
        vertices, [0.,0.,0.], [0.,0.,0.,1.], translation, quaternion, scale)
    local = human._bodyparts_world_to_body_stored_m(
        world, position, orientation, [0.,0.,0.], [0.,0.,0.,1.], 1., 'organ family ' + spec['member_id'])
    inverse = human._matrix_transpose(human._myosim_matrix_from_quaternion_xyzw(orientation))
    stored_normals = []
    for normal in normals:
        n = human._bodyparts_unit_vector(human._myosim_matrix_vector(global_rotation, normal), 'organ family normal')
        stored_normals.append(human._bodyparts_unit_vector(human._myosim_matrix_vector(inverse, n), 'organ family local normal'))
    return body['core_body_index'], body['source_body_id'], np.asarray(local), np.asarray(stored_normals), np.asarray(triangles)


def compose(sources, artifact, registration_path, base_payload, output, *, config=None):
    import numpy as np
    config = configuration() if config is None else config
    selection = source_selection(sources, config)
    _, header, records, vertices, indices, model, rest, reg, owners, checks = context(
        sources, artifact, registration_path, base_payload, config)
    bodies, _, _ = human._myosim_surface_route_context(artifact, reg['source']['myosim']['source']['archive_sha256'])
    packed_records = [records.tobytes()]
    packed_vertices = [vertices.tobytes()]
    packed_indices = [indices.tobytes()]
    nv, ni = len(vertices), len(indices)
    surfaces = []
    for spec in selection['added']:
        owner, sid, local, normals, f = compiled_surface(sources, spec, reg, bodies)
        packed_records.append(struct.pack('<8I', owner, nv, len(local), ni, f.size,
                                          spec['stable_id'], spec['layer_code'], 0))
        packed_vertices.append(np.column_stack([local, normals]).astype('<f4').tobytes())
        packed_indices.append((f.ravel() + nv).astype('<u4').tobytes())
        surfaces.append({**spec, 'source_body_id': int(sid), 'core_body_index': owner,
                         'vertex_count': len(local), 'triangle_count': len(f)})
        nv += len(local)
        ni += f.size
    count = header[1] + len(surfaces)
    require(nv <= 1000000 and ni <= 6000000, 'native geometry capacity')
    abi = config.get('payload_abi', 4)
    raw = lung.HEADER.pack(b'NHANAT1\0', abi, count, nv, ni, header[4], header[5])
    raw += b''.join(packed_records) + b''.join(packed_vertices) + b''.join(packed_indices)
    output.mkdir(parents=True, exist_ok=True)
    path = output / PAYLOAD_NAME
    path.write_bytes(raw)
    manifest = {'schema': 'numi.human.source-organ-family-composite-payload.v1',
                'payload': {'sha256': human.sha256(path), 'abi': abi, 'surfaces': count, 'vertices': nv, 'indices': ni},
                'source_configuration_sha256': human.sha256(config_path(config)), 'base_payload_sha256': human.sha256(base_payload),
                'registration_sha256': human.sha256(registration_path), 'selection': selection,
                'surfaces': surfaces, 'rigid_source_program_checks': checks, 'geometry_repair_applied': False,
                'license': 'CC-BY-4.0 AND CC-BY-SA-4.0 AND Apache-2.0',
                'clinical_registration': False, 'mechanics_admitted': False, 'physical_volume_admitted': False,
                'boundary': config['boundary']}
    human.write_json(output / MANIFEST_NAME, manifest)
    return manifest


def config_path(config):
    return ROOT / config.get('configuration_file', str(CONFIG.relative_to(ROOT)))


def audit(sources, artifact, registration_path, baseline_payload, base_payload,
          payload, native_pack, native_poses, pose, mask=1023, *, config=None,
          native_surface_count=None, prefix_base_payload=None, hidden_source_surface_ids=()):
    import mujoco
    import numpy as np
    from .torso_anatomy_audit import _pack_sections, anatomy_hidden_ids
    from .upper_limb_pose_audit import _pose_qpos
    config = configuration() if config is None else config
    selection = source_selection(sources, config)
    _, old_header, old_records, old_v, old_i, model, rest, reg, owners, checks = context(
        sources, artifact, registration_path, base_payload, config)
    _, header, records, vertices, indices = lung.decode(payload)
    count = config['base_surface_count'] + len(selection['added'])
    native_count = count if native_surface_count is None else native_surface_count
    require(count <= native_count <= 1024, 'native complete surface capacity')
    abi = config.get('payload_abi', 4)
    require(header[:2] == (abi, count) and header[4:] == old_header[4:], 'complete composite header identity')
    require(records[:old_header[1]].tobytes() == old_records.tobytes()
            and vertices[:len(old_v)].tobytes() == old_v.tobytes()
            and indices[:len(old_i)].tobytes() == old_i.tobytes(),
            f'unchanged complete {old_header[1]}-surface baseline')
    manifest = human.read_json(payload.with_name(MANIFEST_NAME))
    require(manifest['source_configuration_sha256'] == human.sha256(config_path(config))
            and manifest['base_payload_sha256'] == human.sha256(base_payload)
            and manifest['registration_sha256'] == human.sha256(registration_path), 'composite provenance')
    require(manifest['payload'] == {'sha256': human.sha256(payload), 'abi': abi, 'surfaces': count,
                                    'vertices': header[2], 'indices': header[3]}, 'composite payload identity')
    require(manifest['selection'] == selection, 'independent source selection/type/topology')
    require(manifest['geometry_repair_applied'] is False and manifest['clinical_registration'] is False
            and manifest['mechanics_admitted'] is False and manifest['physical_volume_admitted'] is False,
            'source-only qualification boundary')
    require({k:v for k,v in manifest['rigid_source_program_checks'].items() if k != 'file'}
            == {k:v for k,v in checks.items() if k != 'file'}, 'source program identity')
    if config.get('prefix_configuration'):
        prefix = config['prefix_configuration']
        path = ROOT / prefix['file']
        require(human.sha256(path) == prefix['sha256'] and prefix_base_payload is not None,
                'prefix configuration/source payload identity')
        base_audit = audit(sources, artifact, registration_path, baseline_payload, prefix_base_payload,
                           base_payload, native_pack, native_poses, pose, mask,
                           config=human.read_json(path), native_surface_count=native_count,
                           hidden_source_surface_ids=hidden_source_surface_ids)
    else:
        base_audit = lung.audit(sources, artifact, registration_path, baseline_payload, base_payload,
                               native_pack, native_poses, pose, mask, native_surface_count=native_count,
                               hidden_source_surface_ids=hidden_source_surface_ids)
    sections = _pack_sections(native_pack)
    pv = np.frombuffer(sections[2][0], '<f4').reshape(-1, 20)
    pi = np.frombuffer(sections[3][0], '<u4')
    primitives = np.frombuffer(sections[4][0], '<u4').reshape(-1, 16)
    instances = np.frombuffer(sections[5][0], '<u4').reshape(-1, 20)
    transforms = np.frombuffer(sections[5][0], '<f4').reshape(-1, 20)
    semantic_codes = [51010,51011,51012,51020,51021,51022,51023,51024,51025,51026,51027,51028,51029,51030,51031]
    all_anatomy = primitives[np.isin(primitives[:, 4], semantic_codes)]
    if native_count > count:
        all_anatomy = all_anatomy[all_anatomy[:,5] <= count]
    require(len(all_anatomy) == count and set(all_anatomy[:, 5]) == set(range(1, count+1)),
            'complete native anatomy identity coverage')
    by_id = {int(p[5]):p for p in all_anatomy}
    snapshot = human.read_json(native_poses)
    hidden = anatomy_hidden_ids(snapshot, hidden_source_surface_ids, native_count)
    require(snapshot['surface_count'] == native_count and snapshot['visible_layer_mask'] == mask, 'native composite pose/profile')
    poses = {b['body_index']: b for b in snapshot['bodies']}
    require(len(poses) == len(snapshot['bodies']) and
            snapshot['registration_fingerprint32'] == header[4] and
            set(map(int, records[:, 0])) <= set(poses), 'unique native pose owner identity')
    if native_count == count:
        require(set(poses) == set(map(int, records[:, 0])), 'exact native pose owner coverage')
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0 if pose is None else _pose_qpos(model, pose, mujoco, np)[0]
    mujoco.mj_forward(model, data)
    matrix = np.asarray(reg['coordinate_system']['global_source_mm_to_myosim_world_m'])
    next_v, next_i = len(old_v), len(old_i)
    rows = []
    require(len(manifest['surfaces']) == len(selection['added']), 'source surface receipt count')
    for spec, declared, record in zip(selection['added'], manifest['surfaces'], records[old_header[1]:], strict=True):
        owner, sid, local, normals, f = source_surface(sources, spec, model, rest, matrix, owners)
        nv, ni = len(local), f.size
        require(record.tolist() == [owner, next_v, nv, next_i, ni, spec['stable_id'], spec['layer_code'], 0],
                'source component record/owner/range')
        require(declared == {**spec, 'source_body_id': int(sid), 'core_body_index': owner,
                             'vertex_count': nv, 'triangle_count': len(f)}, 'source component receipt provenance')
        require(np.array_equal(indices[next_i:next_i+ni]-next_v, f.ravel()), 'payload exact source topology')
        p = by_id[spec['stable_id']]
        first_i, count_i, _, inst = map(int, p[:4])
        require(p[4] == spec['semantic'] and p[6] == owner and count_i == ni and first_i+ni <= len(pi),
                'native component semantic/owner/range')
        require(inst < len(instances), 'native instance range')
        first_v = int(pi[first_i:first_i+ni].min())
        require(first_v+nv <= len(pv) and np.array_equal(pi[first_i:first_i+ni]-first_v, f.ravel()),
                'native exact source topology')
        require(instances[inst, 9] == owner and instances[inst, 10] == 3
                and np.array_equal(instances[inst, 12:16], p[4:8])
                and instances[inst, 11] == (11 if mask & (1 << (spec['layer_code']-1)) and spec['stable_id'] not in hidden else 0),
                'native component binding/visibility')
        require(np.array_equal(transforms[inst, :8], [0,0,0,1,0,0,0,1]), 'native component local transform')
        packed_local, packed_normal = pv[first_v:first_v+nv, :3], pv[first_v:first_v+nv, 4:7]
        require(np.isfinite(packed_local).all() and np.isfinite(packed_normal).all(), 'finite native component geometry')
        q = np.asarray(poses[owner]['orientation_world_xyzw'])
        position = np.asarray(poses[owner]['position_world_m'])
        require(q.shape == (4,) and position.shape == (3,) and np.isfinite(q).all()
                and np.isfinite(position).all() and abs(np.linalg.norm(q)-1.) <= 1e-6,
                'native finite unit component pose')
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, q[[3,0,1,2]] / np.linalg.norm(q))
        rotation = rotation.reshape(3,3)
        expected_world = local @ data.ximat[sid].reshape(3,3).T + data.xipos[sid]
        errors = {'payload_local_error_m': float(np.linalg.norm(vertices[next_v:next_v+nv, :3]-local, axis=1).max()),
                  'native_pack_local_error_m': float(np.linalg.norm(packed_local-local, axis=1).max()),
                  'native_pose_world_error_m': float(np.linalg.norm(packed_local @ rotation.T+position-expected_world, axis=1).max()),
                  'normal_direction_error': float(max(np.linalg.norm(packed_normal-normals, axis=1).max(),
                                                      np.linalg.norm(vertices[next_v:next_v+nv, 3:]-normals, axis=1).max()))}
        unit_error = float(max(np.abs(np.linalg.norm(packed_normal, axis=1)-1).max(),
                               np.abs(np.linalg.norm(vertices[next_v:next_v+nv, 3:], axis=1)-1).max()))
        position_error = float(np.linalg.norm(position-data.xipos[sid]))
        rotation_error = float(np.linalg.norm(rotation-data.ximat[sid].reshape(3,3), axis=0).max())
        rows.append({'member_id': spec['member_id'], 'stable_id': spec['stable_id'], 'layer': spec['layer'],
                     'families': spec['families'], 'myosim_body': spec['myosim_body'], 'core_body_index': owner,
                     'vertex_count': nv, 'triangle_count': len(f), 'topology_exact': True,
                     **errors, 'normal_unit_error': unit_error, 'native_COM_position_error_m': position_error,
                     'native_orientation_unit_witness_error_m': rotation_error,
                     'passed': max(errors.values()) <= 2e-5 and unit_error <= .002
                               and max(position_error, rotation_error) <= 1e-6})
        next_v += nv
        next_i += ni
    require(next_v == len(vertices) and next_i == len(indices), 'complete composite geometry ownership')
    selected_members = set(selection['baseline_source_members']) | {r['member_id'] for r in rows}
    families = [{**r, 'rendered_members': sorted(set(r['expected_members']) & selected_members),
                 'passed': set(r['expected_members']) <= selected_members} for r in selection['families']]
    passed = base_audit['passed'] and all(r['passed'] for r in rows) and all(r['passed'] for r in families)
    return {'schema': 'numi.human.native-organ-family-source-audit.v1', 'passed': passed,
            'surface_count': count, 'native_surface_count': native_count,
            'family_count': len(families), 'families': families,
            'required_unique_source_members': selection['required_member_count'],
            'shared_family_members': selection['shared_family_members'], 'rows': rows, 'baseline_audit': base_audit,
            'maximum_added_native_source_error_m': max(r['native_pose_world_error_m'] for r in rows),
            'allowed_geometry_and_normal_error': 2e-5, 'allowed_COM_and_orientation_witness_error': 1e-6,
            'allowed_normal_unit_error': .002, 'visible_layer_mask': mask,
            'qualification': {'source_geometry': passed, 'complete_declared_family_membership': passed,
                              'disjoint_tissue': False, 'connected_lumen': False, 'physical_volume': False,
                              'clinical_registration': False, 'mechanics': False},
            'inputs': {k: {'path': str(p.resolve()), 'sha256': human.sha256(p)} for k,p in {
                'configuration': config_path(config), 'registration': registration_path, 'baseline_payload': baseline_payload,
                'base_payload': base_payload, 'payload': payload, 'payload_manifest': payload.with_name(MANIFEST_NAME),
                'native_pack': native_pack, 'native_poses': native_poses}.items()},
            'pose_coordinates': pose, 'boundary': config['boundary']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['compose', 'audit'])
    parser.add_argument('--configuration', type=Path, default=CONFIG)
    for name in ['sources', 'artifact', 'registration', 'base-payload', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['baseline-payload', 'payload', 'native-pack', 'native-poses']:
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--prefix-base-payload', type=Path)
    parser.add_argument('--mask', type=int, default=1023)
    parser.add_argument('--raw-source-rest', action='store_true')
    parser.add_argument('--pose-q', nargs=2, action='append', default=[])
    args = parser.parse_args()
    config = human.read_json(args.configuration)
    require(args.configuration.resolve() == config_path(config).resolve(), 'configuration path identity')
    if args.raw_source_rest and args.pose_q:
        parser.error('raw source rest cannot have pose coordinates')
    if args.mode == 'compose':
        result = compose(args.sources, args.artifact, args.registration, args.base_payload, args.output, config=config)
    else:
        if not all([args.baseline_payload,args.payload,args.native_pack,args.native_poses]):
            parser.error('audit requires baseline-payload, payload, native-pack and native-poses')
        pose = None if args.raw_source_rest else tuple((int(i),float(v)) for i,v in args.pose_q)
        result = audit(args.sources,args.artifact,args.registration,args.baseline_payload,args.base_payload,
                       args.payload,args.native_pack,args.native_poses,pose,args.mask,
                       config=config, prefix_base_payload=args.prefix_base_payload)
        human.write_json(args.output, result)
    print(json.dumps({'mode': args.mode, 'passed': result.get('passed'), 'payload': result.get('payload'),
                      'surface_count': result.get('surface_count')}))
    return 0 if result.get('passed',True) else 2


if __name__ == '__main__':
    raise SystemExit(main())
