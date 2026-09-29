"""Source-authored lung-lobe/pleural surfaces in the existing native anatomy ABI.

Atlas envelopes are visual anatomy, not pulmonary parenchymal microstructure,
clinical registration, fluid domains, materials, motion or mechanics.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'config/zanatomy-thorax-source.v1.json'
HEADER = struct.Struct('<8s5I32s')


def require(condition, message):
    if not condition:
        raise ValueError('lung envelope: ' + message)


def configuration():
    return json.loads(CONFIG.read_text())


def source_data(config=None):
    config = configuration() if config is None else config
    require(hashlib.sha256((ROOT / config['export']['exporter_file']).read_bytes()).hexdigest()
            == config['export']['exporter_sha256'], 'source exporter hash')
    compressed = (ROOT / config['export']['file']).read_bytes()
    require(hashlib.sha256(compressed).hexdigest() == config['export']['compressed_sha256'], 'compressed source hash')
    raw = gzip.decompress(compressed)
    require(hashlib.sha256(raw).hexdigest() == config['export']['sha256'], 'source export hash')
    d = json.loads(raw)
    require(d['schema'] == 'numi.human.zanatomy-thorax-blender-export.v1', 'source schema')
    require(d['source']['blend_sha256'] == config['source']['blend']['sha256']
            and d['source']['exporter_sha256'] == config['export']['exporter_sha256']
            and d['source']['registration_sha256'] == config['registration_sha256']
            and d['source']['scene_unit_scale'] == 1 and d['source']['scripts_autoexec'] is False,
            'source provenance/units')
    expected = [s['object_name'] for s in config['objects']]
    require([o['object_name'] for o in d['objects']] == expected and len(set(expected)) == 6, 'complete source objects')
    require(d['source_lung_mesh_members'] == sorted(expected[:-1]), 'complete source lung collection')
    require({a['object_name']: a['bodyparts_member_id'] for a in d['registration_anchors']}
            == config['registration_anchor_members'] and len(d['registration_anchors']) == 27,
            'source registration anchor identity')
    return d


def area_centroid(vertices, triangles):
    import numpy as np
    v, f = np.asarray(vertices, float), np.asarray(triangles, int)
    require(v.ndim == 2 and v.shape[1] == 3 and np.isfinite(v).all()
            and f.ndim == 2 and f.shape[1] == 3 and f.min() >= 0 and f.max() < len(v), 'source geometry arrays')
    q = v[f]
    weights = np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1)
    require(weights.sum() > 0, 'nonzero source surface area')
    return (q.mean(axis=1) * weights[:, None]).sum(axis=0) / weights.sum()


def registration(d, sources: Path, config=None):
    """Fit only the declared 13 anchors; retain the other 14 as diagnostics."""
    import numpy as np
    from . import model as human
    config = configuration() if config is None else config
    rows = []
    for a in d['registration_anchors']:
        archive, member, obj = human._bodyparts_obj_member(sources, 'is_a', a['bodyparts_member_id'])
        v, f = human._bodyparts_obj_triangles(obj, member)
        rows.append({'object_name': a['object_name'], 'member_id': a['bodyparts_member_id'],
                     'source_member_sha256': hashlib.sha256(obj).hexdigest(),
                     'role': 'fit' if a['object_name'] in config['fit_anchor_names'] else 'held_out',
                     'source_centroid_m': area_centroid(a['vertices_world_m'], a['triangles']).tolist(),
                     'target_centroid_m': (area_centroid(v, f) * .001).tolist()})
    fitting = [r for r in rows if r['role'] == 'fit']
    require(len(fitting) == 13 and len(rows) - len(fitting) == 14, 'disjoint fit/held-out anchor counts')
    x, y = (np.array([r[key] for r in fitting]) for key in ['source_centroid_m', 'target_centroid_m'])
    xc, yc = x-x.mean(0), y-y.mean(0)
    require(np.linalg.matrix_rank(xc) == 3 and np.linalg.matrix_rank(yc) == 3, 'registration rank')
    u, _, vt = np.linalg.svd(xc.T @ yc)
    rotation = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    scale = float((yc * (xc @ rotation.T)).sum() / (xc * xc).sum())
    translation = y.mean(0) - scale * rotation @ x.mean(0)
    require(scale > 0 and np.isfinite(scale) and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            and abs(np.linalg.det(rotation)-1) < 1e-12, 'proper uniform similarity')
    matrix = np.eye(4);matrix[:3, :3] = rotation * scale;matrix[:3, 3] = translation
    for r in rows:
        r['centroid_error_m'] = float(np.linalg.norm(matrix[:3, :3] @ r['source_centroid_m']
                                                   + translation - r['target_centroid_m']))
    metrics = {}
    for role in ['fit', 'held_out']:
        errors = np.array([r['centroid_error_m'] for r in rows if r['role'] == role])
        metrics[role] = {'count': len(errors), 'rms_m': float(np.sqrt(np.mean(errors**2))),
                         'p90_m': float(np.quantile(errors, .9)), 'maximum_m': float(errors.max())}
    return {'method': 'proper_uniform_similarity_of_13_area_weighted_thorax_bone_centroids',
            'source_world_m_to_bodyparts_world_m': matrix.tolist(), 'uniform_scale': scale,
            'rotation_determinant': float(np.linalg.det(rotation)), 'rows': rows, 'metrics': metrics,
            'clinical_admitted': False, 'boundary': 'Fourteen held-out atlas centroid discrepancies are diagnostics, not exact corresponding landmarks or a clinical acceptance threshold.'}


def decode(path: Path):
    import numpy as np
    raw = path.read_bytes()
    require(len(raw) >= 60, 'payload header size')
    magic, abi, n, nv, ni, fingerprint, source = HEADER.unpack_from(raw)
    require(magic == b'NHANAT1\0' and abi in [2, 3, 4, 5] and 0 < n <= 1024
            and 0 < nv <= 1000000 and 0 < ni <= 6000000 and ni % 3 == 0,
            'payload header')
    offset = 60 + n*32;end = offset+nv*24
    require(len(raw) == end+ni*4, 'payload byte count')
    records = np.frombuffer(raw, '<u4', count=n*8, offset=60).reshape(-1, 8)
    vertices = np.frombuffer(raw, '<f4', count=nv*6, offset=offset).reshape(-1, 6)
    indices = np.frombuffer(raw, '<u4', count=ni, offset=end)
    require(np.isfinite(vertices).all(), 'finite payload vertices')
    return raw, (abi, n, nv, ni, fingerprint, source), records, vertices, indices


def oracle_model(sources, artifact, registration_path):
    import mujoco
    from myo_sim.build.compose import build_model
    from . import model as human
    config = configuration()
    require(human.sha256(registration_path) == config['registration_sha256'], 'bone registration hash')
    checks = human._require_myosim_rigid_program(sources, artifact)
    reg = json.loads(registration_path.read_text())
    _, owners = human._bodyparts_runtime_bindings(reg, artifact)
    m = build_model('myofullbody');rest = mujoco.MjData(m);rest.qpos[:] = m.qpos0;mujoco.mj_forward(m, rest)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    require(sid == owners['torso'][1]['source_body_id'], 'source torso owner')
    return m, rest, sid, owners['torso'][0], reg, checks


def compose(sources: Path, artifact: Path, registration_path: Path, base_payload: Path, output: Path):
    import numpy as np
    from . import model as human
    config = configuration();d = source_data(config)
    require(human.sha256(base_payload) == config['base_payload_sha256'], 'base anatomy payload hash')
    raw, header, records, base_v, base_i = decode(base_payload)
    require(header[0] == 2 and header[1] == config['base_surface_count'], 'base anatomy ABI/count')
    m, rest, sid, owner, reg, checks = oracle_model(sources, artifact, registration_path)
    alignment = registration(d, sources, config)
    global_matrix = np.asarray(reg['coordinate_system']['global_source_mm_to_myosim_world_m'])
    mm = np.diag([1000., 1000., 1000., 1.])
    world_matrix = global_matrix @ mm @ np.asarray(alignment['source_world_m_to_bodyparts_world_m'])
    vertices = [base_v];indices = [base_i];new_records = [];nv, ni = header[2:4]
    for spec, surface in zip(config['objects'], d['objects'], strict=True):
        v = np.asarray(surface['vertices_world_m']);n = np.asarray(surface['normals_world'])
        world = v @ world_matrix[:3, :3].T + world_matrix[:3, 3]
        local = (world-rest.xipos[sid]) @ rest.ximat[sid].reshape(3, 3)
        normal = n @ world_matrix[:3, :3].T;normal /= np.linalg.norm(normal, axis=1)[:, None]
        normal = normal @ rest.ximat[sid].reshape(3, 3)
        f = np.asarray(surface['triangles'], dtype='<u4').ravel()
        new_records.append(struct.pack('<8I', owner, nv, len(v), ni, len(f), spec['stable_id'], spec['layer_code'], 0))
        vertices.append(np.column_stack([local, normal]).astype('<f4'));indices.append(f+nv)
        nv += len(v);ni += len(f)
    payload = HEADER.pack(b'NHANAT1\0', 3, len(records)+6, nv, ni, header[4], header[5])
    payload += records.tobytes()+b''.join(new_records)+b''.join(v.tobytes() for v in vertices)+b''.join(i.astype('<u4').tobytes() for i in indices)
    output.mkdir(parents=True, exist_ok=True);path = output/'thorax-lung-envelope.nhanatomy';path.write_bytes(payload)
    manifest = {'schema': 'numi.human.zanatomy-lung-envelope-composite-payload.v1',
                'payload': {'sha256': human.sha256(path), 'abi': 3, 'surfaces': 310, 'vertices': nv, 'indices': ni},
                'source_configuration_sha256': human.sha256(CONFIG), 'source_export_sha256': config['export']['sha256'],
                'base_payload_sha256': human.sha256(base_payload), 'registration_sha256': human.sha256(registration_path),
                'alignment': alignment, 'world_matrix': world_matrix.tolist(), 'surfaces': config['objects'],
                'source_topology': source_topology(),
                'rigid_source_program_checks': checks, 'license': 'CC-BY-4.0 AND CC-BY-SA-4.0 AND Apache-2.0',
                'coverage': {'right_lobes': 3, 'left_lobes': 2, 'pleural_surfaces': 1, 'baseline_surfaces': 304},
                'geometry_repair_applied': False, 'parenchymal_microstructure': False, 'mechanics_admitted': False,
                'boundary': config['boundary']}
    human.write_json(output/'thorax-lung-envelope.manifest.json', manifest)
    return manifest


@lru_cache(maxsize=1)
def source_topology():
    import numpy as np
    from .cardiac_cavity_geometry import analyze_topology
    rows = []
    for surface in source_data()['objects']:
        v, f = np.asarray(surface['vertices_world_m']), np.asarray(surface['triangles'])
        t = analyze_topology(surface['vertices_world_m'], surface['triangles'])
        keys = ['vertex_count', 'face_count', 'boundary_edge_count', 'nonmanifold_edges',
                'orientation_defect_edges', 'degenerate_face_ids', 'duplicate_face_ids',
                'vertex_manifold_defect_ids', 'unused_vertex_ids', 'repeated_vertex_face_ids',
                'face_component_count', 'euler_characteristic',
                'closed_oriented_manifold_candidate']
        q = v[f];vol = float(np.einsum('ij,ij->i', q[:, 0], np.cross(q[:, 1], q[:, 2])).sum()/6)
        rows.append({'object_name': surface['object_name'], **{k: len(t[k]) if isinstance(t[k], list) else t[k] for k in keys},
                     'signed_surface_integral_m3': vol, 'fluid_or_tissue_volume_admitted': False,
                     'geometry_modified': False, 'self_intersections': 'not_checked'})
    return rows


def audit(sources, artifact, registration_path, base_payload, payload, native_pack, native_poses, pose, mask=255,
          *, native_surface_count=310, hidden_source_surface_ids=()):
    import mujoco
    import numpy as np
    from . import model as human
    from .torso_anatomy_audit import _pack_sections, audit_torso_anatomy, anatomy_hidden_ids
    from .upper_limb_pose_audit import _pose_qpos
    config = configuration();d = source_data(config)
    require(human.sha256(base_payload) == config['base_payload_sha256'], 'base anatomy payload hash')
    old_raw, old_header, old_records, old_v, old_i = decode(base_payload)
    raw, header, records, vertices, indices = decode(payload)
    require(header[0:2] == (3, 310) and header[4:] == old_header[4:], 'composite header identity')
    require(records[:304].tobytes() == old_records.tobytes() and vertices[:len(old_v)].tobytes() == old_v.tobytes()
            and indices[:len(old_i)].tobytes() == old_i.tobytes(), 'unchanged complete baseline geometry')
    manifest = json.loads(payload.with_name('thorax-lung-envelope.manifest.json').read_text())
    require(manifest['payload']['sha256'] == human.sha256(payload)
            and manifest['source_configuration_sha256'] == human.sha256(CONFIG)
            and manifest['source_export_sha256'] == config['export']['sha256'], 'composite manifest source identity')
    require(manifest['payload'] == {'sha256': human.sha256(payload), 'abi': 3, 'surfaces': 310,
                                   'vertices': header[2], 'indices': header[3]}
            and manifest['base_payload_sha256'] == human.sha256(base_payload)
            and manifest['registration_sha256'] == human.sha256(registration_path), 'composite manifest input identity')
    require(manifest['source_topology'] == source_topology(), 'source topology diagnostic')
    alignment = registration(d, sources, config)
    require(manifest['alignment'] == alignment and manifest['surfaces'] == config['objects'], 'source registration or surface provenance')
    m, rest, sid, owner, reg, checks = oracle_model(sources, artifact, registration_path)
    require({k:v for k,v in manifest['rigid_source_program_checks'].items() if k != 'file'}
            == {k:v for k,v in checks.items() if k != 'file'} and manifest['geometry_repair_applied'] is False
            and manifest['parenchymal_microstructure'] is False and manifest['mechanics_admitted'] is False,
            'composite source program or qualification boundary')
    g = np.asarray(reg['coordinate_system']['global_source_mm_to_myosim_world_m']) @ np.diag([1000.,1000.,1000.,1.])
    world_matrix = g @ np.asarray(alignment['source_world_m_to_bodyparts_world_m'])
    require(np.array_equal(manifest['world_matrix'], world_matrix), 'source composite world matrix')
    base_audit = audit_torso_anatomy(sources, artifact, registration_path, base_payload, native_pack, native_poses,
                                    pose, native_surface_count=native_surface_count, visible_layer_mask=mask,
                                    hidden_source_surface_ids=hidden_source_surface_ids)
    snapshot = json.loads(native_poses.read_text())
    hidden = anatomy_hidden_ids(snapshot, hidden_source_surface_ids, native_surface_count)
    require(snapshot.get('visible_layer_mask') == mask and 0 < mask <= 32767
            and 310 <= native_surface_count <= 1024, 'native visibility profile')
    sections = _pack_sections(native_pack)
    pv = np.frombuffer(sections[2][0], '<f4').reshape(-1,20)
    pi = np.frombuffer(sections[3][0], '<u4')
    primitives = np.frombuffer(sections[4][0], '<u4').reshape(-1,16)
    iu = np.frombuffer(sections[5][0], '<u4').reshape(-1,20)
    transform = np.frombuffer(sections[5][0], '<f4').reshape(-1,20)
    selected = primitives[np.isin(primitives[:,4], [51023,51024])]
    require(len(selected) == 6 and set(selected[:,5]) == set(range(305,311)), 'native complete lung/pleural coverage')
    native_by_id = {int(r[5]):r for r in selected}
    data = mujoco.MjData(m);data.qpos[:] = m.qpos0 if pose is None else _pose_qpos(m,pose,mujoco,np)[0]
    mujoco.mj_forward(m,data)
    native_pose = next(b for b in snapshot['bodies'] if b['body_index'] == owner)
    quaternion = np.asarray(native_pose['orientation_world_xyzw'])
    position = np.asarray(native_pose['position_world_m'])
    native_rotation = np.empty(9)
    mujoco.mju_quat2Mat(native_rotation, quaternion[[3, 0, 1, 2]] / np.linalg.norm(quaternion))
    native_rotation = native_rotation.reshape(3, 3)
    position_error = float(np.linalg.norm(position - data.xipos[sid]))
    orientation_error = float(np.linalg.norm(native_rotation - data.ximat[sid].reshape(3, 3), axis=0).max())
    rows = []
    next_vertex, next_index = len(old_v), len(old_i)
    for spec, surface, record in zip(config['objects'],d['objects'],records[304:],strict=True):
        body,fv,nv,fi,ni,stable,layer,reserved = map(int,record)
        require(body == owner and stable == spec['stable_id'] and layer == spec['layer_code'] and reserved == 0,
                'lung surface owner/identity')
        v, f = np.asarray(surface['vertices_world_m']),np.asarray(surface['triangles'])
        face_normals = np.cross(v[f[:, 1]]-v[f[:, 0]], v[f[:, 2]]-v[f[:, 0]])
        n = np.zeros_like(v)
        for corner in range(3): np.add.at(n, f[:, corner], face_normals)
        lengths = np.linalg.norm(n, axis=1)
        n[lengths <= 1e-12] = v[lengths <= 1e-12]-v.mean(axis=0)
        require(bool((np.linalg.norm(n, axis=1) > 1e-12).all()), 'source lung degenerate normal')
        n /= np.linalg.norm(n, axis=1)[:, None]
        require(float(np.linalg.norm(n-np.asarray(surface['normals_world']), axis=1).max()) <= 2e-5,
                'source exported normal direction')
        require(nv == len(v) and ni == f.size and fv+nv <= len(vertices) and fi+ni <= len(indices), 'lung surface ranges')
        require(fv == next_vertex and fi == next_index, 'contiguous source geometry ownership')
        next_vertex += nv; next_index += ni
        require(np.array_equal(indices[fi:fi+ni]-fv,f.ravel()), 'exact source lung topology')
        world = v@world_matrix[:3,:3].T+world_matrix[:3,3]
        local = (world-rest.xipos[sid])@rest.ximat[sid].reshape(3,3)
        expected_world = local@data.ximat[sid].reshape(3,3).T+data.xipos[sid]
        normal = n@world_matrix[:3,:3].T;normal /= np.linalg.norm(normal,axis=1)[:,None]
        normal = normal@rest.ximat[sid].reshape(3,3)
        p = native_by_id[stable];pf,pc,_,inst = map(int,p[:4])
        require(int(p[4]) == (51023 if layer==7 else 51024) and int(p[6]) == owner, 'native lung semantic/owner')
        require(pc == ni and pf+pc <= len(pi) and inst < len(iu), 'native lung ranges')
        first = int(pi[pf:pf+pc].min())
        require(np.array_equal(pi[pf:pf+pc]-first,f.ravel()) and first+nv <= len(pv), 'native exact lung topology')
        require(bool(np.isfinite(pv[first:first+nv,:3]).all())
                and bool(np.isfinite(pv[first:first+nv,4:7]).all()), 'finite native lung geometry/normals')
        require(int(iu[inst,9]) == owner and int(iu[inst,10]) == 3 and np.array_equal(iu[inst,12:16],p[4:8]),
                'native lung instance identity')
        require(int(iu[inst,11]) == (11 if mask&(1<<(layer-1)) and stable not in hidden else 0), 'native lung visibility')
        require(np.array_equal(transform[inst,:8],[0,0,0,1,0,0,0,1]), 'native lung local transform')
        errors = {'payload_local_error_m':float(np.linalg.norm(vertices[fv:fv+nv,:3]-local,axis=1).max()),
                  'native_pack_local_error_m':float(np.linalg.norm(pv[first:first+nv,:3]-local,axis=1).max()),
                  'native_pose_world_error_m':float(np.linalg.norm(pv[first:first+nv,:3]@native_rotation.T+position-expected_world,axis=1).max()),
                  'native_normal_error':float(np.linalg.norm(pv[first:first+nv,4:7]-normal,axis=1).max()),
                  'payload_normal_error':float(np.linalg.norm(vertices[fv:fv+nv,3:]-normal,axis=1).max())}
        normal_unit_error = float(max(np.abs(np.linalg.norm(vertices[fv:fv+nv,3:],axis=1)-1).max(),
                                      np.abs(np.linalg.norm(pv[first:first+nv,4:7],axis=1)-1).max()))
        rows.append({'object_name':surface['object_name'],'stable_id':stable,'layer':spec['layer'],
                     'vertex_count':nv,'triangle_count':len(f),'topology_exact':True,'visible':bool(mask&(1<<(layer-1))),
                     **errors,'native_COM_position_error_m':position_error,
                     'native_orientation_unit_witness_error_m':orientation_error,'normal_unit_error':normal_unit_error,
                     'passed':max(errors.values()) <= 2e-5 and normal_unit_error <= .002
                              and max(position_error, orientation_error) <= 1e-6})
    require(len(indices) == int(records[-1,3]+records[-1,4]) and len(vertices) == int(records[-1,1]+records[-1,2]),
            'complete composite geometry ownership')
    passed = base_audit['passed'] and all(r['passed'] for r in rows)
    return {'schema':'numi.human.native-lung-envelope-source-audit.v1','passed':passed,
            'surface_count':310,'vertex_count':len(vertices),'baseline_audit':base_audit,'rows':rows,
            'maximum_lung_native_source_error_m':max(r['native_pose_world_error_m'] for r in rows),
            'atlas_registration':alignment,'source_topology':source_topology(),'visible_layer_mask':mask,
            'allowed_geometry_and_normal_error':2e-5,'allowed_COM_and_orientation_witness_error':1e-6,
            'allowed_normal_unit_error':.002,'qualification':{'source_geometry':passed,'clinical_registration':False,
            'mechanics':False,'parenchymal_microstructure':False},'boundary':config['boundary']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['compose','audit'])
    for name in ['sources','artifact','registration','base-payload','output']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['payload','native-pack','native-poses']:p.add_argument('--'+name,type=Path)
    p.add_argument('--mask',type=int,default=255);p.add_argument('--raw-source-rest',action='store_true')
    p.add_argument('--pose-q',nargs=2,action='append',default=[])
    a=p.parse_args()
    if a.raw_source_rest and a.pose_q:p.error('raw source rest cannot have pose coordinates')
    if a.mode=='compose':r=compose(a.sources,a.artifact,a.registration,a.base_payload,a.output)
    else:
        if not all([a.payload,a.native_pack,a.native_poses]):p.error('audit requires payload, native-pack and native-poses')
        pose=None if a.raw_source_rest else tuple((int(i),float(v)) for i,v in a.pose_q)
        r=audit(a.sources,a.artifact,a.registration,a.base_payload,a.payload,a.native_pack,a.native_poses,pose,a.mask)
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'mode':a.mode,'passed':r.get('passed'),'payload':r.get('payload'),'surface_count':r.get('surface_count')}))
    return 0 if r.get('passed',True) else 2


if __name__=='__main__':raise SystemExit(main())
