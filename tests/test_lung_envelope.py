"""Source atlas integrity, held-out registration and native ABI3 corruption checks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from numilab_human import lung_envelope as lung
from numilab_human.torso_anatomy_audit import _pack_sections
from test_torso_anatomy_source_audit import _mutate_pack


def test_source_objects_and_authored_modifiers_are_complete():
    d = lung.source_data()
    assert len(d['objects']) == 6 and len(d['source_lung_mesh_members']) == 5
    assert sum(len(o['vertices_world_m']) for o in d['objects']) == 77300
    assert sum(len(o['triangles']) for o in d['objects']) == 154432
    assert all(o['modifiers'][0]['type'] == 'SUBSURF' and o['modifiers'][0]['levels'] == 1
               for o in d['objects'])
    assert [m['type'] for m in d['objects'][-1]['modifiers']] == ['SUBSURF', 'SOLIDIFY', 'SOLIDIFY']
    assert d['source']['scripts_autoexec'] is False and d['source']['scene_unit_scale'] == 1


def test_rehashed_source_omission_cannot_pass_collection_contract(tmp_path):
    import gzip
    d = lung.source_data();d['objects'].pop(0)
    config = lung.configuration();raw = json.dumps(d).encode();compressed = gzip.compress(raw, mtime=0)
    path = tmp_path/'source.gz';path.write_bytes(compressed)
    config['export'].update(file=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                            compressed_sha256=hashlib.sha256(compressed).hexdigest())
    with pytest.raises(ValueError, match='complete source objects'):lung.source_data(config)


def test_lobe_openings_and_right_lower_source_defects_are_reported():
    rows = lung.source_topology()
    assert [r['boundary_edge_count'] for r in rows] == [14,16,20,34,60,0]
    assert all(not r['closed_oriented_manifold_candidate'] for r in rows[:5])
    assert rows[1]['nonmanifold_edges'] == 8 and rows[1]['vertex_manifold_defect_ids'] == 9
    assert rows[-1]['closed_oriented_manifold_candidate'] and rows[-1]['face_component_count'] == 8
    assert all(r['fluid_or_tissue_volume_admitted'] is False for r in rows)


@pytest.fixture(scope='module')
def inputs():
    root = os.environ.get('NUMILAB_HUMAN_LUNG_ENVELOPE_EVIDENCE')
    if not root:pytest.skip('selected native lung-envelope evidence required')
    root = Path(root).resolve();repo = lung.ROOT
    return [repo/'Sources',repo/'Build/myosim-fullbody',
            repo/'Build/knee-parity-registration-20260929/candidate.v6.registration.json',
            repo/'Build/lung-source-coverage-20260929/payload/bodyparts3d-myosim-torso-anatomy.nhanatomy',
            root/'payload/thorax-lung-envelope.nhanatomy'],root


def test_held_out_anchors_do_not_change_the_fitted_transform(inputs):
    args,_ = inputs;d = lung.source_data();before = lung.registration(d,args[0])
    moved = copy.deepcopy(d)
    for anchor in moved['registration_anchors']:
        if anchor['object_name'] not in lung.configuration()['fit_anchor_names']:
            for v in anchor['vertices_world_m']:v[0] += .1
    after = lung.registration(moved,args[0])
    assert before['source_world_m_to_bodyparts_world_m'] == after['source_world_m_to_bodyparts_world_m']
    assert before['metrics']['fit'] == after['metrics']['fit']
    assert after['metrics']['held_out']['rms_m'] > .09
    assert before['metrics']['held_out']['rms_m'] < .0045 and before['clinical_admitted'] is False


def test_proper_registration_preserves_atlas_lung_laterality(inputs):
    args,_ = inputs;d = lung.source_data();r = lung.registration(d,args[0])
    matrix = np.asarray(r['source_world_m_to_bodyparts_world_m'])
    assert r['rotation_determinant'] > .999999999 and r['uniform_scale'] > 0
    for o in d['objects'][:5]:
        centroid = lung.area_centroid(o['vertices_world_m'],o['triangles'])
        x = (matrix[:3,:3]@centroid+matrix[:3,3])[0]
        assert x > 0 if 'left lung' in o['object_name'] else x < 0


def test_native_abi2_retains_published_packed_geometry(inputs,tmp_path):
    args,root = inputs
    old = lung.ROOT/'Build/lung-source-coverage-20260929/projected-neutral'
    cmd = json.loads((old/'command.json').read_text());cmd[0] = str(root/'native.final/myosim-visual-probe')
    cmd[4] = str(tmp_path/'views');result = subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0,result.stderr
    current = _pack_sections(next((tmp_path/'views').glob('*.mrvpack')))
    previous = _pack_sections(next((old/'views').glob('*.mrvpack')))
    assert all(current[k] == previous[k] for k in [2,3,4,5])


@pytest.fixture(scope='module',params=[(name,pose,mask)
    for name,pose in [('raw-source-rest',None),('projected-neutral',()),
                      ('torso-flexion',((7,-.4),(8,.1),(9,.2)))] for mask in [63,64,128,255]],
    ids=lambda q:f'{q[0]}-{q[2]}')
def native(inputs,request):
    args,root = inputs;name,pose,mask = request.param;views = root/'final-native'/f'{name}-mask{mask}'/'views'
    return [*args,next(views.glob('*.mrvpack')),next(views.glob('*.torso-anatomy-poses.json')),pose,mask]


def test_all_310_surfaces_and_visibility_are_source_verified(native):
    report = lung.audit(*native)
    assert report['passed'] and report['surface_count'] == 310
    assert report['maximum_lung_native_source_error_m'] < 2e-5
    assert sum(r['vertex_count'] for r in report['rows']) == 77300
    assert report['baseline_audit']['passed'] and report['baseline_audit']['surface_count'] == 304
    assert all(r['topology_exact'] for r in report['rows'])


@pytest.fixture(scope='module')
def neutral(inputs):
    args,root = inputs;v = root/'final-native/projected-neutral-mask64/views'
    return [*args,next(v.glob('*.mrvpack')),next(v.glob('*.torso-anatomy-poses.json')),(),64]


@pytest.mark.parametrize('corruption',['owner','semantic','topology','displacement','normal','nonfinite','visibility','pose'])
def test_lung_oracle_detects_rehashed_geometry_and_pose_corruption(neutral,tmp_path,corruption):
    args = list(neutral);sections = _pack_sections(args[5])
    p = np.frombuffer(sections[4][0],'<u4').reshape(-1,16)
    ix = int(np.flatnonzero(p[:,5] == 305)[0]);row = p[ix];dest = tmp_path/'changed.mrvpack'
    if corruption in ['owner','semantic']:
        field,value = (24,7) if corruption == 'owner' else (16,51024)
        _mutate_pack(args[5],dest,4,lambda raw,off,_:struct.pack_into('<I',raw,off+64*ix+field,value))
    elif corruption == 'visibility':
        _mutate_pack(args[5],dest,5,lambda raw,off,_:struct.pack_into('<I',raw,off+80*int(row[3])+44,0))
    elif corruption == 'topology':
        def swap(raw,off,_):
            offset = off+4*int(row[0]);a,b = struct.unpack_from('<2I',raw,offset);struct.pack_into('<2I',raw,offset,b,a)
        _mutate_pack(args[5],dest,3,swap)
    elif corruption in ['displacement','normal','nonfinite']:
        vertex = int(np.frombuffer(sections[3][0],'<u4')[int(row[0])])
        def shift(raw,off,_):
            offset = off+80*vertex+(16 if corruption == 'normal' else 0)
            value = float('nan') if corruption == 'nonfinite' else struct.unpack_from('<f',raw,offset)[0]+.005
            struct.pack_into('<f',raw,offset,value)
        _mutate_pack(args[5],dest,2,shift)
    else:
        d = json.loads(args[6].read_text());next(b for b in d['bodies'] if b['body_index']==20)['position_world_m'][0] += .02
        dest = tmp_path/'poses.json';dest.write_text(json.dumps(d));args[6] = dest
    if corruption != 'pose':args[5] = dest
    if corruption in ['owner','semantic','visibility','topology','nonfinite']:
        with pytest.raises(ValueError):lung.audit(*args)
    else:
        result = lung.audit(*args)
        assert not result['passed'] and not result['qualification']['source_geometry']


@pytest.mark.parametrize('field',['position','normal','owner','baseline'])
def test_rehashed_payload_must_preserve_source_geometry(neutral,tmp_path,field):
    args = list(neutral);raw = bytearray(args[4].read_bytes())
    _,header,records,_,_ = lung.decode(args[4]);first = int(records[304,1]);vertex_offset = 60+32*header[1]+24*first
    if field in ['position','normal']:
        offset = vertex_offset+(12 if field == 'normal' else 0)
        struct.pack_into('<f',raw,offset,struct.unpack_from('<f',raw,offset)[0]+.005)
    else:struct.pack_into('<I',raw,60+32*(304 if field == 'owner' else 0),7)
    payload = tmp_path/args[4].name;payload.write_bytes(raw)
    manifest = json.loads(args[4].with_name('thorax-lung-envelope.manifest.json').read_text())
    manifest['payload']['sha256'] = hashlib.sha256(raw).hexdigest()
    payload.with_name('thorax-lung-envelope.manifest.json').write_text(json.dumps(manifest));args[4] = payload
    if field in ['owner','baseline']:
        with pytest.raises(ValueError):lung.audit(*args)
    else:assert not lung.audit(*args)['passed']


@pytest.mark.parametrize('field',['alignment','source_topology','surfaces','base_payload_sha256'])
def test_payload_receipt_cannot_forge_registration_or_source_coverage(neutral,tmp_path,field):
    args = list(neutral);payload = tmp_path/args[4].name;payload.write_bytes(args[4].read_bytes())
    manifest = json.loads(args[4].with_name('thorax-lung-envelope.manifest.json').read_text())
    if field == 'alignment':manifest[field]['metrics']['held_out']['rms_m'] = 0
    elif field == 'source_topology':manifest[field][0]['closed_oriented_manifold_candidate'] = True
    elif field == 'surfaces':manifest[field][0]['object_name'] = 'Superior lobe of left lung'
    else:manifest[field] = '0'*64
    payload.with_name('thorax-lung-envelope.manifest.json').write_text(json.dumps(manifest));args[4] = payload
    with pytest.raises(ValueError):lung.audit(*args)


@pytest.mark.parametrize('corruption',['abi','layer','mask_zero','mask_overflow'])
def test_native_abi3_rejects_unknown_layers_and_visibility(neutral,inputs,tmp_path,corruption):
    _,root = inputs;cmd = json.loads((root/'final-native/projected-neutral-mask64/command.json').read_text())
    raw = bytearray(neutral[4].read_bytes())
    if corruption == 'abi':struct.pack_into('<I',raw,8,4)
    if corruption == 'layer':struct.pack_into('<I',raw,60+32*304+24,9)
    payload = tmp_path/'invalid.nhanatomy';payload.write_bytes(raw)
    cmd[4] = str(tmp_path/'views');cmd[cmd.index('--torso-anatomy-payload')+1] = str(payload)
    if corruption.startswith('mask_'):cmd[-1] = '0' if corruption == 'mask_zero' else '256'
    result = subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    assert result.returncode != 0 and not list((tmp_path/'views').glob('*.png'))
