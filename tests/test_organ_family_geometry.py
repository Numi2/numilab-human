"""Complete source families, typed spaces/ducts and independent native geometry."""
import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess

import numpy as np
import pytest

from numilab_human import organ_family_geometry as g
from numilab_human import lung_envelope as lung
from numilab_human.torso_anatomy_audit import _pack_sections
from test_torso_anatomy_source_audit import _mutate_pack


@pytest.fixture(scope='module')
def evidence():
    selected=os.environ.get('NUMILAB_HUMAN_ORGAN_FAMILY_EVIDENCE')
    if not selected:pytest.skip('selected native/source organ-family evidence required')
    root=Path(selected).resolve();repo=g.ROOT
    args=[repo/'Sources',repo/'Build/myosim-fullbody',
          repo/'Build/knee-parity-registration-20260929/candidate.v6.registration.json',
          repo/'Build/lung-source-coverage-20260929/payload/bodyparts3d-myosim-torso-anatomy.nhanatomy',
          repo/'Build/lung-envelope-20260929/payload/thorax-lung-envelope.nhanatomy',
          root/'payload.final'/g.PAYLOAD_NAME]
    return root,args


@pytest.fixture(scope='module')
def selection(evidence):
    return g.source_selection(evidence[1][0])


def test_declared_families_are_complete_and_spaces_are_distinct(selection):
    assert len(selection['families']) == 18 and selection['required_member_count'] == 378
    added=selection['added']
    assert len(added) == 77 and [r['stable_id'] for r in added] == list(range(311,388))
    assert {k:sum(r['layer']==k for r in added) for k in g.configuration()['layer_registry']} == {
        'organ_component':18,'vessel':40,'duct':15,'cavity_reference':4}
    assert {r['member_id'] for r in added if r['layer']=='cavity_reference'} == {'FJ2422','FJ2423','FJ2424','FJ2425'}
    assert len(selection['shared_family_members']) == 8
    assert selection['baseline_source_members']['FJ1932']['myosim_body']=='Abdomen'
    assert next(r for r in added if r['member_id']=='FJ3659')['myosim_body']=='Abdomen'
    assert all(not r['source_topology']['repair_applied'] and not r['source_topology']['physics_admitted'] for r in added)


@pytest.mark.parametrize('field',['source_lock','template','baseline','family_omission','shared_owner','type_ambiguity'])
def test_source_selection_fails_closed_on_contract_drift(evidence,monkeypatch,field):
    config=g.configuration()
    if field=='source_lock':config['source_lock_sha256']='0'*64
    elif field in ['template','baseline']:config[field if field=='template' else 'baseline_map']['sha256']='0'*64
    elif field=='family_omission':del config['family_bindings']['pancreas']
    elif field=='shared_owner':config['member_owner_overrides'].clear()
    else:config['layer_registry']['duct']['source_types'].append(['FMA67112','immaterial anatomical entity'])
    monkeypatch.setattr(g,'configuration',lambda:copy.deepcopy(config))
    with pytest.raises(ValueError):g.source_selection(evidence[1][0])


@pytest.mark.parametrize('changed',['isa_BP3D_4.0_obj_99.zip','partof_BP3D_4.0_obj_99.zip'])
def test_same_size_changed_archive_fails_before_source_selection(evidence,tmp_path,changed):
    for filename in ['isa_BP3D_4.0_obj_99.zip','partof_BP3D_4.0_obj_99.zip']:
        source=evidence[1][0]/filename;target=tmp_path/filename
        if filename==changed:
            with target.open('wb') as f:f.truncate(source.stat().st_size)
        else:target.symlink_to(source)
    with pytest.raises(ValueError,match='source archive identity'):g.source_selection(tmp_path)


@pytest.mark.parametrize('corruption',['missing','duplicate','extra'])
def test_rehashed_template_cannot_truncate_or_expand_source_family(evidence,tmp_path,monkeypatch,corruption):
    config=g.configuration();template=json.loads((g.ROOT/config['template']['file']).read_text())
    family=next(r for r in template['regions'] if r['id']=='pancreas')
    if corruption=='missing':family['member_ids'].pop()
    elif corruption=='duplicate':family['member_ids'].append(family['member_ids'][0])
    else:family['member_ids'].append('FJ1932')
    path=tmp_path/'template.json';path.write_text(json.dumps(template))
    config['template']={'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    monkeypatch.setattr(g,'configuration',lambda:copy.deepcopy(config))
    with pytest.raises(ValueError,match='complete source family membership'):g.source_selection(evidence[1][0])


@pytest.fixture(scope='module',params=[(name,pose,mask)
    for name,pose in [('raw-source-rest',None),('projected-neutral',()),('torso-flexion',((7,-.4),(8,.1),(9,.2)))]
    for mask in [63,256,512,1023]],ids=lambda q:f'{q[0]}-{q[2]}')
def native(evidence,request):
    root,args=evidence;name,pose,mask=request.param;views=root/'final-native'/f'{name}-mask{mask}'/'views'
    return [*args,next(views.glob('*.mrvpack')),next(views.glob('*.torso-anatomy-poses.json')),pose,mask]


def test_every_source_surface_and_owner_is_independently_verified(native):
    result=g.audit(*native)
    assert result['passed'] and result['surface_count']==387 and result['family_count']==18
    assert result['required_unique_source_members']==378 and len(result['rows'])==77
    assert result['baseline_audit']['passed'] and result['baseline_audit']['surface_count']==310
    assert all(r['passed'] for r in result['families'])
    assert result['maximum_added_native_source_error_m']<2e-5
    assert result['qualification']=={'source_geometry':True,'complete_declared_family_membership':True,
        'disjoint_tissue':False,'connected_lumen':False,'physical_volume':False,'clinical_registration':False,'mechanics':False}


@pytest.fixture(scope='module')
def neutral(evidence):
    root,args=evidence;views=root/'final-native/projected-neutral-mask256/views'
    return [*args,next(views.glob('*.mrvpack')),next(views.glob('*.torso-anatomy-poses.json')),(),256]


@pytest.mark.parametrize('corruption',['owner','semantic','topology','position','normal','nonfinite','visibility','pose','missing','duplicate'])
def test_rehashed_native_data_cannot_forge_source_geometry(neutral,tmp_path,corruption):
    args=list(neutral);sections=_pack_sections(args[6]);primitives=np.frombuffer(sections[4][0],'<u4').reshape(-1,16)
    ix=int(np.flatnonzero(primitives[:,5]==311)[0]);p=primitives[ix];out=tmp_path/'changed.mrvpack'
    if corruption in ['owner','semantic','missing','duplicate']:
        field,value={'owner':(24,20 if p[6]==7 else 7),'semantic':(16,51025),'missing':(16,59999),'duplicate':(20,312)}[corruption]
        _mutate_pack(args[6],out,4,lambda raw,offset,_:struct.pack_into('<I',raw,offset+64*ix+field,value))
    elif corruption=='topology':
        def swap(raw,offset,_):
            offset+=4*int(p[0]);a,b=struct.unpack_from('<2I',raw,offset);struct.pack_into('<2I',raw,offset,b,a)
        _mutate_pack(args[6],out,3,swap)
    elif corruption=='visibility':
        _mutate_pack(args[6],out,5,lambda raw,offset,_:struct.pack_into('<I',raw,offset+80*int(p[3])+44,11))
    elif corruption=='pose':
        d=json.loads(args[7].read_text());next(b for b in d['bodies'] if b['body_index']==7)['position_world_m'][0]+=.02
        out=tmp_path/'poses.json';out.write_text(json.dumps(d));args[7]=out
    else:
        vertex=int(np.frombuffer(sections[3][0],'<u4')[int(p[0])])
        def shift(raw,offset,_):
            offset+=80*vertex+(16 if corruption=='normal' else 0)
            value=float('nan') if corruption=='nonfinite' else struct.unpack_from('<f',raw,offset)[0]+.005
            struct.pack_into('<f',raw,offset,value)
        _mutate_pack(args[6],out,2,shift)
    if corruption!='pose':args[6]=out
    if corruption in ['owner','semantic','topology','nonfinite','visibility','missing','duplicate']:
        with pytest.raises(ValueError):g.audit(*args)
    else:assert not g.audit(*args)['passed']


@pytest.mark.parametrize('field',['position','normal','owner','baseline','selection','surface_receipt','qualification'])
def test_rehashed_payload_and_receipt_cannot_forge_source_proof(neutral,tmp_path,field):
    args=list(neutral);raw=bytearray(args[5].read_bytes());_,h,records,_,_=lung.decode(args[5])
    manifest=json.loads(args[5].with_name(g.MANIFEST_NAME).read_text())
    if field in ['position','normal']:
        offset=60+32*h[1]+24*int(records[310,1])+(12 if field=='normal' else 0)
        struct.pack_into('<f',raw,offset,struct.unpack_from('<f',raw,offset)[0]+.005)
    elif field in ['owner','baseline']:
        row=310 if field=='owner' else 0
        struct.pack_into('<I',raw,60+32*row,20 if records[row,0]==7 else 7)
    elif field=='selection':manifest['selection']['added'][0]['source_topology']['physics_admitted']=True
    elif field=='surface_receipt':manifest['surfaces'][0]['layer']='cavity_reference'
    else:manifest['physical_volume_admitted']=True
    payload=tmp_path/g.PAYLOAD_NAME;payload.write_bytes(raw);manifest['payload']['sha256']=hashlib.sha256(raw).hexdigest()
    payload.with_name(g.MANIFEST_NAME).write_text(json.dumps(manifest));args[5]=payload
    if field in ['position','normal']:assert not g.audit(*args)['passed']
    else:
        with pytest.raises(ValueError):g.audit(*args)


def test_independent_oracle_detects_corrupt_compiler_coordinate_helper(neutral,tmp_path,monkeypatch):
    original=g.human._bodyparts_world_to_body_stored_m
    def corrupt(*args,**kwargs):
        result=original(*args,**kwargs)
        for vertex in result:vertex[0]+=.01
        return result
    with monkeypatch.context() as m:
        m.setattr(g.human,'_bodyparts_world_to_body_stored_m',corrupt)
        g.compose(neutral[0],neutral[1],neutral[2],neutral[4],tmp_path)
    args=list(neutral);args[5]=tmp_path/g.PAYLOAD_NAME
    report=g.audit(*args)
    assert not report['passed'] and min(r['payload_local_error_m'] for r in report['rows'])>.0099


@pytest.mark.parametrize('abi,old',[('ABI1','torso-organ-coverage-20260929/current-neutral.v2'),
    ('ABI2','lung-source-coverage-20260929/projected-neutral'),
    ('ABI3','lung-envelope-20260929/final-native/projected-neutral-mask64')])
def test_old_abi_native_geometry_is_unchanged(evidence,tmp_path,abi,old):
    root,_=evidence;previous=g.ROOT/'Build'/old
    if not (previous/'command.json').exists():pytest.fail('retained historical command missing: '+str(previous))
    cmd=json.loads((previous/'command.json').read_text());cmd[0]=str(root/'native/myosim-visual-probe');cmd[4]=str(tmp_path/'views')
    if '--focus-body' in cmd:cmd[cmd.index('--focus-body')]='--focus-body-index'
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    before=_pack_sections(next((previous/'views').glob('*.mrvpack')));after=_pack_sections(next((tmp_path/'views').glob('*.mrvpack')))
    assert all(before[k]==after[k] for k in [2,3,4,5])


@pytest.mark.parametrize('corruption',['abi','layer','mask_zero','mask_overflow','repeat_mask','no_payload'])
def test_native_abi4_rejects_invalid_input_before_capture(evidence,neutral,tmp_path,corruption):
    root,_=evidence;cmd=json.loads((root/'final-native/projected-neutral-mask256/command.json').read_text())
    raw=bytearray(neutral[5].read_bytes())
    if corruption=='abi':struct.pack_into('<I',raw,8,5)
    if corruption=='layer':struct.pack_into('<I',raw,60+32*310+24,11)
    payload=tmp_path/'invalid.nhanatomy';payload.write_bytes(raw);cmd[4]=str(tmp_path/'views')
    cmd[cmd.index('--torso-anatomy-payload')+1]=str(payload)
    if corruption.startswith('mask_'):cmd[-1]='0' if corruption=='mask_zero' else '1024'
    if corruption=='repeat_mask':cmd+=['--torso-anatomy-layer-mask','256']
    if corruption=='no_payload':del cmd[cmd.index('--torso-anatomy-payload'):cmd.index('--torso-anatomy-payload')+2]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    assert result.returncode!=0 and not list((tmp_path/'views').glob('*.png'))
