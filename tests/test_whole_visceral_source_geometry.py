"""Cumulative source anatomy, actual native frames and hostile receipt changes."""
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
    selected=os.environ.get('NUMILAB_HUMAN_WHOLE_VISCERAL_EVIDENCE')
    if not selected:pytest.skip('selected native whole-visceral evidence required')
    root=Path(selected).resolve();repo=g.ROOT
    config=json.loads((repo/'config/source-organ-family-composite.v2.json').read_text())
    args=[repo/'Sources',repo/'Build/myosim-fullbody',
          repo/'Build/knee-parity-registration-20260929/candidate.v6.registration.json',
          repo/'Build/lung-source-coverage-20260929/payload/bodyparts3d-myosim-torso-anatomy.nhanatomy',
          repo/'Build/organ-family-coverage-20260929/payload.final'/g.PAYLOAD_NAME,
          root/'payload.verified'/g.PAYLOAD_NAME]
    prefix=repo/'Build/lung-envelope-20260929/payload/thorax-lung-envelope.nhanatomy'
    return root,args,config,prefix


def audit(evidence,args):
    return g.audit(*args,config=evidence[2],prefix_base_payload=evidence[3])


def native_args(evidence,name='projected-neutral',pose=(),mask=63):
    root,args,*_=evidence;v=root/'final-native'/name/'views'
    return [*args,next(v.glob('*.mrvpack')),next(v.glob('*.torso-anatomy-poses.json')),pose,mask]


def test_complete_46_families_preserve_exact_source_types(evidence):
    s=g.source_selection(evidence[1][0],evidence[2]);a=s['added']
    assert len(s['families'])==46 and s['required_member_count']==571 and len(a)==192
    assert [r['stable_id'] for r in a]==list(range(388,580))
    counts={k:sum(x['layer']==k for x in a) for k in evidence[2]['layer_registry']}
    assert counts=={'organ_component':85,'vessel':0,'duct':8,'cavity_reference':2,
        'neural_region_reference':54,'ventricular_region_reference':5,'junction_reference':1,
        'ocular_region_reference':23,'ocular_muscle_reference':14}
    assert len(s['shared_family_members'])==12
    assert s['shared_family_members']['FJ2599']==['small_intestine','large_intestine']
    assert next(x for x in a if x['member_id']=='FJ2571')['myosim_body']=='pelvis'
    assert all({'concept_id':'FMA78447','label':'region of ventricular system of brain'} in x['source_is_a_types']
               for x in a if x['layer']=='ventricular_region_reference')
    assert all({'concept_id':'FMA5022','label':'muscle organ'} in x['source_is_a_types']
               for x in a if x['layer']=='ocular_muscle_reference')
    bad=[x for x in a if not x['source_topology']['exact_coordinate_quotient']['closed_oriented_manifold_candidate']]
    assert len(bad)==15 and all(not x['source_topology']['physics_admitted'] for x in a)


@pytest.mark.parametrize('change',['missing_brain','duplicate_eye','wrong_ventricular_type','shared_rectum_owner'])
def test_source_contract_refuses_rehashed_truncation_or_typing(evidence,tmp_path,change):
    c=copy.deepcopy(evidence[2]);t=json.loads((g.ROOT/c['template']['file']).read_text())
    if change=='missing_brain':next(x for x in t['regions'] if x['id']=='brain')['member_ids'].pop()
    elif change=='duplicate_eye':
        eye=next(x for x in t['regions'] if x['id']=='right_eye');eye['member_ids'].append(eye['member_ids'][0])
    elif change=='wrong_ventricular_type':c['layer_registry']['neural_region_reference']['excluded_source_types']=[]
    else:c['member_owner_overrides'].pop('FJ2571')
    if change in ['missing_brain','duplicate_eye']:
        p=tmp_path/'family.json';p.write_text(json.dumps(t));c['template']={'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    with pytest.raises(ValueError):g.source_selection(evidence[1][0],c)


def test_oracle_triangulates_relative_and_texture_indices_independently():
    raw=b'v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf -4/1/1 -3/2/1 -2/3/1 -1/4/1\n'
    v,f=g.oracle_triangles(raw,'quad.obj')
    assert v.shape==(4,3) and f.tolist()==[[0,1,2],[0,2,3]]


@pytest.mark.parametrize('raw',[b'v nan 0 0\nf 1 1 1',b'v 0 0 0\nf 0 1 1',b'v 0 0 0\nf 1 2 3'])
def test_oracle_refuses_nonfinite_zero_and_escaped_indices(raw):
    with pytest.raises(ValueError):g.oracle_triangles(raw,'invalid.obj')


@pytest.mark.parametrize('name,pose',[('raw-source-rest',None),('projected-neutral',()),('torso-flexion',((7,-.4),(8,.1),(9,.2)))])
def test_all_579_surfaces_keep_prefix_topology_and_actual_body_frames(evidence,name,pose):
    r=audit(evidence,native_args(evidence,name,pose))
    assert r['passed'] and r['surface_count']==579 and r['family_count']==46 and len(r['rows'])==192
    assert r['required_unique_source_members']==571 and r['baseline_audit']['surface_count']==387
    assert r['baseline_audit']['native_surface_count']==579 and r['baseline_audit']['passed']
    assert r['maximum_added_native_source_error_m']<2e-5
    assert {x['myosim_body'] for x in r['rows']}=={'head','neck','torso','Abdomen','pelvis'}
    assert all(x['topology_exact'] for x in r['rows'])
    assert all(not r['qualification'][k] for k in ['disjoint_tissue','connected_lumen','physical_volume','clinical_registration','mechanics'])


@pytest.mark.parametrize('change',['owner','semantic','topology','position','normal','visibility','missing','duplicate'])
def test_rehashed_native_packet_cannot_fake_new_geometry(evidence,tmp_path,change):
    args=native_args(evidence);sections=_pack_sections(args[6]);p=np.frombuffer(sections[4][0],'<u4').reshape(-1,16)
    ix=int(np.flatnonzero((p[:,4]==51027)&(p[:,5]>=388))[0]);row=p[ix];out=tmp_path/'mutated.mrvpack'
    if change in ['owner','semantic','missing','duplicate']:
        field,value={'owner':(24,20),'semantic':(16,51028),'missing':(16,59999),'duplicate':(20,int(row[5])+1)}[change]
        _mutate_pack(args[6],out,4,lambda raw,offset,_:struct.pack_into('<I',raw,offset+64*ix+field,value))
    elif change=='topology':
        def swap(raw,offset,_):
            offset+=4*int(row[0]);a,b=struct.unpack_from('<2I',raw,offset);struct.pack_into('<2I',raw,offset,b,a)
        _mutate_pack(args[6],out,3,swap)
    elif change=='visibility':
        _mutate_pack(args[6],out,5,lambda raw,offset,_:struct.pack_into('<I',raw,offset+80*int(row[3])+44,11))
    else:
        vertex=int(np.frombuffer(sections[3][0],'<u4')[int(row[0])])
        def shift(raw,offset,_):
            offset+=80*vertex+(16 if change=='normal' else 0)
            struct.pack_into('<f',raw,offset,struct.unpack_from('<f',raw,offset)[0]+.005)
        _mutate_pack(args[6],out,2,shift)
    args[6]=out
    try:
        if change in ['position','normal']:assert not audit(evidence,args)['passed']
        else:
            with pytest.raises(ValueError):audit(evidence,args)
    finally:
        # This is a synthetic, hash-repaired mutation, not retained native evidence.
        (tmp_path/'mutation-sha256.txt').write_text(hashlib.sha256(out.read_bytes()).hexdigest()+'\n');out.unlink()


@pytest.mark.parametrize('change',['head','neck','pelvis','zero_quaternion','nan_quaternion','duplicate','fingerprint'])
def test_pose_owner_identity_and_new_frames_fail_closed(evidence,tmp_path,change):
    args=native_args(evidence);d=json.loads(args[7].read_text());owner={'head':23,'neck':22,'pelvis':128}.get(change,23)
    row=next(x for x in d['bodies'] if x['body_index']==owner)
    if change in ['head','neck','pelvis']:row['position_world_m'][0]+=.02
    elif change=='zero_quaternion':row['orientation_world_xyzw']=[0,0,0,0]
    elif change=='nan_quaternion':row['orientation_world_xyzw'][0]=float('nan')
    elif change=='duplicate':d['bodies'].append(copy.deepcopy(row))
    else:d['registration_fingerprint32']+=1
    p=tmp_path/'poses.json';p.write_text(json.dumps(d));args[7]=p
    if change in ['head','neck','pelvis']:assert not audit(evidence,args)['passed']
    else:
        with pytest.raises(ValueError):audit(evidence,args)


@pytest.mark.parametrize('change',['position','normal','owner','old_prefix','physics'])
def test_rehashed_payload_preserves_all_old_and_new_sources(evidence,tmp_path,change):
    args=native_args(evidence);raw=bytearray(args[5].read_bytes());_,h,records,_,_=lung.decode(args[5])
    manifest=json.loads(args[5].with_name(g.MANIFEST_NAME).read_text())
    if change in ['position','normal']:
        offset=60+32*h[1]+24*int(records[387,1])+(12 if change=='normal' else 0)
        struct.pack_into('<f',raw,offset,struct.unpack_from('<f',raw,offset)[0]+.005)
    elif change in ['owner','old_prefix']:struct.pack_into('<I',raw,60+32*(387 if change=='owner' else 310),20)
    else:manifest['mechanics_admitted']=True
    p=tmp_path/g.PAYLOAD_NAME;p.write_bytes(raw);manifest['payload']['sha256']=hashlib.sha256(raw).hexdigest()
    p.with_name(g.MANIFEST_NAME).write_text(json.dumps(manifest));args[5]=p
    if change in ['position','normal']:assert not audit(evidence,args)['passed']
    else:
        with pytest.raises(ValueError):audit(evidence,args)


def test_independent_face_oracle_detects_compiler_parser_winding_bug(evidence,tmp_path,monkeypatch):
    original=g.human._bodyparts_obj_triangles
    def wrong(*a,**k):
        vertices,triangles=original(*a,**k);triangles=[list(t) for t in triangles]
        triangles[0][0],triangles[0][1]=triangles[0][1],triangles[0][0]
        return vertices,triangles
    with monkeypatch.context() as m:
        m.setattr(g.human,'_bodyparts_obj_triangles',wrong)
        g.compose(evidence[1][0],evidence[1][1],evidence[1][2],evidence[1][4],tmp_path,config=evidence[2])
    args=native_args(evidence);args[5]=tmp_path/g.PAYLOAD_NAME
    with pytest.raises(ValueError,match='payload exact source topology'):audit(evidence,args)


@pytest.mark.parametrize('abi,old',[(1,'torso-organ-coverage-20260929/current-neutral.v2'),
    (2,'lung-source-coverage-20260929/projected-neutral'),
    (3,'lung-envelope-20260929/final-native/projected-neutral-mask64'),
    (4,'organ-family-coverage-20260929/final-native/projected-neutral-mask256')])
def test_abi1_through_4_geometry_is_unchanged_with_new_reader(evidence,tmp_path,abi,old):
    root=evidence[0];prior=g.ROOT/'Build'/old
    cmd=json.loads((prior/'command.json').read_text());cmd[0]=str(root/'native/myosim-visual-probe');cmd[4]=str(tmp_path/'views')
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    (tmp_path/'native-command.json').write_text(json.dumps(cmd));(tmp_path/'native-stdout').write_text(r.stdout);(tmp_path/'native-stderr').write_text(r.stderr)
    assert r.returncode==0,r.stderr
    before=_pack_sections(next((prior/'views').glob('*.mrvpack')));after=_pack_sections(next((tmp_path/'views').glob('*.mrvpack')))
    assert all(before[k]==after[k] for k in [2,3,4,5])


@pytest.mark.parametrize('change',['abi','layer','mask_zero','mask_overflow','abi4_mask','repeat_mask'])
def test_native_rejects_unknown_abi_layers_and_masks_before_capture(evidence,tmp_path,change):
    root=evidence[0];cmd=json.loads((root/'final-native/projected-neutral/command.json').read_text())['argv']
    raw=bytearray(evidence[1][5].read_bytes())
    if change=='abi':struct.pack_into('<I',raw,8,6)
    elif change=='layer':struct.pack_into('<I',raw,60+32*387+24,16)
    elif change=='abi4_mask':raw=bytearray(evidence[1][4].read_bytes())
    p=tmp_path/'changed.nhanatomy';p.write_bytes(raw);cmd[4]=str(tmp_path/'views');cmd[cmd.index('--torso-anatomy-payload')+1]=str(p)
    if change in ['mask_zero','mask_overflow','abi4_mask']:cmd[cmd.index('--torso-anatomy-layer-mask')+1]={'mask_zero':'0','mask_overflow':'32768','abi4_mask':'1024'}[change]
    if change=='repeat_mask':cmd+=['--torso-anatomy-layer-mask','63']
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    (tmp_path/'command.json').write_text(json.dumps(cmd));(tmp_path/'stderr').write_text(r.stderr)
    assert r.returncode!=0 and not list((tmp_path/'views').glob('*.png'))
