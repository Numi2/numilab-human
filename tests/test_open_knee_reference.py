"""Admission tests: silent source substitutions must not create runnable cases."""
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from numilab_human.open_knee_reference import (
    audit_program, cylindrical_chain_audit, freeze_reference_case, run_reference_case, xml_record,
)


def source():
    return ET.fromstring('''<febio_spec version="2.5">
      <Module type="solid"/>
      <Material><material id="1" name="bone" type="rigid body"/></Material>
      <Boundary><rigid rb="1" node_set="proximal"/></Boundary>
      <Discrete><discrete_material id="1" type="nonlinear spring">
        <force lc="1">1</force></discrete_material>
        <discrete dmat="1" discrete_set="spring"/></Discrete>
      <LoadData><loadcurve id="1" type="linear"><point>0,0</point>
        <point>1,1</point></loadcurve></LoadData>
      <Step><Contact><contact type="sliding-elastic" surface_pair="pair">
        <laugon>0</laugon><penalty>0.1</penalty></contact></Contact></Step>
      </febio_spec>''')


def geometry():
    return ET.fromstring('''<febio_spec><Geometry>
      <NodeSet name="proximal"><node id="1"/></NodeSet>
      <Elements name="tissue" mat="1" type="tet4"><elem id="1">1,2,3,4</elem></Elements>
      <Surface name="a"/><Surface name="b"/>
      <SurfacePair name="pair"><master surface="a"/><slave surface="b"/></SurfacePair>
      <DiscreteSet name="spring"><delem>1,2</delem></DiscreteSet>
      </Geometry></febio_spec>''')


def test_reference_closure_is_not_native_admission():
    result = audit_program(source(), geometry())
    assert result['reference_errors'] == []
    assert result['native_admission']['executable'] is False
    assert all(x['status'] == 'retained_not_executed'
               for x in result['native_admission']['sections'])


@pytest.mark.parametrize('tag,code', [('NodeSet', 'unresolved_node_set'),
    ('DiscreteSet', 'unresolved_discrete_set'), ('SurfacePair', 'unresolved_surface_pair')])
def test_missing_sets_rejected(tag, code):
    g = geometry()
    parent = g.find('Geometry')
    parent.remove(parent.find(tag))
    assert code in {e['code'] for e in audit_program(source(), g)['reference_errors']}


def test_loadcurve_and_rigid_id_references_are_checked():
    r = source()
    r.find('Boundary/rigid').set('rb', '2')
    r.find('Discrete/discrete_material/force').set('lc', '2')
    assert {e['code'] for e in audit_program(r, geometry())['reference_errors']} == {
        'unresolved_rb', 'unresolved_lc'}


def test_unknown_nested_material_parameter_and_order_survive():
    r = source()
    mat = r.find('Material/material')
    extra = ET.SubElement(mat, 'future_law', {'type': 'must-not-default'})
    ET.SubElement(extra, 'coefficient', {'lc': '1'}).text = '0.0139'
    ET.SubElement(extra, 'coefficient').text = '116.22'
    record = xml_record(mat)
    assert [c['text'] for c in record['children'][0]['children']] == ['0.0139', '116.22']
    result = audit_program(r, geometry())
    assert result['native_admission']['executable'] is False
    assert len(result['native_admission']['sections']) == len(r)


def test_distributed_fibres_require_complete_local_element_ids():
    r = source()
    data = ET.SubElement(ET.SubElement(r, 'MeshData'), 'ElementData',
                         {'var': 'fiber', 'elem_set': 'tissue'})
    ET.SubElement(data, 'elem', {'lid': '2'}).text = '1,0,0'
    assert 'incomplete_element_data' in {e['code'] for e in audit_program(r, geometry())['reference_errors']}


def chain():
    r = ET.Element('febio_spec')
    for i, axis in enumerate(('1,0,0', '0,1,0', '0,0,1')):
        joint = ET.SubElement(r, 'constraint', {'type': 'rigid cylindrical joint',
                                              'name': f'Patellar_{i}'})
        for name, value in {'body_a':str(i), 'body_b':str(i+1),
                            'joint_axis':axis, 'joint_origin':'2,3,5',
                            'prescribed_translation':'0', 'prescribed_rotation':'0'}.items():
            ET.SubElement(joint,name).text = value
    return r


def test_cylindrical_chain_has_six_free_directions_and_preserves_graph():
    r = chain()
    # XML order is not graph order.
    last = r[-1]; r.remove(last); r.insert(0, last)
    result = cylindrical_chain_audit(r)
    assert result['rank'] == 6
    assert result['body_material_ids'] == [0,1,2,3]
    assert len(result['free_coordinates']) == 6
    for joint in r:
        joint.find('prescribed_rotation').text = '1'
    assert cylindrical_chain_audit(r)['rank'] == 3


def test_parallel_axes_do_not_masquerade_as_six_independent_directions():
    r = chain()
    for joint in r:
        joint.find('joint_axis').text = '1,0,0'
    assert cylindrical_chain_audit(r)['rank'] == 2


def test_source_identity_drift_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match='source identity drift'):
        freeze_reference_case(tmp_path, tmp_path/'out', human_revision='test', matter_revision='test')
    assert not (tmp_path/'out').exists()


def test_freeze_retains_full_program_and_never_rebinds_missing_geometry(tmp_path, monkeypatch):
    import hashlib
    import json
    from numilab_human import open_knee
    directory = tmp_path / 'source'
    directory.mkdir()
    r = source()
    ET.SubElement(r, 'Geometry', {'from': r'C:\source\Geometry_custom.feb'})
    (directory/'FeBio_custom.feb').write_bytes(ET.tostring(r))
    (directory/'Geometry.feb').write_bytes(ET.tostring(geometry()))
    (directory/'ModelProperties.xml').write_text('<metadata/>')
    (directory/'license.txt').write_text('fixture')
    monkeypatch.setattr(open_knee, 'EXPECTED_HASHES', {
        p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()})
    out = tmp_path/'case'
    result = freeze_reference_case(directory, out, human_revision='h', matter_revision='m')
    assert result['geometry_binding'] == 'diagnostic_only'
    assert result['status'] == 'blocked_reference_dependencies'
    assert result['reference_errors'][0]['code'] == 'missing_authored_include'
    assert result['reference_solver']['status'] == 'not_run'
    assert result['reference_solver']['version'] is None
    doc = json.loads((out/'mechanical-program.json').read_text())['document']
    assert doc == xml_record(r)
    assert (out/'source/FeBio_custom.feb').read_bytes() == (directory/'FeBio_custom.feb').read_bytes()
    with pytest.raises(FileExistsError):
        freeze_reference_case(directory, out, human_revision='h', matter_revision='m')


def runnable_case(tmp_path, monkeypatch):
    import hashlib
    from numilab_human import open_knee
    directory = tmp_path / 'source'
    directory.mkdir()
    r = source()
    ET.SubElement(r, 'Geometry', {'from': r'C:\source\Geometry_custom.feb'})
    (directory/'FeBio_custom.feb').write_bytes(ET.tostring(r))
    (directory/'Geometry.feb').write_bytes(ET.tostring(geometry()))
    (directory/'Geometry_custom.feb').write_bytes(ET.tostring(geometry()))
    (directory/'license.txt').write_text('fixture')
    monkeypatch.setattr(open_knee, 'EXPECTED_HASHES', {
        p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
        if p.name != 'Geometry_custom.feb'})
    out = tmp_path/'case'
    result = freeze_reference_case(directory, out, human_revision='h', matter_revision='m')
    assert result['reference_errors'] == []
    assert result['geometry_binding'] == 'authored_dependency'
    return directory, out


def test_runnable_case_relocates_only_include_and_retains_source_bytes(tmp_path, monkeypatch):
    directory, out = runnable_case(tmp_path, monkeypatch)
    original = (directory/'FeBio_custom.feb').read_bytes()
    assert (out/'source/FeBio_custom.feb').read_bytes() == original
    assert (out/'execution/FeBio_custom.feb').read_bytes() == original.replace(
        b'C:\\source\\Geometry_custom.feb', b'../source/Geometry_custom.feb')


def test_frozen_geometry_mutation_rejected_before_solver_runs(tmp_path, monkeypatch):
    _, out = runnable_case(tmp_path, monkeypatch)
    (out/'source/Geometry_custom.feb').write_text('changed')
    with pytest.raises(ValueError, match='frozen source identity drift'):
        run_reference_case(out, Path('/bin/false'))


def test_solver_failure_is_retained_and_cannot_be_overwritten(tmp_path, monkeypatch):
    import json
    import shutil
    _, out = runnable_case(tmp_path, monkeypatch)
    binary = Path(shutil.which('false'))
    result = run_reference_case(out, binary)
    assert result['reference_solver']['returncode'] != 0
    assert result['reference_solver']['status'] == 'failed_missing_log'
    assert not result['qualification']['source_equivalence']
    assert result == json.loads((out/'receipt.json').read_text())
    with pytest.raises(ValueError, match='already attempted'):
        run_reference_case(out, binary)


@pytest.mark.parametrize('version,exitcode,expected', [
    ('2.9.1', 0, 'completed_source_protocol'),
    ('3.0.0', 0, 'failed_or_nonmatching_reference'),
    ('2.9.1', 1, 'failed_or_nonmatching_reference'),
])
def test_completed_protocol_does_not_promote_source_equivalence(tmp_path, monkeypatch, version, exitcode, expected):
    import shutil
    from types import SimpleNamespace
    from numilab_human.open_knee_febio_log import CONNECTOR_IDS, FIELDS, RIGID_BODY_IDS
    from numilab_human import open_knee_reference
    _, out = runnable_case(tmp_path, monkeypatch)
    # Synthetic process output exercises receipt admission only, not mechanics.
    text = f'--- version-{version} ---\n------- converged at time : 2\n'
    for i, (name, (width, _)) in enumerate(FIELDS.items(), 1):
        text += f'\nData Record #{i}\n=====\nStep = 1\nTime = 2\nData = {name}\n'
        ids = CONNECTOR_IDS if name.startswith('Rigid_Connector') else RIGID_BODY_IDS
        text += ''.join(str(j) + ' ' + ' '.join(['0']*width) + '\n' for j in sorted(ids))
    text += '\nNumber of time steps completed ... : 1\nN O R M A L   T E R M I N A T I O N\n'

    def process(*args, **kwargs):
        (kwargs['cwd']/'FeBio_custom.log').write_text(text)
        return SimpleNamespace(returncode=exitcode)

    monkeypatch.setattr(open_knee_reference.subprocess, 'run', process)
    result = run_reference_case(out, Path(shutil.which('false')))
    assert result['reference_solver']['status'] == expected
    assert not any(result['qualification'].values())


def test_comparison_version_and_config_are_explicit_and_frozen(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    from numilab_human import open_knee_reference
    _, out = runnable_case(tmp_path, monkeypatch)
    config = tmp_path/'superlu.xml'
    config.write_text('<febio_config version="1.0"><linear_solver type="superlu"/></febio_config>')
    original = config.read_bytes()
    def process(command, **kwargs):
        assert command[-2] == '-config'
        assert Path(command[-1]).read_bytes() == original
        config.write_text('modified after capture')
        (kwargs['cwd']/'FeBio_custom.log').write_text('version-2.9.0')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(open_knee_reference.subprocess, 'run', process)
    result = run_reference_case(out, Path(shutil.which('false')), config=config, comparison_version='2.9.0')
    run = result['reference_solver']
    assert run['comparison_version'] == '2.9.0'
    assert run['status'] == 'failed_or_nonmatching_reference'
    assert (out/'execution/reference-config.xml').read_bytes() == original
    assert not any(result['qualification'].values())


@pytest.mark.parametrize('version', ['2.9.1', 'latest', ''])
def test_invalid_comparison_version_rejected_before_run(tmp_path, monkeypatch, version):
    _, out = runnable_case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='comparison version'):
        run_reference_case(out, Path('/bin/false'), comparison_version=version)


def test_febio3_frame_translation_preserves_fiber_and_isochoric_prestrain():
    import numpy as np
    from numilab_human.open_knee_reference import _translate_febio3_material_frames
    root = ET.Element('febio_spec')
    materials = ET.SubElement(root, 'Material')
    for name in ['QAT','ACL','PCL','MCL','LCL','PTL']:
        m = ET.SubElement(materials, 'material', name=name, type='uncoupled prestrain elastic')
        e = ET.SubElement(m, 'elastic', type='trans iso Mooney-Rivlin')
        ET.SubElement(e, 'c1').text = '1.95'
        p = ET.SubElement(m, 'prestrain', type='in-situ stretch')
        ET.SubElement(p, 'stretch', lc='3').text = '1'
        ET.SubElement(m, 'fiber', type='vector').text = '.36,-.48,.8'
    before = ET.tostring(root)
    records = _translate_febio3_material_frames(root)
    assert len(records) == 6
    for material, record in zip(materials, records):
        a = np.array(record['normalized_axis']);d = np.array(record['transverse_seed'])
        c = np.cross(a, d);c /= np.linalg.norm(c)
        b = np.cross(c, a);q = np.column_stack([a,b,c])
        assert np.allclose(q.T @ q, np.eye(3), atol=1e-15)
        assert np.linalg.det(q) == pytest.approx(1, abs=1e-15)
        assert np.allclose(q @ [1,0,0], a, atol=1e-15)
        for stretch in [1,1.016,1.027,1.034]:
            f = q @ np.diag([stretch,stretch**-.5,stretch**-.5]) @ q.T
            expected = np.eye(3)*stretch**-.5 + np.outer(a,a)*(stretch-stretch**-.5)
            assert np.allclose(f, expected, atol=1e-15)
            assert np.linalg.det(f) == pytest.approx(1, abs=1e-15)
        # Undo only the declared change; every other node/attribute/value
        # must still match the input, including its load-curve assignment.
        material.remove(material.find('mat_axis'))
        material.find('elastic').remove(material.find('elastic/fiber'))
        ET.SubElement(material,'fiber',type='vector').text = '.36,-.48,.8'
    assert ET.tostring(root) == before


def test_translated_deck_cannot_be_reported_as_original_version(tmp_path, monkeypatch):
    import json
    _, out = runnable_case(tmp_path, monkeypatch)
    receipt = out/'receipt.json'
    r = json.loads(receipt.read_text())
    r['execution_deck']['required_comparison_version'] = '3.0.0'
    receipt.write_text(json.dumps(r))
    with pytest.raises(ValueError, match='explicit comparison version'):
        run_reference_case(out, Path('/bin/false'))
