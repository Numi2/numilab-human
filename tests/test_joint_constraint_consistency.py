"""Independent source-constraint oracles, not loaded/clinical qualification."""
import copy
import math
import os
from pathlib import Path

import numpy as np
import pytest

from numilab_human import model as human
from numilab_human.joint_constraint_consistency import (
    joint_equality_driver_domain_audit, polynomial_driver_witnesses,
    source_equality_projection_oracle,
)
from numilab_human.upper_limb_pose_audit import (
    PROJECTED_JOINT_RANGE_TOLERANCE, _pose_joint_range_context,
    PoseAuditError, _finish_pose_audit, _pose_qpos, _project_joint_equalities,
)


def test_stationary_witness_finds_conflict_hidden_between_passing_endpoints():
    witnesses = polynomial_driver_witnesses([0., 4., -4., 0., 0.], [0., 1.], 0., 0., np)
    assert [row['driver_value'] for row in witnesses] == pytest.approx([0., .5, 1.])
    assert [row['dependent_value'] for row in witnesses] == pytest.approx([0., 1., 0.])
    assert witnesses[1]['kind'] == 'numerical_polynomial_stationary_point'


@pytest.mark.parametrize('coefficients,domain,driver_ref,dependent_ref,expected', [
    ([.1], [-1., 2.], 0., .2, [.3, .3]),
    ([0., 4., -4.], [.2, 1.2], .2, .3, [.3, 1.3]),
    ([.0256, -.256, .96, -1.6, 1.], [0., 1.], 0., 0., [0., .1296]),
    ([.3, .2], [.4, .4], 0., 0., [.38, .38]),
])
def test_constant_reference_shift_and_flat_quartic_extrema(coefficients, domain, driver_ref, dependent_ref, expected):
    witnesses = polynomial_driver_witnesses(coefficients, domain, driver_ref, dependent_ref, np)
    values = [row['dependent_value'] for row in witnesses]
    assert [min(values), max(values)] == pytest.approx(expected, abs=1e-12)
    assert all(domain[0] <= row['driver_value'] <= domain[1] for row in witnesses)


@pytest.mark.parametrize('coefficients,domain', [([math.nan], [0.,1.]), ([1.], [2.,1.]), ([], [0.,1.])])
def test_invalid_domain_inputs_rejected(coefficients, domain):
    with pytest.raises(RuntimeError, match='invalid polynomial or driver bounds'):
        polynomial_driver_witnesses(coefficients, domain, 0., 0., np)


@pytest.fixture
def source_model():
    mujoco = pytest.importorskip('mujoco')
    model = mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/>
      <worldbody><body><joint name="driver" ref=".2"/><geom size=".05"/>
        <body pos="0 0 .2"><joint name="dependent" ref="-.1"/><geom size=".05"/>
        </body></body></worldbody><equality><joint name="coupling" joint1="dependent"
        joint2="driver" polycoef=".01 2 0 0 0"/></equality></mujoco>''')
    return model, mujoco


def test_projection_preserves_joint_reference_offsets_against_engine_oracle(source_model):
    model, mujoco = source_model
    q, count, _ = _pose_qpos(model, ((0,.3),), mujoco, np)
    assert count == 1 and q[1] == pytest.approx(.11)
    data=mujoco.MjData(model); data.qpos[:]=q; mujoco.mj_forward(model,data)
    check=source_equality_projection_oracle(model,data,mujoco,PROJECTED_JOINT_RANGE_TOLERANCE)
    assert check['passed'] and check['complete_unique_coverage']
    assert check['maximum_absolute_residual'] <= 1e-15
    # The previous absolute-coordinate calculation looks plausible, but the
    # source engine independently rejects its reference-frame mistake.
    data.qpos[1]=.01+2*.3; mujoco.mj_forward(model,data)
    check=source_equality_projection_oracle(model,data,mujoco,PROJECTED_JOINT_RANGE_TOLERANCE)
    assert not check['passed'] and check['maximum_absolute_residual'] == pytest.approx(.5)


def test_inactive_authored_equality_does_not_move_the_pose(source_model):
    model, mujoco=source_model; model.eq_active0[0]=False
    q=np.asarray(model.qpos0).copy(); before=q.copy()
    count,correction=_project_joint_equalities(model,q,mujoco)
    assert count==0 and correction==0. and np.array_equal(q,before)
    data=mujoco.MjData(model); data.qpos[:]=q; mujoco.mj_forward(model,data)
    check=source_equality_projection_oracle(model,data,mujoco,PROJECTED_JOINT_RANGE_TOLERANCE)
    assert check['passed'] and check['expected_active_joint_equalities']==0


def test_missing_or_duplicate_constraint_rows_cannot_claim_projection_coverage(source_model):
    from types import SimpleNamespace
    model,mujoco=source_model
    for indices in ([], [0,0]):
        data=SimpleNamespace(nefc=len(indices),efc_type=[int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)]*len(indices),
                             efc_id=indices,efc_pos=[0.]*len(indices))
        check=source_equality_projection_oracle(model,data,mujoco,PROJECTED_JOINT_RANGE_TOLERANCE)
        assert not check['passed'] and not check['complete_unique_coverage']


def test_independent_oracle_failure_rejects_an_otherwise_passing_pose_audit(source_model):
    model,mujoco=source_model
    data=mujoco.MjData(model);data.qpos[:]=[.3,.61];mujoco.mj_forward(model,data)
    oracle=source_equality_projection_oracle(model,data,mujoco,PROJECTED_JOINT_RANGE_TOLERANCE)
    receipt={'status':'passed_test_pose','inputs':{
                 'registration':{'sha256':'unit-test-registration'},
                 'runtime_reference':{'rigid':{'sha256':'unit-test-rigid'}}},
             'default_frame_maximum_centroid_residual_m':0.,'default_frame_maximum_allowed_residual_m':1e-9,
             'poses':[{'name':'corrupted','continuity':[],'bilateral_gap_parity':[],
                       'source_equality_projection_oracle':oracle}]}
    with pytest.raises(PoseAuditError,match='differs from MuJoCo oracle') as error:
        _finish_pose_audit(receipt,'lower-limb')
    assert error.value.result['poses'][0]['source_equality_projection_oracle']['maximum_absolute_residual']==pytest.approx(.5)


def test_unbounded_or_chained_driver_domain_is_retained_as_unverified():
    driver={'id':0,'name':'driver','qpos_address':0,'limited':False,'range':[0.,0.],'type':3}
    dep={'id':1,'name':'dependent','qpos_address':1,'limited':True,'range':[-1.,1.],'type':3}
    eq={'id':0,'name':'coupling','dependent_joint':1,'master_joint':0,'polycoef':[0.,1.,0.,0.,0.],
        'dependent_reference':0.,'master_reference':0.}
    source={'joints':[driver,dep],'joint_equalities':[eq]}
    joined=[{'source_joint_id':1,'native_dof':{'flags':human._MR_DOF_POSITION_LIMIT,'position_range':[-1.,1.]},'core_limit_status':'enforced'}]
    receipt=joint_equality_driver_domain_audit(source,joined,np,1e-9)
    assert receipt['unverified_driver_domain_count']==1 and receipt['status']=='unverified_driver_domains'
    altered=copy.deepcopy(source);altered['joint_equalities'][0]['master_joint']=1
    receipt=joint_equality_driver_domain_audit(altered,joined,np,1e-9)
    assert receipt['rows'][0]['reason']=='driver_is_itself_equality_dependent_requires_composition'


@pytest.fixture(scope='module')
def wholebody():
    required=['NUMILAB_HUMAN_MOTION_SOURCES','NUMILAB_HUMAN_NATIVE_REFERENCE_ARTIFACT','NUMILAB_HUMAN_MOTION_REPAIRED']
    if any(not os.environ.get(x) for x in required):
        pytest.skip('exact pinned whole-body source and native inputs were not supplied')
    import json,mujoco
    from myo_sim.build.compose import build_model
    from numilab_human.myosim_export import export_fullbody
    sources,artifact,registration=[Path(os.environ[x]) for x in required]
    reference,_=human._bodyparts_runtime_bindings(json.loads(registration.read_text()),artifact)
    model=build_model('myofullbody')
    joined=_pose_joint_range_context(artifact,reference,model,mujoco)
    return model,mujoco,export_fullbody(sources),joined


def test_all_source_driver_domain_conflict_witnesses_join_engine_constraints(wholebody):
    model,mujoco,exported,joined=wholebody
    check=joint_equality_driver_domain_audit(exported,joined,np,PROJECTED_JOINT_RANGE_TOLERANCE)
    assert check['expected_source_equalities']==check['evaluated_equalities']==51
    assert check['source_range_conflict_count']==38
    assert check['unverified_driver_domain_count']==0
    original=copy.deepcopy(exported['joint_equalities'])
    count=0
    for row in check['rows']:
        for witness in row['witnesses']:
            if not witness['source_range_conflict']:continue
            pose=((row['driver_q_index'],witness['driver_value']),) if row['driver_q_index'] is not None else ()
            q,_,_=_pose_qpos(model,pose,mujoco,np)
            assert q[row['dependent_q_index']]==pytest.approx(witness['dependent_value'],abs=1e-12)
            data=mujoco.MjData(model);data.qpos[:]=q;mujoco.mj_forward(model,data)
            oracle=source_equality_projection_oracle(model,data,mujoco,PROJECTED_JOINT_RANGE_TOLERANCE)
            assert oracle['passed'] and oracle['measured_joint_equalities']==51
            assert witness['source_range_violation']>PROJECTED_JOINT_RANGE_TOLERANCE
            count+=1
    assert count>=38 and exported['joint_equalities']==original
