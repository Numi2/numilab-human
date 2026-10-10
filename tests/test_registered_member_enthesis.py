import copy
import math

import numpy as np
import pytest

from numilab_human import model as human


def fixture():
    target = {
        "core_body_index": 140,
        "default_com_position_world_m": [1., 2., 3.],
        "default_inertial_quaternion_world_xyzw": [0., 0., math.sqrt(.5), math.sqrt(.5)],
    }
    def anchor(member, translation):
        return {
            "source": {"member_id": member},
            "target": {"name": "toes_r", "core_body_index": 140},
            "registration": {"source_obj_mm_to_core_inertial_body_m": [
                [.001, 0., 0., translation[0]],
                [0., .001, 0., translation[1]],
                [0., 0., .001, translation[2]],
                [0., 0., 0., 1.],
            ]},
        }
    records = {
        ("toes_r", "ray1"): anchor("ray1", [0., 0., 0.]),
        ("toes_r", "ray2"): anchor("ray2", [.004, -.002, .001]),
    }
    return target, records


def test_named_enthesis_uses_its_own_member_transform_on_shared_toes_body():
    target, records = fixture()
    points = [[10., 20., 30.], [0., 0., 0.]]
    first = human._bodyparts_registered_member_world_vertices(points, "toes_r", "ray1", records, target)
    second = human._bodyparts_registered_member_world_vertices(points, "toes_r", "ray2", records, target)
    # The ray offset is expressed in the toes frame and rotates with that body.
    np.testing.assert_allclose(np.asarray(second) - first, [[.002, .004, .001]] * 2, atol=1e-14)
    np.testing.assert_allclose(first[0], [.98, 2.01, 3.03], atol=1e-14)


def test_equal_member_registrations_preserve_legacy_body_transform_exactly():
    target, records = fixture()
    records[("toes_r", "ray2")]["registration"] = copy.deepcopy(records[("toes_r", "ray1")]["registration"])
    points = [[10., 20., 30.], [0., 0., 0.]]
    expected = human._bodyparts_source_mm_to_body_world(
        points, target["default_com_position_world_m"],
        target["default_inertial_quaternion_world_xyzw"], [0., 0., 0.], [0., 0., 0., 1.], 1.,
    )
    assert human._bodyparts_registered_member_world_vertices(points, "toes_r", "ray2", records, target) == expected


@pytest.mark.parametrize("fault", ["missing", "member", "body", "core", "matrix"])
def test_registered_enthesis_fails_closed_on_missing_or_mismatched_anchor(fault):
    target, records = fixture()
    anchor = records[("toes_r", "ray2")]
    if fault == "missing":
        del records[("toes_r", "ray2")]
    elif fault == "member":
        anchor["source"]["member_id"] = "ray1"
    elif fault == "body":
        anchor["target"]["name"] = "toes_l"
    elif fault == "core":
        anchor["target"]["core_body_index"] = 154
    else:
        del anchor["registration"]["source_obj_mm_to_core_inertial_body_m"]
    with pytest.raises(human.ImportError):
        human._bodyparts_registered_member_world_vertices([[0., 0., 0.]], "toes_r", "ray2", records, target)
