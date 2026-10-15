"""Witness-local refinement selection; no geometry admission is implied."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

tools=Path(__file__).resolve().parents[1]/"tools"
sys.path.insert(0,str(tools))
spec=importlib.util.spec_from_file_location("contact_refinement",tools/"refine_resting_skin_contact_patch.py")
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def test_local_patch_is_the_union_of_actual_witness_faces():
    result=dict(count=3,triangle_pairs=[[8,3],[4,8],[3,4]])
    np.testing.assert_array_equal(tool.self_contact_refinement_faces(result,10),[3,4,8])


@pytest.mark.parametrize("result",[
    dict(count=2,triangle_pairs=[[0,1]]),
    dict(count=1,triangle_pairs=[[0.,1.]]),
    dict(count=1,triangle_pairs=[[1,1]]),
    dict(count=1,triangle_pairs=[[-1,1]]),
    dict(count=1,triangle_pairs=[[0,10]]),
    dict(count=1,triangle_pairs=[[0,1,2]]),
    dict(count=0,triangle_pairs=[]),
])
def test_missing_truncated_or_aliased_witnesses_are_not_refined(result):
    with pytest.raises(ValueError):
        tool.self_contact_refinement_faces(result,10)
