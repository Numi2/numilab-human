"""An aperture declaration must identify every directed source boundary edge."""
import copy
import numpy as np
import pytest
from numilab_human.model import ImportError as HumanImportError
from numilab_human.resting_skin_boundaries import audit_retained_capture_boundaries

def inputs():
    # Two open triangular surfaces stand in for two named source apertures.
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    positions = np.array([[0,0,0],[1,0,0],[0,1,0],
                          [2,0,0],[3,0,0],[2,1,0]], dtype=np.float32)
    cap = (positions, faces.copy(), None, np.arange(6, dtype=np.uint64)+100)
    interfaces = [dict(interface="ocular surface opening; passive eyeball interface",
                       source_vertex_ids=list(row)) for row in ([0,1,2],[3,4,5])]
    return faces, 6, cap, 100, interfaces

def test_exact_directed_source_rings_survive_capture_without_geometry_change():
    args=inputs(); before=copy.deepcopy(args)
    result=audit_retained_capture_boundaries(*args)
    assert result["passed"] and result["unexplained_boundary_edge_count"] == 0
    assert result["boundary_loop_count"] == 2 and result["boundary_edge_count"] == 6
    assert not result["closed_volume_or_enclosure_admitted"]
    assert not result["self_intersection_checked"]
    assert np.array_equal(args[0],before[0])
    assert all(a is b is None or np.array_equal(a,b)
               for a,b in zip(args[2],before[2]))

def test_rotating_a_directed_cycle_preserves_its_identity():
    args=list(inputs());args[4][0]["source_vertex_ids"]=[1,2,0]
    assert audit_retained_capture_boundaries(*args)["passed"]

@pytest.mark.parametrize("mutation",[
    "reverse_ring","foreign_ring","duplicate_ring","wrong_offset",
    "same_count_changed_capture_faces","duplicate_capture_gids","new_hole",
    "unnamed_interface","nonfinite_capture",
])
def test_same_counts_or_nearby_geometry_cannot_authorize_a_different_boundary(mutation):
    args=list(inputs())
    if mutation=="reverse_ring":args[4][0]["source_vertex_ids"]=[2,1,0]
    elif mutation=="foreign_ring":args[4][0]["source_vertex_ids"]=[0,1,5]
    elif mutation=="duplicate_ring":args[4][1]=copy.deepcopy(args[4][0])
    elif mutation=="wrong_offset":args[3]=101
    elif mutation=="same_count_changed_capture_faces":args[2][1][0]=[0,1,5]
    elif mutation=="duplicate_capture_gids":args[2][3][0]=args[2][3][1]
    elif mutation=="new_hole":args[0]=args[0][:1]
    elif mutation=="unnamed_interface":args[4][0]["interface"]="unknown opening"
    elif mutation=="nonfinite_capture":args[2][0][0,0]=np.nan
    with pytest.raises((HumanImportError,ValueError)):
        audit_retained_capture_boundaries(*args)
