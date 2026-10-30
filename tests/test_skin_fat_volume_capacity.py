import importlib.util
from pathlib import Path
import numpy as np
import pytest

# Also supports direct execution from an isolated source snapshot.
_here = Path(__file__).resolve()
_owner = (_here.parent / "skin_fat_volume_capacity.py")
if not _owner.exists():
    _owner = _here.parents[1] / "src/numilab_human/skin_fat_volume_capacity.py"
_spec = importlib.util.spec_from_file_location("skin_fat_volume_capacity", _owner)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)
TET = np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]], np.int64)

def test_simple_and_disconnected_bounds():
    one = m.closed_shell_bound(TET, label="one")
    two = m.closed_shell_bound(np.vstack((TET,TET+4)), label="two")
    assert (one["boundary_euler_characteristic"], one["necessary_minimum_cells"]) == (2,1)
    assert (two["boundary_euler_characteristic"], two["necessary_minimum_cells"]) == (4,2)

def test_genus_one_bound():
    idx = lambda i,j: (i%3)*3+(j%3)
    faces = []
    for i in range(3):
        for j in range(3):
            a,b,c,d = idx(i,j),idx(i+1,j),idx(i+1,j+1),idx(i,j+1)
            faces += [[a,b,c],[a,c,d]]
    bound = m.closed_shell_bound(faces, label="torus")
    assert bound["boundary_euler_characteristic"] == 0
    assert bound["necessary_minimum_cells"] == 9

@pytest.mark.parametrize("bad",[
    TET[:3], np.vstack((TET,TET[:1])), TET[:,[0,0,2]],
    np.vstack((TET[:3],TET[3:,[0,2,1]])),
])
def test_reject_invalid_boundary(bad):
    with pytest.raises(ValueError):
        m.closed_shell_bound(bad,label="bad")

def test_combined_shared_inner_nodes_and_capacity():
    surfaces = {role:{"faces":TET} for role in ("O","I","D")}
    result = m.volume_capacity_report(TET,TET,surfaces,max_nodes=12,max_cells=2)
    assert result["status"] == "PASS_NECESSARY_BOUNDS_ONLY"
    assert result["physical_admission"] is False
    assert result["necessary_combined_nodes"] == 12
    assert result["necessary_combined_cells"] == 2
    assert m.volume_capacity_report(TET,TET,surfaces,max_nodes=11)["status"] == "FAIL_NATIVE_CAPACITY"
    assert m.volume_capacity_report(TET,TET,surfaces,max_cells=1)["status"] == "FAIL_NATIVE_CAPACITY"
