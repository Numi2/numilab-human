from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from human_stable_cell_order import replacement_in_reference_order

def test_order_and_complete_ancestry_stay_separate():
 cells=np.array([[0,1,2,3],[0,1,3,4]],dtype=np.int32);off=np.array([0,2,3]);parents=np.array([7,9,8])
 reference=cells[::-1][:,[1,0,3,2]]
 out,o,p,perm=replacement_in_reference_order(cells,off,parents,reference)
 assert np.array_equal(out,reference) and np.array_equal(perm,[1,0])
 assert np.array_equal(o,[0,1,3]) and np.array_equal(p,[8,7,9])

@pytest.mark.parametrize('failure',['changed','duplicate','empty_ancestry','domain'])
def test_wrong_identity_or_ancestry_is_rejected(failure):
 cells=np.array([[0,1,2,3],[0,1,3,4]]);reference=cells.copy();off=np.array([0,1,2]);parents=np.array([5,6])
 if failure=='changed':reference[0,0]=9
 if failure=='duplicate':reference[1]=reference[0]
 if failure=='empty_ancestry':off[1]=0
 if failure=='domain':reference=reference[:1]
 with pytest.raises(ValueError):replacement_in_reference_order(cells,off,parents,reference)
