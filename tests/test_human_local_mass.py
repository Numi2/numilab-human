from pathlib import Path
import sys,numpy as np,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_human_local_mass import interface_mass_scope

def test_only_baseline_cells_touching_changed_interface_enter_mass_scope():
 cells=np.array([[0,1,2,3],[4,5,6,7],[7,8,9,10]])
 operations=[{'new_cells':[[0,1,2,4]]}]
 assert interface_mass_scope(cells,operations,11).tolist()==[True,True,False]
 assert not interface_mass_scope(cells,[],11).any()

@pytest.mark.parametrize('bad',[[[0,1,2,11]],[[0,1,2,-1]],[[0.,1.,2.,3.]],[[0,1,2]]])
def test_invalid_original_node_id_cannot_expand_mass_scope(bad):
 with pytest.raises(ValueError):interface_mass_scope(np.array([[0,1,2,3]]),[{'new_cells':bad}],11)
