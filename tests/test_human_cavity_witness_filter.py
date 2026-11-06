"""Native-witness filtering must never guess incomplete or foreign indices."""
import importlib.util,sys
from pathlib import Path
import pytest
TOOLS=Path(__file__).resolve().parents[1]/"tools"
sys.path.insert(0,str(TOOLS))
spec=importlib.util.spec_from_file_location("filter_cavities",TOOLS/"filter_human_cavities_from_native_witness.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_only_exact_failures_are_excluded():
 w=dict(complete=True,diagnostic_only=True,cavity_indices=[1,4])
 assert m.retained_indices(6,w)==[0,2,3,5]
@pytest.mark.parametrize("rows",[[1,1],[-1],[6],[1.0],[True],[]])
def test_invalid_or_duplicate_witness_is_rejected(rows):
 with pytest.raises(ValueError):m.retained_indices(6,dict(complete=True,diagnostic_only=True,cavity_indices=rows))
def test_partial_failure_list_is_rejected():
 with pytest.raises(ValueError):m.retained_indices(6,dict(complete=False,diagnostic_only=True,cavity_indices=[1]))
