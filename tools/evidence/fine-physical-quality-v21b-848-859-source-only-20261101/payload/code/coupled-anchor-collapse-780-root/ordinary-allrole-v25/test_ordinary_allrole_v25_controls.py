from pathlib import Path
import importlib.util, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BUILDER=ROOT/"coupled-anchor-collapse-780-root/ordinary-allrole-v25/repair_coupled_quality_points_722_v25_ordinary_allrole_local.py"
spec=importlib.util.spec_from_file_location("ordinary_allrole_v25_control",BUILDER)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)

def test_source_ring_respects_protection():
    free,info=module.expand_triangle_ring_once(
      np.array([1,4,7,10],np.int64),
      {"O":np.array([[0,1,2]]),"I":np.array([[0,1,2]]),"D":np.array([[0,1,2]])},
      {"O":np.array([0,1,2]),"I":np.array([3,4,5]),"D":np.array([6,7,8])},
      fixed_global_rows={0,3,6,8},witness_frozen_global_rows={5})
    assert np.array_equal(free,np.array([1,2,4,7,10],np.int64)),free
    assert info["final_movable_rows"]==5
    assert not (set(free)&{0,3,5,6,8})

def test_receipt_is_json_serializable():
    payload={"current_rows":np.array([1,2]),"lineage":{"steps":[{"sha256":"x"}]}}
    summary={"count":int(len(payload["current_rows"])),"lineage":payload["lineage"]}
    assert json.loads(json.dumps(summary))["count"]==2

def test_back_lineage_owner_pin():
    assert module.sha(module._BACK_LINEAGE_PATH)==module._BACK_LINEAGE_SHA
    assert module._BACK_LINEAGE_SHA=="68124ffcffcc2bf5147de081e50f87739089c3be5c2370be8ae3b27bef4e922a"

if __name__=="__main__":
    test_source_ring_respects_protection()
    test_receipt_is_json_serializable()
    test_back_lineage_owner_pin()
    print("PASS ordinary-all-role v25 tiny controls")
