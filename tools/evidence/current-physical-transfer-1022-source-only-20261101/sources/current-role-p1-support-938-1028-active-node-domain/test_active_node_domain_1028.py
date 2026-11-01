import importlib.util
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OWNER=HERE/"current_domain_p1_export_937_active_node_domain_1028.py"
spec=importlib.util.spec_from_file_location("active937",OWNER)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def reject(fn):
    try: fn()
    except ValueError: return
    raise AssertionError("invalid active-node mapping unexpectedly passed")

points=np.asarray([[0,0,0],[9,9,9],[1,0,0],[0,1,0],[8,8,8]],dtype=np.float32)
active=np.asarray([0,2,3],dtype=np.int64)
node=np.column_stack((active+1,points[active])).astype(np.float64)
receipt=mod.validate_active_role_domain(node,active,points,points,active,5)
assert receipt["active_rows"]==3 and receipt["inactive_rows"]==2
assert receipt["active_coverage"]=="exact_bijection"
reject(lambda: mod.validate_active_role_domain(node,active,points,points,np.asarray([0,2]),5))
wrong=node.copy();wrong[1,0]=5
reject(lambda: mod.validate_active_role_domain(wrong,active,points,points,active,5))
wrong_points=points.copy();wrong_points[4,0]=7
reject(lambda: mod.validate_active_role_domain(node,active,points,wrong_points,active,5))
print("PASS active rows exact, full inactive support retained, missing map/wrong ID/wrong support coordinates reject")
