
from pathlib import Path
import importlib.util
import numpy as np

BUILDER = Path(__file__).resolve().parent / "repair_coupled_quality_points_722_witness_v16.py"
spec = importlib.util.spec_from_file_location("one_ring_v16_under_test", BUILDER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# One triangle ring only: seed 0 adds 1/2; it does not recursively add 3/4/5.
faces={"D":np.asarray([[0,1,2],[2,3,4],[4,5,6]],np.int64),
       "O":np.empty((0,3),np.int64),"I":np.empty((0,3),np.int64)}
grows={"D":np.arange(10,17,dtype=np.int64),
       "O":np.arange(30,33,dtype=np.int64),"I":np.asarray([100,101,102],np.int64)}
movable,info=mod.expand_triangle_ring_once([10],faces,grows)
assert movable.tolist()==[10,11,12]
assert 13 not in movable and info["new_ring_only_rows"]==2
assert info["role_incidence"]["D"]["incident_source_faces"]==1

# Fixed core/rim/route and witness-frozen rows are reapplied after ring expansion.
movable,info=mod.expand_triangle_ring_once([10],faces,grows,fixed_global_rows=[11],witness_frozen_global_rows=[12])
assert movable.tolist()==[10]
assert info["fixed_or_witness_rows_reexcluded_after_expansion"]==2

# A shared I global row identity is used by both Skin and Fat volume views.
# Its source-ring row enters the single shared global free set once.
i_faces={"D":np.empty((0,3),np.int64),"O":np.empty((0,3),np.int64),
         "I":np.asarray([[0,1,2]],np.int64)}
i_global={"D":np.asarray([],np.int64),"O":np.asarray([],np.int64),
          "I":np.asarray([700,701,702],np.int64)}
skin_i=np.asarray([700,701,702],np.int64);fat_i=np.asarray([700,701,702],np.int64)
assert np.array_equal(skin_i,fat_i) and skin_i[0]==700
shared,ireceipt=mod.expand_triangle_ring_once([700],i_faces,i_global)
assert shared.tolist()==[700,701,702]
assert ireceipt["role_incidence"]["I"]["incident_source_faces"]==1

print("PASS_V16_SINGLE_RING_FIXED_MASK_AND_SHARED_I_CONTROLS")
