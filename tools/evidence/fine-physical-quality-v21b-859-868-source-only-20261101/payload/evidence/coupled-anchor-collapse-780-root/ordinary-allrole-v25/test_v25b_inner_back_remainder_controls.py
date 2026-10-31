from pathlib import Path
import importlib.util, sys
import numpy as np

HERE=Path(__file__).resolve()
spec=importlib.util.spec_from_file_location("remainder_builder_controls",HERE.parent/"repair_coupled_quality_points_722_v25b_inner_back_remainder.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

assert m._typed_archive_scalar_text(np.asarray("abc"),"x","fixture")=="abc"
assert m._typed_archive_scalar_text(np.asarray(["abc"]),"x","fixture")=="abc"
for bad in (np.asarray(["a","b"]),np.asarray(1),np.asarray("")):
    try:m._typed_archive_scalar_text(bad,"x","fixture")
    except ValueError:pass
    else:raise AssertionError("invalid text scalar was accepted")

t=np.asarray([[0,3,6,9],[3,9,10,11],[4,7,12,13]],np.int64)
bad=np.asarray([0,2],np.int64)
ordinary=np.asarray([4],np.int64)
selected=m.select_bad_cells_without_ordinary_free_corner(t,bad,ordinary)
assert np.array_equal(selected,np.asarray([0],np.int64)),selected
maps={"O":np.asarray([0,1,2]),"I":np.asarray([3,4,5,6]),"D":np.asarray([6,7,8])}
faces={"O":np.asarray([[0,1,2]]),"I":np.asarray([[0,1,2],[0,2,3]]),"D":np.asarray([[0,1,2]])}
free,info=m.expand_inner_back_remainder_once(selected,faces,maps,t,
    actual_back_fat_rows=np.asarray([10,11,12]),
    fixed_global_rows=np.asarray([0,1,2,7,8]),
    witness_frozen_global_rows=np.asarray([5]))
fs=set(map(int,free))
assert {3,4,6,9,10,11}.issubset(fs),fs
assert not ({0,1,2,5,7,8,12}&fs),fs
assert info["selected_bad_cells"]==1
assert info["selected_interior_corner_rows"]==1
assert info["authenticated_actual_back_Fat_rows_in_selected_corner_star"]==2
assert info["O_source_rows_never_released"] is True
print("PASS targeted remainder tiny controls",{"selected_cells":selected.tolist(),"free_rows":sorted(fs),"info":info})
