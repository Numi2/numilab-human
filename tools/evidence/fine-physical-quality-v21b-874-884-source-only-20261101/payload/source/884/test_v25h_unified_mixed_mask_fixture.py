#!/usr/bin/env python3
"""Tiny control: mixed ordinary + inner-back bad cell remains in one patch."""
from pathlib import Path
import importlib.util, json, sys
import numpy as np

BUILDER=Path(__file__).with_name("repair_coupled_quality_points_722_v25h_unified_inner_back_ordinary.py")
spec=importlib.util.spec_from_file_location("v25h_fixture_builder",BUILDER)
if spec is None or spec.loader is None: raise RuntimeError("cannot import v25h builder")
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)
# One tetra has an ordinary movable corner (0) and a locked I-back corner (1).
# O scope row 6 is also in the ordinary ring and must stay excluded.
t=np.asarray([[0,1,2,3]],np.int64)
faces={"O":np.asarray([[0,1,2]],np.int64),"I":np.asarray([[0,1,2]],np.int64),"D":np.asarray([[0,1,2]],np.int64)}
maps={"O":np.asarray([0,6,7],np.int64),"I":np.asarray([1,4,5],np.int64),"D":np.asarray([2,8,9],np.int64)}
ordinary,inner,union,oi,ii=M.build_unified_free_sets(
    np.asarray([0],np.int64),faces,maps,t,np.asarray([1],np.int64),
    ordinary_fixed_rows=np.asarray([1,6],np.int64),
    inner_fixed_rows=np.asarray([6],np.int64),witness_frozen_rows=np.empty(0,np.int64))
assert 0 in ordinary, "ordinary corner 0 should remain movable"
assert 1 not in ordinary, "I-back row 1 must remain locked in the ordinary-only mask"
assert 1 in inner and 1 in union, "I-back row 1 must be released by the unified inner mask"
assert 6 not in ordinary and 6 not in inner and 6 not in union, "outer O scope row 6 must remain fixed"
patches,roster,_,_=M._build_unified_block_local_patches(
    np.asarray([0],np.int64),ordinary,inner,faces,maps,t,np.asarray([1],np.int64),
    fixed_global_rows=np.asarray([6],np.int64),witness_frozen_global_rows=np.empty(0,np.int64),point_count=10)
assert len(patches)==1 and np.array_equal(patches[0]["seed_bad_cells"],np.asarray([0],np.int64))
assert 0 in patches[0]["free_rows"] and 1 in patches[0]["free_rows"]
assert 6 not in patches[0]["free_rows"]
assert np.array_equal(patches[0]["affected_cells"],np.asarray([0],np.int64))
assert roster["bad_cells_with_at_least_one_free_corner"]==1
assert roster["bad_cells_without_a_free_corner"]==0
assert roster["all_free_corner_bad_cells_covered_by_full_star"] is True
print(json.dumps({"status":"PASS_TINY_MIXED_MASK_FIXTURE","ordinary_free":ordinary.tolist(),
    "inner_free":inner.tolist(),"union_free":union.tolist(),"patch_count":len(patches),
    "bad_cells_covered":roster["bad_cells_with_at_least_one_free_corner"],
    "O_scope_6_fixed":6 not in union,"I_back_1_released_by_union":1 in union},sort_keys=True))
