#!/usr/bin/env python3
"""Tiny geometry fixture for local-Fat to shared-global back-row mapping."""
from pathlib import Path
import importlib.util
import numpy as np

HERE=Path(__file__).resolve().parent
VERIFIER=HERE/'verify_ordinary_allrole_v25h2_global_back_domain_root.py'
spec=importlib.util.spec_from_file_location('v25h2_fixture_subject',VERIFIER)
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
P=V.M
# One bad tet and a good neighbor; quality metrics are deterministic fixture values.
points=np.zeros((13,3),np.float32)
points[:9]=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[2,0,0],[0,-1,0]],np.float32)
tets=np.asarray([[0,1,2,3],[0,1,3,8]],np.int64)
faces={'O':np.asarray([[0,1,2]],np.int64),'I':np.asarray([[0,1,2]],np.int64),'D':np.asarray([[0,1,2]],np.int64)}
maps={'O':np.asarray([0,1,2],np.int64),'I':np.asarray([3,4,5],np.int64),'D':np.asarray([3,6,7],np.int64)}
surfaces={r:{'faces':faces[r]} for r in ('O','I','D')}
# Nonidentity Fat row map: local row 1 is global point 8, the extra full-star point.
fat_to_global=np.asarray([11,8,12,10],np.int64)
fat_back_local=np.asarray([1],np.int64);fat_back_global=fat_to_global[fat_back_local]
ordinary_fixed=np.asarray([0,1,2,3],np.int64);inner_fixed=np.asarray([0],np.int64);frozen=np.empty(0,np.int64)
assert fat_back_local.tolist()==[1] and fat_back_global.tolist()==[8]
class FixedMetrics:
 @staticmethod
 def measurements(_points,_tets):return np.asarray([1.0,1.0]),np.asarray([1.0e-6,1.0e-4])
 @staticmethod
 def passes(det,q):return (np.asarray(det)>0)&(np.asarray(q)>=1.0e-5)
old=V.G.M.M;V.G.M.M=FixedMetrics
try:
 rec_o,rec_i,rec_u,bad,good,patches,components=V.reconstruct_masks(points,tets,surfaces,maps,fat_back_global,ordinary_fixed,inner_fixed,frozen)
 prod_o,prod_i,prod_u,_,_=P.build_unified_free_sets(np.asarray([0],np.int64),faces,maps,tets,fat_back_global,ordinary_fixed,inner_fixed,frozen)
 wrong_o,wrong_i,wrong_u,_,_,_,_=V.reconstruct_masks(points,tets,surfaces,maps,fat_back_local,ordinary_fixed,inner_fixed,frozen)
finally:V.G.M.M=old
assert bad.tolist()==[0] and np.flatnonzero(good).tolist()==[1]
assert sorted(rec_o)==np.asarray(prod_o,np.int64).tolist() and sorted(rec_i)==np.asarray(prod_i,np.int64).tolist() and sorted(rec_u)==np.asarray(prod_u,np.int64).tolist()
assert sorted(rec_o)==[]
assert sorted(rec_i)==[3,4,5,6,7,8]
assert sorted(rec_u)==[3,4,5,6,7,8]
assert not np.array_equal(wrong_i,rec_i) and 8 not in set(map(int,wrong_i))
assert len(patches)==1 and patches[0]['seed_bad_cells'].tolist()==[0]
assert patches[0]['affected_cells'].tolist()==[0,1]
assert len(components)==1 and components[0]['status']=='PATCHED'
print({'status':'PASS_V25H2_GLOBAL_BACK_ROW_AND_MASK_EQUIVALENCE','fat_local_to_global':fat_to_global.tolist(),
 'actual_back_local_rows':fat_back_local.tolist(),'actual_back_global_rows':fat_back_global.tolist(),
 'ordinary_count':len(rec_o),'inner_count':len(rec_i),'union_count':len(rec_u),
 'ordinary_rows':sorted(rec_o),'inner_rows':sorted(rec_i),'union_rows':sorted(rec_u),
 'wrong_local_namespace_would_miss_global_back_row':8 not in set(map(int,wrong_i)),
 'bad_cell_full_star':patches[0]['affected_cells'].tolist()})
