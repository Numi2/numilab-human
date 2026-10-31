from pathlib import Path
import importlib.util, sys
import numpy as np

SRC=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/coupled-anchor-collapse-780-root/ordinary-allrole-v25g-blocklocal/repair_coupled_quality_points_722_v25g_ordinary_blocklocal.py")
spec=importlib.util.spec_from_file_location("ordinary_block_local_v25g_fixture",SRC)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

# Bad-face components sharing one free row remain distinct sequential patches.
# This prevents healthy one-ring overlap from creating a single global torso LP.
t=np.asarray([[0,1,2,3],[0,4,5,6]],np.int64)
faces={r:np.empty((0,3),np.int64) for r in ("O","I","D")}
maps={r:np.arange(7,dtype=np.int64) for r in ("O","I","D")}
patches,roster,offsets,by_node=m._build_ordinary_block_local_patches(
    np.asarray([0,1]),np.asarray([0,1]),np.asarray([0]),faces,maps,t,7)
assert len(patches)==2
assert [x["free_rows"].tolist() for x in patches]==[[0],[0]]
assert [x["affected_cells"].tolist() for x in patches]==[[0,1],[0,1]]
assert roster["overlapping_free_rows_allowed_and_sequentially_revalidated"] is True
assert roster["all_selected_bad_cells_covered"] is True

# Role source rings remain within the ordinary free set and use complete stars.
t2=np.asarray([[0,1,2,3]],np.int64)
faces2={"O":np.asarray([[0,1,2]],np.int64),
        "I":np.asarray([[0,1,3]],np.int64),
        "D":np.asarray([[1,2,3]],np.int64)}
maps2={r:np.arange(4,dtype=np.int64) for r in ("O","I","D")}
p2,r2,_,_=m._build_ordinary_block_local_patches(
    np.asarray([0]),np.asarray([0]),np.arange(4,dtype=np.int64),faces2,maps2,t2,4)
assert len(p2)==1 and p2[0]["free_rows"].tolist()==[0,1,2,3]
assert p2[0]["affected_cells"].tolist()==[0]
assert set(r2["patches"][0]["ring_by_role"])=={"O","I","D"}

# Ordinary mode supports the established 0.5mm cap; the inner-back mode
# retains its 50um cap. Empty patches test dispatch bounds without optimization.
p=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
t=np.asarray([[0,1,2,3]],np.int64)
d,q=m.M.measurements(p,t);passed=np.asarray(m.M.passes(d,q),bool)
off,by=m._point_cell_incidence(t,len(p))
candidate,diag=m.propose_remainder_block_local(p,t,np.asarray([16]),np.asarray([0]),
    d,q,passed,[],5.e-4,.1,.1,off,by,ordinary_mode=True)
assert np.array_equal(candidate.view(np.uint32),p.view(np.uint32))
assert diag["status"]=="NO_LEXICOGRAPHIC_PROGRESS"
raised=False
try:
    m.propose_remainder_block_local(p,t,np.asarray([16]),np.asarray([0]),
        d,q,passed,[],5.e-4,.1,.1,off,by,ordinary_mode=False)
except ValueError:
    raised=True
assert raised
print("PASS separate bad-face patches with sequential overlap, role rings/full stars, and ordinary-vs-inner cap controls")
