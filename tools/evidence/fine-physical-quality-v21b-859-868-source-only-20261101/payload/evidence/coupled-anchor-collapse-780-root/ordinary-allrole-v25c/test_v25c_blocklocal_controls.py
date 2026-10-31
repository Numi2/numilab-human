from pathlib import Path
import importlib.util, sys
import numpy as np

HERE=Path(__file__).resolve()
spec=importlib.util.spec_from_file_location("v25c_blocklocal_controls",HERE.parent/"repair_coupled_quality_points_722_v25c_inner_back_remainder_blocklocal.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

t=np.asarray([[0,3,6,9],[3,9,10,11],[4,7,12,13]],np.int64)
components=m._bad_face_components(t,np.asarray([0,1,2]))
assert [x.tolist() for x in components]==[[0],[1],[2]],components
face_join=np.asarray([[0,1,2,3],[0,1,2,4],[0,1,5,6]],np.int64)
components=m._bad_face_components(face_join,np.asarray([0,1,2]))
assert [x.tolist() for x in components]==[[0,1],[2]],components

offsets,by_node=m._point_cell_incidence(t,14)
assert m._incident_cells_by_rows([3],offsets,by_node).tolist()==[0,1]
maps={"O":np.asarray([0,1,2]),"I":np.asarray([3,4,5,6]),"D":np.asarray([6,7,8])}
faces={"O":np.asarray([[0,1,2]]),"I":np.asarray([[0,1,2],[0,2,3]]),"D":np.asarray([[0,1,2]])}
patches,receipt,offsets,by_node=m._build_remainder_patches(
    np.asarray([0,2]),np.asarray([0]),faces,maps,t,np.asarray([10,11,12]),
    np.asarray([0,1,2,7,8]),np.asarray([5]),14)
assert len(patches)==1 and patches[0]["seed_bad_cells"].tolist()==[0]
assert patches[0]["free_rows"].tolist()==[3,4,6,9,10,11]
assert patches[0]["affected_cells"].tolist()==[0,1,2]
assert receipt["patch_free_union_rows"]==6
assert receipt["point_cell_incidence_built_once"] is True
# A synthetic one-node sliver confirms the local fallback can move one point
# through a native quality threshold while keeping its positive volume gate.
p0=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,2.e-5]],np.float32)
tt=np.asarray([[0,1,2,3]],np.int64)
det,q=m.Q._M.measurements(p0,tt);good=np.asarray(m.Q._M.passes(det,q),bool)
candidate,diag=m._single_node_quality_candidate(p0,tt,3,np.asarray([0]),5.e-5,
    good,good,None,__import__("time").monotonic()+2.)
assert candidate is not None,diag
newdet,newq=m.Q._M.measurements(candidate,tt)
assert np.all(newdet>0) and np.all(m.Q._M.passes(newdet,newq)),diag
assert np.array_equal(candidate[[0,1,2]].view(np.uint32),p0[[0,1,2]].view(np.uint32))
assert float(np.abs(candidate[3].astype(np.float64)-p0[3].astype(np.float64)).sum())<=5.e-5
print("PASS single-node fallback synthetic threshold",diag)
print("PASS block-local roster controls",{"components":[x.tolist() for x in components],
    "seed_cells":patches[0]["seed_bad_cells"].tolist(),"free_rows":patches[0]["free_rows"].tolist(),
    "affected_cells":patches[0]["affected_cells"].tolist(),"receipt":receipt})
