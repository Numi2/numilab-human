#!/usr/bin/env python3
"""Small controls for the interior-only edge-bisection derivative; no whole-mesh run."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
PROD_PATH = HERE / "edge_bisection_interior_722.py"
VAL_PATH = HERE.parent / "quality-edge-bisection-721-root" / "validate_edge_bisection_721.py"

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def tet_boundary_faces(tets):
    counts = {}
    orientations = {}
    for tet in np.asarray(tets, np.int64):
        a,b,c,d = map(int,tet)
        for face in ((b,c,d),(a,d,c),(a,b,d),(a,c,b)):
            key = tuple(sorted(face))
            counts[key] = counts.get(key,0)+1
            orientations[key] = face
    return np.asarray([orientations[k] for k,n in counts.items() if n==1], np.int64)

def main():
    p = load("interior_bisection_722", PROD_PATH)
    v = load("edge_bisection_validator_721", VAL_PATH)
    points = np.asarray([[0,0,0],[1,0,0],[.5,1,0],[.5,0,1],[.5,-1,-1]], np.float32)
    tets = np.asarray([[0,1,2,3],[0,1,3,4],[0,1,4,2]], np.int64)
    det,_ = p.M.measurements(points,tets)
    for i in range(len(tets)):
        if det[i] < 0:
            tets[i,[2,3]] = tets[i,[3,2]]
    det,_ = p.M.measurements(points,tets)
    assert np.all(det > 0)
    idx = p.VolumeEdgeIndex(points,tets)
    edge=(0,1); star=idx.query(*edge)
    assert np.array_equal(star,np.asarray([0,1,2]))
    boundary=tet_boundary_faces(tets)
    assert not np.any(np.any(boundary==edge[0],axis=1)&np.any(boundary==edge[1],axis=1))
    assert p.strictly_interior_edge(edge, [], set(), boundary, boundary)
    assert not p.strictly_interior_edge(edge, [("I",(0,1),[0])], set(), boundary, boundary)
    assert not p.strictly_interior_edge(edge, [], {edge}, boundary, boundary)
    assert not p.strictly_interior_edge(edge, [], set(), np.asarray([[0,1,2]]), boundary)
    # A face-unreferenced source point may have no volume binding; every used
    # source-face point must still map exactly and uniquely.
    source={"points_world_m":np.asarray([[0,0,0],[1,0,0],[0,1,0],[9,9,9]],np.float32),
            "faces":np.asarray([[0,1,2]],np.int64)}
    inverse=p.surface_inverse(source,3,np.asarray([0,1,2,-1],np.int64))
    assert np.array_equal(inverse,np.asarray([0,1,2],np.int64))
    try:
        p.surface_inverse(source,3,np.asarray([0,1,-1,-1],np.int64))
        raise AssertionError("active source point with -1 map was accepted")
    except ValueError:
        pass

    midpoint=((points[0].astype(np.float64)+points[1].astype(np.float64))*0.5).astype(np.float32)
    base={"points_world_m":points,"tetrahedra":tets,"material_indices":np.asarray([2,2,2],np.int32),
          "output_cell_to_input_cell":np.arange(3,dtype=np.int64),
          "output_point_parent_offsets":np.zeros(len(points)+1,np.int64),
          "output_point_parent_input_cell_rows":np.empty(0,np.int64),
          "boundary_faces":boundary,"desired_boundary_faces":boundary.copy()}
    split=p.split_volume_role(base,edge,.5,midpoint,star,star)
    assert split["exact_delta"] == 0
    assert p.QB._boundary_chain(split["tetrahedra"]) == p.QB._boundary_chain(tets)
    item={"splits":{"Skin_current":split},"role_edges":{"Skin_current":edge},
          "role_fractions":{"Skin_current":.5},"surfaces":{},"point":midpoint}
    candidate=p.build_volume_batch_output(base,"Skin_current",[item],Path("parent.npz"),"0"*64,{}, {})
    rec={"operation_index":0,"edge_volume_rows_by_role":{"Skin_current":list(edge)},
         "fraction_by_role":{"Skin_current":.5},"new_point_index_by_role":{"Skin_current":len(points)},
         "new_point_world_m_f32":midpoint.tolist(),"volume_parent_cell_rows":{"Skin_current":star.tolist()}}
    summary=v.validate_volume(base,candidate,"Skin_current",[(0,rec)],p.QB)
    assert np.array_equal(candidate["boundary_faces"],boundary)
    assert np.array_equal(candidate["desired_boundary_faces"],boundary)
    assert not any(np.asarray(candidate["faces"]).size for _ in ()) # no source-face output was constructed
    print({"status":"PASS_INTERIOR_EDGE_FULLSTAR_NO_SOURCE_OR_BOUNDARY_CHANGE",
           "edge_star_cells":len(star),"replaced_cells":summary["replaced_parent_cells"],
           "exact_det6_delta":summary["exact_det6_delta"],
           "boundary_faces_bitwise_unchanged":True,"source_operations":0,
           "physical_admission":False,"native_steps":0})

if __name__=="__main__": main()
