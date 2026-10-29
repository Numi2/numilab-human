#!/usr/bin/env python3
import numpy as np
from refine_conforming_tetra_edges_530 import refine_edges, signed_volumes

def verify(r, x, t):
    assert np.array_equal(r.points_reference_f64[:len(x)],x)
    for i in range(len(r.points_reference_f64)):
        a,b=r.node_source_offsets[i:i+2]
        assert r.node_source_weights[a:b].sum()==1
        assert np.allclose(r.node_source_weights[a:b]@x[r.node_source_rows[a:b]],
                           r.points_reference_f64[i],rtol=0,atol=1e-15)
    for i,parent in enumerate(r.child_source_tet_rows):
        for node in r.tetrahedra[i]:
            a,b=r.node_source_offsets[node:node+2]
            assert set(r.node_source_rows[a:b])<=set(t[parent])
    old=signed_volumes(x,t);new=signed_volumes(r.points_reference_f64,r.tetrahedra)
    assert np.allclose(new,old[r.child_source_tet_rows]*r.child_source_measure_fractions,rtol=1e-13,atol=1e-16)
    assert np.allclose(np.bincount(r.child_source_tet_rows,weights=new,minlength=len(t)),old,rtol=1e-13,atol=1e-16)

def run():
    x=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.],[0,0,-1.]])
    t=np.array([[0,1,2,3],[0,2,1,4]],dtype=np.int64)
    r=refine_edges(x,t,np.array([[0,1],[1,2],[0,2],[1,0]],dtype=np.int64))
    verify(r,x,t)
    assert len(r.inserted_node_rows)==3
    # Both sides of the shared material interface have identical subdivision.
    interface=[]
    for material in [0,1]:
        faces=[]
        for tet in r.tetrahedra[r.child_source_tet_rows==material]:
            for tri in [tet[[1,2,3]],tet[[0,3,2]],tet[[0,1,3]],tet[[0,2,1]]]:
                if np.all(r.points_reference_f64[tri,2]==0):faces.append(tuple(sorted(map(int,tri))))
        interface.append(sorted(faces))
    assert interface[0]==interface[1] and len(interface[0])==4
    unchanged=refine_edges(x,t,np.empty((0,2),dtype=np.int64));verify(unchanged,x,t)
    assert np.array_equal(unchanged.tetrahedra,t)
    for bad in [np.array([[0,0]]),np.array([[0,99]]),np.array([[3,4]]),np.array([[0.,1.]])]:
        try:refine_edges(x,t,bad)
        except ValueError:pass
        else:raise AssertionError("invalid edge accepted")
    try:refine_edges(x,t[:,[1,0,2,3]],np.array([[0,1]]))
    except ValueError:pass
    else:raise AssertionError("inverted source accepted")
    # Later marked edges may use deterministic inserted IDs.
    nested=refine_edges(x,t,np.array([[0,1],[0,5]],dtype=np.int64));verify(nested,x,t)
    assert np.array_equal(nested.points_reference_f64[-1],np.array([.25,0,0]))
    return {"status":"PASS","shared_interface_triangles":4,"original_nodes_unchanged":True,
            "source_cell_measure_closure":True,"source_node_barycentric_reconstruction":True,
            "invalid_domain_negative_tests":True,"nested_refinement":True}
if __name__=="__main__":
    import json
    print(json.dumps(run()))
