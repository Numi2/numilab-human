#!/usr/bin/env python3
"""Check boundary ancestry survives Boolean splitting and containment stays explicit."""
import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).parent
spec=importlib.util.spec_from_file_location("inventory448",p/"inventory_connected_cavity_witnesses_448.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
v=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
t=np.array([[0,1,2,3]],np.int64)
f,owners=m.boundary(t,np.array([91]))
assert len(f)==4 and np.array_equal(owners,np.full(4,91))
s,ident=m.tagged(v,f)
cut=s^m.mf.Manifold.cube([.6,.6,.6]).translate([.1,.1,.1])
assert cut.volume()>0
w=m.witness_faces(cut.to_mesh64(),ident,4)
assert np.array_equal(w,np.array([0])),w # only the tetrahedron's sloping face survives
contained=s^m.mf.Manifold.cube([.1,.1,.1]).translate([.1,.1,.1])
assert contained.volume()>0 and len(m.witness_faces(contained.to_mesh64(),ident,4))==0
# Merge shared-node stars: independent edits may not separately own one cell.
cells=np.array([[0,1,2,3],[3,4,5,6],[10,11,12,13]],np.int64)
assert m.components(np.array([0,4,10]),cells)==[[0,1],[2]]
print("PASS: split-face ancestry, explicit contained-solid gap, overlapping-star ownership")
