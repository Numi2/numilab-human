"""Check every material boundary pair touched by the absolute-volume correction."""
import ctypes
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent
old=root.parent/"resting-clearance-and-atrium-016/repaired-tetgen/mesh.node"
new=root/"absolute-volume-repaired-tetgen/mesh.node"
facet=root/"paired-repaired-registration/facet-ancestry.csv"
predicate=root.parent/"connected-fold-repair-62/exact-pair.dylib"
compare=ctypes.CDLL(str(predicate)).beyond
three=np.ctypeslib.ndpointer(dtype=np.uint32,ndim=1,flags="C_CONTIGUOUS")
compare.argtypes=[np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags="C_CONTIGUOUS"),three,three]
compare.restype=ctypes.c_uint
before,after=(np.loadtxt(p,skiprows=1,usecols=(1,2,3)) for p in (old,new))
moved=np.any(before!=after,axis=1)
with facet.open() as f:
    nodes=("joint_node0","joint_node1","joint_node2")
    records=csv.DictReader(f)
    triangles=np.asarray([tuple(int(row[k])-1 for k in nodes) for row in records],dtype=np.int32)
assert len(triangles)==833108
affected=np.flatnonzero(np.any(moved[triangles],axis=1))
def measure(positions):
    unique,inv=np.unique(positions,axis=0,return_inverse=True)
    unique=np.ascontiguousarray(unique,dtype=np.float64)
    tri=np.ascontiguousarray(inv.ravel()[triangles],dtype=np.uint32)
    xyz=unique[tri]
    high,low=xyz.max(axis=1),xyz.min(axis=1)
    checked=0
    crossing=[]
    valid=True
    seen=set()
    for a in affected:
        e=xyz[a,1:]-xyz[a,:1]
        valid=valid and float(np.linalg.norm(np.cross(*e)))>0.
        nearby=np.flatnonzero(np.all((high[a]>=low)&(low[a]<=high),axis=1))
        for b in nearby:
            key=tuple(sorted((int(a),int(b))))
            if a==b or key in seen:
                continue
            seen.add(key)
            left,right=tri[a],tri[b]
            if sorted(left)==sorted(right):
                assert any(np.array_equal(np.roll(left[::-1],i),right) for i in range(3))
                continue
            checked+=1
            if compare(unique,left,right):
                crossing.append(list(key))
    return {"crossings":len(crossing),"crossed_faces":crossing,
            "exact_predicate_pairs":checked,"affected_faces_nondegenerate":valid}
baseline,candidate=(measure(x) for x in (before,after))
report={"baseline":baseline,"candidate":candidate,
    "moved_nodes":np.flatnonzero(moved).tolist(),"affected_boundary_faces":affected.tolist(),
    "boundary_faces":len(triangles),"scope":"All pairs incident to a changed material boundary face at the reference pose.",
    "predicate_sha256":hashlib.sha256(predicate.read_bytes()).hexdigest()}
(root/"absolute-volume-surface-delta.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
raise SystemExit(not candidate["affected_faces_nondegenerate"] or candidate["crossings"]!=0)
