"""Differentially test the BVH against the owner's unpruned exact pair predicate."""
import ctypes,itertools,sys
from pathlib import Path
import numpy as np
from compare_hand_interfaces import query

def run(exe,lib):
    arg=np.ctypeslib.ndpointer(flags='C_CONTIGUOUS')
    pred=ctypes.CDLL(str(lib)).batch
    pred.argtypes=[arg,arg,arg,ctypes.c_uint,arg];pred.restype=None
    rng=np.random.default_rng(6192026)
    for case in range(14):
        # Coplanar, neighboring, coincident, disjoint and near-coplanar cases.
        x=rng.normal(size=(32,3)).astype('f4').astype('f8')
        if case%3==0:x[:,2]=0
        if case%3==1:x[:,2]*=1e-6
        x[11:14]=x[:3]
        f=np.array([[0,1,2],[1,2,3],[11,12,13]]+[rng.choice(32,3,replace=False).tolist() for i in range(64)],dtype='u4')
        normal=np.cross(x[f[:,1]]-x[f[:,0]],x[f[:,2]]-x[f[:,0]])
        f=f[np.any(normal!=0,axis=1)]
        change=(np.zeros(len(f),bool) if case==0 else np.ones(len(f),bool) if case==1 else rng.random(len(f))<.23)
        pairs=np.array([(a,b) for a,b in itertools.combinations(range(len(f)),2) if change[a] or change[b]],dtype='u4').reshape(-1,2)
        hits=np.zeros(len(pairs),'u1');pred(x,f,pairs,len(pairs),hits)
        actual=query(exe,x,f,change)
        assert actual==set(map(tuple,pairs[hits.astype(bool)].tolist())),case
    print('14 deterministic differential fixtures passed')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
