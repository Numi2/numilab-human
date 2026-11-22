import ctypes, json, math, sys
from fractions import Fraction
from pathlib import Path
import numpy as np
lib=ctypes.CDLL(sys.argv[1]); f=ctypes.c_float
e=lib.checkEdge;e.argtypes=[f,f,f,f,ctypes.POINTER(f)]
rng=np.random.default_rng(20261122)
n=50000
high=rng.integers(0,2**32,(n,2),dtype=np.uint32).view(np.float32)
low=rng.integers(0,2**32,(n,2),dtype=np.uint32).view(np.float32)
low[:n//2]=0
high[n//3:2*n//3,1]=high[n//3:2*n//3,0]
low[n//3:2*n//3,1]=low[n//3:2*n//3,0]
proved_rows=[]
tested=proved=0
for (ah,bh),(al,bl) in zip(high,low):
    words=tuple(map(float,(ah,al,bh,bl)))
    result=f(); accepted=e(*words,ctypes.byref(result))
    if accepted<0:raise ValueError(('mismatch with shared arithmetic',words))
    if accepted:
        a,b,c,d=map(Fraction.from_float,words)
        if a+b-c-d!=Fraction.from_float(result.value):raise ValueError(('not exact',words))
        proved_rows.append([*words,result.value])
        proved+=1
    tested+=1
if len(sys.argv)==3: np.asarray(proved_rows,dtype='<f4').tofile(sys.argv[2])
assert lib.checkCell()==0 and 0<proved<tested
print(json.dumps(dict(cases=tested,proved=proved,legacy_fallback=tested-proved,seed=20261122,exact_rational_and_two_word_mismatches=0,rigid_cell_admitted=True,collapsed_and_inverted_rejected=True),indent=2))
