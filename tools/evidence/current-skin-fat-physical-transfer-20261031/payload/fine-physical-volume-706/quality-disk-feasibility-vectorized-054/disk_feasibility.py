"""Batch the existing quality-only polygon feasibility precheck."""
from functools import lru_cache
from itertools import combinations
import numpy as np

@lru_cache(maxsize=32)
def triangle_slots(n):
    if n<3 or n>64:raise ValueError('polygon loop size outside bounded domain')
    keys=tuple(combinations(range(n),3))
    rows=np.asarray(keys,np.int64)
    rows.setflags(write=False)
    return keys,rows

def feasible(points,loop,minimum_altitude_m,minimum_angle_degrees,quality_function):
    keys,slots=triangle_slots(len(loop))
    triangles=np.asarray(loop,np.int64)[slots]
    altitude,angle=quality_function(points,triangles)
    eligible=(altitude>=minimum_altitude_m)&(angle>=minimum_angle_degrees)
    q=dict(zip(keys,map(bool,eligible)))
    n=len(loop);d=[[False]*n for _ in range(n)]
    for i in range(n-1):d[i][i+1]=True
    for span in range(2,n):
        for i in range(n-span):
            j=i+span
            d[i][j]=any(q[i,k,j] and d[i][k] and d[k][j] for k in range(i+1,j))
    return d[0][n-1]
