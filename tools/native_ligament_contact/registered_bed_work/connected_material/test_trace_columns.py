"""Analytic connected and disconnected two-layer tetrahedral samples."""
import itertools
from pathlib import Path
import numpy as np
from prepare_connected_columns import native_functions

def main(surface_lib,volume_lib):
    lower,walk=native_functions(surface_lib,volume_lib)
    grid=np.array([(x,y,z) for z in range(3) for y in range(2) for x in range(2)],float)
    cells=[];material=[]
    index=lambda v:int(v[0]+2*v[1]+4*v[2])
    for layer in range(2):
        for axes in itertools.permutations(range(3)):
            point=np.array([0,0,layer]);vertices=[index(point)]
            for axis in axes:
                point[axis]+=1;vertices.append(index(point))
            tetra=grid[vertices]
            if np.linalg.det(tetra[1:]-tetra[0])<0:vertices[1],vertices[2]=vertices[2],vertices[1]
            cells.append(vertices);material.append(15+layer)
    cells=np.array(cells,"i4");mat=np.array(material,"i4");outer=np.array([[0,1,3],[0,3,2]],"i4")
    xy=np.array([[.345,.678],[.83,.22],[1.1,.3]])
    def solve(x,y,c):
        layers=(x[:,2]+1).astype("i4")
        first=np.empty(3,"i4");h=np.empty(3);beta=np.empty((3,3));info=np.empty((3,5),"i4");vals=np.empty((3,4))
        assert lower(y,len(y),outer,2,xy,3,first,h,beta)==0
        assert walk(x,y,len(y),c,mat,len(c),layers,outer,2,first,xy,h,3,info,vals)==0
        return info,vals
    info,z=solve(grid,grid,cells)
    assert list(info[:,0])==[0,0,1] and np.allclose(z[:2],[[0,1,2,1],[0,1,2,1]])
    deform=grid.copy();deform[4:8,2]=.7;deform[8:,2]=1.5
    info,z=solve(grid,deform,cells);assert (info[:2,0]==0).all() and np.allclose(z[:2,1:3],[.7,1.5])
    disconnected=np.vstack([grid,grid[4:8]]);other=cells.copy();replace=(mat[:,None]==16)&(cells<8)
    other[replace]+=8
    info,_=solve(disconnected,disconnected,other);assert list(info[:,0])==[5,5,1]
    inverted=grid.copy();inverted[4:8,2]=-.1
    info,_=solve(grid,inverted,cells);assert list(info[:,0])==[3,3,1]
    print("Two connected rays and an outside ray verified under reference, deformed, disconnected and inverted geometry.")

if __name__=="__main__":
    import sys
    main(*(Path(x).resolve() for x in sys.argv[1:]))
