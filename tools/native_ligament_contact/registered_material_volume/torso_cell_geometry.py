"""Signed material volume and normalized tetrahedral Jacobian at emitted positions."""
import numpy as np

def geom(x,t):
    edge=x[t[:,1:]]-x[t[:,:1]]
    v=np.einsum('ij,ij->i',edge[:,0],np.cross(edge[:,1],edge[:,2]))/6.
    lengths=np.linalg.norm(x[t[:,[0,0,0,1,1,2]]]-x[t[:,[1,2,3,2,3,3]]],axis=2)
    return v,6*v/lengths.max(1)**3
