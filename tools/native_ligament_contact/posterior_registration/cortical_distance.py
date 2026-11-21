"""Reusable exact-feature closest-point query for authoring, with welded cortical normals."""
import ctypes as c
from pathlib import Path
import numpy as np
from prepare_posterior_muscles import mesh_volume,closed
class CorticalSurface:
    def __init__(self,lib,x,f):
        self.library=c.CDLL(str(Path(lib).resolve()));a=self.library
        P=np.ctypeslib.ndpointer(dtype='float64',flags='C_CONTIGUOUS');I=np.ctypeslib.ndpointer(dtype='int32',flags='C_CONTIGUOUS')
        a.cortex_create.argtypes=[P,c.c_int,I,c.c_int];a.cortex_create.restype=c.c_void_p
        a.cortex_query.argtypes=[c.c_void_p,P,c.c_int,P,I];a.cortex_query.restype=None
        a.cortex_destroy.argtypes=[c.c_void_p];a.cortex_destroy.restype=None
        x,iv=np.unique(x,axis=0,return_inverse=True);f=iv[f]
        if any(closed(x,f).values()):raise ValueError('Cortical query requires closed, oriented material')
        if mesh_volume(x,f)<0:f=f[:,[0,2,1]]
        self.handle=a.cortex_create(np.ascontiguousarray(x,dtype='f8'),len(x),np.ascontiguousarray(f,dtype='i4'),len(f))
    def query(self,points):
        p=np.ascontiguousarray(points,dtype='f8');r=np.empty((len(p),7));i=np.empty(len(p),'i4')
        self.library.cortex_query(self.handle,p,len(p),r,i)
        return r[:,0],r[:,1:4],r[:,4:7],i
    def __del__(self):
        if getattr(self,'handle',None):self.library.cortex_destroy(self.handle)
