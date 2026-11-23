"""Construct a continuous extension of articulated motion on the registered material tetrahedra.

The gradient term fits the local rotation of each tetrahedron. A volume-weighted
screening term fits the recorded anatomical carrier displacement. Both terms
define a reduction basis; they are not additional forces in the scene.
"""
import ctypes
from pathlib import Path
import numpy as np
from scipy import sparse

class Factor:
    def __init__(self, stiffness, library):
        self.lib=ctypes.CDLL(str(library.resolve()))
        ptr=np.ctypeslib.ndpointer
        integer=ctypes.c_int
        self.lib.make_factor.argtypes=[integer,ptr('int64',flags='C_CONTIGUOUS'),ptr('int32',flags='C_CONTIGUOUS'),ptr('float64',flags='C_CONTIGUOUS')]
        self.lib.make_factor.restype=ctypes.c_void_p
        self.lib.solve_factor.argtypes=[ctypes.c_void_p,integer,integer,ptr('float64',flags='F_CONTIGUOUS')]
        self.lib.solve_factor.restype=None
        self.lib.free_factor.argtypes=[ctypes.c_void_p]
        self.lib.free_factor.restype=None
        diagonal=stiffness.diagonal()
        if not np.isfinite(diagonal).all() or diagonal.min()<=0: raise ValueError('Invalid material stiffness')
        self.scale=diagonal**-.5
        scaling=sparse.diags(self.scale)
        lower=sparse.tril(scaling@stiffness@scaling,format='csc')
        lower.sort_indices()
        self.storage=(lower.indptr.astype('i8'),lower.indices.astype('i4'),lower.data)
        self.handle=self.lib.make_factor(len(diagonal),*self.storage)
        if not self.handle: raise ValueError('Material extension is not positive definite')

    def solve(self,rhs):
        work=np.asfortranarray(self.scale[:,None]*rhs)
        self.lib.solve_factor(self.handle,*work.shape,work)
        return self.scale[:,None]*work

    def close(self):
        if self.handle:
            self.lib.free_factor(self.handle)
            self.handle=None

def assemble(x,t,material,shear,node_donors,body_count,element_scale=None):
    n=len(x);stiff=sparse.csr_matrix((n,n))
    gradient_map=sparse.csr_matrix((n,body_count*4))
    measure=np.zeros(n)
    for first in range(0,len(t),40000):
        cells=t[first:first+40000]
        edges=x[cells[:,1:]]-x[cells[:,:1]]
        volume=np.linalg.det(edges)/6
        if not np.all(np.isfinite(volume)&(volume>0)):raise ValueError('Invalid material volume')
        weight=volume*shear[material[first:first+len(cells)]]
        if element_scale is not None:
            weight*=element_scale[first:first+len(cells)]
        g=np.linalg.inv(edges).transpose(0,2,1)
        g=np.concatenate((-g.sum(axis=1,keepdims=True),g),axis=1)
        nodes=np.broadcast_to(cells[:,:,None],(len(cells),4,4))
        other=np.broadcast_to(cells[:,None,:],nodes.shape)
        kernel=weight[:,None,None]*(g@g.transpose(0,2,1))
        stiff+=sparse.coo_matrix((kernel.ravel(),(nodes.ravel(),other.ravel())),shape=stiff.shape).tocsr()
        measure+=np.bincount(cells.ravel(),weights=np.repeat(weight/4,4),minlength=n)
        for k in range(3):
            coefficient=np.broadcast_to((g[:,:,k]*weight[:,None]/4)[:,:,None],nodes.shape)
            column=4*node_donors[other]+k
            gradient_map+=sparse.coo_matrix((coefficient.ravel(),(nodes.ravel(),column.ravel())),shape=gradient_map.shape).tocsr()
    return stiff,gradient_map,measure

def sample_map(x,donor,origin,body_count):
    n=len(x);lever=np.column_stack((x-origin[donor],np.ones(n)))
    return sparse.csr_matrix((lever.ravel(),(4*donor[:,None]+np.arange(4)).astype('i4').ravel(),
                              np.arange(0,4*n+1,4,dtype='i4')),shape=(n,4*body_count))

class Extension:
    def __init__(self,stiff,gradient,measure,raw,anchor_rows,anchor_map,length,library):
        if not np.isfinite(length) or length<=0: raise ValueError('Invalid extension length')
        self.fixed=anchor_rows;self.anchor_map=anchor_map
        self.free=np.setdiff1d(np.arange(len(measure)),anchor_rows)
        screen=measure/length**2
        matrix=stiff+sparse.diags(screen)
        self.matrix=matrix[self.free][:,self.free]
        source=gradient+sparse.diags(screen)@raw
        self.rhs=(source[self.free]-matrix[self.free][:,anchor_rows]@anchor_map).tocsr()
        self.factor=Factor(self.matrix,library)

    def forward(self,coeff):
        x=np.empty((len(self.free)+len(self.fixed),coeff.shape[1]))
        rhs=self.rhs@coeff
        x[self.free]=self.factor.solve(rhs)
        x[self.fixed]=self.anchor_map@coeff
        residual=self.matrix@x[self.free]-rhs
        return x,np.linalg.norm(residual,axis=0)/np.maximum(np.linalg.norm(rhs,axis=0),1e-30)

    def transpose(self,load):
        return self.rhs.T@self.factor.solve(load[self.free])+self.anchor_map.T@load[self.fixed]
