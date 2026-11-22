"""Differentiate initial material volume acceleration in a shared lumped-mass solid."""
import numpy as np


def cofactor_rows(edges):
    return np.stack((np.cross(edges[:,1],edges[:,2]),np.cross(edges[:,2],edges[:,0]),np.cross(edges[:,0],edges[:,1])),axis=1)


def volume_rate(position, acceleration, cells, free_cells, reference_det, mass, basis, force_gradient_jac=None):
    # At zero initial velocity J'' is grad(J) dot a. Uniform gravity cancels.
    edges=position[cells[:,1:]]-position[cells[:,:1]]
    cof=cofactor_rows(edges)/reference_det[:,None,None]
    dJ=np.concatenate((-cof.sum(1,keepdims=True),cof),axis=1)
    rates=np.einsum('mka,mka->m',dJ,acceleration[free_cells])
    if force_gradient_jac is None:
        return rates
    ae=acceleration[free_cells[:,1:]]-acceleration[free_cells[:,:1]]
    dc=np.stack([np.cross(ae[:,i],edges[:,j])+np.cross(edges[:,i],ae[:,j])
                 for i,j in ((1,2),(2,0),(0,1))],axis=1)/reference_det[:,None,None]
    hJa=np.concatenate((-dc.sum(1,keepdims=True),dc),axis=1)
    B=basis.reshape(len(mass),3,-1)
    # The physical internal force is the negative of the potential gradient.
    da=-(force_gradient_jac@basis).reshape(len(mass),3,-1)/mass[:,None,None]
    return np.einsum('mka,mkav->mv',hJa,B[free_cells])+np.einsum('mka,mkav->mv',dJ,da[free_cells])
