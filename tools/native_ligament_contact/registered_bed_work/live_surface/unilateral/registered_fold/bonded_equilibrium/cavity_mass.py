"""Evaluate conservative replacement masses on the actual installed tetrahedra."""
import numpy as np

def cavity_delta(source,old_mass,retained,replacement,replacement_mass,num_nodes):
    retired=np.setdiff1d(np.arange(len(source)),retained)
    dm=np.zeros(num_nodes)
    for k in range(4):
        np.add.at(dm,source[retired,k],-old_mass[retired]/4)
        np.add.at(dm,replacement[:,k],replacement_mass/4)
    active=np.unique(np.r_[source[retired].ravel(),replacement.ravel()])
    total=float(sum(replacement_mass))
    if abs(dm.sum())>1e-10*total:raise ValueError('Different total material mass')
    return dm,active,total
