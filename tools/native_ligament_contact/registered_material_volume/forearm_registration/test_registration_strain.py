import numpy as np
from certify_registration_strain import compare
X=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[0,0,1]])
T=np.array([[0,1,2,3]])
ids=np.array([31])
rot=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
if compare(X,X,X@rot+.75,T,ids)['failed']:raise AssertionError('Rigid motion')
shear=X.copy();shear[:,0]+=4*shear[:,1]
r=compare(X,X,shear,T,ids)
if r['failed']!=[31] or r['min_J']!=1:raise AssertionError('Isochoric shear escaped')
if not compare(X,X,-X,T,ids)['failed']:raise AssertionError('Inversion escaped')
print('rigid motion, determinant-one shear, and inversion checks passed')
