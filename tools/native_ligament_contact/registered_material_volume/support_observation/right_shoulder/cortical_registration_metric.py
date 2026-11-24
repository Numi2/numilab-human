"""Allow an inferred muscle footprint inside cortex to move only toward the measured cortical exit."""
import numpy as np
from scipy.sparse import coo_matrix,kron,eye
from scipy.sparse.linalg import splu
class DirectedMaterialMetric:
    def __init__(self,K,free,normals):
        self.n=len(free);rows=[];cols=[];data=[];col=0
        for k,node in enumerate(free):
            basis=normals[node][:,None] if node in normals else np.eye(3)
            for a in range(basis.shape[1]):
                for axis in range(3):
                    if basis[axis,a]:
                        rows.append(3*k+axis);cols.append(col);data.append(basis[axis,a])
                col+=1
        Z=coo_matrix((data,(rows,cols)),shape=(3*self.n,col)).tocsc()
        self.factor=splu((Z.T@kron(K[free][:,free],eye(3),format='csc')@Z).tocsc())
        self.Z=Z
    def solve(self,rhs):
        n,k=rhs.shape
        if n!=self.n or k%3:raise ValueError('Invalid vector-valued correction')
        b=rhs.reshape(n,-1,3).transpose(0,2,1).reshape(n*3,-1)
        u=self.Z@self.factor.solve(self.Z.T@b)
        return u.reshape(n,3,-1).transpose(0,2,1).reshape(n,k)
