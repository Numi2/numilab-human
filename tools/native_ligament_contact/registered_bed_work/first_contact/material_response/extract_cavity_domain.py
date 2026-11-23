"""Extract the marked material side of a constrained tetrahedralization."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def faces(t):
    # Each local facet is opposite the tetrahedron vertex at the same index.
    return t[:, [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]].reshape(-1, 3)

def oriented_plc(path):
    with path.open() as f:
        n=int(f.readline().split()[0])
        for _ in range(n):f.readline()
        m=int(f.readline().split()[0]);boundary=[]
        for _ in range(m):
            if f.readline().split()!=['1','0']:raise ValueError('Non-triangular PLC')
            p=list(map(int,f.readline().split()))
            if len(p)!=4 or p[0]!=3:raise ValueError('Invalid facet')
            boundary.append(p[1:])
    return n,np.asarray(boundary,dtype=int)

def extract(P,T,oriented):
    if (T.ndim != 2 or T.shape[1] != 4 or not len(T)
            or oriented.ndim != 2 or oriented.shape[1] != 3 or not len(oriented)):
        raise ValueError('Empty or malformed cavity')
    oriented=np.asarray(oriented,dtype=T.dtype)
    if np.min(T)<0 or np.max(T)>=len(P) or np.min(oriented)<0 or np.max(oriented)>=len(P):
        raise ValueError('Invalid cavity point ID')
    if len(np.unique(np.sort(oriented,axis=1),axis=0))!=len(oriented):
        raise ValueError('Repeated constrained facet')
    sf=faces(T);key=np.sort(sf,axis=1)
    unique,first,inv,count=np.unique(key,axis=0,return_index=True,return_inverse=True,return_counts=True)
    if count.max()>2:raise ValueError('Nonmanifold tetrahedralization')
    def packed(a):return np.ascontiguousarray(a).view(np.dtype([('i',a.dtype),('j',a.dtype),('k',a.dtype)])).ravel()
    base=packed(unique);wanted=packed(np.sort(oriented,axis=1))
    where=np.searchsorted(base,wanted)
    matched=(where<len(base));matched[matched]&=base[where[matched]]==wanted[matched]
    if not np.all(matched):
        raise ValueError(f'{int((~matched).sum())} mandatory PLC facets missing in mesh')
    mark=np.zeros(len(unique),bool);mark[where]=True
    order=np.argsort(inv,kind='stable');offset=np.r_[0,count.cumsum()]
    connect=np.flatnonzero((count==2)&~mark)
    a=order[offset[connect]]//4;b=order[offset[connect]+1]//4
    graph=sparse.coo_matrix((np.ones(len(a),bool),(a,b)),shape=(len(T),len(T))).tocsr()
    nc,g=connected_components(graph,directed=False);tag=np.zeros(nc,dtype='i4')
    for face,position in zip(oriented,where):
        for row in order[offset[position]:offset[position+1]]:
            tet=row//4;opp=T[tet,row%4]
            u,v,w=P[face].astype(np.longdouble);side=np.dot(P[opp].astype(np.longdouble)-u,np.cross(v-u,w-u))
            if not np.isfinite(side) or not side:raise ValueError('Indeterminate constrained facet side')
            value=1 if side<0 else -1
            if tag[g[tet]] not in (0,value):raise ValueError('Conflicting material side in the same cavity component')
            tag[g[tet]]=value
    if not np.all(tag):raise ValueError('Unclassified constrained mesh component')
    kept=T[tag[g]>0]
    ff,cc=np.unique(np.sort(faces(kept),axis=1),axis=0,return_counts=True)
    if not np.array_equal(ff[cc==1],np.unique(np.sort(oriented,axis=1),axis=0)):
        raise ValueError('Material extraction does not preserve exactly the required facets')
    return kept
