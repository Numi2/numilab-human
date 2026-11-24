from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def density(v):
    x=np.loadtxt(v/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8');t=np.loadtxt(v/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    lab=np.loadtxt(v/'current-child.labels.csv',delimiter=',',skiprows=1,dtype='i8');mass=np.fromfile(v/'cell-mass.f64','<f8')
    w=np.empty(len(t));cent=np.empty((len(t),3))
    for i in range(0,len(t),65536):
        pts=x[t[i:i+65536]];a,b,c=(pts[:,j]-pts[:,0] for j in (1,2,3))
        w[i:i+len(a)]=(a[:,0]*(b[:,1]*c[:,2]-b[:,2]*c[:,1])-a[:,1]*(b[:,0]*c[:,2]-b[:,2]*c[:,0])+a[:,2]*(b[:,0]*c[:,1]-b[:,1]*c[:,0]))/6
        cent[i:i+len(a)]=pts.mean(1)
    return x,t,lab,mass,w,cent

def run(root):
    v=root/'remaining-fold/collar-2/fold-volume';x,t,lab,m,w,c=density(v)
    bad=np.loadtxt(v/'density-outliers.txt',ndmin=2)[:,0].astype(int)
    for tag in np.unique(lab[bad,1:],axis=0):
        b=bad[(lab[bad,1:]==tag).all(1)];eligible=np.flatnonzero((lab[:,1:]==tag).all(1));dist=cKDTree(c[b]).query(c[eligible])[0]
        high=950 if tag[0]==16 else 1100
        print(tag,'bad',len(b),'rho range',min(m[b]/w[b]),max(m[b]/w[b]),'excess kg',np.maximum(0,m[b]-high*w[b]).sum())
        for r in (.001,.003,.01,.03,.2):
            a=eligible[dist<=r];print(round(r*1e3),len(a),'headroom kg',np.maximum(0,(high-1e-7)*w[a]-m[a]).sum())
if __name__=='__main__':run(Path(__file__).resolve().parent)
