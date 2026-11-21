"""World-space triangle queries for the registered digital tendon corridor."""
import numpy as np

def closest(p,tri):
    # Project to all three clamped segments, plus the orthogonal projection in the facet.
    a,b,c=tri.transpose(1,0,2)
    e=b-a;g=c-a;rel=p-a
    ee=np.einsum('ij,ij->i',e,e);gg=np.einsum('ij,ij->i',g,g);eg=np.einsum('ij,ij->i',e,g)
    de=np.einsum('ij,ij->i',rel,e);dg=np.einsum('ij,ij->i',rel,g)
    den=ee*gg-eg*eg
    u=np.divide(de*gg-dg*eg,den,out=np.zeros_like(den),where=den>0)
    v=np.divide(dg*ee-de*eg,den,out=np.zeros_like(den),where=den>0)
    bary=np.column_stack((1-u-v,u,v))
    pts=np.einsum('ij,ijk->ik',bary,tri);sq=np.einsum('ij,ij->i',pts-p,pts-p)
    sq[(bary.min(1)<0)|(den<=0)]=np.inf
    for i,j in [(0,1),(1,2),(2,0)]:
        d=tri[:,j]-tri[:,i];l=np.einsum('ij,ij->i',d,d)
        t=np.clip(np.divide(np.einsum('ij,ij->i',p-tri[:,i],d),l,out=np.zeros_like(l),where=l>0),0,1)
        h=tri[:,i]+t[:,None]*d;s=np.einsum('ij,ij->i',h-p,h-p)
        take=s<sq;sq[take]=s[take];pts[take]=h[take];bary[take]=0;bary[take,i]=1-t[take];bary[take,j]=t[take]
    i=int(np.argmin(sq));return i,pts[i],bary[i],float(np.sqrt(sq[i]))

class Surface:
    def __init__(self,v,f):
        self.v=v;self.f=f;self.tri=v[f].astype(float);self.lo=self.tri.min(1);self.hi=self.tri.max(1)
    def nearby(self,p,r):
        return np.flatnonzero(np.all((self.lo<=p+r)&(self.hi>=p-r),axis=1))
    def nearest(self,p):
        for r in (.001,.003,.008,.02,.05,.2,2.):
            ids=self.nearby(p,r)
            if len(ids):
                i,q,b,d=closest(p,self.tri[ids]);
                if d<=r:return int(ids[i]),q,b,d
        raise ValueError('No nearest face')
    def ray(self,p,n,lim):
        end=p+lim*n;ids=np.flatnonzero(np.all((self.lo<=np.maximum(p,end)+1e-12)&(self.hi>=np.minimum(p,end)-1e-12),axis=1))
        if not len(ids):return []
        a,b,c=self.tri[ids].transpose(1,0,2);e=b-a;g=c-a
        h=np.cross(n,g);den=np.einsum('ij,ij->i',e,h);inv=np.divide(1,den,out=np.zeros_like(den),where=abs(den)>1e-18)
        offset=p-a;u=np.einsum('ij,ij->i',offset,h)*inv;k=np.cross(offset,e);v=(k@n)*inv;t=np.einsum('ij,ij->i',g,k)*inv
        ok=(abs(den)>1e-18)&(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)&(t>=0)&(t<=lim)
        hits=sorted((float(x),int(y)) for x,y in zip(t[ok],ids[ok]));out=[]
        for hit in hits:
            if not out or hit[0]-out[-1][0]>1e-9:out.append(hit)
        return out
    def inside(self,p):
        counts=[len(self.ray(p,n/np.linalg.norm(n),3))%2 for n in (np.array([.019,.12,1]),np.array([-.017,.15,1]),np.array([.07,.25,1]))]
        if len(set(counts))>1:raise ValueError(f'Ambiguous local enclosure at {p}: {counts}')
        return bool(counts[0])
