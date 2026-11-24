"""Shared derivatives map a proposed shoulder edit into every registered source and skinning frame."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from igl.copyleft import cgal
from assemble_material import frames
from registration_stages import parts
from repair_registered_surface import state
from prepare_infraspinatus import dets
from prepare_shoulder import read

def separating_axis(a,b):
    e=a[[1,2,0]]-a;d=b[[1,2,0]]-b
    n,m=np.cross(e[0],e[1]),np.cross(d[0],d[1])
    axes=np.vstack((n,m,np.cross(e[:,None,:],d[None,:,:]).reshape(-1,3),np.cross(n,e),np.cross(m,d)))
    mag=np.linalg.norm(axes,axis=1);axes=axes[mag>1e-12]/mag[mag>1e-12,None]
    axes=np.vstack((axes,-axes));g=(b@axes.T).min(0)-(a@axes.T).max(0)
    k=g.argmax();return axes[k],g[k]

class MaterialPoses:
    def __init__(self,sid,case,cage,source):
        poses=frames(case);a=read(case/'launch.json')['argv'];payload=Path(a[a.index('--soft-tissue-payload')+1])
        z=np.load(source/f'bank-{sid}.npz');p=np.load(cage/f'bank-{sid}.npz');_,_,owners,_,aff=parts(payload,sid)
        w=z['w'].astype('f4').astype(float);self.ref=z['x'];self.f=np.ascontiguousarray(z['f']);self.t=p['t']
        self.centroid=self.ref.mean(0);t=self.t;Y=p['accepted'];self.n=len(w);self.w=w;self.owners=owners;self.aff=aff;self.poses=poses
        mats=[np.einsum('vb,bij->vij',w,[Rotation.from_quat(fr[b]['quaternion_xyzw']).as_matrix()@A for b,(A,_)in zip(owners,aff)])for fr in poses]
        self.inverse=np.linalg.inv(mats[-1]);boundary=[self.inverse]+[v@self.inverse for v in mats]
        self.maps=[];self.base=[];self.vol=[];self.coords=[];E=Y[t[:,1:]]-Y[t[:,:1]];dy=dets(Y,t);q=0
        for name,outer in zip(['reference','rig_reference','opening','accepted'],boundary):
            X=p[name];H=np.linalg.solve(E,X[t[:,1:]]-X[t[:,:1]]).transpose(0,2,1)
            M=np.zeros((len(Y),3,3));weight=np.zeros(len(Y))
            for ids in t.T:np.add.at(M,ids,H*dy[:,None,None]);np.add.at(weight,ids,dy)
            M/=weight[:,None,None];M[:len(w)]=outer;self.maps.append(M);self.coords.append(X);q+=1
        self.base=self.xyz(np.zeros_like(Y));self.vol=[dets(x,t)for x in self.base]
        if any((v<=0).any()for v in self.vol):raise ValueError('Nonpositive base')
    def xyz(self,delta):
        src=(self.ref+np.einsum('nij,nj->ni',self.inverse,delta[:self.n])*1e-3).astype('f4').astype(float)
        borders=[src]+[state(src,self.w,self.aff,self.owners,p).astype('f4').astype(float)for p in self.poses]
        result=[]
        for X,M,b in zip(self.coords,self.maps,borders):
            y=(X-X.mean(0))*1000+np.einsum('nij,nj->ni',M,delta);y[:self.n]=(b-X.mean(0))*1000;result.append(y)
        return result
    def safe(self,delta):
        for x,v in zip(self.xyz(delta),self.vol):
            if (dets(x,self.t)/v < .08).any() or len(cgal.remesh_self_intersections(np.ascontiguousarray(x[:self.n]),self.f,detect_only=True)[2]):return False
        return True
    def restrictions(self,now,future):
        t=self.t;current=self.xyz(now);trial=self.xyz(future);rules=[]
        for i,(x,y,v,M)in enumerate(zip(current,trial,self.vol,self.maps)):
            J=dets(y,t)/v
            for k in np.flatnonzero(J<.14)[np.argsort(J[J<.14],kind='stable')[:8]]:
                ids=t[k];edge=x[ids[1:]]-x[ids[0]];g=np.cross(edge[[1,2,0]],edge[[2,0,1]])/v[k]
                gg=-np.vstack((-g.sum(0),g));gradient=np.einsum('ni,nij->nj',gg,M[ids])
                rules.append((('pose',i,int(k)),ids,gradient,.14-np.linalg.det(edge)/v[k]))
            pairs=cgal.remesh_self_intersections(np.ascontiguousarray(y[:self.n]),self.f,detect_only=True)[2]
            for a,b in pairs[:16]:
                A,B=self.f[a],self.f[b];n,gap=separating_axis(x[A],x[B])
                if gap<=1e-9:continue
                u=A[np.argmax(y[A]@n)];vv=B[np.argmin(y[B]@n)];ids=np.array([u,vv])
                grad=np.array([n@M[u],-n@M[vv]])
                rules.append((('separation',i,int(u),int(vv)),ids,grad,.05*gap-(x[vv]-x[u])@n))
        return rules
