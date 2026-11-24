"""Restore intermediate closed-muscle volume in the actual authoring P1 metric before Float32 emission."""
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.sparse.linalg import splu
from repair_registered_surface import signed_volume_gradient,volume
from prepare_infraspinatus import dets

class MuscleVolume:
    def __init__(self,xyz,t,f,pins):
        self.f=f;ref=dets(xyz,t);edges=xyz[t[:,1:]]-xyz[t[:,:1]]
        if np.any(ref<=0):raise ValueError('Invalid muscle field')
        inv=np.linalg.inv(edges);g=np.concatenate((-inv.sum(2,keepdims=True),inv),axis=2).transpose(0,2,1)
        E=(g@g.transpose(0,2,1))*ref[:,None,None]/6
        ii,jj=np.broadcast_arrays(t[:,:,None],t[:,None,:]);mass=np.bincount(t.ravel(),weights=np.repeat(ref/24,4),minlength=len(xyz))
        length=2*np.median(np.linalg.norm(xyz[f[:,1]]-xyz[f[:,0]],axis=1))
        K=coo_matrix((E.ravel(),(ii.ravel(),jj.ravel())),shape=(len(xyz),len(xyz))).tocsc()+diags(mass/length**2)
        self.free=np.setdiff1d(np.arange(len(xyz)),pins);self.op=splu(K[self.free][:,self.free].tocsc())
        self.start=xyz;self.goal=volume(xyz,f)
    def correct(self,dx):
        x=self.start+dx
        for _ in range(4):
            d=volume(x,self.f)-self.goal
            if abs(d)<abs(self.goal)*1e-10:break
            g=signed_volume_gradient(x,self.f)[self.free]
            u=self.op.solve(g);den=np.sum(g*u)
            if den<=0:raise ValueError('Missing independent volume coordinate')
            x[self.free]-=u*(d/den)
        shift=x-self.start-dx
        if np.linalg.norm(shift,axis=1).max()>5e-5 or abs(volume(x,self.f)/self.goal-1)>1e-8:raise ValueError('Unresolved volume restoration')
        return x-self.start
