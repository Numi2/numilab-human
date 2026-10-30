"""Batched source-tet projection for offline typed reference anchors only.
Evaluates requested point/tet pairs; makes no global nearest-source claim."""
import numpy as np
def project_pairs(points,parent_rows,parent_points,parent_tets,parent_inverses):
 p=np.asarray(points,np.float64);rows=np.asarray(parent_rows,np.int64)
 if p.ndim!=2 or p.shape[1]!=3 or rows.shape!=(len(p),) or not np.isfinite(p).all():raise ValueError('invalid query pair domain')
 if np.any(rows<0)or np.any(rows>=len(parent_tets)):raise ValueError('parent row range')
 if not len(p):return {'projected':p.copy(),'bary':np.empty((0,4)),'distance':np.empty(0)}
 xyz=np.asarray(parent_points,np.float64)[np.asarray(parent_tets)[rows]]
 if not np.isfinite(xyz).all():raise ValueError('nonfinite parent vertices')
 best=np.full(len(p),np.inf);bary=np.zeros((len(p),4));projected=np.empty_like(p)
 def consider(b,valid):
  nonlocal best,bary,projected
  q=np.einsum('ij,ijk->ik',b,xyz);d=np.einsum('ij,ij->i',q-p,q-p)
  take=valid&np.isfinite(d)&(d<best)
  best[take]=d[take];bary[take]=b[take];projected[take]=q[take]
 for j in range(4):
  b=np.zeros((len(p),4));b[:,j]=1;consider(b,np.ones(len(p),bool))
 for j in range(4):
  for k in range(j+1,4):
   delta=xyz[:,k]-xyz[:,j];norm2=np.einsum('ij,ij->i',delta,delta)
   if np.any(norm2<=0)or not np.isfinite(norm2).all():raise ValueError('degenerate source edge')
   u=np.clip(np.einsum('ij,ij->i',p-xyz[:,j],delta)/norm2,0,1)
   b=np.zeros((len(p),4));b[:,j]=1-u;b[:,k]=u;consider(b,np.ones(len(p),bool))
 for j,k,l in ((1,2,3),(0,3,2),(0,1,3),(0,2,1)):
  u=xyz[:,k]-xyz[:,j];v=xyz[:,l]-xyz[:,j];w=p-xyz[:,j];normal=np.cross(u,v)
  norm2=np.einsum('ij,ij->i',normal,normal)
  if np.any(norm2<=0)or not np.isfinite(norm2).all():raise ValueError('degenerate source face')
  bk=np.einsum('ij,ij->i',np.cross(w,v),normal)/norm2
  bl=np.einsum('ij,ij->i',np.cross(u,w),normal)/norm2;bj=1-bk-bl
  b=np.zeros((len(p),4));b[:,j]=bj;b[:,k]=bk;b[:,l]=bl
  consider(b,np.min(b,axis=1)>=0)
 v=np.einsum('ijk,ik->ij',np.asarray(parent_inverses)[rows],p-xyz[:,0])
 b=np.column_stack((1-v.sum(axis=1),v));consider(b,np.min(b,axis=1)>=0)
 if not np.isfinite(best).all()or np.any(bary<0)or np.max(np.abs(bary.sum(axis=1)-1))>1e-12:raise ValueError('invalid bounded anchor projection')
 return {'projected':projected,'bary':bary,'distance':np.sqrt(best)}
