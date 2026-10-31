"""Positive source-tetrahedron sites inside each receiver's transport ball."""
import numpy as np
def source_tetrahedron_ball_sites(child_points,parent_rows,base_barycentric,source_points,source_tets,radius_m):
 p=np.asarray(child_points,dtype=np.float64);pr=np.asarray(parent_rows,dtype=np.int64);b0=np.asarray(base_barycentric,dtype=np.float64)
 if p.shape!=(len(pr),3)or b0.shape!=(len(pr),4)or not np.isfinite(p).all()or not np.isfinite(b0).all()or radius_m<=0:raise ValueError("invalid source-ball request")
 if not len(pr):return {"pair_rows":np.empty(0,dtype=np.int64),"barycentric":np.empty((0,4)),"points":np.empty((0,3)),"distance_m":np.empty(0)}
 tet=np.asarray(source_points,dtype=np.float64)[np.asarray(source_tets)[pr]]
 goals=[np.full(4,.25)]
 goals.extend(np.eye(4))
 for j in range(4):
  for k in range(j+1,4):
   v=np.zeros(4);v[j]=v[k]=.5;goals.append(v)
 for j in range(4):
  v=np.full(4,1/3);v[j]=0;goals.append(v)
 goals=np.asarray(goals)*(1-1e-8)+.25e-8
 start=np.einsum("ij,ijk->ik",b0,tet)
 end=np.einsum("gj,ijk->igk",goals,tet)
 delta=end-start[:,None,:];offset=start-p
 a=np.einsum("igj,igj->ig",delta,delta)
 b=np.einsum("ij,igj->ig",offset,delta)
 base_distance=np.linalg.norm(offset,axis=1)
 if np.any(base_distance>radius_m+1e-14):raise ValueError("source-ball base is outside radius")
 rho=radius_m-np.minimum(1e-10,np.maximum(0,radius_m-base_distance)*.25)
 c=np.einsum("ij,ij->i",offset,offset)-rho*rho
 disc=np.sqrt(np.maximum(0,b*b-a*c[:,None]))
 alpha=np.ones_like(a)
 outside=np.linalg.norm(end-p[:,None,:],axis=2)>rho[:,None]
 positive=outside&(a>0)
 stable=positive&(b>=0);other=positive&~stable
 denominator=b+disc
 alpha[stable]=np.divide(-np.broadcast_to(c[:,None],a.shape)[stable],denominator[stable],out=np.zeros(stable.sum()),where=denominator[stable]!=0)
 alpha[other]=(-b[other]+disc[other])/a[other]
 alpha[outside&(a==0)]=0
 alpha=np.clip(alpha,0,1);alpha[outside]*=1-1e-12
 bary=b0[:,None,:]+alpha[:,:,None]*(goals[None,:,:]-b0[:,None,:])
 bary=np.concatenate([b0[:,None,:],bary],axis=1)
 points=np.einsum("igj,ijk->igk",bary,tet)
 distance=np.linalg.norm(points-p[:,None,:],axis=2)
 rows=np.repeat(np.arange(len(pr),dtype=np.int64),bary.shape[1])
 bary=bary.reshape(-1,4);points=points.reshape(-1,3);distance=distance.ravel()
 valid=(distance<radius_m)&np.all(bary>0,axis=1)
 rows=rows[valid];bary=bary[valid];points=points[valid];distance=distance[valid]
 if not np.isfinite(bary).all()or np.max(np.abs(bary.sum(axis=1)-1),initial=0)>2e-12:raise ValueError("invalid source-ball output")
 return {"pair_rows":rows,"barycentric":bary,"points":points,"distance_m":distance}
