"""Deterministic Float32 emission preserving distinct exact-double geometry vertices."""
from itertools import product
import numpy as np

def emit(points64,faces,source32):
 P,ix=np.unique(np.asarray(points64,dtype=np.float64),axis=0,return_inverse=True);F=ix[np.asarray(faces,dtype=np.int64)]
 Etest=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]);_,ctest=np.unique(np.sort(Etest,axis=1),axis=0,return_counts=True)
 if np.any(ctest>2):P=np.asarray(points64,dtype=np.float64).copy();F=np.asarray(faces,dtype=np.int64).copy()
 Y=P.astype(np.float32)
 sourcekeys={tuple(p.view(np.uint32)) for p in np.asarray(source32,dtype=np.float32)}
 # Prioritize exact measured/source points so their bits never move.
 pinned=np.array([tuple(v.view(np.uint32)) in sourcekeys and np.array_equal(v.astype(float),p) for v,p in zip(Y,P)])
 seenpins=set()
 for i in np.flatnonzero(pinned):
  key=tuple(Y[i].view(np.uint32))
  if key in seenpins:pinned[i]=False
  else:seenpins.add(key)
 order=np.r_[np.flatnonzero(pinned),np.flatnonzero(~pinned)]
 incident=[[] for _ in range(len(P))]
 for f,t in enumerate(F):
  for v in t:incident[v].append(f)
 T=P[F];N=np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]);used={};nudges=[]
 for i in order:
  key=tuple(Y[i].view(np.uint32))
  if key not in used:used[key]=int(i);continue
  assert not pinned[i], 'Distinct pinned source vertices collapse in Float32'
  chosen=None;rows=np.asarray(incident[i]);tri=F[rows];ref=N[rows]
  for distance in [1,2,3,4,6,8,12,16]:
   choices=[]
   for steps in product([-distance,0,distance],repeat=3):
    v=Y[i].copy()
    for axis,s in enumerate(steps):
     for _ in range(abs(s)):v[axis]=np.nextafter(v[axis],np.float32(np.inf if s>0 else -np.inf))
    k=tuple(v.view(np.uint32))
    if k in used:continue
    choices.append((float(np.sum((v.astype(float)-P[i])**2)),k,v))
   for _,k,v in sorted(choices,key=lambda x:(x[0],x[1])):
    a=Y[tri].astype(float);a[tri==i]=v.astype(float);norm=np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]);dots=np.einsum('ij,ij->i',norm,ref)
    if np.all(dots>0):chosen=(k,v);break
   if chosen:break
  if chosen is None:raise ValueError(f'No safe Float32 lattice point for geometry vertex {i}')
  k,v=chosen;old=Y[i].copy();Y[i]=v;used[k]=int(i);nudges.append({'vertex':int(i),'from_bits':old.view(np.uint32).tolist(),'to_bits':v.view(np.uint32).tolist(),'distance_m':float(np.linalg.norm(v.astype(float)-P[i]))})
 E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]);_,counts=np.unique(np.sort(E,axis=1),axis=0,return_counts=True)
 return Y,F,{'unique_f64_points':len(P),'pinned_source_points':int(pinned.sum()),'nudged_vertices':nudges,'nonmanifold_edges':int(np.count_nonzero(counts>2)),'open_edges':int(np.count_nonzero(counts==1))}
