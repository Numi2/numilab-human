"""Z2 tree/cotree annotations and shortest nontrivial loops in a covering graph."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

def prepare(P,F):
 E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]);E.sort(axis=1);U,iv,ct=np.unique(E,axis=0,return_inverse=True,return_counts=True)
 if np.any(ct!=2):raise ValueError('Closed two-manifold required')
 lens=np.linalg.norm(P[U[:,0]]-P[U[:,1]],axis=1);N=len(P);A=coo_matrix((np.r_[lens,lens],(np.r_[U[:,0],U[:,1]],np.r_[U[:,1],U[:,0]])),shape=(N,N)).tocsr();root=int(U[0,0]);_,pa=dijkstra(A,indices=root,return_predecessors=True);lookup={tuple(e):i for i,e in enumerate(U)};T=np.zeros(len(U),bool)
 for i,p in enumerate(pa):
  if p>=0:T[lookup[tuple(sorted((i,int(p))))]]=True
 sides=np.argsort(iv,kind='stable').reshape(-1,2)%len(F);parent=np.arange(len(F));D=np.zeros(len(U),bool)
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 for e in np.argsort(-lens):
  if T[e]:continue
  a,b=map(find,sides[e])
  if a==b:continue
  parent[b]=a;D[e]=True
 L=np.flatnonzero(~T&~D);k=len(L)
 if k>6:raise ValueError('Cover restricted to <=6 homology generators')
 h=np.zeros(len(U),np.uint64);parity=np.zeros(len(F),np.uint64)
 for i,e in enumerate(L):
  bit=np.uint64(1<<i);h[e]=bit;parity[sides[e,0]]^=bit;parity[sides[e,1]]^=bit
 adj=[[]for _ in F]
 for e in np.flatnonzero(D):
  a,b=map(int,sides[e]);adj[a].append((b,e));adj[b].append((a,e))
 order=[0];parents=np.full(len(F),-1,np.int64);pe=np.full(len(F),-1,np.int64);parents[0]=0
 for a in order:
  for b,e in adj[a]:
   if parents[b]>=0:continue
   parents[b]=a;pe[b]=e;order.append(b)
 if len(order)!=len(F):raise ValueError('Dual tree not connected')
 for a in order[:0:-1]:h[pe[a]]=parity[a];parity[parents[a]]^=parity[a]
 # Cocycle sums around every triangle must vanish exactly.
 faceh=h[iv[:len(F)]]^h[iv[len(F):2*len(F)]]^h[iv[2*len(F):]]
 assert not faceh.any()
 S=1<<k;rows=[];cols=[];weights=[]
 for s in range(S):
  rows.extend([U[:,0]+s*N,U[:,1]+s*N]);cols.extend([U[:,1]+(s^h.astype(np.int64))*N,U[:,0]+(s^h.astype(np.int64))*N]);weights.extend([lens,lens])
 cover=coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(S*N,S*N)).tocsr()
 return cover,U,h,L,sides,N,S

def loops(P,F,root_count=24):
 cover,U,h,L,sides,N,S=prepare(P,F)
 if S==1:return []
 seeds=np.unique(U[L]);rng=np.random.default_rng(902);band=np.flatnonzero((P[:,1]<-.9)&(np.abs(P[:,0])>.18));seeds=np.unique(np.r_[seeds,rng.choice(band,size=min(root_count,len(band)),replace=False)])
 result=[];seen=set()
 for root in seeds:
  dist,pa=dijkstra(cover,indices=int(root),return_predecessors=True)
  for signature in range(1,S):
   cur=int(root+signature*N)
   if not np.isfinite(dist[cur]):continue
   path=[]
   while cur!=root and cur>=0:path.append(cur);cur=int(pa[cur])
   path.append(int(root));path=path[::-1];vv=np.asarray(path)%N;state=np.asarray(path)//N
   # Remove the contractible stem by extracting a shortest repeated physical-vertex loop with changed signature.
   best=None;at={}
   for i,v in enumerate(vv):
    for j in at.get(int(v),[]):
     if state[j]!=state[i]:
      ids=vv[j:i];length=float(np.linalg.norm(P[ids]-np.roll(P[ids],1,axis=0),axis=1).sum())
      if best is None or length<best[0]:best=(length,ids.copy())
    at.setdefault(int(v),[]).append(i)
   if best is None:continue
   length,ids=best;key=tuple(sorted(map(int,ids)))
   if key in seen:continue
   seen.add(key);span=float(np.linalg.norm(np.ptp(P[ids],axis=0)));result.append((length,span,ids))
 return sorted(result,key=lambda v:(v[0],v[1]))
