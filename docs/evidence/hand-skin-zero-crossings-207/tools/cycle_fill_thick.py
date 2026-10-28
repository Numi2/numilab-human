from pathlib import Path
import sys,json,time,heapq
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'authoring-deps'))
import manifold3d as mf
from repair_envelope_stable import close_for_boolean,from_mesh
from emit_f32 import emit

def cycles(P,F,root):
 E=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); E.sort(axis=1)
 U,iv=np.unique(E,axis=0,return_inverse=True); lens=np.linalg.norm(P[U[:,0]]-P[U[:,1]],axis=1)
 adj=coo_matrix((np.r_[lens,lens],(np.r_[U[:,0],U[:,1]],np.r_[U[:,1],U[:,0]])),shape=(len(P),len(P))).tocsr()
 dist,parent=dijkstra(adj,indices=root,return_predecessors=True)
 edges={tuple(e):i for i,e in enumerate(U)}; tree=np.zeros(len(U),bool)
 for i,p in enumerate(parent):
  if p>=0:tree[edges[tuple(sorted((i,int(p))))]]=True
 # A dual spanning tree using only edges outside primal tree.
 eidorder=np.argsort(iv,kind='stable'); sides=eidorder.reshape(-1,2)%len(F)
 fa=np.arange(len(F)); rank=np.zeros(len(F),np.int32)
 def find(a):
  while fa[a]!=a:fa[a]=fa[fa[a]];a=fa[a]
  return a
 dual=np.zeros(len(U),bool)
 for e in np.argsort(-lens):
  if tree[e]:continue
  a,b=map(find,sides[e])
  if a==b:continue
  if rank[a]<rank[b]:a,b=b,a
  fa[b]=a
  if rank[a]==rank[b]:rank[a]+=1
  dual[e]=True
 out=[]
 for e in np.flatnonzero(~tree&~dual):
  a,b=map(int,U[e]); aa=[];bb=[];v=a
  while v>=0:aa.append(v);v=int(parent[v])
  amap={v:i for i,v in enumerate(aa)};v=b
  while v not in amap and v>=0:bb.append(v);v=int(parent[v])
  if v<0:continue
  path=aa[:amap[v]+1]+bb[::-1]
  Q=P[path];length=float(np.linalg.norm(Q-np.roll(Q,1,axis=0),axis=1).sum());span=float(np.linalg.norm(np.ptp(Q,axis=0)))
  out.append((length,span,np.array(path)))
 return sorted(out,key=lambda x:(x[0],x[1]))

def export(body,out,source,source_cap_m,baseline):
 mesh=body.to_mesh64();P=np.asarray(mesh.vert_properties)[:,:3];F=np.asarray(mesh.tri_verts,dtype=np.int64);fi=np.asarray(mesh.face_id,dtype=np.int64);ri=np.asarray(mesh.run_index,dtype=np.int64)//3;oi=np.asarray(mesh.run_original_id,dtype=np.int64);fo=np.empty(len(F),np.int64)
 for i,o in enumerate(oi):fo[ri[i]:ri[i+1]]=o
 cap=(fo==source)&(fi>=source_cap_m);F=F[~cap][:,[0,2,1]];fo=fo[~cap];fi=fi[~cap]
 np.savez_compressed(out.with_suffix('.raw.npz'),points_world_m_f64=P,source_O_faces=F,envelope_source_original_id=fo,envelope_source_face_row=fi)
 Y,G,report=emit(P,F,baseline);used=np.unique(G);remap=np.full(len(Y),-1);remap[used]=np.arange(len(used));Y=Y[used];G=remap[G]
 np.savez_compressed(out,points_world_m=Y,source_O_faces=G,envelope_source_original_id=fo,envelope_source_face_row=fi)
 return report

def outer(body):
 parts=body.decompose();positive=[p for p in parts if p.volume()>0];negative=[p for p in parts if p.volume()<0]
 if len(positive)!=1:raise ValueError('Unexpected positive solid components '+str(len(positive)))
 return positive[0],negative

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--run',default='envelope-009-cycles');ap.add_argument('--max-rounds',type=int,default=8);ap.add_argument('--span-mm',type=float,default=8.);args=ap.parse_args()
 out=HERE/args.run;out.mkdir(exist_ok=False);z=np.load(HERE/'envelope-006-local-lattice/candidate.npz');X=z['points_world_m'];F=z['source_O_faces'];V,C,_,_=close_for_boolean(X,F[:,[0,2,1]]);body=from_mesh(V,C,np.arange(len(C)));sid=int(np.asarray(body.to_mesh64().run_original_id)[0]);t=time.monotonic();rep=[]
 parts=body.decompose();main=max(parts,key=lambda a:a.volume());voids=[p for p in parts if p.volume()<0]
 print(json.dumps({'stage':'decompose','volumes': [p.volume() for p in parts],'genus':main.genus()}),flush=True)
 # Dropping a closed negative-volume nested boundary fills that cavity; retain it separately as a witness.
 for i,p in enumerate(voids):
  pm=p.to_mesh64();np.savez_compressed(out/f'void-{i}.npz',points=np.asarray(pm.vert_properties)[:,:3],faces=np.asarray(pm.tri_verts))
 body=main
 for round in range(args.max_rounds):
  g=body.genus();print(json.dumps({'stage':'round','round':round,'genus':g,'faces':body.num_tri(),'elapsed_s':time.monotonic()-t}),flush=True)
  if g==0:break
  mesh=body.to_mesh64();P=np.asarray(mesh.vert_properties)[:,:3];Q=np.asarray(mesh.tri_verts,dtype=np.int64)
  roots=[]
  for sign in [-1,1]:
   indices=np.flatnonzero(sign*P[:,0]>.18)
   for v in [indices[np.argmin(P[indices,1])],indices[np.argmax(P[indices,0]*sign)],indices[np.argmin(P[indices,2])],indices[np.argmax(P[indices,2])]]:roots.append(int(v))
  rng=np.random.default_rng(731+round)
  handrows=np.flatnonzero(np.abs(P[:,0])>.18)
  roots+=list(map(int,rng.choice(handrows,size=min(40,len(handrows)),replace=False)))
  cs=[]
  for root in roots:cs.extend(cycles(P,Q,root))
  cs.sort(key=lambda v:(v[0],v[1]));seen=set();accepted=0
  print(json.dumps({'stage':'cycles','round':round,'shortest':[(a*1000,b*1000,len(c)) for a,b,c in cs[:8]]}),flush=True)
  for length,span,ids in cs:
   if span>args.span_mm/1000:continue
   key=tuple(sorted(map(int,ids)))
   if key in seen:continue
   seen.add(key);pts=P[ids];normal=np.linalg.svd(pts-pts.mean(axis=0),full_matrices=False)[2][-1];epsilon=0.00015
   hull=mf.Manifold.hull_points(np.array(np.vstack([pts+epsilon*normal,pts-epsilon*normal]),dtype=np.float64,copy=True)).as_original()
   trial,newvoids=outer(body+hull);ng=trial.genus()
   if trial.status()!=mf.Error.NoError or ng>=g:continue
   entry={'round':round,'old_genus':g,'new_genus':ng,'cycle_length_mm':length*1000,'span_mm':span*1000,'hull_volume_mm3':hull.volume()*1e9,'added_volume_mm3':(trial.volume()-body.volume())*1e9,'cycle_vertex_rows':ids.tolist(),'hull_original_id':hull.original_id(),'filled_new_void_volumes_mm3':[p.volume()*1e9 for p in newvoids]}
   np.savez_compressed(out/f'fill-{len(rep):03}.npz',cycle_points=pts,points=np.asarray(hull.to_mesh64().vert_properties)[:,:3],faces=np.asarray(hull.to_mesh64().tri_verts));rep.append(entry);print(json.dumps({'stage':'accepted_fill',**entry}),flush=True);body=trial;g=ng;accepted+=1
   if ng==0:break
   # cycles refer to prior geometry; additions are safe as points but recompute after a few fills.
   if accepted>=8:break
  (out/'progress.json').write_text(json.dumps(rep,indent=2))
  body,extra_voids=outer(body.simplify(1e-6))
  try:export(body,out/f'round-{round:02}.npz',sid,len(F),X)
  except ValueError as e:print(json.dumps({'stage':'emission_deferred','reason':str(e)}),flush=True)
  if not accepted:print('NO_SMALL_FILL_PROGRESS',flush=True);break
 emission=export(body,out/'candidate.npz',sid,len(F),X);(out/'report.json').write_text(json.dumps({'genus':body.genus(),'fills':rep,'voids':len(voids),'elapsed_s':time.monotonic()-t,'emission':emission},indent=2));print(json.dumps({'stage':'done','genus':body.genus(),'fills':len(rep),'elapsed_s':time.monotonic()-t}),flush=True)
if __name__=='__main__':main()
