#!/usr/bin/env python3
"""Read-only preparation: identify wider local remesh patches around all measured D slivers."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
BASE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
IN=BASE/'deep-core-void-coalescence-366/candidate-f32-typed.npz'
OUT=BASE/'deep-core-connected-patch-preflight-369/patch-preflight.json'
EDGE_TARGET=5e-6
ALT_TARGET=1e-6
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def main():
 z=np.load(IN,allow_pickle=False);P=np.asarray(z["points_world_m"],np.float32);F=np.asarray(z["faces"],np.int64);face_orig=np.asarray(z["face_source_original_id"],np.int64)
 tri=P[F].astype(np.float64)
 e=np.stack([np.linalg.norm(tri[:,1]-tri[:,0],axis=1),np.linalg.norm(tri[:,2]-tri[:,1],axis=1),np.linalg.norm(tri[:,0]-tri[:,2],axis=1)],axis=1)
 area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1);edge_min=e.min(1);alt=area2/e.max(1);lo=tri.min(1);hi=tri.max(1);roi=np.ones(len(F),bool);boxes=[]
 bad=np.flatnonzero(roi&((edge_min<EDGE_TARGET)|(alt<ALT_TARGET)))
 edge_faces=collections.defaultdict(list)
 for fi,(a,b,c) in enumerate(F):
  for u,v in ((a,b),(b,c),(c,a)):edge_faces[tuple(sorted((int(u),int(v))))].append(fi)
 nbr=[set() for _ in range(len(F))]
 for rows in edge_faces.values():
  if len(rows)==2:
   a,b=rows;nbr[a].add(b);nbr[b].add(a)
 unseen=set(map(int,bad));components=[]
 while unseen:
  root=min(unseen);unseen.remove(root);stack=[root];comp={root}
  while stack:
   f=stack.pop()
   for g in nbr[f]:
    if g in unseen and roi[g] and (edge_min[g]<EDGE_TARGET or alt[g]<ALT_TARGET):
     unseen.remove(g);comp.add(g);stack.append(g)
  components.append(comp)
 components.sort(key=lambda x:(-len(x),min(x)));summaries=[]
 for ci,comp in enumerate(components):
  ring_sets=[];cur=set(comp)
  for ring in range(3):
   ring_sets.append(cur.copy());nxt=set(cur)
   for f in cur:nxt.update(nbr[f])
   cur=nxt
  item={"component":ci,"core_face_count":len(comp),"core_face_rows":sorted(comp),"source_ids":sorted(set(map(int,face_orig[list(comp)]))),"core_min_edge_m":float(edge_min[list(comp)].min()),"core_min_altitude_m":float(alt[list(comp)].min()),"rings":[]}
  for n,patch in enumerate(ring_sets):
   fs=np.asarray(sorted(patch),np.int64);used=np.unique(F[fs].ravel());counts=collections.Counter()
   for f in fs:
    a,b,c=map(int,F[f]);counts[tuple(sorted((a,b)))]+=1;counts[tuple(sorted((b,c)))]+=1;counts[tuple(sorted((c,a)))]+=1
   bd=[e for e,c in counts.items() if c==1];degree=collections.Counter(v for e in bd for v in e);graph=collections.defaultdict(list)
   for a,b in bd:graph[a].append(b);graph[b].append(a)
   unseenv=set(graph);loops=[]
   while unseenv:
    seed=min(unseenv);unseenv.remove(seed);st=[seed];vs={seed}
    while st:
     v=st.pop()
     for w in graph[v]:
      if w in unseenv:unseenv.remove(w);vs.add(w);st.append(w)
    loops.append({"vertices":len(vs),"edges":sum(len(graph[v]) for v in vs)//2,"all_degree2":all(len(graph[v])==2 for v in vs)})
   X=P[used].astype(np.float64);center=X.mean(0);_,sv,vt=np.linalg.svd(X-center,full_matrices=False);normal=vt[-1];res=np.abs((X-center)@normal)
   item["rings"].append({"ring":n,"faces":int(len(fs)),"vertices":int(len(used)),"boundary_edges":len(bd),"boundary_non2_vertices":sum(v!=2 for v in degree.values()),"boundary_loops":loops,"plane_rms_m":float(np.sqrt(np.mean(res**2))),"plane_p95_m":float(np.quantile(res,.95)),"plane_max_m":float(res.max()),"min_edge_m":float(edge_min[fs].min()),"min_altitude_m":float(alt[fs].min()),"source_ids":sorted(set(map(int,face_orig[fs])))})
  summaries.append(item)
 result={"schema":"wide-patch-remesh-preflight.v1","input":str(IN),"input_sha256":sha(IN),"point_count":len(P),"face_count":len(F),"global_edge_lt5um":int(np.sum(edge_min<EDGE_TARGET)),"global_altitude_lt1um":int(np.sum(alt<ALT_TARGET)),"measured_roi_boxes":boxes,"roi_face_count":int(roi.sum()),"roi_bad_face_count":int(len(bad)),"component_count":len(components),"components":summaries,"read_only":True,"remesh_run":False}
 OUT.write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({k:result[k] for k in ("point_count","face_count","global_edge_lt5um","global_altitude_lt1um","roi_face_count","roi_bad_face_count","component_count")},sort_keys=True))
if __name__=="__main__":main()
