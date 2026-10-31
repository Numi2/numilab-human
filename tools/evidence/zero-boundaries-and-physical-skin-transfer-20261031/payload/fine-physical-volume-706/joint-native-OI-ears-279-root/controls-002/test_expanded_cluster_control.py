#!/usr/bin/env python3
import importlib.util, json, sys
from pathlib import Path
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
S=H/"fixed-boundary-quality-topology-196"
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
base=module("baseexpandedcontrol",S/"run_fixed_boundary_dp_batch_197.py")
ears=module("earsexpandedcontrol",H/"joint_native_lazy_ears_275_root.py")
ears.json=json
helper=module("helperexpandedcontrol",base.HELPER)
disk=module("diskexpandedcontrol",base.DISK)
def expanded(F,bad,protected,cap=100):
 edge={};vfaces=[set() for _ in range(int(F.max())+1)]
 for fi,(x,y,z) in enumerate(F):
  for u,v in ((int(x),int(y)),(int(y),int(z)),(int(z),int(x))):edge.setdefault((min(u,v),max(u,v)),[]).append(fi)
  for v in (int(x),int(y),int(z)):vfaces[v].add(fi)
 faceadj=[set() for _ in range(len(F))]
 for inc in edge.values():
  if len(inc)>1:
   for fi in inc:faceadj[fi].update(x for x in inc if x!=fi)
 pending=set(map(int,bad));components=[]
 while pending:
  seed=min(pending);pending.remove(seed);stack=[seed];comp={seed}
  while stack:
   fi=stack.pop()
   for nb in faceadj[fi]:
    if nb in pending:pending.remove(nb);stack.append(nb);comp.add(nb)
  components.append(sorted(comp))
 out=[]
 for comp in components:
  verts=set(map(int,F[comp].ravel()));rows=set(comp)
  for v in verts:rows.update(vfaces[v])
  star=set(rows)
  for fi in star:rows.update(faceadj[fi])
  rr=np.asarray(sorted(rows),np.int64);pv=set(map(int,F[rr].ravel()));loop=helper.patchloop(F,rr)
  if len(rr)<=cap and loop is not None and not ((pv&protected)-set(map(int,loop))):out.append((len(rr),len(loop),int(min(comp)),tuple(map(int,rr)),loop,pv))
 return out,components

def main():
 n=7;P=np.asarray([(i*1e-3,j*1e-3,0.) for j in range(n) for i in range(n)],np.float32);F=[]
 for j in range(n-1):
  for i in range(n-1):
   a=j*n+i;b=a+1;c=a+n;d=c+1;F.extend(((a,b,d),(a,d,c)))
 F=np.asarray(F,np.int64);seed=2*(3*(n-1)+3);candidates,components=expanded(F,[seed],set())
 assert len(components)==1 and len(candidates)==1
 _,_,_,rows,loop,_=candidates[0];assert len(rows)<=100 and len(loop)<100
 ext=helper.external(loop,helper.edgeindex(F,len(P)),np.asarray(rows,np.int64),set())
 class Pred:
  def _records(self,lattice,tris):return [(0,0,0,0,tuple(map(int,tris[0])))]
  def _audit_record_pairs(self,pairs,*,same_surface):return {"count":0}
 class Filtered:
  def _audit_pair_prepared_first(self,index,recs,*,same_surface):return {"count":0}
 pred=Pred();filtered=Filtered();lattice=[(0,0,0)]*len(P)
 new,recs,detail=ears.collision_aware_lazy_ears(loop,P,ext,object(),pred,filtered,lattice,[],1,disk,max_nodes=5000,max_seconds=2,minimum_altitude_m=1e-9,max_solutions=8,quality_floor=0.0)
 assert len(new)>0 and len(recs)==len(new) and detail["new_bad_face_count"]==0
 try:ears.collision_aware_lazy_ears(loop,P,ext,object(),pred,filtered,lattice,[],1,disk,max_nodes=5000,max_seconds=2,minimum_altitude_m=1.,max_solutions=8,quality_floor=0.0)
 except ValueError as exc:failure=str(exc)
 else:raise AssertionError("failure fixture unexpectedly found a valid ear plan")
 assert "no exact-clean lazy ear triangulation" in failure and "search_nodes" in failure
 print(json.dumps({"status":"PASS_EXPANDED_CONNECTED_CLUSTER_STARS_OUTER_RING_AND_EAR_SUCCESS_FAILURE_CONTROLS","fixture_faces":len(F),"seed_face":seed,"expanded_patch_faces":len(rows),"boundary_loop_vertices":len(loop),"success_new_faces":len(new),"success_solutions":detail["statistics"]["complete_exact_clean_solutions"],"failure_type":"ValueError","failure_has_search_stats":True,"runtime_json_binding_for_frozen_helper":True},indent=2))
if __name__=="__main__":main()
