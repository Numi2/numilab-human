#!/usr/bin/env python3
"""Compact physical target cavities using the existing247 topology/volume owner."""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,importlib.util,sys,json,time,hashlib
import numpy as np
H=Path(__file__).resolve().parent
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
C=load('compact291cavity247',H/'remesh_current_surface_cavities_247_root.py')
K=load('compact291inc286',H/'direct_skin_owner_union_caps_286_root.py')
J=load('compact291joint272',H/'audit_joint_native_apex_272_root.py')
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def face_owners(f,t,order,off):
 v=min(map(int,f),key=lambda x:int(off[x+1]-off[x]));cand=np.unique(order[off[v]:off[v+1]]//4)
 mask=np.ones(len(cand),bool)
 for x in f:mask &= np.any(t[cand]==x,axis=1)
 return cand[mask]
def group_components(p,t,mat,selected,missing,extrakeys):
 selected=np.asarray(sorted(selected),np.int64);parent=np.arange(len(selected))
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=int(parent[a])
  return a
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 first={}
 for i,cell in enumerate(t[selected]):
  for v in cell:
   v=int(v)
   if v in first:union(i,first[v])
   else:first[v]=i
 for f in missing:
  ids=[first[int(v)] for v in f if int(v) in first]
  if len(ids)!=3:raise ValueError('missing target facet lacks seeded vertex support')
  for i in ids[1:]:union(ids[0],i)
 buckets=defaultdict(list)
 for i,r in enumerate(selected):buckets[find(i)].append(int(r))
 assigned=defaultdict(list)
 for f in missing:assigned[find(first[int(f[0])])].append(f)
 comps=[]
 for label,rows in sorted(buckets.items()):
  rows=np.asarray(rows,np.int64);old=C.m.boundary(t[rows])
  shell=Counter()
  for f in old:
   if K.facekey(f) not in extrakeys:shell[K.facekey(f)]+=K.facesign(f)
  nf=np.asarray(assigned[label],np.int64).reshape(-1,3)
  for f in nf:shell[K.facekey(f)]+=K.facesign(f)
  shell={k:v for k,v in shell.items() if v}
  if any(abs(v)!=1 for v in shell.values()):raise ValueError('nonmanifold target cavity multiplicity')
  b=np.asarray([k if sg>0 else (k[0],k[2],k[1]) for k,sg in sorted(shell.items())],np.int64).reshape(-1,3)
  empty=not len(b)
  ok,edge=(True,0) if empty else C.closed(b)
  top=C.manifold_topology(b);g=np.unique(b);ix=np.full(len(p),-1,np.int64);ix[g]=np.arange(len(g))
  d,q=C.m.m.measurements(p,t[rows])
  comps.append({'id':len(comps),'rows':rows,'global_points':g,'faces_local':ix[b],
   'closed':ok and not top['bad_vertex_links'] and (empty or top['boundary_components']==1),'empty_shell_retirement':empty,
   'edge_failures':edge,'old_bad':int(np.count_nonzero(~C.m.m.passes(d,q))),
   'old_nonpositive':int(np.count_nonzero(d<=0)),'old_cells':len(rows),'new_desired_faces':len(nf),'topology':top,
   'joint_no_native_apex_faces':[]})
 return comps
def prepare(p,t,mat,desired,bb,missing,extra,maximum_cells=20000,maximum_rounds=32):
 order,off=K.node_incidence(t,len(p));selected=set();extrakeys=C.sortedkeys(extra);desiredkeys=C.sortedkeys(desired)
 retired=set(map(int,np.unique(bb)))-set(map(int,np.unique(desired)))
 for v in retired:selected.update(map(int,np.unique(order[off[v]:off[v+1]]//4)))
 for f in extra:
  hit=face_owners(f,t,order,off)
  if len(hit)!=1:raise ValueError('extra boundary face not uniquely owned')
  selected.add(int(hit[0]))
 if not selected:return [],{'selected_cells':0,'closure_history':[],'retired_boundary_nodes':len(retired)}
 history=[]
 for iteration in range(maximum_rounds):
  comps=group_components(p,t,mat,selected,missing,extrakeys);new=set();blocked=[];internal_bad=0
  for c in comps:
   if len(np.unique(mat[c['rows']]))!=1:raise ValueError('mixed material cavity')
   b=c['global_points'][c['faces_local']]
   if not c['closed']:
    nodes=set(map(int,C.bad_edge_nodes(b)))|set(map(int,c['topology']['witness_nodes']))
    for v in nodes:new.update(map(int,np.unique(order[off[v]:off[v+1]]//4)))
   approx=J.score(p,b);cand=np.flatnonzero(approx<J.GATE*(1-1e-10))
   if len(cand):
    vv=np.unique(b[cand]);ip=C.m.m.integer_points(p[vv]);mp={int(v):i for i,v in enumerate(vv)}
    for fi in cand:
     f=b[fi];reason,_=J.exact_terms(ip,[mp[int(v)] for v in f])
     if reason is None:continue
     witness={'face':f.tolist(),'reason':reason,'fixed_target':K.facekey(f) in desiredkeys}
     c['joint_no_native_apex_faces'].append(witness)
     if witness['fixed_target']:blocked.append(witness);continue
     owners=face_owners(f,t,order,off);add=[int(x) for x in owners if int(x) not in selected and int(mat[x])==int(mat[c['rows'][0]])]
     if not add:blocked.append(dict(witness,unavailable_matching_material_neighbor=True))
     else:new.update(add);internal_bad+=1
  new-=selected
  history.append({'iteration':iteration,'selected_cells':len(selected),'components':len(comps),
   'unclosed_components':sum(not c['closed'] for c in comps),'internal_joint_cut_witnesses':internal_bad,
   'blocked_witnesses':len(blocked),'new_cells':len(new)})
  if not new:break
  if len(selected|new)>maximum_cells:
   history[-1]['cell_budget_stop']=True;break
  selected|=new
 for c in comps:
  c['topological_closed']=c['closed'];c['closed']=c['closed'] and not c['joint_no_native_apex_faces']
 return comps,{'selected_cells':sum(len(c['rows']) for c in comps),'retired_boundary_nodes':len(retired),
  'closure_history':history,'maximum_cells':maximum_cells,'selection':'extra-face owners plus retired-boundary-node stars; exact internal joint-witness growth',
  'all_target_components_admissible':all(c['closed'] for c in comps)}
def fixture():
 p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],np.float32)
 t=C.m.m.orient(p,[[0,1,2,3],[0,2,1,4]]);bb=C.m.boundary(t);desired=C.m.boundary(t[1:])
 ak=C.sortedkeys(bb);dk=C.sortedkeys(desired)
 ex=np.asarray([f for f in bb if K.facekey(f) not in dk],np.int64);mi=np.asarray([f for f in desired if K.facekey(f) not in ak],np.int64)
 comps,r=prepare(p,t,np.full(2,15,np.int32),desired,bb,mi,ex)
 assert len(comps)==1 and len(comps[0]['rows'])==1 and r['retired_boundary_nodes']==1
 # The disappeared wedge needs no replacement; empty expected shell is a valid retirement.
 assert len(comps[0]['global_points'])==0 and comps[0]['empty_shell_retirement'] and comps[0]['closed']
 p2=np.array([[.5,.5,1],[0,0,0],[1,0,0],[1,1,0],[0,1,0]],np.float32)
 t2=C.m.m.orient(p2,[[0,1,2,3],[0,1,3,4]])
 want=C.m.m.orient(p2,[[0,1,2,4],[0,2,3,4]])
 b2=C.m.boundary(t2);d2=C.m.boundary(want);a2=C.sortedkeys(b2);k2=C.sortedkeys(d2)
 ex2=np.asarray([f for f in b2 if K.facekey(f) not in k2],np.int64)
 mi2=np.asarray([f for f in d2 if K.facekey(f) not in a2],np.int64)
 c2,r2=prepare(p2,t2,np.full(2,15,np.int32),d2,b2,mi2,ex2)
 assert len(c2)==1 and c2[0]['closed'] and not c2[0]['empty_shell_retirement']
 assert C.sortedkeys(c2[0]['global_points'][c2[0]['faces_local']])==k2
 return {'status':'PASS_COMPACT_RETIRED_VERTEX_OWNER_SEED_CONTROL','selected_cells':r['selected_cells'],
  'avoided_retained_neighbor_cells':1,'physical_admission':False,'native_steps':0}
def main(a):
 tic=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 p=v['points_world_m'];t=v['tetrahedra'];mat=v['material_indices'];desired=v['desired_boundary_faces'];bb=C.m.boundary(t)
 ak=C.sortedkeys(bb);dk=C.sortedkeys(desired)
 missing=np.asarray([f for f in desired if K.facekey(f) not in ak],np.int64).reshape(-1,3)
 extra=np.asarray([f for f in bb if K.facekey(f) not in dk],np.int64).reshape(-1,3)
 comps,prep=prepare(p,t,mat,desired,bb,missing,extra,a.maximum_cells)
 po=[0];fo=[0];co=[0]
 for c in comps:po.append(po[-1]+len(c['global_points']));fo.append(fo[-1]+len(c['faces_local']));co.append(co[-1]+len(c['rows']))
 shell=a.output/'connected-cavity-shells.npz'
 np.savez_compressed(shell,point_offsets=po,face_offsets=fo,cell_offsets=co,
  global_point_rows=np.concatenate([c['global_points'] for c in comps]) if comps else np.empty(0,np.int64),
  local_faces=np.concatenate([c['faces_local'] for c in comps]) if comps else np.empty((0,3),np.int64),
  input_cell_rows=np.concatenate([c['rows'] for c in comps]) if comps else np.empty(0,np.int64))
 report={'status':'PASS_COMPACT_TARGET_CAVITY_PREPARATION' if prep.get('all_target_components_admissible',False) else 'COMPACT_TARGET_CAVITY_WITNESSES_REMAIN',
  'physical_admission':False,'native_steps':0,'input':str(a.input),'input_sha256':sha(a.input),
  'script_sha256':sha(Path(__file__)),'preparation':prep,'components':[{k:v for k,v in c.items() if k not in ('rows','global_points','faces_local')} for c in comps],
  'output':str(shell),'output_sha256':sha(shell),'elapsed_s':time.monotonic()-tic}
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='components'}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fixture',action='store_true');ap.add_argument('--input',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--maximum-cells',type=int,default=20000);a=ap.parse_args()
 print(json.dumps(fixture())) if a.fixture else main(a)
