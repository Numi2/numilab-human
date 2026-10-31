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
OWNER=H.parent/'deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py'
sys.path.insert(0,str(OWNER.parent.parent))
exact=__import__('numilab_human.cardiac_cavity_intersections',fromlist=['_audit_pair'])
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def face_owners(f,t,order,off):
 v=min(map(int,f),key=lambda x:int(off[x+1]-off[x]));cand=np.unique(order[off[v]:off[v+1]]//4)
 mask=np.ones(len(cand),bool)
 for x in f:mask &= np.any(t[cand]==x,axis=1)
 return cand[mask]
def missing_face_groups(first, faces):
 faces=np.asarray(faces,np.int64).reshape(-1,3)
 parent=np.arange(len(faces),dtype=np.int64)
 def find(a):
  a=int(a)
  while parent[a]!=a:parent[a]=parent[parent[a]];a=int(parent[a])
  return a
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 first_face={}
 for fi,f in enumerate(faces):
  for v0 in f:
   v=int(v0)
   if v in first_face:union(fi,first_face[v])
   else:first_face[v]=fi
 buckets=defaultdict(list)
 for fi in range(len(faces)):buckets[find(fi)].append(fi)
 out=[]
 for members in sorted(buckets.values(),key=lambda x:x[0]):
  anchors=sorted({int(first[int(v)]) for fi in members for v in faces[fi] if int(v) in first})
  if not anchors:raise ValueError("missing target face component has no selected tetrahedron anchor")
  out.append((members,anchors))
 return out
def attach_missing_face_components(first,faces,union):
 groups=missing_face_groups(first,faces)
 for _,anchors in groups:
  for anchor in anchors[1:]:union(anchors[0],anchor)
 return groups
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
  for v0 in cell:
   v=int(v0)
   if v in first:union(i,first[v])
   else:first[v]=i
 missing=np.asarray(missing,np.int64).reshape(-1,3)
 face_groups=attach_missing_face_components(first,missing,union)
 buckets=defaultdict(list)
 for i,r in enumerate(selected):buckets[find(i)].append(int(r))
 assigned=defaultdict(list)
 for members,anchors in face_groups:
  label=find(anchors[0])
  for fi in members:assigned[label].append(missing[fi])
 comps=[]
 for label,rows in sorted(buckets.items()):
  rows=np.asarray(rows,np.int64);oldb=C.m.boundary(t[rows])
  shell=Counter()
  for f in oldb:
   if K.facekey(f) not in extrakeys:shell[K.facekey(f)]+=K.facesign(f)
  nf=np.asarray(assigned[label],np.int64).reshape(-1,3)
  for f in nf:shell[K.facekey(f)]+=K.facesign(f)
  shell={k:v for k,v in shell.items() if v}
  if any(abs(v)!=1 for v in shell.values()):raise ValueError("nonmanifold target cavity multiplicity")
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
def prepare(p,t,mat,desired,bb,missing,extra,maximum_cells=20000,maximum_rounds=32,additional_seed_cells=()):
 order,off=K.node_incidence(t,len(p));selected=set();extrakeys=C.sortedkeys(extra);desiredkeys=C.sortedkeys(desired)
 retired=set(map(int,np.unique(bb)))-set(map(int,np.unique(desired)))
 for v in retired:selected.update(map(int,np.unique(order[off[v]:off[v+1]]//4)))
 for f in extra:
  hit=face_owners(f,t,order,off)
  if len(hit)!=1:raise ValueError('extra boundary face not uniquely owned')
  selected.add(int(hit[0]))
 # Every live vertex on a missing typed-target facet contributes its full current tet star.
 # Facet components may also contain target vertices with no current volume star; those are
 # retained in the shell and assigned only through a connected missing-face component anchored
 # to at least one live selected cell.
 for v0 in np.unique(missing):
  v=int(v0)
  if off[v+1]>off[v]:selected.update(map(int,np.unique(order[off[v]:off[v+1]]//4)))
 seed_rows=np.asarray(additional_seed_cells,np.int64).reshape(-1)
 if len(np.unique(seed_rows))!=len(seed_rows):raise ValueError('duplicate prior seed cell rows')
 if len(seed_rows) and (seed_rows.min()<0 or seed_rows.max()>=len(t)):raise ValueError('prior seed cell row outside current input domain')
 selected.update(map(int,seed_rows))
 if not selected:
  if len(missing):raise ValueError('missing typed target surface has no live volume anchor')
  return [],{'selected_cells':0,'closure_history':[],'retired_boundary_nodes':len(retired)}
 history=[];shell_cache={}
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
  self_pairs=0;self_cut_faces=0
  if not new:
   for c in comps:
    if not c['closed'] or c['joint_no_native_apex_faces'] or c['empty_shell_retirement']:continue
    b=c['global_points'][c['faces_local']];cachekey=b.tobytes()
    gate=shell_cache.get(cachekey)
    if gate is None:
     rec,_=exact.float32_surface_records(p[c['global_points']],c['faces_local'])
     gate=exact._audit_pair(rec,rec,same_surface=True);shell_cache[cachekey]=gate
    c['exact_shell_self_pairs']=int(gate['count']);self_pairs+=int(gate['count'])
    touched=set()
    for a,brow in gate['triangle_pairs']:
     for fi in (a,brow):
      f=b[int(fi)]
      if K.facekey(f) in desiredkeys:continue
      fk=K.facekey(f)
      if fk in touched:continue
      touched.add(fk);owners=face_owners(f,t,order,off)
      add=[int(x) for x in owners if int(x) not in selected and int(mat[x])==int(mat[c['rows'][0]])]
      if add:new.update(add);self_cut_faces+=1
      else:blocked.append({'face':f.tolist(),'reason':'self_intersection_cut_has_no_available_neighbor'})
    if gate['count'] and not touched:blocked.append({'reason':'target_faces_self_intersect','component':c['id']})
  new-=selected
  history.append({'iteration':iteration,'selected_cells':len(selected),'components':len(comps),
   'unclosed_components':sum(not c['closed'] for c in comps),'internal_joint_cut_witnesses':internal_bad,
   'blocked_witnesses':len(blocked),'exact_shell_self_pairs':self_pairs,'self_cut_witness_faces':self_cut_faces,'new_cells':len(new)})
  if not new:break
  if len(selected|new)>maximum_cells:
   history[-1]['cell_budget_stop']=True;break
  selected|=new
 for c in comps:
  c['topological_closed']=c['closed'];c['closed']=c['closed'] and not c['joint_no_native_apex_faces'] and c.get('exact_shell_self_pairs',0)==0
 return comps,{'selected_cells':sum(len(c['rows']) for c in comps),'retired_boundary_nodes':len(retired),
  'closure_history':history,'maximum_cells':maximum_cells,'selection':'extra-face owners, retired-boundary-node stars, and all live missing-target vertex stars; missing-face components joined through shared vertices to live anchors; exact internal joint-witness growth',
  'all_target_components_admissible':all(c['closed'] for c in comps)}
def fixture():
 # A shared unused target vertex joins two missing-facet records; distinct live
 # anchors must merge, while an unanchored missing component must fail closed.
 test_parent=list(range(3))
 def test_find(a):
  while test_parent[a]!=a:test_parent[a]=test_parent[test_parent[a]];a=test_parent[a]
  return a
 def test_union(a,b):
  a,b=test_find(a),test_find(b)
  if a!=b:test_parent[max(a,b)]=min(a,b)
 groups=attach_missing_face_components({10:0,11:1,12:2},np.asarray([[10,90,91],[11,90,92],[12,93,94]],np.int64),test_union)
 assert len(groups)==2 and test_find(0)==test_find(1) and test_find(1)!=test_find(2)
 try:missing_face_groups({},np.asarray([[90,91,92]],np.int64))
 except ValueError:pass
 else:raise AssertionError("unanchored target component accepted")
 p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
 t=C.m.m.orient(p,[[0,1,2,3]]);bb=C.m.boundary(t);desired=np.empty((0,3),np.int64)
 ex=np.asarray(bb,np.int64).reshape(-1,3);mi=np.empty((0,3),np.int64)
 comps,r=prepare(p,t,np.full(1,15,np.int32),desired,bb,mi,ex)
 assert len(comps)==1 and len(comps[0]['rows'])==1 and r['retired_boundary_nodes']==4
 assert len(comps[0]['global_points'])==0 and comps[0]['empty_shell_retirement'] and comps[0]['closed']
 # Missing target facets still rebuild a valid cavity after all live target vertex stars are seeded.
 p2=np.array([[.5,.5,1],[0,0,0],[1,0,0],[1,1,0],[0,1,0]],np.float32)
 t2=C.m.m.orient(p2,[[0,1,2,3],[0,1,3,4]])
 want=C.m.m.orient(p2,[[0,1,2,4],[0,2,3,4]])
 b2=C.m.boundary(t2);d2=C.m.boundary(want);a2=C.sortedkeys(b2);k2=C.sortedkeys(d2)
 ex2=np.asarray([f for f in b2 if K.facekey(f) not in k2],np.int64)
 mi2=np.asarray([f for f in d2 if K.facekey(f) not in a2],np.int64)
 c2,r2=prepare(p2,t2,np.full(2,15,np.int32),d2,b2,mi2,ex2)
 assert len(c2)==1 and c2[0]['closed'] and not c2[0]['empty_shell_retirement']
 assert C.sortedkeys(c2[0]['global_points'][c2[0]['faces_local']])==k2
 return {'status':'PASS_COMPACT_TARGET_COMPONENT_ANCHOR_CONTROLS','selected_cells':r['selected_cells'],
  'missing_face_component_anchor_union':True,'unanchored_component_fail_closed':True,
  'missing_target_live_vertex_stars_seeded':True,'physical_admission':False,'native_steps':0}
def main(a):
 tic=time.monotonic();a.output.mkdir(parents=True,exist_ok=False)
 with np.load(a.input) as z:v={k:z[k] for k in z.files}
 p=v['points_world_m'];t=v['tetrahedra'];mat=v['material_indices'];desired=v['desired_boundary_faces'];bb=C.m.boundary(t)
 ak=C.sortedkeys(bb);dk=C.sortedkeys(desired)
 missing=np.asarray([f for f in desired if K.facekey(f) not in ak],np.int64).reshape(-1,3)
 extra=np.asarray([f for f in bb if K.facekey(f) not in dk],np.int64).reshape(-1,3)
 seed_rows=np.empty(0,np.int64);seed_info=None
 if a.seed_cell_map is not None:
  if a.seed_cell_receipt is None or not a.seed_cell_map_sha256 or not a.seed_cell_receipt_sha256:raise ValueError('seed map requires both pinned map and receipt')
  if sha(a.seed_cell_map)!=a.seed_cell_map_sha256 or sha(a.seed_cell_receipt)!=a.seed_cell_receipt_sha256:raise ValueError('seed map/receipt SHA mismatch')
  seed_receipt=json.loads(a.seed_cell_receipt.read_text())
  if seed_receipt.get('status')!='PASS_EXACT_ORDERED_TET_MATERIAL_SEED_REBASE' or seed_receipt.get('physical_admission') is not False:raise ValueError('seed receipt is not source-only exact rebase')
  if seed_receipt.get('seed_rows',{}).get('sha256')!=a.seed_cell_map_sha256:raise ValueError('seed receipt does not bind seed map')
  base_pin=json.loads(str(np.asarray(v.get('fat_current_target_rebind_source_asset_sha256_json','{}')).item()))
  if base_pin.get('base_candidate')!=seed_receipt.get('current_candidate',{}).get('sha256'):raise ValueError('seed map base candidate differs from target-rebound input lineage')
  with np.load(a.seed_cell_map,allow_pickle=False) as z:seed_rows=np.asarray(z['current_candidate_cell_rows'],np.int64)
  if len(seed_rows)!=seed_receipt.get('selected_cells') or len(np.unique(seed_rows))!=len(seed_rows):raise ValueError('seed row count/injectivity mismatch')
  seed_info={'path':str(a.seed_cell_map),'sha256':a.seed_cell_map_sha256,'receipt_path':str(a.seed_cell_receipt),'receipt_sha256':a.seed_cell_receipt_sha256,'base_candidate_sha256':seed_receipt['current_candidate']['sha256'],'count':len(seed_rows)}
 elif a.seed_cell_receipt is not None or a.seed_cell_map_sha256 or a.seed_cell_receipt_sha256:raise ValueError('seed map pins supplied without seed map')
 comps,prep=prepare(p,t,mat,desired,bb,missing,extra,a.maximum_cells,additional_seed_cells=seed_rows)
 prep['prior_seed_cell_count']=len(seed_rows);prep['prior_seed_cell_map']=seed_info
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
 ap=argparse.ArgumentParser();ap.add_argument('--fixture',action='store_true');ap.add_argument('--input',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--maximum-cells',type=int,default=20000)
 ap.add_argument('--seed-cell-map',type=Path);ap.add_argument('--seed-cell-map-sha256');ap.add_argument('--seed-cell-receipt',type=Path);ap.add_argument('--seed-cell-receipt-sha256');a=ap.parse_args()
 print(json.dumps(fixture())) if a.fixture else main(a)
