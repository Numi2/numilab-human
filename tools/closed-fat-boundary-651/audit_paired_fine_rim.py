#!/usr/bin/env python3
from pathlib import Path
import collections, csv, hashlib, importlib.util, json, math, os, signal, sys, time
import numpy as np

P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
OUT=Path(__file__).resolve().parent/'output'
ASSET=P/'conforming-fit-006-native-assets-545/run-002-operational-reference/registered-assets'
COARSE=P/'reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006'
I_PATH=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz'
D_PATH=P/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/full-native-self-028/current-collar-plan-032/engine-20-ops-one-sided624-run-006/candidate-f32-linked-remesh-typed.npz'
I_ANCHOR=P/'root-connected-layer-fit-219/canonical-registration-diagnostic-228/attempt-004-parent-paired-projection/I-current-anchor.npz'
D_ANCHOR=P/'root-connected-layer-fit-219/canonical-registration-diagnostic-228/attempt-004-parent-paired-projection/D-current-anchor.npz'
I_TEMP=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/inputs/I-temporary-closed.npz'
D_TEMP=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/inputs/D-temporary-closed.npz'
PARENT=P/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv.nodes.csv'
MANIFEST=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json'
F32_BONES=P/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz'
EXACT_OWNER=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001/tools/cardiac_cavity_intersections.py')
LOADER=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py'
BONE_HELPER=P/'bone-barrier-remeshing-609/bone_barrier.py'
EXACT_OWNER_SHA='f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d'
EXPECTED={
 I_PATH:'5d46ee99df2613e9f30a43075af95a6402beffe26392cd9f017d8c1a154bf410',
 D_PATH:'c17794a1012f76924dcd9e4e754bb026c774deb5a3d37e2f1c79635ccfe9dc83',
}
ROLE_OFFSET={'I':2411,'D':4822}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()

def rows(path):
 with Path(path).open(newline='') as f: return list(csv.DictReader(f))

def loadmod(name,path):
 spec=importlib.util.spec_from_file_location(name,path); mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def boundary_edges(faces):
 edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),axis=0)
 und=np.sort(edges,axis=1)
 u,inv,cnt=np.unique(und,axis=0,return_inverse=True,return_counts=True)
 dirs=np.where(edges[:,0]<edges[:,1],1,-1)
 bal=np.bincount(inv,weights=dirs,minlength=len(u)).astype(np.int64)
 return u,cnt,bal

def oriented_closed(faces):
 u,cnt,bal=boundary_edges(faces)
 return bool(len(cnt)>0 and np.all(cnt==2) and np.all(bal==0)),u,cnt,bal

def connected_faces(faces, counts):
 n=len(faces); parent=list(range(n)); rank=bytearray(n)
 def find(x):
  while parent[x]!=x:
   parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a=find(a);b=find(b)
  if a==b:return
  if rank[a]<rank[b]:a,b=b,a
  parent[b]=a
  if rank[a]==rank[b]:rank[a]+=1
 edge_face=np.tile(np.arange(n,dtype=np.int64),3)
 _,inv,_=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),axis=0),axis=1),axis=0,return_inverse=True,return_counts=True)
 order=np.argsort(inv,kind='stable')
 groups=np.split(edge_face[order],np.cumsum(counts)[:-1])
 for g in groups:
  if len(g)==2: union(int(g[0]),int(g[1]))
 roots=np.fromiter((find(i) for i in range(n)),dtype=np.int64,count=n)
 _,labels=np.unique(roots,return_inverse=True)
 return labels

def progress(stage,**fields):
 rec={'stage':stage,'elapsed_s':time.monotonic()-START,**fields}
 (OUT/'progress.json').write_text(json.dumps(rec,indent=2,sort_keys=True)+'\n')
 print(json.dumps(rec),flush=True)

def check_connected_faces_fixture():
 # Three independent components: two closed tetrahedral shells and a pair
 # of triangles joined by a shared edge. Each must receive one stable label.
 faces=np.asarray([[0,1,2],[0,3,1],[1,3,2],[2,3,0],
                   [4,5,6],[4,7,5],[5,7,6],[6,7,4],
                   [8,9,10],[9,8,11]],dtype=np.int64)
 _,counts,_=boundary_edges(faces)
 labels=connected_faces(faces,counts)
 groups=[set(np.flatnonzero(labels==labels[i]).tolist()) for i in (0,4,8)]
 if groups!=[{0,1,2,3},{4,5,6,7},{8,9}]:raise RuntimeError(f'face component mapping fixture failed: {groups}')
 if len(set(int(labels[i]) for i in (0,4,8)))!=3:raise RuntimeError('disconnected fixture components share a label')
 return {'status':'PASS','face_groups':[sorted(g) for g in groups]}

def main():
 global START
 OUT.mkdir(parents=True,exist_ok=False); START=time.monotonic()
 component_fixture=check_connected_faces_fixture()
 progress('connected-face-component-fixture-pass',fixture=component_fixture)
 inputs=[I_PATH,D_PATH,I_ANCHOR,D_ANCHOR,I_TEMP,D_TEMP,PARENT,MANIFEST,F32_BONES,ASSET/'receipt.json',COARSE/'receipt.json',P/'paired-rim-006-632/run-001/report.json']
 pins={str(p):sha(p) for p in inputs}
 for p,h in EXPECTED.items():
  if pins[str(p)]!=h: raise ValueError(f'fine candidate pin mismatch: {p} {pins[str(p)]}')
 asset_receipt=json.loads((ASSET/'receipt.json').read_text())
 for p,h in asset_receipt['output_sha256'].items():
  if sha(p)!=h:raise ValueError(f'545 registered asset changed: {p}')
 coarse_report=json.loads((P/'paired-rim-006-632/run-001/report.json').read_text())
 if coarse_report.get('candidate_sha256')!=sha(COARSE/'candidate.npz') or coarse_report.get('status')!='PASS_TOPOLOGY_AND_TET_BOUNDARY_ONLY':raise ValueError('coarse paired-rim source receipt mismatch')
 progress('pins-verified',pinned_files=len(pins),fine_I_sha256=pins[str(I_PATH)],fine_D_sha256=pins[str(D_PATH)])
 with np.load(I_PATH,allow_pickle=False) as z: ip=np.asarray(z['points_world_m']);iff=np.asarray(z['faces'],dtype=np.int64)
 with np.load(D_PATH,allow_pickle=False) as z: dp=np.asarray(z['points_world_m']);dff=np.asarray(z['faces'],dtype=np.int64)
 if ip.ndim!=2 or dp.ndim!=2 or ip.shape[1]!=3 or dp.shape[1]!=3 or not np.isfinite(ip).all() or not np.isfinite(dp).all():raise ValueError('invalid fine point arrays')
 if not np.array_equal(ip.astype(np.float32).astype(ip.dtype),ip) or not np.array_equal(dp.astype(np.float32).astype(dp.dtype),dp):raise ValueError('fine surface points are not canonical Float32 values')
 def get_boundary(faces):
  edge_list=collections.Counter()
  for tri in faces:
   for a,b in ((int(tri[0]),int(tri[1])),(int(tri[1]),int(tri[2])),(int(tri[2]),int(tri[0]))):
    edge_list[tuple(sorted((a,b)))]+=1
  if any(n>2 for n in edge_list.values()):raise ValueError('nonmanifold fine open surface edge')
  return {edge for edge,n in edge_list.items() if n==1}
 ibound=get_boundary(iff);dbound=get_boundary(dff)
 if len(ibound)!=45 or len(dbound)!=45:raise ValueError(f'expected 45 fine head-rim edges, got I={len(ibound)} D={len(dbound)}')
 def one_boundary_loop(bound):
  adj=collections.defaultdict(set)
  for a,b in bound:adj[a].add(b);adj[b].add(a)
  if any(len(v)!=2 for v in adj.values()):raise ValueError('fine boundary is not a union of loops')
  seen=set();lens=[]
  for s in adj:
   if s in seen:continue
   q=[s];comp=set()
   while q:
    v=q.pop()
    if v in comp:continue
    comp.add(v);q.extend(adj[v]-comp)
   seen|=comp;lens.append(len(comp))
  return sorted(lens),set(adj)
 iloops,iverts=one_boundary_loop(ibound);dloops,dverts=one_boundary_loop(dbound)
 if iloops!=[21,24] or dloops!=[21,24]:raise ValueError(f'unexpected ring loop sizes: I={iloops} D={dloops}')
 progress('fine-boundary-enumerated',I_faces=len(iff),I_edges=len(ibound),I_loops=iloops,D_faces=len(dff),D_edges=len(dbound),D_loops=dloops)

 # The anchor files retain the 330 source-node identity. A ring point must match
 # exactly one anchor point in its packed F32 representation and have one-hot
 # (not triangle/barycentric) ancestry to a role-local source node.
 fine_by_parent={}; role_ring={}
 anchor_info={}
 parent_current_delta={r:[] for r in ('I','D')}
 for role,points,verts in (('I',ip,iverts),('D',dp,dverts)):
  anchor=I_ANCHOR if role=='I' else D_ANCHOR
  with np.load(anchor,allow_pickle=False) as z:
   ap=np.asarray(z['points_world_m']);parent=np.asarray(z['vertex_source_parent_node_id'],dtype=np.int64);face=np.asarray(z['vertex_source_face_row'],dtype=np.int64);bary=np.asarray(z['vertex_source_barycentric'],dtype=np.float64);topid=np.asarray(z['vertex_registered_topology_id'],dtype=np.int64)
   source_hash=str(z['source_input_sha256'][0]); frame=str(z['coordinate_frame'][0])
  if frame!='D37 compiled parent current world coordinates, metres; F32 source rows retained':raise ValueError('unexpected anchor coordinate frame')
  if len(ap)!=len(parent) or len(face)!=len(parent) or bary.shape!=(len(parent),3):raise ValueError('anchor array domains disagree')
  anchor_by_bits=collections.defaultdict(list)
  for j,p in enumerate(ap):anchor_by_bits[np.asarray(p,dtype=np.float32).tobytes()].append(j)
  role_offset=ROLE_OFFSET[role];mapping=[];parent_to_fine={}
  for v in sorted(verts):
   ids=anchor_by_bits.get(np.asarray(points[v],dtype=np.float32).tobytes(),[])
   if len(ids)!=1:raise ValueError(f'{role} fine boundary vertex {v} maps to {len(ids)} exact anchor rows')
   a=int(ids[0]);pid=int(parent[a])
   if pid<role_offset or pid>=role_offset+2411 or face[a]!=-1 or not np.array_equal(bary[a],np.zeros(3)):
    raise ValueError(f'{role} boundary vertex {v} lacks one-hot parent330 node ancestry: anchor={a} parent={pid} face={face[a]}')
   local=pid-role_offset
   if local in parent_to_fine:raise ValueError(f'{role} duplicate ring parent row {local}')
   parent_to_fine[local]=int(v)
   mapping.append((int(v),a,pid,local,int(topid[a])))
  if len(parent_to_fine)!=45:raise ValueError('ring correspondence row count mismatch')
  fine_by_parent[role]={ROLE_OFFSET[role]+local:v for local,v in parent_to_fine.items()}
  role_ring[role]=mapping
  anchor_info[role]={'anchor_path':str(anchor),'anchor_sha256':sha(anchor),'anchor_source_input_sha256':source_hash,'anchor_rows':len(parent),'boundary_rows':len(mapping),'one_hot_parent_nodes':True,'topology_id_unique_within_role':len({x[4] for x in mapping})==45}
 # Check each ring identity against the immutable parent330 node ledger and 545 role map.
 parent_rows=rows(PARENT)
 if len(parent_rows)!=7233 or [int(r['parent_node_index']) for r in parent_rows]!=list(range(7233)):raise ValueError('parent330 node ledger row contract changed')
 parent_by_id={int(r['parent_node_index']):r for r in parent_rows}
 cm=rows(ASSET/'volume/combined-node-map.csv')
 coarse_parent_to_role_row={r:{} for r in ('I','D')}
 coarse_joint={}
 for row in cm:
  role=row['registered_surface_layer']
  if role not in ('I','D'):continue
  joint=int(row['joint_node_id']);role_row=int(row['registered_surface_vertex_row']);parent_id=int(row['source_parent_node_row'])
  value=(role,role_row,parent_id)
  if joint in coarse_joint and coarse_joint[joint]!=value:raise ValueError(f'conflicting duplicate 545 joint node id {joint}')
  coarse_joint[joint]=value
  if parent_id>=0:
   if parent_id in coarse_parent_to_role_row[role] and coarse_parent_to_role_row[role][parent_id]!=role_row:raise ValueError(f'conflicting parent330 node mapping in coarse {role}: {parent_id}')
   coarse_parent_to_role_row[role][parent_id]=role_row
 for role,points in (('I',ip),('D',dp)):
  for v,a,pid,local,tid in role_ring[role]:
   src=parent_by_id[pid]
   srcp=np.asarray([float(src[f'current_{k}_m']) for k in 'xyz'],dtype=np.float32)
   parent_current_delta[role].append(float(np.linalg.norm(np.asarray(points[v],dtype=np.float64)-srcp.astype(np.float64))))
   if coarse_parent_to_role_row[role].get(pid)!=local:raise ValueError(f'{role} ring parent row does not map to same 545 role row')
 # Ensure the actual 545 registered role positions retain those same parent rows.
 for role in ('I','D'):
  reg=rows(ASSET/f'registered/registered-{role}.csv')
  if len(reg)!=2497:raise ValueError('545 registered role row count changed')
  role_offset=ROLE_OFFSET[role]
  for _,_,pid,local,_ in role_ring[role]:
   rr=reg[local]
   coarsep=np.asarray([float(rr['w'+k]) for k in 'xyz'],dtype=np.float32)
   sourcep=np.asarray([float(parent_by_id[pid][f'current_{k}_m']) for k in 'xyz'],dtype=np.float32)
   if not np.array_equal(coarsep,sourcep):raise ValueError(f'{role} 545 source row {local} differs from source330 current F32')

 # Convert the exact 545 fat headwall facets through joint-node -> immutable
 # parent330 node row -> fine boundary vertex. No nearest-row lookup is used.
 rim_rows=[r for r in rows(ASSET/'volume/facet-ancestry.csv') if r['source_partition']=='fat' and r['face_source_role']=='donor_fat_ID_headwall']
 if len(rim_rows)!=90:raise ValueError(f'expected 90 registered fat headwall facets, got {len(rim_rows)}')
 rim_faces=[];rim_parent=[];rim_source_rows=[]
 for x in rim_rows:
  out=[]
  for k in range(3):
   joint=int(x[f'joint_node{k}'])
   if joint not in coarse_joint:raise ValueError(f'headwall joint node {joint} missing from combined map')
   role,role_row,pid=coarse_joint[joint]
   if role not in ('I','D') or pid<0:raise ValueError(f'headwall node lacks I/D parent330 row: {joint} {role} {pid}')
   if pid not in fine_by_parent[role]:raise ValueError(f'headwall parent330 row {pid} is not on fine {role} boundary')
   out.append(fine_by_parent[role][pid] + (len(ip) if role=='D' else 0))
  if len(set(out))!=3:raise ValueError('degenerate mapped headwall triangle')
  rim_faces.append(out);rim_parent.append([int(coarse_joint[int(x[f'joint_node{k}'])][2]) for k in range(3)]);rim_source_rows.append(int(x['face_source_row']))
 rim_faces=np.asarray(rim_faces,dtype=np.int64)
 # Every fine ring edge must be represented once by the mapped rim on each role.
 for role,off,bound in (('I',0,ibound),('D',len(ip),dbound)):
  got=collections.Counter()
  for tri in rim_faces:
   tri=tuple(map(int,tri))
   role_nodes=[v for v in tri if (v>=off if role=='D' else v<len(ip))]
   role_nodes=[v-off for v in role_nodes]
   if len(role_nodes)==2:got[tuple(sorted(role_nodes))]+=1
  if set(got)!=bound or any(n!=1 for n in got.values()):raise ValueError(f'{role} rim edges do not exactly match the fine open boundary')
 # Cross-role pairing is in the 545 shared role-local row domain; source330 IDs stay role-specific.
 source_edges={}
 for role,bound in (('I',ibound),('D',dbound)):
  inv={v:local for v,_,pid,local,_ in role_ring[role]}
  source_edges[role]={tuple(sorted((inv[a],inv[b]))) for a,b in bound}
 if source_edges['I']!=source_edges['D']:raise ValueError('fine I/D head loops do not share the same paired 545 role-row edge set')
 np.savez_compressed(OUT/'fine-rim-source-map.npz',I_fine_vertex_rows=np.asarray([x[0] for x in role_ring['I']],np.int64),I_anchor_rows=np.asarray([x[1] for x in role_ring['I']],np.int64),I_parent330_node_rows=np.asarray([x[2] for x in role_ring['I']],np.int64),I_coarse545_role_rows=np.asarray([x[3] for x in role_ring['I']],np.int64),D_fine_vertex_rows=np.asarray([x[0] for x in role_ring['D']],np.int64),D_anchor_rows=np.asarray([x[1] for x in role_ring['D']],np.int64),D_parent330_node_rows=np.asarray([x[2] for x in role_ring['D']],np.int64),D_coarse545_role_rows=np.asarray([x[3] for x in role_ring['D']],np.int64),fat_headwall_source_face_rows=np.asarray(rim_source_rows,np.int64),fat_headwall_parent330_node_rows=np.asarray(rim_parent,np.int64),fine_fat_rim_faces=np.asarray(rim_faces,np.int64))
 progress('source330-paired-rim-mapped',I_ring_vertices=45,D_ring_vertices=45,headwall_triangles=90,paired_545_role_row_edge_set=True)

 # Compose the actual fine fat boundary using the fine I outer surface, reversed
 # fine D inner surface, and the source-authenticated 90-triangle rim.
 points=np.concatenate((ip.astype(np.float32),dp.astype(np.float32)),axis=0)
 reversed_d=dff[:,[0,2,1]]+len(ip)
 base_faces=np.vstack((iff,reversed_d))
 rim_source_records,_=loadmod('remesh635',LOADER).load_exact_owner(),None
 pred=rim_source_records
 def check_indexed_point_location_fixture():
  vertices=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
  faces=np.asarray([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],dtype=np.int64)
  records,_=pred.float32_surface_records(vertices,faces,source_seam_groups=())
  prepared=pred.prepare_point_location(records)
  expected={'inside':'inside','outside':'outside','boundary':'boundary'}
  queries={'inside':np.asarray([.1,.1,.1],np.float32),'outside':np.asarray([2,2,2],np.float32),'boundary':np.asarray([0,0,0],np.float32)}
  results={}
  for name,point in queries.items():
   key=pred.float32_point_lattice_key(point)
   baseline=pred.point_location(key,records)['location']
   indexed=pred.point_location_prepared_indexed(key,prepared)['location']
   if baseline!=expected[name] or indexed!=baseline:
    raise RuntimeError(f'indexed exact point-location fixture mismatch {name}: baseline={baseline} indexed={indexed}')
   results[name]={'baseline':baseline,'indexed':indexed}
  return {'status':'PASS','results':results}
 indexed_point_location_fixture=check_indexed_point_location_fixture()
 if sha(pred.__file__)!=EXACT_OWNER_SHA:raise ValueError('exact predicate owner hash changed')
 base_records,_=pred.float32_surface_records(points,base_faces,source_seam_groups=())
 rim_records,_=pred.float32_surface_records(points,rim_faces,source_seam_groups=())
 def query_report(queries,index,*,same_surface):
  def candidate_pairs():
   for query,other in pred._query_surface_aabb(queries,index):yield query,other
  return pred._audit_record_pairs(candidate_pairs(),same_surface=same_surface)
 base_index=pred._prepare_surface_aabb(base_records)
 progress('exact-rim-to-complete-I-and-reversed-D',base_faces=len(base_faces),rim_faces=len(rim_faces))
 rim_base=query_report(rim_records,base_index,same_surface=True)
 rim_self=pred._audit_record_pairs(pred._aabb_candidate_pairs(rim_records,rim_records,same_surface=True),same_surface=True)
 if rim_base['count'] or rim_self['count']:
  np.savez_compressed(OUT/'rim-exact-witnesses.npz',rim_base_pairs=np.asarray(rim_base['triangle_pairs'],np.int64).reshape(-1,2),rim_self_pairs=np.asarray(rim_self['triangle_pairs'],np.int64).reshape(-1,2))
  raise ValueError(f'exact rim surface gate failed: rim/base={rim_base["count"]}, rim/self={rim_self["count"]}')
 progress('exact-rim-surface-audits-pass',rim_base_pairs=rim_base['count'],rim_self_pairs=rim_self['count'],rim_base_allowed_shared_pairs=rim_base['allowed_shared_vertex_or_edge_pairs'],rim_self_allowed_shared_pairs=rim_self['allowed_shared_vertex_or_edge_pairs'])
 partial={'schema':'numi.paired-fine-rim-surface-stage.v1','status':'PASS_EXACT_RIM_SURFACE_AUDITS','fine_I_sha256':sha(I_PATH),'fine_D_sha256':sha(D_PATH),'source330_nodes_sha256':sha(PARENT),'545_facet_ancestry_sha256':sha(ASSET/'volume/facet-ancestry.csv'),'545_combined_node_map_sha256':sha(ASSET/'volume/combined-node-map.csv'),'rim_faces':len(rim_faces),'exact_owner_sha256':sha(pred.__file__),'rim_vs_complete_fine_I_and_reversed_D':{'exact_pairs':rim_base['count'],'aabb_candidates':rim_base['aabb_candidate_pairs'],'allowed_shared_pairs':rim_base['allowed_shared_vertex_or_edge_pairs']},'rim_self':{'exact_pairs':rim_self['count'],'aabb_candidates':rim_self['aabb_candidate_pairs'],'allowed_shared_pairs':rim_self['allowed_shared_vertex_or_edge_pairs']},'physical_admission':False,'runtime_created':0,'accepted_physical_steps':0}
 (OUT/'exact-rim-surface-stage.json').write_text(json.dumps(partial,indent=2,sort_keys=True)+'\n')
 # Check the full closed fat material boundary, including edge orientation.
 def check_orientation(rim):
  full=np.vstack((base_faces,rim))
  ok,edges,counts,balance=oriented_closed(full)
  return full,ok,edges,counts,balance
 full_faces,closed,edge_keys,edge_counts,edge_balance=check_orientation(rim_faces)
 source_orientation='as_registered_545'
 if not closed:
  flipped=rim_faces[:,[0,2,1]]
  alt,altclosed,_,_,_=check_orientation(flipped)
  if altclosed:
   full_faces=alt;rim_faces=flipped;closed=True;source_orientation='reversed_545_headwall_winding_to_close_fine_boundary'
 if not closed:raise ValueError('fine fat boundary is not a closed consistently oriented 2-manifold after source rim mapping')
 np.savez_compressed(OUT/'fine-fat-rim-oriented-faces.npz',source_headwall_face_rows=np.asarray(rim_source_rows,np.int64),rim_orientation_applied=np.asarray(source_orientation),oriented_rim_faces=np.asarray(rim_faces,np.int64))
 # Identify true face-connected components and their oriented signed volumes.
 _,edge_counts,edge_balance=boundary_edges(full_faces)
 if not np.all(edge_counts==2) or not np.all(edge_balance==0):raise ValueError('closed/oriented edge incidence failed')
 labels=connected_faces(full_faces,edge_counts)
 ncomp=int(labels.max())+1
 used=np.unique(full_faces)
 if len(used)!=len(np.unique(used)):raise ValueError('unreachable')
 # Unreferenced D storage rows are excluded from all component/solid calculations.
 origin=points[used].astype(np.float64).mean(axis=0)
 tri=points[full_faces].astype(np.float64)-origin
 terms=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2]))/6.0
 comp_volumes=[math.fsum(map(float,terms[labels==i])) for i in range(ncomp)]
 records,_=pred.float32_surface_records(points,full_faces,source_seam_groups=())
 comp_rows=[[] for _ in range(ncomp)]
 comp_vertices=[set() for _ in range(ncomp)]
 for i,label in enumerate(labels):
  comp_rows[int(label)].append(records[i])
  comp_vertices[int(label)].update(map(int,full_faces[i]))
 if any(len(v)==0 for v in comp_vertices):raise ValueError('empty face component')
 # Ask the existing 427 manifold owner to reject any edge-closed but
 # nonmanifold vertex contact before using exact component point-location.
 sys.path.insert(0,'/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/authoring-deps')
 import manifold3d as mf
 used=np.unique(full_faces);inverse=np.full(len(points),-1,dtype=np.int64);inverse[used]=np.arange(len(used))
 compact_points=np.ascontiguousarray(points[used],dtype=np.float64);compact_faces=np.ascontiguousarray(inverse[full_faces],dtype=np.uint64)
 fat=mf.Manifold(mf.Mesh64(vert_properties=compact_points,tri_verts=compact_faces))
 if fat.status()!=mf.Error.NoError:raise ValueError(f'427 Manifold solid owner rejected fine fat boundary: {fat.status()}')
 signed_total=math.fsum(comp_volumes);fat_volume=float(fat.volume());closure_error=fat_volume-signed_total
 if abs(closure_error)>max(1e-12,abs(signed_total)*1e-8):raise ValueError(f'427 solid volume disagrees with oriented component volume: {closure_error}')
 np.savez_compressed(OUT/'fine-fat-closed-boundary.npz',points_world_m_f64=compact_points,faces=np.asarray(compact_faces,np.uint32),fine_I_point_count=np.asarray(len(ip),np.int64),fine_D_point_count=np.asarray(len(dp),np.int64),fine_I_faces=np.asarray(iff,np.int64),reversed_fine_D_faces=np.asarray(reversed_d,np.int64),oriented_rim_faces=np.asarray(rim_faces,np.int64))
 containment=[]
 component_bounds=[]
 for rows_for_component in comp_rows:
  lo=tuple(min(row[1][axis] for row in rows_for_component) for axis in range(3))
  hi=tuple(max(row[2][axis] for row in rows_for_component) for axis in range(3))
  component_bounds.append((lo,hi))
 prepared_components={}
 def prepared_component(j):
  if j not in prepared_components:
   rows_for_component=comp_rows[j]
   limit=pred.MAX_PREPARED_POINT_LOCATION_FACES
   snapshots=[pred.prepare_point_location(rows_for_component[k:k+limit]) for k in range(0,len(rows_for_component),limit)]
   if not snapshots or len(snapshots)>pred.MAX_POINT_LOCATION_CHUNKS:
    raise ValueError(f'component {j} exceeds bounded indexed point-location chunks: faces={len(rows_for_component)} chunks={len(snapshots)}')
   prepared_components[j]=tuple(snapshots)
   progress('exact-component-index-prepared',component=j,faces=len(rows_for_component),chunks=len(snapshots))
  return prepared_components[j]
 progress('exact-component-containment-start',components=ncomp,component_face_counts=[len(rows) for rows in comp_rows])
 for i in range(ncomp):
  states=None;sample_used=None;boundary_trials=0
  for sample in sorted(comp_vertices[i]):
   p=pred.float32_point_lattice_key(points[sample]);trial=[];boundary=False
   for j in range(ncomp):
    if i==j:continue
    lo,hi=component_bounds[j]
    if any(p[axis]<lo[axis] or p[axis]>hi[axis] for axis in range(3)):
     trial.append({'component':j,'location':'outside','crossings':0,'method':'exact_lattice_component_aabb_separation'})
     continue
    snapshots=prepared_component(j)
    if len(snapshots)==1:
     loc=pred.point_location_prepared_indexed(p,snapshots[0])
    else:
     loc=pred.point_location_prepared_indexed_chunks(p,snapshots)
    if loc['location']=='boundary':boundary=True;break
    if loc['location'] not in ('inside','outside'):raise ValueError(f'indeterminate exact indexed containment {i} versus {j}: {loc}')
    trial.append({'component':j,'location':loc['location'],'crossings':loc['crossings'],'method':'exact_prepared_indexed_point_location'})
   if boundary:boundary_trials+=1;continue
   states=trial;sample_used=sample;break
  if states is None:raise ValueError(f'no nonboundary representative vertex for component {i}; boundary_samples={boundary_trials}')
  depth=sum(s['location']=='inside' for s in states)
  expected_sign=1 if depth%2==0 else -1
  actual_sign=1 if comp_volumes[i]>0 else -1
  containment.append({'component':i,'faces':int(np.count_nonzero(labels==i)),'used_vertices':len(comp_vertices[i]),'signed_volume_m3':comp_volumes[i],'nesting_depth':depth,'expected_orientation_sign':expected_sign,'actual_orientation_sign':actual_sign,'orientation_matches_nesting':expected_sign==actual_sign,'point_location_sample_vertex':sample_used,'boundary_samples_skipped':boundary_trials,'against_other_components':states})
  progress('exact-component-classified',component=i,nesting_depth=depth,orientation_matches=expected_sign==actual_sign)
 if not all(r['orientation_matches_nesting'] for r in containment):raise ValueError('component orientation disagrees with exact nesting parity')
 partial.update({'status':'PASS_CLOSED_ORIENTED_FINE_FAT_BOUNDARY','closed_boundary_faces':len(full_faces),'used_vertices':len(used),'component_count':ncomp,'component_signed_volumes_m3':comp_volumes,'component_containment_and_orientation':containment,'signed_volume_m3':signed_total,'manifold_status':str(fat.status()),'manifold_volume_m3':fat_volume,'volume_closure_error_m3':closure_error,'closed_boundary_npz_sha256':sha(OUT/'fine-fat-closed-boundary.npz'),'closed_rim_faces_npz_sha256':sha(OUT/'fine-fat-rim-oriented-faces.npz')})
 (OUT/'closed-fat-boundary-stage.json').write_text(json.dumps(partial,indent=2,sort_keys=True)+'\n')
 progress('exact-closed-fat-boundary-pass',components=ncomp,used_vertices=len(used),unused_D_storage_vertices=len(dp)-len(np.unique(full_faces[full_faces>=len(ip)]-len(ip))),signed_volume_m3=math.fsum(comp_volumes),component_signs=[1 if v>0 else -1 for v in comp_volumes])

 # Exact rim-vs-all185 surface checks, in both the source-order native Float32
 # replay and the original Float64 common-lattice bank operands.
 F32_SHA='e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5'
 MANIFEST_SHA='0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c'
 if sha(F32_BONES)!=F32_SHA or sha(MANIFEST)!=MANIFEST_SHA:raise ValueError('bone input source pins changed')
 with np.load(F32_BONES,allow_pickle=False) as z:
  bpoints=np.asarray(z['points_world_f32']);bfaces=np.asarray(z['faces'],np.int64);offsets=np.asarray(z['bank_face_offsets'],np.int64);bbody=np.asarray(z['bank_body'],np.int64);bbone=np.asarray(z['bank_bone'],np.int64)
 if len(offsets)!=186 or len(bbody)!=185 or len(bbone)!=185 or offsets[0]!=0 or offsets[-1]!=len(bfaces):raise ValueError('native F32 185-bank row map mismatch')
 frecords,_=pred.float32_surface_records(bpoints,bfaces,source_seam_groups=())
 fidx=pred._prepare_surface_aabb(frecords)
 progress('exact-rim-vs-185-native-F32-bones',rim_faces=len(rim_faces),bone_faces=len(bfaces),banks=185)
 f32_report=query_report(rim_records,fidx,same_surface=False)
 f32_pairs=np.asarray(f32_report['triangle_pairs'],np.int64).reshape(-1,2)
 f32_counts=[]
 if len(f32_pairs):
  bankidx=np.searchsorted(offsets,f32_pairs[:,1],side='right')-1
  f32_counts=[{'bank_index':int(k),'body':int(bbody[k]),'bone':int(bbone[k]),'pairs':int(np.count_nonzero(bankidx==k))} for k in np.unique(bankidx)]
 operands=json.loads(MANIFEST.read_text())['operands']
 if len(operands)!=185:raise ValueError('F64 bone manifest incomplete')
 bonehelper=loadmod('bonebarrier635',BONE_HELPER)
 f64_pairs=[];f64_counts=[];f64_aabb=0
 ridx=pred._prepare_surface_aabb(rim_records)
 progress('exact-rim-vs-185-original-F64-bones',rim_faces=len(rim_faces),banks=185)
 for oi,op in enumerate(operands):
  path=Path(op['operand_path'])
  if sha(path)!=op['operand_sha256']:raise ValueError(f'F64 bank input changed: {path}')
  with np.load(path,allow_pickle=False) as z:xx=z['points_world_m_f64'];ff=np.asarray(z['faces_f64'],np.int64)
  if len(ff)!=int(op['face_count']):raise ValueError('F64 bank face count changed')
  keys=tuple(bonehelper.exact_common_lattice_key(v) for v in xx)
  brecords=pred._records(keys,ff);bidx=pred._prepare_surface_aabb(brecords)
  q=query_report(rim_records,bidx,same_surface=False)
  f64_aabb+=q['aabb_candidate_pairs']
  if q['count']:
   f64_counts.append({'manifest_index':op['index'],'body':op['body'],'bone':op['bone'],'pairs':q['count']})
   f64_pairs.extend((int(op['index']),int(a),int(b)) for a,b in q['triangle_pairs'])
  if (oi+1)%25==0:progress('F64 bank queries complete',banks_complete=oi+1,contact_pairs=len(f64_pairs))
 f64_pairs_arr=np.asarray(f64_pairs,np.int64).reshape(-1,3)
 np.savez_compressed(OUT/'rim-bone-exact-witnesses.npz',native_F32_rim_bone_rows=f32_pairs,original_F64_bank_rim_bone_rows=f64_pairs_arr)
 if f32_report['count'] or len(f64_pairs_arr):raise ValueError('rim has exact intersections with one or more all185 bone operands; retained witnesses')
 progress('exact-rim-all185-bone-pass',F32_pairs=f32_report['count'],F64_pairs=len(f64_pairs_arr),F32_AABB=f32_report['aabb_candidate_pairs'],F64_AABB=f64_aabb)

 # Existing 427 solid owner diagnostic, using the authenticated closed fine fat
 # surface, not independent temporary caps. Per-bank volumes are not union sums.
 manifest_rows=json.loads(MANIFEST.read_text())['operands'];boolean_rows=[]
 progress('427-solid-fat-bone-intersections',fat_volume_m3=fat_volume,banks=185)
 for oi,op in enumerate(manifest_rows):
  path=Path(op['operand_path'])
  with np.load(path,allow_pickle=False) as z:vv=np.asarray(z['points_world_m_f64'],np.float64);ff=np.asarray(z['faces_f64'],np.uint64)
  bank=mf.Manifold(mf.Mesh64(vert_properties=np.ascontiguousarray(vv),tri_verts=np.ascontiguousarray(ff)))
  if bank.status()!=mf.Error.NoError:raise ValueError(f'427 bone operand invalid index={op["index"]}: {bank.status()}')
  inter=fat^bank
  if inter.status()!=mf.Error.NoError:raise ValueError(f'427 fat/bone boolean failed index={op["index"]}: {inter.status()}')
  vol=float(inter.volume())
  boolean_rows.append({'manifest_index':int(op['index']),'body':int(op['body']),'bone':int(op['bone']),'intersection_m3':vol,'positive_intersection':vol>0.0})
  if (oi+1)%25==0:progress('427 bone booleans complete',banks_complete=oi+1,positive_pairs=sum(r['positive_intersection'] for r in boolean_rows))
 result={
  'schema':'numi.paired-fine-headwall-rim-audit.v1','status':'PASS_FINE_PAIRED_RIM_GEOMETRY_DIAGNOSTIC',
  'scope':'source330-authenticated 545 paired headwall facets mapped to current fine I/D boundary rows; exact rim/self/bone surface checks; oriented closed fine fat material boundary and 427-style solid-volume/bone intersection diagnostic. Not native runtime or physical admission.',
  'physical_admission':False,'runtime_created':0,'accepted_physical_steps':0,
  'input_pins':pins,'source_roles':anchor_info,
  'fine_surfaces':{'I':{'path':str(I_PATH),'sha256':sha(I_PATH),'points':len(ip),'faces':len(iff),'open_boundary_edges':len(ibound),'boundary_loop_sizes':iloops},'D':{'path':str(D_PATH),'sha256':sha(D_PATH),'points':len(dp),'faces':len(dff),'open_boundary_edges':len(dbound),'boundary_loop_sizes':dloops}},
  'source330_pairing':{'parent_nodes':len(parent_rows),'role_offsets':ROLE_OFFSET,'I_and_D_paired_545_role_row_edge_sets_equal':True,'coarse545_role_rows_are_parent330_source_rows':True,'fine_boundary_F32_matches_anchor_bits':True,'coarse545_role_rows_match_source330_current_F32':True,'fine_to_source330_current_residual_m_by_role':{r:{'max':max(parent_current_delta[r]),'p95':float(np.quantile(parent_current_delta[r],.95)),'min':min(parent_current_delta[r])} for r in ('I','D')},'fine_source_residual_is_reported_not_used_as_a_new_acceptance_tolerance':True,'mapped_fat_headwall_faces':len(rim_faces),'source_face_rows':rim_source_rows,'all_mapped_faces_use_authenticated_I_D_boundary_nodes':True,'rim_orientation':source_orientation},
  'exact_rim_surface_audits':{'owner_path':str(EXACT_OWNER),'owner_sha256':sha(pred.__file__),'rim_vs_complete_fine_I_and_reversed_D':{'pairs':rim_base['count'],'AABB_candidates':rim_base['aabb_candidate_pairs'],'allowed_shared_edge_or_vertex_pairs':rim_base['allowed_shared_vertex_or_edge_pairs']},'rim_self':{'pairs':rim_self['count'],'AABB_candidates':rim_self['aabb_candidate_pairs'],'allowed_shared_edge_or_vertex_pairs':rim_self['allowed_shared_vertex_or_edge_pairs']}},
  'closed_fat_boundary':{'faces':len(full_faces),'oriented_edge_closed':True,'edge_count_two_and_orientation_balance_zero':True,'face_connected_components':containment,'unused_D_storage_vertices_ignored':int(len(dp)-len(np.unique(full_faces[full_faces>=len(ip)]-len(ip)))),'signed_component_volume_sum_m3':signed_total,'manifold_status':str(fat.status()),'manifold_volume_m3':fat_volume,'volume_closure_error_m3':closure_error},
  'rim_all185_bone_surface_audits':{'native_F32':{'banks':185,'faces':len(bfaces),'exact_pairs':f32_report['count'],'AABB_candidates':f32_report['aabb_candidate_pairs'],'contact_banks':f32_counts},'original_F64':{'banks':185,'exact_pairs':len(f64_pairs_arr),'AABB_candidates':f64_aabb,'contact_banks':f64_counts}},
  '427_fat_bone_intersection_diagnostic':{'status':'COMPLETED_DIAGNOSTIC_ONLY','fat_volume_m3':fat_volume,'banks':boolean_rows,'positive_bank_pairs':sum(r['positive_intersection'] for r in boolean_rows),'note':'Per-bank intersections can overlap and must not be summed as a union; this Boolean diagnostic does not establish native contact, mass mapping, thickness, mechanics, or physical admission.'},
  'component_label_fixture':component_fixture,'indexed_point_location_fixture':indexed_point_location_fixture,'provenance':{'coarse_632_report_sha256':sha(P/'paired-rim-006-632/run-001/report.json'),'source330_nodes_sha256':sha(PARENT),'545_facet_ancestry_sha256':sha(ASSET/'volume/facet-ancestry.csv'),'545_combined_node_map_sha256':sha(ASSET/'volume/combined-node-map.csv'),'fine_map_npz_sha256':sha(OUT/'fine-rim-source-map.npz'),'F32_bone_replay_sha256':sha(F32_BONES),'F64_bone_manifest_sha256':sha(MANIFEST),'exact_owner_sha256':sha(EXACT_OWNER)},
  'elapsed_s':time.monotonic()-START
 }
 (OUT/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':result['status'],'fine_fat_volume_m3':fat_volume,'component_count':ncomp,'rim_surface_pairs':rim_base['count'],'rim_self_pairs':rim_self['count'],'F32_bone_pairs':f32_report['count'],'F64_bone_pairs':len(f64_pairs_arr),'positive_bone_intersections':result['427_fat_bone_intersection_diagnostic']['positive_bank_pairs'],'report_sha256':sha(OUT/'report.json'),'elapsed_s':result['elapsed_s']},indent=2),flush=True)

if __name__=='__main__':
 try:
  main()
 except BaseException as e:
  if OUT.exists():
   failure={'status':'FAILED_OR_INCOMPLETE','error':repr(e),'physical_admission':False,'runtime_created':0,'accepted_physical_steps':0}
   with (OUT/'failure.json').open('w') as f:json.dump(failure,f,indent=2)
  raise
