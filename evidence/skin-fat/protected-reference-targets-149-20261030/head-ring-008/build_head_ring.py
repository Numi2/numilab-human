from pathlib import Path
import numpy as np,json,hashlib,csv
from scipy.optimize import linear_sum_assignment
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
B=P/'geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union'
basep=P/'reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz'
paths={'O':P/'surface-transfer-checkpoint-392/payload/O/candidate.npz','I':B/'dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz','D':P/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/full-native-self-028/current-collar-plan-032/engine-20-ops-one-sided624-run-006/candidate-f32-linked-remesh-typed.npz'}
out=P/'protected-149-current-surface-crosswalk-001/head-ring-135-root-008';out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def meshboundary(f):
 edges=np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1)
 e,c=np.unique(edges,axis=0,return_counts=True)
 assert np.all(c<=2)
 boundary=e[c==1];v,deg=np.unique(boundary,return_counts=True)
 assert np.all(deg==2)
 return boundary,v
z=np.load(basep,allow_pickle=False);p=z['points_current_world_m_f32'];pins=z['pinned_volume_node_rows'];parent=z['parent_volume_node_row']
rows=[];gates={}
for role,path in paths.items():
 a=np.load(path,allow_pickle=False);q=a['points_world_m'];f=a['faces']
 fine_edges,fine_v=meshboundary(f)
 coarse_edges,coarse_v=meshboundary(z['coarse_boundary_faces_'+role])
 assert len(fine_v)==45 and len(coarse_v)==45,(role,len(fine_v),len(coarse_v))
 old=[int(x) for x in pins if x in set(map(int,coarse_v))]
 assert len(old)==45 and set(old)==set(map(int,coarse_v))
 distances=np.linalg.norm(p[old,None,:].astype(float)-q[fine_v][None,:,:].astype(float),axis=2)
 i,j=linear_sum_assignment(distances)
 assert np.array_equal(i,np.arange(45))
 vals=distances[i,j]
 assert vals.max()<=7e-6,(role,vals.max())
 # Each source node has a unique match within the bounded tolerance.
 assert np.all(np.sum(distances<=7e-6,axis=1)==1),(role,'ambiguous')
 mapping={int(s):int(fine_v[k]) for s,k in zip(old,j)}
 mapped_edges={tuple(sorted((mapping[int(x)],mapping[int(y)]))) for x,y in coarse_edges}
 assert mapped_edges==set(map(tuple,fine_edges.tolist())),(role,'edge mismatch')
 # Explicitly preserve both source eye/head boundary cycles; no caps are authored.
 adj={int(x):[] for x in fine_v}
 for x,y in fine_edges:adj[int(x)].append(int(y));adj[int(y)].append(int(x))
 left=set(adj);cycles=[]
 while left:
  todo=[next(iter(left))];seen=set()
  while todo:
   u=todo.pop()
   if u in seen:continue
   seen.add(u);todo.extend(adj[u])
  left-=seen;cycles.append(len(seen))
 assert sorted(cycles)==[21,24],cycles
 for oldrow,k in zip(old,j):
  v=int(fine_v[k]);res=q[v].astype(float)-p[oldrow].astype(float)
  rows.append({'canonical_protected_row':int(np.flatnonzero(pins==oldrow)[0]),'base_volume_node_row':oldrow,'source330_parent_node_row':int(parent[oldrow]),'role':role,'current_surface_vertex_row':v,'current_x_m':float(q[v,0]),'current_y_m':float(q[v,1]),'current_z_m':float(q[v,2]),'source_to_current_residual_x_m':float(res[0]),'source_to_current_residual_y_m':float(res[1]),'source_to_current_residual_z_m':float(res[2]),'source_to_current_distance_m':float(np.linalg.norm(res)),'surface_sha256':sha(path),'mapping_kind':'bounded_unique_head_boundary_vertex_with_exact_edge_isomorphism'})
 gates[role]={'source_path':str(path),'sha256':sha(path),'boundary_vertices':45,'boundary_edges':45,'boundary_cycles':sorted(cycles),'max_source_to_current_distance_m':float(vals.max()),'unique_within_7um':True,'complete_edge_set_isomorphism':True}
rows.sort(key=lambda r:r['canonical_protected_row'])
head_indices={r['canonical_protected_row'] for r in rows}
assert len(rows)==135 and len(head_indices)==135
other_indices=sorted(set(range(len(pins)))-head_indices)
assert len(other_indices)==14
canonical_d_rows=[790,1333,804,1719,1790,1557,1478,752,1842,1831,1591,1104,692,1987]
expected_source_nodes={4822+r for r in canonical_d_rows}
actual_source_nodes={int(parent[pins[i]]) for i in other_indices}
assert actual_source_nodes==expected_source_nodes,(actual_source_nodes,expected_source_nodes)
csvp=out/'head-ring-135-targets.csv'
with csvp.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
receipt={'base_npz':str(basep),'base_sha256':sha(basep),'canonical_protected_rows':149,'head_rows_covered':135,'remaining_D_attachment_rows':14,'remaining_D_attachment_canonical_protected_rows':other_indices,'gates':gates,'csv_sha256':sha(csvp),'physical_admission':False,'runtime':0,'limitations':['Bounded geometric and complete boundary-graph correspondence; this does not replace historical source330 tet registration in the native importer.','These are declared open head boundary loops. No caps or anatomical closure are invented.','Output targets are not yet volume nodes; local retet and owner checks remain mandatory.']}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
