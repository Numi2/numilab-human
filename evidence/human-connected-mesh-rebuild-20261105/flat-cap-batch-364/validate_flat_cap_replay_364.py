"""Certify geometric bounds for the actual accepted degree-three cap replay."""
from pathlib import Path
import hashlib,json
import numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
D=W/'flat-cap-batch-rebuild-364/proposal'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def distance(x,tri):
 a,b,c=tri;e=b-a;g=c-a
 uv=np.linalg.solve(np.array([[e@e,e@g],[e@g,g@g]]),np.array([e@(x-a),g@(x-a)]))
 if min(uv)>=0 and uv.sum()<=1:return float(np.linalg.norm(x-a-uv[0]*e-uv[1]*g))
 return float(min(np.linalg.norm(x-y-np.clip((x-y)@(z-y)/((z-y)@(z-y)),0,1)*(z-y)) for y,z in [(a,b),(b,c),(c,a)]))
# Known face-, edge-, and vertex-projection cases.
tri=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
for point,expected in [([.2,.2,1.],1.),([.6,.6,0.],np.sqrt(.02)),([-1.,-1.,0.],np.sqrt(2.))]:
 assert abs(distance(np.array(point),tri)-expected)<1e-14
source=H/'all-quality-937-b5-backtracking-root/proposal/candidate-O-typed-current.npz'
with np.load(source) as z:surface_points=z['points_world_m_f32'].astype(float);sf=z['faces']
with np.load(H/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz') as z:
 p=z['points_reference_world_m_f64'].astype(np.float32).astype(float);m=z['current_typed_role_source_row_to_joint_node_O']
f=m[sf].copy();alive=np.ones(len(f),bool);anc={};records=[];groups=[]
ops=json.loads((D/'report.json').read_text())['operations']
for i,o in enumerate(ops):
 if 'O' not in o['roles']:continue
 drop=o['drop'];keep=o['keep'];fr=np.flatnonzero(alive&np.any(f==drop,axis=1))
 assert fr.tolist()==o['source_face_rows']['O'] and len(fr)==3
 neighbors=np.setdiff1d(f[fr].ravel(),[drop]);assert len(neighbors)==3 and keep in neighbors
 old=f[fr].copy();new=old.copy();new[new==drop]=keep
 live=~np.any(np.diff(np.sort(new,axis=1),axis=1)==0,axis=1)
 assert live.sum()==1 and set(new[live][0])==set(neighbors)
 source_nodes=np.flatnonzero(np.isin(m,[drop,*neighbors]))
 assert np.array_equal(p[m[source_nodes]],surface_points[source_nodes])
 bound=distance(p[drop],p[neighbors])+1e-12
 assert bound<=.0005
 origins=set().union(*(anc.get(int(row),{int(row)}) for row in fr))
 touched=[g for g in groups if g['faces']&origins]
 merged={'faces':origins.copy(),'sum_bound':bound,'operations':[i]}
 for g in touched:
  merged['faces']|=g['faces'];merged['sum_bound']+=g['sum_bound'];merged['operations']+=g['operations'];groups.remove(g)
 groups.append(merged)
 for row in fr:anc[int(row)]=origins
 f[fr]=new;alive[fr[~live]]=False
 records.append({'operation':i,'drop':drop,'keep':keep,'source_face_rows':fr.tolist(),'patch_hausdorff_bound_m':bound})
with np.load(D/'candidate-O-typed-current.npz') as z:
 assert np.array_equal(f[alive],m[z['faces']])
 assert np.array_equal(surface_points,z['points_world_m_f32'].astype(float))
maximum=max(g['sum_bound'] for g in groups)
assert maximum<=.0005
result={'status':'PASS_ACTUAL_CAP_REPLAY_GEOMETRIC_BOUND','input_source':{'path':str(source),'sha256':sha(source)},'proposal_report_sha256':sha(D/'report.json'),'executed_owner_sha256':sha(__file__),'patches':records,'connected_change_components':[{'operations':sorted(g['operations']),'accumulated_hausdorff_bound_m':g['sum_bound']} for g in groups],'global_reference_surface_hausdorff_bound_m':maximum,'gate_m':.0005,'coordinates_unchanged':True,'actual_final_topology_exactly_replayed':True,'physical_admission':False,'proof':'Each old degree3 cap maps affinely to its base by mapping the tip to its closest point in the base. The mapped fan covers the base. Both directed distances are bounded by the tip displacement. Compose bounds for overlapping original-face ancestry; disjoint components use their maximum. The bound applies to reference outer-surface geometry, not dynamic force/strain agreement.'}
(D/'cap-replay-geometric-bound.json').write_text(json.dumps(result,indent=2)+'\n')
print('CAP_REPLAY_PASS',len(records),'caps',len(groups),'components','maximum_bound_m',maximum)

