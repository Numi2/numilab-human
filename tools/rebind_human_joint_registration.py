import sys
"""Rebind certified node registrations by immutable role/raw node identity, not nearest geometry."""
from pathlib import Path
import csv,gzip,json,hashlib,time,importlib.util
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
C=(Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path('/Users/numi/Work/human-integrated-gaps-20261104/joint-boundary-rebuild-64c'))/'compiler'
old=H/'current-native-1024-root/current-433-1037'
oldhand=H/'current-role-p1-support-938-1025-active-source-map/actual-001/217-handoff'
newhand=C/'handoff938'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
pins={}
def verify(p,expected):
 actual=sha(p)
 if actual!=expected:raise ValueError(('pin mismatch',str(p),expected,actual))
 pins[str(p)] = actual
def csvrows(p):
 with (gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open()) as f:yield from csv.DictReader(f)
start=time.time();r=load(old/'receipt.json')
for k in ('candidate_npz','candidate_node_map'):verify(r[k]['path'],r[k]['sha256'])
for hand in (oldhand,newhand):
 rr=load(hand/'adapter-receipt.json')
 for name in ('skin-fat-combined-node-map.csv.gz','skin-fat-combined.node'):
  verify(hand/name,rr['outputs'][name]['sha256'])
with np.load(old/'candidate-current-typed-volume.npz') as z:xyz=z['points_reference_world_m_f64'].astype(np.float32)
newxyz=np.loadtxt(newhand/'skin-fat-combined.node',skiprows=1,usecols=(1,2,3),dtype=np.float32)
oldxyz=np.loadtxt(oldhand/'skin-fat-combined.node',skiprows=1,usecols=(1,2,3),dtype=np.float32)
assert np.array_equal(xyz,oldxyz)
rich=list(csvrows(old/'candidate-node-map.csv'))
assert len(rich)==len(xyz)
bykey={}
for x in csvrows(oldhand/'skin-fat-combined-node-map.csv.gz'):
 key=(x['source_partition'],int(x['source_tetgen_node_id']))
 assert key not in bykey
 bykey[key]=int(x['joint_node_id'])-1
for i,x in enumerate(rich):assert int(x['volume_node_row'])==i
out=C/'registration217';out.mkdir(exist_ok=False)
fields=['joint_node_id','source_partition','source_tetgen_node_id','registered_I_source_vertex_row','merge_peer_partition','merge_peer_tetgen_node_id','registered_surface_layer','registered_surface_vertex_row','role','current_x_m','current_y_m','current_z_m','donor_body','parent_tet_row',*[f'parent_bary{k}' for k in range(4)],*[f'parent_current_residual_{a}_m' for a in 'xyz'],'parent_registration_distance_m','geometric_donor_body','anatomical_region_id']
construction=load(C.parent/'report.json');moves={int(op['node']):op for op in construction.get('interior_relocations',[])}
lookup=parent_owner=owner217=None;relocated=[]
if moves:
 checks=load(C.parent/'interior-relocation-checks.json');assert checks['status']=='PASS_INDEPENDENT_INTERIOR_RELOCATION_REPLAY' and checks['nodes']==len(moves)
 verify(C.parent/'report.json',sha(C.parent/'report.json'));verify(C.parent/'interior-relocation-checks.json',sha(C.parent/'interior-relocation-checks.json'))
 source=H/'current-native-1024-root/current-217-1036/register_current_combined_volume_nodes_217_1036.py'
 spec=importlib.util.spec_from_file_location('registration217',source);owner217=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner217)
 capture=H.parent/'native-physical-parent-mass-330/native-source-identity-control-003'
 parent,parent_owner=owner217.read_parent_current(capture/'compiled-parent-cell-mass.csv',source.parent);lookup=owner217.make_lookup(parent,parent_owner)
 verify(source,sha(source))
counts={};seen=set()
with (out/'candidate-node-map.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for raw in csvrows(newhand/'skin-fat-combined-node-map.csv.gz'):
  key=(raw['source_partition'],int(raw['source_tetgen_node_id']))
  oldrow=bykey[key];jid=int(raw['joint_node_id']);newrow=jid-1;reg=rich[oldrow]
  moved=not np.array_equal(xyz[oldrow],newxyz[newrow])
  if moved:
   assert oldrow in moves and np.array_equal(xyz[oldrow].astype(float),moves[oldrow]['old_world_m']) and np.array_equal(newxyz[newrow].astype(float),moves[oldrow]['new_world_m']),('undocumented coordinate change',key)
  role=raw['registered_surface_layer'] or 'interior'
  assert role==reg['role'],('role change',key,role,reg['role'])
  if role!='interior':
   assert int(raw['registered_surface_vertex_row'])==int(reg['current_typed_source_surface_vertex_row']),('source row change',key)
  if moved:
   assert role=='interior',('surface registration changed',key)
   material=15 if key[0]=='skin' else 16
   triple=(material,int(reg['geometric_donor_body']),int(reg['anatomical_region_id']))
   assert triple in lookup,('missing exact source tuple',key,triple)
   best=owner217.nearest_for_tuple(newxyz[newrow].astype(float),triple,lookup,parent_owner,.02,1e-10)
   assert best['distance']<=.02 and tuple(best['tuple'])==triple,('relocation parent outside unchanged bound',key)
   reg=dict(reg);old_parent=int(reg['source_parent_tet_row']);reg['source_parent_tet_row']=str(best['tet'])
   for k in range(4):reg[f'source_parent_bary{k}']=format(float(best['beta'][k]),'.17g')
   for a,k in zip('xyz',range(3)):reg[f'historical_registration_residual_{a}_m']=format(float(best['residual'][k]),'.17g')
   reg['historical_registration_distance_m']=format(best['distance'],'.17g')
   relocated.append(dict(original_joint_node=oldrow,current_joint_node=newrow,source_partition=key[0],source_node_id=key[1],old_parent=old_parent,new_parent=int(best['tet']),semantic_tuple=list(triple),distance_m=float(best['distance'])))
  row=dict(raw);row['role']=role;row['donor_body']=reg['frame_owner_body'];row['parent_tet_row']=reg['source_parent_tet_row']
  for k in range(4):row[f'parent_bary{k}']=reg[f'source_parent_bary{k}']
  for a,k in zip('xyz',range(3)):
   row[f'parent_current_residual_{a}_m']=reg[f'historical_registration_residual_{a}_m'];row[f'current_{a}_m']=format(float(newxyz[newrow,k]),'.17g')
  row['parent_registration_distance_m']=reg['historical_registration_distance_m']
  for k in ('geometric_donor_body','anatomical_region_id'):row[k]=reg[k]
  w.writerow(row);counts[role]=counts.get(role,0)+1;seen.add(newrow)
assert seen==set(range(len(newxyz)))
assert {v['original_joint_node'] for v in relocated}==set(moves)
report={'schema':'numi.h706.current-217-rich-registration-execution.v1','status':'PASS_EXACT_POINT_REGISTRATION_REBIND_REQUIRES_433_VALIDATION','input_hashes':pins,'output':{'path':str(out/'candidate-node-map.csv'),'sha256':sha(out/'candidate-node-map.csv')},'counts':counts,'unique_nodes':len(seen),'unchanged_f32_coordinates':not bool(relocated),'typed_surface_coordinates_and_registration_unchanged':True,'changed_parent_registrations':len(relocated),'interior_relocation_witnesses':relocated,'wall_s':time.time()-start,'physical_admission':False,'native_steps':0}
(out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
