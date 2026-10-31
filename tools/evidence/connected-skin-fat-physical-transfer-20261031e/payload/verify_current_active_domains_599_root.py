"""Replay active-domain preparation on the frozen current598 physical pair."""
from pathlib import Path
import sys,time,json,hashlib
import numpy as np
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'active-domain-compaction-565-root'))
from compact_physical_skin_domains import prepare_paired_active_domains
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def load(p):
 with np.load(p) as z:return {k:z[k] for k in ('points_world_m','tetrahedra','material_indices')}
def quality(p,t):
 x=p[t].astype(np.float64)
 d=np.einsum('ij,ij->i',x[:,1]-x[:,0],np.cross(x[:,2]-x[:,0],x[:,3]-x[:,0]))
 e=np.zeros(len(t))
 for a,b in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)):e=np.maximum(e,np.linalg.norm(x[:,a]-x[:,b],axis=1))
 return int(np.count_nonzero((d/6<=1e-18)|(d/e**3<1e-5))),float(d.min()),float((d/e**3).min())
start=time.monotonic();out=H/'active-domain-compaction-599-root';out.mkdir(exist_ok=False)
cp=H/'checkpoint-current598-root.json';assert sha(cp)=='f5be01b9695bfbb0bee0e9f63967a54c8a9641d0b425a010353e8b282356a2ac'
state=json.loads(cp.read_text())['current_verified_state_20261031']
vol={}
for role in ('Skin','Fat'):
 entry=state[role+'_current'];assert sha(entry['path'])==entry['sha256']
 vol[role]=load(entry['path'])
 assert vol[role]['points_world_m'].dtype==np.float32
 assert np.all(vol[role]['material_indices']==(15 if role=='Skin' else 16))
lookups={role:{tuple(v.view(np.uint32)):i for i,v in enumerate(a['points_world_m'])} for role,a in vol.items()}
assert all(len(lookups[role])==len(a['points_world_m']) for role,a in vol.items())
role_inputs={}
for role,targets in [('O',('Skin',)),('I',('Skin','Fat')),('D',('Fat',))]:
 e=state['source_surfaces'][role];assert sha(e['path'])==e['sha256']
 with np.load(e['path']) as z:
  p=z['points_world_m_f32'];f=z['faces']
  assert np.array_equal(np.asarray(z['points_world_m'],np.float32).view(np.uint32),p.view(np.uint32))
 maps={target:np.asarray([lookups[target].get(tuple(v.view(np.uint32)),-1) for v in p],np.int64) for target in targets}
 assert all(np.all(m>=0) for m in maps.values())
 role_inputs[role]={'points_world_m':p,'faces':f,'volume_node_rows':maps}
r=prepare_paired_active_domains(vol['Skin']['points_world_m'],vol['Skin']['tetrahedra'],
 vol['Fat']['points_world_m'],vol['Fat']['tetrahedra'],role_inputs)
a={};counts={};offset=0;quality_rows={}
for role,v in r['volumes'].items():
 n=len(vol[role]['tetrahedra']);current=r['tetrahedra'][offset:offset+n]
 assert np.array_equal(r['points_world_m'][current].view(np.uint32),
  vol[role]['points_world_m'][vol[role]['tetrahedra']].view(np.uint32))
 before=quality(vol[role]['points_world_m'],vol[role]['tetrahedra'])
 after=quality(r['points_world_m'],current)
 assert before==after and before[0]==state[role+'_current']['native_quality_failures']
 quality_rows[role]={'native_quality_failures':before[0],'min_signed_det6_m3':before[1],
 'min_normalized_det':before[2],'all_cell_corner_bits_and_order_unchanged':True}
 counts[role]={'source_points':len(vol[role]['points_world_m']),'active_points':len(v['points_world_m']),
 'retired_unreferenced_points':int(np.count_nonzero(v['input_node_to_output_node']<0)),'cells':n}
 for key in ('output_node_to_input_node','input_node_to_output_node','output_node_to_joint_node'):a[role+'_'+key]=v[key]
 offset+=n
role_rows={}
for role,v in r['roles'].items():
 for key in ('output_source_vertex_to_input_source_vertex','input_source_vertex_to_output_source_vertex','joint_node_rows'):a[role+'_'+key]=v[key]
 for target,rows in v['volume_node_rows'].items():a[role+'_'+target+'_volume_node_rows']=rows
 original=role_inputs[role]
 assert np.array_equal(v['output_source_vertex_to_input_source_vertex'][v['faces']],original['faces'])
 role_rows[role]={'original_source_points':len(original['points_world_m']),
 'active_role_points':len(v['points_world_m']),'faces':len(v['faces']),
 'all_face_corners_order_source_identity_and_coordinates_unchanged':True}
cross=out/'active-domain-source-crosswalk.npz';np.savez_compressed(cross,**a)
receipt={'status':'PASS_CURRENT598_EXACT_ACTIVE_PHYSICAL_DOMAIN_AND_DERIVED_ROLE_PREPARATION',
 'physical_admission':False,'native_steps':0,'checkpoint':pin(cp),'producer':pin(Path(__file__)),
 'implementation':pin(H/'active-domain-compaction-565-root/compact_physical_skin_domains.py'),
 'controls':pin(H/'active-domain-compaction-565-root/controls-receipt.json'),
 'crosswalk':pin(cross),'volumes':counts,'roles':role_rows,'quality':quality_rows,
 'combined_nodes':len(r['points_world_m']),'combined_cells':len(r['tetrahedra']),
 'only_shared_registered_I_welded':True,'no_coordinate_or_cell_changes':True,
 'cell_mass_material_and_parent_row_order_unchanged':True,
 'mass_allocated_by_this_stage':False,'old_217_writer_limits':{'combined_nodes':300000,'combined_cells':200000},
 'requires_capacity_qualified_current_native_emitter':True,
 'limitations':['This is exact geometry and namespace preparation, not a native import or motion run.',
 'Retired source rows remain recoverable through the complete input/output source crosswalks.',
 'No mass is assigned, no node is constrained, and no native quality threshold changes.',
 'The historical 217 combined writer must not be used with its old fixed capacity limits.'],
 'elapsed_s':time.monotonic()-start}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
