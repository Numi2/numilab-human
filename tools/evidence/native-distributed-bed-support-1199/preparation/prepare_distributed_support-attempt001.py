from pathlib import Path
import sys,json,hashlib,copy
import numpy as np
ROOT=Path('/Users/n/numi-human-retained-delivery-20261009')
OUT=Path(__file__).resolve().parent
OWNER=Path('/Users/n/numi-human-free-apex-two-family-1178')
sys.path.insert(0,str(OWNER/'src'))
from numilab_human import resting_scene as owner
PARENT=ROOT/'contoured-bed-reference-1196/resting-supine-scene-contoured-5mm.manifest.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
scene=json.loads(PARENT.read_text())
rigid_path=Path(scene['source']['rigid']['path'])
skin_path=Path(scene['source']['skin']['path'])
order_path=rigid_path.with_name('myosim-fullbody-reference.manifest.json')
rigid=owner.load_rigid(rigid_path);skin=owner.load_skin(skin_path,rigid)
order=json.loads(order_path.read_text())['core_tree']['body_order']
world,body,error=owner.source_shell_world(rigid,skin)
rotation=owner._rotation_xyzw(np.asarray(scene['pose']['root_delta_quaternion_xyzw']))
pivot=rigid['q'][:3]
posed=owner._row_times_rotation_transpose(world-pivot,rotation)+pivot
posed[:,2]+=scene['pose']['root_translation_xyz_m'][2]-rigid['q'][2]
old=scene['bed']['support_witnesses']
assert len(old)==32 and [x['region'] for x in old[:8]]==['head','thorax','thorax','thorax','pelvis','pelvis','pelvis','pelvis']
selected=copy.deepcopy(old[:8]);used={x['vertex_index'] for x in selected}
groups=[]
for side in ('r','l'):
 hand=[name for name in order if name.endswith('_'+side) and not name.startswith('__') and
       any(x in name for x in ('mc_','thumb_','proxph_','midph','distph','lunate_','scaphoid_','pisiform_','triquetrum_','capitate_','trapezium_','trapezoid_','hamate_'))]
 groups.extend([(f'upper_arm_{side}',[f'humerus_{side}']),
                (f'forearm_{side}',[f'ulna_{side}',f'radius_{side}']),
                (f'rigid_hand_{side}',hand),
                (f'thigh_{side}',[f'femur_{side}',f'patella_{side}']),
                (f'shin_{side}',[f'tibia_{side}']),
                (f'foot_{side}',[f'talus_{side}',f'calcn_{side}',f'toes_{side}'])])
details=[]
for label,names in groups:
 ids=[order.index(name) for name in names]
 candidates=np.flatnonzero(np.isin(body,ids)&(error<=.01))
 assert len(candidates)>=20,(label,len(candidates))
 # Longest posed coordinate span defines two nonoverlapping axial halves.
 # Select one posterior surface seed per half near its axial/lateral centre.
 axis=int(np.argmax(np.ptp(posed[candidates,:],axis=0)))
 assert axis in (0,1),(label,axis)
 half_order=candidates[np.lexsort((candidates,posed[candidates,axis]))]
 halves=np.array_split(half_order,2)
 chosen=[]
 for band,ids_in_half in enumerate(halves):
  dorsal=ids_in_half[posed[ids_in_half,2]<=np.quantile(posed[ids_in_half,2],.2)]
  centre=np.median(posed[ids_in_half,:2],axis=0)
  distance=np.sum((posed[dorsal,:2]-centre)**2,axis=1)
  ranked=dorsal[np.lexsort((dorsal,distance))]
  vertex=next(int(v) for v in ranked if int(v) not in used)
  used.add(vertex);chosen.append(vertex)
  selected.append({'region':f'{label}_axial_half_{band}','vertex_index':vertex,
                   'core_body_index':int(body[vertex]),
                   'dominant_binding_error_m':float(error[vertex]),
                   'source_shell_gap_from_global_min_m':float(posed[vertex,2]-posed[:,2].min()),
                   'region_minimum_world_z_m':float(posed[ids_in_half,2].min())})
 details.append({'region':label,'bodies':names,'body_indices':ids,'source_skin_vertices':len(candidates),
                 'axial_coordinate':axis,'seeds':chosen,'seed_separation_m':float(np.linalg.norm(posed[chosen[0]]-posed[chosen[1]]))})
assert len(selected)==32 and len(used)==32
rows=owner._local_witnesses(rigid,skin,world,selected,rotation,pivot,scene['pose']['root_translation_xyz_m'][2]-rigid['q'][2])
contact=owner.encode_nhcnt1(rigid,rows,friction=scene['bed']['friction'])
# Head/torso/pelvis records are deliberately unchanged; only limb partition seeds change.
old_contact=Path(scene['outputs']['support_contact']['path']).read_bytes()
assert contact[:owner.NHCNT_HEADER.size+8*owner.NHCNT_RECORD.size]==old_contact[:owner.NHCNT_HEADER.size+8*owner.NHCNT_RECORD.size]
contact_path=OUT/'myosim-fullbody-distributed-rigid-digit-support.nhcnt'
assert not contact_path.exists()
contact_path.write_bytes(contact)
scene['outputs']['support_contact'].update(path=str(contact_path),sha256=sha(contact_path),bytes=len(contact))
scene['bed']['support_witnesses']=rows
scene['bed']['contact_scope']='32 source-rest Voronoi regions over the complete registered skin; each accepted GPU step selects its current minimum fixed-bed-gap vertex and full-weight Jacobian. Existing Metal stand is the sole contact force owner.'
scene['bed']['contact_distribution']={'condition':'reference rigid-digit resting support distribution','required_runtime':'rigid digits; released initialization; full registered skin GPU support',
 'algorithm':'Preserve the first eight head/thorax/pelvis seeds. Use two nonoverlapping axial halves for each upper arm, forearm, rigid hand, thigh, shin and foot; within each half select a posterior fifth vertex nearest the projected surface median. Stable vertex-index tie breaking.',
 'purpose':'Provide separated contact regions for proximal and distal limb support using the same 32-region capacity after individual finger DOFs are disabled.',
 'qualification':'Prepared computational support partition; not an equilibrium, reduced drift, anatomy, or clinical result.',
 'parameter_status':'Geometric discretization choices; no measured participant parameters and no force or pose constraints added.',
 'seed_groups':details,'parent_manifest':str(PARENT),'parent_manifest_sha256':sha(PARENT)}
scene['heightfield_provenance']['support_contact']=copy.deepcopy(scene['outputs']['support_contact'])
scene['heightfield_provenance']['source_region_gaps_previous_partition']=scene['heightfield_provenance'].pop('source_region_gaps')
scene['heightfield_provenance']['contact_distribution_generator']={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)}
result=OUT/'resting-supine-scene-distributed-support.manifest.json'
result.write_text(json.dumps(scene,indent=2,sort_keys=True)+'\n')
receipt={'status':'prepared, no native execution','manifest_sha256':sha(result),'support_sha256':sha(contact_path),
 'skin_unchanged':scene['source']['skin']['sha256'],'bed_grid_unchanged':True,'contact_count':32,
 'first_eight_records_exact':True,'all_source_point_fit_error_max_m':max(x['rest_frame_witness_fit_error_m'] for x in rows),
 'groups':details,'source_pins':{str(p):sha(p) for p in [PARENT,rigid_path,skin_path,order_path,Path(owner.__file__),Path(__file__)]}}
(OUT/'preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'manifest':str(result),'manifest_sha256':sha(result),'support_sha256':sha(contact_path),'groups':details}))
