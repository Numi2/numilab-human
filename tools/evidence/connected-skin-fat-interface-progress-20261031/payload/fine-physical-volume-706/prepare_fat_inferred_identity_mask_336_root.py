#!/usr/bin/env python3
"""Pin all Fat anatomical identities and reserve pending boundary cavities."""
from pathlib import Path
import numpy as np,csv,json,hashlib,subprocess
H=Path(__file__).resolve().parent
PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(f):
 with open(f,'rb') as q:return hashlib.file_digest(q,'sha256').hexdigest()
cur=H/'fat614-compact-cavities-004-budget1-satcap-002/connected-fixed-surface-volume-proposal.npz'
assert sha(cur)=='4602edc3b1db3617ab610c21f91309db10892c0e28ff7d6ba5c381491d13df2c'
out=H/'fat-inferred-identity-mask-336-root';out.mkdir(exist_ok=False)
with np.load(cur) as z:p=z['points_world_m'];t=z['tetrahedra'];ret=z['retained_input_cell_rows'];desired=z['desired_boundary_faces']
source=H/'combined-child-handoff-142/fat.node'
sx=np.loadtxt(source,skiprows=1,usecols=(1,2,3)).astype(np.float32)
regmap=H/'combined-child-handoff-142/registered-node-map-177-143-713-source008/registered-node-map.csv'
assert sha(regmap)=='40646a6c501b89046e0e9836790a78b3303ad8f0033fe5281fc64ebb760fccc3'
seen=set();registered_keys=set();counts={'I':0,'D':0,'inferred_original_volume_interior':0}
with regmap.open() as q:
 for row in csv.DictReader(q):
  if row['source_partition']!='fat':continue
  i=int(row['source_tetgen_node_id'])-1
  assert i not in seen and 0<=i<len(sx);seen.add(i)
  xyz=np.asarray([float(row['current_'+a+'_m']) for a in 'xyz'],np.float32)
  assert np.array_equal(xyz.view(np.uint32),sx[i].view(np.uint32))
  role=row['role']
  if role in ('I','D'):
   assert row['source_domain']=='derived_registered_skin_authoring' and row['source_mapping_kind']=='vertex'
   assert row['registered_surface_layer']==role and int(row['source_vertex_row'])>=0 and int(row['registered_surface_vertex_row'])>=0
   registered_keys.add(xyz.tobytes());counts[role]+=1
  else:
   assert role=='interior' and row['source_domain']=='derived_volume_interior_no_surface_ancestry' and row['source_mapping_kind']=='none'
   assert int(row['source_vertex_row'])==int(row['source_face_row'])==int(row['registered_surface_vertex_row'])==-1 and not row['registered_surface_layer']
   counts['inferred_original_volume_interior']+=1
assert len(seen)==len(sx)
typed=[
 H/'joint-native-OI-ears-279-root/I-batch002-run001/typed-280-002/candidate-I-typed-current.npz',
 H/'native-apex-D-singleton-after-paired-293-root/run-001/typed-001/candidate-D-typed-current.npz']
typedpins={}
for f in typed:
 with np.load(f) as z:
  tp=z['points_world_m_f32']
  assert tp.dtype==np.float32
  registered_keys.update(x.tobytes() for x in tp)
 typedpins[str(f)]=sha(f)
protected=np.asarray([i for i,x in enumerate(p) if x.tobytes() in registered_keys],np.int64)
protected=np.union1d(protected,np.unique(desired))
prep=H/'fat614-compact-cavities-004-anchor-stars/connected-cavity-shells.npz'
with np.load(prep) as z:pending=z['input_cell_rows']
# ret directly maps every unchanged candidate cell to c21b; appended rows are
# component1 replacements. Preserve both all pending cavities and replacements.
assert len(np.unique(ret))==len(ret) and np.all(ret>=0)
excluded=np.r_[np.flatnonzero(np.isin(ret,pending)),np.arange(len(ret),len(t))]
mask=out/'identity-and-pending-cavity-mask.npz'
np.savez_compressed(mask,source_input_sha256=np.asarray(sha(cur)),protected_point_rows=protected,excluded_cell_rows=excluded)
report={'status':'PASS_FAT_REGISTERED_IDENTITY_AND_PENDING_CAVITY_MASK','physical_admission':False,'native_steps':0,
 'identity_counts':counts,'source_input':str(cur),'source_input_sha256':sha(cur),'point_count':len(p),'cell_count':len(t),
 'protected_existing_point_rows':len(protected),'excluded_pending_cell_rows':len(excluded),
 'registered_map':str(regmap),'registered_map_sha256':sha(regmap),'source_node':str(source),'source_node_sha256':sha(source),
 'typed_identity_pins':typedpins,'pending_shells':str(prep),'pending_shells_sha256':sha(prep),
 'retained_cell_row_map_rule':'current4602 retained_input_cell_rows maps exact unchanged c21b cells; pending prep004 rows plus every appended comp1 replacement are protected',
 'eligibility':'all existing historical registered I/D coordinates and all current typed I/D coordinates, complete current volume boundary and desired target vertices are fixed; only unregistered volume-interior nodes may retire; no pending cavity cell may change',
 'mask_sha256':sha(mask),'producer_sha256':sha(Path(__file__))}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
subprocess.run([PY,str(H/'retire_fat_inferred_interior_nodes_337_root.py'),'--fixture','--output',str(out/'controls337')],check=True)
subprocess.run([PY,str(H/'retire_fat_inferred_interior_nodes_337_root.py'),'--input',str(cur),'--input-sha256',sha(cur),'--identity-mask',str(mask),'--identity-report',str(out/'report.json'),'--output',str(out/'fat-run001'),'--seconds','45'],check=True)
