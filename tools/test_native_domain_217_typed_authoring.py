from pathlib import Path
import hashlib,json,tempfile
import numpy as np
import build_native_domain_217 as m
from test_native_domain_217 import save_fixture_npzs

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def run():
 with tempfile.TemporaryDirectory() as td:
  root=Path(td)
  points=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
  for layer,origin,role in [('O',1,0),('I',3,1),('D',5,2)]:
   source=root/(layer+'source.npz')
   np.savez(source,identity0=np.array([17]),identity1=np.array([29]))
   schema={str(origin):{'role':layer,'path':str(source),'sha256':sha(source),
                        'identity_keys':['identity0','identity1']}}
   payload=dict(points_world_m=points,faces=np.array([[0,1,2]],np.int64),
    face_source_layer=np.array([role]),face_source_body=np.array([-1]),face_source_bone=np.array([-1]),
    face_source_face_id=np.array([71]),face_source_original_id=np.array([origin]),
    face_source_identity0=np.array([17]),face_source_identity1=np.array([29]),
    face_source_identity_schema_by_original_id_json=np.array([json.dumps(schema)]))
   candidate=root/(layer+'candidate.npz');np.savez(candidate,**payload)
   result=m.load_surface(candidate,layer)
   assert result['ancestry']==[[71]] and result['ancestry_original_ids']==[[origin]]
   assert result['ancestry_parent_rows']==[[0]] and result['source_face_ids'].tolist()==[71]
   assert result['source_run_original_ids'].tolist()==[origin]
   assert result['ancestry_kind'].tolist()==[7]
   bad={**payload,'face_source_identity0':np.array([18])}
   np.savez(root/'bad-source.npz',**bad)
   try:m.load_surface(root/'bad-source.npz',layer)
   except ValueError:pass
   else:raise AssertionError('invalid typed source accepted')
   bad={**payload,'points_world_m':np.array([[0,0,0],[1,0,0],[2,0,0]],np.float32)}
   np.savez(root/'bad-area.npz',**bad)
   try:m.load_surface(root/'bad-area.npz',layer)
   except ValueError:pass
   else:raise AssertionError('zero-area emitted face accepted')
  op,ip,dp,cp=save_fixture_npzs(root)
  old=np.load(cp);original=m.load_surface(op,'O',old)
  perm=np.array([2,0,1]);inverse=np.argsort(perm)
  new={**original,'points':original['points'][perm],
       'faces':inverse[original['faces']]}
  bound=m.rebind_typed_collar_outer(old,new)
  assert np.array_equal(bound['points_world_m'][:3],new['points'])
  assert np.array_equal(bound['points_world_m'][3:],old['points_world_m'][3:])
  assert np.array_equal(bound['donor_O_parent_node_rows'],old['donor_O_parent_node_rows'])
  assert np.array_equal(bound['donor_I_parent_node_rows'],old['donor_I_parent_node_rows'])
  assert np.array_equal(bound['donor_D_parent_node_rows'],old['donor_D_parent_node_rows'])
  assert np.array_equal(bound['points_world_m'][bound['skin_sidewall_faces']],
                        old['points_world_m'][old['skin_sidewall_faces']])
  assert np.array_equal(bound['points_world_m'][bound['fat_sidewall_faces']],
                        old['points_world_m'][old['fat_sidewall_faces']])
  placement=root/'placement.npz'
  offset=np.array([0.125,-0.25,0.5])
  np.savez(placement,native_placement_rotation_row_major=np.eye(3).reshape(9),
           native_placement_translation_m=offset)
  placed=m.place_verified_donor_collar(old,placement)
  moved={**new,'points':(new['points'].astype(float)+offset).astype(np.float32)}
  bound_moved=m.rebind_typed_collar_outer(placed,moved)
  assert np.array_equal(bound_moved['points_world_m'][bound_moved['skin_sidewall_faces']],
     (old['points_world_m'][old['skin_sidewall_faces']].astype(float)+offset).astype(np.float32))
  np.savez(placement,native_placement_rotation_row_major=np.eye(3).reshape(9)*2,
           native_placement_translation_m=offset)
  try:m.place_verified_donor_collar(old,placement)
  except ValueError:pass
  else:raise AssertionError('nonrigid donor collar placement accepted')
  broken={**new,'points':new['points'].copy()};broken['points'][0,0]+=np.float32(.001)
  try:m.rebind_typed_collar_outer(old,broken)
  except ValueError:pass
  else:raise AssertionError('changed exact collar boundary accepted')
 print(json.dumps(dict(typed_OID_ancestry=True,invalid_source_rejected=True,zero_area_rejected=True,
 exact_donor_wall_reindex=True,changed_boundary_rejected=True,physical_admission=False)))
if __name__=='__main__':run()
