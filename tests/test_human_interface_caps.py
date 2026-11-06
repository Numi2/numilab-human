"""Exact local interface repair controls, not native Human admission."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import rebuild_human_interface_caps as m

def fixture():
 p=np.array([[0.,0,-1],[-1,0,0],[0,1,2**-16],[1,0,0],[0,-1,2**-16]],np.float32)
 cells=m.positive_cells(p,[[1,2,3,4],[0,1,2,3],[0,1,3,4]])
 return p,cells

def test_exact_cap_flip_removes_one_cell_and_retains_cavity():
 p,t=fixture();r=m.interface_flip(p,t[0],t[1:])
 assert len(r["cells"])==2 and r["coordinates_unchanged"]
 assert r["exact_oriented_cavity_and_volume"]
 assert r["interface_hausdorff_bound_m"]==2**-16
 assert set(r["removed_edge"])=={1,3}
 assert {tuple(sorted(f)) for f in r["new_interface_faces"]}=={(1,2,4),(2,3,4)}
 assert m.unchanged_cavity_boundary(t,r["cells"])
 assert sum(m.exact_determinants(p,t))==sum(m.exact_determinants(p,r["cells"]))

def test_orientation_and_translation_preserve_the_exact_flip():
 p,t=fixture();p=p[:,[1,2,0]]+np.array([1.,2,4],np.float32)
 r=m.interface_flip(p,t[0],t[1:]);assert r["interface_hausdorff_bound_m"]==2**-16

def test_flat_zero_volume_cap_is_rejected():
 p,t=fixture();p[2,2]=0;p[4,2]=0
 with pytest.raises(ValueError):m.interface_flip(p,t[0],t[1:])

def test_negative_old_orientation_is_rejected():
 p,t=fixture();t[0,[2,3]]=t[0,[3,2]]
 with pytest.raises(ValueError):m.interface_flip(p,t[0],t[1:])

def test_nonstar_and_missing_face_neighbors_are_rejected():
 p,t=fixture();p=np.vstack([p,[0.,0.,2.]]).astype(np.float32);t[2,0]=5
 with pytest.raises(ValueError):m.interface_flip(p,t[0],t[1:])

def test_nonemitted_precision_and_out_of_range_are_rejected():
 p,t=fixture();p=p.astype(float);p[2,2]+=1e-20
 with pytest.raises(ValueError):m.interface_flip(p,t[0],t[1:])
 p,t=fixture();t[0,0]=8
 with pytest.raises(ValueError):m.interface_flip(p,t[0],t[1:])


def test_positive_but_overlapping_old_cells_do_not_pass_volume_certificate():
 p,t=fixture();p[0,2]=1;t=m.positive_cells(p,t)
 with pytest.raises(ValueError):m.interface_flip(p,t[0],t[1:])


def test_face_union_composes_both_ancestries_without_dropping_sources():
 p={"faces":np.array([[0,1,2],[0,2,3]])}
 groups=[("quality_star_patch_union_offsets",["quality_star_patch_union_face_source_"+k for k in ["layer","body","bone","face_id","original_id","identity0","identity1"]]),("face_ancestry_offsets",["face_ancestry_source_face_rows","face_ancestry_source_original_ids","face_ancestry_parent_face_rows"])]
 for offset,lanes in groups:
  p[offset]=np.array([0,1,2])
  for lane in lanes:p[lane]=np.array([5,6])
 m.rewrite_face_unions(p,{0:[0,1],1:[0,1]})
 for offset,lanes in groups:
  assert p[offset].tolist()==[0,2,4]
  for lane in lanes:assert p[lane].tolist()==[5,6,5,6]


def test_joint_ancestry_accepts_identical_shared_interface_records():
 def row(j,role,node):return dict(joint_node_id=str(j),source_partition=role,source_tetgen_node_id=str(node),merge_peer_partition="",merge_peer_tetgen_node_id="")
 old=[row(1,"skin",4),row(1,"fat",8),row(2,"skin",9)]
 current=[row(1,"skin",4),row(1,"fat",8),row(2,"skin",9)]
 assert m.registered_joint_to_original_node(old,current,2).tolist()==[0,1]
 current[1]=row(1,"skin",9)
 with pytest.raises(ValueError):m.registered_joint_to_original_node(old,current,2)

def test_joint_ancestry_rejects_missing_nodes_and_role_id_changes():
 def row(j,node):return dict(joint_node_id=str(j),source_partition="skin",source_tetgen_node_id=str(node),merge_peer_partition="",merge_peer_tetgen_node_id="")
 with pytest.raises(ValueError):m.registered_joint_to_original_node([row(1,4)],[row(1,4)],2)
 with pytest.raises(ValueError):m.registered_joint_to_original_node([row(1,4)],[row(1,5)],1)


def test_bounded_mass_uses_existing_halo_reserve_without_homogenizing():
 old=np.array([900.,900.,1000.]);materials=np.array([15,16,15])
 # Skin expands by0.01; fat contracts by0.01. A nearby skin donor keeps
 # total material mass fixed while the other material remains in range.
 newmat=np.array([15,16,15]);volume=np.array([1.01,.99,1.])
 preferred=np.array([900.,900.,1000.])
 out=m.bounded_patch_mass(old,materials,newmat,volume,preferred)
 np.testing.assert_allclose(out,[909.,900.,991.],rtol=0,atol=1e-12)
 for mat in [15,16]:assert out[newmat==mat].sum()==old[materials==mat].sum()
 # A redundant fourth halo cell remains bitwise at its original mass.
 out=m.bounded_patch_mass(np.r_[old,950.],np.r_[materials,15],np.r_[newmat,15],np.r_[volume,1.],np.r_[preferred,950.])
 assert out[-1]==950.

def test_bounded_mass_rejects_an_unfunded_volume_change():
 with pytest.raises(ValueError):m.bounded_patch_mass([900.,900.],[15,16],[15,16],[1.01,.99],[900.,900.])


def test_context_pins_emitted_interface_and_includes_it_in_audit(tmp_path):
 import hashlib
 old=tmp_path/'old';new=tmp_path/'new';new.mkdir()
 config={'candidate_surfaces':{},'changed_roles':['O','D']}
 for role in 'OID':
  (new/f'candidate-{role}-typed-current.npz').write_bytes(role.encode())
  config['candidate_surfaces'][role]={'path':str(old/f'candidate-{role}-typed-current.npz'),'sha256':'stale'}
 result=m.candidate_context(config,old,new)
 assert result['changed_roles']==['O','D','I']
 for role in 'OID':
  assert result['candidate_surfaces'][role]['sha256']==hashlib.sha256(role.encode()).hexdigest()
  assert result['candidate_surfaces'][role]['path']==str(new/f'candidate-{role}-typed-current.npz')
 assert config['candidate_surfaces']['I']['sha256']=='stale'


def paired_fixture():
 p,t=fixture();p2=p+np.array([4.,0,0],np.float32)
 points=np.vstack([p,p2]);cells=np.vstack([t,t+len(p)])
 labels=np.array([[16,23,23],[15,23,23],[15,23,23],
                  [15,23,23],[16,23,23],[16,23,23]])
 import simplify_human_skin_fat_boundary as mesh
 volume=mesh.measure(points,cells)[0]/6
 mass=volume*np.where(labels[:,0]==16,950.,1000.)
 parts=[]
 for base in [0,3]:
  flip=m.interface_flip(points,cells[base],cells[base+1:base+3])
  parts.append(dict(core_rows=[base,base+1,base+2],rest=[],
                    opposite=[base+1,base+2],new=flip['cells']))
 return points,cells,labels,mass,parts


def test_opposite_caps_balance_a_density_saturated_tissue_without_lost_mass():
 p,t,labels,mass,parts=paired_fixture()
 # Removing the first fat cap alone cannot retain its mass below950kg/m3.
 with pytest.raises(ValueError):m.cap_group_mass(p,t,labels,mass,parts[:1])
 rows,cells,newlabels,newmass=m.cap_group_mass(p,t,labels,mass,parts)
 import simplify_human_skin_fat_boundary as mesh
 volume=mesh.measure(p,cells)[0]/6
 for mat,lo,hi in [(15,900.,1100.),(16,850.,950.)]:
  assert newmass[newlabels[:,0]==mat].sum()==pytest.approx(mass[labels[:,0]==mat].sum(),rel=3e-15)
  density=newmass[newlabels[:,0]==mat]/volume[newlabels[:,0]==mat]
  assert np.all(density>=lo-1e-10) and np.all(density<=hi+1e-10)
 assert len(cells)==4 and len(rows)==6
 assert m.unchanged_cavity_boundary(t,cells)


def test_cap_group_rejects_overlap_ownership_and_capacity():
 p,t,labels,mass,parts=paired_fixture()
 with pytest.raises(ValueError,match='overlapping'):m.cap_group_mass(p,t,labels,mass,[parts[0],parts[0]])
 with pytest.raises(ValueError,match='capacity'):m.cap_group_mass(p,t,labels,mass,[])
 with pytest.raises(ValueError,match='capacity'):m.cap_group_mass(p,t,labels,mass,parts+[parts[0]])
 labels[3:,1]=24
 with pytest.raises(ValueError,match='ownership'):m.cap_group_mass(p,t,labels,mass,parts)


def test_cached_cap_volumes_preserve_the_exact_mass_solution():
 p,t,labels,mass,parts=paired_fixture()
 import simplify_human_skin_fat_boundary as mesh
 for part in parts:part['new_volume']=mesh.measure(p,part['new'])[0]/6
 cached=m.cap_group_mass(p,t,labels,mass,parts,cell_volume=mesh.measure(p,t)[0]/6)
 direct=m.cap_group_mass(p,t,labels,mass,parts)
 for a,b in zip(cached,direct):assert np.array_equal(a,b)
