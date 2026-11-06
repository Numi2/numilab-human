"""Resume must preserve authenticated reference geometry, ancestry and mass rows."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location("skin_fat_boundary",Path(__file__).resolve().parents[1]/"tools/simplify_human_skin_fat_boundary.py")
owner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)

class ResumeTest(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory()
  self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name);self.volume=self.root/"volume.npz";self.mass=self.root/"mass.csv"
  self.points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],np.float32)
  self.labels=np.array([[15,23,23]])
  np.savetxt(self.mass,[[0,1]],delimiter=",",header="child_row,mass_kg",comments="")
  self.write()
 def write(self,**changes):
  fields=dict(points_reference_world_m_f64=self.points.astype(float),volume_tetrahedra=np.array([[0,1,2,3]]),material_index=np.array([15]),geometric_donor_body=np.array([23]),anatomical_region_id=np.array([23]),replacement_parent_offsets=np.array([0,1]),replacement_parent_cell_rows=np.array([0]),source_volume_sha256=np.array("a"*64))
  fields.update(changes);np.savez(self.volume,**fields)
 def load(self,volume_sha=None,mass_sha=None):
  return owner.load_resume_volume(self.volume,volume_sha or owner.sha(self.volume),self.mass,mass_sha or owner.sha(self.mass),self.points,"a"*64,self.labels)
 def test_exact_resume(self):
  cells,labels,parents,mass=self.load()
  np.testing.assert_array_equal(cells,[[0,1,2,3]])
  np.testing.assert_array_equal(labels,self.labels)
  np.testing.assert_array_equal(parents[0],[0])
  np.testing.assert_array_equal(mass,[1.])
 def test_reference_identity_and_coordinates(self):
  for change in [dict(source_volume_sha256=np.array("b"*64)),dict(points_reference_world_m_f64=self.points.astype(float)+.1)]:
   with self.subTest(change=list(change)):
    self.write(**change)
    with self.assertRaises(ValueError):self.load()
 def test_complete_typed_ancestry(self):
  for change in [dict(geometric_donor_body=np.array([24])),dict(replacement_parent_cell_rows=np.array([1])),dict(replacement_parent_offsets=np.array([0,0]),replacement_parent_cell_rows=np.array([],int))]:
   with self.subTest(change=list(change)):
    self.write(**change)
    with self.assertRaises(ValueError):self.load()
 def test_hash_authentication(self):
  with self.assertRaises(ValueError):self.load(volume_sha="b"*64)
  with self.assertRaises(ValueError):self.load(mass_sha="b"*64)
 def test_cell_mass_identity_and_positivity(self):
  for row in [[1,1],[0,0],[0,-1],[0,float("nan")]]:
   with self.subTest(row=row):
    np.savetxt(self.mass,[row],delimiter=",",header="child_row,mass_kg",comments="")
    with self.assertRaises(ValueError):self.load()

class MaterialPartitionLabelTest(unittest.TestCase):
 def test_captured_bonded_skin_fat_donors_remain_distinct(self):
  labels=np.array([[15,95,93],[15,95,93],[16,93,93],[16,93,93]])
  self.assertTrue(owner.uniform_typed_labels_by_material(labels))
  # Production ancestry union is partitioned by material, matching mass routing.
  for material,donor in [(15,95),(16,93)]:
   parents=np.flatnonzero(labels[:,0]==material)
   np.testing.assert_array_equal(labels[parents,1:],[[donor,93],[donor,93]])
 def test_one_material_cannot_mix_donor_or_region(self):
  for labels in [[[15,95,93],[15,93,93]],[[16,93,93],[16,93,95]],
                 [[15,95,93],[16,93,93],[16,99,93]]]:
   with self.subTest(labels=labels):
    self.assertFalse(owner.uniform_typed_labels_by_material(labels))
 def test_invalid_partition_shape_is_rejected(self):
  for labels in [[],[[15,95]],[15,95,93]]:
   self.assertFalse(owner.uniform_typed_labels_by_material(labels))

class ChangedStarMassTest(unittest.TestCase):
 def test_keep_only_neighbor_retains_mass_and_typed_ancestry(self):
  cells=np.array([[0,1,2,3],[0,1,3,4],[0,2,3,4],[1,5,6,7]])
  labels=np.array([[15,10,10],[15,10,10],[15,10,10],[15,20,20]])
  masses=np.array([.9,.9,.9,4.5]);parents=[np.array([i]) for i in range(4)]
  # A different donor touching only the kept vertex cannot block or receive
  # mass from this cavity. The complete star still owns the topology test.
  star=np.arange(4)
  self.assertFalse(owner.uniform_typed_labels_by_material(labels[star]))
  rows=owner.contraction_changed_rows(cells,star,0)
  np.testing.assert_array_equal(rows,[0,1,2])
  self.assertTrue(owner.uniform_typed_labels_by_material(labels[rows]))
  new=cells[rows].copy();new[new==0]=1
  live=~np.any(np.diff(np.sort(new,axis=1),axis=1)==0,axis=1)
  np.testing.assert_array_equal(new[live],[[1,2,3,4]])
  routed=owner.compose_local_cell_mass(masses[rows],labels[rows,0],
   labels[rows[live],0],np.array([.003]),{15:(850.,950.)})
  masses[rows[live]]=routed
  np.testing.assert_allclose(routed,[2.7],rtol=1e-15,atol=0)
  self.assertEqual(masses[3],4.5)
  np.testing.assert_array_equal(parents[3],[3])
  np.testing.assert_array_equal(cells[3],[1,5,6,7])
 def test_direction_and_noncontiguous_star_keep_original_row_ids(self):
  cells=np.array([[8,9,10,11],[0,1,2,3],[1,5,6,7],[0,2,3,4]])
  np.testing.assert_array_equal(owner.contraction_changed_rows(cells,np.array([1,2,3]),1),[1,2])
  np.testing.assert_array_equal(owner.contraction_changed_rows(cells,np.array([1,2,3]),0),[1,3])

class ReferenceCompactionTest(unittest.TestCase):
 def setUp(self):
  self.p=np.array([[9,9,9],[0,0,0],[0,1,0],[1,0,0],[4,4,4],[0,0,1]],np.float32)
  self.t=np.array([[1,2,3,5]]);self.labels=np.array([[16,95,93]])
  self.rows=np.unique(self.t);self.u=np.arange(12).reshape(4,3)*.001
 def match(self,**changes):
  args=dict(p=self.p,t=self.t,labels=self.labels,nodes=self.p[self.rows],globals_=np.arange(10,14),cells=np.array([[10,11,12,13]]),regions=[95],displacements=self.u)
  args.update(changes);return owner.match_reference_deformation(**args)
 def test_exact_active_order_retains_unused_rows_without_coordinate_search(self):
  u,receipt=self.match()
  np.testing.assert_array_equal(u[self.rows],self.u)
  np.testing.assert_array_equal(u[[0,4]],np.zeros((2,3)))
  self.assertEqual(receipt["method"],"exact_sorted_active_source_order")
  self.assertEqual(receipt["unused_rows"],2)
 def test_full_source_order_stays_supported(self):
  u=np.arange(18).reshape(6,3)*.001
  out,receipt=self.match(nodes=self.p,globals_=np.arange(10,16),cells=self.t+10,displacements=u)
  np.testing.assert_array_equal(out,u)
  self.assertEqual(receipt["method"],"exact_full_source_order")
 def test_reference_anchor_labels_do_not_replace_construction_donor(self):
  u,receipt=self.match(regions=[94],require_same_donor=False)
  np.testing.assert_array_equal(u[self.rows],self.u)
  self.assertEqual(receipt["source_vs_capture_label_disagreements"],1)
  self.assertEqual(receipt["label_disagreement_witnesses"][0]["construction_donor"],95)
  self.assertEqual(receipt["label_disagreement_witnesses"][0]["capture_reference_anchor"],94)
 def test_reordered_exact_cells_match_with_donors(self):
  cells=np.array([[1,2,3,5],[2,1,3,5]])
  labels=np.array([[16,95,93],[16,96,93]])
  u,receipt=self.match(t=cells,labels=labels,cells=[[11,10,12,13],[10,11,12,13]],regions=[96,95])
  np.testing.assert_array_equal(u[self.rows],self.u)
  self.assertEqual(receipt["source_vs_capture_label_disagreements"],0)
 def test_wrong_order_topology_donor_or_nonfinite_rejected(self):
  for changes in [dict(nodes=self.p[self.rows[::-1]]),dict(cells=[[10,12,11,13]]),dict(regions=[93]),dict(globals_=[10,11,12,14]),dict(displacements=np.full((4,3),np.nan)),dict(nodes=self.p[:3])]:
   with self.subTest(changes=list(changes)):
    with self.assertRaises(ValueError):self.match(**changes)

class BoundaryContactQualityTest(unittest.TestCase):
 def test_measured_fold_can_improve_worst_cell_despite_log_sum(self):
  old=np.array([1.179095772658468e-5,.001,.001])
  new=np.array([1.2986674107320688e-5,2e-5])
  self.assertTrue(owner.boundary_contact_quality_progress(old,new,1))
 def test_native_floor_and_no_degradation_remain_required(self):
  for new,removed in [([9e-6],1),([1.01e-5],1),([2e-5],0),([float('nan')],1),([],1)]:
   with self.subTest(new=new,removed=removed):
    self.assertFalse(owner.boundary_contact_quality_progress([1.02e-5],new,removed))
 def test_neutral_first_contraction_requires_cell_removal(self):
  self.assertTrue(owner.boundary_contact_quality_progress([1.2e-5,.001],[1.2e-5],1))
  self.assertFalse(owner.boundary_contact_quality_progress([1.2e-5],[1.2e-5],0))
 def test_invalid_original_quality_cannot_be_promoted(self):
  for old in [[],[float('inf')],[9e-6]]:
   self.assertFalse(owner.boundary_contact_quality_progress(old,[2e-5],1))

if __name__=="__main__":unittest.main()
