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

if __name__=="__main__":unittest.main()
