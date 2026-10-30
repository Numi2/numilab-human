from pathlib import Path
import unittest,importlib.util,numpy as np,json,hashlib,time,tempfile
root=Path(__file__).parent;spec=importlib.util.spec_from_file_location('seed056',root/'complete_region_seeds.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Tests(unittest.TestCase):
 def setUp(self):
  self.p=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1],[3,0,0],[4,0,0],[3,1,0],[3,0,1]])
  self.t=np.array([[0,1,2,3],[0,2,1,4],[5,6,7,8]])
 def test_components_are_face_connected(self):
  self.assertEqual(m.components(self.t,[0,1,2]),[[0,1],[2]])
 def test_one_seed_per_disconnected_unowned_region(self):
  s=m.seeds_from_unowned(self.p,self.t,np.array([0,0,0]),15,lambda p:True)
  self.assertEqual(len(s),2);self.assertEqual(sum(len(x['unowned_cell_rows'])for x in s),3)
 def test_owned_cells_are_never_relabelled(self):
  s=m.seeds_from_unowned(self.p,self.t,np.array([15,15,0]),15,lambda p:True)
  self.assertEqual(s[0]['unowned_cell_rows'],[2])
 def test_outside_cell_rejected(self):
  with self.assertRaisesRegex(ValueError,'outside'):m.seeds_from_unowned(self.p,self.t,np.array([0,15,15]),15,lambda p:False)
 def test_collapsed_cell_rejected(self):
  self.p[3]=[0,0,0]
  with self.assertRaisesRegex(ValueError,'collapsed'):m.seeds_from_unowned(self.p,self.t,np.array([0,15,15]),15,lambda p:True)
 def test_poly_preserves_points_facets_holes(self):
  class Owner:
   def validate_poly_sections(self,*args):pass
  source='4 3 0 1\n1 0 0 0 0\n2 1 0 0 0\n3 0 1 0 0\n4 0 0 1 0\n1 1\n1 0 11\n3 1 2 3\n1\n1 .1 .1 .1\n1\n1 .2 .2 .2 15 0\n'
  with tempfile.TemporaryDirectory()as td:
   p=Path(td)/'source.poly';out=Path(td)/'out.poly';p.write_text(source)
   r=m.complete_poly(p,out,[{'point_world_m':[.3,.3,.3],'material':15}],Owner())
   emitted=[x for x in out.read_text().splitlines()if not x.startswith('#')]
   self.assertEqual(emitted[:10],source.splitlines()[:10]);self.assertEqual(r['added_regions'],1);self.assertEqual(emitted[10],'2')
 def test_poly_cannot_cross_tissue_material(self):
  class Owner:
   def validate_poly_sections(self,*args):pass
  with tempfile.TemporaryDirectory()as td:
   p=Path(td)/'source.poly';p.write_text('4 3 0 1\n1 0 0 0 0\n2 1 0 0 0\n3 0 1 0 0\n4 0 0 1 0\n0 1\n0\n1\n1 .2 .2 .2 16 0\n')
   with self.assertRaisesRegex(ValueError,'ownership'):m.complete_poly(p,Path(td)/'out.poly',[{'point_world_m':[.3,.3,.3],'material':15}],Owner())
 def test_unknown_attribute_rejected(self):
  with self.assertRaisesRegex(ValueError,'attribute'):m.seeds_from_unowned(self.p,self.t,np.array([1,15,15]),15,lambda p:True)
if __name__=='__main__':
 start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);result=unittest.TextTestRunner(verbosity=2).run(suite)
 receipt={'status':'PASS'if result.wasSuccessful()else'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source_sha256':hashlib.sha256((root/'complete_region_seeds.py').read_bytes()).hexdigest(),'tests_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'wall_s':time.monotonic()-start}
 (root/'test-receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
 raise SystemExit(not result.wasSuccessful())

