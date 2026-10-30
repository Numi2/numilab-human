import unittest,importlib.util,numpy as np
from pathlib import Path
s=importlib.util.spec_from_file_location("csr",Path(__file__).with_name("canonicalize_microedge_csr_011.py"));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Tests(unittest.TestCase):
 def fixture(self):
  parent={k:np.arange(3,dtype=np.int64) for k in m.FIELDS["O"]}
  current={k:np.arange(5,dtype=np.int64).reshape(-1,1) for k in m.FIELDS["O"]}
  current["points_world_m"]=np.zeros((3,3),np.float32);current["faces"]=np.asarray([[0,1,2]],np.int64)
  return parent,current
 def test_encoding_only(self):
  p,c=self.fixture();o,r=m.canonicalize(p,c,"O")
  self.assertEqual(len(r),4)
  for key in r:self.assertEqual(o[key].shape,(5,));self.assertEqual(o[key].tobytes(),c[key].tobytes())
  self.assertIs(o["faces"],c["faces"])
 def test_unexpected_rank_change_rejected(self):
  p,c=self.fixture();p["unknown"]=np.arange(2);c["unknown"]=np.zeros((2,1))
  with self.assertRaisesRegex(ValueError,"inventory"):m.canonicalize(p,c,"O")
 def test_vector_csr_not_squeezed(self):
  p,c=self.fixture();key=next(iter(m.FIELDS["O"]));c[key]=np.zeros((3,2),np.int64)
  with self.assertRaisesRegex(ValueError,"scalar"):m.canonicalize(p,c,"O")
if __name__=="__main__":unittest.main()
