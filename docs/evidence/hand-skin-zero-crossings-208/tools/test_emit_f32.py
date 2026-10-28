import unittest
import numpy as np
from emit_f32 import emit
class EmissionTests(unittest.TestCase):
 def test_exact_source_vertices_do_not_move(self):
  p=np.array([[.25,.25,.25],[.251,.25,.25],[.25,.251,.25]],dtype=np.float32);f=np.array([[0,1,2]])
  y,g,r=emit(p.astype(float),f,p)
  self.assertEqual({tuple(v.view(np.uint32)) for v in p},{tuple(v.view(np.uint32)) for v in y})
  np.testing.assert_allclose(y[g],p[f],atol=0,rtol=0);self.assertEqual(r['nudged_vertices'],[])
 def test_distinct_double_vertices_get_distinct_f32_points(self):
  p=np.array([[.25,.25,.25],[.25+5e-9,.25,.25],[.25,.251,.25],[.251,.25,.25]],dtype=float);f=np.array([[0,1,2],[1,3,2]])
  y,g,r=emit(p,f,p[[0,2,3]].astype(np.float32))
  self.assertEqual(len(np.unique(y,axis=0)),4);self.assertEqual(len(r['nudged_vertices']),1)
  self.assertGreater(np.linalg.norm(np.cross(y[g[0,1]].astype(float)-y[g[0,0]],y[g[0,2]].astype(float)-y[g[0,0]])),0)
  self.assertIn(tuple(p[0].astype(np.float32).view(np.uint32)),{tuple(v.view(np.uint32)) for v in y})
 def test_exact_property_seams_weld_without_extra_geometry(self):
  p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,0]],dtype=float);f=np.array([[0,1,2],[3,2,1]])
  y,g,r=emit(p,f,p[:3].astype(np.float32));self.assertEqual(len(y),3);self.assertEqual(g[0,0],g[1,0]);self.assertFalse(r['nudged_vertices'])
if __name__=='__main__':unittest.main()
