import unittest
import numpy as np
from topology_check import analyse

class TopologyGuardTests(unittest.TestCase):
 def sphere(self):
  return np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32),np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],np.int64)
 def test_closed_sphere_has_zero_genus(self):
  p,f=self.sphere();a=analyse(p,f,'sphere');self.assertEqual(len(a['components']),1);self.assertEqual(a['components'][0]['genus_from_chi_and_boundary'],0);self.assertEqual(a['edge_incidence_gt2_count'],0)
 def test_closed_torus_is_rejected_as_sphere_topology(self):
  n,m=24,12;p=[];f=[]
  for i in range(n):
   u=2*np.pi*i/n
   for j in range(m):
    v=2*np.pi*j/m;p.append([(2+.5*np.cos(v))*np.cos(u),(2+.5*np.cos(v))*np.sin(u),.5*np.sin(v)])
  for i in range(n):
   for j in range(m):
    a=i*m+j;b=((i+1)%n)*m+j;c=((i+1)%n)*m+(j+1)%m;d=i*m+(j+1)%m;f.extend([[a,b,c],[a,c,d]])
  a=analyse(np.array(p,np.float32),np.array(f,np.int64),'torus');self.assertEqual(a['components'][0]['genus_from_chi_and_boundary'],1);self.assertEqual(a['edge_incidence_gt2_count'],0)
 def test_nested_shell_is_separate_component(self):
  p,f=self.sphere();p2=p*.1+.1;a=analyse(np.vstack([p,p2]),np.vstack([f,f[:,[0,2,1]]+4]),'nested');self.assertEqual(len(a['components']),2)
 def test_unused_points_are_counted(self):
  p,f=self.sphere();a=analyse(np.vstack([p,[8,8,8]]),f,'unused');self.assertEqual(a['unused_point_count'],1)
 def test_duplicate_faces_trigger_nonmanifold_inventory(self):
  p,f=self.sphere();a=analyse(p,np.vstack([f,f[:1]]),'duplicate');self.assertGreater(a['edge_incidence_gt2_count'],0)
if __name__=='__main__':unittest.main()
