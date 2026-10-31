import unittest,numpy as np
from compose_fixed_boundary_volume_patch import compose,oriented_boundary
def orient(p,t):
 t=np.asarray(t,np.int64);u=p[t[:,1:]].astype(float)-p[t[:,:1]].astype(float)
 d=np.einsum('ij,ij->i',u[:,0],np.cross(u[:,1],u[:,2]))
 for i in np.flatnonzero(d<0):t[i,[0,1]]=t[i,[1,0]]
 return t
class Tests(unittest.TestCase):
 def setUp(self):
  q=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[2**-20,2**-20,2**-20]],np.float32);p=np.vstack((q,q+[3,0,0])).astype(np.float32)
  fan=np.array([[4,1,2,3],[0,4,2,3],[0,1,4,3],[0,1,2,4]])
  t=orient(p,np.vstack((fan,fan+5)));original=orient(p,[[0,1,2,3],[5,6,7,8]])
  def volume(t):return dict(points_world_m=p.copy(),points_world_m_f32=p.copy(),tetrahedra=t,material_indices=np.full(len(t),16,np.int32),desired_boundary_faces=oriented_boundary(t)[:,:3])
  self.base=volume(t);self.patch=volume(np.vstack((t[4:],original[:1])));self.current=volume(np.vstack((t[:4],original[1:])))
  self.offset=np.arange(6);self.rows=np.array([4,5,6,7,0,1,2,3]);self.offset=np.array([0,1,2,3,4,8])
 def test_disjoint_geometry_material_and_lineage(self):
  out,w=compose(self.base,self.patch,self.current,self.offset,self.rows)
  self.assertEqual(len(out['tetrahedra']),2);np.testing.assert_array_equal(out['replacement_parent_cell_rows'],np.arange(4));np.testing.assert_array_equal(oriented_boundary(out['tetrahedra']),oriented_boundary(self.current['tetrahedra']))
  np.testing.assert_array_equal(out['points_world_m'].view('u4'),self.base['points_world_m'].view('u4'));self.assertTrue(np.all(out['material_indices']==16))
 def test_overlap_rejected(self):
  with self.assertRaisesRegex(ValueError,'overlaps'):compose(self.base,self.patch,self.patch,self.offset,self.rows)
 def test_coordinate_alias_rejected(self):
  v=self.current.copy();v['points_world_m_f32']=v['points_world_m_f32'].copy();v['points_world_m_f32'][0,0]=1
  with self.assertRaisesRegex(ValueError,'alias'):compose(self.base,self.patch,v,self.offset,self.rows)
 def test_boundary_change_rejected(self):
  v=self.patch.copy();v['tetrahedra']=v['tetrahedra'][:-1];v['material_indices']=v['material_indices'][:-1]
  with self.assertRaisesRegex(ValueError,'boundary'):compose(self.base,v,self.current,np.arange(len(v['tetrahedra'])+1),np.arange(len(v['tetrahedra'])))
 def test_parent_identity_rejected(self):
  bad=self.rows.copy();bad[-1]=99
  with self.assertRaisesRegex(ValueError,'parent'):compose(self.base,self.patch,self.current,self.offset,bad)
 def test_material_identity_rejected(self):
  v=self.current.copy();v['material_indices']=v['material_indices'].copy();v['material_indices'][0]=15
  with self.assertRaisesRegex(ValueError,'overlaps'):compose(self.base,self.patch,v,self.offset,self.rows)
 def test_nonpositive_cell_rejected(self):
  v=self.patch.copy();v['tetrahedra']=v['tetrahedra'].copy();v['tetrahedra'][-1,[0,1]]=v['tetrahedra'][-1,[1,0]]
  with self.assertRaises(ValueError):compose(self.base,v,self.current,self.offset,self.rows)
 def test_empty_cell_domain_rejected(self):
  v=self.patch.copy();v['tetrahedra']=v['tetrahedra'][:0];v['material_indices']=v['material_indices'][:0]
  with self.assertRaisesRegex(ValueError,'cell shape'):compose(self.base,v,self.current,self.offset,self.rows)
 def test_positive_corner_permutation_of_common_cell(self):
  patch=self.patch.copy();patch['tetrahedra']=patch['tetrahedra'].copy()
  patch['tetrahedra'][0]=patch['tetrahedra'][0,[1,2,0,3]]
  out,_=compose(self.base,patch,self.current,self.offset,self.rows)
  self.assertEqual(len(out['tetrahedra']),2)
if __name__=='__main__':unittest.main()
