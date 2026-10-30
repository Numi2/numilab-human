import unittest
import numpy as np
import build_native_domain_217 as owner
from prior_face_incidence import prior_face_incidence

def mesh(points, tets):
    t=np.asarray(tets,np.int64).reshape((-1,4))
    return dict(points=np.asarray(points,np.float32).reshape((-1,3)),tetrahedra=t,
                element_ids=np.arange(len(t),dtype=np.int64)*7+11,
                material_indices=np.where(np.arange(len(t))%2,16,15))

class VolumeIndexTests(unittest.TestCase):
    def test_orientation_multiple_negative_rows(self):
        p=[];t=[]
        for k in range(9):
            base=len(p);p.extend(np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])+[k*3,0,0])
            t.append(np.asarray([0,1,2,3])+base)
        for n in (1,2,3,5,9):
            a=np.asarray(t).copy()
            for row in range(n):a[row]=a[row,[1,0,2,3]]
            cp,ct,vol=owner.positive_f32_tetrahedra(p,a,15)
            np.testing.assert_array_equal(ct,t)
            np.testing.assert_allclose(vol,1/6,rtol=0,atol=0)
            np.testing.assert_array_equal(cp,np.asarray(p,np.float32))

    def test_parity_shared_face_both_orientations_and_order(self):
        p=[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]]
        for t in ([[0,1,2,3],[0,2,1,4]],[[0,2,1,3],[0,1,2,4]]):
            m=mesh(p,t);expected=prior_face_incidence(m);actual=owner._tetrahedron_boundary_faces(m)
            self.assertEqual(list(actual.items()),list(expected.items()))
            self.assertEqual(len(actual),len(expected))
            self.assertEqual(actual.single_occurrence_keys(),{k for k,v in expected.items() if len(v)==1})
            self.assertEqual(actual.get((0,1,2)),expected[(0,1,2)])
            self.assertIsNone(actual.get((999,1000,1001)))

    def test_random_cell_order_and_disconnected_parity(self):
        rng=np.random.default_rng(107);p=[];t=[]
        for k in range(80):
            base=len(p);p.extend(rng.normal(size=(4,3))+[4*k,0,0])
            t.append(rng.permutation(4)+base)
        m=mesh(p,np.asarray(t)[rng.permutation(len(t))])
        self.assertEqual(list(owner._tetrahedron_boundary_faces(m).items()),
                         list(prior_face_incidence(m).items()))

    def test_nonmanifold_witness_parity(self):
        m=mesh([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1],[0,0,2]],
               [[0,1,2,3],[0,2,1,4],[0,1,2,5]])
        messages=[]
        for f in (prior_face_incidence,owner._tetrahedron_boundary_faces):
            with self.assertRaises(ValueError) as caught:f(m)
            messages.append(str(caught.exception))
        self.assertEqual(messages[0],messages[1])

    def test_empty(self):
        m=mesh([],[])
        actual=owner._tetrahedron_boundary_faces(m)
        self.assertEqual(len(actual),0);self.assertEqual(list(actual.items()),[])
        self.assertEqual(actual.get((0,1,2),[]),[])

    def test_metadata_and_index_domain_rejected(self):
        for kind in ('index','metadata'):
            m=mesh([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],[[0,1,2,3]])
            if kind=='index':m['tetrahedra'][0,3]=4
            else:m['element_ids']=np.array([],np.int64)
            with self.assertRaises(ValueError):owner._tetrahedron_boundary_faces(m)

    def test_query_is_scratch_free_and_values_are_independent(self):
        m=mesh([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],[[0,1,2,3]])
        actual=owner._tetrahedron_boundary_faces(m)
        a=actual[(0,1,2)];a[0]['oriented_nodes']=(99,99,99)
        self.assertNotEqual(a,actual[(0,1,2)])
        with self.assertRaises(KeyError):actual[(2,1,0)]

if __name__=='__main__':unittest.main()
