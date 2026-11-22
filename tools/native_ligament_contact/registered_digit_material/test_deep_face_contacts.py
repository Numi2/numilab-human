import unittest
from fractions import Fraction
from trace_deep_face_contacts import piercings

class DeepFaceTest(unittest.TestCase):
    def test_shared_corner_can_still_pierce(self):
        xyz=[(0,0,0),(2,0,0),(0,2,0),(1,1,-1),(0,1,1)]
        xyz={i:tuple(map(Fraction,q)) for i,q in enumerate(xyz)}
        result=piercings([[0,1,2],[0,3,4]],xyz)
        self.assertTrue(any(v['edge']==[3,4] and v['t']==.5 for v in result))

    def test_only_common_corner(self):
        xyz=[(0,0,0),(2,0,0),(0,2,0),(-1,1,1),(-1,2,1)]
        xyz={i:tuple(map(Fraction,q)) for i,q in enumerate(xyz)}
        self.assertEqual(piercings([[0,1,2],[0,3,4]],xyz),[])
