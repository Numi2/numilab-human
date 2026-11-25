import unittest

import numpy as np

from compare_accepted_tissue_forces import differences, regional_response


class AcceptedForcesTests(unittest.TestCase):
    def test_zero_reference_force_keeps_absolute_delta(self):
        r = np.vstack((np.zeros(3),np.eye(3)))
        ref = np.vstack((r,r+2))
        t = np.array([[0,1,2,3],[4,5,6,7]])
        lab = np.array([15,16])
        params = {15:(10,30),16:(5,20)}
        base = regional_response(ref,ref,t,lab,params)
        pos = ref.copy()
        pos[5,0] += .02
        actual = differences(base,regional_response(ref,pos,t,lab,params))
        self.assertEqual(actual[15]["force_delta_L2_N"],0.)
        self.assertGreater(actual[16]["force_delta_L2_N"],0.)
        self.assertIsNone(actual[16]["relative_force_L2"])
        self.assertGreaterEqual(actual[16]["peak_row"],4)


if __name__ == "__main__":
    unittest.main()
