"""Regression checks for source-bound flat-bed contact preparation."""
import copy
import unittest
import numpy as np
from numilab_human import resting_scene as s
from numilab_human.model import ImportError as HumanImportError

class NativeSkinWitnessTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(core_body_index=3,vertex_index=17,local_point_m=[.1,.2,.3],region="ulna_l")]
        self.poses=[dict(body_index=3,position_m=[1.,2.,3.],quaternion_xyzw=[0.,0.,0.,1.])]
        self.points={17:np.array([1.14,2.2,3.3],dtype=np.float32)}

    def test_stale_contact_refits_existing_identity_in_float32(self):
        original=copy.deepcopy(self.rows)
        rows,detail=s.refit_native_skin_witnesses(self.rows,self.points,self.poses)
        self.assertEqual(self.rows,original)
        self.assertEqual((rows[0]["core_body_index"],rows[0]["vertex_index"]),(3,17))
        self.assertEqual(rows[0]["region"],"ulna_l")
        self.assertTrue(detail[0]["changed_local_point"])
        self.assertGreater(detail[0]["lever_arm_change_m"],.039)
        self.assertLessEqual(detail[0]["new_actual_skin_fit_error_m"],2e-6)
        self.assertEqual(rows[0]["world_witness_m"][2],0.)

    def test_matching_witness_retains_exact_float32_local_bytes(self):
        old=np.asarray(self.rows[0]["local_point_m"],dtype=np.float32)
        self.points[17]=np.asarray(self.poses[0]["position_m"])+old.astype(float)
        rows,detail=s.refit_native_skin_witnesses(self.rows,self.points,self.poses)
        self.assertEqual(np.asarray(rows[0]["local_point_m"],dtype=np.float32).tobytes(),old.tobytes())
        self.assertFalse(detail[0]["changed_local_point"])

    def test_float32_quaternion_inverse_matches_existing_rotation_owner(self):
        q=np.array([-.7009092642998509,0.,0.,.7132504491541817],dtype=np.float32)
        self.poses[0]["quaternion_xyzw"]=q.tolist()
        desired=np.array([.14,-.82,.3],dtype=np.float32)
        self.points[17]=np.asarray(self.poses[0]["position_m"])+s._rotation_xyzw(q)@desired.astype(float)
        rows,detail=s.refit_native_skin_witnesses(self.rows,self.points,self.poses)
        self.assertLessEqual(detail[0]["new_actual_skin_fit_error_m"],2e-6)
        np.testing.assert_array_equal(np.asarray(rows[0]["local_point_m"],dtype=np.float32),desired)

    def test_absent_source_slot_is_rejected_instead_of_projected(self):
        with self.assertRaisesRegex(HumanImportError,"absent from authenticated"):
            s.refit_native_skin_witnesses(self.rows,{18:self.points[17]},self.poses)

    def test_duplicate_source_and_body_ids_are_rejected(self):
        with self.assertRaisesRegex(HumanImportError,"duplicate contact source"):
            s.refit_native_skin_witnesses(self.rows*2,self.points,self.poses)
        with self.assertRaisesRegex(HumanImportError,"duplicate accepted"):
            s.refit_native_skin_witnesses(self.rows,self.points,self.poses*2)

    def test_nonfinite_or_nonunit_inputs_cannot_create_contacts(self):
        for field,value in (("position_m",[float("nan"),0.,0.]),
                            ("quaternion_xyzw",[0.,0.,0.,2.])):
            with self.subTest(field=field):
                poses=copy.deepcopy(self.poses);poses[0][field]=value
                with self.assertRaises(HumanImportError):
                    s.refit_native_skin_witnesses(self.rows,self.points,poses)
        with self.assertRaises(HumanImportError):
            s.refit_native_skin_witnesses(self.rows,{17:[float("inf"),0.,0.]},self.poses)

    def test_existing_contact_header_and_source_geometry_identity_retained(self):
        rigid=dict(body_count=4,source_sha256="ab"*32)
        oldrows=[{**self.rows[0],"world_witness_m":[1.,2.,0.]}]
        rows,_=s.refit_native_skin_witnesses(self.rows,self.points,self.poses)
        before=s.encode_nhcnt1(rigid,oldrows,friction=1.)
        after=s.encode_nhcnt1(rigid,rows,friction=1.)
        size=s.NHCNT_HEADER.size
        self.assertEqual(before[:size],after[:size])
        self.assertEqual(s.NHCNT_RECORD.unpack(before[size:])[:2],
                         s.NHCNT_RECORD.unpack(after[size:])[:2])
        self.assertNotEqual(before[size:],after[size:])

if __name__=="__main__":unittest.main()

