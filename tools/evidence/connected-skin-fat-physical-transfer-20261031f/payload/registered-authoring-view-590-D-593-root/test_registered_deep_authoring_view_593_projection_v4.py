import unittest
import numpy as np
from prepare_registered_deep_authoring_view_593_v4 import AdapterError, project_layers, verify_layer3_provenance

class RoleDirectedProjectionControls(unittest.TestCase):
    def fixture(self):
        ids=np.array([62010000,62010001,62010002,100000078,62000001,62010003],dtype=np.int64)
        layers=np.array([3,-1,3,3,3,2],dtype=np.int64)
        c={
          'face_source_original_id':ids.copy(), 'face_source_layer':layers.copy(),
          'face_source_body':np.array([4,4,4,4,4,4],dtype=np.int64),
          'face_source_bone':np.array([8,8,8,8,8,8],dtype=np.int64),
          'face_source_face_id':np.arange(6,dtype=np.int64),
          'face_source_identity0':np.arange(6,dtype=np.int64),
          'face_source_identity1':-np.ones(6,dtype=np.int64),
          'quality_star_patch_union_face_source_original_id':ids.copy(),
          'quality_star_patch_union_face_source_layer':layers.copy(),
          'quality_star_patch_union_face_source_body':np.array([4,4,4,4,4,4],dtype=np.int64),
          'quality_star_patch_union_face_source_bone':np.array([8,8,8,8,8,8],dtype=np.int64),
          'quality_star_patch_union_face_source_face_id':np.arange(6,dtype=np.int64),
          'quality_star_patch_union_face_source_identity0':np.arange(6,dtype=np.int64),
          'quality_star_patch_union_face_source_identity1':-np.ones(6,dtype=np.int64),
        }
        s={
          '62010000':{'role':'D','layer':3,'derived_operation':'connected-D patch','derived_parent_source_ids':[100000078]},
          '62010001':{'role':'D','layer':-1,'derived_operation':'mixed-parent D patch'},
          '62010002':{'role':'D','layer':3},
          '100000078':{'role':'BONE','face_count':10},
          '62000001':{'role':'DERIVED_PATCH','layer':3,'derived_operation':'synthetic helper'},
          '62010003':{'role':'D','layer':2,'derived_operation':'already-D-coded patch'},
        }
        return c,s

    def test_only_derived_role_D_codes_minus1_and3_project_to_authoring2(self):
        c,s=self.fixture(); out,r=project_layers(c,s)
        np.testing.assert_array_equal(out['face_source_layer'],[2,2,3,3,3,2])
        np.testing.assert_array_equal(out['quality_star_patch_union_face_source_layer'],[2,2,3,3,3,2])
        self.assertEqual(r['source_registry_projection_ids'],[62010000,62010001])
        self.assertEqual(r['primary_rows_by_original_source_layer'],{'-1':1,'3':1})
        self.assertEqual(r['csr_rows_by_original_source_layer'],{'-1':1,'3':1})
        self.assertEqual(r['BONE_layer3_primary_rows_unchanged'],1)
        self.assertEqual(r['BONE_layer3_CSR_rows_unchanged'],1)

    def test_layer3_means_bone_parent_provenance_not_D_source_identity(self):
        c,s=self.fixture(); r=verify_layer3_provenance(c,s)
        self.assertEqual(r['derived_D_layer3_ids'],[62010000])
        self.assertEqual(r['distinct_parent_source_ids'],[100000078])
        self.assertTrue(r['all_layer3_parent_source_ids_registered_as_BONE'])
        self.assertTrue(r['all_parent_candidate_primary_tuples_verified_layer3'])
        self.assertIn('3 remains inherited BONE source provenance',r['semantics'])

    def test_layer3_nonderived_D_bone_and_derived_patch_stay_unchanged(self):
        c,s=self.fixture(); out,_=project_layers(c,s)
        for row in (2,3,4):
            self.assertEqual(int(out['face_source_layer'][row]),3)
            self.assertEqual(int(out['quality_star_patch_union_face_source_layer'][row]),3)

    def test_all_nonlayer_source_tuple_arrays_are_bitwise_unchanged(self):
        c,s=self.fixture(); out,_=project_layers(c,s)
        for key in c:
            if not key.endswith('_layer'):
                np.testing.assert_array_equal(out[key],c[key],err_msg=key)

    def test_registry_to_tuple_layer_mismatch_fails_closed(self):
        c,s=self.fixture(); c['face_source_layer'][0]=2
        with self.assertRaisesRegex(AdapterError,'disagree with pinned source-layer codes'):
            project_layers(c,s)

    def test_layer3_D_parent_must_be_registered_BONE(self):
        c,s=self.fixture(); s['100000078']['role']='D'
        with self.assertRaisesRegex(AdapterError,'not registered as BONE'):
            verify_layer3_provenance(c,s)

    def test_face_id_sentinel_and_identity_rows_are_not_rewritten(self):
        c,s=self.fixture();c['face_source_face_id'][1]=-1;c['face_source_identity0'][1]=0;c['face_source_identity1'][1]=-1
        c['quality_star_patch_union_face_source_face_id'][1]=-1;c['quality_star_patch_union_face_source_identity0'][1]=0;c['quality_star_patch_union_face_source_identity1'][1]=-1
        out,_=project_layers(c,s)
        for k in ('face_source_face_id','face_source_identity0','face_source_identity1','quality_star_patch_union_face_source_face_id','quality_star_patch_union_face_source_identity0','quality_star_patch_union_face_source_identity1'):
            np.testing.assert_array_equal(out[k],c[k])

if __name__=='__main__': unittest.main(verbosity=2)
