import unittest
import numpy as np
from prepare_registered_deep_authoring_view_593_v3 import project_layers

class DAuthoringProjectionControls(unittest.TestCase):
    def fixture(self):
        c={
          'face_source_original_id':np.array([62010000,62009999,62000001],dtype=np.int64),
          'face_source_layer':np.array([-1,-1,-1],dtype=np.int64),
          'face_source_face_id':np.array([0,1,2],dtype=np.int64),
          'face_source_identity0':np.array([0,0,0],dtype=np.int64),
          'face_source_identity1':np.array([-1,-1,-1],dtype=np.int64),
          'quality_star_patch_union_face_source_original_id':np.array([62010000,62009999,62000001],dtype=np.int64),
          'quality_star_patch_union_face_source_layer':np.array([-1,-1,-1],dtype=np.int64),
          'quality_star_patch_union_face_source_face_id':np.array([0,1,2],dtype=np.int64),
          'quality_star_patch_union_face_source_identity0':np.array([0,0,0],dtype=np.int64),
          'quality_star_patch_union_face_source_identity1':np.array([-1,-1,-1],dtype=np.int64),
        }
        schema={
          '62010000':{'role':'D','layer':-1,'derived_operation':'connected-patch-retriangulation'},
          '62009999':{'role':'D','layer':-1},
          '62000001':{'role':'DERIVED_PATCH','layer':-1,'derived_operation':'synthetic patch'},
        }
        return c,schema
    def test_only_registry_declared_D_derived_sentinels_project(self):
        c,s=self.fixture();out,r=project_layers(c,s)
        np.testing.assert_array_equal(out['face_source_layer'],[2,-1,-1])
        np.testing.assert_array_equal(out['quality_star_patch_union_face_source_layer'],[2,-1,-1])
        self.assertEqual(r['primary_changed_source_ids'],[62010000])
        self.assertEqual(r['csr_changed_source_ids'],[62010000])
        self.assertEqual(r['primary_derived_patch_ids_left_unchanged'],[62000001])
        self.assertEqual(r['csr_derived_patch_ids_left_unchanged'],[62000001])
    def test_face_identity_and_nonlayer_tuples_are_unchanged(self):
        c,s=self.fixture();out,_=project_layers(c,s)
        for key in c:
            if not key.endswith('_layer'):
                np.testing.assert_array_equal(out[key],c[key],err_msg=key)
        self.assertTrue(np.all(out['face_source_face_id'][0:1]==0))
    def test_source_schema_without_layer_sentinel_does_not_project(self):
        c,s=self.fixture();s['62010000']['layer']=2
        out,r=project_layers(c,s)
        np.testing.assert_array_equal(out['face_source_layer'],[-1,-1,-1])
        self.assertEqual(r['primary_changed_count'],0)
    def test_D_derived_schema_requires_operation(self):
        c,s=self.fixture();s['62010000'].pop('derived_operation')
        out,r=project_layers(c,s)
        np.testing.assert_array_equal(out['face_source_layer'],[-1,-1,-1])
        self.assertEqual(r['primary_changed_count'],0)
    def test_empty_registry_projection_is_noop_for_control(self):
        c,s=self.fixture();out,r=project_layers(c,{})
        np.testing.assert_array_equal(out['face_source_layer'],[-1,-1,-1])
        self.assertEqual(r['primary_changed_count'],0)

if __name__=='__main__': unittest.main(verbosity=2)
