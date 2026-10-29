#!/usr/bin/env python3
from __future__ import annotations
import copy,hashlib,json,sys,unittest
from pathlib import Path
import numpy as np
from native_bank_frames_386 import native_current_world_to_body

class NativeBankFrameTests(unittest.TestCase):
    def record(self,matrix=None):
        if matrix is None:matrix=np.eye(3)
        return {'current':{'position_world_m':[1.,2.,3.],
                'local_to_world_row_major_f32':np.asarray(matrix,np.float32).astype(float).reshape(-1).tolist()}}
    def test_round_trip_uses_actual_inverse(self):
        m=np.array([[0.,-1.00000012,0.],[0.99999994,0.,0.],[0.,0.,1.]],np.float32).astype(float)
        inv,origin,meta=native_current_world_to_body(self.record(m))
        local=np.array([[.14,-.37,.71],[1.,.2,-.8]])
        world=local@m.T+origin
        np.testing.assert_allclose((world-origin)@inv.T,local,rtol=0,atol=5e-16)
        self.assertGreater(float(np.max(np.abs(inv-m.T))),1e-8)
    def test_missing_native_matrix_rejected(self):
        record=self.record();del record['current']['local_to_world_row_major_f32']
        with self.assertRaisesRegex(ValueError,'nine finite'):native_current_world_to_body(record)
    def test_unemitted_precision_rejected(self):
        record=self.record();record['current']['local_to_world_row_major_f32'][0]=1.+1e-12
        with self.assertRaisesRegex(ValueError,'exact emitted Float32'):native_current_world_to_body(record)
    def test_reflection_rejected(self):
        with self.assertRaisesRegex(ValueError,'SO\\(3\\)'):native_current_world_to_body(self.record(np.diag([-1.,1.,1.])))
    def test_nonfinite_or_invalid_matrix_rejected(self):
        for matrix in [np.diag([2.,1.,1.]),np.diag([float('nan'),1.,1.])]:
            with self.assertRaises(ValueError):native_current_world_to_body(self.record(matrix))
    def test_all_current_native_body_frames(self):
        base=Path(__file__).parent
        pose_path=base/'native-physical-parent-mass-330/native-source-identity-control-003/registered-skin-poses-100.json'
        pose=json.loads(pose_path.read_text());frames=pose['all_bodies'];self.assertEqual(len(frames),157)
        maximum_round_trip=0.;maximum_orth=0.
        for index,record in enumerate(frames):
            self.assertEqual(record['body_index'],index)
            inv,origin,meta=native_current_world_to_body(record)
            matrix=np.array(record['current']['local_to_world_row_major_f32']).reshape(3,3)
            maximum_round_trip=max(maximum_round_trip,float(np.max(np.abs(inv@matrix-np.eye(3)))))
            maximum_orth=max(maximum_orth,meta['native_matrix_orthogonality_error'])
        self.assertLess(maximum_round_trip,1e-14)
        print(json.dumps({'native_body_count':len(frames),'maximum_inverse_round_trip_error':maximum_round_trip,
              'maximum_orthogonality_error':maximum_orth,'pose_sha256':hashlib.sha256(pose_path.read_bytes()).hexdigest(),
              'physical_admission':False}))

if __name__=='__main__':unittest.main()
