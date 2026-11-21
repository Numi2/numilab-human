"""Reject a moved rib, a changed rest frame, a missing anchor and changed cardiac bytes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

owner = Path(__file__).resolve().parents[2] / 'src/numilab_human/respiratory_bone_rebind.py'
spec = importlib.util.spec_from_file_location('respiratory_bone_rebind', owner)
bindings = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bindings)

class RespiratoryBindingTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.base = Path(folder.name)
        self.anatomy = self.base / 'anatomy'
        self.anatomy.write_bytes(b'thorax')
        self.anatomy_sha = bindings.file_hash(self.anatomy)
        self.ids = list(range(48, 74))
        self.records = {sid: (sid,) for sid in self.ids + [8, 9]}
        self.changed = [dict(stable_id=sid, member_id=str(sid), core_body=sid) for sid in (8, 9)]
        self.banks = [({}, 'old', self.records), ({'source': {'connected_girdle_parent': {'bone_sha256': 'old', 'changed_anchors': self.changed}}}, 'new', self.records)]
        self.move_rib = self.move_frame = False
        self.bound = dict(bones_payload_sha256='old', anatomy_payload_sha256=self.anatomy_sha,
                          rib_bone_stable_ids=self.ids[:24], sternum_bone_stable_ids=self.ids[24:])
        for sid in range(48, 51):
            self.bound[f'intercostal_313_rib{sid}_motion_attachment'] = dict(
                bones_payload_sha256='old', anatomy_payload_sha256=self.anatomy_sha,
                existing_rib_channel=dict(stable_id=sid, body_index=sid, source_member_id=str(sid), source_member_sha256=str(sid)))
        self.receipt = self.base / 'receipt.json'

    def bank(self, data, sid, body):
        self.assertEqual(body, sid)
        anchor = dict(core_body_index=sid, source_record_index=sid, member_id=str(sid), member_sha256=str(sid),
                      myosim_body={8:'clavicle_r',9:'clavicle_l'}.get(sid,'thorax'))
        new = data[1] == 'new'
        return (anchor, (sid, int(new and self.move_frame)), np.array([int(new and (sid in (8, 9) or (sid == 49 and self.move_rib)))], dtype='<f4'), np.array([[0, 1, 2]]))

    def run_binding(self):
        self.receipt.write_text(json.dumps(dict(functional_bindings=self.bound, provenance=dict(bones_payload_sha256='old'))))
        with patch.object(bindings,'read_material_banks',side_effect=self.banks), patch.object(bindings,'material_bank',self.bank):
            return bindings.rebind(self.base/'old',self.base/'new',self.anatomy,self.receipt)

    def test_complete_thorax(self):
        _, proof = self.run_binding()
        self.assertEqual(proof['identical_respiratory_banks'], self.ids)
        self.assertEqual(len(proof['material_attachments']), 3)

    def test_moved_rib(self):
        self.move_rib = True
        with self.assertRaisesRegex(ValueError,'Unexpected geometry'):
            self.run_binding()

    def test_changed_material_frame(self):
        self.move_frame = True
        with self.assertRaisesRegex(ValueError,'rest frame'):
            self.run_binding()

    def test_missing_attachment(self):
        del self.bound['intercostal_313_rib49_motion_attachment']
        with self.assertRaisesRegex(ValueError,'inventory'):
            self.run_binding()

    def test_cardiac_mismatch(self):
        aux = self.base/'map'
        aux.write_bytes(b'changed')
        definition = {'path': 'map','sha256':hashlib.sha256(b'original').hexdigest()}
        doc = {'provenance':{'cardiac_geometry_binding':{'common_field':dict.fromkeys(('map','domain_boxes','polynomials'),definition)}}}
        with self.assertRaisesRegex(ValueError,'Cardiac sidecar identity'):
            bindings.write_bundle(doc,{},self.receipt,self.base/'out',self.base/'proof')

if __name__ == '__main__':
    unittest.main()
