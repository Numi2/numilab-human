"""Asset admission regressions using synthetic tetrahedra, not anatomy qualification."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from numilab_human.resting_passive_interfaces import build_candidate, CARDIAC_IDS
from numilab_human.resting_anatomy_interface_patch import HEADER, RECORD, normals
from numilab_human.resting_pleura_proxy import _parse_payload, _record_content_bytes


class PassiveInterfaceAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ids = [*CARDIAC_IDS, 2, 3, 5, 13, 398]
        self.passive = [2, 3, 5, 13, 398]
        self.v = np.array([[0,0,0], [.01,0,0], [0,.01,0], [0,0,.01]], dtype='<f4')
        self.f = np.array([[0,2,1], [0,1,3], [0,3,2], [1,2,3]], dtype='<i4')
        packed = np.column_stack((self.v, normals(self.v.astype(float), self.f))).astype('<f4')
        records, vv, ff = [], [], []
        for i, sid in enumerate(self.ids):
            records.append(RECORD.pack(20, i*4, 4, i*12, 12, sid, 1, 0))
            vv.append(packed.tobytes())
            ff.append((self.f.reshape(-1)+i*4).astype('<u4').tobytes())
        raw = HEADER.pack(b'NHANAT1\0', 5, len(self.ids), len(self.ids)*4, len(self.ids)*12, 123, b'x'*32)
        raw += b''.join(records+vv+ff)
        self.payload = self.root/'input.nhanatomy'
        self.payload.write_bytes(raw)
        self.sha = hashlib.sha256(raw).hexdigest()
        self.receipt = {'schema':'numi.human.resting-anatomy-receipt.v1',
            'payload':{'sha256':self.sha},
            'functional_bindings':{'anatomy_payload_sha256':self.sha,
                'passive_viscera_geometry_binding':{'stable_ids':self.passive,'upper_anchor_stable_ids':[3,13]}},
            'provenance':{'source_id_map':{'3':{'name':'pancreas','source_member':'FJ1895'},
                                         '13':{'name':'spleen','source_member':'FJ2561'}},
                'cardiac_geometry_binding':{'output_anatomy_payload_sha256':self.sha,
                    'ventricular_wall_binding':{'output_anatomy_payload_sha256':self.sha,'wall_map':{}}}},
            'qualification':{},'mass_geometry_accounting':{'test_only':True}}
        self.candidates = self.root/'candidates'
        self.candidates.mkdir()
        self.audit = {'source_payload_sha256':self.sha,'self':{'3':{'count':0},'13':{'count':0}},
            'pairs':[{'ids':list(pair),'count':0,'audit_complete':True}
                     for pair in sorted({tuple(sorted((a,b))) for a in (3,13) for b in self.passive if a!=b})],
            'candidate_sha256':{}}
        for sid in (3,13):
            path = self.candidates/f'surface-{sid}.npz'
            np.savez(path, vertices=self.v*.995, faces=self.f)
            self.audit['candidate_sha256'][str(sid)] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.derivation = {'input_payload_sha256':self.sha,'separation_margin_m':.00005}

    def run_candidate(self):
        paths=[]
        for filename,value in [('receipt.json',self.receipt),('audit.json',self.audit),('derivation.json',self.derivation)]:
            p=self.root/filename;p.write_text(json.dumps(value));paths.append(p)
        return build_candidate(self.payload,paths[0],self.payload,self.candidates,paths[1],paths[2],self.root/'output')

    def test_scoped_replacement_preserves_every_other_anatomical_record_and_identity(self):
        result=self.run_candidate()
        _,_,before=_parse_payload(self.payload.read_bytes())
        _,_,after=_parse_payload(Path(result['payload']['path']).read_bytes())
        self.assertEqual(set(before),set(after))
        for sid in self.ids:
            equal=_record_content_bytes(before[sid])==_record_content_bytes(after[sid])
            self.assertEqual(equal,sid not in (3,13))
        r=json.loads((self.root/'output/resting-anatomy-receipt.json').read_text())
        cardiac=r['provenance']['cardiac_geometry_binding']
        self.assertEqual(cardiac['ventricular_wall_binding']['output_anatomy_payload_sha256'],result['payload']['sha256'])
        self.assertEqual(r['provenance']['source_id_map']['3']['source_member'],'FJ1895')
        self.assertEqual(r['mass_geometry_accounting'],self.receipt['mass_geometry_accounting'])

    def test_stale_and_incomplete_evidence_fails_before_output(self):
        cases=[]
        bad=copy.deepcopy(self.receipt);bad['payload']['sha256']='0'*64;cases.append(('receipt',bad))
        bad=copy.deepcopy(self.audit);bad['candidate_sha256']['3']='0'*64;cases.append(('audit',bad))
        bad=copy.deepcopy(self.audit);bad['pairs'].pop();cases.append(('audit',bad))
        bad=copy.deepcopy(self.audit);bad['self']['3']['audit_complete']=False;cases.append(('audit',bad))
        bad=copy.deepcopy(self.audit);bad['pairs'][0]['count']=1;cases.append(('audit',bad))
        for field,value in cases:
            previous=getattr(self,field)
            with self.subTest(field=field):
                setattr(self,field,value)
                with self.assertRaises(ValueError):self.run_candidate()
                self.assertFalse((self.root/'output').exists())
                setattr(self,field,previous)

    def test_source_bound_cardiac_refinement_requires_owner_rebinding(self):
        wall=self.receipt['provenance']['cardiac_geometry_binding']['ventricular_wall_binding']
        wall['wall_map']['local_coefficient_refinement']={}
        with self.assertRaisesRegex(ValueError,'cardiac map refinement'):self.run_candidate()
        self.assertFalse((self.root/'output').exists())

    def test_invalid_candidate_cannot_be_admitted_by_a_zero_count_receipt(self):
        path=self.candidates/'surface-3.npz'
        np.savez(path,vertices=self.v,faces=self.f[:-1])
        self.audit['candidate_sha256']['3']=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):self.run_candidate()
        self.assertFalse((self.root/'output').exists())

    def test_unresolved_source_neighbor_stays_explicit(self):
        self.audit['pairs'][0]={'ids':self.audit['pairs'][0]['ids'],'invalid_geometry':'source zero-area face','audit_complete':False}
        self.run_candidate()
        r=json.loads((self.root/'output/resting-anatomy-receipt.json').read_text())
        unresolved=r['provenance']['passive_reference_interfaces']['unresolved_neighbor_checks']
        self.assertEqual(len(unresolved),1)
        self.assertIn('accepted native breathing cycle',r['qualification']['passive_reference_interfaces'])


if __name__=='__main__':unittest.main()
