"""Source-binding admission regressions; synthetic tetrahedra are not anatomy evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from numilab_human.resting_anatomy_interface_patch import HEADER, RECORD, normals, signed_volume
from numilab_human.resting_passive_interfaces import CARDIAC_IDS
from numilab_human.resting_pleura_proxy import _parse_payload, _record_content_bytes
from numilab_human.resting_taenia_reference import MEMBERS, build_candidate


class TaeniaAdmissionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        self.passive = [3,13,*MEMBERS]; self.ids = [*CARDIAC_IDS,*self.passive]
        self.v = np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]],dtype='<f4')
        self.f = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]],dtype='<i4')
        packed = np.column_stack((self.v,normals(self.v.astype(float),self.f))).astype('<f4')
        records=[];vv=[];ff=[]
        for i,sid in enumerate(self.ids):
            records.append(RECORD.pack(20,4*i,4,12*i,12,sid,1,0));vv.append(packed.tobytes());ff.append((self.f.reshape(-1)+4*i).astype('<u4').tobytes())
        raw=HEADER.pack(b'NHANAT1\0',5,len(self.ids),4*len(self.ids),12*len(self.ids),123,b'x'*32)+b''.join(records+vv+ff)
        self.payload=self.root/'source.nhanatomy';self.payload.write_bytes(raw);self.sha=hashlib.sha256(raw).hexdigest()
        self.receipt={'schema':'numi.human.resting-anatomy-receipt.v1','payload':{'sha256':self.sha},
            'functional_bindings':{'anatomy_payload_sha256':self.sha,'passive_viscera_geometry_binding':{'stable_ids':self.passive}},
            'provenance':{'source_id_map':{str(s):{'source_member':MEMBERS.get(s,'fixture')} for s in self.ids},
                'cardiac_geometry_binding':{'output_anatomy_payload_sha256':self.sha,'ventricular_wall_binding':{'output_anatomy_payload_sha256':self.sha,'wall_map':{}}}},
            'qualification':{},'mass_geometry_accounting':{'fixture_mass_unchanged':True}}
        self.candidate=self.root/'candidate.npz';np.savez(self.candidate,vertices=self.v*.999,faces=self.f)
        self.digest=hashlib.sha256(self.candidate.read_bytes()).hexdigest();_,_,rows=_parse_payload(raw)
        ledger={str(s):hashlib.sha256(_record_content_bytes(rows[s])).hexdigest() for s in self.passive}
        self.audit={'source_payload_sha256':self.sha,'candidate_sha256':self.digest,'self':{'count':0},'source_record_sha256':ledger,
            'source_weld_sha256':'fixture-weld','source_weld_record_sha256':ledger['457'],'exact_audit_sha256':'fixture-exact','predicate_sha256':'fixture-predicate',
            'pairs':[{'ids':[457,s],'count':0} for s in self.passive if s!=457]}
        self.shape={'source_sha256':'fixture-weld','candidate_sha256':self.digest,'sampled_distance_bound_m':.0005,'relative_volume_bound':.04,
            'source_to_candidate':{'max_m':.00001},'candidate_to_source':{'max_m':.00001},
            'source_volume_m3':signed_volume(self.v.astype(float),self.f),'candidate_volume_m3':signed_volume((self.v*.999).astype(float),self.f)}
        self.interfaces={'candidate_sha256':self.digest,'audit_sha256':'fixture-exact','predicate_sha256':'fixture-predicate','interfaces':[]}

    def run_candidate(self):
        paths=[]
        for name,value in [('receipt',self.receipt),('audit',self.audit),('shape',self.shape),('interfaces',self.interfaces)]:
            p=self.root/(name+'.json');p.write_text(json.dumps(value));paths.append(p)
        return build_candidate(self.payload,paths[0],self.candidate,paths[1],paths[2],paths[3],self.root/'output')

    def test_only_taenia_changes_and_mass_and_cardiac_ownership_are_preserved(self):
        result=self.run_candidate();_,_,before=_parse_payload(self.payload.read_bytes());_,_,after=_parse_payload(Path(result['payload']['path']).read_bytes())
        for sid in self.ids:self.assertEqual(_record_content_bytes(before[sid])==_record_content_bytes(after[sid]),sid!=457)
        r=json.loads((self.root/'output/resting-anatomy-receipt.json').read_text())
        self.assertEqual(r['mass_geometry_accounting'],self.receipt['mass_geometry_accounting'])
        self.assertEqual(r['provenance']['source_id_map']['457']['source_member'],'FJ2569')
        self.assertEqual(r['provenance']['cardiac_geometry_binding']['ventricular_wall_binding']['output_anatomy_payload_sha256'],result['payload']['sha256'])

    def test_stale_or_incomplete_evidence_is_rejected(self):
        cases=[]
        a=copy.deepcopy(self.audit);a['source_record_sha256']['3']='changed';cases.append(('audit',a))
        a=copy.deepcopy(self.audit);a['pairs'].pop();cases.append(('audit',a))
        a=copy.deepcopy(self.audit);a['self']['count']=1;cases.append(('audit',a))
        a=copy.deepcopy(self.audit);a['pairs'][0]['invalid_geometry']='degenerate';cases.append(('audit',a))
        a=copy.deepcopy(self.shape);a['source_sha256']='unbound';cases.append(('shape',a))
        a=copy.deepcopy(self.shape);a['candidate_to_source']['max_m']=.00051;cases.append(('shape',a))
        for name,value in cases:
            old=getattr(self,name)
            with self.subTest(name=name):
                setattr(self,name,value)
                with self.assertRaises(ValueError):self.run_candidate()
                self.assertFalse((self.root/'output').exists());setattr(self,name,old)

    def test_a_source_family_does_not_waive_crossings_outside_localized_interface(self):
        pair=next(x for x in self.audit['pairs'] if x['ids'][1]==456);pair.update(count=1,triangle_pairs=[[0,0]])
        self.interfaces['interfaces']=[{'stable_ids':[457,456],'source_members':['FJ2569','FJ2568'],'source_triangle_pairs':[[0,0]],
            'interface':'taenia_convergence_at_caudal_ascending_colon','distance_from_caudal_ascending_extent_m':[.01,.0201]}]
        with self.assertRaisesRegex(ValueError,'localized'):self.run_candidate()
        self.interfaces['interfaces'][0]['distance_from_caudal_ascending_extent_m']=[.01,.015]
        self.assertIn('payload',self.run_candidate())

    def test_unrelated_organ_crossing_and_prebound_cardiac_map_are_rejected(self):
        self.audit['pairs'][0].update(count=1,triangle_pairs=[[0,0]])
        with self.assertRaises(ValueError):self.run_candidate()
        self.audit['pairs'][0]['count']=0
        self.receipt['provenance']['cardiac_geometry_binding']['ventricular_wall_binding']['wall_map']['local_coefficient_refinement']={}
        with self.assertRaisesRegex(ValueError,'cardiac correction'):self.run_candidate()


if __name__=='__main__':unittest.main()
