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





class TaeniaWallRegionTests(unittest.TestCase):
    """Region admission invariants; tetrahedra are engineering fixtures only."""
    stable_id = 456

    def setUp(self):
        from numilab_human.resting_taenia_reference import (
            WALL_REGION_PARENTS, face_coordinate_sha256)
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        v = np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]], dtype="<f4")
        f = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], dtype="<i4")
        self.ids = [self.stable_id, *WALL_REGION_PARENTS, 23]
        chunks, vv, ff, nv, ni = [], [], [], 0, 0
        selected_v, selected_f, owners, parent_faces = [], [], [], []
        for i, sid in enumerate(self.ids):
            positions = v + np.array([i*.03, 0, 0], dtype="<f4")
            packed = np.column_stack((positions, normals(positions.astype(float), f))).astype("<f4")
            chunks.append(RECORD.pack(20, nv, 4, ni, 12, sid, 1, 0))
            vv.append(packed.tobytes()); ff.append((f.reshape(-1)+nv).astype("<u4").tobytes())
            nv += 4; ni += 12
            if sid in WALL_REGION_PARENTS:
                selected_f.extend((f[:2]+len(selected_v)).tolist())
                selected_v.extend(positions.tolist())
                owners.extend([sid, sid]); parent_faces.extend([0, 1])
        raw = HEADER.pack(b"NHANAT1\0", 5, len(self.ids), nv, ni, 123, b"x"*32) + b"".join(chunks+vv+ff)
        self.payload = self.root/"source.nhanatomy"; self.payload.write_bytes(raw)
        self.sha = hashlib.sha256(raw).hexdigest()
        _, _, self.rows = _parse_payload(raw)
        self.source_map = {str(s): {"source_member": MEMBERS.get(s, "fixture")} for s in self.ids}
        self.arrays = dict(vertices=np.asarray(selected_v, dtype="<f4"),
                           faces=np.asarray(selected_f, dtype="<i4"),
                           colon_owner=np.asarray(owners, dtype="<i4"),
                           colon_face=np.asarray(parent_faces, dtype="<i4"))
        self.candidate = self.root/"candidate.npz"; np.savez(self.candidate, **self.arrays)
        self.derivation = {
            "representation": "passive_colon_wall_surface_region_v1", "stable_id": self.stable_id,
            "candidate_sha256": hashlib.sha256(self.candidate.read_bytes()).hexdigest(),
            "source_record_sha256": hashlib.sha256(_record_content_bytes(self.rows[self.stable_id])).hexdigest(),
            "parent_face_coordinate_sha256": {str(s): face_coordinate_sha256(self.rows[s]) for s in WALL_REGION_PARENTS},
            "rectal_scope": "excluded_no_taenia_band",
            "sampled_shape": {
                "distance_is_sampled_not_hausdorff_bound": True,
                "source_to_region": {"sample_count": 1, "maximum_m": .0058 if self.stable_id == 456 else .0038},
                "region_to_source": {"sample_count": 1, "maximum_m": .0049 if self.stable_id == 456 else .0149},
                "face_selection_sampled_distance_limit_m": .005 if self.stable_id == 456 else .002},
            "sensitivity": [{"fixture": "4mm"}, {"fixture": "5mm"}],
            "references": ["Synthetic admission fixture, not anatomical evidence."]}
        self.receipt = {
            "schema": "numi.human.resting-anatomy-receipt.v1", "payload": {"sha256": self.sha},
            "functional_bindings": {"anatomy_payload_sha256": self.sha,
                "passive_viscera_geometry_binding": {"stable_ids": self.ids}},
            "provenance": {"source_id_map": self.source_map,
                "cardiac_geometry_binding": {"output_anatomy_payload_sha256": self.sha,
                    "ventricular_wall_binding": {"output_anatomy_payload_sha256": self.sha, "wall_map": {}}}},
            "qualification": {}, "mass_geometry_accounting": {"reference_total_mass_kg": 72}}

    def prepare(self):
        from numilab_human.resting_taenia_reference import prepare_wall_region
        path = self.root/"derivation.json"; path.write_text(json.dumps(self.derivation))
        return prepare_wall_region(self.rows, self.source_map, self.candidate, path, stable_id=self.stable_id)

    def compile(self):
        from numilab_human.resting_taenia_reference import build_wall_region_candidate
        dp = self.root/"derivation.json"; dp.write_text(json.dumps(self.derivation))
        rp = self.root/"receipt.json"; rp.write_text(json.dumps(self.receipt))
        return build_wall_region_candidate(self.payload, rp, self.candidate, dp, self.root/"output", stable_id=self.stable_id)

    def save_modified_candidate(self):
        np.savez(self.candidate, **self.arrays)
        self.derivation["candidate_sha256"] = hashlib.sha256(self.candidate.read_bytes()).hexdigest()

    def test_region_is_open_nonadditive_and_other_records_are_unchanged(self):
        result = self.compile()
        _, _, after = _parse_payload(Path(result["payload"]["path"]).read_bytes())
        for sid in self.ids:
            self.assertEqual(_record_content_bytes(self.rows[sid]) == _record_content_bytes(after[sid]), sid != self.stable_id)
        r = json.loads((self.root/"output/resting-anatomy-receipt.json").read_text())
        detail = (r["provenance"]["passive_taenia_wall_region"] if self.stable_id == 456 else
                  r["provenance"]["passive_taenia_wall_regions"][str(self.stable_id)])
        self.assertGreater(detail["topology"]["boundary_edge_count"], 0)
        self.assertEqual(detail["exact_self"]["count"], 0)
        self.assertIsNone(detail["independent_volume_m3"])
        self.assertEqual(detail["additional_physical_mass_kg"], 0)
        self.assertEqual(r["mass_geometry_accounting"], self.receipt["mass_geometry_accounting"])
        self.assertEqual(r["provenance"]["source_id_map"][str(self.stable_id)]["source_member"], MEMBERS[self.stable_id])

    def test_stale_parent_and_source_frame_are_rejected(self):
        self.rows[454]["vertices6"][0, 0] += .0001
        with self.assertRaisesRegex(ValueError, "parent colon"): self.prepare()
        self.rows[454]["body_index"] = 128
        with self.assertRaisesRegex(ValueError, "torso frame"): self.prepare()

    def test_geometrically_near_and_reversed_faces_are_not_shared_faces(self):
        self.arrays["vertices"][0, 0] += 1e-7
        self.save_modified_candidate()
        with self.assertRaisesRegex(ValueError, "identically oriented"): self.prepare()
        self.arrays["vertices"][0] = self.rows[454]["vertices6"][0, :3]
        self.arrays["faces"][0] = self.arrays["faces"][0][::-1]
        self.save_modified_candidate()
        with self.assertRaisesRegex(ValueError, "identically oriented"): self.prepare()

    def test_duplicate_parent_face_and_rectal_parent_are_rejected(self):
        self.arrays["colon_face"][1] = 0
        self.save_modified_candidate()
        with self.assertRaisesRegex(ValueError, "repeats"): self.prepare()
        self.arrays["colon_owner"][0] = 459
        self.save_modified_candidate()
        with self.assertRaisesRegex(ValueError, "inventory"): self.prepare()

    def test_source_shape_scope_and_prebound_cardiac_map_fail_closed(self):
        shape = self.derivation["sampled_shape"]
        shape["distance_is_sampled_not_hausdorff_bound"] = False
        with self.assertRaisesRegex(ValueError, "sampled scope"): self.prepare()
        shape["distance_is_sampled_not_hausdorff_bound"] = True
        shape["source_to_region"]["maximum_m"] = float("nan")
        with self.assertRaisesRegex(ValueError, "sampling"): self.prepare()
        excessive = .0061 if self.stable_id == 456 else .0041
        shape["source_to_region"]["maximum_m"] = excessive
        with self.assertRaisesRegex(ValueError, "extent"): self.prepare()
        shape["source_to_region"]["maximum_m"] = .0058 if self.stable_id == 456 else .0038
        self.receipt["provenance"]["cardiac_geometry_binding"]["ventricular_wall_binding"]["wall_map"]["local_coefficient_refinement"] = {}
        with self.assertRaisesRegex(ValueError, "cardiac correction"): self.compile()
        self.assertFalse((self.root/"output").exists())

    def test_parent_ownership_does_not_waive_crossing_regions(self):
        from numilab_human.resting_taenia_reference import face_coordinate_sha256
        self.rows[455]["vertices6"] = self.rows[454]["vertices6"].copy()
        self.arrays["vertices"][4:8] = self.arrays["vertices"][:4]
        self.derivation["parent_face_coordinate_sha256"]["455"] = face_coordinate_sha256(self.rows[455])
        self.save_modified_candidate()
        with self.assertRaisesRegex(ValueError, "self intersection"): self.prepare()


if __name__ == '__main__': unittest.main()


class TaeniaMesocolicaWallRegionTests(TaeniaWallRegionTests):
    stable_id = 457

    def test_mesocolica_uses_specific_coarse_projection_bounds_and_keeps_libera_record(self):
        self.receipt["provenance"]["passive_taenia_wall_region"] = {"stable_id": 456, "sentinel": "preserved"}
        result = self.compile()
        receipt = json.loads(Path(result["receipt"]["path"]).read_text())
        self.assertEqual(receipt["provenance"]["passive_taenia_wall_region"], {"stable_id": 456, "sentinel": "preserved"})
        detail = receipt["provenance"]["passive_taenia_wall_regions"]["457"]
        self.assertEqual(detail["source_member"], "FJ2569")
        self.assertEqual(detail["stable_id"], 457)
        self.assertEqual(detail["independent_volume_m3"], None)
        self.assertEqual(receipt["provenance"]["source_id_map"]["457"]["source_shell_status"],
                         "retained_in_provenance_not_rendered_as_independent_solid")

    def test_mesocolica_sampled_reverse_extent_fails_closed_above_bound(self):
        self.derivation["sampled_shape"]["region_to_source"]["maximum_m"] = .01501
        with self.assertRaisesRegex(ValueError, "extent"):
            self.prepare()
class PassiveRowTransferTests(unittest.TestCase):
    def setUp(self):
        from numilab_human.resting_anatomy_interface_patch import HEADER, RECORD, normals
        from numilab_human.resting_taenia_reference import PASSIVE_REFERENCE_TRANSFER_IDS
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.ids = sorted(set(PASSIVE_REFERENCE_TRANSFER_IDS) | {454, 460, 999})
        self.transfer_ids = set(PASSIVE_REFERENCE_TRANSFER_IDS)
        self.v = np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]], dtype="<f4")
        self.f = np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], dtype="<i4")
        self.header = HEADER; self.record = RECORD
        def payload(offset):
            chunks=[]; vv=[]; ff=[]; nv=ni=0
            for sid in self.ids:
                row_offset = offset if sid in self.transfer_ids else 0.0
                xyz = self.v + np.array([sid*1e-5 + row_offset, 0, 0], dtype="<f4")
                packed = np.column_stack((xyz, normals(xyz.astype(float), self.f))).astype("<f4")
                chunks.append(RECORD.pack(20,nv,4,ni,12,sid,1,0))
                vv.append(packed.tobytes()); ff.append((self.f.reshape(-1)+nv).astype("<u4").tobytes())
                nv += 4; ni += 12
            return HEADER.pack(b"NHANAT1\0",5,len(self.ids),nv,ni,123,b"x"*32)+b"".join(chunks+vv+ff)
        self.base = self.root/"base.nhanatomy"; self.base.write_bytes(payload(0))
        self.passive = self.root/"passive.nhanatomy"; self.passive.write_bytes(payload(.002))
        def receipt(path):
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            source_map={str(s):{"source_member":f"FJ{s}","source_sha256":f"source-{s}",
                "source_owner_metadata":{"member_id":f"FJ{s}"},"name":f"source {s}"} for s in self.ids}
            return {"schema":"numi.human.resting-anatomy-receipt.v1",
                "payload":{"sha256":sha},
                "functional_bindings":{"anatomy_payload_sha256":sha,"passive_viscera_geometry_binding":{"stable_ids":self.ids}},
                "provenance":{"source_id_map":source_map,"cardiac_geometry_binding":{
                    "output_anatomy_payload_sha256":sha,"ventricular_wall_binding":{"output_anatomy_payload_sha256":sha}},
                    "passive_bowel_reference_composition":{"fixture":True}},
                "qualification":{},"mass_geometry_accounting":{"reference_total_mass_kg":72}}
        self.base_receipt=self.root/"base.json";self.base_receipt.write_text(json.dumps(receipt(self.base)))
        self.passive_receipt=self.root/"passive.json";self.passive_receipt.write_text(json.dumps(receipt(self.passive)))

    def compose(self, out=None):
        from numilab_human.resting_taenia_reference import compose_passive_rows_onto_respiratory_base
        return compose_passive_rows_onto_respiratory_base(
            base_payload=self.base,base_receipt=self.base_receipt,
            passive_payload=self.passive,passive_receipt=self.passive_receipt,
            output=out or self.root/"out")

    def test_only_declared_rows_transfer_and_cardio_stays_pending(self):
        result=self.compose()
        _,_,base_rows=_parse_payload(self.base.read_bytes())
        _,_,passive_rows=_parse_payload(self.passive.read_bytes())
        _,_,out_rows=_parse_payload(Path(result["payload"]["path"]).read_bytes())
        from numilab_human.resting_taenia_reference import PASSIVE_REFERENCE_TRANSFER_IDS
        moved=set(PASSIVE_REFERENCE_TRANSFER_IDS)
        for sid in self.ids:
            expected=passive_rows[sid] if sid in moved else base_rows[sid]
            self.assertEqual(_record_content_bytes(expected),_record_content_bytes(out_rows[sid]),sid)
        receipt=json.loads(Path(result["receipt"]["path"]).read_text())
        self.assertEqual(receipt["mass_geometry_accounting"],json.loads(self.base_receipt.read_text())["mass_geometry_accounting"])
        self.assertEqual(receipt["provenance"]["passive_reference_row_transfer"]["transferred_stable_ids"],sorted(moved))
        self.assertIn("not_ready",receipt["provenance"]["passive_reference_row_transfer"]["native_readiness"])

    def test_stale_identity_parent_or_payload_fails_closed(self):
        source=json.loads(self.passive_receipt.read_text())
        source["provenance"]["source_id_map"]["455"]["source_member"]="wrong"
        self.passive_receipt.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError,"source identity"):
            self.compose()
        target=json.loads(self.base_receipt.read_text())
        target["provenance"]["source_id_map"]["2"]["source_sha256"]="wrong"
        self.base_receipt.write_text(json.dumps(target))
        with self.assertRaisesRegex(ValueError,"source identity"):
            self.compose()

    def test_changed_colon_parent_is_rejected(self):
        from numilab_human.resting_anatomy_interface_patch import HEADER, RECORD, normals
        raw=self.passive.read_bytes(); _,_,rows=_parse_payload(raw)
        chunks=[]; vv=[]; ff=[]; nv=ni=0
        for sid in self.ids:
            xyz=np.asarray(rows[sid]["vertices6"][:,:3],dtype="<f4").copy()
            faces=np.asarray(rows[sid]["faces"],dtype="<i4")
            if sid==454: xyz[0,0]+=1e-5
            packed=np.column_stack((xyz,normals(xyz.astype(float),faces))).astype("<f4")
            chunks.append(RECORD.pack(20,nv,len(xyz),ni,faces.size,sid,1,0))
            vv.append(packed.tobytes());ff.append((faces.reshape(-1)+nv).astype("<u4").tobytes())
            nv+=len(xyz);ni+=faces.size
        changed=HEADER.pack(b"NHANAT1\0",5,len(self.ids),nv,ni,123,b"x"*32)+b"".join(chunks+vv+ff)
        self.passive.write_bytes(changed)
        r=json.loads(self.passive_receipt.read_text());sha=hashlib.sha256(changed).hexdigest()
        r["payload"]["sha256"]=sha;r["functional_bindings"]["anatomy_payload_sha256"]=sha
        self.passive_receipt.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,"colon parent 454"):
            self.compose()
