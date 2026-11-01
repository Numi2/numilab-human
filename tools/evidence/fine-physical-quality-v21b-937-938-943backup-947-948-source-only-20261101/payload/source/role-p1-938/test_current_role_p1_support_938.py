#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, tempfile, unittest
from pathlib import Path
import numpy as np
import build_current_role_p1_143_support_938 as b

P1_SOURCE=b.H/"downstream-004-same-material-current-reference-229/native-domain-217-source-007/p1_reference_anchor_217.py"

def load_p1_owner():
    spec=importlib.util.spec_from_file_location("p1_owner_938_fixture",P1_SOURCE)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def fixture_data():
    parent={"tets":np.asarray([[0,1,2,3],[0,2,1,4],[0,3,4,5]],np.int64),
      "materials":np.asarray([15,15,16],np.int32),"donors":np.asarray([2,2,9],np.int32),
      "regions":np.asarray([7,8,99],np.int32),"faces":np.asarray([101,102,999],np.uint32)}
    groups,lookup=b._source_groups(parent,"Skin")
    g7=lookup[(15,2,7)];g8=lookup[(15,2,8)]
    points=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
    maps=[{g7:{0}},{g7:{0}},{g8:{1}},{g8:{1}}]
    ns={"maps":maps,"origin":np.zeros(4,np.uint8),"distance":np.zeros(4,np.float64)}
    node_arrays=b._serialize_node_maps(points,ns,groups,parent)
    p1=b.integrate_p1_cells(np.asarray([[0,1,2,3]],np.int64),maps,groups,parent,"Skin")
    adj,proofs=b.build_region_adjacency(parent,"Skin")
    ext=b.expand_onehop(p1,groups,adj,"Skin",1)
    return parent,groups,points,node_arrays,p1,ext,proofs

class CurrentSupportTests(unittest.TestCase):
    def test_owner_contract_p1_143_roundtrip_and_source_pins(self):
        parent,groups,points,node,p1,ext,proofs=fixture_data()
        self.assertEqual(int(p1["cell_p1_score_denominator"][0]),4)
        self.assertEqual(int(p1["cell_group_p1_score_numerators"].sum()),4)
        self.assertEqual(set(ext["cell_admissible_region_ids"].tolist()),{7,8})
        self.assertIn((7,8),proofs)
        base={**p1,**node,"geometry_sha256":np.asarray("a"*64),
              "current_state_path":np.asarray("/fixture/state.json"),
              "current_state_sha256":np.asarray("b"*64),"volume_sha256":np.asarray("c"*64)}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"output"
            base_path,support_path,base_sha=b._write_support_pair(base,ext,out,"Skin")
            receipt={
              "status":b.ROLE_STATUS["Skin"],"physical_admission":False,"native_steps":0,
              "inputPins":{"source330_parent":{"mass_sha256":"1"*64,"nodes_sha256":"2"*64,
                "tets_sha256":"3"*64,"labels_sha256":"4"*64,"full86_weights_sha256":b.PARENT.with_name(b.PARENT.name+".weights.csv").name},
                "073_receipt":{"element_sha256":"5"*64}}}
            # Use the loader owner's exact immutable full86 digest.
            owner=load_p1_owner()
            receipt["inputPins"]["source330_parent"]["full86_weights_sha256"]=owner.FULL86_SHA256
            rp=out/"receipt.json";rp.write_text(json.dumps(receipt))
            owner.P1_PINS={"skin":{"support_sha256":b.sha(support_path),"receipt_sha256":b.sha(rp),
              "cells":1,"material":15,"status":b.ROLE_STATUS["Skin"]}}
            owner.SKIN_ELE_SHA256="5"*64
            meta={"massSHA256":"1"*64,"nodeSHA256":"2"*64,"tetsSHA256":"3"*64,"labelsSHA256":"4"*64}
            got=owner.load_p1_support("skin",support_path,rp,meta,parent)
            self.assertTrue(np.array_equal(got["cell_group_p1_score_numerators"],[2,2]))
            with np.load(support_path,allow_pickle=False) as z:
                self.assertEqual(str(z["admissibility_source_support_sha256"].item()),base_sha)
                self.assertEqual(int(z["admissibility_max_source_region_hops"]),1)

    def test_role_ele_requires_exact_current_217_handoff(self):
        state={"path":"/fixture/state.json","sha256":"a"*64}
        vol={"path":"/fixture/volume.npz","sha256":"b"*64}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ele=root/"skin.ele";ele.write_text("fixture")
            h={"schema":"numi.h706.current-fine-217-combined-volume-handoff.v1",
              "status":"PREPARED_EXISTING_217_COMBINED_CHILD_VOLUME_HANDOFF_NO_PHYSICAL_ADMISSION",
              "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
              "inputs":{"current_state":state,"skin_volume":vol,
                "canonical_834_report":{"path":"/fixture/834.json","sha256":"d"*64}},
              "owner":{"path":str(b.OWNER217),"sha256":b.OWNER217_SHA,
                "derived_from":{"path":str(b.OWNER217_BASE),"sha256":b.OWNER217_BASE_SHA}},
              "role_meshes":{"Skin":{"active_cell_count":1,"cell_order_preserved":True,"tetrahedra_roundtrip_exact":True}},
              "outputs":{"skin.ele":{"sha256":b.sha(ele)}}}
            rp=root/"adapter-receipt.json";rp.write_text(json.dumps(h))
            q=b._validate_role_ele("Skin",ele,rp,state,vol,{"path":"/fixture/834.json","sha256":"d"*64},1)
            self.assertEqual(q["role_cell_count"],1)
            h["inputs"]["skin_volume"]={"path":vol["path"],"sha256":"c"*64}
            rp.write_text(json.dumps(h))
            with self.assertRaises(ValueError):b._validate_role_ele("Skin",ele,rp,state,vol,{"path":"/fixture/834.json","sha256":"d"*64},1)

    def test_full_controls(self):
        result=b.run_controls()
        self.assertEqual(result["status"],"PASS_CURRENT_ROLE_P1_SUPPORT_CONTROLS")
        self.assertFalse(result["physical_admission"])
        self.assertEqual(result["native_steps"],0)

if __name__=="__main__":unittest.main(verbosity=2)
