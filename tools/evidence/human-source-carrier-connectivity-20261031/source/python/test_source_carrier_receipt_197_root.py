import copy,hashlib,struct,unittest
from source_carrier_receipt_197_root import (
    FLOW_NAMES,CanonicalDigest,source_carrier_packet_digest,source_carrier_array_digest,
    validate_current_input_pins,validate_receipt_mode
)
from verify_recorded_carrier_allocation_197_root import (
    bary_point, build_lp_carriers, INVALID
)

SHA={c*64 for c in "abcdef0123456789"}
def receipt(role="Skin",lp=True):
    su=role.upper()
    pins={
        "input_child_node_sha256":"a"*64,
        "input_child_element_sha256":"b"*64,
        "input_parent_mass_csv_sha256":"c"*64,
        "input_parent_label_csv_sha256":"d"*64,
        "support_binary_sha256":"e"*64,
        "input_region_label_sha256":"9"*64,
    }
    flow={name:{"sha256":"f"*64,"size_bytes":16} for name in FLOW_NAMES}
    if not lp:
        return {
            "status":f"PASS_SOURCE_ONLY_{su}_SOURCE_CARRIER_SHARED_CAPACITY_ALLOCATION",
            "recorded_flow_deterministic_replay":"PASS","physical_admission":False,"native_steps":0,
            "flow_csv":flow,
        }
    return {
        "schema":"numi.human.source-carrier-lp-feasibility-allocation.v1",
        "status":f"PASS_SOURCE_ONLY_{su}_SOURCE_CARRIER_FEASIBLE_ALLOCATION",
        "complete_allocation":True,"physical_admission":False,"native_steps":0,
        "recorded_flow_feasibility":"PASS_SOLVER_FREE","deterministic_optimizer_replay":"NOT_RUN",
        "allocation_construction":"held_dinic_partial_plus_lp_positive_carrier_flows",
        "allocation_solver":"held_dinic_partial_plus_sparse_lp_positive_carrier_flow",
        "cost_optimality_certified":False,
        "semantic_candidate_edge_count_basis":"pinned held+LP proposal graph count; feasibility does not reconstruct optimization graph",
        "augmentation_count_basis":"held partial Dinic subflow only; LP construction is counted separately",
        "held_dinic_augmentation_count":1,"augmentations":1,"lp_positive_variable_count":3,
        "held_positive_flow_rows":1,"held_candidate_edges":7,"lp_candidate_edges":5,
        "candidate_edges":12,"maximum_allowed_candidate_edges":20,
        "flow_csv":flow,"input_sha256":pins,
        "child_mesh_sha256":"0"*64,"input_child_node_sha256":pins["input_child_node_sha256"],
        "input_child_element_sha256":pins["input_child_element_sha256"],
        "input_parent_mass_csv_sha256":pins["input_parent_mass_csv_sha256"],
        "input_parent_label_csv_sha256":pins["input_parent_label_csv_sha256"],
        "support_binary_sha256":pins["support_binary_sha256"],
        "input_region_label_sha256":pins["input_region_label_sha256"],
        "semantic_support_sha256":"1"*64,"transport_receipt_sha256":"2"*64,
        "admissibility_source_support_sha256":"3"*64,
        "admissibility_max_source_region_hops":1,"admissible_region_slots":42,
        "input_parent_mass_sha256":pins["input_parent_mass_csv_sha256"],
        "input_parent_source_sha256":"4"*64,"input_coordinate_frame_sha256":"5"*64,
        "source_carrier_packet_sha256":"6"*64,"source_carrier_array_sha256":"7"*64,
        "source_carrier_flow_trace_sha256":"8"*64,"lp_proposal_sha256":"9"*64,
        "lp_input_manifest_sha256":"a"*64,"held_partial_manifest_sha256":"b"*64,
        "held_carrier_flow_csv_sha256":"c"*64,
        "feasibility_source_sha256":"8"*64,
    }

class ReceiptModeTests(unittest.TestCase):
    def test_lp_status_for_both_roles(self):
        self.assertEqual(validate_receipt_mode(receipt("Skin"),"Skin"),"solver_free_lp_feasibility")
        self.assertEqual(validate_receipt_mode(receipt("Fat"),"Fat"),"solver_free_lp_feasibility")
    def test_legacy_optimizer_path_is_unchanged(self):
        self.assertEqual(validate_receipt_mode(receipt("Skin",False),"Skin"),"deterministic_optimizer_replay")
    def test_wrong_role_rejects(self):
        with self.assertRaises(ValueError):validate_receipt_mode(receipt("Skin"),"Fat")
    def test_lp_truth_fields_fail_closed(self):
        for key,value in (("complete_allocation",False),("physical_admission",True),("native_steps",1),
                          ("recorded_flow_feasibility","NOT_RUN"),("deterministic_optimizer_replay","PASS"),
                          ("cost_optimality_certified",True)):
            bad=receipt();bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_receipt_mode(bad,"Skin")
    def test_candidate_budget_and_held_history_reject(self):
        bad=receipt();bad["maximum_allowed_candidate_edges"]=10
        with self.assertRaises(ValueError):validate_receipt_mode(bad,"Skin")
        bad=receipt();bad["held_dinic_augmentation_count"]=0
        with self.assertRaises(ValueError):validate_receipt_mode(bad,"Skin")
    def test_replay_claim_cannot_be_smuggled_into_lp_branch(self):
        bad=receipt();bad["recorded_flow_deterministic_replay"]="PASS"
        with self.assertRaises(ValueError):validate_receipt_mode(bad,"Skin")
    def test_required_flat_and_nested_pins_match(self):
        r=receipt();expected=dict(r["input_sha256"])
        validate_current_input_pins(r,expected)
        bad=copy.deepcopy(r);bad["input_child_node_sha256"]="f"*64
        with self.assertRaises(ValueError):validate_current_input_pins(bad,expected)
    def test_parent_packet_aliases_match_source_files(self):
        r=receipt()
        expected={"input_child_node_sha256":"a"*64,"input_child_element_sha256":"b"*64,
                  "input_parent_mass_csv_sha256":"c"*64,"input_parent_label_csv_sha256":"d"*64,
                  "support_binary_sha256":"e"*64,"input_region_label_sha256":"9"*64,
                  "input_parent_source_sha256":"4"*64}
        validate_current_input_pins(r,expected)
        bad=copy.deepcopy(r);bad["input_parent_mass_sha256"]="f"*64
        with self.assertRaises(ValueError):validate_current_input_pins(bad,expected)
    def test_wrong_six_csv_contract_rejects(self):
        bad=receipt();bad["flow_csv"].pop("transport-flow.csv")
        with self.assertRaises(ValueError):validate_receipt_mode(bad,"Skin")
    def test_bad_hash_pin_rejects(self):
        bad=receipt();bad["source_carrier_packet_sha256"]="not-a-digest"
        with self.assertRaises(ValueError):validate_receipt_mode(bad,"Skin")
    def test_region_label_input_pin_is_required_and_authenticated(self):
        bad=receipt();bad["input_sha256"].pop("input_region_label_sha256")
        with self.assertRaises(ValueError):validate_receipt_mode(bad,"Skin")
        bad=receipt();bad["input_region_label_sha256"]="f"*64
        with self.assertRaises(ValueError):validate_receipt_mode(bad,"Skin")
        bad=receipt();bad["input_sha256"]["input_region_label_sha256"]="f"*64
        with self.assertRaises(ValueError):validate_receipt_mode(bad,"Skin")
    def test_canonical_digest_matches_cpp_scalar_encoding(self):
        h=CanonicalDigest()
        h.string("x");h.u32(1);h.u64(2);h.real(.5)
        expected=hashlib.sha256(
            struct.pack(">Q",1)+b"x"+struct.pack(">I",1)+
            struct.pack(">Q",2)+struct.pack(">d",.5)
        ).hexdigest()
        self.assertEqual(h.finish(),expected)
    def test_composite_carrier_digests_bind_provenance_and_arrays(self):
        args=("1"*64,"2"*64,"3"*64,"4"*64,"5"*64,[0,1],[10],[1.0],
              [[.25,.25,.25,.25]],[7],[.125])
        packet=source_carrier_packet_digest(*args)
        self.assertEqual(packet,source_carrier_packet_digest(*args))
        changed=list(args);changed[6]=[.5]
        self.assertNotEqual(packet,source_carrier_packet_digest(*changed))
        array_args=(packet,"6"*64,"7"*64,[0,1],[10],[[0,1,2,3]],
                    [[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]],
                    [1.0],[[.25,.25,.25,.25]],[7],[.125],1,
                    "8"*64,"9"*64,"a"*64,"b"*64,"c"*64)
        digest=source_carrier_array_digest(*array_args)
        self.assertEqual(digest,source_carrier_array_digest(*array_args))
        changed=list(array_args);changed[-2]="d"*64
        self.assertNotEqual(digest,source_carrier_array_digest(*changed))

    def test_barycentric_projection_argument_order_and_lp_carrier_build(self):
        import csv,hashlib,tempfile
        from pathlib import Path
        import numpy as np
        points=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        bary=np.asarray([.25,.25,.25,.25])
        expected=np.asarray([.25,.25,.25])
        np.testing.assert_array_equal(bary_point(points,bary),expected)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            held=root/'held.csv'
            with held.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['child_row','parent_row','expanded_parent_row','carrier_row','mass_kg','source_measure_m3','distance_m'])
                w.writeheader();w.writerow({'child_row':0,'parent_row':0,'expanded_parent_row':0,'carrier_row':INVALID,'mass_kg':100.,'source_measure_m3':1.,'distance_m':0.})
            held_manifest=root/'held.json';held_manifest.write_text('{}')
            proposal=root/'proposal.npz';np.savez(proposal,variable_flow_mass_kg=np.asarray([50.]),variable_positive_indices=np.asarray([4]),variable_child_rows=np.asarray([0]),variable_parent_rows=np.asarray([0]),variable_barycentric=np.asarray([[.25,.25,.25,.25]]),variable_distance_m=np.asarray([0.]),held_parent_scale=np.asarray([.5]))
            lp_manifest=root/'lp.json';lp_manifest.write_text('{}')
            feasibility=root/'feasibility.json';feasibility.write_text('{}')
            def digest(p):
                with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
            r={'held_partial_manifest_sha256':digest(held_manifest),'held_carrier_flow_csv_sha256':digest(held),'lp_proposal_sha256':digest(proposal),'lp_input_manifest_sha256':digest(lp_manifest),'feasibility_source_sha256':digest(feasibility),'held_positive_flow_rows':1,'lp_positive_variable_count':1}
            q={'parent_offsets':np.asarray([0,0],dtype=np.uint32),'carrier_barycentric':np.empty((0,4)),'carrier_target_child_rows':np.empty(0,dtype=np.uint32)}
            out=build_lp_carriers(proposal,held,held_manifest,lp_manifest,feasibility,r,q,'a'*64,np.asarray([7],dtype=np.uint32),np.asarray([100.]),np.asarray([1.]),points.reshape((1,4,3)),np.asarray([expected]),np.asarray([expected]),.019353)
            offsets,weights,barys,witnesses,distances,expanded_source,expanded_carrier,sites,expanded_weights,packet_sha,carrierized=out
            np.testing.assert_array_equal(offsets,[0,2])
            np.testing.assert_allclose(weights,[.5,.5],rtol=0,atol=0)
            np.testing.assert_allclose(barys,[[.25]*4,[.25]*4],rtol=0,atol=0)
            np.testing.assert_array_equal(witnesses,[0,0])
            np.testing.assert_array_equal(expanded_source,[0,0])
            np.testing.assert_array_equal(expanded_carrier,[0,1])
            np.testing.assert_allclose(sites,[expected,expected],rtol=0,atol=0)
            self.assertEqual(carrierized,1)
            self.assertEqual(len(packet_sha),64)

if __name__=="__main__":unittest.main()
