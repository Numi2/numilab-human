from pathlib import Path
import importlib.util
import json
import struct
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/"source"
SPEC=importlib.util.spec_from_file_location("lobe_lineage_v2",ROOT/"lobe_lineage_v2.py")
V=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(V)
E=Path("/Users/n/numi-human-resting-evidence-20261005")
BASE=E/"native-lung-transformed-pose-audit-1079/audit_lung_interfaces_1042_pinned.py"
NHA=E/"native-lung-conditioned-final-compose-1078/final/resting-thorax.nhanatomy"
LEDGER=E/"native-lung-307-sliver-reciprocal-flip-1083/candidate-307-309-parent-ledger.json"
CAND=E/"native-lung-307-sliver-reciprocal-flip-1083/candidate-307-309-rows.npz"
SOURCE_MAP=E/"native-lung-lobe-source-lineage-1055-verified/reciprocal-coincident-face-map.jsonl"


def bits(p):
    return [struct.pack("<f",float(x)).hex() for x in p]


def key(p):
    return tuple(int(round(float(x)*1000)) for x in p)


def fixture():
    va=np.asarray([[0,0,0],[1,0,0],[0,1,0]],dtype="<f4")
    vb=np.asarray([[0,0,0],[1,0,0],[0,1,0]],dtype="<f4")
    rows={305:{"vertices6":va,"faces":np.asarray([[0,1,2]])},
          308:{"vertices6":vb,"faces":np.asarray([[0,2,1]])}}
    ta=tuple(tuple(bits(va[i])) for i in [0,1,2])
    tb=tuple(tuple(bits(vb[i])) for i in [0,2,1])
    faces=[{"pair":[305,308],"stable_names":["a","b"],"face_rows":[0,0],
            "source_face_ids":[5,9],"patch_kinds":[1,1],
            "source_parent_relation":"identical_source_parent_triangle",
            "orientation_relation":"opposite_winding",
            "mapped_output_triangle_vertices_bits":[ta,tb]}]
    edges=[]
    for i in range(3):
        u,v=ta[i],ta[(i+1)%3]
        rev=next(j for j in range(3) if tb[j]==v and tb[(j+1)%3]==u)
        edges.append({"pair":[305,308],"endpoint_coordinate_bits":[u,v],
          "incidences_a":[{"face_map_row":0,"face_row":0,"source_face_id":5,"patch_kind":1,"directed_endpoint_bits":[u,v]}],
          "incidences_b":[{"face_map_row":0,"face_row":0,"source_face_id":9,"patch_kind":1,"directed_endpoint_bits":[v,u]}],
          "classification":"boundary","paired_face_edge_directions_opposed":True})
    summaries=[]
    for pair in V.PAIRS:
        if pair==(305,308):
            summaries.append({"pair":list(pair),"full_reciprocal_face_map_count":1,
              "map_status":"exact_source_supported_reciprocal_surface",
              "edge_counts":{"boundary":3,"interior":0}})
        else:
            summaries.append({"pair":list(pair),"full_reciprocal_face_map_count":0,
              "map_status":"explicit_zero_no_map","raw_exact_hit_count":0})
    return rows,faces,edges,summaries


class V2MapTests(unittest.TestCase):
    def test_current_faces_opposite_winding_and_complete_edges_pass(self):
        rows,faces,edges,summaries=fixture()
        result=V.validate_v2_face_edge_maps(face_rows=faces,edge_rows=edges,
            pair_summaries=summaries,final_rows=rows,point_key=key)
        self.assertEqual(result["pairs"][(305,308)]["face_pairs"],{(0,0)})
        self.assertEqual(len(result["pairs"]),10)


    def test_flat_12_octet_coordinate_word_normalizes_exactly(self):
        words=("8b4ce43d","084d5ebd","a2f352bd")
        flat=[words[j][i:i+2] for j in range(3) for i in range(0,8,2)]
        self.assertEqual(V._bits(flat),words)

    def test_producer_octets_and_nested_union_incidence_fields(self):
        rows,faces,edges,summaries=fixture()
        def octets(word): return [word[i:i+2] for i in range(0,8,2)]
        def octet_point(point): return [octets(x) for x in point]
        faces[0]["mapped_output_triangle_vertices_bits"]=[
            [octet_point(p) for p in tri] for tri in faces[0]["mapped_output_triangle_vertices_bits"]]
        faces[0]["source_parent_relation"]="two_parent_inferred_reference_diagonal_flip"
        faces[0]["source_face_ids"]=[[5,6],[9,10]]
        faces[0]["patch_kinds"]=[[1,1],[1,1]]
        for edge in edges:
            edge["endpoint_coordinate_bits"]=[octet_point(p) for p in edge["endpoint_coordinate_bits"]]
            for side_name in ("incidences_a","incidences_b"):
                for inc in edge[side_name]:
                    inc["source_face_ids"]=[5,6] if side_name=="incidences_a" else [9,10]
                    inc["patch_kinds"]=[1,1]
                    inc.pop("source_face_id",None)
                    inc.pop("patch_kind",None)
                    inc["directed_endpoint_bits"]=[octet_point(p) for p in inc["directed_endpoint_bits"]]
        result=V.validate_v2_face_edge_maps(face_rows=faces,edge_rows=edges,
            pair_summaries=summaries,final_rows=rows,point_key=key)
        self.assertEqual(result["pairs"][(305,308)]["face_pairs"],{(0,0)})

    def test_map_rejects_missing_pair_declaration(self):
        rows,faces,edges,summaries=fixture()
        with self.assertRaisesRegex(ValueError,"all ten"):
            V.validate_v2_face_edge_maps(face_rows=faces,edge_rows=edges,
                pair_summaries=summaries[:-1],final_rows=rows,point_key=key)

    def test_map_rejects_changed_current_triangle(self):
        rows,faces,edges,summaries=fixture()
        faces[0]["face_rows"]=[1,0]
        with self.assertRaisesRegex(ValueError,"out of bounds|differs"):
            V.validate_v2_face_edge_maps(face_rows=faces,edge_rows=edges,
                pair_summaries=summaries,final_rows=rows,point_key=key)

    def test_map_rejects_missing_or_misdirected_edge_incidence(self):
        rows,faces,edges,summaries=fixture()
        with self.assertRaisesRegex(ValueError,"incomplete"):
            V.validate_v2_face_edge_maps(face_rows=faces,edge_rows=edges[:-1],
                pair_summaries=summaries,final_rows=rows,point_key=key)
        rows,faces,edges,summaries=fixture()
        edges[0]["paired_face_edge_directions_opposed"]=False
        with self.assertRaisesRegex(ValueError,"edge class"):
            V.validate_v2_face_edge_maps(face_rows=faces,edge_rows=edges,
                pair_summaries=summaries,final_rows=rows,point_key=key)

    def test_map_rejects_nonopposite_face_orientation(self):
        rows,faces,edges,summaries=fixture()
        faces[0]["mapped_output_triangle_vertices_bits"][1]=faces[0]["mapped_output_triangle_vertices_bits"][0]
        rows[308]["faces"][0]=np.asarray([0,1,2])
        with self.assertRaisesRegex(ValueError,"opposite-wound"):
            V.validate_v2_face_edge_maps(face_rows=faces,edge_rows=edges,
                pair_summaries=summaries,final_rows=rows,point_key=key)



class Real1055SourceRowResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=[json.loads(line) for line in SOURCE_MAP.read_text().splitlines() if line.strip()]
        cls.lookup=V.build_1055_source_row_lookup(cls.records)

    def test_all_source_rows_resolve_by_overlay_row_and_identity(self):
        self.assertEqual(len(self.records),29226)
        for ordinal,record in enumerate(self.records):
            resolved,indices=V.resolve_1055_source_parents(
                source_records=self.records,lookup=self.lookup,pair=record["pair"],
                parent_rows=record["face_rows"],source_ids=record["source_face_ids"],
                patch_kinds=record["patch_kinds"])
            self.assertEqual(indices[0],[ordinal])
            self.assertEqual(indices[1],[ordinal])
            self.assertEqual(resolved[0][0]["pair"],record["pair"])
            self.assertEqual(resolved[1][0]["pair"],record["pair"])

    def test_source_row_resolver_derives_omitted_ordinary_patch_kind(self):
        records=[
            {"pair":[305,308],"face_rows":[10,20],"source_face_ids":[100,200],"patch_kinds":[1,1]},
        ]
        lookup=V.build_1055_source_row_lookup(records)
        resolved,ordinals=V.resolve_1055_source_parents(source_records=records,lookup=lookup,
            pair=[305,308],parent_rows=[10,20],source_ids=[100,200])
        self.assertEqual(ordinals,[[0],[0]])
        self.assertEqual([x[0]["patch_kinds"][0] for x in resolved],[1,1])

    def test_source_row_resolver_rejects_ambiguous_omitted_kind(self):
        records=[
            {"pair":[305,308],"face_rows":[10,20],"source_face_ids":[100,200],"patch_kinds":[0,1]},
            {"pair":[305,308],"face_rows":[10,20],"source_face_ids":[100,200],"patch_kinds":[1,0]},
        ]
        lookup=V.build_1055_source_row_lookup(records)
        with self.assertRaisesRegex(ValueError,"missing or non-unique"):
            V.resolve_1055_source_parents(source_records=records,lookup=lookup,
                pair=[305,308],parent_rows=[10,20],source_ids=[100,200])

    def test_source_row_resolver_handles_owner_specific_parent_order(self):
        records=[
            {"pair":[307,309],"face_rows":[10,20],"source_face_ids":[100,200],"patch_kinds":[1,1]},
            {"pair":[307,309],"face_rows":[11,21],"source_face_ids":[101,201],"patch_kinds":[1,1]},
        ]
        lookup=V.build_1055_source_row_lookup(records)
        resolved,ordinals=V.resolve_1055_source_parents(source_records=records,lookup=lookup,
            pair=[307,309],parent_rows=[[10,11],[21,20]],source_ids=[[100,101],[201,200]],
            patch_kinds=[[1,1],[1,1]])
        self.assertEqual(ordinals,[[0,1],[1,0]])
        self.assertEqual(resolved[1][0]["face_rows"],[11,21])

    def test_source_row_resolver_rejects_wrong_face_or_identity(self):
        r=self.records[0]
        bad=list(r["face_rows"]);bad[0]+=1
        with self.assertRaisesRegex(ValueError,"missing or non-unique"):
            V.resolve_1055_source_parents(source_records=self.records,lookup=self.lookup,
                pair=r["pair"],parent_rows=bad,source_ids=r["source_face_ids"],patch_kinds=r["patch_kinds"])

class Real1083ParentUnionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_spec=importlib.util.spec_from_file_location("auditbase1083",BASE)
        cls.base=importlib.util.module_from_spec(base_spec);base_spec.loader.exec_module(cls.base)
        parser=cls.base.load(cls.base.PARSER,"parser_parent_union_test")
        cls.parent_rows=parser.parse_payload(NHA)[1]
        cls.ledger=json.loads(LEDGER.read_text())
        cls.candidate={}
        with np.load(CAND,allow_pickle=False) as z:
            for sid in (307,309):
                cls.candidate[sid]={"vertices6":np.asarray(z[f"row{sid}_vertices6"],dtype="<f4"),
                                    "faces":np.asarray(z[f"row{sid}_faces"],dtype=np.int64)}

    def test_actual_1083_parent_union_replays_for_both_owners(self):
        cases=((307,40752),(307,40858),(309,1806),(309,1912))
        for sid,child in cases:
            with self.subTest(stable_id=sid,child=child):
                r=V.validate_paired_union(stable_id=sid,current_face_row=child,
                    parent_face_rows=[x for x in self.ledger["modified_faces"][str(sid)]["parent_sets_after"][str(child)]],
                    parent_row=self.parent_rows[sid],candidate_row=self.candidate[sid],ledger=self.ledger)
                self.assertEqual(r["status"],"PASS_exact_paired_parent_union")

    def test_parent_union_rejects_missing_or_wrong_parent(self):
        sid,child=307,40752
        with self.assertRaisesRegex(ValueError,"parent set"):
            V.validate_paired_union(stable_id=sid,current_face_row=child,parent_face_rows=[40752],
                parent_row=self.parent_rows[sid],candidate_row=self.candidate[sid],ledger=self.ledger)
        with self.assertRaisesRegex(ValueError,"parent set"):
            V.validate_paired_union(stable_id=sid,current_face_row=child,parent_face_rows=[40752,40859],
                parent_row=self.parent_rows[sid],candidate_row=self.candidate[sid],ledger=self.ledger)

    def test_parent_union_rejects_triangle_or_coordinate_mismatch(self):
        sid,child=307,40752
        bad={"vertices6":self.candidate[sid]["vertices6"].copy(),"faces":self.candidate[sid]["faces"].copy()}
        bad["faces"][child]=bad["faces"][child][::-1]
        with self.assertRaisesRegex(ValueError,"child face"):
            V.validate_paired_union(stable_id=sid,current_face_row=child,parent_face_rows=[40752,40858],
                parent_row=self.parent_rows[sid],candidate_row=bad,ledger=self.ledger)
        bad={"vertices6":self.candidate[sid]["vertices6"].copy(),"faces":self.candidate[sid]["faces"].copy()}
        bad["vertices6"][0,0]+=np.float32(1e-6)
        with self.assertRaisesRegex(ValueError,"changed XYZ"):
            V.validate_paired_union(stable_id=sid,current_face_row=child,parent_face_rows=[40752,40858],
                parent_row=self.parent_rows[sid],candidate_row=bad,ledger=self.ledger)


if __name__=="__main__":
    unittest.main()
