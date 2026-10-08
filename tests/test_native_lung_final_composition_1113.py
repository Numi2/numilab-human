"""Mac-mini integration regressions for the pinned 1113 lung composition."""
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import tempfile
import unittest
import numpy as np

REPO=pathlib.Path(__file__).resolve().parents[1]
EVIDENCE=REPO/"tools/evidence/native-lung-final-selected-composition-1113"
COMPOSER_PATH=EVIDENCE/"compose_candidate_v8.py"
SOURCE_ROOT=pathlib.Path("/Users/n/numi-human-final-lung-composition-001")
BASE=pathlib.Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-conditioned-final-compose-1078/final")
FROZEN=pathlib.Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-final-selected-composition-dryrun-1113/provisional-1105-1106-v8")

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def load_composer():
    spec=importlib.util.spec_from_file_location("numi_lung_composer_1113_test",COMPOSER_PATH)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load exact composer")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

@unittest.skipUnless(
    COMPOSER_PATH.is_file() and BASE.joinpath("resting-thorax.nhanatomy").is_file()
    and FROZEN.joinpath("composition-report.json").is_file()
    and SOURCE_ROOT.joinpath("src/numilab_human/resting_lung_edge_repair.py").is_file(),
    "pinned Mac-mini composition inputs are unavailable",
)
class NativeLungFinalComposition1113Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pins=json.loads((EVIDENCE/"source-pins.json").read_text())
        commit=subprocess.check_output(["git","-C",str(SOURCE_ROOT),"rev-parse","HEAD"],text=True).strip()
        if commit!=cls.pins["source_checkout"]["commit"]:
            raise unittest.SkipTest("source checkout is not the pinned revision")
        dirty=subprocess.check_output(["git","-C",str(SOURCE_ROOT),"status","--porcelain"],text=True).strip()
        if dirty:
            raise unittest.SkipTest("pinned source checkout is dirty")
        if sha(COMPOSER_PATH)!=cls.pins["composer"]["sha256"]:
            raise unittest.SkipTest("composer is not the pinned script")
        for item in cls.pins["direct_owner_imports"]:
            if sha(item["path"])!=item["sha256"]:
                raise unittest.SkipTest(f"owner module is not pinned: {item['module']}")
        cls.tmp=tempfile.TemporaryDirectory(prefix="numi-lung-composition-1113-")
        cls.output=pathlib.Path(cls.tmp.name)/"rerun"
        cls.composer=load_composer()
        cls.composer.OUT=cls.output
        cls.composer.main()
        cls.report=json.loads((cls.output/"composition-report.json").read_text())
        cls.frozen=json.loads((FROZEN/"composition-report.json").read_text())

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,"tmp"): cls.tmp.cleanup()

    def test_1083_parent_records_survive_replacement_and_union_is_explicit(self):
        self.assertTrue(all(v["xyz_exact"] and v["other_faces_exact"] for v in self.report["1083_replay"].values()))
        rows=[json.loads(x) for x in (self.output/"current-reciprocal-face-map-v2.jsonl").read_text().splitlines()]
        unions=[x for x in rows if x.get("source_parent_relation")=="two_parent_inferred_reference_diagonal_flip"]
        self.assertEqual(len(unions),2)
        for row in unions:
            self.assertEqual(row["pair"],[307,309])
            self.assertTrue(all(len(side)==2 for side in row["parent_face_rows_1078"]))
            self.assertTrue(all(len(side)==2 for side in row["parent_face_rows_1055"]))
            self.assertTrue(all(len(side)==2 for side in row["source_face_ids_1055"]))
            self.assertEqual(row["operation_ref"]["sha256"],"df98cf7961a3f791d16d4f7eb7d5b6d63f1ac0b01df72aa22c360b70a34e34e0")
            self.assertEqual(row["source_map_record_ordinal_1055"][0],row["source_map_record_ordinal_1055"][1])

    def test_308_two_stage_inputs_and_normals_match_both_owner_checkpoints(self):
        stages=self.report["308_first_cluster_replay"]
        self.assertEqual([x["ordinal"] for x in stages],[1,2])
        self.assertTrue(all(x["exact_xyz_faces_recomputed_normals"] for x in stages))
        self.assertEqual(set(stages[0]["rows"]),{"308","310"})
        self.assertEqual(set(stages[1]["rows"]),{"308","310"})
        self.assertEqual((stages[0]["rows"]["308"]["drop_index"],stages[0]["rows"]["308"]["keep_index"]),(14267,14266))
        self.assertEqual((stages[0]["rows"]["310"]["drop_index"],stages[0]["rows"]["310"]["keep_index"]),(104412,104411))
        self.assertEqual((stages[1]["rows"]["308"]["drop_index"],stages[1]["rows"]["308"]["keep_index"]),(14266,14261))
        self.assertEqual((stages[1]["rows"]["310"]["drop_index"],stages[1]["rows"]["310"]["keep_index"]),(104411,13100))
        self.assertTrue(all(x["normal_bit_mismatches"]==0 for x in self.report["final_normal_checks"].values()))

    def test_candidate_and_manifest_keep_pending_qualification(self):
        receipt=json.loads((self.output/"final/resting-anatomy-receipt.json").read_text())
        manifest=json.loads((self.output/"final/resting-anatomy-manifest.json").read_text())
        base=json.loads((BASE/"resting-anatomy-receipt.json").read_text())
        self.assertEqual(receipt["qualification"]["source_geometry_candidate"],"Provisional 1078-based selected-patch dry-run. Not a final asset.")
        self.assertEqual(receipt["qualification"]["status"],base["qualification"]["status"])
        for key,value in base["qualification"].items():
            if key!="source_geometry_candidate":
                self.assertEqual(receipt["qualification"][key],value,key)
        self.assertEqual(manifest["qualification"],receipt["qualification"])
        self.assertEqual(manifest["receipt"]["sha256"],sha(self.output/"final/resting-anatomy-receipt.json"))

    def test_final_row310_topology_and_lineage_have_no_stage_sentinels(self):
        lineage=np.load(self.output/"row310-face-lineage.npy",allow_pickle=False)
        _,rows=self.composer.parse_payload(self.output/"final/resting-thorax.nhanatomy")
        top=self.report["final_topology"]["310"]
        self.assertEqual(lineage.shape,(len(rows[310]["faces"]),2))
        self.assertEqual(lineage.shape,(303656,2))
        self.assertTrue(set(np.unique(lineage[:,0]).tolist()).issubset({305,306,307,308,309}))
        self.assertTrue(np.all(lineage[:,1]>=0))
        for sid in (305,306,307,308,309):
            parent_count=len(self.composer.parse_payload(BASE/"resting-thorax.nhanatomy")[1][sid]["faces"])
            self.assertTrue(np.all(lineage[lineage[:,0]==sid,1]<parent_count))
        self.assertEqual((top["euler_characteristic"],top["face_component_count"],top["boundary_edge_count"]),(-32,2,0))
        self.assertTrue(all(v==0 for v in top["defects"].values()))
        self.assertEqual((self.report["final_topology"]["311"]["euler_characteristic"],self.report["final_topology"]["311"]["face_component_count"]),(-2,1))

    def test_historical_stage_lineage_is_distinguished_from_final_row310_map(self):
        supplement=json.loads((EVIDENCE/"row310-lineage-scope-correction.json").read_text())
        history=supplement["retained_historical_field"]
        final=supplement["authoritative_final_lineage"]
        self.assertEqual(history["value"],104975)
        self.assertIn("pre-pleura",history["interpretation"])
        self.assertFalse(final["synthetic_stage_parent_ids_survive"])
        self.assertEqual(final["sha256"],self.report["outputs"]["row310_lineage"]["sha256"])
        self.assertEqual(supplement["candidate"]["receipt_sha256"],self.frozen["outputs"]["final_receipt"]["sha256"])
        self.assertEqual(supplement["verification"]["qualification_effect"],"None. The frozen receipt and manifest remain provisional and unchanged.")

    def test_source_area_volume_and_final_payload_replay_are_stable(self):
        self.assertEqual(sha(self.output/"final/resting-thorax.nhanatomy"),sha(FROZEN/"final/resting-thorax.nhanatomy"))
        self.assertEqual(self.report["area_volume_f32_checks"],self.frozen["area_volume_f32_checks"])
        self.assertTrue(self.report["runtime_area_f32_unchanged"])
        for sid,check in self.report["area_volume_f32_checks"].items():
            self.assertTrue(check["area_f32_equal"],f"row {sid}")
            self.assertTrue(check["volume_f32_equal"],f"row {sid}")


    def test_reciprocal_map_is_complete_and_zero_pairs_are_explicit(self):
        report=json.loads((self.output/"current-reciprocal-map-report-v2.json").read_text())
        actual={tuple(row["pair"]):(row["face_pair_count"],row["status"]) for row in report["pairs"]}
        expected={(a,b) for a in range(305,310) for b in range(a+1,310)}
        self.assertEqual(set(actual),expected)
        self.assertEqual(sum(1 for count,status in actual.values() if status=="explicit_zero_no_map"),6)
        self.assertEqual(actual[(305,308)],(8884,"mapped"))
        self.assertEqual(actual[(306,307)],(8727,"mapped"))
        self.assertEqual(actual[(306,309)],(2010,"mapped"))
        self.assertEqual(actual[(307,309)],(9605,"mapped"))

if __name__=="__main__": unittest.main()
