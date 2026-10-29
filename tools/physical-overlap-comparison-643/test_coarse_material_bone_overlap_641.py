import copy
import unittest
from compare_coarse_material_bone_overlap_641 import compare

def fixture():
    return {"coordinate_field": "reference", "bank_manifest_sha256": "manifest", "script_sha256": "owner",
            "candidate_sha256": "candidate", "status": "coarse_material_bone_volume_inventory",
            "minimum_cell_volume_m3": 1e-12,
            "banks": [{"index": i, "body": i+1, "bone": i+2, "operand_sha256": str(i),
                       "layers": {m: {"intersection_m3": 0.0, "status": "Error.NoError"}
                                  for m in ("skin", "fat")}} for i in range(185)]}
class ComparisonTests(unittest.TestCase):
    def test_unchanged(self):
        a=fixture()
        self.assertEqual(compare(a,a)["positive_bank_count_after"], 0)
    def test_skin_gain_cannot_hide_fat_regression(self):
        a=fixture();a["banks"][45]["layers"]["skin"]["intersection_m3"]=6e-6
        a["banks"][45]["layers"]["fat"]["intersection_m3"]=70e-6
        b=copy.deepcopy(a)
        b["banks"][45]["layers"]["skin"]["intersection_m3"]=4e-6
        b["banks"][45]["layers"]["fat"]["intersection_m3"]=72e-6
        r=compare(a,b)
        self.assertEqual(r["status"], "DIAGNOSTIC_OVERLAP_REGRESSION")
        self.assertEqual(len(r["increased_material_bank_pairs"]), 1)
        self.assertFalse(r["physical_admission"])
    def test_any_positive_preserved_without_tolerance(self):
        a=fixture();b=copy.deepcopy(a)
        b["banks"][1]["layers"]["fat"]["intersection_m3"]=1e-30
        self.assertEqual(compare(a,b)["positive_bank_count_after"],1)
        self.assertEqual(compare(a,b)["status"],"DIAGNOSTIC_OVERLAP_REGRESSION")
    def test_rejects_incomplete_duplicate_or_different_operand(self):
        a=fixture()
        variants=[]
        b=copy.deepcopy(a);b["banks"].pop();variants.append(b)
        b=copy.deepcopy(a);b["banks"][-1]=b["banks"][0];variants.append(b)
        b=copy.deepcopy(a);b["banks"][0]["operand_sha256"]="changed";variants.append(b)
        b=copy.deepcopy(a);b["coordinate_field"]="other";variants.append(b)
        for b in variants:
            with self.subTest(b=b["coordinate_field"]),self.assertRaises(ValueError):compare(a,b)
    def test_rejects_nonfinite_failed_or_negative_intersection(self):
        a=fixture()
        for value,status in [(float("nan"),"Error.NoError"),(-1e-20,"Error.NoError"),(0.0,"Error.NotManifold")]:
            b=copy.deepcopy(a);b["banks"][0]["layers"]["skin"]={"intersection_m3":value,"status":status}
            with self.assertRaises(ValueError):compare(a,b)
    def test_decrease_is_diagnostic_only(self):
        a=fixture();a["banks"][0]["layers"]["fat"]["intersection_m3"]=1e-9
        b=fixture();r=compare(a,b)
        self.assertFalse(r["physical_admission"]);self.assertEqual(r["runtime"],0)
        self.assertEqual(r["material_counts"]["fat"]["decreased"],1)
if __name__=="__main__":unittest.main()
