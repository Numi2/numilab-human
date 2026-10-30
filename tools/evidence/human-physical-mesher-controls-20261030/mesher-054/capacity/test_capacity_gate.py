import unittest
from capacity_gate import evaluate_role_capacity

class CapacityGateTests(unittest.TestCase):
    def setUp(self):
        self.profile={"name":"fixture","max_nodes_per_child":10,"max_cells_per_child":20}
        self.good={"status":"POSITIVE_F32_CELLS_NOT_PHYSICAL_ADMISSION","nodes":10,"tetrahedra":20,"positive_f32_cells":20}
    def test_at_limits_passes_role_counts_only(self):
        r=evaluate_role_capacity(self.good,self.profile)
        self.assertEqual(r["status"],"PASS_PER_ROLE_COUNTS_ONLY")
        self.assertTrue(r["node_limit_pass"]); self.assertTrue(r["cell_limit_pass"])
        self.assertEqual(r["combined_skin_fat_capacity"],"NOT_EVALUATED")
        self.assertFalse(r["physical_admission"])
    def test_node_over_limit_fails(self):
        r=evaluate_role_capacity({**self.good,"nodes":11},self.profile)
        self.assertFalse(r["node_limit_pass"])
    def test_cell_over_limit_fails(self):
        r=evaluate_role_capacity({**self.good,"tetrahedra":21,"positive_f32_cells":21},self.profile)
        self.assertFalse(r["cell_limit_pass"])
    def test_positive_count_mismatch_fails(self):
        r=evaluate_role_capacity({**self.good,"positive_f32_cells":19},self.profile)
        self.assertFalse(r["positive_count_matches_tetrahedra"])
    def test_owner_positive_status_is_required(self):
        bad={**self.good,"status":"INVALID"}
        r=evaluate_role_capacity(bad,self.profile)
        self.assertFalse(r["owner_positive_f32_status_pass"])
        self.assertEqual(r["status"],"FAIL_PER_ROLE_CAPACITY_OR_POSITIVE_CELL_CHECK")
    def test_missing_or_invalid_profile_fails_closed(self):
        r=evaluate_role_capacity({"nodes":4},self.profile)
        self.assertEqual(r["status"],"FAIL_PER_ROLE_CAPACITY_OR_POSITIVE_CELL_CHECK")
        r=evaluate_role_capacity(self.good,{"name":"bad","max_nodes_per_child":0,"max_cells_per_child":20})
        self.assertFalse(r["node_limit_pass"])
if __name__=="__main__": unittest.main()
