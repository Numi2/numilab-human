import importlib.util
import unittest
from pathlib import Path
import numpy as np

MODULE_PATH = Path(__file__).with_name("bridge_adaptive_031h.py")
spec = importlib.util.spec_from_file_location("bridge_adaptive_031h", MODULE_PATH)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

class BridgeIndexControls(unittest.TestCase):
    def test_fat_global_packet_index_maps_to_role_local(self):
        self.assertEqual(bridge.role_packet_row(37436, "existing_packet_anchor", 37436), 0)
        self.assertEqual(bridge.role_packet_row(37440, "existing_packet_anchor", 37436), 4)

    def test_implicit_packet_sentinel_is_not_offset(self):
        self.assertEqual(bridge.role_packet_row(-1, "implicit_centroid", 37436), -1)
        self.assertEqual(bridge.role_packet_row(-1, "source_tet_candidate", 37436), -1)

    def test_interleaved_global_child_uses_role_local_membership(self):
        global_rows = np.asarray([0, 2, 5], dtype=np.int64)
        ctx = {"cglobal": global_rows, "region_children": {69: {2}}}
        self.assertEqual(bridge.role_local_child(5, global_rows), 2)
        self.assertEqual(bridge.role_local_child(2, global_rows), 1)
        self.assertTrue(bridge.role_region_allows(ctx, 69, 5))
        self.assertFalse(bridge.role_region_allows(ctx, 69, 2))
        self.assertFalse(bridge.role_region_allows(ctx, 69, 7))

if __name__ == "__main__":
    unittest.main()
