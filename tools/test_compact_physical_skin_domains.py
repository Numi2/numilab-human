"""Regression tests for physical active-domain and compact role namespaces."""
import copy
import unittest
import numpy as np
from compact_physical_skin_domains import prepare_paired_active_domains, compact_volume


def fixture():
    # Row 6 is no longer on I faces but still supports a Skin interior cell.
    # It must remain physical, without being fabricated as a shared I boundary.
    skin = np.array([[0,0,1],[1,0,1],[0,1,1],
                     [0,0,0],[1,0,0],[0,1,0],[.2,.2,.2],[9,9,9]], np.float32)
    fat = np.array([[0,0,0],[1,0,0],[0,1,0],
                    [0,0,-1],[1,0,-1],[0,1,-1],[.2,.2,.2],[9,9,9]], np.float32)
    t = np.array([[3,4,5,2],[3,4,2,1],[3,0,1,2]], np.int64)
    st = np.vstack((t, [3,4,5,6]))
    tri = np.array([[0,1,2]], np.int64)
    ip = skin[[3,4,5,6,7]]
    inputs = {
        "O": {"points_world_m": skin[[0,1,2,7]], "faces": tri,
              "volume_node_rows": {"Skin": np.array([0,1,2,7])}},
        "I": {"points_world_m": ip, "faces": tri,
              "volume_node_rows": {"Skin": np.array([3,4,5,6,7]),
                                   "Fat": np.array([0,1,2,6,7])}},
        "D": {"points_world_m": fat[[3,4,5,7]], "faces": tri,
              "volume_node_rows": {"Fat": np.array([3,4,5,7])}}}
    return skin, st, fat, t, inputs


class ActiveDomainTests(unittest.TestCase):
    def test_retired_role_point_can_remain_physical_interior(self):
        args = fixture()
        before = [x.copy() for x in args[:4]]
        r = prepare_paired_active_domains(*args)
        self.assertEqual(len(r["points_world_m"]), 10)
        self.assertEqual(len(r["tetrahedra"]), 7)
        self.assertEqual(r["volumes"]["Skin"]["input_node_to_output_node"][6], 6)
        self.assertEqual(r["volumes"]["Fat"]["input_node_to_output_node"][6], -1)
        self.assertEqual(r["roles"]["I"]["input_source_vertex_to_output_source_vertex"][3], -1)
        np.testing.assert_array_equal(r["roles"]["O"]["joint_node_rows"], [0,1,2])
        np.testing.assert_array_equal(r["points_world_m"][r["tetrahedra"]].view(np.uint32),
            np.vstack((args[0][args[1]], args[2][args[3]])).view(np.uint32))
        for old, actual in zip(before, args[:4]):
            np.testing.assert_array_equal(old, actual)

    def test_independent_node_permutations_preserve_cells_and_original_source_ids(self):
        a = list(fixture())
        baseline = prepare_paired_active_domains(*a)
        for k in (0,2):
            perm = np.array([7,4,6,1,5,0,3,2])
            inv = np.argsort(perm)
            a[k] = a[k][perm]
            a[k+1] = inv[a[k+1]]
            label = "Skin" if k == 0 else "Fat"
            for role in a[4].values():
                if label in role["volume_node_rows"]:
                    role["volume_node_rows"][label] = inv[role["volume_node_rows"][label]]
        r = prepare_paired_active_domains(*a)
        np.testing.assert_array_equal(r["points_world_m"][r["tetrahedra"]],
                                      baseline["points_world_m"][baseline["tetrahedra"]])
        for role in ("O","I","D"):
            np.testing.assert_array_equal(r["roles"][role]["faces"], baseline["roles"][role]["faces"])
            np.testing.assert_array_equal(r["roles"][role]["output_source_vertex_to_input_source_vertex"],
                                          baseline["roles"][role]["output_source_vertex_to_input_source_vertex"])

    def test_invalid_or_incomplete_maps_fail_closed(self):
        mutations = [
            lambda a: a[4]["O"]["volume_node_rows"]["Skin"].__setitem__(0, 1),
            lambda a: a[4]["I"]["volume_node_rows"].pop("Fat"),
            lambda a: a[4]["D"]["faces"].__setitem__((0,0), 3), # retired orphan
            lambda a: a[4]["O"]["faces"].__setitem__((0,0), 1), # collapsed face
            lambda a: a[4]["O"].__setitem__("points_world_m", a[4]["O"]["points_world_m"].astype(np.float64)),
        ]
        for mutate in mutations:
            a = list(copy.deepcopy(fixture()))
            mutate(a)
            with self.assertRaises(ValueError):
                prepare_paired_active_domains(*a)

    def test_compaction_keeps_distinct_nearby_and_equal_coordinate_node_ids(self):
        a = fixture()
        p = a[0].copy()
        p[7] = p[6]
        t = np.vstack((a[1], [0,1,2,7]))
        r = compact_volume(p, t)
        self.assertEqual(len(r["points_world_m"]), 8)
        self.assertNotEqual(r["input_node_to_output_node"][6],
                            r["input_node_to_output_node"][7])
        with self.assertRaises(ValueError):
            compact_volume(p, t.astype(np.float64))


if __name__ == "__main__":
    unittest.main()
