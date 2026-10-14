"""Order-independent joins for captured geometry and material rows."""
from __future__ import annotations

import unittest

import numpy as np

from numilab_human.resting_anatomy_verify import gid_index_map, keyed_row_indices


class GIDIndexTests(unittest.TestCase):
    def test_unsorted_ids_reindex_rows_in_requested_order(self):
        source_gids = np.array([501, 17, 309], dtype=np.uint64)
        source_rows = np.array([[5.0], [1.7], [3.09]])
        requested_gids = np.array([309, 501, 309, 17], dtype=np.uint64)

        rows = source_rows[keyed_row_indices(source_gids, requested_gids)]

        np.testing.assert_array_equal(rows[:, 0], [3.09, 5.0, 3.09, 1.7])
        self.assertEqual(gid_index_map(source_gids), {501: 0, 17: 1, 309: 2})

    def test_duplicate_source_gids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate GID: 17"):
            gid_index_map(np.array([17, 17], dtype=np.int64))

    def test_missing_requested_gids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing GIDs"):
            keyed_row_indices(np.array([17, 501], dtype=np.int64),
                              np.array([99], dtype=np.int64))

    def test_noninteger_or_multidimensional_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional integer"):
            gid_index_map(np.array([[17, 501]], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "requested GIDs"):
            keyed_row_indices(np.array([17], dtype=np.int64), np.array([17.0]))


if __name__ == "__main__":
    unittest.main()
