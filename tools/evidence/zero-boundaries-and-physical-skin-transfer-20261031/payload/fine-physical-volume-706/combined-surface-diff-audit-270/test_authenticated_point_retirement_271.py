import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCRIPT = Path(__file__).with_name("audit_combined_current_surface_diff.py")
spec = importlib.util.spec_from_file_location("combined_surface_diff_retirement", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class AuthenticatedPointRetirementControls(unittest.TestCase):
    def setUp(self):
        self.base = np.asarray([[i, i*i, -i] for i in range(8)], dtype=np.float32)
        self.rows = np.asarray([6, 2, 7, 0], dtype=np.int64)
        self.candidate = self.base[self.rows]

    def test_exact_subset_and_compacted_output_rows(self):
        mapped = audit.point_subset_identity_map(
            self.base, self.candidate, "D", [1, 3, 4, 5])
        np.testing.assert_array_equal(mapped, self.rows)

    def test_rejects_unlisted_retired_row(self):
        with self.assertRaisesRegex(ValueError, "missing F32 rows"):
            audit.point_subset_identity_map(
                self.base, self.candidate, "D", [1, 3, 4, 6])

    def test_rejects_moved_or_added_point(self):
        moved = self.candidate.copy()
        moved[1, 0] = np.nextafter(moved[1, 0], np.float32(10))
        with self.assertRaisesRegex(ValueError, "moved or unauthorized new"):
            audit.point_subset_identity_map(
                self.base, moved, "D", [1, 3, 4, 5])

    def test_rejects_duplicate_retirement_authorization(self):
        with self.assertRaisesRegex(ValueError, "duplicate point rows"):
            audit.point_subset_identity_map(
                self.base, self.candidate, "D", [1, 3, 4, 4])

    def test_legacy_bijection_still_rejects_subset(self):
        with self.assertRaisesRegex(ValueError, "shape changed"):
            audit.point_identity_map(self.base, self.candidate, "D")

if __name__ == "__main__":
    unittest.main(verbosity=2)
