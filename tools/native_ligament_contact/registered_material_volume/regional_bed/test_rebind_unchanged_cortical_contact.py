from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from rebind_unchanged_cortical_contact import rebind, sha256


class CorticalBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        self.source, self.old, self.new, self.out = [folder / s for s in ('binding', 'old', 'new', 'out')]
        self.coords = np.eye(4, 3)
        self.write_nodes(self.old, self.coords)
        self.source.write_text(
            'inline constexpr Facet faces[]={\n{0,0,15,2,1,{0,1,2},{1,1,1}},\n};\n'
            'inline constexpr Sample samples[]={\n{2,15,2,1,1.0,0.1},\n};\n'
            'inline constexpr char loaded[]="' + sha256(self.old) + '";\n'
        )
        self.poses = np.zeros((2, 7))
        self.poses[:, -1] = 1

    @staticmethod
    def write_nodes(path, coords):
        rows = np.column_stack((np.arange(1, 5), coords))
        np.savetxt(path, rows, header='4 3 0 0', comments='', fmt='%g')

    def test_distant_change_preserves_every_contact_row(self):
        self.coords[3, 1] += 0.01
        self.write_nodes(self.new, self.coords)
        pose = self.poses.copy()
        pose[0, 0] += 0.01
        result = rebind(self.source, self.old, self.new, self.poses, pose, self.out)
        self.assertEqual(result['contact_nodes'], 3)
        self.assertEqual(self.out.read_text(), self.source.read_text().replace(sha256(self.old), sha256(self.new)))

    def test_changed_sample_or_bone_requires_recooking(self):
        self.coords[2, 1] += 0.01
        self.write_nodes(self.new, self.coords)
        with self.assertRaisesRegex(ValueError, 'facet or sample'):
            rebind(self.source, self.old, self.new, self.poses, self.poses, self.out)
        self.new.write_bytes(self.old.read_bytes())
        pose = self.poses.copy()
        pose[1, 2] += 0.01
        with self.assertRaisesRegex(ValueError, 'body frame'):
            rebind(self.source, self.old, self.new, self.poses, pose, self.out)
        self.assertFalse(self.out.exists())
