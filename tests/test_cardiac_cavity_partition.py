"""Exact Boolean authoring regressions; no physical stepping."""
from fractions import Fraction as F
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from numilab_human import cardiac_cavity_partition as partition
from numilab_human import cardiac_cavity_intersections as predicates
from numilab_human.cardiac_cavity_geometry import extract_cavity_surfaces
from numilab_human.model import ImportError as HumanImportError


class AuthoringTests(unittest.TestCase):
    def test_command_preserves_existing_output_and_rejects_redirection(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "artifact.json"
            target.write_text("original bytes\n")
            with patch.object(partition, "compile_partitions", return_value={}):
                self.assertEqual(partition.main(["--output", str(target)]), 1)
                link = Path(directory) / "redirect.json"
                link.symlink_to(target)
                self.assertEqual(partition.main(["--output", str(link)]), 1)
            self.assertEqual(target.read_text(), "original bytes\n")

    def test_integer_classifier_exact_boundary_and_rational_queries(self):
        vertices = [(0,0,0),(12,0,0),(0,12,0),(0,0,12)]
        records = predicates._records(vertices, [(0,2,1),(0,1,3),(0,3,2),(1,2,3)])
        classifier = partition.IntegerRayClassifier(records)
        for point in ((1,1,1), (F(1,100000000000000000000),)*3,
                      (4,4,4), (-1,-1,-1), (12,0,0), (5,5,5), (0,2,3), (20,20,20)):
            self.assertEqual(classifier.location(point), predicates.point_location(point, records)["location"])

    def test_float_conversion_cannot_merge_rational_vertices(self):
        with self.assertRaisesRegex(HumanImportError, "collapsed"):
            partition.indexed_mesh([((F(1),F(0),F(0)), (F(1)+F(1,2**100),F(0),F(0)), (F(0),F(1),F(0)))])

    def test_global_conversion_is_identical_for_reversed_shared_faces(self):
        tri = ((F(1,3),F(0),F(0)),(F(0),F(1,7),F(0)),(F(0),F(0),F(1,11)))
        a, b = partition.indexed_mesh([tri]), partition.indexed_mesh([tri[::-1]])
        self.assertEqual(a["vertices_m"], b["vertices_m"])
        self.assertEqual(a["triangles"][0], b["triangles"][0][::-1])

    def test_transverse_authoring_rejects_tangent_and_coplanar_contacts(self):
        faces = [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
        vertices = [(F(0),F(0),F(0)),(F(1),F(0),F(0)),(F(0),F(1),F(0)),(F(0),F(0),F(1))]
        def surface(v): return {"vertices":v, "triangles":faces, "source_sha256":"0"*64}
        a = surface(vertices)
        for b in (a, surface([tuple(p[k]+(1 if k==0 else 0) for k in range(3)) for p in vertices])):
            with self.assertRaisesRegex(HumanImportError, "unsupported"):
                partition.construct_arrangement(dict(zip(partition.NAMES, (a,b))))


class PinnedArrangementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cavities = extract_cavity_surfaces()
        cls.source = partition.rational_source_surfaces(cls.cavities)
        cls.rows, cls.provenance = partition.construct_arrangement(cls.source)

    def test_exact_published_coordinates_and_source_hashes(self):
        for source in self.cavities["chambers"][:2]:
            actual = self.source[source["source_id"]]
            self.assertEqual(actual["source_sha256"], source["source"]["sha256"])
            self.assertEqual([[float(x) for x in p] for p in actual["vertices"]],
                             source["exact_coordinate_quotient"]["vertices_m"])

    def test_actual_source_arrangement_counts_and_parent_coverage(self):
        self.assertEqual(self.provenance["source_intersecting_triangle_pair_count"], 42)
        self.assertEqual(self.provenance["source_face_count"], 4162)
        self.assertEqual(len(self.rows), 4330)
        self.assertEqual({(row["source"], row["source_face"]) for row in self.rows},
                         {(name,i) for name in partition.NAMES for i in range(len(self.source[name]["triangles"]))})
        self.assertEqual({name:sum(row["source"] == name and row["other_location"] == "inside" for row in self.rows)
                          for name in partition.NAMES}, {"right_atrium":42, "right_ventricle":56})
        source_points = {p for surface in self.source.values() for p in surface["vertices"]}
        result_points = {p for row in self.rows for p in row["vertices"]}
        self.assertTrue(source_points <= result_points)
        self.assertEqual(len(result_points-source_points), 42)


if __name__ == "__main__":
    unittest.main()
