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
    def test_custom_source_names_must_be_two_distinct_nonempty_strings(self):
        surface = {"vertices": [(0,0,0),(1,0,0),(0,1,0)],
                   "triangles": [(0,1,2)], "source_sha256": "0"*64}
        for names, surfaces in ((("same","same"), {"same": surface}),
                                (("", "other"), {"": surface, "other": surface}),
                                ((1, "other"), {1: surface, "other": surface})):
            with self.assertRaisesRegex(HumanImportError, "distinct named source surfaces"):
                partition.construct_arrangement(surfaces, source_names=names)
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


class NonzeroWindingSelfUnionTests(unittest.TestCase):
    faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))

    @classmethod
    def tetra(cls, offset=(0, 0, 0), scale=4, reverse=False):
        vertices = [
            tuple(F(offset[k]) + F(point[k]) for k in range(3))
            for point in ((0, 0, 0), (scale, 0, 0), (0, scale, 0), (0, 0, scale))
        ]
        faces = cls.faces
        if reverse:
            faces = tuple((a, c, b) for a, b, c in faces)
        return vertices, list(faces)

    @staticmethod
    def combine(parts):
        vertices, faces = [], []
        for part_vertices, part_faces in parts:
            offset = len(vertices)
            vertices.extend(part_vertices)
            faces.extend(tuple(index + offset for index in face) for face in part_faces)
        return vertices, faces

    @staticmethod
    def surface(vertices, faces):
        return {"vertices": vertices, "triangles": faces, "source_sha256": "a" * 64}

    @staticmethod
    def signed_volume(mesh):
        vertices = mesh["vertices_m"]
        return sum((predicates._dot(vertices[face[0]],
                                    predicates._cross(vertices[face[1]], vertices[face[2]]))
                    for face in mesh["triangles"]), F(0)) / 6

    @staticmethod
    def oriented_triangle_key(triangle):
        return min(tuple(triangle[index:] + triangle[:index]) for index in range(3))

    def test_unchanged_closed_tetra_is_retained_exactly(self):
        vertices, faces = self.tetra()
        result = partition.construct_nonzero_winding_self_union(self.surface(vertices, faces))
        output = result["output_mesh"]
        emitted = [tuple(output["vertices_m"][index] for index in face)
                   for face in output["triangles"]]
        authored = [tuple(vertices[index] for index in face) for face in faces]
        self.assertEqual({self.oriented_triangle_key(tri) for tri in emitted},
                         {self.oriented_triangle_key(tri) for tri in authored})
        self.assertEqual(self.signed_volume(output), F(32, 3))
        self.assertEqual(result["input_self_intersection_count"], 0)
        self.assertFalse(result["float32_conversion_audited"])

    def test_overlapping_tetra_self_union_has_exact_union_volume(self):
        first = self.tetra(scale=4)
        second = self.tetra(offset=(1, 1, 1), scale=4)
        vertices, faces = self.combine((first, second))
        result = partition.construct_nonzero_winding_self_union(self.surface(vertices, faces))
        output = result["output_mesh"]
        records = predicates._records(output["vertices_m"], output["triangles"])
        winding = predicates.prepare_signed_winding(records)
        self.assertGreater(result["input_self_intersection_count"], 0)
        self.assertEqual(result["output_self_intersection_audit"]["count"], 0)
        self.assertEqual(result["output_topology"]["closed_oriented_2_manifold"], True)
        self.assertEqual(self.signed_volume(output), F(127, 6))
        self.assertEqual(predicates.signed_winding_number((F(5, 4),) * 3, winding)["winding_number"], 1)
        self.assertEqual(result["output_topology"]["component_count"], 1)

    def test_nested_opposite_cavity_is_preserved(self):
        outer = self.tetra(scale=10)
        inner = self.tetra(offset=(1, 1, 1), scale=2, reverse=True)
        vertices, faces = self.combine((outer, inner))
        result = partition.construct_nonzero_winding_self_union(self.surface(vertices, faces))
        output = result["output_mesh"]
        winding = predicates.prepare_signed_winding(
            predicates._records(output["vertices_m"], output["triangles"]))
        self.assertEqual(result["output_topology"]["component_count"], 2)
        self.assertEqual(self.signed_volume(output), F(496, 3))
        self.assertEqual(predicates.signed_winding_number((F(5, 4),) * 3, winding)["winding_number"], 0)
        self.assertEqual(predicates.signed_winding_number((1, 1, 6), winding)["winding_number"], 1)

    def test_reversed_shell_is_reoriented_outward(self):
        vertices, faces = self.tetra(reverse=True)
        result = partition.construct_nonzero_winding_self_union(self.surface(vertices, faces))
        output = result["output_mesh"]
        winding = predicates.prepare_signed_winding(
            predicates._records(output["vertices_m"], output["triangles"]))
        self.assertEqual(self.signed_volume(output), F(32, 3))
        self.assertEqual(predicates.signed_winding_number((1, 1, 1), winding)["winding_number"], 1)
        self.assertEqual(result["reversed_patch_count"], 1)

    def test_coplanar_overlapping_shells_reject_closed_but_unsupported_arrangement(self):
        first = self.tetra(scale=4)
        second = self.tetra(scale=4)
        vertices, faces = self.combine((first, second))
        with self.assertRaisesRegex(HumanImportError, "coplanar self-intersection"):
            partition.construct_nonzero_winding_self_union(self.surface(vertices, faces))


    def test_point_only_contact_is_rejected(self):
        first = self.tetra(scale=4)
        second_vertices, second_faces = self.tetra(scale=4, reverse=True)
        second_vertices = [tuple(-value for value in point) for point in second_vertices]
        vertices, faces = self.combine((first, (second_vertices, second_faces)))
        with self.assertRaisesRegex(HumanImportError, "point-only self-contact"):
            partition.construct_nonzero_winding_self_union(self.surface(vertices, faces))

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
