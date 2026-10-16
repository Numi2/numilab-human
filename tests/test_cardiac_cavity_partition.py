"""Exact Boolean authoring regressions; no physical stepping."""
import sys
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from numilab_human import cardiac_cavity_intersections as predicates
from numilab_human import cardiac_cavity_partition as partition
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
    def test_refined_transverse_vertex_contact_retains_full_certificate(self):
        from numilab_human import cardiac_partition_certificate as certificate
        faces = [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
        vertices = [(0,0,0),(4,0,0),(0,4,0),(0,0,4)]
        first = {"vertices": vertices + [(1,1,2)],
                 "triangles": faces[:3] + [(1,2,4),(2,3,4),(3,1,4)],
                 "source_sha256": "a"*64}
        second = {"vertices": [tuple(x+1 for x in p) for p in vertices],
                  "triangles": faces, "source_sha256": "b"*64}
        names = ("first", "second")
        sources = dict(zip(names, (first, second)))
        children, report = partition.construct_arrangement(sources, source_names=names)
        self.assertGreater(report["supported_point_contact_count"], 0)
        self.assertTrue(all(type(value) is F
                            for row in children for point in row["vertices"] for value in point))
        self.assertTrue(all(type(value) is F
                            for row in report["intersection_segments_m"]
                            for point in row["endpoints"] for value in point))
        self.assertTrue(certificate.build_triangle_sets(children, source_names=names))
        proof = certificate.certify_partition(
            sources, children, source_names=names,
            expected_geometry_sha256={n:certificate.source_geometry_sha256(sources[n])
                                      for n in names},
            expected_source_component_counts={n:1 for n in names})
        self.assertTrue(proof)
        self.assertEqual({(r["source"], r["source_face"]) for r in children},
                         {(n,i) for n in names for i in range(len(sources[n]["triangles"]))})

    def test_isolated_nonparallel_point_contact_still_fails_closed(self):
        names = ("first", "second")
        surfaces = dict(zip(names, (
            {"vertices": [(0,0,0),(1,0,0),(0,1,0)],
             "triangles": [(0,1,2)], "source_sha256": "a"*64},
            {"vertices": [(0,0,0),(0,-1,1),(0,1,1)],
             "triangles": [(0,1,2)], "source_sha256": "b"*64})))
        with self.assertRaisesRegex(HumanImportError, "point-only face contact is unsupported"):
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

    def test_integer_classifier_fails_closed_after_all_64_rays_are_ambiguous(self):
        records = []
        for attempt in range(1, 65):
            triangle = ((-1, -attempt, 0), (-2, -2 * attempt, 0),
                        (-1, -attempt, -1))
            lower = tuple(min(point[axis] for point in triangle) for axis in range(3))
            upper = tuple(max(point[axis] for point in triangle) for axis in range(3))
            vertex_base = 3 * (attempt - 1)
            records.append((triangle, lower, upper, attempt - 1,
                            (vertex_base, vertex_base + 1, vertex_base + 2)))
        classifier = partition.IntegerRayClassifier(records)
        with self.assertRaisesRegex(HumanImportError,
                                   "indeterminate exact ray classification"):
            classifier.location((0, 0, 0))

    def test_integer_classifier_uses_local_lattice_on_huge_global_coordinates(self):
        scale = 1 << 2800
        offset = (1 << 2900, -(1 << 2898), 1 << 2887)
        vertices = tuple(tuple(offset[k] + scale * point[k] for k in range(3))
                         for point in ((0, 0, 0), (12, 0, 0), (0, 12, 0), (0, 0, 12)))
        records = predicates._records(vertices, [(0,2,1),(0,1,3),(0,3,2),(1,2,3)])
        classifier = partition.IntegerRayClassifier(records)
        for point, expected in (
                (tuple(offset[k] + scale for k in range(3)), "inside"),
                (tuple(offset[k] - scale for k in range(3)), "outside"),
                (offset, "boundary"),
                (tuple(F( offset[k]) + F(scale, 8) for k in range(3)), "inside")):
            with self.subTest(expected=expected):
                self.assertEqual(classifier.location(point), expected)

    def test_float_conversion_cannot_merge_rational_vertices(self):
        with self.assertRaisesRegex(HumanImportError, "collapsed"):
            partition.indexed_mesh([((F(1),F(0),F(0)), (F(1)+F(1,2**100),F(0),F(0)), (F(0),F(1),F(0)))])

    def test_directly_copied_reversed_face_keeps_corner_payload_aligned(self):
        # This models an inherited source face whose vertices and corner maps
        # were copied from an existing row before the owner reverses its winding.
        inherited_face = (17, 29, 41)
        inherited_corner_bary = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        emitted_face, emitted_bary = partition.orient_triangle_corner_payload(
            inherited_face, inherited_corner_bary, reverse=True)
        self.assertEqual(emitted_face, (17, 41, 29))
        self.assertEqual(emitted_bary,
                         (inherited_corner_bary[0], inherited_corner_bary[2],
                          inherited_corner_bary[1]))
        self.assertEqual(tuple(zip(emitted_face, emitted_bary)),
                         tuple((inherited_face[i], inherited_corner_bary[i])
                               for i in (0, 2, 1)))

    def test_triangle_corner_payload_is_unchanged_without_reversal(self):
        face = (3, 5, 7)
        payload = (("a0",), ("a1",), ("a2",))
        self.assertEqual(partition.orient_triangle_corner_payload(
            face, payload), (face, payload))

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

    def test_declared_opposed_full_face_is_retained_but_not_cut_or_ray_classified(self):
        triangle = ((F(0), F(0), F(0)), (F(1), F(0), F(0)), (F(0), F(1), F(0)))
        first = {"vertices": triangle, "triangles": [(0,1,2)], "source_sha256": "a"*64}
        second = {"vertices": triangle, "triangles": [(0,2,1)], "source_sha256": "b"*64}
        children, report = partition.construct_arrangement(
            {"first": first, "second": second}, source_names=("first", "second"),
            declared_feature_pairs={(0,0): "opposed_full_face"})
        self.assertEqual(report["declared_boundary_feature_counts"], {"opposed_full_face": 1})
        self.assertEqual(report["declared_boundary_feature_pair_count"], 1)
        self.assertEqual({(row["source"], row["source_face"], row["other_location"])
                          for row in children},
                         {("first",0,"outside"), ("second",0,"outside")})
        self.assertEqual(len(children), 2)

    def test_explicit_partial_opposed_coplanar_overlap_is_split_as_common_inside_patch(self):
        first_vertices = ((F(0),F(0),F(0)), (F(4),F(0),F(0)), (F(0),F(4),F(0)))
        second_vertices = ((F(4),F(0),F(0)), (F(2),F(1),F(0)), (F(0),F(4),F(0)))
        first = {"vertices": first_vertices, "triangles": [(0,1,2)], "source_sha256": "a"*64}
        second = {"vertices": second_vertices, "triangles": [(0,1,2)], "source_sha256": "b"*64}
        first_exact = tuple(tuple(F(x) for x in point) for point in first_vertices)
        second_exact = tuple(tuple(F(x) for x in point) for point in second_vertices)
        normal = predicates._cross(predicates._sub(first_exact[1], first_exact[0]),
                                   predicates._sub(first_exact[2], first_exact[0]))
        polygon = partition._coplanar_overlap_polygon(first_exact, second_exact, normal)
        self.assertEqual(set(polygon), set(second_exact))

        def boundary_aware_location(point):
            return ("boundary" if partition._point_in_convex_polygon(point, polygon, normal)
                    else "outside")

        with patch.object(partition.IntegerRayClassifier, "location",
                          side_effect=boundary_aware_location):
            children, report = partition.construct_arrangement(
                {"first": first, "second": second}, source_names=("first", "second"),
                coplanar_priority_pairs={(0,0)})

        self.assertEqual(report["coplanar_priority_overlap_pair_count"], 1)
        self.assertEqual(len(report["intersection_segments_m"]), 3)
        self.assertEqual(len([row for row in children if row["source"] == "first"]), 3)
        self.assertEqual(len([row for row in children if row["source"] == "second"]), 1)
        inside = {name: [row["vertices"] for row in children
                         if row["source"] == name and row["other_location"] == "inside"]
                  for name in ("first", "second")}
        self.assertEqual(len(inside["first"]), 1)
        self.assertEqual(len(inside["second"]), 1)
        self.assertEqual(set(inside["first"][0]), set(inside["second"][0]))
        n_first = predicates._cross(predicates._sub(*inside["first"][0][1:]),
                                    predicates._sub(inside["first"][0][2], inside["first"][0][0]))
        n_second = predicates._cross(predicates._sub(*inside["second"][0][1:]),
                                     predicates._sub(inside["second"][0][2], inside["second"][0][0]))
        self.assertLess(predicates._dot(n_first, n_second), 0)
        self.assertTrue(all(row["other_location"] != "boundary" for row in children))

        def area_vector(rows):
            total = (F(0), F(0), F(0))
            for row in rows:
                a, b, c = row["vertices"]
                normal = predicates._cross(predicates._sub(b,a), predicates._sub(c,a))
                total = tuple(total[axis] + normal[axis] for axis in range(3))
            return total
        self.assertEqual(area_vector([row for row in children if row["source"] == "first"]), normal)
        second_normal = predicates._cross(predicates._sub(second_exact[1], second_exact[0]),
                                          predicates._sub(second_exact[2], second_exact[0]))
        self.assertEqual(area_vector([row for row in children if row["source"] == "second"]),
                         second_normal)

    def test_coplanar_priority_rejects_same_winding_or_missing_pairs(self):
        first_vertices = ((F(0),F(0),F(0)), (F(4),F(0),F(0)), (F(0),F(4),F(0)))
        same_winding = {"vertices": first_vertices, "triangles": [(0,1,2)], "source_sha256": "b"*64}
        first = {"vertices": first_vertices, "triangles": [(0,1,2)], "source_sha256": "a"*64}
        with self.assertRaisesRegex(HumanImportError, "opposite winding"):
            partition.construct_arrangement(
                {"first": first, "second": same_winding}, source_names=("first", "second"),
                coplanar_priority_pairs={(0,0)})
        second_vertices = ((F(5),F(0),F(0)), (F(6),F(0),F(0)), (F(5),F(1),F(0)))
        second = {"vertices": second_vertices, "triangles": [(0,1,2)], "source_sha256": "c"*64}
        with self.assertRaisesRegex(HumanImportError, "absent from the exact intersurface contact set"):
            partition.construct_arrangement(
                {"first": first, "second": second}, source_names=("first", "second"),
                coplanar_priority_pairs={(0,0)})

    def test_actual_forearm_coplanar_witness_subdivides_both_edge_supported_faces(self):
        first_vertices = (
            (F(23691324), F(-125515960), F(15897056)),
            (F(23691326), F(-125515984), F(15897049)),
            (F(23691324), F(-125515968), F(15897048)),
        )
        second_vertices = (
            first_vertices[1],
            (F(23691320), F(-125515928), F(15897054)),
            first_vertices[2],
        )
        edge_neighbor = (first_vertices[0][0], first_vertices[0][1],
                         first_vertices[0][2] + 100)
        normal = predicates._cross(predicates._sub(first_vertices[1], first_vertices[0]),
                                   predicates._sub(first_vertices[2], first_vertices[0]))
        polygon = partition._coplanar_overlap_polygon(first_vertices, second_vertices, normal)
        self.assertEqual(set(polygon), {
            first_vertices[1], first_vertices[2],
            (F(23691324), F(-376547896, 3), F(47691152, 3)),
        })

        first = {"vertices": first_vertices + (edge_neighbor,),
                 "triangles": [(0,1,2), (0,2,3)], "source_sha256": "a"*64}
        second = {"vertices": second_vertices, "triangles": [(0,1,2)], "source_sha256": "b"*64}
        with patch.object(partition.IntegerRayClassifier, "location", return_value="outside"):
            rows, report = partition.construct_arrangement(
                {"FDS": first, "FDP": second}, source_names=("FDS", "FDP"),
                coplanar_priority_pairs={(0,0)})

        self.assertEqual(report["coplanar_priority_overlap_pair_count"], 1)
        self.assertEqual(len(report["intersection_segments_m"]), 4)
        fds_target = [row for row in rows if row["source"] == "FDS" and row["source_face"] == 0]
        fds_neighbor = [row for row in rows if row["source"] == "FDS" and row["source_face"] == 1]
        fdp_target = [row for row in rows if row["source"] == "FDP"]
        self.assertEqual((len(fds_target), len(fds_neighbor), len(fdp_target)), (2, 2, 2))
        inside_fds = [row for row in fds_target if row["other_location"] == "inside"]
        inside_fdp = [row for row in fdp_target if row["other_location"] == "inside"]
        self.assertEqual((len(inside_fds), len(inside_fdp)), (1, 1))
        self.assertEqual(set(inside_fds[0]["vertices"]), set(polygon))
        self.assertEqual(set(inside_fdp[0]["vertices"]), set(polygon))
        point = (F(23691324), F(-376547896, 3), F(47691152, 3))
        self.assertTrue(all(point in row["vertices"] for row in fds_neighbor))

    def test_declared_shared_feature_does_not_hide_an_unlisted_coplanar_pair(self):
        triangle = ((F(0), F(0), F(0)), (F(1), F(0), F(0)), (F(0), F(1), F(0)))
        first = {"vertices": triangle, "triangles": [(0,1,2)], "source_sha256": "a"*64}
        second = {"vertices": triangle, "triangles": [(0,2,1), (0,2,1)], "source_sha256": "b"*64}
        with self.assertRaisesRegex(HumanImportError, "coplanar or tangent face intersection is unsupported"):
            partition.construct_arrangement(
                {"first": first, "second": second}, source_names=("first", "second"),
                declared_feature_pairs={(0,0): "opposed_full_face"})

    def test_declared_full_face_requires_opposed_full_geometry(self):
        triangle = ((F(0), F(0), F(0)), (F(1), F(0), F(0)), (F(0), F(1), F(0)))
        first = {"vertices": triangle, "triangles": [(0,1,2)], "source_sha256": "a"*64}
        same_winding = {"vertices": triangle, "triangles": [(0,1,2)], "source_sha256": "b"*64}
        with self.assertRaisesRegex(HumanImportError, "not oppositely oriented"):
            partition.construct_arrangement(
                {"first": first, "second": same_winding}, source_names=("first", "second"),
                declared_feature_pairs={(0,0): "opposed_full_face"})


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

class PositiveWindingSelfUnionTests(unittest.TestCase):
    tetra = staticmethod(NonzeroWindingSelfUnionTests.tetra)
    combine = staticmethod(NonzeroWindingSelfUnionTests.combine)
    surface = staticmethod(NonzeroWindingSelfUnionTests.surface)
    signed_volume = staticmethod(NonzeroWindingSelfUnionTests.signed_volume)

    def test_opposite_overlapping_lobes_positive_material_is_exact_difference(self):
        first = self.tetra(scale=4)
        second = self.tetra(offset=(1, 1, 1), scale=4, reverse=True)
        vertices, faces = self.combine((first, second))
        source = self.surface(vertices, faces)
        # Nonzero material includes both lobes meeting on a four-face edge.
        with self.assertRaisesRegex(HumanImportError, "exactly two incident faces"):
            partition.construct_nonzero_winding_self_union(source)
        result = partition.construct_positive_winding_self_union(source)
        self.assertEqual(result["material_rule"], "strictly_positive")
        self.assertEqual(result["output_self_intersection_audit"]["count"], 0)
        self.assertTrue(result["output_topology"]["closed_oriented_2_manifold"])
        self.assertEqual(self.signed_volume(result["output_mesh"]), F(63, 6))
        winding = predicates.prepare_signed_winding(predicates._records(
            result["output_mesh"]["vertices_m"], result["output_mesh"]["triangles"]))
        for point, expected in [((F(1, 2),) * 3, 1),
                                ((F(5, 4),) * 3, 0),
                                ((F(2),) * 3, 0)]:
            self.assertEqual(predicates.signed_winding_number(point, winding)["winding_number"], expected)
        self.assertEqual(len(result["output_face_ancestry"]),
                         len(result["output_mesh"]["triangles"]))

    def test_positive_material_retains_nested_opposite_cavity(self):
        vertices, faces = self.combine((self.tetra(scale=10),
            self.tetra(offset=(1, 1, 1), scale=2, reverse=True)))
        result = partition.construct_positive_winding_self_union(self.surface(vertices, faces))
        self.assertEqual(result["output_topology"]["component_count"], 2)
        self.assertEqual(self.signed_volume(result["output_mesh"]), F(496, 3))
        winding = predicates.prepare_signed_winding(predicates._records(
            result["output_mesh"]["vertices_m"], result["output_mesh"]["triangles"]))
        self.assertEqual(predicates.signed_winding_number((F(5, 4),) * 3, winding)["winding_number"], 0)
        self.assertEqual(predicates.signed_winding_number((1, 1, 6), winding)["winding_number"], 1)

    def test_positive_material_rejects_globally_reversed_shell(self):
        vertices, faces = self.tetra(reverse=True)
        with self.assertRaisesRegex(HumanImportError, "selected winding material has no boundary"):
            partition.construct_positive_winding_self_union(self.surface(vertices, faces))


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




class CoveredPointUnionTests(unittest.TestCase):
    @staticmethod
    def source():
        outward = [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
        vertices = [(0,0,0),(-2,0,0),(0,-2,0),(0,0,-2),
                    (0,0,0),(2,0,0),(0,2,0),(0,0,2),
                    (-10,-11,-12),(30,-9,-8),(-7,31,-10),(-8,-6,32)]
        faces = [(a,c,b) for a,b,c in outward]
        faces += [tuple(i+4 for i in face) for face in outward]
        faces += [tuple(i+8 for i in face) for face in outward]
        return dict(vertices=vertices,triangles=faces,source_sha256="f"*64)

    def test_explicit_internal_fill_resolves_point_pinch_and_keeps_exact_boundary(self):
        result = partition.construct_positive_winding_self_union(
            self.source(), point_contact_cover_face_indices=[8,9,10,11])
        self.assertGreater(len(result["covered_point_contacts"]), 0)
        self.assertEqual(result["output_self_intersection_audit"]["count"], 0)
        self.assertEqual(result["output_topology"]["component_count"], 1)
        self.assertEqual(len(result["output_mesh"]["triangles"]), 4)
        self.assertEqual({row["source_face"] for row in result["output_face_ancestry"]},
                         {8,9,10,11})
        self.assertTrue(all(row["point"] == (F(0),)*3
                            for row in result["covered_point_contacts"]))

    def test_default_strict_point_contact_rejection_is_unchanged(self):
        with self.assertRaisesRegex(HumanImportError,"point-only self-contact is unsupported"):
            partition.construct_positive_winding_self_union(self.source())

    def test_a_cover_boundary_is_not_a_strict_interior_waiver(self):
        with self.assertRaisesRegex(HumanImportError,"not strictly inside"):
            partition.construct_positive_winding_self_union(
                self.source(),point_contact_cover_face_indices=[0,1,2,3])

    def test_invalid_open_and_reversed_covers_fail_before_construction(self):
        for selected in ([],[True],[8,8],[-1],[12],[8,9,10]):
            with self.subTest(selected=selected), self.assertRaises(HumanImportError):
                partition.construct_positive_winding_self_union(
                    self.source(),point_contact_cover_face_indices=selected)
        source=self.source()
        source["triangles"][8:]=[(a,c,b) for a,b,c in source["triangles"][8:]]
        with self.assertRaisesRegex(HumanImportError,"outward"):
            partition.construct_positive_winding_self_union(
                source,point_contact_cover_face_indices=[8,9,10,11])


if __name__ == "__main__":
    unittest.main()
