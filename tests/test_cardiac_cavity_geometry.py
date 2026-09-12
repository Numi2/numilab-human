from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from numilab_human.cardiac_cavity_geometry import (
    ARCHIVE, ARCHIVE_BYTES, ARCHIVE_SHA256, CAVITIES, ROOT, analyze_topology,
    exact_coordinate_quotient, extract_cavity_surfaces, main, parse_obj,
)
from numilab_human.model import ImportError as HumanImportError
from numilab_human.physiology import canonical, read_json

# Consistently outward tetrahedron; a minimal connected orientable sphere.
VERTICES = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
FACES = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
OBJ = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"


class CardiacCavityGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.surfaces = extract_cavity_surfaces()

    def test_exact_source_identity_and_cavity_membership(self) -> None:
        result = self.surfaces
        self.assertEqual(result['archive'], {'file': ARCHIVE, 'sha256': ARCHIVE_SHA256, 'bytes': ARCHIVE_BYTES})
        for chamber, (source_id, concept, name, member, sha) in zip(result['chambers'], CAVITIES):
            self.assertEqual((chamber['source_id'], chamber['concept_id'], chamber['source_name'], chamber['member_id']),
                             (source_id, concept, name, member))
            self.assertEqual(chamber['source']['sha256'], sha)
            self.assertEqual(chamber['semantic_id'], 'FMA:' + concept[3:])
            self.assertIn('# Bounds(mm):', '\n'.join(chamber['source']['header_comments']))
        self.assertEqual(result['license_provenance']['source_header_license'], 'CC-BY-SA-2.1-Japan')
        self.assertEqual(result['license_provenance']['current_database_license'], 'CC-BY-4.0')
        self.assertFalse(result['source_coordinate_edits'])
        self.assertEqual(result['added_faces'], 0)

    def test_obj_seams_are_identified_without_coordinate_or_triangle_changes(self) -> None:
        for chamber, boundary, identified in zip(self.surfaces['chambers'], [146, 34, 126, 42], [69, 12, 53, 16]):
            quotient = chamber['exact_coordinate_quotient']
            self.assertEqual(chamber['raw_topology']['boundary_edge_count'], boundary)
            self.assertFalse(chamber['raw_topology']['closed_oriented_manifold_candidate'])
            self.assertEqual(quotient['identified_vertex_count'], identified)
            for i, j in enumerate(quotient['source_vertex_to_vertex']):
                self.assertEqual(chamber['vertices_m'][i], quotient['vertices_m'][j])
            for raw, mapped in zip(chamber['triangles'], quotient['triangles']):
                self.assertEqual(mapped, [quotient['source_vertex_to_vertex'][i] for i in raw])
            topology = quotient['topology']
            self.assertTrue(topology['closed_oriented_manifold_candidate'])
            self.assertEqual(topology['face_component_count'], 1)
            self.assertEqual(topology['euler_characteristic'], 2)
            self.assertEqual(topology['vertex_manifold_defect_ids'], [])
            self.assertIsNone(chamber['physical_volume_m3'])
            self.assertIsNone(chamber['mechanical_mass_kg'])
            self.assertEqual(topology['self_intersection_status'], 'not_checked')
            self.assertEqual(topology['interdomain_overlap_status'], 'not_checked')

    def test_source_extraction_is_deterministic(self) -> None:
        self.assertEqual(canonical(self.surfaces), canonical(extract_cavity_surfaces()))

    def test_source_archive_and_relation_locks_cannot_be_rebound(self) -> None:
        for path, value in ((['files', ARCHIVE, 'sha256'], '0' * 64), (['files', ARCHIVE, 'bytes'], 1),
                            (['files', 'partof_element_parts.txt', 'sha256'], '0' * 64),
                            (['files', 'isa_element_parts.txt', 'sha256'], '0' * 64),
                            (['version'], '9'), (['license'], 'invented')):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as temporary:
                lock = read_json(ROOT / 'sources.lock.json')
                row = lock['sources']['bodyparts3d_4']
                for key in path[:-1]:
                    row = row[key]
                row[path[-1]] = value
                target = Path(temporary) / 'lock.json'
                target.write_bytes(canonical(lock))
                with self.assertRaises(HumanImportError):
                    extract_cavity_surfaces(source_lock=target)

    def test_changed_anatomy_source_bytes_fail_before_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sources = Path(temporary)
            for name in ('partof_element_parts.txt', 'isa_element_parts.txt'):
                (sources / name).write_bytes((ROOT / 'Sources' / name).read_bytes())
            with (sources / 'isa_element_parts.txt').open('ab') as stream:
                stream.write(b'\n')
            with self.assertRaisesRegex(HumanImportError, 'hash mismatch'):
                extract_cavity_surfaces(sources=sources)

    def test_triangular_parser_rejects_malformed_or_unrepresented_geometry(self) -> None:
        malformed = [OBJ.replace(b'v 1 0 0', b'v nan 0 0'),
                     OBJ.replace(b'v 1 0 0', b'v 1e999 0 0'),
                     OBJ.replace(b'v 1 0 0', b'v 1e-999999999 0 0'),
                     OBJ.replace(b'v 1 0 0', b'v 1e-322 0 0'),
                     OBJ.replace(b'f 1 3 2', b'f 0 3 2'),
                     OBJ.replace(b'f 1 3 2', b'f 8 3 2'),
                     OBJ.replace(b'f 1 3 2', b'f 1 3 3'),
                     OBJ.replace(b'f 1 3 2', b'f 1 3 2 4'),
                     OBJ.replace(b'f 1 3 2', b'f 1/2 3/2 2/2'),
                     OBJ.replace(b'f 1 3 2', b'f 1//1 3//1 2//1'),
                     OBJ + b'f 1 3 2\n', OBJ + b'f 2 3 1\n',
                     OBJ + b'v 9 9 9\n', OBJ + b'curv 0 1 1 2\n', b'\xff']
        for data in malformed:
            with self.subTest(data=data[:80]), self.assertRaises(HumanImportError):
                parse_obj(data)

    def test_face_indices_and_exact_decimal_quotient(self) -> None:
        negative = OBJ.replace(b'f 1 3 2', b'f -4 -2 -3')
        self.assertEqual(parse_obj(negative)['triangles'], FACES)
        seam = OBJ.replace(b'f 1 3 2', b'v 1.000 0.0 0.000\nf 1 3 5')
        quotient = exact_coordinate_quotient(parse_obj(seam))
        self.assertEqual(quotient['identified_vertex_count'], 1)
        self.assertEqual(quotient['triangles'], FACES)
        self.assertTrue(quotient['topology']['closed_oriented_manifold_candidate'])
        distinct = exact_coordinate_quotient(parse_obj(seam.replace(b'1.000 0.0 0.000', b'1.00001 0.0 0.000')))
        self.assertEqual(distinct['identified_vertex_count'], 0)
        self.assertFalse(distinct['topology']['closed_oriented_manifold_candidate'])
        close = seam.replace(b'1.000 0.0 0.000', b'1.00000000000000000000001 0.0 0.000')
        with self.assertRaisesRegex(HumanImportError, 'collapse'):
            parse_obj(close)

    def test_topology_distinguishes_open_oriented_sphere_and_winding_defect(self) -> None:
        result = analyze_topology(VERTICES, FACES)
        self.assertTrue(result['closed_oriented_manifold_candidate'])
        self.assertEqual(result['euler_characteristic'], 2)
        opened = analyze_topology(VERTICES, FACES[:-1])
        self.assertEqual(opened['boundary_edge_count'], 3)
        self.assertEqual(opened['boundary_loop_count'], 1)
        self.assertFalse(opened['closed_oriented_manifold_candidate'])
        reversed_faces = copy.deepcopy(FACES)
        reversed_faces[0].reverse()
        defect = analyze_topology(VERTICES, reversed_faces)
        self.assertEqual(len(defect['orientation_defect_edges']), 3)
        self.assertFalse(defect['closed_oriented_manifold_candidate'])

    def test_vertex_link_check_rejects_two_closed_shells_pinched_at_one_vertex(self) -> None:
        vertices = VERTICES + [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]
        mapping = [0, 4, 5, 6]
        faces = FACES + [[mapping[i] for i in face] for face in FACES]
        result = analyze_topology(vertices, faces)
        self.assertEqual(result['boundary_edge_count'], 0)
        self.assertEqual(result['nonmanifold_edges'], [])
        self.assertEqual(result['face_component_count'], 2)
        self.assertEqual(result['vertex_manifold_defect_ids'], [0])
        self.assertFalse(result['closed_oriented_manifold_candidate'])

    def test_duplicate_nonmanifold_and_degenerate_faces_are_retained_as_defects(self) -> None:
        result = analyze_topology(VERTICES, FACES + [FACES[0]])
        self.assertEqual(result['duplicate_face_ids'], [4])
        self.assertEqual(len(result['nonmanifold_edges']), 3)
        self.assertFalse(result['closed_oriented_manifold_candidate'])
        flattened = copy.deepcopy(VERTICES)
        flattened[3] = [0., 0., 0.]
        self.assertTrue(analyze_topology(flattened, FACES)['degenerate_face_ids'])
        with self.assertRaises(HumanImportError):
            analyze_topology(VERTICES, [[True, 1, 2]])
        unused = analyze_topology(VERTICES + [[2., 2., 2.]], FACES)
        self.assertEqual(unused['unused_vertex_ids'], [4])
        self.assertFalse(unused['closed_oriented_manifold_candidate'])
        with self.assertRaisesRegex(HumanImportError, 'nonfinite triangle arithmetic'):
            analyze_topology([[0., 0., 0.], [1e308, 0., 0.], [0., 1e308, 0.]], [[0, 1, 2]])

    def test_cli_output_is_immutable_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'surface.json'
            self.assertEqual(main(['--output', str(output)]), 0)
            expected = output.read_bytes()
            self.assertEqual(main(['--output', str(output)]), 0)
            self.assertEqual(output.read_bytes(), expected)
            output.write_text('{}\n')
            self.assertEqual(main(['--output', str(output)]), 1)
            self.assertEqual(output.read_text(), '{}\n')
            link = Path(temporary) / 'redirected.json'
            link.symlink_to(output)
            self.assertEqual(main(['--output', str(link)]), 1)


if __name__ == '__main__':
    unittest.main()
