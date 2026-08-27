from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path
from subprocess import run

from numilab_human.model import (
    _BODYPARTS_MYOSIM_AXIAL_EXTENSIONS,
    _BODYPARTS_MYOSIM_BONE_ANCHORS,
    _BODYPARTS_MYOSIM_CRANIAL_EXTENSIONS,
    _BODYPARTS_MYOSIM_REMAINING_SOURCE_EXTENSIONS,
    _BODYPARTS_MYOSIM_THORACIC_FOOT_EXTENSIONS,
    _BODYPARTS_MYOSIM_TOE_EXTENSIONS,
    _BODYPARTS_MYOSIM_WRIST_HAND_EXTENSIONS,
    _bodyparts_secondary_attachment_weight_lock,
    _bodyparts_project_tendon_attachment_band,
    _bodyparts_drop_interior_tendon_cap_triangles,
    _bodyparts_stitch_tendon_enthesis_band,
    _bodyparts_source_mm_to_body_world,
    _bodyparts_world_to_body_stored_m,
    _bodyparts_skin_bbox_distance_squared,
    _bodyparts_skin_bbox_surface_distance_squared,
    _bodyparts_skin_nearest_surface_bindings,
    _bodyparts_largest_connected_surface_component,
    _bodyparts_skin_outer_surface_component,
    _bodyparts_skin_smooth_visual_normals,
    _bodyparts_skin_surface_index,
    _bodyparts_myosim_surface_specifications,
    _bodyparts_similarity_fit,
    ImportError as HumanImportError,
    bodyparts_foot_collider_preflight,
    bodyparts_foot_registration_receipt_template,
    validate_bodyparts_foot_registration_receipt,
    bodyparts_geometry_preflight,
    bodyparts_foot_registration_template,
    bodyparts_lower_body_attachment_worklist,
    bodyparts_right_calcaneal_tendon_continuity_preview,
    bodyparts_nerve_annotation,
    bodyparts_right_lower_leg_anatomy_preview,
    bodyparts_visual_preview,
    build_rajagopal_distal_pin_preview,
    evaluate_opensim_custom_joint,
    gate_report,
    parse_bodyparts3d,
    parse_opensim_archive,
    parse_opensim,
    rajagopal_custom_joint_gpu_artifacts,
    rajagopal_core_reference_artifact,
    read_json,
    rajagopal_custom_joint_ir,
    rajagopal_millard_muscle_ir,
    numi_human_tendon_endpoint_payload,
    rajagopal_lower_body_pilot,
    rajagopal_walking_contract,
    rajagopal_rigid_skeleton_ir,
    runtime_compatibility_report,
    runtime_checkout_gate,
)
from numilab_human.zanatomy import (
    _project_tendon_attachment_band,
    _stored_normal_from_world,
    _stored_source_from_world,
    _world_from_stored_source,
)


ROOT = Path(__file__).resolve().parents[1]


class ImporterTests(unittest.TestCase):
    def _minimal_myosim_tendon_artifact(self, directory: Path) -> Path:
        directory.mkdir(parents=True)
        source_sha = "11" * 32
        payload = b"".join([
            struct.pack(
                "<8s9I32s", b"NHMYO1\0\0", 1, 2, 1, 2, 0, 2, 1, 0, 0,
                bytes.fromhex(source_sha),
            ),
            struct.pack("<I3f", 0, 0.1, 0.2, 0.3),
            struct.pack("<I3f", 1, 0.4, 0.5, 0.6),
            struct.pack("<4I", 1, 0, 0xFFFFFFFF, 0),
            struct.pack("<4I", 1, 1, 0xFFFFFFFF, 0),
            struct.pack("<4I37f", 0, 0, 2, 0, *([0.0] * 37)),
        ])
        muscle_path = directory / "myosim-fullbody-muscle-reference.nhmyo"
        muscle_path.write_bytes(payload)
        manifest = {
            "schema": "numi.human.myosim-fullbody-reference.v1",
            "source": {"archive_sha256": source_sha},
            "payloads": {
                "rigid": {"file": "fixture.nhrigid", "sha256": "22" * 32,
                          "bytes": 0, "payload_abi": 1},
                "muscles": {"file": muscle_path.name,
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "bytes": len(payload), "payload_abi": 1},
                "support_contact": None,
            },
            "muscles": [{"source_actuator_index": 0, "name": "fixture_muscle"}],
        }
        (directory / "myosim-fullbody-reference.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return directory

    def test_numi_human_tendon_payload_covers_both_route_endpoints_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = self._minimal_myosim_tendon_artifact(directory / "artifact")
            manifest = numi_human_tendon_endpoint_payload(artifact, directory / "output")
            self.assertEqual(manifest["status"], "complete_route_endpoint_mechanical_coverage")
            self.assertEqual(manifest["coverage"]["muscle_count"], 1)
            self.assertEqual(manifest["coverage"]["mechanical_endpoint_count"], 2)
            self.assertEqual(manifest["coverage"]["source_site_point_count"], 2)
            self.assertEqual(manifest["coverage"]["registered_bone_triangle_count"], 0)
            payload = (directory / "output" / manifest["payload"]["file"]).read_bytes()
            self.assertEqual(len(payload), 104 + 2 * 64)
            first = struct.unpack_from("<8I8f", payload, 104)
            second = struct.unpack_from("<8I8f", payload, 168)
            self.assertEqual((first[1], first[3], first[4], first[5]), (0, 0, 0, 0))
            self.assertEqual((second[1], second[3], second[4], second[5]), (1, 1, 1, 0))

    def test_numi_human_tendon_surface_receipt_replaces_only_named_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = self._minimal_myosim_tendon_artifact(directory / "artifact")
            receipt = directory / "surface.json"
            receipt.write_text(json.dumps({
                "schema": "numi.human.tendon-surface-registration.v1",
                "admission": {"mechanical": True},
                "records": [{
                    "muscle": "fixture_muscle", "endpoint": "insertion", "body_index": 1,
                    "bone_member_id": "FJ3360", "bone_stable_id": 7,
                    "source_triangle_index": 12,
                    "triangle_local_m": [[0.3, 0.5, 0.6], [0.5, 0.5, 0.6], [0.4, 0.7, 0.6]],
                    "barycentric": [0.25, 0.5, 0.25],
                }],
            }), encoding="utf-8")
            manifest = numi_human_tendon_endpoint_payload(
                artifact, directory / "output", receipt,
            )
            self.assertEqual(manifest["coverage"]["source_site_point_count"], 1)
            self.assertEqual(manifest["coverage"]["registered_bone_triangle_count"], 1)
            payload = (directory / "output" / manifest["payload"]["file"]).read_bytes()
            insertion = struct.unpack_from("<8I8f", payload, 168)
            triangle = struct.unpack_from("<4I12f", payload, 232)
            self.assertEqual((insertion[4], insertion[5], insertion[6], insertion[7]), (1, 1, 0, 7))
            self.assertAlmostEqual(insertion[8], 0.425)
            self.assertAlmostEqual(insertion[9], 0.55)
            self.assertAlmostEqual(insertion[10], 0.6)
            self.assertEqual(triangle[:4], (1, 7, 12, 0))

    def test_numi_human_tendon_surface_candidate_fails_closed_and_stays_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = self._minimal_myosim_tendon_artifact(directory / "artifact")
            receipt = directory / "surface.json"
            receipt.write_text(json.dumps({
                "schema": "numi.human.tendon-surface-registration.v1",
                "admission": {"mechanical": False, "reason": "fixture candidate"},
                "records": [],
            }), encoding="utf-8")
            with self.assertRaisesRegex(HumanImportError, "not a mechanically admitted"):
                numi_human_tendon_endpoint_payload(artifact, directory / "rejected", receipt)
            manifest = numi_human_tendon_endpoint_payload(
                artifact, directory / "candidate", receipt,
                allow_unadmitted_surface=True,
            )
            self.assertEqual(
                manifest["status"],
                "candidate_route_endpoint_program_not_mechanically_admitted",
            )
            pack = read_json(directory / "candidate" / "numi-human-pack.manifest.json")
            self.assertEqual(pack["status"], manifest["status"])

    def test_skin_bone_envelope_distance_is_zero_inside_and_metric_outside(self) -> None:
        self.assertEqual(
            _bodyparts_skin_bbox_distance_squared(
                [0.0, 0.0, 0.0], [-1.0, -2.0, -3.0], [1.0, 2.0, 3.0],
            ),
            0.0,
        )
        self.assertEqual(
            _bodyparts_skin_bbox_distance_squared(
                [3.0, 5.0, 0.0], [-1.0, -2.0, -3.0], [1.0, 2.0, 3.0],
            ),
            13.0,
        )

    def test_skin_bone_envelope_boundary_distance_distinguishes_overlapping_bones(self) -> None:
        minimum, maximum = [-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]
        self.assertEqual(
            _bodyparts_skin_bbox_surface_distance_squared([0.0, 0.0, 0.0], minimum, maximum),
            1.0,
        )
        self.assertEqual(
            _bodyparts_skin_bbox_surface_distance_squared([1.0, 0.0, 0.0], minimum, maximum),
            0.0,
        )
        self.assertEqual(
            _bodyparts_skin_bbox_surface_distance_squared([3.0, 5.0, 0.0], minimum, maximum),
            13.0,
        )

    def test_skin_source_surface_index_returns_nearest_distinct_bodies(self) -> None:
        index = _bodyparts_skin_surface_index([
            ((0.0, 0.0, 0.0), 7), ((0.1, 0.0, 0.0), 7),
            ((0.2, 0.0, 0.0), 3), ((0.3, 0.0, 0.0), 5),
            ((0.4, 0.0, 0.0), 11), ((0.5, 0.0, 0.0), 13),
        ])
        candidates = _bodyparts_skin_nearest_surface_bindings(index, [0.09, 0.0, 0.0])
        self.assertEqual([binding for _, binding in candidates], [7, 3, 5, 11])

    def test_skin_visual_normal_smoothing_preserves_finite_unit_normals(self) -> None:
        normals = _bodyparts_skin_smooth_visual_normals(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [(0, 1, 2), (0, 2, 3)], iterations=2,
        )
        self.assertEqual(len(normals), 4)
        for normal in normals:
            self.assertAlmostEqual(sum(value * value for value in normal), 1.0)
        self.assertGreater(normals[0][2], 0.0)

    def test_skin_outer_surface_selection_retains_enclosing_source_sheet(self) -> None:
        outer_vertices = [
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0],
        ]
        inner_vertices = [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.0, 0.5, -0.5], [0.0, 0.0, 0.5],
        ]
        tetrahedra = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]
        vertices, triangles, evidence = _bodyparts_skin_outer_surface_component(
            outer_vertices + inner_vertices,
            tetrahedra + [tuple(index + 4 for index in triangle) for triangle in tetrahedra],
            "test-compound-skin",
        )
        self.assertEqual(vertices, outer_vertices)
        self.assertEqual(triangles, tetrahedra)
        self.assertEqual(evidence["source_vertex_count"], 8)
        self.assertEqual(evidence["retained_vertex_count"], 4)

    def test_tendon_component_selection_removes_disconnected_source_sliver(self) -> None:
        main_vertices = [
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0],
        ]
        sliver_vertices = [[5.0, 0.0, 0.0], [5.1, 0.0, 0.0]]
        tetrahedra = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]
        vertices, triangles, evidence = _bodyparts_largest_connected_surface_component(
            main_vertices + sliver_vertices, tetrahedra + [(0, 4, 5)], "test-compound-tendon",
        )
        self.assertEqual(vertices, main_vertices)
        self.assertEqual(triangles, tetrahedra)
        self.assertEqual(evidence["retained_triangle_count"], 4)
        self.assertEqual(evidence["discarded_component_count"], 1)

    def test_tendon_attachment_weight_lock_holds_secondary_bone_insertion(self) -> None:
        weights, evidence = _bodyparts_secondary_attachment_weight_lock(
            [[0.0, 0.0, 0.0], [0.008, 0.0, 0.0], [0.025, 0.0, 0.0]],
            [0.8, 0.8, 0.8],
            [[0.0, 0.0, 0.0]],
        )
        self.assertEqual(weights[0], 0.0)
        self.assertGreater(weights[1], 0.0)
        self.assertLess(weights[1], 0.8)
        self.assertEqual(weights[2], 0.8)
        self.assertEqual(evidence["locked_vertex_count"], 1)
        self.assertEqual(evidence["feathered_vertex_count"], 1)
        self.assertEqual(evidence["nearest_vertex_distance_m"], 0.0)

    def test_tendon_attachment_weight_lock_uses_bone_surface_not_only_vertices(self) -> None:
        weights, evidence = _bodyparts_secondary_attachment_weight_lock(
            [[0.01, 0.01, 0.0]], [0.8],
            [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]],
            [(0, 1, 2)],
        )
        self.assertEqual(weights, [0.0])
        self.assertEqual(evidence["method"],
                         "exact-source-triangle proximity to named secondary BodyParts3D bone mesh")
        self.assertEqual(evidence["nearest_vertex_distance_m"], 0.0)

    def test_zanatomy_tendon_attachment_projection_targets_named_calcaneus_surface(self) -> None:
        bone = [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]]
        projected, evidence = _project_tendon_attachment_band(
            [[0.01, 0.01, 0.002], [0.01, 0.01, 0.002]], [0.0, 0.5], bone, [(0, 1, 2)],
        )
        self.assertAlmostEqual(projected[0][0], 0.01)
        self.assertAlmostEqual(projected[0][1], 0.01)
        self.assertAlmostEqual(projected[0][2], -0.0015)
        self.assertAlmostEqual(projected[1][2], 0.00025)
        self.assertEqual(evidence["fully_locked_vertex_count"], 1)
        self.assertEqual(evidence["feathered_vertex_count"], 1)
        self.assertEqual(evidence["visual_enthesis_inset_m"], 0.0015)
        self.assertIn("interior enthesis inset", evidence["method"])

    def test_bodyparts_tendon_attachment_projection_preserves_source_topology_and_targets_bone(self) -> None:
        bone = [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]]
        source = [[0.01, 0.01, 0.002], [0.01, 0.01, 0.002], [0.03, 0.01, 0.02]]
        projected, evidence = _bodyparts_project_tendon_attachment_band(
            source, [0.0, 0.5, 1.0], bone, [(0, 1, 2)],
        )
        self.assertAlmostEqual(projected[0][0], 0.01)
        self.assertAlmostEqual(projected[0][1], 0.01)
        self.assertAlmostEqual(projected[0][2], -0.005)
        self.assertAlmostEqual(projected[1][2], -0.0015)
        self.assertEqual(projected[2], source[2])
        self.assertEqual(evidence["projected_vertex_count"], 2)
        self.assertEqual(evidence["fully_locked_vertex_count"], 1)
        self.assertEqual(evidence["feathered_vertex_count"], 1)
        self.assertEqual(evidence["visual_enthesis_inset_m"], 0.005)
        self.assertIn("interior enthesis inset", evidence["method"])

    def test_tendon_interior_cap_trim_preserves_attachment_transition_faces(self) -> None:
        triangles, evidence = _bodyparts_drop_interior_tendon_cap_triangles(
            [(0, 1, 2), (1, 2, 3), (0, 3, 4)], [0.0, 0.0, 0.0, 0.5, 1.0], "test-tendon",
        )
        self.assertEqual(triangles, [(1, 2, 3), (0, 3, 4)])
        self.assertEqual(evidence["dropped_fully_locked_interior_cap_triangle_count"], 1)

    def test_tendon_enthesis_stitch_closes_only_the_trimmed_distal_cap_boundary(self) -> None:
        vertices, triangles, attenuation, evidence = _bodyparts_stitch_tendon_enthesis_band(
            [[0.0, 0.0, 0.002], [0.02, 0.0, 0.002], [0.01, 0.02, 0.012]],
            [(0, 1, 2)], [0.0, 0.0, 1.0],
            [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]], [(0, 1, 2)], "test-tendon",
        )
        self.assertEqual(len(vertices), 5)
        self.assertEqual(len(triangles), 3)
        self.assertEqual(attenuation, [0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(evidence["source_boundary_edge_count"], 1)
        self.assertEqual(evidence["generated_triangle_count"], 2)
        self.assertAlmostEqual(vertices[3][2], 0.00035)
        self.assertAlmostEqual(vertices[4][2], 0.00035)

    def test_bodyparts_anchor_binding_round_trip_preserves_projected_tendon_coordinates(self) -> None:
        source_mm = [[100.0, -25.0, 60.0]]
        body_position = [0.5, -0.3, 0.2]
        identity = [0.0, 0.0, 0.0, 1.0]
        local_translation = [0.1, 0.2, -0.1]
        world = _bodyparts_source_mm_to_body_world(
            source_mm, body_position, identity, local_translation, identity, 1.0,
        )
        restored = _bodyparts_world_to_body_stored_m(
            world, body_position, identity, local_translation, identity, 1.0, "test",
        )
        for actual, expected in zip(restored[0], [0.1, -0.025, 0.06], strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_zanatomy_source_transform_round_trip_preserves_rotated_vectors(self) -> None:
        rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        translation = [0.4, -0.3, 0.2]
        stored = [0.12, -0.08, 0.31]
        world = _world_from_stored_source(stored, translation, rotation, 1.7)
        reconstructed = _stored_source_from_world(world, translation, rotation, 1.7)
        for actual, expected in zip(reconstructed, stored, strict=True):
            self.assertAlmostEqual(actual, expected)
        normal = _stored_normal_from_world([0.0, 1.0, 0.0], rotation)
        self.assertEqual(normal, [1.0, 0.0, 0.0])

    def test_zanatomy_supplement_scope_is_four_named_surfaces_and_matching_calcaneus(self) -> None:
        configuration = read_json(ROOT / "config/zanatomy-calf-visual-supplement.v1.json")
        self.assertEqual(configuration["source"]["license"], "CC-BY-SA-4.0")
        surfaces = [entry for entry in configuration["objects"] if entry["layer"] != "bone"]
        self.assertEqual([entry["id"] for entry in surfaces], [
            "lateral_gastrocnemius", "medial_gastrocnemius", "soleus", "calcaneal_tendon",
        ])
        self.assertEqual([entry["base_stable_id"] for entry in surfaces], [1, 3, 5, 7])
        self.assertEqual(surfaces[-1]["body_indices"], [131, 136, 138])
        overlay = next(entry for entry in configuration["objects"] if entry["layer"] == "bone")
        self.assertEqual(overlay["id"], "calcaneus_overlay")
        self.assertEqual(overlay["body_index"], 138)
        self.assertEqual(overlay["base_stable_id"], 7)

    def test_fullbody_surface_map_is_mirrored_and_explicit(self) -> None:
        surfaces = _bodyparts_myosim_surface_specifications()
        self.assertEqual(len(surfaces), 150)
        self.assertEqual(len({surface["member_id"] for surface in surfaces}), 150)
        self.assertEqual(sum(surface.get("layer", "muscle") == "muscle" for surface in surfaces), 148)
        self.assertEqual(sum(surface.get("layer") == "tendon" for surface in surfaces), 2)
        left_gastrocnemius = next(surface for surface in surfaces if surface["member_id"] == "FJ1394M")
        self.assertEqual(left_gastrocnemius["source_name"], "lateral head of left gastrocnemius")
        self.assertEqual(left_gastrocnemius["myosim_muscles"], ["gaslat_l"])

    def test_fullbody_surface_map_pins_native_right_upper_limb_inspection_ids(self) -> None:
        surfaces = _bodyparts_myosim_surface_specifications()
        selected = {
            stable_id: surfaces[stable_id - 1]
            for stable_id in range(69, 110, 2)
        }
        self.assertEqual(
            [surface["source_name"] for surface in selected.values()],
            [
                "abdominal part of right pectoralis major",
                "clavicular part of right pectoralis major",
                "sternocostal part of right pectoralis major",
                "acromial part of right deltoid",
                "clavicular part of right deltoid",
                "spinal part of right deltoid",
                "right supraspinatus",
                "right infraspinatus muscle",
                "right subscapularis",
                "right teres major",
                "right teres minor",
                "right coracobrachialis",
                "long head of right triceps brachii",
                "lateral head of right triceps brachii",
                "medial head of right triceps brachii",
                "right anconeus",
                "right supinator",
                "long head of right biceps brachii",
                "short head of right biceps brachii",
                "right brachialis",
                "right brachioradialis",
            ],
        )
        self.assertTrue(all(surface.get("layer", "muscle") == "muscle" for surface in selected.values()))

    def test_visual_skeleton_extension_preserves_the_validated_fit_set(self) -> None:
        fit_anchors = [
            anchor for anchor in _BODYPARTS_MYOSIM_BONE_ANCHORS
            if anchor.get("registration_anchor", True)
        ]
        extension = [
            anchor for anchor in _BODYPARTS_MYOSIM_BONE_ANCHORS
            if not anchor.get("registration_anchor", True)
        ]
        self.assertEqual(len(fit_anchors), 17)
        self.assertEqual(len(_BODYPARTS_MYOSIM_CRANIAL_EXTENSIONS), 8)
        self.assertEqual(len(_BODYPARTS_MYOSIM_THORACIC_FOOT_EXTENSIONS), 34)
        self.assertEqual(len(_BODYPARTS_MYOSIM_REMAINING_SOURCE_EXTENSIONS), 4)
        self.assertEqual(len(_BODYPARTS_MYOSIM_WRIST_HAND_EXTENSIONS), 52)
        self.assertEqual(len(_BODYPARTS_MYOSIM_TOE_EXTENSIONS), 38)
        self.assertEqual(len(_BODYPARTS_MYOSIM_AXIAL_EXTENSIONS), 22)
        self.assertEqual(len(_BODYPARTS_MYOSIM_BONE_ANCHORS), 184)
        baseline_extension = {
                ("right hip bone", "pelvis"), ("left hip bone", "pelvis"),
                ("right fibula", "tibia_r"), ("left fibula", "tibia_l"),
                ("right talus", "talus_r"), ("left talus", "talus_l"),
                ("right patella", "patella_r"), ("left patella", "patella_l"),
                ("body of sternum", "torso"),
        }
        self.assertTrue(baseline_extension.issubset(
            {(anchor["bodyparts_name"], anchor["myosim_body"]) for anchor in extension}
        ))
        self.assertEqual(
            len({anchor["member_id"] for anchor in _BODYPARTS_MYOSIM_BONE_ANCHORS}),
            len(_BODYPARTS_MYOSIM_BONE_ANCHORS),
        )

    def test_bodyparts_similarity_fit_keeps_proper_axes_and_positive_scale(self) -> None:
        source = [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0], [0.0, 0.0, 3.0],
        ]
        # target = 2 * (source-z, source-x, source-y) + translation.
        target = [
            [4.0 + 2.0 * point[2], -1.0 + 2.0 * point[0], 7.0 + 2.0 * point[1]]
            for point in source
        ]
        fit = _bodyparts_similarity_fit(source, target)
        self.assertEqual(fit["axis_permutation"], [2, 0, 1])
        self.assertEqual(fit["axis_signs"], [1, 1, 1])
        self.assertAlmostEqual(fit["scale_after_mm_to_m"], 2.0)
        self.assertAlmostEqual(fit["rms_residual_m"], 0.0)
        self.assertTrue(all(abs(value) < 1.0e-12 for value in fit["residuals_m"]))

    def test_foot_collider_preflight_emits_only_source_local_enclosure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sources = Path(temporary) / "Sources"
            sources.mkdir()
            archive = sources / "isa_BP3D_4.0_obj_99.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "isa_BP3D_4.0_obj_99/FJ1.obj",
                    "v -2 -1 3\nv 4 -1 3\nv -2 5 3\nf 1 2 3\n",
                )
            anatomy = {
                "source_id": "fixture-bodyparts", "version": "4.0",
                "archives": [{"hierarchy": "is_a", "file": archive.name, "sha256": "a" * 64}],
                "components": [
                    {"concept_id": "FMA1", "name": "right calcaneus", "anatomy_class": "bone", "hierarchy": "is_a",
                     "element_meshes": [{"element_id": "FJ1", "mesh_present": True}]},
                ],
            }
            model = {
                "source_id": "fixture-rajagopal", "source_file": "fixture.osim", "source_sha256": "b" * 64,
                "bodies": [{"id": body_id} for body_id in ("calcn_r", "toes_r", "calcn_l", "toes_l")],
            }
            result = bodyparts_foot_collider_preflight(sources, anatomy, model)
            mesh = result["per_foot"][0]["source_meshes"][0]
            self.assertEqual(result["status"], "source_local_proxy_candidates_not_admitted")
            self.assertEqual(mesh["geometry"]["bounds_mm"], {"minimum": [-2.0, -1.0, 3.0], "maximum": [4.0, 5.0, 3.0]})
            self.assertEqual(mesh["source_local_proxy_candidate"]["center_mm"], [1.0, 2.0, 3.0])
            self.assertEqual(mesh["source_local_proxy_candidate"]["half_extents_mm"], [3.0, 3.0, 0.0])
            self.assertNotIn("transform", mesh["source_local_proxy_candidate"])

    def test_foot_registration_template_requires_exact_source_bodies_and_has_no_transform(self) -> None:
        anatomy = {
            "source_id": "fixture-bodyparts", "version": "4.0",
            "archives": [{"hierarchy": "is_a", "file": "fixture.zip", "sha256": "a" * 64}],
            "components": [
                {"concept_id": "FMA1", "name": "right calcaneus", "anatomy_class": "bone", "hierarchy": "is_a",
                 "element_meshes": [{"element_id": "FJ1", "mesh_present": True}]},
                {"concept_id": "FMA2", "name": "left toes", "anatomy_class": "bone", "hierarchy": "is_a",
                 "element_meshes": [{"element_id": "FJ2", "mesh_present": True}]},
            ],
        }
        model = {
            "source_id": "fixture-rajagopal", "source_file": "fixture.osim", "source_sha256": "b" * 64,
            "bodies": [{"id": body_id} for body_id in ("calcn_r", "toes_r", "calcn_l", "toes_l")],
        }
        result = bodyparts_foot_registration_template(anatomy, model)
        self.assertEqual(result["walking_contact_bodies"], ["calcn_r", "toes_r", "calcn_l", "toes_l"])
        self.assertEqual(result["registrations"][0]["bodyparts_candidate_count"], 1)
        self.assertEqual(result["registrations"][1]["bodyparts_candidate_count"], 0)
        self.assertEqual(result["registrations"][3]["bodyparts_candidate_count"], 1)
        self.assertEqual(result["registrations"][0]["registration"]["status"], "requires_explicit_reviewed_transform")
        self.assertNotIn("transform", result["registrations"][0]["registration"])
        with self.assertRaises(HumanImportError):
            bodyparts_foot_registration_template(anatomy, {"bodies": []})

    def test_foot_registration_receipt_template_is_pinned_but_not_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sources = Path(temporary) / "Sources"
            sources.mkdir()
            archive = sources / "isa_BP3D_4.0_obj_99.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "isa_BP3D_4.0_obj_99/FJ1.obj",
                    "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                )
            anatomy = {
                "source_id": "fixture-bodyparts", "version": "4.0",
                "archives": [{"hierarchy": "is_a", "file": archive.name, "sha256": "a" * 64}],
                "components": [{
                    "concept_id": "FMA1", "name": "right calcaneus",
                    "anatomy_class": "bone", "hierarchy": "is_a",
                    "element_meshes": [{"element_id": "FJ1", "mesh_present": True}],
                }],
            }
            model = {
                "source_id": "fixture-rajagopal", "source_file": "fixture.osim",
                "source_sha256": "b" * 64,
                "bodies": [{"id": body_id} for body_id in (
                    "calcn_r", "toes_r", "calcn_l", "toes_l"
                )],
            }
            receipt = bodyparts_foot_registration_receipt_template(sources, anatomy, model)
        self.assertEqual(
            receipt["status"], "not_a_registration_or_collider_manifest"
        )
        self.assertEqual(len(receipt["preflight_sha256"]), 64)
        right_calcaneus = receipt["receipts"][0]
        self.assertEqual(right_calcaneus["opensim_body"], "calcn_r")
        self.assertEqual(right_calcaneus["source_meshes"][0]["source"]["member_id"], "FJ1")
        self.assertEqual(
            right_calcaneus["reviewed_registration"]["multi_angle_visual_review"]
            ["minimum_distinct_views"], 3
        )
        self.assertNotIn(
            "transform", right_calcaneus["reviewed_registration"]
        )

    def test_foot_registration_receipt_validator_requires_complete_review_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sources = Path(temporary) / "Sources"
            sources.mkdir()
            archive = sources / "isa_BP3D_4.0_obj_99.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "isa_BP3D_4.0_obj_99/FJ1.obj",
                    "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                )
            anatomy = {
                "source_id": "fixture-bodyparts", "version": "4.0",
                "archives": [{"hierarchy": "is_a", "file": archive.name, "sha256": "a" * 64}],
                "components": [
                    {"concept_id": f"FMA{index}", "name": name, "anatomy_class": "bone", "hierarchy": "is_a",
                     "element_meshes": [{"element_id": "FJ1", "mesh_present": True}]}
                    for index, name in enumerate((
                        "right calcaneus", "right toes", "left calcaneus", "left toes",
                    ), 1)
                ],
            }
            model = {
                "source_id": "fixture-rajagopal", "source_file": "fixture.osim",
                "source_sha256": "b" * 64,
                "bodies": [{"id": body_id} for body_id in (
                    "calcn_r", "toes_r", "calcn_l", "toes_l"
                )],
            }
            receipt = bodyparts_foot_registration_receipt_template(sources, anatomy, model)
            with self.assertRaises(HumanImportError):
                validate_bodyparts_foot_registration_receipt(receipt, sources, anatomy, model)
            for index, entry in enumerate(receipt["receipts"]):
                entry["reviewed_registration"] = {
                    "axis_and_unit_conversion": {
                        "axis_permutation": [0, 1, 2], "axis_signs": [1, 1, 1],
                        "scale_m_per_source_unit": 0.001, "reviewer": "fixture reviewer",
                    },
                    "source_to_body_rest_transform": {
                        "matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                                   [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                    },
                    "multi_angle_visual_review": {
                        "reviewer": "fixture reviewer",
                        "views": [{
                            "id": view, "artifact_sha256": f"{index:064x}",
                            "maximum_landmark_residual_mm": 0.0,
                        } for view in ("front", "side", "rear")],
                    },
                }
                entry["reviewed_contact"] = {
                    "proxy_geometry": {"shape": "box", "body_frame": entry["opensim_body"], "parameters": {"half_extents_m": [0.1, 0.1, 0.1]}},
                    "collision_exclusions": [],
                    "calibration": {
                        "friction": 0.8, "normal_stiffness": 1000.0,
                        "normal_damping": 10.0, "restitution": 0.0,
                        "evidence_sha256": f"{index + 4:064x}", "reviewer": "fixture reviewer",
                    },
                }
            result = validate_bodyparts_foot_registration_receipt(receipt, sources, anatomy, model)
        self.assertEqual(
            result["status"], "structurally_complete_not_physics_or_walking_qualified"
        )
        self.assertEqual(result["reviewed_foot_bodies"], ["calcn_r", "toes_r", "calcn_l", "toes_l"])

    def test_lower_body_attachment_worklist_never_promotes_name_matches_to_bindings(self) -> None:
        anatomy = {"source_id": "fixture", "version": "4.0", "archives": {}, "components": [
            {"concept_id": "FMA1", "name": "right calcaneus", "anatomy_class": "bone",
             "element_meshes": [{"element_id": "FJ1", "mesh_present": True}]},
            {"concept_id": "FMA2", "name": "left toe muscle", "anatomy_class": "muscle_surface",
             "element_meshes": [{"element_id": "FJ2", "mesh_present": True}]},
        ]}
        result = bodyparts_lower_body_attachment_worklist(anatomy, {"bodies": []})
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["candidates"][0]["status"], "candidate_requires_rest_frame_registration")
        self.assertEqual(result["foot_collider_work"]["status"], "blocked_by_registration_and_calibration")
    def test_walking_contract_requires_source_mobile_root_and_complete_muscles(self) -> None:
        model = {
            "source_id": "fixture", "source_file": "fixture.osim", "source_sha256": "0" * 64,
            "bodies": [{"id": "pelvis"}],
            "joints": [{"id": "ground_pelvis", "kind": "CustomJoint", "parent_frame": "/ground",
                "child_frame": "/bodyset/pelvis", "coordinates": [
                    {"id": name, "default_value": 0.0, "range": [-1.0, 1.0], "clamped": "true", "locked": "false"}
                    for name in ("pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz")],
                "frames": [], "motion_axes": [{"id": f"axis{i}", "coordinates": name, "axis": axis,
                    "function_kind": "LinearFunction", "function_parameters": {"coefficients": [1.0, 0.0]}}
                    for i, (name, axis) in enumerate(zip(("pelvis_tilt", "pelvis_list", "pelvis_rotation", "pelvis_tx", "pelvis_ty", "pelvis_tz"),
                        ([0,0,1], [1,0,0], [0,1,0], [1,0,0], [0,1,0], [0,0,1])), 1)]}],
            "muscles": [], "wrap_objects": [], "model_id": "fixture",
        }
        # The compiler deliberately rejects incomplete muscle admission rather than inventing actions.
        with self.assertRaises(HumanImportError):
            rajagopal_walking_contract(model)

    def test_lower_body_pilot_uses_four_temporary_foot_pads(self) -> None:
        model = parse_opensim(
            ROOT / "Sources/RajagopalLaiUhlrich2023.osim",
            "rajagopal_lai_uhlrich_2023",
        )
        result = rajagopal_lower_body_pilot(model)
        self.assertEqual(result["contact"]["mode"], "temporary_engineering_pads")
        self.assertEqual(result["contact"]["full_dimensions_m"], [0.06, 0.03, 0.06])
        self.assertEqual(
            [pad["body"] for pad in result["contact"]["pads"]],
            ["calcn_r", "toes_r", "calcn_l", "toes_l"],
        )
        self.assertEqual(result["policy"]["action_count"], 80)
    def test_bodyparts_visual_preview_preserves_one_source_member(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "Sources"
            source.mkdir()
            archive = source / "isa_BP3D_4.0_obj_99.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(
                    "isa_BP3D_4.0_obj_99/FJ2810.obj",
                    "v 0 0 0\nv 1000 0 0\nv 0 1000 0\nf 1 2 3\n",
                )
            result = bodyparts_visual_preview(source, Path(temporary) / "Preview")
            preview = Path(temporary) / "Preview"
            self.assertEqual(result["geometry"]["vertex_count"], 3)
            self.assertEqual(result["geometry"]["triangle_count"], 1)
            self.assertEqual(result["geometry"]["maximum_mm"], [1000.0, 1000.0, 0.0])
            self.assertTrue((preview / "FJ2810-source-static.glb").is_file())
            self.assertEqual((preview / "FJ2810-source-static.glb").read_bytes()[:4], b"glTF")

    def test_right_lower_leg_anatomy_preview_preserves_source_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "Sources"
            source.mkdir()
            archive = source / "isa_BP3D_4.0_obj_99.zip"
            members = (
                "FJ3365", "FJ3381", "FJ3387", "FJ3366", "FJ3385", "FJ3360",
                "FJ1394", "FJ1397", "FJ1437", "FJ1439", "FJ1405",
            )
            with zipfile.ZipFile(archive, "w") as bundle:
                for index, member in enumerate(members):
                    bundle.writestr(
                        f"isa_BP3D_4.0_obj_99/{member}.obj",
                        f"v {index} 0 0\nv {index + 1} 0 0\nv {index} 1 0\nf 1 2 3\n",
                    )
            preview = Path(temporary) / "Preview"
            result = bodyparts_right_lower_leg_anatomy_preview(source, preview)
            self.assertEqual(result["geometry"]["surface_count"], len(members))
            self.assertEqual(result["geometry"]["vertex_count"], 3 * len(members))
            self.assertEqual(result["geometry"]["triangle_count"], len(members))
            self.assertEqual(
                {surface["layer"] for surface in result["surfaces"]},
                {"bone", "muscle", "tendon"},
            )
            glb = preview / "bodyparts3d-right-lower-leg-anatomy-source-static.glb"
            self.assertEqual(glb.read_bytes()[:4], b"glTF")

    def test_calcaneal_tendon_preview_preserves_authored_normals_and_continuity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "Sources"
            source.mkdir()
            archive = source / "isa_BP3D_4.0_obj_99.zip"
            members = (
                "FJ3387", "FJ3366", "FJ3385", "FJ3360",
                "FJ1394", "FJ1397", "FJ1437", "FJ1405",
            )
            with zipfile.ZipFile(archive, "w") as bundle:
                for member in members:
                    bundle.writestr(
                        f"isa_BP3D_4.0_obj_99/{member}.obj",
                        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                        "vn 0 0 1\nvn 0 0 1\nvn 0 0 1\n"
                        "f 1//1 2//2 3//3\n",
                    )
            preview = Path(temporary) / "Preview"
            result = bodyparts_right_calcaneal_tendon_continuity_preview(source, preview)
            self.assertEqual(result["geometry"]["surface_count"], len(members))
            self.assertEqual(result["geometry"]["authored_normal_surface_count"], len(members))
            self.assertEqual(result["geometry"]["generated_normal_surface_count"], 0)
            self.assertTrue(all(
                relationship["distance_mm"] == 0.0
                for relationship in result["source_mesh_proximity"]
            ))
            glb = preview / "bodyparts3d-right-calcaneal-tendon-continuity-source-static.glb"
            self.assertEqual(glb.read_bytes()[:4], b"glTF")

    def test_numi_workspace_command_describes_itself(self) -> None:
        command = ROOT / ".numi/commands/human"
        result = run([command, "--numi-describe"], capture_output=True, text=True, check=True)
        self.assertEqual(
            result.stdout,
            "Build NumiLab Human source artifacts and run native full-body muscle and articulated-visual references.\n",
        )

    def test_numi_workspace_native_visual_command_rejects_missing_paths_before_python(self) -> None:
        command = ROOT / ".numi/commands/human"
        result = run(
            [command, "myosim-native-visuals"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "usage: numi human myosim-native-visuals <artifact-directory> <output-directory>\n",
        )

    def test_numi_workspace_muscle_bone_visual_command_rejects_missing_paths_before_python(self) -> None:
        command = ROOT / ".numi/commands/human"
        result = run(
            [command, "myosim-native-muscle-bone-visuals"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "usage: numi human myosim-native-muscle-bone-visuals <artifact-directory> "
            "<bodyparts3d-myosim-major-bones.nhbones> <output-directory> "
            "[--muscle-step-seconds <1e-6..1e-3>] "
            "[--dimension <512..2048; multiple-of-64>]\n",
        )

    def test_numi_workspace_fullbody_soft_tissue_visual_command_rejects_missing_paths_before_python(self) -> None:
        command = ROOT / ".numi/commands/human"
        result = run(
            [command, "myosim-native-fullbody-soft-tissue-visuals"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "usage: numi human myosim-native-fullbody-soft-tissue-visuals "
            "<artifact-directory> <bodyparts3d-myosim-major-bones.nhbones> "
            "<bodyparts3d-myosim-soft-tissue.nhtissue> <output-directory> "
            "[--dimension <512..2048; multiple-of-64>]\n",
        )

    def test_numi_workspace_torso_anatomy_visual_command_rejects_missing_paths_before_python(self) -> None:
        command = ROOT / ".numi/commands/human"
        result = run(
            [command, "myosim-native-torso-anatomy-visuals"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "usage: numi human myosim-native-torso-anatomy-visuals "
            "<artifact-directory> <bodyparts3d-myosim-major-bones.nhbones> "
            "<bodyparts3d-myosim-torso-anatomy.nhanatomy> <output-directory> "
            "[--dimension <512..2048; multiple-of-64>]\n",
        )

    def test_torso_anatomy_map_is_unique_and_source_backed(self) -> None:
        mapping = read_json(ROOT / "config/bodyparts3d-myosim-torso-anatomy-map.v1.json")
        self.assertEqual(mapping["schema"], "numi.human.bodyparts3d-myosim-torso-anatomy-map.v1")
        entries = mapping["entries"]
        self.assertEqual(len(entries), 12)
        self.assertEqual({entry["layer"] for entry in entries}, {"organ", "vessel", "nerve"})
        self.assertEqual(len({entry["member_id"] for entry in entries}), len(entries))
        source_relations = {
            tuple(line.split("\t"))
            for line in (ROOT / "Sources/partof_element_parts.txt").read_text(encoding="utf-8").splitlines()
        }
        for entry in entries:
            self.assertIn(
                (entry["concept_id"], entry["source_name"], entry["member_id"]),
                source_relations,
            )

    def test_numi_workspace_supported_muscle_surface_command_rejects_missing_paths_before_python(self) -> None:
        command = ROOT / ".numi/commands/human"
        result = run(
            [command, "myosim-native-supported-muscle-soft-tissue-visuals"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "usage: numi human myosim-native-supported-muscle-soft-tissue-visuals "
            "<artifact-directory> <bodyparts3d-myosim-major-bones.nhbones> "
            "<bodyparts3d-myosim-soft-tissue.nhtissue> <output-directory> "
            "<focus-body-index> [--muscle-step-seconds <1e-6..1e-3>] "
            "[--muscle-step-count <1..64>] "
            "[--muscle-activation <0..1>] "
            "[--dimension <512..2048; multiple-of-64>]\n",
        )

    def test_numi_workspace_posterior_tendon_inspection_rejects_missing_paths_before_python(self) -> None:
        command = ROOT / ".numi/commands/human"
        result = run(
            [command, "myosim-native-posterior-tendon-inspection"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "usage: numi human myosim-native-posterior-tendon-inspection "
            "<artifact-directory> <bodyparts3d-myosim-major-bones.nhbones> "
            "<bodyparts3d-myosim-soft-tissue.nhtissue> <output-directory> "
            "[--dimension <512..2048; multiple-of-64>]\n",
        )

    def test_numi_workspace_upper_limb_inspection_rejects_missing_paths_before_python(self) -> None:
        command = ROOT / ".numi/commands/human"
        result = run(
            [command, "myosim-native-upper-limb-inspection"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "usage: numi human myosim-native-upper-limb-inspection "
            "<artifact-directory> <bodyparts3d-myosim-major-bones.nhbones> "
            "<bodyparts3d-myosim-soft-tissue.nhtissue> <output-directory> "
            "[--muscle-step-seconds <1e-6..1e-3>] "
            "[--muscle-step-count <1..64>] "
            "[--muscle-activation <0..1>] "
            "[--dimension <512..2048; multiple-of-64>]\n",
        )

    def test_numi_workspace_right_upper_limb_flexion_drive_rejects_missing_paths_before_python(self) -> None:
        command = ROOT / ".numi/commands/human"
        result = run(
            [command, "myosim-native-right-upper-limb-flexion-drive"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(
            result.stderr,
            "usage: numi human myosim-native-right-upper-limb-flexion-drive "
            "<artifact-directory> <bodyparts3d-myosim-major-bones.nhbones> "
            "<bodyparts3d-myosim-soft-tissue.nhtissue> <output-directory> "
            "[--dimension <512..2048; multiple-of-64>]\n",
        )

    def test_opensim_parser_retains_mechanical_fields(self) -> None:
        source = """<?xml version=\"1.0\"?>
<OpenSimDocument Version=\"40000\"><Model name=\"fixture\">
  <gravity>0 -9.81 0</gravity>
  <BodySet><objects><Body name=\"pelvis\"><mass>11.2</mass><mass_center>0 0.1 0</mass_center><inertia_xx>1</inertia_xx><inertia_yy>2</inertia_yy><inertia_zz>3</inertia_zz></Body></objects></BodySet>
  <JointSet><objects><CustomJoint name=\"hip\"><socket_parent_frame>/ground</socket_parent_frame><socket_child_frame>/bodyset/pelvis</socket_child_frame><coordinates><Coordinate name=\"hip_flexion\"><default_value>0</default_value><range>-1 1</range><clamped>true</clamped><locked>false</locked></Coordinate></coordinates><frames><PhysicalOffsetFrame name=\"hip_center\"><socket_parent>/bodyset/pelvis</socket_parent><translation>0 0.1 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame></frames><SpatialTransform><TransformAxis name=\"rotation1\"><coordinates>hip_flexion</coordinates><axis>0 0 1</axis><function><LinearFunction><coefficients>1 0</coefficients></LinearFunction></function></TransformAxis></SpatialTransform></CustomJoint></objects></JointSet>
  <ForceSet><objects><Millard2012EquilibriumMuscle name=\"iliacus\"><max_isometric_force>1000</max_isometric_force><optimal_fiber_length>0.1</optimal_fiber_length><tendon_slack_length>0.2</tendon_slack_length><minimum_activation>0.01</minimum_activation><ActiveForceLengthCurve><min_norm_active_fiber_length>0.5</min_norm_active_fiber_length></ActiveForceLengthCurve><GeometryPath><PathPointSet><objects><PathPoint name=\"origin\"><socket_parent_frame>/bodyset/pelvis</socket_parent_frame><location>0 0 0</location></PathPoint></objects></PathPointSet><PathWrapSet><objects><PathWrap name=\"wrap\"><socket_wrap_object>/bodyset/pelvis/wrap_object</socket_wrap_object></PathWrap></objects></PathWrapSet></GeometryPath></Millard2012EquilibriumMuscle></objects></ForceSet>
</Model></OpenSimDocument>"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.osim"
            path.write_text(source, encoding="utf-8")
            result = parse_opensim(path, "fixture")
        self.assertEqual(result["model_id"], "fixture")
        self.assertEqual(result["bodies"][0]["mass_kg"], 11.2)
        self.assertEqual(
            result["bodies"][0]["inertia_kg_m2"],
            {"xx": 1.0, "yy": 2.0, "zz": 3.0, "xy": 0.0, "xz": 0.0, "yz": 0.0},
        )
        self.assertEqual(result["joints"][0]["coordinates"][0]["id"], "hip_flexion")
        self.assertEqual(result["muscles"][0]["parameters"]["tendon_slack_length"], 0.2)
        self.assertEqual(result["muscles"][0]["path_points"][0]["parent_frame"], "/bodyset/pelvis")
        self.assertEqual(result["joints"][0]["frames"][0]["id"], "hip_center")
        self.assertEqual(result["joints"][0]["motion_axes"][0]["coordinates"], "hip_flexion")
        self.assertEqual(result["joints"][0]["motion_axes"][0]["function_kind"], "LinearFunction")
        self.assertEqual(
            result["joints"][0]["motion_axes"][0]["function_parameters"]["coefficients"],
            [1.0, 0.0],
        )
        self.assertEqual(result["muscles"][0]["path_wraps"][0]["wrap_object"], "/bodyset/pelvis/wrap_object")
        self.assertEqual(result["muscles"][0]["parameters"]["minimum_activation"], 0.01)
        self.assertEqual(
            result["muscles"][0]["curves"]["ActiveForceLengthCurve"]["parameters"]
            ["min_norm_active_fiber_length"],
            0.5,
        )

    def test_opensim3_parser_retains_body_owned_joint_frames(self) -> None:
        source = """<?xml version=\"1.0\"?>
<OpenSimDocument Version=\"30000\"><Model name=\"legacy\">
  <gravity>0 -9.81 0</gravity>
  <BodySet><objects><Body name=\"cerv7\"><mass>0.5</mass><mass_center>0 0 0</mass_center><inertia>1 1 1 0 0 0</inertia><Joint><CustomJoint name=\"neck\"><parent_body>spine</parent_body><location_in_parent>0 0.1 0</location_in_parent><orientation_in_parent>0 0 0</orientation_in_parent><location>0 0 0</location><orientation>0 0 0</orientation><CoordinateSet><objects><Coordinate name=\"pitch\"><default_value>0</default_value><range>-1 1</range></Coordinate></objects></CoordinateSet></CustomJoint></Joint></Body></objects></BodySet>
  <ForceSet><objects></objects></ForceSet>
</Model></OpenSimDocument>"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.osim"
            path.write_text(source, encoding="utf-8")
            result = parse_opensim(path, "legacy")
        joint = result["joints"][0]
        self.assertEqual(result["opensim_document_version"], "30000")
        self.assertTrue(joint["legacy_opensim3"])
        self.assertEqual(joint["parent_frame"], "spine")
        self.assertEqual(joint["child_frame"], "cerv7")
        self.assertEqual(joint["frames"][0]["translation_m"], [0.0, 0.1, 0.0])
        self.assertEqual(joint["coordinates"][0]["id"], "pitch")

    def test_bodyparts_parser_preserves_two_hierarchies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "isa_parts_list_e.txt").write_text(
                "concept id\trepresentation id\ten\nFMA5018\tBP1\tbone organ\nFMA9611\tBP2\tfemur\n",
                encoding="utf-8",
            )
            (source / "partof_parts_list_e.txt").write_text(
                "concept id\trepresentation id\ten\nFMA7157\tBP3\tnervous system\nFMA5865\tBP4\tcranial nerve\n",
                encoding="utf-8",
            )
            (source / "isa_inclusion_relation_list.txt").write_text(
                "parent id\tparent name\tchild id\tchild name\nFMA5018\tbone organ\tFMA9611\tfemur\n",
                encoding="utf-8",
            )
            (source / "partof_inclusion_relation_list.txt").write_text(
                "parent id\tparent name\tchild id\tchild name\nFMA7157\tnervous system\tFMA5865\tcranial nerve\n",
                encoding="utf-8",
            )
            (source / "isa_element_parts.txt").write_text(
                "concept id\tname\telement file id\nFMA9611\tfemur\tFJ100\n",
                encoding="utf-8",
            )
            (source / "partof_element_parts.txt").write_text(
                "concept id\tname\telement file id\nFMA5865\tcranial nerve\tFJ200\n",
                encoding="utf-8",
            )
            for archive_name, members in {
                "isa_BP3D_4.0_obj_99.zip": ("FJ100.obj", "FJ101.obj"),
                "partof_BP3D_4.0_obj_99.zip": ("FJ200.obj",),
            }.items():
                with zipfile.ZipFile(source / archive_name, "w") as archive:
                    for member in members:
                        archive.writestr(
                            member,
                            "o fixture\n"
                            "v 0 0 0\n"
                            "v 1 0 0\n"
                            "v 0 1 0\n"
                            "v 0 0 1\n"
                            "f 1 3 2\n"
                            "f 1 2 4\n"
                            "f 2 3 4\n"
                            "f 3 1 4\n",
                        )
            result = parse_bodyparts3d(source, ROOT / "config/anatomy-classification.v1.json")
            geometry = bodyparts_geometry_preflight(source, result)
        lookup = {item["concept_id"]: item for item in result["components"]}
        self.assertEqual(len(result["hierarchy_edges"]), 2)
        self.assertEqual(lookup["FMA9611"]["anatomy_class"], "bone")
        self.assertEqual(lookup["FMA5865"]["anatomy_class"], "nerve_surface")
        self.assertTrue(lookup["FMA9611"]["mesh_present"])
        self.assertEqual(lookup["FMA9611"]["element_meshes"][0]["element_id"], "FJ100")
        self.assertTrue(lookup["FMA5865"]["element_meshes"][0]["mesh_present"])
        self.assertEqual(geometry["summary"]["mesh_count"], 3)
        self.assertEqual(geometry["summary"]["closed_2_manifold_candidates"], 3)
        self.assertEqual(geometry["summary"]["invalid_face_reference_count"], 0)
        self.assertEqual(geometry["archives"][0]["meshes"][0]["vertex_count"], 4)
        self.assertEqual(geometry["archives"][0]["meshes"][0]["face_count"], 4)

    def test_nerve_annotation_preserves_components_and_incident_hierarchy_edges(self) -> None:
        annotation = bodyparts_nerve_annotation(
            {
                "source_id": "bodyparts3d_4",
                "version": "4.0",
                "archives": [{"file": "isa.zip", "sha256": "a" * 64}],
                "components": [
                    {
                        "concept_id": "FMA7157",
                        "representation_id": "BP1",
                        "hierarchy": "is_a",
                        "anatomy_class": "nerve_surface",
                        "element_meshes": [{"element_id": "FJ1", "mesh_present": True}],
                    },
                    {
                        "concept_id": "FMA5018",
                        "representation_id": "BP2",
                        "hierarchy": "is_a",
                        "anatomy_class": "bone",
                        "element_meshes": [{"element_id": "FJ2", "mesh_present": True}],
                    },
                ],
                "hierarchy_edges": [
                    {
                        "hierarchy": "is_a",
                        "parent_id": "FMA7157",
                        "parent_name": "nervous system",
                        "child_id": "FMA5865",
                        "child_name": "cranial nerve",
                    },
                    {
                        "hierarchy": "is_a",
                        "parent_id": "FMA5018",
                        "parent_name": "anatomical structure",
                        "child_id": "FMA23881",
                        "child_name": "bone",
                    },
                ],
            }
        )
        self.assertEqual(annotation["component_count"], 1)
        self.assertEqual(annotation["hierarchy_edge_count"], 1)
        self.assertEqual(annotation["numi_role"], "anatomical_geometry_only")

    def test_gate_report_keeps_unavailable_sources_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            lock = {
                "sources": {
                    "bodyparts3d_4": {"files": {"body.zip": {"role": "geometry", "sha256": "abc"}}},
                    "rajagopal_lai_uhlrich_2023": {"sha256": "def"},
                    "mobl_arms_upper_extremity": {"release_file": "upper.zip", "license": "non-commercial"},
                }
            }
            report = gate_report(
                sources=source,
                upper_archive=None,
                source_lock=lock,
                runtime_contract=read_json(ROOT / "config/numi-runtime-contract.v1.json"),
            )
        self.assertEqual(report["source_artifacts"]["bodyparts3d_4"][0]["status"], "missing")
        self.assertEqual(report["source_artifacts"]["mobl_arms_upper_extremity"]["status"], "missing_authenticated_archive")
        self.assertEqual(report["gates"][0]["status"], "blocked")
        self.assertEqual(report["runtime_checkout"]["status"], "not_provided")

    def test_runtime_checkout_gate_rejects_a_missing_runtime_root(self) -> None:
        report = runtime_checkout_gate(
            Path("/nonexistent/numilab-human-test-runtime"),
            read_json(ROOT / "config/numi-runtime-contract.v1.json"),
        )
        self.assertEqual(report["status"], "missing")

    def test_gate_report_exposes_the_active_myosim_fullbody_route(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            archive = source / "myosim" / "myo.tar.gz"
            archive.parent.mkdir()
            archive.write_bytes(b"pinned-myosim-source")
            checkout = archive.parent / "checkout" / "myo_sim" / "build"
            checkout.mkdir(parents=True)
            (checkout / "compose.py").write_text("# fixture\n", encoding="utf-8")
            lock = {
                "sources": {
                    "bodyparts3d_4": {"files": {}},
                    "rajagopal_lai_uhlrich_2023": {"sha256": "def"},
                    "mobl_arms_upper_extremity": {
                        "release_file": "upper.zip", "license": "non-commercial",
                    },
                    "myosim_fullbody": {
                        "storage_dir": "myosim",
                        "archive_file": "myo.tar.gz",
                        "archive_sha256": hashlib.sha256(
                            b"pinned-myosim-source"
                        ).hexdigest(),
                        "expected_file": "myo_sim/build/compose.py",
                        "revision": "fixture-revision",
                        "license": "Apache-2.0",
                        "role": "active full-body mechanics",
                    },
                }
            }
            report = gate_report(
                sources=source,
                upper_archive=None,
                source_lock=lock,
                runtime_contract=read_json(ROOT / "config/numi-runtime-contract.v1.json"),
            )
        self.assertEqual(
            report["source_artifacts"]["myosim_fullbody"]["status"],
            "verified",
        )
        active_gate = next(
            gate for gate in report["gates"]
            if gate["id"] == "active_myosim_fullbody_mechanics"
        )
        self.assertEqual(
            active_gate["status"], "qualified_static_device_reference"
        )

    def test_gate_report_exposes_the_selected_free_human_foundation_stack(self) -> None:
        source_lock = read_json(ROOT / "sources.lock.json")
        public_upper = source_lock["sources"]["mobl_arms_ceinms_41_public_mirror"]
        report = gate_report(
            sources=ROOT / "Sources",
            upper_archive=None,
            upper_public_model=ROOT / "Sources" / public_upper["model_file"],
            source_lock=source_lock,
            runtime_contract=read_json(ROOT / "config/numi-runtime-contract.v1.json"),
        )
        free_foundation_gate = next(
            gate for gate in report["gates"]
            if gate["id"] == "free_human_foundation_source_stack"
        )
        original_bimanual_gate = next(
            gate for gate in report["gates"]
            if gate["id"] == "source_faithful_import"
        )
        self.assertEqual(
            free_foundation_gate["status"],
            "ready_for_source_import_unimanual_upper_variant",
        )
        self.assertEqual(original_bimanual_gate["status"], "blocked")

    def test_public_mobl_reports_its_real_runtime_lowering_boundaries(self) -> None:
        model = parse_opensim(
            ROOT / "Sources" / "MOBL_ARMS_41.osim",
            "mobl_arms_ceinms_41_public_mirror",
        )
        report = runtime_compatibility_report(
            model,
            read_json(ROOT / "config/numi-runtime-contract.v1.json"),
        )
        self.assertEqual(report["skeleton"]["status"], "blocked")
        self.assertEqual(report["skeleton"]["massless_source_bodies"], ["thorax"])
        self.assertEqual(
            report["source_model"]["wrap_object_kinds"],
            {
                "WrapCylinder": 49,
                "WrapEllipsoid": 23,
                "WrapSphere": 4,
                "WrapTorus": 11,
            },
        )
        self.assertEqual(report["muscle_tendon"]["status"], "blocked")
        self.assertEqual(
            report["muscle_tendon"]["unsupported_source_wrap_kinds"],
            {"WrapEllipsoid": 23, "WrapSphere": 4, "WrapTorus": 11},
        )

    def test_runtime_compatibility_preserves_unsupported_joints_and_muscle_evidence_gates(self) -> None:
        model = {
            "model_id": "fixture",
            "joints": [
                {"kind": "PinJoint", "motion_axes": []},
                {"kind": "CustomJoint", "motion_axes": [{"function_kind": "SimmSpline"}]},
                {"kind": "UniversalJoint", "motion_axes": []},
            ],
            "muscles": [
                {
                    "kind": "Millard2012EquilibriumMuscle",
                    "path_points": [{}, {}],
                    "path_wraps": [{}],
                }
            ],
            "wrap_objects": [{}],
        }
        report = runtime_compatibility_report(
            model,
            read_json(ROOT / "config/numi-runtime-contract.v1.json"),
        )
        self.assertEqual(report["skeleton"]["status"], "blocked")
        self.assertEqual(report["skeleton"]["unsupported_joint_kinds"], {"UniversalJoint": 1})
        self.assertEqual(
            report["muscle_tendon"]["status"], "compatible_bounded_static_equilibrium"
        )
        self.assertEqual(report["source_model"]["muscle_path_wraps"], 1)
        self.assertEqual(report["source_model"]["muscle_curve_kinds"], {})

    def test_locked_zero_universal_joint_lowers_exactly_as_fixed(self) -> None:
        report = runtime_compatibility_report(
            {
                "model_id": "locked-universal-fixture",
                "joints": [
                    {
                        "id": "locked_wrist",
                        "kind": "UniversalJoint",
                        "coordinates": [
                            {"id": "q0", "locked": "true", "default_value": 0.0},
                            {"id": "q1", "locked": "true", "default_value": 0.0},
                        ],
                        "motion_axes": [],
                    }
                ],
                "muscles": [],
                "wrap_objects": [],
            },
            read_json(ROOT / "config/numi-runtime-contract.v1.json"),
        )
        self.assertEqual(report["skeleton"]["status"], "compatible")
        self.assertEqual(report["skeleton"]["unsupported_joint_kinds"], {})
        self.assertEqual(
            report["skeleton"]["exact_locked_joint_lowerings"][0]["numi_primitive"],
            "fixed",
        )

    def test_custom_joint_evaluator_keeps_function_values_and_derivatives(self) -> None:
        joint = {
            "id": "fixture_custom",
            "kind": "CustomJoint",
            "coordinates": [{"id": "q", "default_value": 0.25}],
            "motion_axes": [
                {
                    "id": "rotation1",
                    "coordinates": "q",
                    "axis": [1.0, 0.0, 0.0],
                    "function_kind": "LinearFunction",
                    "function_parameters": {"coefficients": [2.0, 1.0]},
                },
                {
                    "id": "rotation2",
                    "coordinates": "q",
                    "axis": [0.0, 1.0, 0.0],
                    "function_kind": "PolynomialFunction",
                    "function_parameters": {"coefficients": [3.0, 2.0, 1.0]},
                },
                {
                    "id": "rotation3",
                    "coordinates": "q",
                    "axis": [0.0, 0.0, 1.0],
                    "function_kind": "SimmSpline",
                    "function_parameters": {"x": [0.0, 1.0], "y": [0.0, 2.0]},
                },
                {
                    "id": "translation1",
                    "coordinates": "",
                    "axis": [1.0, 0.0, 0.0],
                    "function_kind": "Constant",
                    "function_parameters": {"value": 4.0},
                },
                {
                    "id": "translation2",
                    "coordinates": "",
                    "axis": [0.0, 1.0, 0.0],
                    "function_kind": "Constant",
                    "function_parameters": {"value": 0.0},
                },
                {
                    "id": "translation3",
                    "coordinates": "",
                    "axis": [0.0, 0.0, 1.0],
                    "function_kind": "Constant",
                    "function_parameters": {"value": 0.0},
                },
            ],
        }
        result = evaluate_opensim_custom_joint(joint, coordinate_velocities={"q": -0.5})
        self.assertEqual(result["coordinate_values"], {"q": 0.25})
        self.assertEqual(result["coordinate_velocities"], {"q": -0.5})
        self.assertEqual(result["axes"][0]["displacement"], 1.5)
        self.assertEqual(result["axes"][0]["derivative"], 2.0)
        self.assertEqual(result["axes"][1]["displacement"], 1.6875)
        self.assertEqual(result["axes"][1]["derivative"], 3.5)
        self.assertEqual(result["axes"][1]["second_derivative"], 6.0)
        self.assertEqual(result["axes"][2]["displacement"], 0.5)
        self.assertEqual(result["axes"][2]["derivative"], 2.0)
        self.assertEqual(result["axes"][3]["spatial_kind"], "translation")
        self.assertEqual(
            result["spatial_transform"]["translation_parent_frame_m"], [4.0, 0.0, 0.0]
        )
        self.assertEqual(len(result["spatial_transform"]["motion_subspace_parent_frame"]), 1)
        self.assertEqual(len(result["spatial_transform"]["motion_subspace_dot_parent_frame"]), 1)

    def test_custom_joint_ir_carries_source_tables_and_default_test_vector(self) -> None:
        joint = {
            "id": "fixture_custom",
            "kind": "CustomJoint",
            "coordinates": [{"id": "q", "default_value": 0.0}],
            "motion_axes": [
                {
                    "id": f"axis_{index}",
                    "coordinates": "q" if index == 0 else "",
                    "axis": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ][index],
                    "function_kind": "LinearFunction" if index == 0 else "Constant",
                    "function_parameters": (
                        {"coefficients": [1.0, 0.0]} if index == 0 else {"value": 0.0}
                    ),
                }
                for index in range(6)
            ],
        }
        result = rajagopal_custom_joint_ir(
            {
                "source_id": "fixture",
                "source_file": "fixture.osim",
                "source_sha256": "b" * 64,
                "model_id": "fixture",
                "joints": [joint],
            }
        )
        self.assertEqual(result["joint_count"], 1)
        self.assertEqual(result["schema"], "numi.human.opensim-custom-joint-ir.v2")
        self.assertEqual(result["function_kinds"], {"Constant": 5, "LinearFunction": 1})
        self.assertEqual(result["joints"][0]["default_value_test_vector"]["axes"][0]["displacement"], 0.0)
        self.assertEqual(len(result["joints"][0]["unit_velocity_test_vectors"]), 1)

    def test_custom_joint_gpu_artifact_uses_the_pinned_core_abi(self) -> None:
        joint = {
            "id": "fixture_custom",
            "kind": "CustomJoint",
            "coordinates": [{"id": "q", "default_value": 0.25}],
            "motion_axes": [
                {
                    "id": "rotation1",
                    "coordinates": "q",
                    "axis": [1.0, 0.0, 0.0],
                    "function_kind": "LinearFunction",
                    "function_parameters": {"coefficients": [2.0, 1.0]},
                },
                *[
                    {
                        "id": f"axis_{index}",
                        "coordinates": "",
                        "axis": [
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ][index - 1],
                        "function_kind": "Constant",
                        "function_parameters": {"value": 0.0},
                    }
                    for index in range(1, 6)
                ],
            ],
        }
        manifest, artifacts = rajagopal_custom_joint_gpu_artifacts(
            {
                "source_id": "fixture",
                "source_file": "fixture.osim",
                "source_sha256": "c" * 64,
                "model_id": "fixture",
                "joints": [joint],
            }
        )
        program = artifacts[Path("opensim-spatial-programs/fixture_custom.mrospatial")]
        self.assertEqual(len(program), 2512)
        self.assertEqual(struct.unpack_from("<4I", program), (1, 1, 0, 0))
        self.assertEqual(struct.unpack_from("<4I", program, 16), (1, 0, 2, 0))
        self.assertEqual(manifest["program_count"], 1)
        self.assertEqual(manifest["programs"][0]["coordinate_ids"], ["q"])
        default_input = artifacts[
            Path("opensim-spatial-programs/fixture_custom.default.mrospatialinput")
        ]
        self.assertEqual(len(default_input), 64)
        self.assertEqual(struct.unpack("<16f", default_input)[:8], (0.25,) + (0.0,) * 7)
        unit_input = artifacts[
            Path("opensim-spatial-programs/fixture_custom.velocity-q.mrospatialinput")
        ]
        self.assertEqual(struct.unpack("<16f", unit_input)[8:], (1.0,) + (0.0,) * 7)

    def test_millard_muscle_ir_preserves_curve_path_and_wrap_records(self) -> None:
        parameters = {
            "max_isometric_force": 1000.0,
            "optimal_fiber_length": 0.1,
            "tendon_slack_length": 0.2,
            "pennation_angle_at_optimal": 0.1,
            "ignore_tendon_compliance": "false",
            "fiber_damping": 0.1,
            "default_activation": 0.01,
            "minimum_activation": 0.01,
            "TendonForceLengthCurve": [],
        }
        result = rajagopal_millard_muscle_ir(
            {
                "source_id": "fixture",
                "source_file": "fixture.osim",
                "source_sha256": "d" * 64,
                "model_id": "fixture",
                "bodies": [{"id": "pelvis"}, {"id": "femur"}],
                "wrap_objects": [{"id": "pelvis_wrap"}],
                "muscles": [
                    {
                        "id": "iliacus",
                        "kind": "Millard2012EquilibriumMuscle",
                        "parameters": parameters,
                        "curves": {
                            "ActiveForceLengthCurve": {"parameters": {}},
                            "FiberForceLengthCurve": {"parameters": {}},
                            "ForceVelocityCurve": {"parameters": {}},
                            "TendonForceLengthCurve": {"parameters": {}},
                        },
                        "path_points": [
                            {"parent_frame": "/bodyset/pelvis", "location_m": [0.0, 0.0, 0.0]},
                            {"parent_frame": "/bodyset/femur", "location_m": [0.0, 0.1, 0.0]},
                            {"parent_frame": "/bodyset/femur", "location_m": [0.1, 0.1, 0.0]},
                        ],
                        "path_wraps": [
                            {
                                "wrap_object": "pelvis_wrap",
                                "method": "hybrid",
                                "range": [2.0, 3.0],
                            }
                        ],
                        "source_xml": "<Millard2012EquilibriumMuscle />",
                    }
                ],
            }
        )
        self.assertEqual(result["muscle_count"], 1)
        self.assertEqual(result["path_point_count"], 3)
        self.assertEqual(result["path_wrap_count"], 1)
        self.assertEqual(result["muscles"][0]["parameters"], parameters)
        self.assertEqual(result["muscles"][0]["path_wraps"][0]["method"], "hybrid")
        self.assertEqual(result["muscles"][0]["path_wraps"][0]["range"], [2.0, 3.0])

    def test_rigid_skeleton_ir_resolves_ground_and_custom_joint_program_link(self) -> None:
        result = rajagopal_rigid_skeleton_ir(
            {
                "source_id": "fixture",
                "source_file": "fixture.osim",
                "source_sha256": "e" * 64,
                "model_id": "fixture",
                "bodies": [{"id": "pelvis"}],
                "joints": [
                    {
                        "id": "ground_pelvis",
                        "kind": "CustomJoint",
                        "parent_frame": "ground",
                        "child_frame": "/bodyset/pelvis",
                        "coordinates": [],
                        "frames": [],
                        "motion_axes": [],
                        "source_xml": "<CustomJoint />",
                    }
                ],
            }
        )
        self.assertEqual(result["root_body"], "pelvis")
        self.assertEqual(result["body_count"], 1)
        self.assertEqual(result["joint_count"], 1)
        self.assertEqual(result["joints"][0]["parent_body"], None)
        self.assertEqual(
            result["joints"][0]["lowering"]["program_file"],
            "opensim-spatial-programs/ground_pelvis.mrospatial",
        )

    def test_core_reference_artifact_keeps_rigid_tree_and_function_program_abi(self) -> None:
        axes = [
            {
                "id": f"axis_{index}",
                "coordinates": "pelvis_tilt" if index == 0 else "",
                "axis": [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ][index],
                "function_kind": "LinearFunction" if index == 0 else "Constant",
                "function_parameters": (
                    {"coefficients": [1.0, 0.0]} if index == 0 else {"value": 0.0}
                ),
            }
            for index in range(6)
        ]
        manifest, payload = rajagopal_core_reference_artifact(
            {
                "source_id": "fixture",
                "source_file": "fixture.osim",
                "source_sha256": "f" * 64,
                "model_id": "fixture",
                "bodies": [
                    {
                        "id": "pelvis",
                        "mass_kg": 11.2,
                        "mass_center_m": [0.0, 0.1, 0.0],
                        "inertia_kg_m2": {
                            "xx": 1.0, "xy": 0.0, "xz": 0.0,
                            "yy": 2.0, "yz": 0.0, "zz": 3.0,
                        },
                    }
                ],
                "joints": [
                    {
                        "id": "ground_pelvis",
                        "kind": "CustomJoint",
                        "parent_frame": "ground",
                        "child_frame": "/bodyset/pelvis",
                        "coordinates": [
                            {
                                "id": "pelvis_tilt", "default_value": 0.25,
                                "range": [-1.0, 1.0], "clamped": True,
                            }
                        ],
                        "frames": [],
                        "motion_axes": axes,
                    }
                ],
            }
        )
        header = struct.unpack_from("<8s9I32s", payload)
        self.assertEqual(header[:10], (b"NHRIGID1", 1, 5, 1, 2, 1, 1, 1, 1, 0))
        self.assertEqual(header[10], bytes.fromhex("f" * 64))
        self.assertEqual(manifest["body_order"], ["__ground__", "pelvis"])
        self.assertEqual(manifest["function_based_program_count"], 1)
        self.assertEqual(manifest["payload"]["bytes"], len(payload))
        self.assertEqual(len(payload), 3276)

    def test_opensim_archive_parser_preserves_selected_member_and_archive_hash(self) -> None:
        source = """<?xml version=\"1.0\"?>
<OpenSimDocument Version=\"40000\"><Model name=\"fixture\">
  <BodySet><objects><Body name=\"pelvis\"><mass>1</mass><mass_center>0 0 0</mass_center><inertia_xx>1</inertia_xx><inertia_yy>1</inertia_yy><inertia_zz>1</inertia_zz></Body></objects></BodySet>
  <JointSet><objects /></JointSet><ForceSet><objects /></ForceSet>
</Model></OpenSimDocument>"""
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "upper.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("nested/bimanual_fixture.osim", source)
            result = parse_opensim_archive(archive, "upper")
        self.assertEqual(result["model_id"], "fixture")
        self.assertEqual(result["source_file"], "nested/bimanual_fixture.osim")
        self.assertEqual(result["source_archive"]["file"], "upper.zip")
        self.assertEqual(len(result["source_archive"]["sha256"]), 64)

    def test_rajagopal_distal_pin_preview_preserves_source_rigid_bodies(self) -> None:
        bodies = []
        for identifier in ("tibia_r", "talus_r", "calcn_r", "toes_r"):
            bodies.append(
                {
                    "id": identifier,
                    "mass_kg": 1.0,
                    "mass_center_m": [0.0, 0.0, 0.0],
                    "inertia_kg_m2": {
                        "xx": 1.0,
                        "yy": 1.0,
                        "zz": 1.0,
                        "xy": 0.0,
                        "xz": 0.0,
                        "yz": 0.0,
                    },
                }
            )
        joints = []
        for name, parent, child in (
            ("ankle_r", "tibia_r", "talus_r"),
            ("subtalar_r", "talus_r", "calcn_r"),
            ("mtp_r", "calcn_r", "toes_r"),
        ):
            joints.append(
                {
                    "id": name,
                    "kind": "PinJoint",
                    "parent_frame": f"{parent}_offset",
                    "child_frame": f"{child}_offset",
                    "coordinates": [{"id": name + "_coordinate", "range": [-1.0, 1.0]}],
                    "frames": [
                        {
                            "id": f"{parent}_offset",
                            "parent_frame": f"/bodyset/{parent}",
                            "translation_m": [0.0, -0.1, 0.0],
                            "orientation_rad": [0.0, 0.0, 0.0],
                        },
                        {
                            "id": f"{child}_offset",
                            "parent_frame": f"/bodyset/{child}",
                            "translation_m": [0.0, 0.0, 0.0],
                            "orientation_rad": [0.0, 0.0, 0.0],
                        },
                    ],
                }
            )
        urdf, report = build_rajagopal_distal_pin_preview(
            {
                "source_id": "fixture",
                "source_file": "fixture.osim",
                "source_sha256": "a" * 64,
                "model_id": "fixture",
                "bodies": bodies,
                "joints": joints,
                "muscles": [],
            },
            "right",
        )
        self.assertIn('<robot name="numilab_human_rajagopal_right_distal_pin_preview">', urdf)
        self.assertIn('<joint name="ankle_r" type="revolute">', urdf)
        self.assertEqual(report["joint_lowering"][0]["axis_in_child_body_frame"], [0.0, 0.0, 1.0])
        self.assertEqual(report["included_bodies"][0]["inertia_kg_m2"]["xx"], 1.0)
        self.assertEqual(report["excluded_source_muscles"], 0)

    def test_gate_report_parses_available_upper_archive_for_runtime_compatibility(self) -> None:
        source = """<?xml version=\"1.0\"?>
<OpenSimDocument Version=\"40000\"><Model name=\"upper_fixture\">
  <BodySet><objects><Body name=\"upper\"><mass>1</mass><mass_center>0 0 0</mass_center><inertia_xx>1</inertia_xx><inertia_yy>1</inertia_yy><inertia_zz>1</inertia_zz></Body></objects></BodySet>
  <JointSet><objects><PinJoint name=\"elbow\"><parent_body>ground</parent_body><child_body>upper</child_body><coordinates><Coordinate name=\"flexion\"><default_value>0</default_value><range>-1 1</range></Coordinate></coordinates></PinJoint></objects></JointSet><ForceSet><objects /></ForceSet>
</Model></OpenSimDocument>"""
        lock = {
            "sources": {
                "bodyparts3d_4": {"files": {}},
                "rajagopal_lai_uhlrich_2023": {"sha256": "def"},
                "mobl_arms_upper_extremity": {"release_file": "upper.zip", "license": "non-commercial"},
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "upper.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("bimanual_fixture.osim", source)
            report = gate_report(
                sources=directory,
                upper_archive=archive,
                source_lock=lock,
                runtime_contract=read_json(ROOT / "config/numi-runtime-contract.v1.json"),
            )
        upper = report["runtime_compatibility"]["upper_extremities"]
        self.assertEqual(upper["source_model"]["id"], "upper_fixture")
        self.assertEqual(upper["skeleton"]["status"], "compatible")

    def test_gate_report_parses_pinned_public_upper_model_as_unimanual_variant(self) -> None:
        source = """<?xml version=\"1.0\"?>
<OpenSimDocument Version=\"40000\"><Model name=\"upper_fixture\">
  <BodySet><objects><Body name=\"upper\"><mass>1</mass><mass_center>0 0 0</mass_center><inertia_xx>1</inertia_xx><inertia_yy>1</inertia_yy><inertia_zz>1</inertia_zz></Body></objects></BodySet>
  <JointSet><objects><PinJoint name=\"elbow\"><parent_body>ground</parent_body><child_body>upper</child_body><coordinates><Coordinate name=\"flexion\"><default_value>0</default_value><range>-1 1</range></Coordinate></coordinates></PinJoint></objects></JointSet><ForceSet><objects /></ForceSet>
</Model></OpenSimDocument>"""
        lock = {
            "sources": {
                "bodyparts3d_4": {"files": {}},
                "rajagopal_lai_uhlrich_2023": {"sha256": "def"},
                "mobl_arms_upper_extremity": {
                    "release_file": "upper.zip", "license": "non-commercial",
                },
                "mobl_arms_ceinms_41_public_mirror": {
                    "model_file": "MOBL_ARMS_41.osim",
                    "sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "license": "non-commercial",
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = directory / "MOBL_ARMS_41.osim"
            model.write_text(source)
            report = gate_report(
                sources=directory,
                upper_archive=None,
                upper_public_model=model,
                source_lock=lock,
                runtime_contract=read_json(ROOT / "config/numi-runtime-contract.v1.json"),
            )
        artifact = report["source_artifacts"]["mobl_arms_upper_extremity"]
        upper = report["runtime_compatibility"]["upper_extremities"]
        self.assertEqual(artifact["source_variant"], "public_unimanual_mirror")
        self.assertEqual(artifact["status"], "ready_for_import_public_mirror")
        self.assertEqual(upper["source_model"]["id"], "upper_fixture")
        self.assertEqual(upper["skeleton"]["status"], "compatible")


if __name__ == "__main__":
    unittest.main()
