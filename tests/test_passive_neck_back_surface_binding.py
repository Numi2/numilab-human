from __future__ import annotations

import unittest
from unittest.mock import patch

from numilab_human import model
from numilab_human.model import ImportError as ModelImportError


class PassiveNeckBackSurfaceBindingTests(unittest.TestCase):
    def test_neck_back_rows_have_explicit_laterality_and_anatomical_supports(self) -> None:
        surfaces = model._bodyparts_myosim_surface_specifications()
        self.assertEqual(len(surfaces), 158)
        self.assertEqual(
            [(row["member_id"], row["source_name"]) for row in surfaces[150:]],
            [
                ("FJ1595", "right sternocleidomastoid"),
                ("FJ1573", "left sternocleidomastoid"),
                ("FJ1520", "ascending part of right trapezius"),
                ("FJ1520M", "ascending part of left trapezius"),
                ("FJ1554", "transverse part of right trapezius"),
                ("FJ1554M", "transverse part of left trapezius"),
                ("FJ1521", "descending part of right trapezius"),
                ("FJ1521M", "descending part of left trapezius"),
            ],
        )
        right_scm, left_scm = surfaces[150:152]
        self.assertEqual(
            [row["body"] for row in right_scm["passive_visual_binding"]["supports"]],
            ["torso", "clavicle_r", "head"],
        )
        self.assertEqual(
            [row["body"] for row in left_scm["passive_visual_binding"]["supports"]],
            ["torso", "clavicle_l", "head"],
        )
        self.assertEqual(
            right_scm["passive_visual_binding"]["supports"][2]["mirror_source_member_ids"],
            ["FJ3281", "FJ3309"],
        )
        self.assertEqual(
            [row["body"] for row in surfaces[157]["passive_visual_binding"]["supports"]],
            ["head", "cervical_spine", "torso", "clavicle_l", "scapula_l"],
        )
        self.assertTrue(all(not row.get("myosim_muscles") for row in surfaces[150:]))

    def test_default_keeps_routed_inventory_and_explicit_subset_retains_stable_ids(self) -> None:
        surfaces = model._bodyparts_myosim_surface_specifications()
        selected = model._bodyparts_myosim_selected_visual_surfaces(surfaces, None)
        self.assertEqual([sid for sid, _ in selected], list(range(1, 151)))
        self.assertEqual([row for _, row in selected], surfaces[:150])
        selected = model._bodyparts_myosim_selected_visual_surfaces(surfaces, {152, 157})
        self.assertEqual([(sid, row["member_id"]) for sid, row in selected],
                         [(152, "FJ1573"), (157, "FJ1521")])

    def test_passive_surface_map_rejects_non_object_binding(self) -> None:
        with patch.object(
            model,
            "read_json",
            return_value={
                "schema": "numi.human.bodyparts3d-myosim-surface-map.v1",
                "entries": [
                    {
                        "member_id": "FJ1595",
                        "source_name": "right sternocleidomastoid",
                        "passive_visual_binding": [],
                    }
                ],
            },
        ):
            with self.assertRaisesRegex(ModelImportError, "binding must be an object"):
                model._bodyparts_myosim_surface_specifications()

    def test_registered_support_weighting_handles_five_frames_and_exact_lock(self) -> None:
        weights, report = model._bodyparts_registered_attachment_support_weights(
            [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]],
            ["head", "cervical_spine", "torso", "clavicle", "scapula"],
            {
                "head": [[0.0, 0.0, 0.0]],
                "cervical_spine": [[0.0, 0.01, 0.0]],
                "torso": [[0.0, 0.02, 0.0]],
                "clavicle": [[0.0, 0.03, 0.0]],
                "scapula": [[0.0, 0.04, 0.0]],
            },
            0.004,
            0.030,
            0.003,
            "test surface",
        )
        self.assertEqual(len(weights), 2)
        self.assertEqual(len(weights[0]), 5)
        self.assertEqual(weights[0], [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(sum(weights[1]), 1.0)
        self.assertEqual(report["support_body_count"], 5)
        self.assertEqual(report["method"], "exact_euclidean_nearest_registered_attachment_support_vertex_proximity")

    def test_registered_support_weighting_rejects_empty_tissue(self) -> None:
        with self.assertRaisesRegex(ModelImportError, "no passive tissue vertices"):
            model._bodyparts_registered_attachment_support_weights(
                [],
                ["torso", "head"],
                {"torso": [[0.0, 0.0, 0.0]], "head": [[1.0, 0.0, 0.0]]},
                0.004,
                0.030,
                0.003,
                "empty surface",
            )


if __name__ == "__main__":
    unittest.main()
