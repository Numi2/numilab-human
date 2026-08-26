from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from numilab_human.model import parse_bodyparts3d, parse_opensim


ROOT = Path(__file__).resolve().parents[1]


class ImporterTests(unittest.TestCase):
    def test_opensim_parser_retains_mechanical_fields(self) -> None:
        source = """<?xml version=\"1.0\"?>
<OpenSimDocument Version=\"40000\"><Model name=\"fixture\">
  <gravity>0 -9.81 0</gravity>
  <BodySet><objects><Body name=\"pelvis\"><mass>11.2</mass><mass_center>0 0.1 0</mass_center><inertia_xx>1</inertia_xx><inertia_yy>2</inertia_yy><inertia_zz>3</inertia_zz></Body></objects></BodySet>
  <JointSet><objects><CustomJoint name=\"hip\"><socket_parent_frame>/ground</socket_parent_frame><socket_child_frame>/bodyset/pelvis</socket_child_frame><coordinates><Coordinate name=\"hip_flexion\"><default_value>0</default_value><range>-1 1</range><clamped>true</clamped><locked>false</locked></Coordinate></coordinates></CustomJoint></objects></JointSet>
  <ForceSet><objects><Millard2012EquilibriumMuscle name=\"iliacus\"><max_isometric_force>1000</max_isometric_force><optimal_fiber_length>0.1</optimal_fiber_length><tendon_slack_length>0.2</tendon_slack_length><GeometryPath><PathPointSet><objects><PathPoint name=\"origin\"><socket_parent_frame>/bodyset/pelvis</socket_parent_frame><location>0 0 0</location></PathPoint></objects></PathPointSet></GeometryPath></Millard2012EquilibriumMuscle></objects></ForceSet>
</Model></OpenSimDocument>"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.osim"
            path.write_text(source, encoding="utf-8")
            result = parse_opensim(path, "fixture")
        self.assertEqual(result["model_id"], "fixture")
        self.assertEqual(result["bodies"][0]["mass_kg"], 11.2)
        self.assertEqual(result["joints"][0]["coordinates"][0]["id"], "hip_flexion")
        self.assertEqual(result["muscles"][0]["parameters"]["tendon_slack_length"], 0.2)
        self.assertEqual(result["muscles"][0]["path_points"][0]["parent_frame"], "/bodyset/pelvis")

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
            for archive_name, members in {
                "isa_BP3D_4.0_obj_99.zip": ("BP1.obj", "BP2.obj"),
                "partof_BP3D_4.0_obj_99.zip": ("BP3.obj", "BP4.obj"),
            }.items():
                with zipfile.ZipFile(source / archive_name, "w") as archive:
                    for member in members:
                        archive.writestr(member, "o fixture\nv 0 0 0\n")
            result = parse_bodyparts3d(source, ROOT / "config/anatomy-classification.v1.json")
        lookup = {item["concept_id"]: item for item in result["components"]}
        self.assertEqual(len(result["hierarchy_edges"]), 2)
        self.assertEqual(lookup["FMA9611"]["anatomy_class"], "bone")
        self.assertEqual(lookup["FMA5865"]["anatomy_class"], "nerve_surface")
        self.assertTrue(lookup["FMA9611"]["mesh_present"])


if __name__ == "__main__":
    unittest.main()
