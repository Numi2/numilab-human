from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from subprocess import run

from numilab_human.model import gate_report, parse_bodyparts3d, parse_opensim


ROOT = Path(__file__).resolve().parents[1]


class ImporterTests(unittest.TestCase):
    def test_numi_workspace_command_describes_itself(self) -> None:
        command = ROOT / ".numi/commands/human"
        result = run([command, "--numi-describe"], capture_output=True, text=True, check=True)
        self.assertEqual(
            result.stdout,
            "Build provenance-locked NumiLab Human v1 source artifacts.\n",
        )

    def test_opensim_parser_retains_mechanical_fields(self) -> None:
        source = """<?xml version=\"1.0\"?>
<OpenSimDocument Version=\"40000\"><Model name=\"fixture\">
  <gravity>0 -9.81 0</gravity>
  <BodySet><objects><Body name=\"pelvis\"><mass>11.2</mass><mass_center>0 0.1 0</mass_center><inertia_xx>1</inertia_xx><inertia_yy>2</inertia_yy><inertia_zz>3</inertia_zz></Body></objects></BodySet>
  <JointSet><objects><CustomJoint name=\"hip\"><socket_parent_frame>/ground</socket_parent_frame><socket_child_frame>/bodyset/pelvis</socket_child_frame><coordinates><Coordinate name=\"hip_flexion\"><default_value>0</default_value><range>-1 1</range><clamped>true</clamped><locked>false</locked></Coordinate></coordinates><frames><PhysicalOffsetFrame name=\"hip_center\"><socket_parent>/bodyset/pelvis</socket_parent><translation>0 0.1 0</translation><orientation>0 0 0</orientation></PhysicalOffsetFrame></frames><SpatialTransform><TransformAxis name=\"rotation1\"><coordinates>hip_flexion</coordinates><axis>0 0 1</axis><function><LinearFunction><coefficients>1 0</coefficients></LinearFunction></function></TransformAxis></SpatialTransform></CustomJoint></objects></JointSet>
  <ForceSet><objects><Millard2012EquilibriumMuscle name=\"iliacus\"><max_isometric_force>1000</max_isometric_force><optimal_fiber_length>0.1</optimal_fiber_length><tendon_slack_length>0.2</tendon_slack_length><GeometryPath><PathPointSet><objects><PathPoint name=\"origin\"><socket_parent_frame>/bodyset/pelvis</socket_parent_frame><location>0 0 0</location></PathPoint></objects></PathPointSet><PathWrapSet><objects><PathWrap name=\"wrap\"><socket_wrap_object>/bodyset/pelvis/wrap_object</socket_wrap_object></PathWrap></objects></PathWrapSet></GeometryPath></Millard2012EquilibriumMuscle></objects></ForceSet>
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
        self.assertEqual(result["joints"][0]["frames"][0]["id"], "hip_center")
        self.assertEqual(result["joints"][0]["motion_axes"][0]["coordinates"], "hip_flexion")
        self.assertEqual(result["joints"][0]["motion_axes"][0]["function_kind"], "LinearFunction")
        self.assertEqual(result["muscles"][0]["path_wraps"][0]["wrap_object"], "/bodyset/pelvis/wrap_object")

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
                "isa_BP3D_4.0_obj_99.zip": ("FJ100.obj",),
                "partof_BP3D_4.0_obj_99.zip": ("FJ200.obj",),
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
        self.assertEqual(lookup["FMA9611"]["element_meshes"][0]["element_id"], "FJ100")
        self.assertTrue(lookup["FMA5865"]["element_meshes"][0]["mesh_present"])

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
            report = gate_report(sources=source, upper_archive=None, source_lock=lock)
        self.assertEqual(report["source_artifacts"]["bodyparts3d_4"][0]["status"], "missing")
        self.assertEqual(report["source_artifacts"]["mobl_arms_upper_extremity"]["status"], "missing_authenticated_archive")
        self.assertEqual(report["gates"][0]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
