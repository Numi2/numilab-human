from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from subprocess import run

from numilab_human.model import (
    bodyparts_geometry_preflight,
    build_rajagopal_distal_pin_preview,
    evaluate_opensim_custom_joint,
    gate_report,
    parse_bodyparts3d,
    parse_opensim_archive,
    parse_opensim,
    read_json,
    rajagopal_custom_joint_ir,
    runtime_compatibility_report,
    runtime_checkout_gate,
)


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

    def test_runtime_compatibility_does_not_silently_lower_custom_joints_or_muscles(self) -> None:
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
        self.assertEqual(report["skeleton"]["unsupported_joint_kinds"], {"CustomJoint": 1, "UniversalJoint": 1})
        self.assertEqual(report["muscle_tendon"]["status"], "blocked")
        self.assertEqual(report["source_model"]["muscle_path_wraps"], 1)
        self.assertEqual(report["source_model"]["muscle_curve_kinds"], {})

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
        result = evaluate_opensim_custom_joint(joint)
        self.assertEqual(result["coordinate_values"], {"q": 0.25})
        self.assertEqual(result["axes"][0]["displacement"], 1.5)
        self.assertEqual(result["axes"][0]["derivative"], 2.0)
        self.assertEqual(result["axes"][1]["displacement"], 1.6875)
        self.assertEqual(result["axes"][1]["derivative"], 3.5)
        self.assertEqual(result["axes"][2]["displacement"], 0.5)
        self.assertEqual(result["axes"][2]["derivative"], 2.0)
        self.assertEqual(result["axes"][3]["spatial_kind"], "translation")

    def test_custom_joint_ir_carries_source_tables_and_default_test_vector(self) -> None:
        joint = {
            "id": "fixture_custom",
            "kind": "CustomJoint",
            "coordinates": [{"id": "q", "default_value": 0.0}],
            "motion_axes": [
                {
                    "id": f"axis_{index}",
                    "coordinates": "q" if index == 0 else "",
                    "axis": [1.0, 0.0, 0.0],
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
        self.assertEqual(result["function_kinds"], {"Constant": 5, "LinearFunction": 1})
        self.assertEqual(result["joints"][0]["default_value_test_vector"]["axes"][0]["displacement"], 0.0)

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


if __name__ == "__main__":
    unittest.main()
