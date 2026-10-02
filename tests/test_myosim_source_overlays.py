"""Fail-closed replay checks for the explicit source-bound MyoSim overlay."""

import hashlib
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

from numilab_human.myosim_source_overlays import apply_myo_sim_source_overlays


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = ROOT / "Sources" / "myosim"
OVERLAY_PATH = SOURCE_PACKAGE / "source-overlays" / \
    "myosim-left-knee-translation2-range.v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MyoSimSourceOverlayTests(unittest.TestCase):
    def test_archive_extraction_gets_exact_idempotent_range_overlay(self):
        overlay = json.loads(OVERLAY_PATH.read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "myosim"
            package.mkdir()
            shutil.copy2(
                SOURCE_PACKAGE / overlay["source"]["archive"],
                package / overlay["source"]["archive"],
            )
            overlay_target = package / OVERLAY_PATH.relative_to(SOURCE_PACKAGE)
            overlay_target.parent.mkdir(parents=True)
            shutil.copy2(OVERLAY_PATH, overlay_target)
            target = package / overlay["change"]["file"]
            target.parent.mkdir(parents=True)
            archive_member_suffix = (
                "myo_sim/models/leg/assets/myolegs_chain.xml"
            )
            with tarfile.open(package / overlay["source"]["archive"], "r:gz") as archive:
                members = [member for member in archive.getmembers()
                           if member.name.endswith(archive_member_suffix)]
                self.assertEqual(len(members), 1)
                member_stream = archive.extractfile(members[0])
                self.assertIsNotNone(member_stream)
                target.write_bytes(member_stream.read())

            self.assertEqual(
                sha256(target), overlay["change"]["file_sha256_before"]
            )
            first = apply_myo_sim_source_overlays(root)
            self.assertEqual(
                sha256(target), overlay["change"]["file_sha256_after"]
            )
            self.assertEqual(
                first["range_after_m"], [-0.006792, -7.69254e-11]
            )
            self.assertEqual(apply_myo_sim_source_overlays(root), first)

            target.write_bytes(target.read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimeError, "both the pinned original"):
                apply_myo_sim_source_overlays(root)

    def test_compiled_bilateral_ranges_cover_projected_knee_poses(self):
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
        from numilab_human.lower_limb_pose_audit import POSE_SUITE
        from numilab_human.upper_limb_pose_audit import _pose_qpos

        model = build_model("myofullbody")
        joints = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in (
                "knee_angle_translation2_l", "knee_angle_translation2_r",
                "knee_angle_rotation2_l", "knee_angle_rotation2_r",
            )
        }
        left_translation_range = [float(value) for value in model.jnt_range[joints["knee_angle_translation2_l"]]]
        right_translation_range = [float(value) for value in model.jnt_range[joints["knee_angle_translation2_r"]]]
        self.assertTrue(model.jnt_limited[joints["knee_angle_translation2_l"]])
        self.assertEqual(left_translation_range, [-0.006792, -7.69254e-11])
        self.assertEqual(left_translation_range, [-value for value in right_translation_range[::-1]])
        left_rotation_range = [float(value) for value in model.jnt_range[joints["knee_angle_rotation2_l"]]]
        right_rotation_range = [float(value) for value in model.jnt_range[joints["knee_angle_rotation2_r"]]]
        self.assertTrue(model.jnt_limited[joints["knee_angle_rotation2_l"]])
        self.assertEqual(left_rotation_range, [-0.00167821, 0.0335354])
        self.assertTrue(model.jnt_limited[joints["knee_angle_rotation2_r"]])
        self.assertEqual(left_rotation_range, right_rotation_range)

        selected = dict(POSE_SUITE)
        for pose_name in (
            "neutral", "bilateral_knee_flexion", "bilateral_deep_crouch",
            "bilateral_functional_crouch",
        ):
            qpos, _, _ = _pose_qpos(model, selected[pose_name], mujoco, np)
            for name, joint in joints.items():
                value = float(qpos[int(model.jnt_qposadr[joint])])
                limits = model.jnt_range[joint]
                self.assertGreaterEqual(value, float(limits[0]) - 1e-12, f"{pose_name}:{name}")
                self.assertLessEqual(value, float(limits[1]) + 1e-12, f"{pose_name}:{name}")


if __name__ == "__main__":
    unittest.main()
