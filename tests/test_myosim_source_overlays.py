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
PATELLA_REBASE_PATH = ROOT / "config" / \
    "myosim-patella-neutral-coordinate-rebase.v1.json"


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
                assets_suffix = "myo_sim/models/leg/assets/myolegs_assets.xml"
                assets_members = [member for member in archive.getmembers()
                                  if member.name.endswith(assets_suffix)]
                self.assertEqual(len(assets_members), 1)
                assets_stream = archive.extractfile(assets_members[0])
                self.assertIsNotNone(assets_stream)
                assets_target = package / "checkout/myo_sim/models/leg/assets/myolegs_assets.xml"
                assets_target.parent.mkdir(parents=True, exist_ok=True)
                assets_target.write_bytes(assets_stream.read())

            self.assertEqual(
                sha256(target), overlay["change"]["file_sha256_before"]
            )
            first = apply_myo_sim_source_overlays(root)
            self.assertEqual(
                first["range_after_m"], [-0.006792, -7.69254e-11]
            )
            rebase = first["additional_source_overlays"][0]
            rebase_manifest = json.loads(PATELLA_REBASE_PATH.read_text())
            self.assertEqual(
                rebase["id"],
                "myosim-bilateral-patella-qpos0-neutral-coordinate-rebase-20261003",
            )
            for row in rebase_manifest["coordinate_reparameterization"]["files"]:
                self.assertEqual(
                    sha256(package / row["file"]),
                    row["file_sha256_after"],
                )
            chain_rebase = next(
                row for row in rebase_manifest["coordinate_reparameterization"]["files"]
                if row["file"] == overlay["change"]["file"]
            )
            self.assertEqual(sha256(target), chain_rebase["file_sha256_after"])
            self.assertEqual(apply_myo_sim_source_overlays(root), first)

            assets_target = package / "checkout/myo_sim/models/leg/assets/myolegs_assets.xml"
            assets_target.write_bytes(assets_target.read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimeError, "pinned patella rebase original"):
                apply_myo_sim_source_overlays(root)

            target.write_bytes(target.read_bytes() + b"\n")
            with self.assertRaisesRegex(RuntimeError, "both the pinned original"):
                apply_myo_sim_source_overlays(root)

    def test_compiled_bilateral_ranges_cover_projected_knee_poses(self):
        import mujoco
        import numpy as np
        from numilab_human.lower_limb_pose_audit import _patellar_anteriority
        from numilab_human.myosim_bone_proximity import _compiled_meshes_by_body
        from myo_sim.build.compose import build_model
        from numilab_human.lower_limb_pose_audit import POSE_SUITE
        from numilab_human.upper_limb_pose_audit import _pose_qpos

        model = build_model("myofullbody")
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        meshes_by_body = _compiled_meshes_by_body(model, mujoco, np)
        for side in ("r", "l"):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"patella_{side}")
            knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"knee_angle_{side}")
            patella_meshes = meshes_by_body[body]
            self.assertTrue(patella_meshes)
            vertices = np.concatenate([mesh["vertices"] for mesh in patella_meshes])
            vertices = np.einsum(
                "ki,ji->kj", vertices, data.xmat[body].reshape(3, 3)
            ) + data.xpos[body]
            anteriority = _patellar_anteriority(
                vertices, data.xanchor[knee], [0.0, -1.0, 0.0],
                side=side, member_id=f"compiled_patella_{side}",
            )
            self.assertTrue(anteriority["passed"], anteriority)
            self.assertEqual(anteriority["vertices_posterior_or_on_knee_anchor_plane"], 0)

        neutral_qpos, _, _ = _pose_qpos(model, (), mujoco, np)
        for name in (
            "knee_angle_beta_translation1_r", "knee_angle_beta_translation1_l",
            "knee_angle_beta_translation2_r", "knee_angle_beta_translation2_l",
        ):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_index = int(model.jnt_qposadr[joint])
            self.assertAlmostEqual(
                float(neutral_qpos[qpos_index]), float(model.qpos0[qpos_index]), delta=1e-12,
            )
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

        # The rebase translates each dependent coordinate and equality constant
        # together, preserving the projected source pose while making the raw
        # default configuration anterior at both knee-anchor planes.
        rebase_ranges = {
            "knee_angle_beta_translation1_r": [-0.0751923, 0.0],
            "knee_angle_beta_translation1_l": [-0.0751923, 0.0],
            "knee_angle_beta_translation2_r": [-0.03, 0.0],
            "knee_angle_beta_translation2_l": [-0.03, 0.0],
        }
        for name, expected in rebase_ranges.items():
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.assertEqual([float(value) for value in model.jnt_range[joint]], expected)
        for side in ("r", "l"):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"patella_{side}")
            self.assertEqual(model.body_pos[body].tolist(), [0.0443292, -0.4187881, 0.0])
            for coordinate in ("translation1", "translation2"):
                equality = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_EQUALITY,
                    f"knee_angle_beta_{coordinate}_constraint_{side}",
                )
                self.assertEqual(float(model.eq_data[equality, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
