"""Render reproducible multi-angle MyoSim source-model validation frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import zlib
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_png(path: Path, image: object) -> None:
    """Write RGB uint8 image data without adding a graphics dependency."""
    import numpy as np

    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise RuntimeError("MuJoCo renderer did not produce RGB pixels")
    height, width, _ = pixels.shape
    scanlines = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    payload = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(scanlines, level=9)) + chunk(b"IEND", b"")
    path.write_bytes(payload)


def _visual_qpos(model: object, mujoco: object, raw_source_rest: bool):
    import numpy as np

    if raw_source_rest:
        return np.asarray(model.qpos0, dtype=float).copy(), {
            "pose_state": "literal_unprojected_source_qpos0",
            "source_joint_equalities_projected": 0,
            "maximum_equality_coordinate_correction": 0.0,
        }
    from .upper_limb_pose_audit import _pose_qpos
    qpos, count, correction = _pose_qpos(model, (), mujoco, np)
    if count <= 0:
        raise RuntimeError("MyoSim visual source model has no active joint equalities")
    return qpos, {
        "pose_state": "source_equality_projected_neutral",
        "source_joint_equalities_projected": count,
        "maximum_equality_coordinate_correction": correction,
    }


def _patella_body_center_anteriority(model: object, data: object, mujoco: object,
                                    raw_source_rest: bool) -> list[dict[str, object]]:
    """Reject a displayed neutral pose with either patella behind its knee.

    The compiled BodyParts3D full-mesh audit is separate. This small source-
    visual preflight catches an omitted/changed equality projection before a
    new rendered source image can be mistaken for current neutral anatomy.
    """
    import numpy as np

    # Neutral MyoSim source world anterior is -Y. The pelvis body's local
    # frame is rotated relative to source world and must not rotate this axis.
    anterior = np.array([0., -1., 0.])
    rows = []
    for side in ("r", "l"):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "patella_" + side)
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_" + side)
        if body < 0 or joint < 0:
            raise RuntimeError(f"MyoSim visual is missing patella/knee source identity for {side}")
        offset = float(np.dot(np.asarray(data.xpos[body])-np.asarray(data.xanchor[joint]), anterior))
        if not np.isfinite(offset):
            raise RuntimeError(f"MyoSim visual patella {side} anteriority is nonfinite")
        passed = offset >= .025
        rows.append({"side": side, "patella_body": "patella_" + side,
                     "knee_joint": "knee_angle_" + side,
                     "source_body_center_anterior_offset_m": offset,
                     "source_specific_minimum_m": .025,
                     "passed_display_gate": passed})
        if not raw_source_rest and not passed:
            raise RuntimeError(
                f"MyoSim projected visual would place patella_{side} behind/too near "
                f"the knee anchor: center anterior offset {offset:.6g} m; "
                "source-specific display minimum 0.025 m")
    return rows


def render(sources: Path, output: Path, raw_source_rest: bool = False) -> dict[str, object]:
    # MuJoCo chooses a GL backend when imported, so set this before the import.
    os.environ.setdefault("MUJOCO_GL", "glfw")
    try:
        import mujoco
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - source environment only
        raise RuntimeError("MyoSim rendering requires mujoco and the pinned myo-sim checkout") from error
    archive = sources / "myosim" / "myo_sim-33c89c2b.tar.gz"
    if not archive.is_file():
        raise RuntimeError(f"MyoSim source archive is absent: {archive}")
    output.mkdir(parents=True, exist_ok=True)
    model = build_model("myofullbody")
    data = mujoco.MjData(model)
    qpos, pose = _visual_qpos(model, mujoco, raw_source_rest)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    pose["patella_body_center_anteriority"] = _patella_body_center_anteriority(
        model, data, mujoco, raw_source_rest)
    # The authored model declares a 640x480 MuJoCo offscreen framebuffer.
    renderer = mujoco.Renderer(model, height=480, width=640)
    views = {
        "front": {"azimuth_degrees": 90.0, "elevation_degrees": -5.0},
        "side": {"azimuth_degrees": 0.0, "elevation_degrees": -5.0},
        "rear": {"azimuth_degrees": -90.0, "elevation_degrees": -5.0},
    }
    outputs = []
    for name, camera_settings in views.items():
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (-0.02, 0.10, 1.10)
        camera.distance = 3.0
        camera.azimuth = camera_settings["azimuth_degrees"]
        camera.elevation = camera_settings["elevation_degrees"]
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
        path = output / f"myosim-fullbody-{name}.png"
        _write_png(path, image)
        outputs.append({"id": name, "file": path.name, "sha256": _sha256(path), **camera_settings})
    renderer.close()
    return {
        "schema": "numi.human.myosim-source-visual-validation.v2",
        "source": {
            "model": "myofullbody",
            "revision": "33c89c2bde282553dde3f526768eb3bdcfaa7649",
            "archive_sha256": _sha256(archive),
            "mujoco_version": mujoco.__version__,
        },
        "pose": pose,
        "views": outputs,
        "evidence_boundary": (
            "These are rendered MyoSim source frames from three camera angles in the explicitly "
            "recorded source pose. The projected neutral pose applies active source joint "
            "equalities before rendering; raw unprojected qpos0 is diagnostic only and "
            "can place the patella behind the knee anchor. "
            "The source-visual display gate checks patella body centers only; the separate "
            "compiled-bone audit checks every patella vertex across posed states. "
            "They validate source-model visibility, not Core-native rendering, contact, or a locomotion rollout."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--raw-source-rest", action="store_true",
        help="render literal unprojected qpos0 for source diagnostics only",
    )
    arguments = parser.parse_args(argv)
    try:
        manifest = render(
            arguments.sources.resolve(), arguments.output.resolve(), arguments.raw_source_rest,
        )
    except RuntimeError as error:
        print(f"numilab-human MyoSim visual: {error}", file=sys.stderr)
        return 2
    manifest_path = arguments.output.resolve() / "myosim-fullbody-source-visual.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
