"""Check whether the Open Knee clearance pose has a source-joint admission path.

The Open Knee displacement and the MyoSim body frame are different sources.
This audit deliberately does not equate their coordinate axes: it checks the
stronger prerequisite that the full-body patella has an independent joint
coordinate available at a fixed knee angle under its authored equalities.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from myo_sim.build.compose import build_model
from numilab_human.myosim_visual import _visual_qpos


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Sources/myosim/myo_sim-33c89c2b.tar.gz"
CHECKOUT = ROOT / "Sources/myosim/checkout"
POSE = ROOT / "Docs/media/patellofemoral-pose-clearance-20260930/receipt.json"
EXTENSOR = ROOT / "Docs/media/patellar-extensor-pose-candidate-20260930/receipt.json"
OUTPUT = ROOT / "Docs/media/patellar-joint-pose-admission-20260930/receipt.json"
ARCHIVE_SHA256 = "280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975"
SOURCE_FILES = (
    "myo_sim/models/leg/assets/myolegs_assets.xml",
    "myo_sim/models/leg/assets/myolegs_chain.xml",
    "myo_sim/build/compose.py",
)
JOINT_KINDS = ("translation1", "translation2", "rotation1")
KNEE_ANGLES_RAD = (0.0, 0.9, 2.0)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeError("patellar joint admission: " + detail)


def source_bytes() -> tuple[dict[str, str], ET.Element]:
    require(sha(ARCHIVE.read_bytes()) == ARCHIVE_SHA256,
            "pinned MyoSim archive changed")
    hashes: dict[str, str] = {}
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for relative in SOURCE_FILES:
            members = [member for member in archive.getmembers()
                       if member.name.endswith("/" + relative) and member.isfile()]
            require(len(members) == 1, f"archive lacks unique {relative}")
            stream = archive.extractfile(members[0])
            require(stream is not None, f"archive cannot read {relative}")
            archived = stream.read()
            require((CHECKOUT / relative).read_bytes() == archived,
                    f"checkout differs from pinned archive: {relative}")
            hashes[relative] = sha(archived)
            if relative.endswith("myolegs_assets.xml"):
                equality_xml = ET.fromstring(archived)
    return hashes, equality_xml


def joint_id(model: mujoco.MjModel, name: str) -> int:
    identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    require(identifier >= 0, "missing joint " + name)
    return identifier


def polynomial(coefficients: np.ndarray, angle: float) -> float:
    return float(sum(float(coefficient) * angle**power
                     for power, coefficient in enumerate(coefficients[:5])))


def run() -> dict:
    source_hashes, equality_xml = source_bytes()
    pose = json.loads(POSE.read_text())
    extensor = json.loads(EXTENSOR.read_text())
    require(pose["status"] == "unadopted_bilateral_geometric_pose_candidate"
            and extensor["status"] == "bounded_coherent_extensor_geometry_candidate"
            and extensor["pose_receipt_sha256"] == sha(POSE.read_bytes())
            and extensor["patella_rigid_joint_pose_updated"] is False,
            "Open Knee candidate receipts are not the expected unadopted pair")

    model = build_model("myofullbody")
    qpos, pose_state = _visual_qpos(model, mujoco, False)
    require(pose_state["pose_state"] == "source_equality_projected_neutral",
            "whole-body reference pose is not equality projected")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    sides: dict[str, dict] = {}
    for side, suffix in (("left", "l"), ("right", "r")):
        body_name = "patella_" + suffix
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        require(body >= 0, "missing " + body_name)
        first = int(model.body_jntadr[body])
        actual = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
                  for index in range(first, first + int(model.body_jntnum[body]))}
        expected = {f"knee_angle_beta_{kind}_{suffix}" for kind in JOINT_KINDS}
        require(actual == expected, body_name + " gained an unaudited joint")
        knee_name = "knee_angle_" + suffix
        knee = joint_id(model, knee_name)
        require(abs(float(qpos[model.jnt_qposadr[knee]])) < 1e-12,
                "projected reference knee is not neutral")
        rows = []
        for kind in JOINT_KINDS:
            name = f"knee_angle_beta_{kind}_{suffix}"
            eq_name = f"knee_angle_beta_{kind}_constraint_{suffix}"
            joint = joint_id(model, name)
            equality = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
            require(equality >= 0, "missing equality " + eq_name)
            require(int(model.eq_type[equality]) == int(mujoco.mjtEq.mjEQ_JOINT)
                    and int(model.eq_obj1id[equality]) == joint
                    and int(model.eq_obj2id[equality]) == knee
                    and bool(model.eq_active0[equality]),
                    "inactive or redirected source equality " + eq_name)
            xml = equality_xml.find(f".//equality/joint[@name='{eq_name}']")
            require(xml is not None and xml.get("joint1") == name
                    and xml.get("joint2") == knee_name,
                    "compiled equality differs from source XML: " + eq_name)
            coefficients = np.asarray(model.eq_data[equality, :5], dtype=float)
            authored = np.asarray([float(value) for value in
                                   xml.attrib["polycoef"].split()], dtype=float)
            require(np.array_equal(coefficients, authored),
                    "compiled equality coefficients differ from source XML")
            address = int(model.jnt_qposadr[joint])
            neutral = polynomial(coefficients, 0.0)
            require(abs(float(qpos[address]) - neutral) < 1e-12,
                    "projected patella joint violates source equality")
            rows.append({
                "joint": name,
                "joint_qpos_address": address,
                "equality": eq_name,
                "equality_coefficients": authored.tolist(),
                "neutral_coordinate": neutral,
                "source_prescribed_coordinates_at_knee_angles": {
                    str(angle): polynomial(coefficients, angle)
                    for angle in KNEE_ANGLES_RAD
                },
            })
        vector = np.asarray(pose["sides"][side]["prescribed_pose_translation_m"],
                            dtype=float)
        require(np.allclose(vector, extensor["sides"][side]["pose_translation_m"],
                            rtol=0, atol=1e-15)
                and abs(float(np.linalg.norm(vector)) - 20e-6) < 1e-12,
                "clearance vector drifted")
        sides[side] = {
            "source_body": body_name,
            "source_knee_joint": knee_name,
            "projected_neutral_body_center_world_m": np.asarray(data.xpos[body]).tolist(),
            "authored_body_joints": rows,
            "open_knee_candidate_translation_m": vector.tolist(),
            "open_knee_candidate_magnitude_m": float(np.linalg.norm(vector)),
            "independent_exact_source_joint_coordinates_at_fixed_knee_angle": 0,
            "whole_body_pose_admitted": False,
        }
    return {
        "schema": "numi.human.patellar-joint-pose-admission.v1",
        "status": "blocked_source_joint_admission",
        "source_archive_sha256": ARCHIVE_SHA256,
        "source_file_sha256": source_hashes,
        "source_model": "myofullbody",
        "mujoco_version": mujoco.__version__,
        "projected_pose": pose_state,
        "open_knee_pose_receipt_sha256": sha(POSE.read_bytes()),
        "open_knee_extensor_receipt_sha256": sha(EXTENSOR.read_bytes()),
        "sides": sides,
        "boundary": (
            "Both source patellar bodies have exactly three joints, and each is "
            "prescribed by an active knee-angle equality. At a fixed knee angle, "
            "the exact authored coordinate law has no independent patellar pose "
            "degree of freedom. MuJoCo equality constraints have solver compliance; "
            "this is a source-law admission check, not a claim of infinite stiffness. "
            "Open Knee mesh vectors and MyoSim world axes are different source frames "
            "and were not mapped here. The geometric candidate remains unadopted; "
            "no native coupled contact/load or anatomical calibration is qualified."
        ),
    }


if __name__ == "__main__":
    receipt = run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": receipt["status"],
                      "sides": {side: row["whole_body_pose_admitted"]
                                for side, row in receipt["sides"].items()},
                      "receipt": str(OUTPUT.relative_to(ROOT))}, sort_keys=True))
