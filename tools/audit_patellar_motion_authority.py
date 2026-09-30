"""Bind the Open Knee and MyoSim patellar motion contracts to pinned sources.

FEBio prescribes tibiofemoral flexion but leaves the three patellar joints
free. MyoSim instead prescribes three patellar coordinates from knee angle.
This conflict matters before assigning an anatomy claim to live flexion.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from myo_sim.build.compose import build_model
from numilab_human.open_knee import EXPECTED_HASHES
from tools.audit_patellar_joint_pose_admission import (
    ARCHIVE_SHA256, joint_id, source_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
FEBIO = ROOT / "Sources/open-knee-oks003/FeBio_custom.feb"
JOINT_ADMISSION = ROOT / "Docs/media/patellar-joint-pose-admission-20260930/receipt.json"
OUTPUT = ROOT / "Docs/media/patellar-motion-authority-20260930/receipt.json"
PATELLAR_CONSTRAINTS = (
    "Patellar_Extension_Flexion",
    "Patellar_Lateral_Tilt",
    "Patellar_Lateral_Rotation",
)
PATELLAR_JOINT_KINDS = ("translation1", "translation2", "rotation1")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("patellar motion authority: " + message)


def run() -> dict:
    febio_bytes = FEBIO.read_bytes()
    require(sha(febio_bytes) == EXPECTED_HASHES["FeBio_custom.feb"],
            "pinned Open Knee FEBio source changed")
    root = ET.fromstring(febio_bytes)
    source_hashes, myosim_xml = source_bytes()
    admission_bytes = JOINT_ADMISSION.read_bytes()
    admission = json.loads(admission_bytes)
    require(admission["status"] == "blocked_source_joint_admission"
            and admission["source_archive_sha256"] == ARCHIVE_SHA256,
            "existing MyoSim source-law admission changed")
    constraints = root.findall(".//Step/Constraints/constraint")
    named = [item for item in constraints if item.get("name") is not None]
    by_name = {item.get("name"): item for item in named}
    require(len(by_name) == len(named) and
            all(name in by_name for name in PATELLAR_CONSTRAINTS) and
            "Extension_Flexion" in by_name,
            "source rigid-joint constraints changed")
    knee = by_name["Extension_Flexion"]
    require(knee.get("type") == "rigid cylindrical joint" and
            knee.findtext("prescribed_rotation") == "1" and
            knee.findtext("prescribed_translation") == "0" and
            knee.find("rotation").get("lc") == "9",
            "source tibiofemoral flexion authority changed")
    patellar = []
    for name in PATELLAR_CONSTRAINTS:
        item = by_name[name]
        require(item.get("type") == "rigid cylindrical joint" and
                item.findtext("prescribed_rotation") == "0" and
                item.findtext("prescribed_translation") == "0" and
                item.find("rotation").get("lc") is None and
                item.find("translation").get("lc") is None,
                f"source patellar joint became prescribed: {name}")
        patellar.append({
            "name": name,
            "body_a_material_id": int(item.findtext("body_a")),
            "body_b_material_id": int(item.findtext("body_b")),
            "joint_origin_source_mm": [float(value) for value in
                                        item.findtext("joint_origin").split(",")],
            "joint_axis_source": [float(value) for value in
                                  item.findtext("joint_axis").split(",")],
            "rotation_prescribed": False,
            "translation_prescribed": False,
        })
    rigid_ties = {item.get("name"): item for item in
                  root.findall(".//Boundary/rigid")}
    for name in ("PTC_With_PTB", "QAT_With_PTB", "PTL_With_PTB"):
        require(name in rigid_ties and rigid_ties[name].get("rb") == "1",
                f"source patellar bone tie changed: {name}")
    model = build_model("myofullbody")
    require(mujoco.__version__ == "3.12.0", "MuJoCo version changed")
    sides = {}
    for side, suffix in (("left", "l"), ("right", "r")):
        knee_joint = joint_id(model, f"knee_angle_{suffix}")
        joint_rows = []
        for kind in PATELLAR_JOINT_KINDS:
            name = f"knee_angle_beta_{kind}_{suffix}"
            equality_name = f"knee_angle_beta_{kind}_constraint_{suffix}"
            dependent = joint_id(model, name)
            equality = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name)
            xml = myosim_xml.find(
                f".//equality/joint[@name='{equality_name}']")
            require(equality >= 0 and xml is not None and
                    bool(model.eq_active0[equality]) and
                    int(model.eq_type[equality]) == int(mujoco.mjtEq.mjEQ_JOINT) and
                    int(model.eq_obj1id[equality]) == dependent and
                    int(model.eq_obj2id[equality]) == knee_joint and
                    xml.get("joint1") == name and
                    xml.get("joint2") == f"knee_angle_{suffix}",
                    f"{side} active source patellar equality changed: {kind}")
            coefficients = np.asarray(model.eq_data[equality, :5], dtype=float)
            authored = np.asarray(
                [float(value) for value in xml.get("polycoef").split()],
                dtype=float)
            require(np.array_equal(coefficients, authored),
                    f"{side} source patellar polynomial changed: {kind}")
            joint_rows.append({
                "joint": name,
                "equality": equality_name,
                "dependent_q_index": int(model.jnt_qposadr[dependent]),
                "master_knee_q_index": int(model.jnt_qposadr[knee_joint]),
                "polycoef": coefficients.tolist(),
                "active": True,
            })
        require(len(joint_rows) == 3 and
                admission["sides"][side]
                ["independent_exact_source_joint_coordinates_at_fixed_knee_angle"] == 0,
                f"{side} prior admission disagrees with source equalities")
        sides[side] = {"myosim_patellar_equalities": joint_rows,
                       "open_knee_patellar_motion_prescribed": False,
                       "motion_authority_compatible": False}
    return {
        "schema": "numi.human.patellar-motion-authority.v1",
        "status": "open_knee_free_patella_conflicts_with_myosim_prescription",
        "open_knee_febio_sha256": sha(febio_bytes),
        "myosim_archive_sha256": ARCHIVE_SHA256,
        "myosim_source_file_sha256": source_hashes,
        "prior_joint_admission_receipt_sha256": sha(admission_bytes),
        "open_knee_tibiofemoral_flexion": {
            "constraint": "Extension_Flexion",
            "rotation_prescribed": True,
            "translation_prescribed": False,
            "loadcurve_id": 9,
        },
        "open_knee_patellar_constraints": patellar,
        "open_knee_patellar_bone_rigid_ties":
            sorted(name for name in rigid_ties if name in
                   ("PTC_With_PTB", "QAT_With_PTB", "PTL_With_PTB")),
        "sides": sides,
        "live_free_patella_mode_qualified": False,
        "boundary": (
            "The pinned FEBio model supplies free patellar cylindrical-joint "
            "motion under contact and tendon/ligament loads, not a measured "
            "patellar trajectory. MyoSim instead prescribes three patellar "
            "coordinates from knee angle on each side. This source-contract "
            "audit does not qualify an alternative joint law, a released "
            "native patella, loaded cartilage, or clinical anatomy."
        ),
    }


if __name__ == "__main__":
    result = run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "receipt": str(OUTPUT),
                      "open_knee_free_patellar_constraints": len(
                          result["open_knee_patellar_constraints"]),
                      "myosim_active_patellar_equalities_per_side": 3},
                     sort_keys=True))
