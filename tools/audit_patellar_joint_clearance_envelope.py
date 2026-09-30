"""Scan source patellar translation1 around flexed Open Knee cartilage poses.

Any nonzero perturbation violates the authored MyoSim equality. This is a
diagnostic of the clearance/kinematic mismatch, not a proposed joint law.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from myo_sim.build.compose import build_model
from numilab_human.open_knee import parse_source
from numilab_human.upper_limb_pose_audit import _pose_qpos
from tools.audit_patellar_cartilage_live_pose_transport import (
    body_pose, local_set, place, pt_l_geodesic_weights, signed_six,
    visual_locals,
)
from tools.export_patellofemoral_matter_input import HEADER
from tools.verify_patellofemoral_intersections import _compiled_meshes, _intersections
from tools.verify_patellofemoral_surface import payload


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Sources/open-knee-oks003"
GEOMETRY = ROOT / "Docs/media/patellar-cartilage-tie-clearance-20260930/receipt.json"
FRAME = ROOT / "Docs/media/patellar-cartilage-live-pose-transport-20260930/receipt.json"
PROJECTED = ROOT / "Build/open-knee-projected-visual-frame-20260930"
OUTPUT = ROOT / "Docs/media/patellar-joint-clearance-envelope-20260930/receipt.json"
ANGLES_RADIANS = (0.1, 0.9)
TRANSLATION1_OFFSETS_METERS = (-0.008, -0.006, -0.004, -0.002,
                              -0.001, -0.0005, 0.0, 0.0005, 0.001)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("patellar joint clearance envelope: " + message)


def run() -> dict:
    source = parse_source(SOURCE)
    geometry_bytes = GEOMETRY.read_bytes()
    frame_bytes = FRAME.read_bytes()
    geometry = json.loads(geometry_bytes)
    frame = json.loads(frame_bytes)
    require(geometry["status"] == "bilateral_tie_fixed_cartilage_geometry_candidate"
            and frame["status"] ==
            "projected_payload_matches_live_reference_but_flexion_crosses"
            and frame["geometry_receipt_sha256"] == sha(geometry_bytes),
            "source-bound prerequisite receipts changed")
    model = build_model("myofullbody")
    require(mujoco.__version__ == "3.12.0", "MuJoCo version changed")
    data = mujoco.MjData(model)
    neutral_q, _, _ = _pose_qpos(model, (), mujoco, np)
    data.qpos[:] = neutral_q
    mujoco.mj_forward(model, data)
    sides = {}
    for side, suffix in (("left", "l"), ("right", "r")):
        stem = ("open-knee-oks003-left" if side == "left"
                else "open-knee-oks003-right-mirrored")
        path = PROJECTED / side / f"{stem}.nhknee"
        manifest_bytes = path.with_suffix(".manifest.json").read_bytes()
        require(sha(path.read_bytes()) ==
                frame["sides"][side]["projected_nhknee_payload_sha256"] and
                sha(manifest_bytes) ==
                frame["sides"][side]["projected_nhknee_manifest_sha256"],
                f"{side} projected frame candidate changed")
        manifest = json.loads(manifest_bytes)
        decoded = payload(path, manifest, source)
        meshes = _compiled_meshes(decoded)
        locals_ = visual_locals(decoded)
        patella_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"patella_{suffix}")
        femur_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"femur_{suffix}")
        tibia_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"tibia_{suffix}")
        knee_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"knee_angle_{suffix}")
        translation_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT,
            f"knee_angle_beta_translation1_{suffix}")
        require(min(patella_body, femur_body, tibia_body,
                    knee_joint, translation_joint) >= 0,
                f"{side} source body or joint missing")
        ptc_row = decoded["regions"]["PTC"]
        ptc_first, ptc_count = ptc_row["first_node"], ptc_row["node_count"]
        ptc_rest = decoded["positions"][ptc_first:ptc_first + ptc_count]
        candidate_row = geometry["sides"][side]["diagnostic_native_input"]
        candidate_bytes = (ROOT / candidate_row["path"]).read_bytes()
        require(sha(candidate_bytes) == candidate_row["sha256"],
                f"{side} PTC clearance field changed")
        ptc_current = np.frombuffer(
            candidate_bytes, dtype="<f4", offset=HEADER.size + 12 * ptc_count,
            count=3 * ptc_count).reshape(-1, 3).astype(np.float64)
        neutral_patella_pose = body_pose(data, patella_body)
        neutral_tibia_pose = body_pose(data, tibia_body)
        ptc_local = (np.asarray(locals_[ptc_first:ptc_first + ptc_count],
                                dtype=np.float64) +
                     (ptc_current - ptc_rest) @ neutral_patella_pose[1])
        fmc_row = decoded["regions"]["FMC"]
        fmc_first, fmc_count = fmc_row["first_node"], fmc_row["node_count"]
        fmc_local = np.asarray(locals_[fmc_first:fmc_first + fmc_count],
                               dtype=np.float64)
        ptb_row = decoded["regions"]["PTB"]
        ptb_first, ptb_count = ptb_row["first_node"], ptb_row["node_count"]
        ptb_local = np.asarray(locals_[ptb_first:ptb_first + ptb_count],
                               dtype=np.float64)
        pt_l_row = decoded["regions"]["PTL"]
        pt_l_first, pt_l_count = pt_l_row["first_node"], pt_l_row["node_count"]
        pt_l_rest = np.asarray(decoded["positions"][
            pt_l_first:pt_l_first + pt_l_count], dtype=np.float64)
        pt_l_cells = np.asarray(decoded["tetrahedra"][
            pt_l_row["first_tet"]:pt_l_row["first_tet"] +
            pt_l_row["tet_count"]], dtype=np.int64) - pt_l_first
        pt_l_patellar = local_set(source, "PTL", "PTL_@_PTB_TiesNodes")
        pt_l_tibial = local_set(source, "PTL", "PTL_@_TBB_TiesNodes")
        pt_l_weight = pt_l_geodesic_weights(
            pt_l_rest, pt_l_cells, pt_l_patellar, pt_l_tibial)
        pt_l_reference_six = signed_six(pt_l_rest, pt_l_cells)
        require(np.all(pt_l_reference_six > 0),
                f"{side} PTL reference tetrahedra are not positive")
        pt_l_patellar_local = ((pt_l_rest - neutral_patella_pose[0]) @
                               neutral_patella_pose[1])
        pt_l_tibial_local = ((pt_l_rest - neutral_tibia_pose[0]) @
                            neutral_tibia_pose[1])
        knee_origin_body = np.asarray(
            manifest["registration"]["target_knee_origin_femur_body_m"],
            dtype=np.float64)
        rows = {}
        for angle in ANGLES_RADIANS:
            qpos, _, _ = _pose_qpos(
                model, ((int(model.jnt_qposadr[knee_joint]), angle),),
                mujoco, np)
            source_coordinate = float(qpos[model.jnt_qposadr[translation_joint]])
            samples = []
            for offset in TRANSLATION1_OFFSETS_METERS:
                candidate_coordinate = source_coordinate + offset
                lower, upper = model.jnt_range[translation_joint]
                if not lower <= candidate_coordinate <= upper:
                    continue
                qpos[model.jnt_qposadr[translation_joint]] = candidate_coordinate
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                patella_pose = body_pose(data, patella_body)
                femur_pose = body_pose(data, femur_body)
                ptc = place(ptc_local, patella_pose).astype("<f4").astype(np.float64)
                fmc = place(fmc_local, femur_pose).astype("<f4").astype(np.float64)
                contact = _intersections({
                    "PTC": (ptc, meshes["PTC"][1]),
                    "FMC": (fmc, meshes["FMC"][1]),
                })
                ptb = place(ptb_local, patella_pose)
                femur_origin = (np.asarray(data.xmat[femur_body]).reshape(3, 3) @
                                knee_origin_body + data.xpos[femur_body])
                anterior_offset = float(femur_origin[1] - ptb[:, 1].mean())
                pt_l_patellar_world = place(pt_l_patellar_local, patella_pose)
                pt_l_tibial_world = place(
                    pt_l_tibial_local, body_pose(data, tibia_body))
                mapped_pt_l = (
                    pt_l_weight[:, None] * pt_l_patellar_world +
                    (1.0 - pt_l_weight[:, None]) * pt_l_tibial_world)
                mapped_pt_l[pt_l_patellar] = pt_l_patellar_world[pt_l_patellar]
                mapped_pt_l[pt_l_tibial] = pt_l_tibial_world[pt_l_tibial]
                pt_l_jacobian = (signed_six(mapped_pt_l, pt_l_cells) /
                                 pt_l_reference_six)
                samples.append({
                    "translation1_offset_from_source_m": offset,
                    "translation1_coordinate_m": candidate_coordinate,
                    "violates_source_equality": offset != 0.0,
                    "patellar_bone_anterior_offset_from_femoral_knee_origin_m":
                        anterior_offset,
                    "ptl_continuum_map_preflight": {
                        "minimum_jacobian": float(pt_l_jacobian.min()),
                        "maximum_jacobian": float(pt_l_jacobian.max()),
                        "tetrahedra_below_native_0_05_gate":
                            int(np.count_nonzero(pt_l_jacobian < 0.05)),
                        "tetrahedra_above_native_20_gate":
                            int(np.count_nonzero(pt_l_jacobian > 20.0)),
                    },
                    "articular_intersections": {
                        "exact_crossing_pairs":
                            contact["exact_segment_or_polygon_crossing_pairs"],
                        "exact_point_pairs": contact["exact_point_contact_pairs"],
                        "maximum_intersection_segment_m":
                            contact["maximum_intersection_segment_m"],
                    },
                })
            baseline = next((item for item in samples
                             if item["translation1_offset_from_source_m"] == 0.0),
                            None)
            require(baseline is not None and
                    baseline["articular_intersections"]["exact_crossing_pairs"] ==
                    frame["sides"][side]["poses"][str(angle)]
                    ["projected_payload_candidate_articular_intersections"]
                    ["exact_segment_or_polygon_crossing_pairs"],
                    f"{side} {angle} source equality pose no longer matches frame audit")
            rows[str(angle)] = {
                "source_translation1_coordinate_m": source_coordinate,
                "source_translation1_range_m":
                    model.jnt_range[translation_joint].tolist(),
                "samples": samples,
            }
        sides[side] = rows
    return {
        "schema": "numi.human.patellar-joint-clearance-envelope-diagnostic.v1",
        "status": "source_joint_perturbation_diagnostic_only",
        "geometry_receipt_sha256": sha(geometry_bytes),
        "pose_transport_receipt_sha256": sha(frame_bytes),
        "mujoco_version": mujoco.__version__,
        "sides": sides,
        "joint_law_adopted": False,
        "boundary": (
            "Each nonzero translation1 offset violates the active source "
            "MyoSim knee-to-patella equality. Exact PTC/FMC surface crossing "
            "counts, anterior offsets and Python PTL map ratios are "
            "geometric diagnostics only; zero crossings would not qualify "
            "native tendon dynamics, loaded contact, anatomy, or a new "
            "source joint law."
        ),
    }


if __name__ == "__main__":
    result = run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "receipt": str(OUTPUT),
                      "crossings": {side: {angle: [
                          (row["translation1_offset_from_source_m"],
                           row["articular_intersections"]["exact_crossing_pairs"])
                          for row in entry["samples"]]
                          for angle, entry in values.items()}
                          for side, values in result["sides"].items()}},
                     sort_keys=True))
