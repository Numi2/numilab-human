"""Measure the unadopted PTC clearance field in live MyoSim knee poses.

NHKNEE1 restWorld is compiled at source qpos0. Its visualLocal coordinates
use the source inertial body frame, so a projected or flexed Human pose must
move PTC with patella and FMC with femur before testing their separation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from myo_sim.build.compose import build_model
from numilab_human.myosim_visual import _visual_qpos
from numilab_human.open_knee import (
    HEADER_STRUCT, NODE_STRUCT, NODE_SET_STRUCT, REGION_STRUCT,
    SURFACE_PAIR_STRUCT, SURFACE_STRUCT, VISUAL_BODY_ROLE, parse_source,
)
from numilab_human.upper_limb_pose_audit import _pose_qpos
from tools.audit_patellar_cartilage_tie_clearance import source_indices
from tools.export_patellofemoral_matter_input import HEADER
from tools.verify_patellofemoral_intersections import (
    _compiled_meshes, _intersections,
)
from tools.verify_patellofemoral_surface import payload


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Sources/open-knee-oks003"
GEOMETRY = ROOT / "Docs/media/patellar-cartilage-tie-clearance-20260930/receipt.json"
OUTPUT = ROOT / "Docs/media/patellar-cartilage-live-pose-transport-20260930/receipt.json"
COMPILED = ROOT / "Build/patellofemoral-surface-20260930"
PROJECTED = ROOT / "Build/open-knee-projected-visual-frame-20260930"
FLEXION_RADIANS = (0.0, 0.1, 0.9)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("patellar cartilage live pose transport: " + message)


def node_offset(decoded: dict) -> int:
    raw = decoded["raw"]
    (_, _, header_bytes, region_count, _, _, surface_count, _,
     node_set_count, _, pair_count, _, _, _) = HEADER_STRUCT.unpack_from(raw)
    offset = (header_bytes + region_count * REGION_STRUCT.size +
              surface_count * SURFACE_STRUCT.size +
              node_set_count * NODE_SET_STRUCT.size +
              pair_count * SURFACE_PAIR_STRUCT.size)
    return offset


def visual_locals(decoded: dict) -> np.ndarray:
    return np.ndarray(decoded["positions"].shape, dtype="<f4",
                      buffer=decoded["raw"], offset=node_offset(decoded) + 16,
                      strides=(NODE_STRUCT.size, 4))


def body_pose(data: mujoco.MjData, body: int) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(data.xipos[body], dtype=np.float64).copy(),
            np.asarray(data.ximat[body], dtype=np.float64).reshape(3, 3).copy())


def place(locals_: np.ndarray, pose: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    position, rotation = pose
    return locals_ @ rotation.T + position


def maximum_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - b.astype(np.float64),
                                axis=1).max())


def signed_six(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    a, b, c, d = (points[cells[:, index]] for index in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)


def local_set(source, region_name: str, set_name: str) -> np.ndarray:
    ids = {identifier: index for index, identifier in
           enumerate(source.regions[region_name].node_ids)}
    return np.asarray([ids[identifier] for identifier in source.node_sets[set_name]],
                      dtype=np.int64)


def pt_l_geodesic_weights(points: np.ndarray, cells: np.ndarray,
                          patellar_tie: np.ndarray,
                          tibial_tie: np.ndarray) -> np.ndarray:
    edges = np.concatenate([cells[:, [a, b]] for a, b in
                            ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))])
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    require(np.all(lengths > 0), "PTL graph contains a zero-length edge")
    row = np.concatenate((edges[:, 0], edges[:, 1]))
    col = np.concatenate((edges[:, 1], edges[:, 0]))
    graph = coo_matrix((np.tile(lengths, 2), (row, col)),
                       shape=(len(points), len(points))).tocsr()
    patellar = dijkstra(graph, directed=False, indices=patellar_tie,
                        min_only=True)
    tibial = dijkstra(graph, directed=False, indices=tibial_tie,
                      min_only=True)
    require(np.all(np.isfinite(patellar)) and np.all(np.isfinite(tibial)) and
            not np.intersect1d(patellar_tie, tibial_tie).size,
            "PTL tie sets are disconnected or overlap")
    a = np.maximum(patellar, 1e-12) ** -2
    b = np.maximum(tibial, 1e-12) ** -2
    weight = a / (a + b)
    weight[patellar_tie] = 1.0
    weight[tibial_tie] = 0.0
    return weight


def run() -> dict:
    geometry_bytes = GEOMETRY.read_bytes()
    geometry = json.loads(geometry_bytes)
    require(geometry["status"] == "bilateral_tie_fixed_cartilage_geometry_candidate",
            "unadopted geometry identity changed")
    source = parse_source(SOURCE)
    model = build_model("myofullbody")
    require(mujoco.__version__ == "3.12.0", "MuJoCo version changed")
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    default_poses = {}
    for side, suffix in (("left", "l"), ("right", "r")):
        for name in ("femur", "patella"):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     f"{name}_{suffix}")
            require(body >= 0, f"missing source body {name}_{suffix}")
            default_poses[(side, name)] = body_pose(data, body)

    sides = {}
    for side, suffix in (("left", "l"), ("right", "r")):
        stem = ("open-knee-oks003-left" if side == "left"
                else "open-knee-oks003-right-mirrored")
        path = COMPILED / side / f"{stem}.nhknee"
        manifest_path = path.with_suffix(".manifest.json")
        manifest_bytes = manifest_path.read_bytes()
        require(sha(path.read_bytes()) == geometry["sides"][side]["compiled_payload_sha256"]
                and sha(manifest_bytes) == geometry["sides"][side]["compiled_manifest_sha256"],
                f"{side} NHKNEE1 identity changed")
        decoded = payload(path, json.loads(manifest_bytes), source)
        projected_path = PROJECTED / side / f"{stem}.nhknee"
        projected_manifest_bytes = projected_path.with_suffix(
            ".manifest.json").read_bytes()
        projected_manifest = json.loads(projected_manifest_bytes)
        require(projected_manifest["status"] ==
                "equality_projected_visual_frame_candidate" and
                projected_manifest["visual_reference_frame"]["pose"] ==
                "source_equality_projected_neutral",
                f"{side} projected frame candidate identity changed")
        projected_decoded = payload(projected_path, projected_manifest, source)
        require(len(decoded["raw"]) == len(projected_decoded["raw"]) and
                node_offset(decoded) == node_offset(projected_decoded),
                f"{side} projected payload layout changed")
        different = np.flatnonzero(
            np.frombuffer(decoded["raw"], dtype=np.uint8) !=
            np.frombuffer(projected_decoded["raw"], dtype=np.uint8))
        local_byte = different - node_offset(decoded)
        require(len(different) > 0 and
                np.all((local_byte >= 0) &
                       (local_byte < len(decoded["positions"]) * NODE_STRUCT.size)) and
                np.all(((local_byte % NODE_STRUCT.size >= 16) &
                        (local_byte % NODE_STRUCT.size < 28)) |
                       ((local_byte % NODE_STRUCT.size >= 32) &
                        (local_byte % NODE_STRUCT.size < 44))),
                f"{side} projected frame changed bytes outside node locals")
        neutral_q, neutral_state = _visual_qpos(model, mujoco, False)
        require(neutral_state["pose_state"] == "source_equality_projected_neutral",
                "neutral pose is not equality projected")
        data.qpos[:] = neutral_q
        mujoco.mj_forward(model, data)
        projected_reference_poses = {}
        for owner in ("femur", "patella", "tibia"):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     f"{owner}_{suffix}")
            projected_reference_poses[owner] = body_pose(data, body)
        projected_locals_all = visual_locals(projected_decoded)
        projected_visual_errors = {}
        for name, row in decoded["regions"].items():
            sl = slice(row["first_node"], row["first_node"] + row["node_count"])
            owner = VISUAL_BODY_ROLE[name]
            projected_visual_errors[name] = maximum_distance(
                place(np.asarray(projected_locals_all[sl], dtype=np.float64),
                      projected_reference_poses[owner]),
                decoded["positions"][sl])
        require(max(projected_visual_errors.values()) < 2e-6,
                f"{side} projected visual frame misses source restWorld")
        locals_all = visual_locals(decoded)
        articular = _compiled_meshes(decoded)
        regions = {}
        for name, owner in (("PTC", "patella"), ("FMC", "femur"),
                            ("PTL", "patella")):
            row = decoded["regions"][name]
            sl = slice(row["first_node"], row["first_node"] + row["node_count"])
            rest = np.asarray(decoded["positions"][sl], dtype=np.float64)
            local = np.asarray(locals_all[sl], dtype=np.float64)
            require(maximum_distance(place(local, default_poses[(side, owner)]), rest)
                    < 2e-6, f"{side} {name} source default body frame drifted")
            regions[name] = {"rest": rest, "local": local, "owner": owner}
        candidate_row = geometry["sides"][side]["diagnostic_native_input"]
        candidate_path = ROOT / candidate_row["path"]
        candidate_bytes = candidate_path.read_bytes()
        require(sha(candidate_bytes) == candidate_row["sha256"],
                f"{side} cartilage candidate identity changed")
        ptc = regions["PTC"]
        current = np.frombuffer(candidate_bytes, dtype="<f4",
                                offset=HEADER.size + 12 * len(ptc["rest"]),
                                count=3 * len(ptc["rest"])).reshape(-1, 3).astype(np.float64)
        require(sha(current.astype("<f4").tobytes()) ==
                candidate_row["candidate_current_ptc_positions_sha256"],
                f"{side} cartilage current block changed")
        tie = source_indices(source, "PTC_@_PTB_TiesNodes")
        require(len(tie) == candidate_row["fixed_node_count"] and
                np.array_equal(current[tie], ptc["rest"][tie]),
                f"{side} PTC/PTB fixed tie changed")
        ptc["current_local"] = (ptc["local"] +
                                (current - ptc["rest"]) @
                                default_poses[(side, "patella")][1])
        knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                 f"knee_angle_{suffix}")
        require(knee >= 0, f"{side} source knee joint missing")
        knee_q = int(model.jnt_qposadr[knee])
        pt_l = regions["PTL"]
        pt_l_row = decoded["regions"]["PTL"]
        pt_l_cells = np.asarray(decoded["tetrahedra"][
            pt_l_row["first_tet"]:pt_l_row["first_tet"] +
            pt_l_row["tet_count"]], dtype=np.int64) - pt_l_row["first_node"]
        pt_l_patellar = local_set(source, "PTL", "PTL_@_PTB_TiesNodes")
        pt_l_tibial = local_set(source, "PTL", "PTL_@_TBB_TiesNodes")
        pt_l_weight = pt_l_geodesic_weights(
            pt_l["rest"], pt_l_cells, pt_l_patellar, pt_l_tibial)
        pt_l_reference_six = signed_six(pt_l["rest"], pt_l_cells)
        require(np.all(pt_l_reference_six > 0),
                f"{side} source PTL tetrahedra are not positive")
        pose_rows = {}
        for flexion in FLEXION_RADIANS:
            qpos, equality_count, correction = _pose_qpos(
                model, ((knee_q, flexion),), mujoco, np)
            if flexion == 0.0:
                neutral_q, neutral_state = _visual_qpos(model, mujoco, False)
                require(np.array_equal(qpos, neutral_q) and
                        neutral_state["pose_state"] == "source_equality_projected_neutral",
                        "source neutral projection changed")
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            live = {}
            for name in ("PTC", "FMC"):
                owner = regions[name]["owner"]
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                         f"{owner}_{suffix}")
                local = (regions[name]["current_local"] if name == "PTC"
                         else regions[name]["local"])
                live[name] = place(local, body_pose(data, body)).astype("<f4")
            meshes = {name: (live[name].astype(np.float64), articular[name][1])
                      for name in ("PTC", "FMC")}
            contacts = _intersections(meshes)
            pt_l_transforms = {}
            for owner in ("patella", "tibia"):
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                         f"{owner}_{suffix}")
                require(body >= 0, f"{side} source {owner} body missing")
                default_pose = default_poses.get((side, owner))
                if default_pose is None:
                    # The tibial reference was not needed by PTC/FMC above.
                    data_default = mujoco.MjData(model)
                    data_default.qpos[:] = model.qpos0
                    mujoco.mj_forward(model, data_default)
                    default_pose = body_pose(data_default, body)
                    default_poses[(side, owner)] = default_pose
                local = (pt_l["rest"] - default_pose[0]) @ default_pose[1]
                pt_l_transforms[owner] = place(local, body_pose(data, body))
            mapped_pt_l = (pt_l_weight[:, None] * pt_l_transforms["patella"] +
                           (1.0 - pt_l_weight[:, None]) *
                           pt_l_transforms["tibia"])
            mapped_pt_l[pt_l_patellar] = pt_l_transforms["patella"][pt_l_patellar]
            mapped_pt_l[pt_l_tibial] = pt_l_transforms["tibia"][pt_l_tibial]
            pt_l_jacobian = signed_six(mapped_pt_l, pt_l_cells) / pt_l_reference_six
            projected_pt_l_transforms = {}
            for owner in ("patella", "tibia"):
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                         f"{owner}_{suffix}")
                projected_reference = projected_reference_poses[owner]
                local = ((pt_l["rest"] - projected_reference[0]) @
                         projected_reference[1])
                projected_pt_l_transforms[owner] = place(
                    local, body_pose(data, body))
            projected_pt_l = (
                pt_l_weight[:, None] * projected_pt_l_transforms["patella"] +
                (1.0 - pt_l_weight[:, None]) *
                projected_pt_l_transforms["tibia"])
            projected_pt_l[pt_l_patellar] = projected_pt_l_transforms[
                "patella"][pt_l_patellar]
            projected_pt_l[pt_l_tibial] = projected_pt_l_transforms[
                "tibia"][pt_l_tibial]
            projected_pt_l_jacobian = (
                signed_six(projected_pt_l, pt_l_cells) / pt_l_reference_six)
            projected_candidate_local = (
                np.asarray(projected_locals_all[
                    decoded["regions"]["PTC"]["first_node"]:
                    decoded["regions"]["PTC"]["first_node"] +
                    decoded["regions"]["PTC"]["node_count"]], dtype=np.float64) +
                (current - ptc["rest"]) @
                projected_reference_poses["patella"][1])
            patella_body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"patella_{suffix}")
            projected_candidate_live = place(
                projected_candidate_local, body_pose(data, patella_body))
            femur_body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"femur_{suffix}")
            fmc_row = decoded["regions"]["FMC"]
            projected_fmc_live = place(
                np.asarray(projected_locals_all[
                    fmc_row["first_node"]:fmc_row["first_node"] +
                    fmc_row["node_count"]], dtype=np.float64),
                body_pose(data, femur_body))
            projected_contacts = _intersections({
                "PTC": (projected_candidate_live.astype("<f4").astype(np.float64),
                        articular["PTC"][1]),
                "FMC": (projected_fmc_live.astype("<f4").astype(np.float64),
                        articular["FMC"][1]),
            })
            # The live runner reconstructs non-FEM visuals from restWorld in
            # the projected frame. Measure the correction it must apply to
            # the old payload-local visual, not an error in that live visual.
            old_payload_visual_correction = None
            if flexion == 0.0:
                old_payload_visual_correction = {
                    name: maximum_distance(regions[name]["rest"], live[name])
                    for name in ("PTC", "FMC")
                }
            pose_rows[str(flexion)] = {
                "source_joint_equalities_projected": equality_count,
                "maximum_source_joint_correction": correction,
                "ptc_bone_tie_maximum_current_displacement_from_visual_m":
                    maximum_distance(
                        place(ptc["local"][tie],
                              body_pose(data, mujoco.mj_name2id(
                                  model, mujoco.mjtObj.mjOBJ_BODY,
                                  f"patella_{suffix}"))).astype("<f4"),
                        live["PTC"][tie]),
                "default_frame_payload_visual_articular_intersections": contacts,
                "hypothetical_default_reference_to_live_ptl_map": {
                    "tetrahedra": len(pt_l_cells),
                    "minimum_jacobian": float(pt_l_jacobian.min()),
                    "maximum_jacobian": float(pt_l_jacobian.max()),
                    "tetrahedra_below_native_0_05_gate":
                        int(np.count_nonzero(pt_l_jacobian < 0.05)),
                    "tetrahedra_above_native_20_gate":
                        int(np.count_nonzero(pt_l_jacobian > 20.0)),
                    "maximum_node_displacement_m":
                        maximum_distance(mapped_pt_l, pt_l["rest"]),
                },
                "projected_reference_to_live_ptl_continuum_map": {
                    "minimum_jacobian": float(projected_pt_l_jacobian.min()),
                    "maximum_jacobian": float(projected_pt_l_jacobian.max()),
                    "tetrahedra_below_native_0_05_gate":
                        int(np.count_nonzero(projected_pt_l_jacobian < 0.05)),
                    "tetrahedra_above_native_20_gate":
                        int(np.count_nonzero(projected_pt_l_jacobian > 20.0)),
                    "maximum_node_displacement_m":
                        maximum_distance(projected_pt_l, pt_l["rest"]),
                },
                "projected_vs_default_payload_visual_ptc_difference_m":
                    maximum_distance(projected_candidate_live, live["PTC"]),
                "projected_payload_candidate_articular_intersections":
                    projected_contacts,
                "projected_payload_candidate_neutral_current_error_m":
                    maximum_distance(projected_candidate_live, current)
                    if flexion == 0.0 else None,
                "old_payload_visual_correction_to_live_rest_m":
                    old_payload_visual_correction,
            }
        sides[side] = {
            "nhknee_payload_sha256": sha(path.read_bytes()),
            "projected_nhknee_payload_sha256": sha(projected_path.read_bytes()),
            "projected_nhknee_manifest_sha256": sha(projected_manifest_bytes),
            "projected_candidate_changed_node_local_bytes": len(different),
            "projected_visual_rest_maximum_error_m":
                max(projected_visual_errors.values()),
            "projected_visual_rest_error_by_region_m": projected_visual_errors,
            "candidate_nhcar_sha256": sha(candidate_bytes),
            "ptc_ptb_tie_nodes": len(tie),
            "poses": pose_rows,
            "projected_frame_candidate_flexion_admitted": all(
                pose_rows[str(angle)]["projected_payload_candidate_articular_intersections"]
                ["static_noninterpenetration_qualified"] and
                pose_rows[str(angle)]["projected_reference_to_live_ptl_continuum_map"]
                ["tetrahedra_below_native_0_05_gate"] == 0
                for angle in FLEXION_RADIANS),
        }
    return {
        "schema": "numi.human.patellar-cartilage-live-pose-transport-preflight.v2",
        "status": "projected_payload_matches_live_reference_but_flexion_crosses",
        "geometry_receipt_sha256": sha(geometry_bytes),
        "source_model": "myofullbody",
        "mujoco_version": mujoco.__version__,
        "sides": sides,
        "whole_body_cartilage_adopted": False,
        "native_loaded_contact_qualified": False,
        "boundary": (
            "The opt-in projected visual/anchor frame makes raw NHKNEE1 "
            "visual locals match equality-projected restWorld. The existing "
            "live Human runner already reconstructs non-FEM visuals from "
            "restWorld in that frame, so the old payload correction is not "
            "a live-render error. The local PTC clearance field crosses FMC "
            "under source patellar flexion. The default-frame raw payload "
            "visual separation is not an admission result. The right knee "
            "is a mirrored left specimen. This is a geometric preflight, "
            "not a native accepted Human/Matter step or clinical claim."
        ),
    }


if __name__ == "__main__":
    result = run()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "receipt": str(OUTPUT),
                      "sides": {side: {pose: row["projected_payload_candidate_articular_intersections"]
                                       ["exact_segment_or_polygon_crossing_pairs"]
                                       for pose, row in result["sides"][side]["poses"].items()}
                                for side in result["sides"]}}, sort_keys=True))
