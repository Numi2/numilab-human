"""Export the authored MyoSim full-body model into a dependency-free IR.

This module deliberately runs in the pinned source environment rather than
adding MuJoCo or MyoSim as runtime dependencies of ``numilab-human``.  The
result contains the compiled MuJoCo model values that define the physical
model, along with a compact default-pose oracle used by the native Core probe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix_to_quaternion(matrix: object, mujoco: object) -> list[float]:
    import numpy as np

    result = np.empty(4, dtype=float)
    mujoco.mju_mat2Quat(result, np.asarray(matrix, dtype=float))
    # MuJoCo stores quaternions as wxyz; Core's payload ABI is xyzw.
    return [float(result[1]), float(result[2]), float(result[3]), float(result[0])]


def _mujoco_quaternion_to_xyzw(value: object) -> list[float]:
    values = [float(component) for component in value]
    if len(values) != 4:
        raise RuntimeError("MuJoCo quaternion did not contain four components")
    return [values[1], values[2], values[3], values[0]]


def _matrix_columns(value: object, np: object) -> object:
    """Return MuJoCo's row-major 3x3 matrix as a NumPy matrix.

    MuJoCo publishes world rotations in row-major form.  Keeping this tiny
    conversion next to the support-witness exporter prevents the contact
    artifact from relying on MJCF's convenient, but less authoritative,
    pre-compilation ``fromto``/Euler spellings.
    """
    matrix = np.asarray(value, dtype=float)
    if matrix.size != 9:
        raise RuntimeError("MuJoCo did not publish a 3x3 rotation matrix")
    return matrix.reshape(3, 3)


def _support_point_on_geometry(
    *,
    geom_type: int,
    sphere_type: int,
    capsule_type: int,
    ellipsoid_type: int,
    center: object,
    rotation: object,
    size: object,
    ground_normal: object,
    np: object,
) -> object:
    """Return the source primitive's lowest point against an authored plane.

    These are exact support witnesses for the *compiled* MuJoCo capsules and
    ellipsoids—not a BodyParts3D enclosure or a new collision approximation.
    The native runtime later treats each witness as a unilateral plane-contact
    point and is deliberately scoped to a stance/support snapshot.
    """
    direction = -ground_normal
    if geom_type == sphere_type:
        return center + direction * float(size[0])
    if geom_type == capsule_type:
        axis = rotation[:, 2]
        half_length = float(size[1])
        endpoint = axis * (half_length if float(np.dot(axis, direction)) >= 0.0 else -half_length)
        return center + endpoint + direction * float(size[0])
    if geom_type == ellipsoid_type:
        local_direction = rotation.T @ direction
        radii = np.asarray(size[:3], dtype=float)
        denominator = float(np.linalg.norm(radii * local_direction))
        if not denominator > 0.0:
            raise RuntimeError("MuJoCo support ellipsoid has a degenerate radius")
        local_support = (radii * radii * local_direction) / denominator
        return center + rotation @ local_support
    raise RuntimeError(f"MyoSim support geometry has unsupported MuJoCo type {geom_type}")


def export_fullbody(sources: Path) -> dict[str, object]:
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - executed in source environment
        raise RuntimeError(
            "MyoSim export requires the pinned myo-sim source environment: "
            "install the checked-out package and mujoco>=3.4, then pass it with --python"
        ) from error

    checkout = sources / "myosim" / "checkout"
    archive = sources / "myosim" / "myo_sim-33c89c2b.tar.gz"
    expected = checkout / "myo_sim" / "build" / "compose.py"
    if not expected.is_file():
        raise RuntimeError(f"MyoSim checkout is absent or incomplete: {expected}")
    if not archive.is_file():
        raise RuntimeError(f"MyoSim source archive is absent: {archive}")

    model = build_model("myofullbody")
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    if model.na:
        data.act[:] = 0.5
    if model.nu:
        data.ctrl[:] = 0.5
    mujoco.mj_forward(model, data)

    root_ids = [
        index for index in range(model.njnt)
        if int(model.jnt_type[index]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if len(root_ids) != 1:
        raise RuntimeError(f"myofullbody must contain exactly one free root, found {len(root_ids)}")
    root_joint = root_ids[0]
    root_body = int(model.jnt_bodyid[root_joint])
    if root_body <= 0 or int(model.body_parentid[root_body]) != 0:
        raise RuntimeError("myofullbody free root must be a direct world child")

    body_joints: dict[int, list[int]] = {index: [] for index in range(1, model.nbody)}
    joints: list[dict[str, object]] = []
    for index in range(model.njnt):
        body_id = int(model.jnt_bodyid[index])
        if body_id == 0:
            raise RuntimeError(f"joint {index} is attached to world")
        body_joints[body_id].append(index)
        joints.append(
            {
                "id": index,
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index),
                "body": body_id,
                "type": int(model.jnt_type[index]),
                "qpos_address": int(model.jnt_qposadr[index]),
                "dof_address": int(model.jnt_dofadr[index]),
                "position_body_m": [float(value) for value in model.jnt_pos[index]],
                "axis_body": [float(value) for value in model.jnt_axis[index]],
                "range": [float(value) for value in model.jnt_range[index]],
                "limited": bool(model.jnt_limited[index]),
                "armature": float(model.dof_armature[int(model.jnt_dofadr[index])]),
            }
        )

    joint_equalities: list[dict[str, object]] = []
    joint_equality_type = int(mujoco.mjtEq.mjEQ_JOINT)
    for index in range(model.neq):
        if not bool(model.eq_active0[index]):
            continue
        if int(model.eq_type[index]) != joint_equality_type:
            raise RuntimeError(
                f"myofullbody active equality {index} is not a supported joint equality"
            )
        dependent = int(model.eq_obj1id[index])
        master = int(model.eq_obj2id[index])
        if not 0 <= dependent < model.njnt or master >= model.njnt:
            raise RuntimeError(f"myofullbody joint equality {index} has invalid joints")
        dependent_q = int(model.jnt_qposadr[dependent])
        master_q = int(model.jnt_qposadr[master]) if master >= 0 else -1
        joint_equalities.append({
            "id": index,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index),
            "dependent_joint": dependent,
            "master_joint": master,
            "dependent_reference": float(model.qpos0[dependent_q]),
            "master_reference": float(model.qpos0[master_q]) if master_q >= 0 else 0.0,
            "polycoef": [float(value) for value in model.eq_data[index, :5]],
            "solref": [float(value) for value in model.eq_solref[index]],
            "solimp": [float(value) for value in model.eq_solimp[index]],
        })

    bodies: list[dict[str, object]] = []
    for index in range(1, model.nbody):
        bodies.append(
            {
                "id": index,
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index),
                "parent": int(model.body_parentid[index]),
                "mass_kg": float(model.body_mass[index]),
                "inertial_position_body_m": [float(value) for value in model.body_ipos[index]],
                "inertial_quaternion_body_xyzw": _mujoco_quaternion_to_xyzw(model.body_iquat[index]),
                "inertia_kg_m2": [float(value) for value in model.body_inertia[index]],
                "default_body_position_world_m": [float(value) for value in data.xpos[index]],
                "default_body_quaternion_world_xyzw": _matrix_to_quaternion(data.xmat[index], mujoco),
                "default_com_position_world_m": [float(value) for value in data.xipos[index]],
                "default_inertial_quaternion_world_xyzw": _matrix_to_quaternion(
                    data.ximat[index], mujoco
                ),
            }
        )

    # Full-body MyoSim's locomotor contact surface is authored as five
    # primitives per foot.  Preserve those compiled primitive witnesses rather
    # than deriving boxes from a different visual-anatomy coordinate system.
    plane_type = int(mujoco.mjtGeom.mjGEOM_PLANE)
    sphere_type = int(mujoco.mjtGeom.mjGEOM_SPHERE)
    capsule_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    ellipsoid_type = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    ground_candidates = [
        index for index in range(model.ngeom)
        if int(model.geom_type[index]) == plane_type and int(model.geom_bodyid[index]) == 0
    ]
    if len(ground_candidates) != 1:
        raise RuntimeError(
            "myofullbody must compile exactly one world-attached plane for support export"
        )
    ground_id = ground_candidates[0]
    ground_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, ground_id)
    ground_rotation = _matrix_columns(data.geom_xmat[ground_id], np)
    ground_normal = ground_rotation[:, 2]
    normal_length = float(np.linalg.norm(ground_normal))
    if not normal_length > 0.0:
        raise RuntimeError("MyoSim ground plane has a degenerate normal")
    ground_normal = ground_normal / normal_length
    ground_point = np.asarray(data.geom_xpos[ground_id], dtype=float)
    support_names = {
        f"{prefix}_col{ordinal}_{side}"
        for prefix in ("foot", "bofoot")
        for ordinal in range(1, 5)
        for side in ("r", "l")
    }
    support_geometries: list[dict[str, object]] = []
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        if name not in support_names:
            continue
        geom_type = int(model.geom_type[index])
        if geom_type not in {sphere_type, capsule_type, ellipsoid_type}:
            raise RuntimeError(
                f"MyoSim support geometry {name} has unsupported MuJoCo type {geom_type}"
            )
        body = int(model.geom_bodyid[index])
        if body <= 0:
            raise RuntimeError(f"MyoSim support geometry {name} is not attached to a body")
        rotation = _matrix_columns(data.geom_xmat[index], np)
        center = np.asarray(data.geom_xpos[index], dtype=float)
        size = np.asarray(model.geom_size[index], dtype=float)
        support_world = _support_point_on_geometry(
            geom_type=geom_type,
            sphere_type=sphere_type,
            capsule_type=capsule_type,
            ellipsoid_type=ellipsoid_type,
            center=center,
            rotation=rotation,
            size=size,
            ground_normal=ground_normal,
            np=np,
        )
        inertia_rotation = _matrix_columns(data.ximat[body], np)
        support_local = inertia_rotation.T @ (
            support_world - np.asarray(data.xipos[body], dtype=float)
        )
        support_geometries.append(
            {
                "id": index,
                "name": name,
                "body": body,
                "body_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                "type": geom_type,
                "type_name": (
                    "sphere" if geom_type == sphere_type else
                    "capsule" if geom_type == capsule_type else "ellipsoid"
                ),
                "size_m": [float(value) for value in size],
                "support_point_local_com_m": [float(value) for value in support_local],
                "support_point_world_m": [float(value) for value in support_world],
                "default_signed_plane_distance_m": float(
                    np.dot(support_world - ground_point, ground_normal)
                ),
                # MuJoCo combines pair friction conservatively; preserving both
                # source values lets the native contact artifact make that rule
                # visible instead of silently inventing a generic coefficient.
                "friction_tangential": float(max(
                    float(model.geom_friction[index, 0]),
                    float(model.geom_friction[ground_id, 0]),
                )),
            }
        )
    expected_support_names = {
        f"foot_col{ordinal}_{side}"
        for ordinal in (1, 3, 4)
        for side in ("r", "l")
    } | {
        f"bofoot_col{ordinal}_{side}"
        for ordinal in (1, 2)
        for side in ("r", "l")
    }
    found_support_names = {str(entry["name"]) for entry in support_geometries}
    if found_support_names != expected_support_names:
        raise RuntimeError(
            "myofullbody support geometry set drifted: expected "
            + ", ".join(sorted(expected_support_names))
            + "; found " + ", ".join(sorted(found_support_names))
        )
    support_geometries.sort(key=lambda entry: str(entry["name"]))

    muscle_actuators: list[int] = []
    spatial_tendon = int(mujoco.mjtTrn.mjTRN_TENDON)
    muscle_dyn = int(mujoco.mjtDyn.mjDYN_MUSCLE)
    muscle_gain = int(mujoco.mjtGain.mjGAIN_MUSCLE)
    muscle_bias = int(mujoco.mjtBias.mjBIAS_MUSCLE)
    for actuator in range(model.nu):
        if (
            int(model.actuator_trntype[actuator]) != spatial_tendon
            or int(model.actuator_dyntype[actuator]) != muscle_dyn
            or int(model.actuator_gaintype[actuator]) != muscle_gain
            or int(model.actuator_biastype[actuator]) != muscle_bias
        ):
            raise RuntimeError(
                f"actuator {actuator} is not an authored MuJoCo muscle-tendon element"
            )
        tendon = int(model.actuator_trnid[actuator, 0])
        if tendon < 0:
            raise RuntimeError(f"muscle actuator {actuator} has no tendon transmission")
        muscle_actuators.append(actuator)
    if len(muscle_actuators) != model.nu:
        raise RuntimeError("myofullbody contains non-muscle controls")

    required_sites: set[int] = set()
    required_geoms: set[int] = set()
    routes: dict[int, list[dict[str, object]]] = {}
    supported_wraps = {
        int(mujoco.mjtWrap.mjWRAP_SITE): "site",
        int(mujoco.mjtWrap.mjWRAP_SPHERE): "sphere",
        int(mujoco.mjtWrap.mjWRAP_CYLINDER): "cylinder",
    }
    for actuator in muscle_actuators:
        tendon = int(model.actuator_trnid[actuator, 0])
        address = int(model.tendon_adr[tendon])
        count = int(model.tendon_num[tendon])
        route: list[dict[str, object]] = []
        for route_index in range(address, address + count):
            source_type = int(model.wrap_type[route_index])
            object_id = int(model.wrap_objid[route_index])
            kind = supported_wraps.get(source_type)
            if kind is None:
                raise RuntimeError(
                    f"tendon {tendon} has unsupported source wrap type {source_type}"
                )
            if kind == "site":
                required_sites.add(object_id)
            else:
                required_geoms.add(object_id)
            route.append(
                {
                    "kind": kind,
                    "source_id": object_id,
                    "side_site_source_id": (
                        int(round(float(model.wrap_prm[route_index]))) if kind != "site" else -1
                    ),
                }
            )
        if len(route) < 2 or route[0]["kind"] != "site" or route[-1]["kind"] != "site":
            raise RuntimeError(f"tendon {tendon} is not a spatial site route")
        for left, middle, right in zip(route, route[1:], route[2:]):
            if middle["kind"] != "site" and (left["kind"] != "site" or right["kind"] != "site"):
                raise RuntimeError(f"tendon {tendon} has a non-site-bounded wrap")
        for node in route:
            side_id = int(node["side_site_source_id"])
            if side_id >= 0:
                required_sites.add(side_id)
        routes[tendon] = route

    sites = {
        index: {
            "id": index,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, index),
            "body": int(model.site_bodyid[index]),
            "position_body_m": [float(value) for value in model.site_pos[index]],
        }
        for index in sorted(required_sites)
    }
    geoms = {}
    for index in sorted(required_geoms):
        geom_type = int(model.geom_type[index])
        if geom_type not in {
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        }:
            raise RuntimeError(f"wrap geometry {index} has unsupported MuJoCo geometry type {geom_type}")
        geoms[index] = {
            "id": index,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index),
            "body": int(model.geom_bodyid[index]),
            "type": geom_type,
            "position_body_m": [float(value) for value in model.geom_pos[index]],
            "quaternion_body_xyzw": _mujoco_quaternion_to_xyzw(model.geom_quat[index]),
            "radius_m": float(model.geom_size[index, 0]),
        }

    muscles = []
    for actuator in muscle_actuators:
        tendon = int(model.actuator_trnid[actuator, 0])
        muscles.append(
            {
                "id": actuator,
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator),
                "tendon": tendon,
                "route": routes[tendon],
                "length_range_m": [float(value) for value in model.actuator_lengthrange[actuator]],
                "acceleration_scale": float(model.actuator_acc0[actuator]),
                "control_range": [float(value) for value in model.actuator_ctrlrange[actuator]],
                "gain_parameters": [float(value) for value in model.actuator_gainprm[actuator]],
                "bias_parameters": [float(value) for value in model.actuator_biasprm[actuator]],
                "dynamic_parameters": [float(value) for value in model.actuator_dynprm[actuator]],
                "oracle_length_m": float(data.ten_length[tendon]),
                "oracle_force_n_at_activation_0_5": float(data.actuator_force[actuator]),
            }
        )

    return {
        "schema": "numi.human.myosim-mujoco-export.v1",
        "source": {
            "id": "myosim_fullbody",
            "repository": "https://github.com/MyoHub/myo_sim",
            "revision": "33c89c2bde282553dde3f526768eb3bdcfaa7649",
            "archive": archive.name,
            "archive_sha256": _sha256(archive),
            "license": "Apache-2.0",
            "mujoco_version": mujoco.__version__,
        },
        "model": {
            "name": "myofullbody",
            "body_count_with_world": int(model.nbody),
            "joint_count": int(model.njnt),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "tendon_count": int(model.ntendon),
            "root_joint": root_joint,
            "root_body": root_body,
            "gravity_m_s2": [float(value) for value in model.opt.gravity],
            "timestep_seconds": float(model.opt.timestep),
            "default_qpos": [float(value) for value in model.qpos0],
        },
        "support_contact": {
            "ground": {
                "source_geom_id": ground_id,
                "name": ground_name,
                "point_world_m": [float(value) for value in ground_point],
                "normal_world": [float(value) for value in ground_normal],
                "friction_tangential": float(model.geom_friction[ground_id, 0]),
            },
            "geometries": support_geometries,
        },
        "bodies": bodies,
        "joints": joints,
        "joint_equalities": joint_equalities,
        "sites": list(sites.values()),
        "wrap_geometries": list(geoms.values()),
        "muscles": muscles,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = export_fullbody(arguments.sources.resolve())
    except RuntimeError as error:
        print(f"numilab-human MyoSim export: {error}", file=sys.stderr)
        return 2
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
