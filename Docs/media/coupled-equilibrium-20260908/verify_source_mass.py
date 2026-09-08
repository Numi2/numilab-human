"""Independent MuJoCo mass-action check of the offline native equilibrium."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import math

MASS_TOLERANCE = 1e-6
GRAVITY_TOLERANCE = 2e-6

def audit_vectors(native_force, source_force, native_gravity, source_gravity):
    vectors = (native_force, source_force, native_gravity, source_gravity)
    if any(len(v) != 128 or any(type(x) not in (int, float) or not math.isfinite(x) for x in v) for v in vectors):
        raise ValueError("invalid generalized force vectors")
    def error(a, b):
        return max(abs(x-y) / (1 + abs(x) + abs(y)) for x, y in zip(a, b))
    mass = error(native_force, source_force)
    gravity = error(native_gravity, source_gravity)
    if mass > MASS_TOLERANCE or gravity > GRAVITY_TOLERANCE:
        raise ValueError(f"source mass/gravity discrepancy: {mass}, {gravity}")
    return {"status": "offline_source_mass_action_passed",
            "maximum_relative_mass_action_error": mass,
            "maximum_relative_gravity_error": gravity,
            "mass_tolerance": MASS_TOLERANCE, "gravity_tolerance": GRAVITY_TOLERANCE}

def main():
    import mujoco
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('--certificate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = next(p for p in Path(__file__).resolve().parents if (p / 'src/numilab_human').is_dir())
    oracle = base / 'Docs/media/curved-support-20260908/verify_source_curved.py'
    geometry_output = args.output.with_suffix('.geometry.json')
    sys.argv = [str(oracle), '--certificate', str(args.certificate), '--output', str(geometry_output)]
    env = runpy.run_path(str(oracle), run_name='__main__')
    model, data, manifest = env['model'], env['data'], env['manifest']
    raw = args.certificate.read_text()
    def record(prefix):
        matches = [json.loads(line[len(prefix):]) for line in raw.splitlines() if line.startswith(prefix)]
        assert len(matches) == 1
        return matches[0]
    q = np.array(record('compiled_equilibrium_q='), dtype=np.float64)
    reaction = record('compiled_equilibrium_reactions=')
    root_body = env['body']
    com_matrix = np.empty(9)
    inertia_matrix = np.empty(9)
    mujoco.mju_quat2Mat(com_matrix, np.array([q[6], q[3], q[4], q[5]]))
    mujoco.mju_quat2Mat(inertia_matrix, model.body_iquat[root_body])
    rotation = com_matrix.reshape(3, 3) @ inertia_matrix.reshape(3, 3).T
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    data.qpos[:3] = q[:3] - rotation @ model.body_ipos[root_body]
    data.qpos[3:7] = quat
    for joint in manifest['core_tree']['source_joint_map']:
        data.qpos[model.jnt_qposadr[joint['source_joint_id']]] = q[joint['core_q_index']]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    source_mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, source_mass)
    nv = manifest['core_tree']['nv']
    transform = np.zeros((model.nv, nv))
    # Native free velocity is COM linear/world angular. Source free velocity is
    # body-origin linear/body-local angular. Scalar source velocities map directly.
    offset = rotation @ model.body_ipos[root_body]
    x, y, z = offset
    transform[:3, :3] = np.eye(3)
    transform[:3, 3:6] = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    transform[3:6, 3:6] = rotation.T
    for joint in manifest['core_tree']['source_joint_map']:
        transform[model.jnt_dofadr[joint['source_joint_id']], joint['core_v_index']] = 1
    acceleration = np.array(reaction['acceleration'])
    inertial_force = transform.T @ source_mass @ transform @ acceleration
    native_force = np.array(reaction['force_residual'])
    gravity = transform.T @ data.qfrc_bias
    native_gravity = np.array(reaction['gravity_target'])
    mass_error = np.abs(inertial_force - native_force)
    gravity_error = np.abs(gravity - native_gravity)
    report = {
        'schema': 'numi.human.offline-source-mass-action.v1',
        'mujoco_version': mujoco.__version__,
        'certificate_sha256': hashlib.sha256(args.certificate.read_bytes()).hexdigest(),
        'source_archive_sha256': env['report']['source_archive_sha256'],
        'pose_precision': 'native FP64 compiler output; source inertial parameters retain source precision',
        'maximum_absolute_mass_action_error': float(mass_error.max()),
        'maximum_relative_mass_action_error': float(np.max(mass_error / (1 + np.abs(inertial_force) + np.abs(native_force)))),
        'maximum_absolute_gravity_error': float(gravity_error.max()),
        'maximum_relative_gravity_error': float(np.max(gravity_error / (1 + np.abs(gravity) + np.abs(native_gravity)))),
        'native_force_residual': native_force.tolist(),
        'source_mass_times_native_acceleration': inertial_force.tolist(),
        'source_gravity': gravity.tolist(),
        'native_gravity': native_gravity.tolist(),
        'boundary': 'independent offline mass action and gravity; no stepping or source-compliant dynamic claim',
    }
    report.update(audit_vectors(native_force.tolist(), inertial_force.tolist(), native_gravity.tolist(), gravity.tolist()))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, list)}, indent=2))

if __name__ == "__main__":
    main()
