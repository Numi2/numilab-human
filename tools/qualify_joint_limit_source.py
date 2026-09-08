#!/usr/bin/env python3
"""Emit pinned MuJoCo constraintUpdate oracles, never run a Human controller."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def f32(x):
    return float(np.float32(x))


def generate():
    if mujoco.__version__ != "3.12.0":
        raise RuntimeError("source limit qualification requires MuJoCo 3.12.0")
    cases = []
    definitions = [
        ("lower", "hinge", -.30, .03125, [.02, 1], [.7, .9, .1, .4, 2], True),
        ("upper", "hinge", .55, .03125, [.02, 1], [.7, .9, .1, .4, 2], True),
        ("inactive", "hinge", 0, .03125, [.02, 1], [.7, .9, .1, .4, 2], True),
        ("exact_lower_margin", "slide", -.125, .125, [.02, 1], [.7, .9, .1, .4, 2], True),
        ("exact_upper_margin", "slide", .375, .125, [.02, 1], [.7, .9, .1, .4, 2], True),
        ("inside_margin", "slide", -.234375, .03125, [.02, 1], [.7, .9, .1, .4, 2], True),
        ("direct", "hinge", -.30, 0, [-2500, -90], [.8, .95, .1, .7, 3], True),
        ("refsafe", "slide", .55, 0, [.00001, .75], [.8, .95, .1, .5, 2], True),
        ("refsafe_off", "slide", .55, 0, [.0001, .75], [.8, .95, .1, .5, 2], False),
        ("both_sides", "slide", .125, .5, [-300, -20], [.6, .85, .2, .5, 2], True),
        ("constant_impedance", "hinge", -.30, 0, [.02, 1], [.9, .9, .1, .5, 2], True),
        ("zero_width", "slide", -.30, 0, [.02, 1], [.7, .9, 0, .5, 2], True),
        ("linear_impedance", "slide", .51, .03125, [.02, 1], [.7, .9, .1, .5, 1], True),
        ("clamped_impedance", "hinge", -.30, 0, [.02, 1], [1, 1, .1, 0, .5], True),
        ("zero_stiffness", "slide", -.30, 0, [0, -5], [.7, .9, .1, .5, 2], True),
    ]
    for name, kind, position, margin, solref, solimp, refsafe in definitions:
        position, margin = f32(position), f32(margin)
        solref, solimp = list(map(f32, solref)), list(map(f32, solimp))
        h = f32(.001)
        xml = f'''<mujoco><compiler angle="radian"/>
<option timestep="{h}" gravity="0 0 0" integrator="Euler" jacobian="dense">
<flag refsafe="{'enable' if refsafe else 'disable'}"/></option><worldbody>
<body><freejoint/><geom type="sphere" size=".1" mass="2" contype="0" conaffinity="0"/>
<body pos="0 0 .2"><joint type="{kind}" axis="0 1 0" limited="true"
range="-.25 .5" margin="{margin}" solreflimit="{' '.join(map(str, solref))}"
solimplimit="{' '.join(map(str, solimp))}"/>
<geom type="capsule" size=".02 .2" mass="1" contype="0" conaffinity="0"/>
</body></body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        state = mujoco.MjData(model)
        state.qpos[7] = position
        state.qvel[6] = f32(-.03)
        mujoco.mj_forward(model, state)
        if model.nv != 7 or np.any(state.efc_type != mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT):
            raise RuntimeError("oracle fixture has unexpected constraints")
        J = state.efc_J.reshape(state.nefc, model.nv).copy()
        R, aref = state.efc_R.copy(), state.efc_aref.copy()
        free = f32(.04)
        increment = free - state.qvel[6]
        expected_lin = [[0., 0., 0., 0.], [0., 0., 0., 0.]]
        for i in range(state.nefc):
            side = 0 if J[i, 6] > 0 else 1
            expected_lin[side] = [J[i, 6], h * aref[i] - J[i, 6] * increment,
                                  1 / R[i], state.efc_pos[i] - state.efc_margin[i]]
        deltas = [f32(x) for x in (0., -.1, .1, -2., 2.)]
        for row in expected_lin:
            if row[2]:
                deltas += [f32(row[1] / row[0] + x) for x in (-.02, .02)]
        candidates = []
        for delta in deltas:
            jar = np.ascontiguousarray(J[:, 6] * ((increment + delta) / h) - aref)
            mujoco.mj_constraintUpdate(model, state, jar, None, 0)
            impulse = float(h * state.qfrc_constraint[6])
            tangent = float(np.sum(J[jar < 0, 6] ** 2 / R[jar < 0]))
            candidates.append({"delta": delta, "residual": impulse, "tangent": tangent,
                               "source_jar": jar.tolist(), "source_force": state.efc_force.tolist(),
                               "source_state": state.efc_state.tolist()})
        cases.append({"name": name, "xml": xml, "xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                      "h": h, "flags": int(refsafe), "q": position, "v0": float(state.qvel[6]),
                      "v_free": free, "range": [-.25, .5], "margin": margin,
                      "solref": solref, "solimp": solimp,
                      "source_inverse_weight": float(model.dof_invweight0[6]),
                      "linearization": expected_lin, "candidates": candidates})
    return {"schema": "numi.human.mujoco-limit-oracle.v1", "mujoco_version": mujoco.__version__,
            "source": "https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_core_constraint.c",
            "method": "mj_forward then mj_constraintUpdate at explicit acceleration candidates; no stepping",
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(generate(), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
