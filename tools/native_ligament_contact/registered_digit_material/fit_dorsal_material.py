"""Fit one connected physical skin/fat displacement across the bilateral distal digital corridors."""

import json
from pathlib import Path
import numpy as np
from material_context import sha
from torso_cell_geometry import geom


def ramp(x):
    t = np.clip(x, 0, 1)
    return t * t * (3 - 2 * t)


def make_field(x, meta):
    # Both hands are nearly edge-on in this supported supine frame. One signed
    # world-x displacement fixes the remaining two coordinates and gives an affine
    # determinant constraint for every tetrahedron. Eligibility is anatomical.
    field = np.zeros(len(x))
    for side in ("right", "left"):
        value = np.zeros(len(x))
        for name, r in meta["digits"].items():
            if name.endswith("_l") != (side == "left"):
                continue
            p = np.array(r["trial_enthesis"]["world_m"])
            axis = np.array(r["axis"])
            up = np.array(r["dorsal"])
            lat = np.array(r["lateral"])
            if (
                np.sign(up[0]) != (1 if side == "left" else -1)
                or np.argmax(abs(up)) != 0
            ):
                raise ValueError("Unsupported authoring frame")
            delta = x - p
            ax = delta @ axis
            h = delta @ up
            y = delta @ lat
            # Smooth to zero past the terminal and middle phalanx. The taper reaches zero on the palmar side.
            g = (
                ramp((ax + 0.023) / 0.013)
                * ramp((0.012 - ax) / 0.007)
                * ramp((0.012 - abs(y)) / 0.007)
                * ramp((h + 0.003) / 0.003)
            )
            value = np.maximum(value, g / abs(up[0]))
        field += value * (1 if side == "left" else -1)
    return field


def inputs(run):
    j = json.loads((run / "launch.json").read_text())
    env = j["environment"]
    p = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"

    def external(key, ident):
        path = Path(env[p + key])
        assert sha(path) == env[p + ident]
        return path

    ref = external("TETGEN_NODE", "TETGEN_NODE_SHA256")
    tet = external("TETGEN_ELEMENTS", "TETGEN_ELEMENTS_SHA256")
    cur = Path(env["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
    assert sha(cur) == j["pins"][str(cur)]
    x = np.loadtxt(ref, skiprows=1, usecols=(1, 2, 3)).astype("f4")
    y = np.loadtxt(cur, skiprows=1, usecols=(1, 2, 3)).astype("f4")
    z = np.loadtxt(tet, skiprows=1, dtype="i4")
    assert np.array_equal(z[:, 0], np.arange(1, len(z) + 1))
    return (
        x,
        y,
        z[:, 1:5] - 1,
        z[:, 5],
        dict(
            reference_sha256=sha(ref),
            installed_pose_sha256=sha(cur),
            elements_sha256=sha(tet),
        ),
    )


def evaluate(x, t):
    v, q = geom(x.astype(float), t)
    return dict(
        bad=int(((v <= 1e-18) | (q < 1e-5)).sum()),
        inverted=int((v <= 0).sum()),
        minimum_volume=float(v.min()),
        minimum_shape=float(q.min()),
    )
