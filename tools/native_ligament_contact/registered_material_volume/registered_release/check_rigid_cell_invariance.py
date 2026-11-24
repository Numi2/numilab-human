"""Run a rigid-motion patch test on the actual registered tetrahedra."""

import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from supported_material_admission import geom


def check(volume, out):
    load = lambda p: np.loadtxt(p, skiprows=1, usecols=(1, 2, 3))
    original = load(volume / "current.node")
    reference = load(volume / "mesh.node")
    tet = (
        np.loadtxt(volume / "mesh.ele", skiprows=1, usecols=(1, 2, 3, 4), dtype="i4")
        - 1
    )
    mass = np.fromfile(volume / "node-mass.f64", "<f8")
    center = mass @ original / mass.sum()
    axis = np.array([1.0, 2.0, -3.0])
    axis /= np.linalg.norm(axis)
    chunks = np.array_split(tet, 24)

    def metrics(z):
        return np.concatenate([np.stack(geom(z, t)) for t in chunks], axis=1)

    ref = metrics(reference)[0]
    v0, h0 = metrics(original)

    def counts(v, h):
        j = v / ref
        bad = (v <= 1.0e-18) | (j < 0.4) | (j > 2.5) | (h < 1.0e-5)
        return dict(
            inverted=int((v <= 0).sum()),
            failed_quality=int(bad.sum()),
            below_shape=int((h < 1.0e-5).sum()),
            maximum_relative_volume_error=float(np.max(abs(v / v0 - 1))),
            min_J=float(j.min()),
            max_J=float(j.max()),
        )

    baseline = counts(v0, h0)
    trials = []
    for theta, dz in ((0.0, 0.001), (0.0001, 0.0), (0.005, 0.0), (0.05, 0.0)):
        z = (
            Rotation.from_rotvec(theta * axis).apply(original - center)
            + center
            + [0.0, 0.0, dz]
        )
        # Both stores model Float32 GPU buffers. Reconstruct in float64 before predicates
        # to isolate coordinate storage from arithmetic in a particular validator.
        hi = z.astype("f4")
        lo = (z - hi.astype("f8")).astype("f4")
        du = (z - original).astype("f4").astype("f8")
        displacement = z - reference
        upper = displacement.astype("f4").astype("f8")
        lower = (displacement - upper).astype("f4").astype("f8")
        r = dict(rotation_rad=theta, translation_z_m=dz)
        for key, positions in [
            ("exact_double", z),
            ("absolute_f32", hi.astype("f8")),
            ("reference_plus_f32_displacement", original + du),
            ("high_low_f32", hi.astype("f8") + lo.astype("f8")),
            ("native_reference_high_low", reference + (upper + lower)),
        ]:
            r[key] = counts(*metrics(positions))
        trials.append(r)
    out.write_text(
        json.dumps(
            dict(
                registered=digest(volume / "current.node"),
                reference=digest(volume / "mesh.node"),
                elements=digest(volume / "mesh.ele"),
                baseline=baseline,
                center_m=center.tolist(),
                rotation_axis=axis.tolist(),
                trials=trials,
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        [
            (
                r["rotation_rad"],
                r["translation_z_m"],
                [
                    (k, r[k]["inverted"], r[k]["failed_quality"])
                    for k in (
                        "exact_double",
                        "absolute_f32",
                        "reference_plus_f32_displacement",
                        "high_low_f32",
                    )
                ],
            )
            for r in trials
        ]
    )


if __name__ == "__main__":
    check(*(Path(s).resolve() for s in sys.argv[1:]))
