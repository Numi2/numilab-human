"""Differentiate the native multi-node force, including rows beyond the point activation band."""

import json
from pathlib import Path
import subprocess
import tempfile
import sys
import numpy as np
from material_context import sha


def run(root):
    dt = 1 / 65536
    width = 1e-5
    faces = np.array([[0, 1, 2], [2, 1, 3], [3, 1, 4]], dtype="u4")
    area = np.array([0.0002, 0.0003, 0.00007])
    mass = np.array([0.001, 0.004, 0.005, 0.0005, 0.002])
    dual = np.bincount(faces.ravel(), weights=np.repeat(area / 3, 3), minlength=5)
    ref = width * np.array([0.73, 1.02, 0.89, 1.1, 1.25])
    direction = np.array([0.6, -0.3, 0.2, -0.8, 0.1])
    per = [[] for _ in mass]
    for f, tri in enumerate(faces):
        for k, n in enumerate(tri):
            per[n].append(3 * f + k)
    bounds = np.asarray(
        [[sum(len(a) for a in per[:n]), len(p)] for n, p in enumerate(per)], dtype="u4"
    )

    def pad(a):
        return np.c_[a, np.zeros(len(a), dtype=a.dtype)]

    records = {}
    with tempfile.TemporaryDirectory() as tmp:
        r = Path(tmp)
        pad(faces).astype("<u4").tofile(r / "faces")
        np.c_[area, 1.0 / dual[faces]].astype("<f4").tofile(r / "measure")
        bounds.tofile(r / "range")
        np.concatenate(per).astype("<u4").tofile(r / "corners")
        np.array([dt, width], "<f4").tofile(r / "params")

        def evaluate(z, d, old, v, tag):
            np.c_[z, mass, v, np.zeros_like(z)].astype("<f4").tofile(r / "nodes")
            np.c_[old, mass, v, np.zeros_like(z)].astype("<f4").tofile(r / "accepted")
            fields = np.zeros((len(d), 4), "<f4")
            fields[:, 2] = d
            fields.tofile(r / "direction")
            p = r / "out"
            subprocess.run(
                [
                    str(root / "check-metal"),
                    str(root / "test.metallib"),
                    str(r),
                    str(p),
                    tag,
                ],
                capture_output=True,
                check=True,
            )
            return np.fromfile(p, "<f4").reshape(len(d), 4).astype(float)

        for name, vel in [
            ("static", np.zeros(5)),
            ("closing", np.array([-1.0, -2.0, -0.4, 0.15, -1.0])),
        ]:
            initial = evaluate(ref, direction, ref, vel, name)
            cols = [
                evaluate(ref, col, ref, vel, "unit")[:, 1] for col in np.eye(len(ref))
            ]
            matrix = np.asarray(cols).T
            norm = np.linalg.norm(matrix)
            symmetry = np.linalg.norm(matrix - matrix.T) / norm
            eig = np.linalg.eigvalsh((matrix + matrix.T) / 2)
            h = width * 0.001
            low = evaluate(ref - h * direction, direction, ref, vel, "lower")[:, 0]
            high = evaluate(ref + h * direction, direction, ref, vel, "upper")[:, 0]
            fd = (low - high) / (2 * h) * dt
            parity = np.linalg.norm(initial[:, 1] - fd) / np.linalg.norm(fd)
            if not (
                initial[1, 0] > 0
                and initial[3, 0] > 0
                and initial[4, 0] == 0
                and symmetry < 3e-6
                and eig.min() >= -1e-6 * eig.max()
                and parity < 2e-3
            ):
                raise ValueError(
                    f"{name}: invalid boundary response: force={initial[:, 0]} symmetry={symmetry}, minimum eigenvalue={eig.min()}, derivative error={parity}"
                )
            records[name] = dict(
                normal_force_N=float(initial[:, 0].sum() / dt),
                symmetry=symmetry,
                directional_derivative_error=parity,
                eigenvalues=eig.tolist(),
                nonlocal_forces_N=(initial[[1, 3], 0] / dt).tolist(),
                offdiagonal_action=initial[:, 3].tolist(),
            )
        if records["closing"]["normal_force_N"] <= records["static"]["normal_force_N"]:
            raise ValueError("Closing branch did not increase normal response")
    return dict(
        metal=sha(root / "test.metallib"),
        source=sha(root / "plane_response.metalinc"),
        rows=records,
    )


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]).resolve()), indent=2, allow_nan=False))
