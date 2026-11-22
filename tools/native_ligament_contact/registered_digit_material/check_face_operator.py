"""Check the actual M6 face force and tangent at a registered supported configuration."""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from material_context import sha


class NativeFace:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.build = json.loads((self.root / "build.json").read_text())
        for name, key in [("face-check", "host"), ("face.metallib", "library")]:
            if sha(self.root / name) != self.build[key]:
                raise ValueError("Metal validation executable changed")

    def compare(self, points, face, area, density, dt, width, direction, force, action):
        source = np.zeros((len(face), 2, 4), dtype="<f4")
        source[:, 0, :3] = points[face, 2]
        source[:, 0, 3] = area
        source[:, 1, :3] = density[face]
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "faces"
            dst = Path(directory) / "response"
            source.tofile(src)
            command = [
                str(self.root / "face-check"),
                str(self.root / "face.metallib"),
                str(src),
                str(dst),
                str(dt),
                str(width),
            ]
            state = subprocess.run(command, capture_output=True, text=True, check=True)
            values = (
                np.fromfile(dst, dtype="<f4").astype(float).reshape(len(face), 3, 4)
            )
        if not np.isfinite(values).all() or np.any(values[:, 0, :3] < 0):
            raise ValueError("Invalid Metal contact response")
        a = values[:, 1, :3].T
        b = values[:, 2, :3].T
        rows = np.array(
            [[a[0], a[1], a[2]], [a[1], b[0], b[1]], [a[2], b[1], b[2]]]
        ).transpose(2, 0, 1)
        computed_force = np.bincount(
            face.ravel(), weights=values[:, 0, :3].ravel(), minlength=len(points)
        )
        nodal_action = np.einsum("fij,fj->fi", rows, direction[face])
        computed_action = np.bincount(
            face.ravel(), weights=nodal_action.ravel(), minlength=len(points)
        )

        def error(actual, reference):
            denom = np.linalg.norm(reference)
            return float(np.linalg.norm(actual - reference) / (denom if denom else 1.0))

        eigen = np.linalg.eigvalsh(rows)
        if np.any(eigen[:, 0] < -2e-6 * eigen[:, -1]):
            raise ValueError("Indefinite Metal face tangent")
        f = error(computed_force, force)
        k = error(computed_action, action)
        if not (f < 1e-4 and k < 1e-4):
            raise ValueError(
                f"Metal and reference support disagree force={f}, tangent={k}"
            )
        return {
            "device_sample": state.stdout.strip(),
            "metallib": self.build["library"],
            "force_relative_error": f,
            "hessian_action_relative_error": k,
            "normal_N": float(computed_force.sum()),
        }
