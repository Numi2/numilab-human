"""Cross-check the fitted physical support against the production Metal force and matrix-vector product."""

import json, sys, subprocess, tempfile
from pathlib import Path
import numpy as np
from material_context import sha
from compare_bed_sampling import integrate
from fit_native_bed_barrier_load import force


def verify(root):
    device = root / "material-nodes"
    ledger = json.loads((root / "wrench-fit/result.json").read_text())
    with np.load(root / "load/support.npz") as z:
        d = dict(z)
    with np.load(root / "wrench-fit/preload.npz") as z:
        emitted = z["coordinates"]
    t = d["f"]
    area = d["area"]
    dt, w = float(d["dt"]), float(d["width"])
    mass = d["mass"]
    n = len(mass)
    unit_mass = force(0.85 * w, mass, dt, w) / force(0.85 * w, 1.0, dt, w)
    inv = d["density"] / unit_mass
    codes = np.argsort(t.ravel(), kind="stable").astype("u4")
    lengths = np.bincount(t.ravel(), minlength=n)
    ranges = np.c_[lengths.cumsum() - lengths, lengths].astype("<u4")
    center = np.asarray(ledger["gravity"]["center_m"])
    rows = {}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        np.c_[t, np.zeros(len(t))].astype("<u4").tofile(p / "faces")
        np.c_[area, inv[t]].astype("<f4").tofile(p / "measure")
        ranges.tofile(p / "range")
        codes.tofile(p / "corners")
        np.array([dt, w], "<f4").tofile(p / "params")
        xy = d["y"][:, :2]
        direction = np.sin(3 * xy[:, 0]) + np.cos(5 * xy[:, 1])
        a = np.zeros((n, 4), "<f4")
        a[:, 2] = direction
        a.tofile(p / "direction")
        for name in ("initial", "emitted"):
            y = d["y"].copy()
            if name == "emitted":
                y[d["free"]] = emitted
            a[:, 0] = y[:, 2]
            a[:, 1] = mass
            a[:, 2] = 0
            a.tofile(p / "nodes")
            a.tofile(p / "accepted")
            proc = subprocess.run(
                [
                    str(device / "check-metal"),
                    str(device / "test.metallib"),
                    str(p),
                    str(p / "out"),
                    name,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            g = np.fromfile(p / "out", "<f4").reshape(-1, 4).astype(float)
            f = g[:, 0] / dt
            ref, _, K = integrate(
                y, t, area, d["density"], dt, w, 4, normal_direction=direction
            )
            force_rel = np.linalg.norm(f - ref) / np.linalg.norm(ref)
            k_rel = np.linalg.norm(g[:, 1] / dt**2 - K) / np.linalg.norm(K)
            torque = [
                float((xy[:, 1] - center[1]) @ f),
                float(-(xy[:, 0] - center[0]) @ f),
            ]
            row = dict(
                normal_N=float(f.sum()),
                torque_Nm=torque,
                force_error_relative=force_rel,
                tangent_action_error_relative=k_rel,
                device=proc.stdout.strip(),
                nodes_loaded=int((f > 0).sum()),
            )
            rows[name] = row
            if not np.isfinite(g).all() or max(force_rel, k_rel) > 1e-5:
                raise ValueError(row)
    target = ledger["weight_N"]
    if (
        abs(rows["emitted"]["normal_N"] - target) > 0.02
        or np.linalg.norm(rows["emitted"]["torque_Nm"]) > 0.01
    ):
        raise ValueError(rows)
    result = dict(
        rows=rows,
        mesh=d["t"].shape[0],
        production_shader_library=sha(device / "build/NumiMatter.metallib"),
        test_library=sha(device / "test.metallib"),
        fit=sha(root / "wrench-fit/result.json"),
    )
    (root / "native-wrench.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
