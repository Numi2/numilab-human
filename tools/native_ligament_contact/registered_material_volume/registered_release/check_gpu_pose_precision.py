"""Exercise the production GPU determinant on rigidly moved skin/fat cells."""

from pathlib import Path
import json, sys, subprocess, tempfile
import numpy as np
from scipy.spatial.transform import Rotation
from compile_connected_joint import digest
from supported_material_admission import geom


def check(volume, fixture, trial, out):
    ground = json.loads(trial.read_bytes())
    law = json.loads((fixture / "test.json").read_bytes())
    if law["failures"] or digest(fixture / "test.metal") != law["kernel_sha256"]:
        raise ValueError("Incorrect Metal predicate")
    load = lambda n: np.loadtxt(volume / n, skiprows=1, usecols=(1, 2, 3))
    x = load("mesh.node")
    y = load("current.node")
    T = (
        np.loadtxt(volume / "mesh.ele", skiprows=1, usecols=(1, 2, 3, 4), dtype="i4")
        - 1
    )
    if ground["registered"] != digest(volume / "current.node") or ground[
        "elements"
    ] != digest(volume / "mesh.ele"):
        raise ValueError("Unexpected test geometry")
    o = np.array(ground["center_m"])
    axis = np.array(ground["rotation_axis"])
    theta = 0.005
    z = Rotation.from_rotvec(axis * theta).apply(y - o) + o
    hp = z.astype("f4")
    parts = np.array_split(T, 24)
    vp = np.concatenate([geom(hp.astype(float), s)[0] for s in parts])
    bad = np.flatnonzero(vp <= 0)
    controls = np.linspace(0, len(T) - 1, 151, dtype=int)
    indices = np.unique(np.r_[bad, controls])
    threshold = next(
        s["absolute_f32"]["inverted"]
        for s in ground["trials"]
        if s["rotation_rad"] == theta
    )
    if not len(bad) == threshold:
        raise ValueError("Rigid storage test drift")
    t = T[indices]
    data = []
    expected = []
    for mode in (False, True):
        state = np.zeros((len(t), 4, 6, 4), dtype="f4")
        state[:, :, 0, :3] = hp[t]
        state[:, :, 2, :3] = x[t]
        delta = z[t] - x[t]
        hi = delta.astype("f4")
        state[:, :, 4, :3] = hi
        state[:, :, 5, :3] = delta - hi
        state[:, :, 4, 3] = int(mode)
        data.append(state)
        expected.extend([True] * len(t) if mode else vp[indices] > 0)
    n = 2 * len(t)
    bounds = np.zeros((n, 4), "f4")
    bounds[:, 3] = 1.0
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        np.concatenate(data).tofile(p / "nodes")
        bounds.tofile(p / "bounds")
        subprocess.run(
            [
                str(fixture / "test"),
                str(fixture / "test.metallib"),
                str(p / "nodes"),
                str(p / "bounds"),
                str(p / "out"),
            ],
            check=True,
        )
        r = np.fromfile(p / "out", "f4").reshape(-1, 4)
    expected = np.array(expected)
    fail = int(np.sum((r[:, 0] != expected) | (r[:, 1] != expected)))
    rec = dict(
        device=law["device"],
        kernel=digest(fixture / "test.metallib"),
        kernel_source=law["kernel_sha256"],
        study=digest(trial),
        rotation_rad=theta,
        unique_cells=len(indices),
        mode_cell_cases=n,
        absolute_storage_rejected=int((r[: len(t), 1] == 0).sum()),
        reference_high_low_rejected=int((r[len(t) :, 1] == 0).sum()),
        mismatches=fail,
    )
    out.write_text(json.dumps(rec, indent=2) + "\n")
    if fail:
        raise ValueError("GPU disagrees with storage-mode orientation")


if __name__ == "__main__":
    check(*(Path(s).resolve() for s in sys.argv[1:]))
