"""Check every restricted step against the exact affine plane gap, including grazing cases."""

from pathlib import Path
import json, sys, subprocess, tempfile
import numpy as np
from material_context import sha


def check(root):
    rng = np.random.default_rng(4921)
    n = 16384
    floor = 10 ** rng.uniform(-9, -5, n)
    gap = floor * (1 + 10 ** rng.uniform(-5, 4, n))
    dz = rng.uniform(-1, 1, n) * 10 ** rng.uniform(-8, -2, n)
    dz[::5] = 0.0
    err = floor * rng.uniform(1e-7, 2e-6, n)
    X = np.c_[gap, floor, dz, err].astype("<f4")
    with tempfile.TemporaryDirectory() as td:
        f = Path(td)
        X.tofile(f / "in")
        gpu = subprocess.run(
            [
                str(root / "check-bound"),
                str(root / "test.metallib"),
                str(f / "in"),
                str(f / "out"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        alpha = np.fromfile(f / "out", "<f4").astype(float)
    gap, floor, dz, err = X.T.astype(float)
    witness = gap + alpha * dz - floor
    if (
        len(alpha) != n
        or not np.isfinite(alpha).all()
        or np.min(alpha) < 0
        or np.max(alpha) > 1
        or np.min(witness) <= 0
    ):
        raise ValueError("Unsafe static plane sweep")
    if not np.all(alpha[(dz == 0) & ((gap - floor) > 2 * err)] == 1) or not np.any(
        alpha[dz < 0] < 0.01
    ):
        raise ValueError("Missing inward or grazing response")
    res = dict(
        device=gpu.stdout.strip(),
        cases=n,
        limited=int((alpha < 1).sum()),
        min_accepted_gap_above_floor_m=float(witness.min()),
        production_helper=sha(root / "fixed_plane_bound.metalinc"),
        metal=sha(root / "build/NumiMatter.metallib"),
    )
    (root / "sweep.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve())
