"""Check the retained exact cardiac model against the first rejected target."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def run(run_dir, output):
    launch = json.loads((run_dir / "launch.json").read_text())
    argv = launch["argv"]
    receipt = Path(argv[argv.index("--resting-anatomy-receipt") + 1])
    field = json.loads(receipt.read_text())["provenance"]["cardiac_geometry_binding"]["common_field"]
    witness = json.loads((run_dir / "common-failure.json").read_text())["first_failure_latch"]
    assert witness["common_status"][0] == 5
    point = np.array(witness["common_first"] + witness["common_second"][:3])
    m3 = witness["chamber_targets_m3"] + field["material_target_volumes_m3"]
    target = np.asarray(m3) / np.asarray(field["source_reference_volumes_m3"])

    def input_field(key):
        spec = field[key]
        raw = (receipt.parent / spec["path"]).read_bytes()
        assert sha(raw) == spec["sha256"]
        return np.frombuffer(raw, dtype="<f4").astype(float)

    boxes = input_field("domain_boxes").reshape(-1, 2, 8)[:, :, :7]
    coefficients = input_field("polynomials").reshape(7, 120)
    degrees = np.asarray([e for e in itertools.product(range(4), repeat=7) if sum(e) <= 3])

    def values(q):
        return coefficients @ np.prod(q ** degrees, axis=1)

    def jacobian(q):
        column = []
        for d in range(7):
            terms = degrees.copy()
            terms[:, d] = np.maximum(terms[:, d] - 1, 0)
            column.append(coefficients @ (np.prod(q ** terms, axis=1) * degrees[:, d]))
        return np.stack(column, axis=1)

    def residual(q):
        return max(abs(values(q) - target))

    exact = residual(point)
    assert abs(exact - witness["common_diagnostics"][0]) < 2.e-6
    best = (float("inf"), None, None)
    for i, (lower, upper) in enumerate(boxes):
        q = np.clip(point, lower, upper)
        for _ in range(8):
            r = values(q) - target
            j = jacobian(q)
            # Numerical diagnostic only. This search does not prove a global minimum.
            inequality = np.column_stack((np.concatenate((j, -j)), -np.ones(14)))
            step = linprog(
                [0] * 7 + [1], A_ub=inequality, b_ub=np.concatenate((-r, r)),
                bounds=list(zip(lower - q, upper - q)) + [(0, None)], method="highs",
            )
            assert step.success, step.message
            q = np.clip(q + step.x[:7], lower, upper)
            if np.linalg.norm(step.x[:7], ord=np.inf) < 1.e-10:
                break
        if residual(q) < best[0]:
            best = (residual(q), i, q)
    result = dict(
        first_failed_step=witness["step"],
        last_accepted_time_s=0.002*witness["step"],
        atlas_sha256=field["domain_boxes"]["sha256"],
        tolerance=field["solver"]["relative_volume_tolerance"],
        unrestricted_residual=exact,
        best_box_residual=best[0],
        best_box_id=best[1],
        best_box_coordinates=best[2].tolist(),
        status="numerical search across every retained box",
        chamber_targets_m3=m3[:4],
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.run.resolve(), args.output.resolve())
