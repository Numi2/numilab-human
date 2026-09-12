"""Compare retained contact-gap secants with native point Jacobians.

Diagnostic only: midpoint Jacobians approximate a finite nonlinear interval.
This neither runs physics nor supplies a contact/convergence acceptance gate.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct


def floats(hexadecimal, *, sample=False):
    raw = bytes.fromhex(hexadecimal)
    if len(raw) % 4:
        raise ValueError("invalid FP32 arena extent")
    values = struct.unpack("<" + str(len(raw) // 4) + "f", raw)
    if not all(math.isfinite(value) for index, value in enumerate(values)
               if not sample or index % 40 >= 4):
        raise ValueError("nonfinite arena")
    return values


def inspect(raw, timestep_ns):
    rows = [json.loads(line.split("=", 1)[1]) for line in raw.decode().splitlines()
            if line.startswith("human_support_iterate=")]
    if timestep_ns <= 0 or len(rows) < 2:
        raise ValueError("missing diagnostic interval or timestep")
    dt = timestep_ns / 1_000_000_000
    output = []
    for previous, current in zip(rows, rows[1:]):
        if previous["root"] != current["root"]:
            continue
        before, after = previous["arenas"], current["arenas"]
        samples, next_samples = floats(before["sample"], sample=True), floats(after["sample"], sample=True)
        jacobian, next_jacobian = floats(before["jacobian"]), floats(after["jacobian"])
        delta, next_delta = floats(before["delta_v"]), floats(after["delta_v"])
        # NMContactSampleGPU ABI34: 10 float4 blocks, gap is block1.w.
        count, capacity = len(samples) // 40, len(delta)
        if (len(samples) % 40 or len(next_samples) != len(samples)
                or len(jacobian) != count * 3 * capacity
                or len(next_jacobian) != len(jacobian) or len(next_delta) != capacity):
            raise ValueError("incompatible native diagnostic arenas")
        discrepancies = []
        for row in range(count):
            normal = samples[row * 40 + 8:row * 40 + 11]
            if normal != next_samples[row * 40 + 8:row * 40 + 11]:
                raise ValueError("normal changed across diagnostic interval")
            observed = (next_samples[row * 40 + 7] - samples[row * 40 + 7]) / dt
            predicted = sum(normal[axis] * 0.5 * (
                jacobian[(row * 3 + axis) * capacity + dof]
                + next_jacobian[(row * 3 + axis) * capacity + dof])
                * (next_delta[dof] - delta[dof])
                for axis in range(3) for dof in range(capacity))
            discrepancies.append({"row": row, "observed_gap_change_m": observed * dt,
                                  "predicted_gap_change_m": predicted * dt,
                                  "difference_m_per_s": observed - predicted})
        output.append({"root": current["root"], "from_iteration": previous["iteration"],
                       "to_iteration": current["iteration"], "certificate": current["certificate"],
                       "support_rows": count,
                       "l2_difference_m_per_s": math.sqrt(sum(
                           row["difference_m_per_s"] ** 2 for row in discrepancies)),
                       "largest_difference": max(discrepancies,
                           key=lambda row: abs(row["difference_m_per_s"]))})
    return {"schema": "numi.human.support-secant-diagnostic.v1", "timestep_nanoseconds": timestep_ns,
            "log_sha256": hashlib.sha256(raw).hexdigest(), "intervals": output,
            "qualification": "diagnostic_only_midpoint_jacobian_is_a_finite_interval_approximation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--timestep-ns", type=int, required=True)
    args = parser.parse_args()
    raw = args.log.read_bytes()
    if args.log.suffix == ".gz":
        raw = gzip.decompress(raw)
    print(json.dumps(inspect(raw, args.timestep_ns), indent=2))
