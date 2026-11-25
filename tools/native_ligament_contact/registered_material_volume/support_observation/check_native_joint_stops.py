"""Measure authored source-stop reactions and excursions in an accepted dense release."""

import argparse
import csv
import json
import math
import struct
from pathlib import Path

from assess_supported_release import digest


def vector(record, name, size):
    values = [float(x) for x in record[name].split(";")]
    if len(values) != size or not all(map(math.isfinite, values)):
        raise ValueError(f"Invalid physical trace {name}")
    return values


def measure(case):
    cfg = json.loads((case / "launch.json").read_bytes())
    rigid = Path(cfg["argv"][1])
    if digest(rigid) != cfg["pins"][str(rigid)]:
        raise ValueError("Different rigid source")
    data = rigid.read_bytes()
    magic, abi, engine, sources, bodies, joints, nq, nv, *_ = struct.unpack_from("<8s10I", data)
    offset = 80 + 96 + 48 + 160 * bodies + 144 * joints
    if ((magic, abi, engine) != (b"NHRIGID2", 1, 5) or
            len(data) != offset + 64 * nv + 4 * (nq + nv) + 32 * sources):
        raise ValueError("Unsupported MyoSim articulation layout")
    stops = {}
    for i in range(nv):
        row = struct.unpack_from("<8I8f", data, offset + 64 * i)
        if row[5] & 4:
            if row[3] != i or row[2] >= nq or not row[8] < row[9]:
                raise ValueError("Invalid joint stop")
            stops[i] = (row[2], row[8], row[9])
    traces = case / "native-traces"
    path = traces / "resting-com-q-integration.csv"
    observations = []
    with path.open() as f:
        for rec in csv.DictReader(f):
            before = vector(rec, "q_before_f32_semicolon", nq)
            q = vector(rec, "q_accepted_f32_semicolon", nq)
            v0 = vector(rec, "v_before_f32_semicolon", nv)
            v1 = vector(rec, "v_accepted_f32_semicolon", nv)
            impulses = vector(rec, "gpu_stand_source_limit_impulses_f32_by_local_v_index_semicolon", nv)
            dt = float(rec["dt_s"])
            if (not math.isfinite(dt) or dt <= 0 or int(rec["accepted_step"]) != len(observations) + 1
                    or int(rec["gpu_stand_source_limit_impulse_count"]) != nv):
                raise ValueError("Invalid accepted source-limit record")
            active = [i for i, value in enumerate(impulses) if value != 0]
            if any(i not in stops for i in active):
                raise ValueError("Reaction without an authored joint stop")
            outside = {}
            outward_at_closed_stop = {}
            for i, (iq, lower, upper) in stops.items():
                gap = min(q[iq] - lower, upper - q[iq])
                if before[iq] <= lower and v1[i] < 0:
                    outward_at_closed_stop[i] = -v1[i]
                elif before[iq] >= upper and v1[i] > 0:
                    outward_at_closed_stop[i] = v1[i]
                if gap < 0:
                    outside[i] = -gap
            witnesses = {}
            for i in active:
                iq, lower, upper = stops[i]
                witnesses[i] = dict(before=before[iq], after=q[iq], lower=lower,
                    upper=upper, impulse=impulses[i], terminal_velocity=v1[i])
            delta = [(b - a) / dt for a, b in zip(v0, v1, strict=True)]
            peak = max(range(6, nv), key=lambda i: abs(delta[i]))
            observations.append(dict(step=int(rec["accepted_step"]), dt_s=dt,
                active_stops=len(active), stop_witnesses=witnesses,
                outside_exact_bounds=outside, outward_velocity_at_closed_stop=outward_at_closed_stop,
                peak_internal_v_index=peak,
                peak_internal_acceleration=abs(delta[peak])))
    log = (case / "stdout.log").read_text().splitlines()
    def line(prefix):
        match, = [s for s in log if s.startswith(prefix)]
        return dict(t.split("=", 1) for t in match.split() if "=" in t)

    support = line("resting_fem_skin_bed_accepted_root_reaction ")
    owner = line("resting_bed_contact_owner=")
    init = line("resting_initialization=dynamic_release ")
    timing = line("resting_integrated_body=completed ")
    if (int(support["articulated_support_rows"]) != 0 or
            owner["resting_bed_contact_owner"] != "fem_outer_skin"):
        raise ValueError("The scene used another bed owner")
    executed = json.loads((case / "execution.json").read_bytes())
    simulated = float(timing["simulated_s"])
    if (executed["exit"] != 0 or not observations or
            not math.isclose(math.fsum(step["dt_s"] for step in observations), simulated,
                             rel_tol=1e-6, abs_tol=1e-15)):
        raise ValueError("Native coupled run did not complete")
    return dict(launch_sha256=digest(case / "launch.json"), rigid_sha256=digest(rigid),
                q_trace_sha256=digest(path), authored_stops=len(stops),
                observations=observations, bed_normal_N=float(support["mean_normal_force_N"]),
                rigid_bed_rows=0, static_maximum_acceleration=float(init["maximum_acceleration_residual"]),
                static_root_evaluated=int(init["floating_root_residual_evaluated"]),
                static_balanced=init["balanced"] == "true", simulated_s=simulated,
                body_wall_s=float(timing["wall_s"]), launch_to_exit_s=executed["wall_s"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-opening-reaction", type=int, action="append", default=[],
                        help="require this authored velocity coordinate to react at its opening stop")
    opts = parser.parse_args()
    report = measure(opts.case)
    opts.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for index in opts.require_opening_reaction:
        opening = report["observations"][0]
        witness = opening["stop_witnesses"].get(index)
        if opening["step"] != 1 or witness is None:
            raise SystemExit(f"Missing opening joint-stop impulse on coordinate {index}")
        at_lower = witness["before"] == witness["lower"] and witness["impulse"] > 0
        at_upper = witness["before"] == witness["upper"] and witness["impulse"] < 0
        if not (at_lower or at_upper):
            raise SystemExit(f"Impulse does not oppose the closed stop on coordinate {index}")
