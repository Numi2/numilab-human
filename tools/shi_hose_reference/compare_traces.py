"""Compare already generated C++ source traces. This program never steps physics."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 10001:
        raise ValueError("Expected at least ten cycles sampled at one millisecond")
    rows = [{key: float(value) for key, value in row.items()} for row in rows]
    if any(not math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError("Nonfinite source trace")
    if rows[0]["time_s"] != 0 or rows[-1]["time_s"] < 10:
        raise ValueError("Source trace does not cover the requested initial transient")
    return rows


def family(name):
    if name.endswith("_volume_m3"):
        return "chamber_volume_m3"
    if name.endswith("_pressure_Pa"):
        return "pressure_Pa"
    if name.endswith("_outflow_m3_per_s"):
        return "flow_m3_per_s"
    if name.endswith("_elastance_Pa_per_m3"):
        return "elastance_Pa_per_m3"
    return None


def compare(a, b):
    if len(a) != len(b) or list(a[0]) != list(b[0]):
        raise ValueError("Trace schemas or sample counts differ")
    if any(x["time_s"] != y["time_s"] for x, y in zip(a, b)):
        raise ValueError("Trace sample times differ")
    metrics = []
    grouped = {}
    for key in a[0]:
        diffs = [abs(x[key] - y[key]) for x, y in zip(a, b)]
        index = max(range(len(diffs)), key=diffs.__getitem__)
        amplitude = max(abs(row[key]) for row in b)
        metric = {
            "name": key,
            "maximum_absolute_difference": diffs[index],
            "time_at_maximum_seconds": b[index]["time_s"],
            "maximum_difference_over_reference_amplitude": diffs[index] / max(amplitude, 1e-30),
            "root_mean_square_difference": math.sqrt(sum(x*x for x in diffs)/len(diffs)),
        }
        metrics.append(metric)
        group = family(key)
        if group and diffs[index] > grouped.get(group, {}).get("maximum_absolute_difference", -1):
            grouped[group] = metric
    return {"families": grouped, "columns": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_1e10", type=Path)
    parser.add_argument("trace_1e12", type=Path)
    parser.add_argument("trace_1e13", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    paths = [args.trace_1e10, args.trace_1e12, args.trace_1e13]
    rows = [read(path) for path in paths]
    coarse = compare(rows[0], rows[1])
    tight = compare(rows[1], rows[2])
    report = {
        "schema": "NumiHuman.CellML-reference-refinement.v1",
        "source_revision": "a679cdc2e97429fb5280af8132c119758626c1f2",
        "scope": "Pairwise numerical refinement, not a rigorous exact-solution error bound or biological calibration",
        "traces": [{"file": path.name, "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
        "samples": len(rows[0]),
        "coarse_1e10_vs_1e12": coarse,
        "tight_1e12_vs_1e13": tight,
        "refinement_decreases_pressure_flow_volume_maxima": all(
            tight["families"][key]["maximum_absolute_difference"] < coarse["families"][key]["maximum_absolute_difference"]
            for key in ["pressure_Pa", "flow_m3_per_s", "chamber_volume_m3"]),
        "initial_native_hydraulic_state_SI": [rows[2][0]["native_hydraulic_"+str(i)] for i in range(20)],
        "final_native_hydraulic_state_SI": [rows[2][-1]["native_hydraulic_"+str(i)] for i in range(20)],
        "final_cycle_observable_ranges": {
            key: {"minimum": min(row[key] for row in rows[2] if row["time_s"] >= rows[2][-1]["time_s"]-1),
                  "maximum": max(row[key] for row in rows[2] if row["time_s"] >= rows[2][-1]["time_s"]-1)}
            for key in rows[2][0] if family(key)},
    }
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"refinement_decreases": report["refinement_decreases_pressure_flow_volume_maxima"],
                      "tight_families": tight["families"]}, indent=2))
    if not report["refinement_decreases_pressure_flow_volume_maxima"]:
        raise SystemExit("Source reference refinement did not improve every physical family")


if __name__ == "__main__":
    main()
