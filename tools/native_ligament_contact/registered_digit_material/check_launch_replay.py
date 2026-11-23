"""Verify the public entrypoint repeats the qualified physical and geometric endpoint."""

from pathlib import Path
import re, sys, json
from registered_forearms import sha


def compare(full, repeat):
    def read(directory, name):
        return json.loads((directory / name).read_text())

    a = read(full, "terminal.receipt.json")
    b = read(repeat, "terminal.receipt.json")
    ex = read(repeat, "execution.json")
    x = read(full, "launch.json")
    y = read(repeat, "launch.json")
    args = lambda j: [v for i, v in enumerate(j["argv"]) if i != 4]
    if args(x) != args(y) or x["pins"] != y["pins"] or x["cwd"] != y["cwd"]:
        raise ValueError("Physical inputs differ")
    ignored = {
        "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS",
        "NUMI_HUMAN_BED_QUADRATURE_CAPTURE",
        "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT",
    }
    if {k: v for k, v in x["environment"].items() if k not in ignored} != {
        k: v for k, v in y["environment"].items() if k not in ignored
    }:
        raise ValueError("Different physical settings")
    keys = (
        "accepted_step",
        "accepted_time_s",
        "physical_endpoint",
        "surface_audit_endpoint",
        "accepted_body_state_sha256",
        "accepted_respiration_state_sha256",
        "pack_file_sha256",
        "base_pack_file_sha256",
    )
    matches = {k: a[k] == b[k] for k in keys}
    names = (
        "resting-coupled.csv",
        "resting-com-support-impulses.csv",
        "resting-surface-audit.csv",
        "resting-support-selected-skin-witness.csv",
    )
    files = {f: sha(full / f) == sha(repeat / f) for f in names}
    if (
        ex["exit_code"]
        or b["accepted_step"] != 8000
        or b["physical_endpoint"] != "accepted"
        or not all(matches.values())
        or not all(files.values())
    ):
        raise ValueError(f"Native replay diverged: {matches}, {files}")
    log = (repeat / "stdout.log").read_text()
    measures = {}
    for n in ("resting_integrated_body=completed", "resting_integrated_throughput"):
        (line,) = re.findall("^" + n + " .*$", log, re.M)
        v = dict(re.findall(r"(\w+)=([^ ]+)", line))
        measures[n] = dict(
            simulated_s=float(v["simulated_s"]),
            wall_s=float(v["wall_s"]),
            factor=float(v["real_time_factor"]),
        )
    return dict(
        repeated_native_fields=matches,
        repeated_trace_digests=files,
        seconds=b["accepted_time_s"],
        accepted_steps=b["accepted_step"],
        new_launch_sha256=sha(repeat / "launch.json"),
        terminal_pack=b["pack_file_sha256"],
        native_measures=measures,
        cold_wall_seconds=ex["cold_wall_s"],
        cold_realtime=b["accepted_time_s"] / ex["cold_wall_s"],
    )


if __name__ == "__main__":
    print(json.dumps(compare(*(Path(p) for p in sys.argv[1:])), indent=2))
