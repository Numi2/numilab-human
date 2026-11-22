"""Compare the first rejected driver trial with its physically identical control."""

from pathlib import Path
import csv, json, sys
import numpy as np
from certify_heart import sha


def measure(control, case, output):
    def read(p):
        with (p / "resting-coupled.csv").open() as f:
            rows = list(csv.DictReader(f))
        return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}

    c, t = map(read, (control, case))
    l0, l1 = [json.loads((p / "launch.json").read_text()) for p in (control, case)]
    argv = l1["argv"][:]
    i = argv.index("--resting-drive-intervention")
    dose = argv[i + 1 : i + 4]
    del argv[i : i + 4]
    argv[4] = l0["argv"][4]
    if (
        set(c) != set(t)
        or argv != l0["argv"]
        or dose != ["15", "35", "0.6"]
        or l0["pins"] != l1["pins"]
    ):
        raise ValueError("Unmatched control")
    env0, env1 = [dict(l["environment"]) for l in (l0, l1)]
    for env in (env0, env1):
        env.pop("NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT")
    if env0 != env1 or env0["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] != "1":
        raise ValueError("Configuration mismatch")
    clock = t["time_s"]
    n = len(clock)
    early = clock < 15.0
    if not np.array_equal(clock, c["time_s"][:n]) or not all(
        np.isfinite(x).all() for x in t.values()
    ):
        raise ValueError("Unmatched timestamps")
    parity = all(np.array_equal(c[k][:n][early], t[k][early]) for k in t)
    f = json.loads((case / "failure.json").read_text())
    latch = f["first_failure_latch"]
    idx = latch["step"]
    rr = json.loads((case / "execution.json").read_text())
    assert (
        parity
        and latch["common_status"][0] == 5
        and rr["native_exit_code"]
        and not rr["watchdog"]
    )
    assert (
        f["respiratory_accepted_status"][0] == idx
        and f["accepted_common_coordinates"] == f["candidate_common_coordinates"]
    )
    assert latch["brain_candidate_metadata"][1] > latch["brain_accepted_metadata"][1]
    measures = {}
    for k in (
        "inspired_volume_accum_ml",
        "diaphragm_mm",
        "pleural_pa",
        "PaO2_mmhg",
        "PaCO2_mmhg",
        "lung_volume_ml",
        "breaths",
        "complete_filling_ejection_cycles",
    ):
        measures[k] = dict(
            control=float(c[k][n - 1]),
            intervention=float(t[k][-1]),
            difference=float(t[k][-1] - c[k][n - 1]),
        )
    res = dict(
        matched_pre_dose_columns=len(t),
        matched_pre_dose_samples=int(early.sum()),
        trace_through_s=clock[-1],
        rejected_step=idx + 1,
        last_accepted_step=idx,
        accepted_s=idx * 0.002,
        wall_s=rr["wall_seconds"],
        first_latched_status=latch["common_status"],
        accepted_brain_time_us=latch["brain_accepted_metadata"][1],
        uncommitted_brain_time_us=latch["brain_candidate_metadata"][1],
        measurements_at_last_common_sample=measures,
        control_trace_sha256=sha(control / "resting-coupled.csv"),
        treatment_trace_sha256=sha(case / "resting-coupled.csv"),
        failure_sha256=sha(case / "failure.json"),
        completed_intervention=False,
    )
    output.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    measure(*(Path(a).resolve() for a in sys.argv[1:]))
