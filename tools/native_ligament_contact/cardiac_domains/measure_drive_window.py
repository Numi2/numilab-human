"""Measure respiratory and gas response on matched native timestamps."""

import csv
import json
import sys
from pathlib import Path
import numpy as np
from certify_heart import sha


def trial(control, treatment, out):
    def read(case):
        launch = json.loads((case / "launch.json").read_text())
        run = json.loads((case / "execution.json").read_text())
        terminal = json.loads((case / "terminal.receipt.json").read_text())
        if (
            run["native_exit_code"]
            or run["watchdog"]
            or terminal["accepted_step"] != 30000
            or terminal["physical_endpoint"] != "accepted"
            or launch["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] != "1"
        ):
            raise ValueError("Native run did not reach the accepted physical endpoint")
        with (case / "resting-coupled.csv").open() as f:
            rows = list(csv.DictReader(f))
        data = {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}
        if not all(np.isfinite(v).all() for v in data.values()):
            raise ValueError("Nonfinite native trace")
        return data, launch, run, terminal

    c, l0, r0, s0 = read(control)
    t, l1, r1, s1 = read(treatment)
    keys = list(c)
    argv = l1["argv"][:]
    i = argv.index("--resting-drive-intervention")
    if argv[i + 1 : i + 4] != ["15", "35", "0.6"]:
        raise ValueError("Intervention differs")
    del argv[i : i + 4]
    argv[4] = l0["argv"][4]
    if argv != l0["argv"] or set(keys) != set(t) or l0["pins"] != l1["pins"]:
        raise ValueError("Unmatched source or output columns")
    # Only receiver paths may change. Neither physiological nor controller flags can differ.
    e0 = dict(l0["environment"])
    e1 = dict(l1["environment"])
    for k in e0:
        if (
            k in e1
            and e0[k] != e1[k]
            and k == "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"
        ):
            e1[k] = e0[k]
    if e1 != e0:
        raise ValueError("Native configurations differ")
    time = c["time_s"]
    if not np.array_equal(t["time_s"], time) or not np.all(np.diff(time) > 0):
        raise ValueError("Native clocks differ")
    same = all(np.array_equal(c[k][time < 15], t[k][time < 15]) for k in keys)

    def change(data, key, start, stop):
        return float(
            np.interp(stop, time, data[key]) - np.interp(start, time, data[key])
        )

    def avg(d, a, b):
        inside = (time > a) & (time < b)
        timeline = np.r_[a, time[inside], b]
        return float(np.trapezoid(np.interp(timeline, time, d), timeline) / (b - a))

    vi = [change(d, "inspired_volume_accum_ml", 15, 35) for d in (c, t)]
    gas = {}
    for k in ("PaO2_mmhg", "PaCO2_mmhg"):
        delta = t[k] - c[k]
        last = avg(delta, 30, 35)
        late = avg(delta, 55, 60)
        gas[k] = dict(
            end_mean_difference=last,
            recovery_mean_difference=late,
            peak_absolute_difference=float(np.max(np.abs(delta))),
            halved=abs(late) <= abs(last) * 0.5,
        )
    mechanics = {}
    for k in ("diaphragm_mm", "airflow_ml_s", "pleural_pa", "lung_volume_ml"):
        keep = (time >= 15) & (time < 35)
        mechanics[k] = [
            dict(min=float(d[k][keep].min()), max=float(d[k][keep].max()))
            for d in (c, t)
        ]
    checks = dict(
        matches_pre_intervention=same,
        ventilation_reduced_five_percent=vi[1] < 0.95 * vi[0],
        gas_directions=gas["PaO2_mmhg"]["end_mean_difference"] < 0
        and gas["PaCO2_mmhg"]["end_mean_difference"] > 0,
        both_gas_differences_halved=all(v["halved"] for v in gas.values()),
    )
    values = {}
    for k in (
        "breaths",
        "complete_filling_ejection_cycles",
        "blood_error_ml",
        "oxygen_balance_error_stpd_ml",
        "co2_balance_error_stpd_ml",
        "aortic_ejected_ml",
        "pulmonary_ejected_ml",
    ):
        values[k] = dict(terminal=float(t[k][-1]), max_abs=float(np.max(np.abs(t[k]))))
    data = dict(
        checks=checks,
        control_run_sha256=sha(control / "launch.json"),
        treatment_run_sha256=sha(treatment / "launch.json"),
        source_binary=sha(Path(argv[0])),
        accepted_s=s1["accepted_time_s"],
        samples=len(time),
        inspired_ml=vi,
        inspired_volume_fractional_change=vi[1] / vi[0] - 1,
        gas=gas,
        mechanics=mechanics,
        physiology=values,
        treatment_wall_s=r1["wall_seconds"],
        treatment_end_to_end_factor=s1["accepted_time_s"] / r1["wall_seconds"],
        control_trace=sha(control / "resting-coupled.csv"),
        treatment_trace=sha(treatment / "resting-coupled.csv"),
    )
    out.write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: data[k]
                for k in ("checks", "inspired_ml", "gas", "treatment_end_to_end_factor")
            }
        )
    )


if __name__ == "__main__":
    trial(*(Path(p).resolve() for p in sys.argv[1:]))
