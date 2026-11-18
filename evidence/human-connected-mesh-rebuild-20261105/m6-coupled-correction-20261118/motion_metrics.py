"""Summarize accepted motion without counting step zero or failed trials."""
import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean


def trace(run):
    def rows(path):
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    state = rows(run / "native" / "resting-coupled.csv")
    terminal = {}
    measured = {}
    for line in (run / "stdout.log").read_text().splitlines():
        if line.startswith("stand_terminal_state="):
            terminal = json.loads(line.split("=",1)[1])
        elif line.startswith("resting_integrated_throughput "):
            measured = dict(re.findall(r"\b(\w+)=([^ ]+)",line))
    manifest = json.loads((run / "launch.json").read_text())
    outcome = json.loads((run / "execution.json").read_text())
    result = dict(outcome=outcome, config=manifest["mode"])
    if state:
        last = state[-1]
        assert not any(float(step[key]) != 0 for step in state for key in (
            "root_assistance_n", "root_assistance_nm"
        ))
        for key in ("step","time_s","breaths","complete_filling_ejection_cycles",
                    "aortic_ejected_ml","pulmonary_ejected_ml","blood_ml","PaO2_mmhg",
                    "PaCO2_mmhg","tidal_ml","last_lv_stroke_ml"):
            result[key] = float(last[key])
        if terminal:
            assert not terminal["root_assistance"]
            assert terminal["step_count"] == int(last["step"])
            result["final_q"] = terminal["q"]
        for key in ("blood_error_ml","oxygen_balance_error_stpd_ml","co2_balance_error_stpd_ml"):
            result[f"max_abs_{key}"] = max(abs(float(step[key])) for step in state)
    support_path = run / "native" / "resting-com-momentum-diagnostic.csv"
    if state and support_path.is_file():
        support = rows(support_path)
        end_time = float(state[-1]["time_s"])
        late = [row for row in support if float(row["time_s"]) > end_time - 2]
        if late:
            result["last_2s_bed_load_n"] = mean(float(row["normal_force_last_physical_step_n"]) for row in late)
            result["reference_weight_n"] = float(late[-1]["total_body_mass_kg"]) * 9.81
    surface_path = run / "native" / "resting-surface-audit.csv"
    if surface_path.is_file():
        surface_rows = rows(surface_path)
        result["surface_samples"] = len(surface_rows)
        for field in ("mesh_zero_area_triangles","mesh_nonfinite_area_triangles",
                      "vertices_below_1mm","nonfinite_skin_vertices"):
            result["max_"+field] = max((int(step[field]) for step in surface_rows), default=0)
    if measured:
        result["viewer_inclusive_rtf"] = float(measured["real_time_factor"])
        result["timing_includes"] = measured["includes"]
        result["timing_excludes"] = measured["excludes"]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    a = p.parse_args()
    print(json.dumps({r.name:trace(r) for r in a.runs},indent=2))
