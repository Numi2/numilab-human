"""Report matched native motion and support with declared scope and units."""
import csv
import json
import math
from pathlib import Path
import re
from statistics import fmean
import sys


def read_table(directory, name):
    with (directory / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit(root, native):
    execution = json.loads((root / "execution.json").read_text())
    trace = read_table(native, "resting-coupled.csv")
    support = read_table(native, "resting-com-momentum-diagnostic.csv")
    geometry = read_table(native, "resting-surface-audit.csv")
    steps = [int(row["step"]) for row in trace]
    assert all(a < b for a,b in zip(steps, steps[1:]))
    assert [int(row["accepted_step"]) for row in support] == steps
    times = [float(row["time_s"]) for row in trace]
    pressure = [float(row["pleural_pa"]) for row in trace]
    assert all(math.isfinite(p) for p in pressure)
    st = support[-1]
    mass = float(st["total_body_mass_kg"])
    terminal = trace[-1]
    load = [row for row in support if float(row["time_s"]) > times[-1] - 2]
    bed_n = fmean(float(row["normal_force_last_physical_step_n"]) for row in load)
    bed_impulse_n = sum(float(row["world_external_support_impulse_z_ns"]) for row in load)
    # The support vector samples the last physical root. Gravity is
    # integrated over the entire reporting segment, so normalize to one root.
    gravity_impulse_n = -sum(float(row["gravity_impulse_z_ns"]) / int(row["segment_steps"]) for row in load)
    speed = math.hypot(*(float(st[f"com_v{k}_m_s"]) for k in "xyz"))
    source = (root / "stdout.log").read_text(errors="replace")
    calibration = re.search(r"resting_mass_calibration [^\n]+", source)
    timing = re.search(r"accepted_start_step=\d+ accepted_end_step=\d+ [^\n]+real_time_factor=(\S+)", source)
    if timing is None and (root / "timing.json").is_file():
        existing = json.loads((root / "timing.json").read_text())
        timing = re.search(r"real_time_factor=(\S+)", existing["integrated_motion"])
    presentation = re.findall(r"resting_present_profile.*?total_present_wall_ms=(\S+)", source)
    mode = sorted(set(row["geometry_mode"] for row in geometry))
    return {
        "scope": "Released articulated native scene, thirty-two condensed bed-contact sections, reported bed and mesh validity samples only",
        "native_exit_code": execution.get("native_exit_code", execution.get("exit_code")),
        "accepted_samples": len(trace),
        "final_accepted_step": steps[-1],
        "final_accepted_time_s": times[-1],
        "wall_seconds": execution["wall_seconds"],
        "launch_to_exit_rtf": times[-1] / execution["wall_seconds"],
        "viewer_inclusive_motion_rtf": float(timing.group(1)) if timing else None,
        "presentation_callbacks": len(presentation),
        "body_mass_kg": mass,
        "mass_calibration": calibration.group(0) if calibration else None,
        "late_two_seconds_last_step_support_force_mean_n": bed_n,
        "late_two_seconds_sampled_impulse_to_weight_ratio": bed_impulse_n / gravity_impulse_n,
        "final_center_of_mass_speed_m_s": speed,
        "max_root_force_n": max(abs(float(row["root_assistance_n"])) for row in trace),
        "max_root_torque_nm": max(abs(float(row["root_assistance_nm"])) for row in trace),
        "max_abs_mass_residual_kg": max(abs(float(row["body_mass_residual_kg"])) for row in trace),
        "pleural_pa_range": [min(pressure), max(pressure)],
        "breaths": int(terminal["breaths"]),
        "filling_ejection_cycles": int(terminal["complete_filling_ejection_cycles"]),
        "terminal_o2_mmhg": float(terminal["PaO2_mmhg"]),
        "terminal_co2_mmhg": float(terminal["PaCO2_mmhg"]),
        "surface_samples": len(geometry),
        "min_skin_to_bed_gap_m": min(float(row["min_skin_bed_gap_m"]) for row in geometry),
        "max_vertices_below_1mm": max(int(row["vertices_below_1mm"]) for row in geometry),
        "max_invalid_visual_skin_vertices": max(int(row["nonfinite_skin_vertices"]) for row in geometry),
        "max_invalid_visual_triangles": max(int(row["mesh_nonfinite_area_triangles"]) +
                                           int(row["mesh_zero_area_triangles"]) for row in geometry),
        "max_functional_volume_relative_error": max(float(row["max_functional_volume_relative_error"])
                                                    for row in geometry),
        "geometry_mode": mode,
    }


if __name__ == "__main__":
    control, candidate = map(Path, sys.argv[1:])
    print(json.dumps({"control": audit(control, control / "native"),
                      "candidate": audit(candidate, candidate)}, indent=2))
