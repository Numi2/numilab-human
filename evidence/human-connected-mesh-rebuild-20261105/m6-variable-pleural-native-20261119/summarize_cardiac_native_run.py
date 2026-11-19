"""Summarize variable-pressure motion from the native accepted-state traces."""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from fit_cardiac_volume_grid import hash_file
from verify_cardiac_tube_membership import exact_inside, read_tubes


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def field(rows, key):
    return np.asarray([float(row[key]) for row in rows])


def bounds(rows, keys):
    result = {}
    for k in keys:
        x = field(rows, k)
        assert np.isfinite(x).all(), k
        result[k] = [float(x.min()), float(x.max())]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    run = a.run
    n = run / "native"
    cfg = json.loads((run / "launch.json").read_text())
    execution = json.loads((run / "execution.json").read_text())
    assert execution["exit_code"] == 0
    assert cfg["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] == "1"
    step = int(execution["requested_steps"])
    receipt_path = n / "accepted-geometry" / f"step-{step}.receipt.json"
    receipt = json.loads(receipt_path.read_text())
    assert int(receipt["accepted_step"]) == step and receipt["physical_endpoint"] == "accepted"
    assert receipt["surface_audit_endpoint"] == "passed"
    p = read_csv(n / "resting-coupled.csv")
    s = read_csv(n / "resting-surface-audit.csv")
    m = read_csv(n / "resting-com-momentum-diagnostic.csv")
    assert int(p[-1]["step"]) == int(s[-1]["step"]) == int(m[-1]["accepted_step"]) == step
    for k in ("root_assistance_n", "root_assistance_nm", "external_blood_ml"):
        assert (field(p, k) == 0).all(), k
    for k in ("common_coordinate_solver_status", "functional_geometry_status", "nonfinite_skin_vertices",
              "mesh_nonfinite_area_triangles", "mesh_zero_area_triangles"):
        assert (field(s, k) == 0).all(), k
    boxes = (n / "common-field-domain-boxes-f32.bin").stat().st_size // 64
    tubes = read_tubes()
    coordinates = ("RA", "RV", "LA", "LV", "RA_material", "ventricular_material", "LA_material")
    sampled_tube_steps = []
    for row in s:
        domain = int(row["common_coordinate_domain_box"])
        if domain >= boxes:
            q = np.asarray([row["common_coordinate_" + k] for k in coordinates], dtype="<f4")
            assert exact_inside(q, tubes[domain-boxes]), row["step"]
            sampled_tube_steps.append((int(row["step"]), domain-boxes))
    rates = {}
    stdout = (run / "stdout.log").read_text()
    for tag in ("resting_integrated_throughput", "resting_integrated_body=completed"):
        line = next(row for row in stdout.splitlines() if row.startswith(tag+" "))
        rates[tag] = float(re.search(r"real_time_factor=(\S+)", line).group(1))
    timing = {"simulated_seconds": receipt["accepted_time_s"],
              "launch_to_exit_seconds": execution["wall_seconds"],
              "launch_to_exit_rtf": receipt["accepted_time_s"]/execution["wall_seconds"],
              "motion_rtf_including_native_presentation": rates["resting_integrated_throughput"],
              "body_run_rtf": rates["resting_integrated_body=completed"],
              "native_presentation_callbacks": stdout.count("resting_present_profile step=")}
    limits = bounds(p, ("pleural_pa", "alveolar_pa", "airflow_ml_s", "lung_volume_ml", "diaphragm_mm",
                         "rib_mm", "PaO2_mmhg", "PaCO2_mmhg", "blood_error_ml", "tidal_ml",
                         "oxygen_balance_error_stpd_ml", "co2_balance_error_stpd_ml"))
    assert limits["pleural_pa"][1] > limits["pleural_pa"][0]
    support = bounds(m, ("normal_force_last_physical_step_n", "com_z_m"))
    accounted = [row for row in m if row["unaccounted_delta_p_z_ns"]]
    if accounted:
        support.update(bounds(accounted, ("unaccounted_delta_p_z_ns",)))
    support["momentum_rows_with_residual"] = len(accounted)
    result = dict(device="Apple M6", run=str(run), variable_pleural_pressure=True,
        launch_sha256=hash_file(run/"launch.json"), terminal_receipt_sha256=hash_file(receipt_path),
        terminal_pack_sha256=receipt["pack_file_sha256"], timing=timing,
        complete_breaths=int(p[-1]["breaths"]), complete_filling_ejection_cycles=int(p[-1]["complete_filling_ejection_cycles"]),
        active_controller=bool(np.ptp(field(p,"diaphragm_excitation"))>0),
        cardiac_volume_residual_max=float(field(s,"common_coordinate_normalized_residual").max()),
        native_geometry_volume_error_max=float(field(s,"max_functional_volume_relative_error").max()),
        sampled_tube_steps=sampled_tube_steps, sampled_geometry_rows=len(s),
        ranges=limits, support=support, terminal_bed_normal_force_n=float(m[-1]["normal_force_last_physical_step_n"]),
        no_recorded_root_assistance=True, whole_body_crossing_count_measured_by_this_tool=False)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
