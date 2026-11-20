"""Compare accepted clocks, mass, ligament inventory and release behavior."""
import csv
import json
import math
import re

from build_hand_trial import HERE, digest
from run_hand_trial import BASE

CASE_NAMES = {"control": BASE.parent,
              "one_node": HERE / "hand-interface-replay",
              "connected": HERE / "connected-interface-replay"}


def numbers(row, key):
    m = re.search(r"(?<![A-Za-z0-9_])" + key + r"=([-+.0-9eE]+)(?: |$)", row)
    if m is None:
        raise ValueError(f"missing {key} in {row}")
    return float(m.group(1))


def compare(path):
    stdout = (path / "stdout.log").read_text().splitlines()
    traces = path / "native/resting-coupled.csv"
    with traces.open() as trace:
        accepted = list(csv.DictReader(trace))
    mass = next(row for row in stdout if row.startswith("whole_body_tissue_mass_reference="))
    runtime = next(row for row in stdout if row.startswith("resting_tissue_shared_runtime=bound "))
    skin = next(row for row in stdout if row.startswith("resting_skin_fat_whole_body_preparation=accepted "))
    pose = next(row for row in stdout if row.startswith("resting_initialization=dynamic_release "))
    physiology = {
        key: [float(row[key]) for row in accepted]
        for key in ("time_s", "pleural_pa", "normal_impulse_ns", "root_assistance_n", "root_assistance_nm")
    }
    result = {
        "source_launch_sha256": digest(path / "launch.json"),
        "raw_stdout_sha256": digest(path / "stdout.log"),
        "accepted_csv_sha256": digest(traces),
        "accepted_steps": len(accepted),
        "publication_trace": physiology,
        "whole_body_mass_kg": numbers(mass, "original_total_kg"),
        "skin_mass_debit_kg": numbers(skin, "skin_mass_debit_kg"),
        "fat_mass_debit_kg": numbers(skin, "fat_mass_debit_kg"),
        "cooked_tissue_mass_kg": numbers(mass, "cooked_tissue_kg"),
        "fem_uploaded_mass_kg": numbers(runtime, "FEM_mass_kg"),
        "skin_fem_nodes": int(numbers(skin, "shared_fem_nodes")),
        "skin_fem_tets": int(numbers(skin, "shared_fem_tets")),
        "native_registered_ligament_rows": int(numbers(runtime, "registered_ligament_force_rows")),
        "release_max_acceleration_residual": numbers(pose, "maximum_acceleration_residual"),
        "release_balance": " balanced=true" in pose,
        "release_support_newtons": numbers(pose, "support_force_n"),
    }
    wall = json.loads((path / "execution.json").read_text())["wall_seconds"]
    result["process_wall_seconds"] = wall
    result["process_realtime_ratio"] = (physiology["time_s"][-1] if accepted else 0.) / wall
    return result


def main():
    import sys

    names = sys.argv[1:] or list(CASE_NAMES)
    assert names[0] == "control"
    results = {name: compare(CASE_NAMES[name]) for name in names}
    for result in results.values():
        for key in ("whole_body_mass_kg","skin_mass_debit_kg","fat_mass_debit_kg",
                    "skin_fem_nodes","skin_fem_tets",
                    "native_registered_ligament_rows"):
            assert result[key] == results["control"][key], key
        assert math.isclose(result["cooked_tissue_mass_kg"], results["control"]["cooked_tissue_mass_kg"],
                            rel_tol=0., abs_tol=1e-12)
        result["fem_upload_mass_change_kg"] = result["fem_uploaded_mass_kg"] - results["control"]["fem_uploaded_mass_kg"]
        assert all(x == 0. for key in ("root_assistance_n", "root_assistance_nm")
                   for x in result["publication_trace"][key])
    print(json.dumps(results, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
