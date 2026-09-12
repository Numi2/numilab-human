#!/usr/bin/env python3
"""Verify captured partition evidence against current owners; optionally rebuild."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from numilab_human import cardiac_cavity_partition as partition
from numilab_human.physiology import canonical, read_json
from qualify_cardiac_partition import OWNERS, SOURCES, MODULES, digest

EVIDENCE = ROOT / "Docs/media/cardiac-partition-20260912"
SCOPE = {"source_union_preserved_exactly_in_rational_construction": True,
    "source_exclusive_regions_preserved_exactly_in_rational_construction": True,
    "both_emitted_candidates_have_disjoint_interiors": True,
    "anatomical_valve_interface_selected": False, "cardiac_phase_registered": False,
    "subject_or_body_frame_registered": False, "blood_tissue_mass_partition": False,
    "physiological_calibration": False, "standing_walking": False}


def require(ok, message):
    if not ok:
        raise ValueError("cardiac partition evidence: "+message)


def verify(evidence, *, recompute=False):
    execution = read_json(evidence / "execution.json")
    require(execution["schema"] == "HumanPack.cardiac-partition-execution.v1" and execution["status"] == "pass",
            "execution did not pass")
    require(execution["scope"] == "offline_exact_geometry_and_regressions" and execution["physical_stepping"] is False,
            "execution scope differs")
    require(execution["platform"]["system"] == "Darwin" and execution["platform"]["machine"] == "arm64" and
            execution["platform"]["chip"] == "Apple M4 Pro", "physical qualification host differs")
    expected_inputs = {name: digest(ROOT / name) for name in OWNERS+SOURCES}
    require(execution["inputs_before"] == execution["inputs_after"] == expected_inputs, "owner or source hash drift")
    require(set(execution["outputs"]) == {"partition.json", "compile.log", "tests.log"}, "output coverage differs")
    for name, expected in execution["outputs"].items():
        require(digest(evidence / name) == expected, "captured output hash drift: "+name)
    require([r["name"] for r in execution["runs"]] == ["compile", "tests"], "run coverage differs")
    for run in execution["runs"]:
        require(type(run["exit_code"]) is int and run["exit_code"] == 0 and run["log"] == run["name"]+".log" and
                run["log_sha256"] == execution["outputs"][run["log"]], "failed or unbound run")
    compile_command = execution["runs"][0]["command"]
    require(len(compile_command) == 3 and Path(compile_command[0]).is_absolute() and
            compile_command[0].endswith("/.numi/commands/human-circulation-partition") and
            compile_command[1] == "--output" and Path(compile_command[2]).is_absolute() and
            Path(compile_command[2]).name == "partition.json", "captured authoring command differs")
    require(execution["runs"][1]["command"] == [execution["platform"]["python_executable"], "-m", "unittest", "-v", *MODULES],
            "captured regression command differs")
    result = read_json(evidence / "partition.json")
    require(result["schema"] == partition.SCHEMA and set(result["qualification"]) == set(SCOPE) and
            all(result["qualification"][key] is value for key, value in SCOPE.items()), "qualification scope differs")
    require(result["selection"] is None and result["physical_stepping"] is False and
            result["native_payload_changed"] is False and result["hydraulic_parameters_changed"] is False and
            result["added_mechanical_mass_kg"] == 0, "unsupported physical promotion")
    require(set(result["candidates"]) == {name+"_priority" for name in partition.NAMES}, "candidate coverage differs")
    require(result["source_geometry"] == partition.extract_cavity_surfaces(), "source extraction drift")
    for candidate in result["candidates"].values():
        require(candidate["four_cavity_interiors_disjoint"] is True and candidate["biological_selection"] is False and
                candidate["mechanical_mass_assigned"] is False, "candidate scope differs")
        require(candidate["geometry_sha256"] == partition._sha({"regions": candidate["regions"],
                                                                 "shared_interface": candidate["shared_interface"]}), "candidate geometry hash drift")
    if recompute:
        rebuilt = partition.compile_partitions(progress=lambda text: print(text, file=sys.stderr, flush=True))
        require(canonical(result) == canonical(rebuilt), "recomputed artifact differs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.evidence, recompute=args.recompute)
        print("PASS: current source/owner hashes, captured execution, both geometric candidates; physiological and mass claims remain false")
        print("shared source volume mL:", result["shared_source_volume_ml"])
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("FAIL:", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
