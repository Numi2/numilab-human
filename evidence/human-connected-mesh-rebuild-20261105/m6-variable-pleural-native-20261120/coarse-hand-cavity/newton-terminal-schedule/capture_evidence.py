"""Retain the compact, reproducible result of the terminal-schedule pair."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def capture(source, stage):
    for name in ("README.md", "experiment.md", "preregistration.sha256", "build_trial.py",
                 "run_trial.sh", "capture_evidence.py", "full-parity.json", "staged-prefix.json",
                 "staged-work.json", "full-fields.json", "staged-fields.json", "ram-volume.log"):
        path = source / name
        if path.is_file():
            shutil.copyfile(path, stage / name)
    for name in ("app", "library", "shader"):
        for path in (source / name).rglob("*"):
            if path.name in ("identity.json", "commands.json") or path.suffix == ".patch":
                shutil.copyfile(path, stage / (name + "-" + path.name))
    for mode in ("full", "staged"):
        base = source / mode
        for name in ("launch.json", "execution.json", "native/resting-coupled.csv"):
            path = base / name
            if path.is_file():
                shutil.copyfile(path, stage / f"{mode}-{path.name}")
        prefixes = ("matter_solver_domain_norms=", "matter_fgmres_active_work=", "resting_common_failure ",
                    "myosim_articulated_visual=failed")
        with (base / "stderr.log").open() as stream, (stage / f"{mode}-solver.log").open("x") as out:
            out.writelines(line for line in stream if line.startswith(prefixes))
        prefixes = ("human_staged_solver_attempt", "human_staged_solver_rollback", "resting_integrated_body=",
                    "resting_fem_skin_bed_terminal_history ", "resting_fascia_fem_terminal_audit=",
                    "resting_FEM_terminal_audit=")
        with gzip.open(base / "stdout.log.gz", "rt") as stream, (stage / f"{mode}-terminal.log").open("x") as out:
            out.writelines(line for line in stream if line.startswith(prefixes))
    full = json.loads((source / "full-parity.json").read_text())
    prefix = json.loads((source / "staged-prefix.json").read_text())
    assessment = {
        "reference_matches_prior_coupled_trace": full["first_accepted_difference"] is None,
        "candidate_matches_reference_coupled_trace": prefix["first_accepted_difference"] is None,
        "candidate_accepted_roots": prefix["candidate"]["accepted_steps"],
        "candidate_zero_root_assistance_in_trace": prefix["candidate"]["zero_root_assistance"],
    }
    if (source / "staged-work.json").is_file():
        pair = json.loads((source / "staged-work.json").read_text())
        fields = json.loads((source / "staged-fields.json").read_text())
        assessment.update(
            staged_rollback_count=len(pair["candidate"].get("verified_rollbacks", [])),
            body_time_reduction=pair["body_time_reduction"],
            force_failures=len(fields["force_regions_over_1pct"]),
            strain_failures=len(fields["strain_metrics_over_1pct"]),
            unchanged_connectivity_and_mass=all(fields[key] for key in
                ("exact_element_connectivity", "reference_positions_match", "nodal_masses_match")),
        )
    with (stage / "assessment.json").open("x") as stream:
        json.dump(assessment, stream, indent=2, allow_nan=False)
        stream.write("\n")
    checksums = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                 for path in sorted(stage.iterdir()) if path.is_file()]
    (stage / "SHA256SUMS").write_text("".join(checksums))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("stage", type=Path)
    capture(**vars(parser.parse_args()))
