"""Summarize the extended matched release and keep the support test explicit."""
import csv
import json
from pathlib import Path
import re

from compare import summarize, sha256

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "parallel-operators-128"
REFERENCE = ROOT / "parallel-operators-64"

def trace(directory):
    return (directory / "native/resting-coupled.csv").read_bytes().splitlines(keepends=True)

def record(prefix):
    with (RUN / "stdout.log").open() as log:
        matches = [line.rstrip() for line in log if line.startswith(prefix)]
    assert len(matches) == 1, (prefix, len(matches))
    return matches[0]

def field(line, name):
    return re.search(r"\b" + name + r"=(\S+)", line)[1]

def main():
    evidence = summarize(RUN, require_domain_gate=False)
    assert evidence["exit"] == 0 and evidence["roots_certified"] == 128
    with (RUN / "stderr.log").open() as stream:
        certificates = [json.loads(line.partition("=")[2]) for line in stream
                        if line.startswith("fem_newton_terminal_certificate=")]
    assert [row["root"] for row in certificates] == list(range(128))
    statuses = [status["code"] for row in certificates for status in row["status"]]
    assert len(statuses) == 128 and all(code == 0 for code in statuses)
    evidence["native_terminal_certificate_count"] = len(statuses)
    evidence["largest_terminal_relative_mechanical_residual"] = max(
        obj["nonlinear"][0] for row in certificates for obj in row["objects"])
    target_misses = {r["root"] for r in evidence["domain_target_misses"]}
    evidence["terminal_witnesses_for_tighter_target_misses"] = [
        row for row in certificates if row["root"] in target_misses
    ]
    baseline_rows = trace(REFERENCE)
    assert len(baseline_rows) == 65
    prefix_equal = trace(RUN)[:65] == baseline_rows
    assert prefix_equal
    assert sha256(RUN / "initial-skin-fem.csv") == sha256(REFERENCE / "initial-skin-fem.csv")
    assert sha256(RUN / "skin-bone-witness.csv") == sha256(REFERENCE / "skin-bone-witness.csv")
    shell = json.loads((ROOT / "motion-128-shell.json").read_text())
    reaction = record("resting_fem_skin_bed_accepted_root_reaction ")
    initial_force = record("resting_initialization=dynamic_release ")
    ligament = record("resting_fascia_fem_terminal_audit=")
    start = record("initial_skin_bone_solid_audit ")
    assert ligament.startswith("resting_fascia_fem_terminal_audit=passed ")
    assert field(ligament, "registered_ligament_force_rows") == "34"
    force = float(field(reaction, "mean_normal_force_N"))
    with (RUN / "native/resting-coupled.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    pressure = [float(r["pleural_pa"]) for r in rows]
    shell_counts = {
        frame: sum(shell[f"{frame}_pairs_by_layer"].values())
        for frame in ("initial", "terminal")
    }
    evidence.update(
        exact_64_root_trace_prefix=prefix_equal,
        pleural_pressure_range_pa=[min(pressure), max(pressure)],
        terminal_bed_force_N=force,
        terminal_bed_contact_count=int(field(reaction, "valid_contact_sample_count")),
        terminal_bed_weight_fraction=force / (97.13195069110952 * 9.81),
        initial_force_balance=initial_force,
        startup_skin_bone_witness=start,
        terminal_ligament_check=ligament,
        shell_crossing_counts=shell_counts,
        shell_scope=shell["scope"],
        launch_sha256=sha256(RUN / "launch.json"),
        native_trace_sha256=sha256(RUN / "native/resting-coupled.csv"),
    )
    (ROOT / "motion-128-summary.json").write_text(json.dumps(evidence, indent=2) + "\n")

if __name__ == "__main__":
    main()
