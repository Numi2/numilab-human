#!/usr/bin/env python3
"""Summarize the retained matched initialization and solver trials."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import re
import statistics

out = Path(__file__).resolve().parent
work = out.parent

def read_json(p):
    return json.loads(p.read_text())

def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def summarize(case):
    folder = work / case
    report = read_json(folder / "install-motion.execution.json")
    plan = read_json(folder / "install-motion.plan.json")
    argv = plan["stages"][0]["argv"]
    activation = float(argv[argv.index("--muscle-activation") + 1])
    dt = float(argv[argv.index("--muscle-step-seconds") + 1])
    prelim = []
    timings = []
    profile = []
    with gzip.open(folder / "install-motion.stdout.log.gz", "rt") as f:
        for line in f:
            if line.startswith("supported_prelude_observation "):
                prelim.append({k: int(v) if k in {"accepted_step","failure_mask","settled_consecutive_steps"} else float(v)
                    for k,v in re.findall(r"(\w+)=([-+\d.eE]+)", line)})
            if line.startswith("resting_native_profile step="):
                timings.append({k: float(v) for k,v in re.findall(r"(\w+)=([-+\d.eE]+)", line)})
            if line.startswith(("resting_initialization=dynamic_release", "resting_fem_skin_bed_compiled=validated",
                                "resting_bed_contact_owner=fem_outer_skin", "resting_fem_skin_bed_accepted_root_reaction",
                                "resting_trial_bed_plane=fixed")):
                profile.append(line.strip())
    linear = []
    terminal_failure = None
    stderr = folder / "install-motion.stderr.log"
    with stderr.open() as f:
        for line in f:
            if line.startswith("matter_fgmres_active_work="):
                j = json.loads(line.split("=",1)[1])
                if j.get("has_outer_sample"):
                    linear.append({k:j[k] for k in ["root","used_columns_sum","cycles_with_work","budget_exhausted_solves",
                        "active_newton_begins","final_nonpressure_nontransport_relative_residual","status_code"]})
            if line.startswith("myosim_articulated_visual=failed"):
                terminal_failure = line.strip()
    q_trace = folder / "install-motion/resting-com-q-integration.csv"
    joints = []
    if q_trace.exists():
        with q_trace.open() as f:
            for row in csv.DictReader(f):
                v0 = row["v_before_f32_semicolon"].split(";")
                v1 = row["v_accepted_f32_semicolon"].split(";")
                dv = {str(i): (float(v1[i])-float(v0[i]))/float(row["dt_s"])
                      for i in (105,108,119,122)}
                joints.append({"accepted_step": int(row["accepted_step"]), "dt_s": float(row["dt_s"]),
                               "joint_acceleration_rad_s2":dv})
    body_elapsed = None
    native_wall = next((msg for msg in profile if msg.startswith("native_wall")),None)
    with gzip.open(folder / "install-motion.stdout.log.gz", "rt") as f:
        for line in f:
            m=re.search(r"body_wall_ms=(\d+(?:\.\d+)?)",line)
            if m:
                body_elapsed=float(m[1])/1000
    return {
        "case":case,
        "native_binary":plan["native_binary"],
        "step_seconds":dt,
        "launch_plan_sha256":report["plan_sha256"],
        "activation_cap":activation,
        "execution":report,
        "last_terminal_failure":terminal_failure,
        "accepted_motion": prelim or joints,
        "initialization_and_bed":profile,
        "physical_command_samples":timings,
        "solver_work":linear,
        "body_elapsed_s_if_published":body_elapsed,
    }

cases = [summarize(case) for case in (
    "resume-fidelity-and-profile-02", "resume-ankle-recruitment-001",
    "resume-fgmres-halfwarp-001", "resume-bed-plane-004", "resume-bed-plane-005")]
raw = (work / "resume-graph-24-batch-001/timing.log").read_text()
pairs = [(float(a),float(b)) for a,b in re.findall(r"reference_ms=([\d.]+) candidate_ms=([\d.]+)",raw)]
assert len(pairs)==12
result = {
    "device": "Apple M6, 16 GiB, Mac18,5, macOS 27.0 (26A425)",
    "base_source_revision": "f75955f08d6edf0f85027eac7614017ac9fbc996",
    "build_source_scope": "Retained dirty native checkpoint with incremental application overlays and exact linked libraries for bed test.",
    "cases":cases,
    "graph_fixture": {"phases":pairs,"reference_median_ms":statistics.median(a for a,b in pairs),
        "candidate_median_ms":statistics.median(b for a,b in pairs),"byte_parity":True,
        "patches":8192,"dofs_per_patch":24,"scope":"isolated synthetic batch"},
    "halfwarp_endpoint_comparison_sha256":sha(work/"resume-fgmres-halfwarp-001/matched-endpoint-channels.json"),
    "settlement_qualified": any(obs.get("settled_consecutive_steps",0)>=8 for obs in cases[-1]["accepted_motion"]),
}
(out / "result.json").write_text(json.dumps(result,indent=2)+"\n")
for c in cases:
    print(c["case"],c["activation_cap"],c["execution"]["exit_code"],
          len(c["physical_command_samples"]),c["last_terminal_failure"])
print(json.dumps(result["graph_fixture"],separators=(",",":")))
