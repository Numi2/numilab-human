"""Fail-closed receipt for native PTL impulse and cruciate-contact runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATTER = Path("/Users/home/numi-lab-cardiac-native-20260930")
FOLDER = ROOT / "Docs/media/cruciate-source-contact-20260930"
OUTPUT = FOLDER / "ptl-trajectory-native-receipt.json"
RUNS = (("one", 1, True, 0), ("two", 2, True, 0),
        ("eight", 8, True, 1), ("untouched", 1, False, 1))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("PTL trajectory audit: " + message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([a-z][a-z0-9_]+)=([^ ]+)", line))


def vector(value: str) -> tuple[float, float, float]:
    result = tuple(float(component) for component in value.split(","))
    require(len(result) == 3 and all(math.isfinite(x) for x in result),
            "nonfinite vector")
    return result


def run() -> dict:
    prior = json.loads((FOLDER / "ptl-momentum-one-step-receipt.json").read_text())
    binary = Path(prior["binary"]["path"])
    source = MATTER / "apps/numilab_human_myosim_visual_probe.mm"
    shader = MATTER / "matter/src/metal/numi_human.metalinc"
    for name, entry in prior["inputs"].items():
        require(sha(Path(entry["path"])) == entry["sha256"],
                f"pinned {name} input drifted")
    matter_commit = subprocess.check_output(
        ["git", "-C", str(MATTER), "rev-parse", "--short", "HEAD"],
        text=True).strip()
    require(matter_commit == "a334f99", "Matter implementation revision drifted")
    witness_path = FOLDER / "ptl-trajectory-contact-witness.json"
    witness = json.loads(witness_path.read_text())
    require(witness["source_contact_pair"] == "PCL_To_ACL" and
            witness["accepted_steps_before_rejection"] == 5 and
            witness["native_step_finish_strict_crossing"] and
            not witness["source_strict_crossing"] and
            not witness["candidate_strict_crossing"],
            "source-bound dynamic contact witness drifted")

    receipts: dict[str, dict] = {}
    for label, steps, candidate, exit_code in RUNS:
        log = FOLDER / f"ptl-trajectory-{label}-step.log"
        if label == "untouched":
            log = FOLDER / f"ptl-trajectory-{label}.log"
        text = log.read_text()
        argv = prior["run"]["argv"].copy()
        argv[4] = str(ROOT / f"Build/ptl-trajectory-20260930/final-gate-{label}")
        argv[argv.index("--muscle-step-count") + 1] = str(steps)
        if not candidate:
            argv.remove("--open-knee-cruciate-initialization-diagnostic")
        receipt = {"argv": argv, "requested_steps": steps,
                   "initialization": "local_acl_diagnostic" if candidate
                                     else "untouched_source",
                   "observed_exit_code": exit_code,
                   "log": str(log.relative_to(ROOT)),
                   "log_sha256": sha(log)}
        if exit_code == 0:
            account = next((line for line in text.splitlines()
                            if line.startswith("open_knee_ptl_force_accounting=")),
                           None)
            accepted = next((line for line in text.splitlines()
                             if line.startswith("open_knee_live_tissue_fem=accepted")),
                            None)
            terminal = next((line for line in text.splitlines()
                             if line.startswith("stand_terminal_state=")), None)
            require(account is not None and accepted is not None and
                    terminal is not None, f"{label} accepted evidence missing")
            a, b = fields(account), fields(accepted)
            end = json.loads(terminal[len("stand_terminal_state="):])
            relative = float(a["impulse_plus_momentum_relative_to_l1"])
            require(int(float(a["accepted_reaction_impulse_steps"])) == steps and
                    end["step_count"] == steps and relative <= 0.02 and
                    b["qualification_scope"] ==
                        "unqualified_local_acl_initialization" and
                    b["replay"] == "bitwise" and
                    b["rollback"] == "verified",
                    f"{label} PTL acceptance or evidence boundary drifted")
            receipt["ptl"] = {key: a[key] for key in (
                "route_terminal_force_xyz_n",
                "accepted_reaction_impulse_xyz_ns",
                "accepted_reaction_impulse_steps",
                "accepted_reaction_impulse_l1_ns",
                "impulse_plus_momentum_residual_ns",
                "impulse_plus_momentum_relative_to_l1",
                "continuum_momentum_rate_xyz_n")}
            receipt["qualification_scope"] = b["qualification_scope"]
            receipt["replay"] = b["replay"]
            receipt["rollback"] = b["rollback"]
        else:
            failure = re.search(
                r"human_joint_failure completed_steps=(\d+) "
                r"matter_status=(\d+) matter_failing_index=(\d+)", text)
            partial = next((line for line in text.splitlines()
                            if line.startswith("open_knee_ptl_partial_impulse=")),
                           None)
            require(failure is not None and partial is not None and
                    int(failure[1]) == (5 if label == "eight" else 0) and
                    int(failure[2]) == 6 and
                    int(float(fields(partial)["accepted_steps"])) ==
                        int(failure[1]),
                    f"{label} fail-closed contact or impulse count drifted")
            receipt["accepted_steps_before_contact_rejection"] = int(failure[1])
            receipt["matter_status"] = int(failure[2])
            receipt["partial_impulse"] = fields(partial)
        receipts[label] = receipt

    one = receipts["one"]["ptl"]
    two = receipts["two"]["ptl"]
    one_impulse = vector(one["accepted_reaction_impulse_xyz_ns"])
    two_rate = vector(two["continuum_momentum_rate_xyz_n"])
    missed_second = math.sqrt(sum(
        (one_impulse[axis] + two_rate[axis] * 2.0e-6) ** 2
        for axis in range(3))) / float(one["accepted_reaction_impulse_l1_ns"])
    require(missed_second > 0.02,
            "missing-second-step negative control would pass PTL gate")

    return {
        "schema": "numi.human.ptl-accepted-trajectory-native.v1",
        "matter_commit": matter_commit,
        "binary": {"path": str(binary), "sha256": sha(binary)},
        "matter_source_sha256": sha(source),
        "matter_shader_sha256": sha(shader),
        "inputs": prior["inputs"],
        "runs": receipts,
        "contact_witness_sha256": sha(witness_path),
        "negative_control_missing_second_step_relative_error": missed_second,
        "qualified": False,
        "boundary": "The accepted-step PTL gate closes one and two 1-microsecond "
                    "diagnostic steps only. Untouched source rejects step zero; "
                    "the local ACL initialization rejects attempted step six "
                    "at a newly crossing PCL/ACL contact. No sustained loaded "
                    "knee or whole-body anatomy qualification follows.",
    }


if __name__ == "__main__":
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"binary_sha256": result["binary"]["sha256"],
                      "accepted_steps": [
                          result["runs"][name]["requested_steps"]
                          for name in ("one", "two")],
                      "first_contact_failure_step":
                          result["runs"]["eight"]
                              ["accepted_steps_before_contact_rejection"]},
                     sort_keys=True))
