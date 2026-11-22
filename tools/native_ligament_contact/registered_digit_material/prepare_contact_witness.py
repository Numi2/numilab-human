"""Inspect only the actual deformable-contact minimum at the failed Newton iterate."""

import json
import sys
from pathlib import Path

from material_context import sha


def create(parent, failed, out):
    stop = json.loads((failed / "failure.json").read_text())
    logs = (failed / "stderr.log").read_text().splitlines()
    (run,) = [
        json.loads(s.split("=", 1)[1])
        for s in logs
        if s.startswith("fem_backtracking_trials=")
    ]
    if run["root"] != stop["segment_start"]:
        raise ValueError("Mismatched trace")
    source = json.loads((parent / "launch.json").read_text())
    env = source["environment"]
    if (
        env.get("NM_FEM_TILED_DEFORMABLE_LIMIT") != "1"
        or env.get("NM_FEM_DEFORMABLE_LIMIT_TRACE_MINIMA") != "1"
    ):
        raise ValueError("Expected existing tiled owner")
    if (
        "NM_FEM_CONTACT_TRACE_ROOT" in env
        or "NM_FEM_DEFORMABLE_LIMIT_TRACE_SLOT" in env
    ):
        raise ValueError("Conflicting full capture")
    env["NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT"] = str(run["root"])
    env["NM_FEM_CONTACT_TRACE_ITERATION"] = str(run["iteration"])
    out.mkdir(exist_ok=False)
    (out / "launch.json").write_text(json.dumps(source, indent=2) + "\n")
    (out / "source.json").write_text(
        json.dumps(
            {
                "parent_launch": sha(parent / "launch.json"),
                "failed_step": sha(failed / "failure.json"),
                "selected_object": run["object"],
                "root": run["root"],
                "iteration": run["iteration"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    create(*(Path(v).resolve() for v in sys.argv[1:]))
