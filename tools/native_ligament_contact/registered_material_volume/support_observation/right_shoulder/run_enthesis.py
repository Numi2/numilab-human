"""Install the recooked functional attachment in the matched full scene and retain its accepted trace."""

import json
import os
import sys
from pathlib import Path

from bind_registered_shoulder import bind
from compile_connected_joint import digest
from run_cavity_motion import execute_native


def run(control, authored, out, steps=None):
    env0 = json.loads((control / "launch.json").read_bytes())
    c = bind(env0, authored)
    a = c["argv"]
    env = c["environment"]
    pins = c["pins"]
    if steps is not None:
        if not 1 <= steps <= 4 or a.count("--muscle-step-count") != 1:
            raise ValueError("Invalid diagnostic prefix")
        old_count = a[a.index("--muscle-step-count") + 1]
        a[a.index("--muscle-step-count") + 1] = str(steps)
        for name in (
            "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS",
            "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP",
        ):
            if name in env:
                if env[name] != old_count:
                    raise ValueError("Only a terminal capture can be retimed")
                env[name] = str(steps)
    if steps is not None and steps > 1:
        for key in (
            "NM_FEM_NEWTON_TRACE_ROOT",
            "NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT",
            "NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT",
            "NM_FEM_BACKTRACKING_TRACE_ROOT",
        ):
            if key in env:
                env[key] = str(steps - 1)
    out.mkdir()
    a[4] = str(out / "native-traces")
    for k, v in list(env.items()):
        if v.startswith(str(control) + "/") and v not in pins:
            env[k] = str(out / Path(v).relative_to(control))
    for v in set(a + list(env.values())) & set(pins):
        if digest(Path(v)) != pins[v]:
            raise ValueError("Stale native input " + Path(v).name)
    c["enthesis_control"] = digest(control / "launch.json")
    (out / "launch.json").write_text(json.dumps(c, indent=2) + "\n")
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("NM_", "NUMI_"))}
    result = execute_native(
        a, c["cwd"], clean | env, out, timeout_s=900, minimum_free_bytes=4 * 2**30
    )
    (out / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    run(
        *(Path(s).resolve() for s in sys.argv[1:4]),
        int(sys.argv[4]) if len(sys.argv) > 4 else None,
    )
