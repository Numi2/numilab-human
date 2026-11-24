"""Install the recooked functional attachment in the matched full scene and retain its accepted trace."""

import json
import os
import sys
from pathlib import Path

from bind_registered_shoulder import bind
from compile_connected_joint import digest
from run_cavity_motion import execute_native


def run(control, authored, out):
    env0 = json.loads((control / "launch.json").read_bytes())
    c = bind(env0, authored)
    a = c["argv"]
    env = c["environment"]
    pins = c["pins"]
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
    run(*(Path(s).resolve() for s in sys.argv[1:]))
