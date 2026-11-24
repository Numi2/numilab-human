"""Run the matched native scene with opt-in sparse observations, preserving its physics settings."""

import json
import os
import sys
from pathlib import Path

from compile_connected_joint import digest
from run_cavity_motion import execute_native


def run(case, build, out, steps=None):
    cfg = json.loads((case / "launch.json").read_bytes())
    spec = json.loads((build / "build.json").read_bytes())
    if spec["control"] != digest(case / "launch.json"):
        raise ValueError("Unmatched dense scene")
    args, env, pins = cfg["argv"], cfg["environment"], cfg["pins"]
    if env["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] != "1":
        raise ValueError("Different thoracic loading")
    out.mkdir()
    args[4] = str(out / "native-traces")
    for k, v in list(env.items()):
        if v.startswith(str(case) + "/") and v not in pins:
            env[k] = str(out / Path(v).relative_to(case))
    for file, key in [
        ("libmetalrobo.dylib", "library"),
        ("NumiMatter.metallib", "metal"),
    ]:
        p = build / file
        if digest(p) != spec[key]:
            raise ValueError("Unmatched instrumentation")
        pins[str(p)] = spec[key]
    env["DYLD_LIBRARY_PATH"] = str(build) + ":" + env["DYLD_LIBRARY_PATH"]
    env["NUMI_MATTER_METALLIB_OVERRIDE"] = str(build / "NumiMatter.metallib")
    env["NUMI_HUMAN_REGISTERED_FORCE_ROUTE_PREFIX"] = str(out / "contact")
    if steps is not None:
        if not 1 <= steps <= 4:
            raise ValueError("Invalid observation prefix")
        i = args.index("--steps") + 1
        args[i] = str(steps)
    for v in set(args + list(env.values())) & set(pins):
        if digest(Path(v)) != pins[v]:
            raise ValueError("Stale native input " + v)
    cfg.update(route_control=spec["control"], route_build=digest(build / "build.json"))
    (out / "launch.json").write_text(json.dumps(cfg, indent=2) + "\n")
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("NM_", "NUMI_"))}
    result = execute_native(
        args, cfg["cwd"], clean | env, out, minimum_free_bytes=4 * 2**30, timeout_s=900
    )
    (out / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    run(
        *(Path(v).resolve() for v in sys.argv[1:4]),
        int(sys.argv[4]) if len(sys.argv) > 4 else None,
    )
