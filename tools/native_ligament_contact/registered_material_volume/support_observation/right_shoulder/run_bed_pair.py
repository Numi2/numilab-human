"""Run the authored lower plane with identical contact observation and material ownership."""

import json
import os
import sys
from pathlib import Path

from compile_connected_joint import digest
from run_cavity_motion import execute_native


def run(parent, bed, out):
    c = json.loads((parent / "launch.json").read_bytes())
    src = json.loads((bed / "result.json").read_bytes())
    argv = c["argv"]
    env = c["environment"]
    pins = c["pins"]
    i = argv.index("--support-contact-payload") + 1
    if (
        digest(Path(argv[0])) != src["original_program"]
        or digest(Path(argv[i])) != src["original_bed"]
        or src["control"] != c["route_control"]
    ):
        raise ValueError("Incompatible bed control")
    for j, name, key in [
        (0, "numi-human-native", "new_program"),
        (i, Path(argv[i]).name, "new_bed"),
    ]:
        p = bed / name
        if digest(p) != src[key]:
            raise ValueError("Bed input changed")
        argv[j] = str(p)
        pins[str(p)] = src[key]
    argv[4] = str(out / "native-traces")
    for k, v in list(env.items()):
        if v.startswith(str(parent) + "/") and v not in pins:
            env[k] = str(out / Path(v).relative_to(parent))
    for v in set(argv + list(env.values())) & set(pins):
        if digest(Path(v)) != pins[v]:
            raise ValueError("Changed model input")
    out.mkdir()
    c["bed_pair"] = dict(
        control=digest(parent / "launch.json"), authored=digest(bed / "result.json")
    )
    (out / "launch.json").write_text(json.dumps(c, indent=2) + "\n")
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("NM_", "NUMI_"))}
    result = execute_native(
        argv, c["cwd"], clean | env, out, minimum_free_bytes=4 * 2**30, timeout_s=900
    )
    (out / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    run(*(Path(x).resolve() for x in sys.argv[1:]))
