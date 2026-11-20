"""Run one matched third-root replay, preserving predecessor outputs."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import time

from build_hand_trial import HERE, OUT, CONNECTED_OUT, digest

BASE = (HERE.parent / "loaded-initialization-023/"
        "same-source-parent-16-compensated-vascular-vascular-predictor-"
        "trace-third-root-reset-direction-bound-parallel-constitutive-screen/launch.json")
TRIAL = HERE / "hand-interface-replay"

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--connected", action="store_true")
    parser.add_argument("--trace-fourth", action="store_true")
    options = parser.parse_args()
    if options.trace_fourth and not options.connected:
        parser.error("--trace-fourth requires --connected")
    binary_dir = CONNECTED_OUT if options.connected else OUT
    trial = HERE / "connected-interface-replay" if options.connected else TRIAL
    if options.trace_fourth:
        trial = trial.with_name(trial.name + "-fourth")
    launch = json.loads(BASE.read_text())
    for path, expected in launch["pins"].items():
        assert digest(path) == expected, path
    argv = launch["argv"]
    old_binary = argv[0]
    assert digest(old_binary) == launch["native_binary_sha256"]
    argv[0] = str(binary_dir / "numi-human-native")
    argv[4] = str(trial / "native")
    launch["pins"].pop(old_binary)
    launch["pins"][argv[0]] = digest(argv[0])
    launch["native_binary_sha256"] = launch["pins"][argv[0]]
    launch["comparison_launch_sha256"] = digest(BASE)
    environment = launch["environment"]
    for key, name in {
        "NUMI_HUMAN_SKIN_REFERENCE_DUMP": "registered-skin-native.csv",
        "NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP": "registered-skin-poses.json",
        "NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP": "tissue-mass.txt",
    }.items():
        assert key in environment
        environment[key] = str(trial / name)
    environment["NUMI_HUMAN_COARSE_HAND_IFACE_TRIAL"] = "1"
    if options.trace_fourth:
        for key in ("NM_FEM_BACKTRACKING_TRACE_ROOT", "NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT",
                    "NM_VASCULAR_TRACE_ROOT"):
            environment[key] = "3"
    trial.mkdir()
    (trial / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NM_", "NUMI_"))}
    env.update(environment)
    start = time.monotonic()
    with (trial / "stdout.log").open("xb") as out, (trial / "stderr.log").open("xb") as err:
        try:
            result = subprocess.run(argv, env=env, cwd=launch["cwd"], check=False,
                                    timeout=540, stdout=out, stderr=err)
            receipt = {"exit": result.returncode, "stop": "exit"}
        except subprocess.TimeoutExpired:
            receipt = {"exit": 124, "stop": "timeout"}
    receipt["wall_seconds"] = time.monotonic() - start
    (trial / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)

if __name__ == "__main__":
    main()
