"""Run one matched third-root replay, preserving predecessor outputs."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import time

from build_hand_trial import HERE, OUT, CONNECTED_OUT, WITNESS_OUT, CLEARANCE_OUT, digest

BASE = (HERE.parent / "loaded-initialization-023/"
        "same-source-parent-16-compensated-vascular-vascular-predictor-"
        "trace-third-root-reset-direction-bound-parallel-constitutive-screen/launch.json")
TRIAL = HERE / "hand-interface-replay"

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--connected", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--trace-fourth", action="store_true")
    mode.add_argument("--capture-three", action="store_true")
    mode.add_argument("--contact-rhs", action="store_true")
    mode.add_argument("--solid-witness", action="store_true")
    mode.add_argument("--clearance", action="store_true")
    parser.add_argument("--clearance-trace-ninth", action="store_true")
    options = parser.parse_args()
    if options.clearance_trace_ninth and not options.clearance:
        parser.error("--clearance-trace-ninth requires --clearance")
    if (options.trace_fourth or options.capture_three or options.contact_rhs or options.solid_witness or options.clearance) and not options.connected:
        parser.error("the extended trials require --connected")
    binary_dir = CLEARANCE_OUT if options.clearance else WITNESS_OUT if options.solid_witness else CONNECTED_OUT if options.connected else OUT
    trial = HERE / "connected-interface-replay" if options.connected else TRIAL
    if options.trace_fourth:
        trial = trial.with_name(trial.name + "-fourth")
    if options.capture_three:
        trial = trial.with_name(trial.name + "-capture-three")
    if options.contact_rhs:
        trial = trial.with_name(trial.name + "-contact-rhs")
    if options.solid_witness:
        trial = trial.with_name(trial.name + "-solid-witness")
    if options.clearance:
        trial = HERE / ("hand-clearance-ninth-root" if options.clearance_trace_ninth
                        else "hand-clearance-native-replay")
    launch = json.loads(BASE.read_text())
    for path, expected in launch["pins"].items():
        assert digest(path) == expected, path
    argv = launch["argv"]
    old_binary = argv[0]
    assert digest(old_binary) == launch["native_binary_sha256"]
    argv[0] = str(binary_dir / "numi-human-native")
    argv[4] = str(trial / "native")
    if options.capture_three:
        argv[argv.index("--muscle-step-count") + 1] = "3"
    if options.contact_rhs or options.solid_witness:
        argv[argv.index("--muscle-step-count") + 1] = "1"
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
    if options.clearance:
        environment["NUMI_HUMAN_HAND_CLEARANCE_TRIAL"] = "1"
    else:
        environment["NUMI_HUMAN_COARSE_HAND_IFACE_TRIAL"] = "1"
    if options.solid_witness or options.clearance:
        environment["NUMI_HUMAN_HAND_CONTACT_WITNESS_CSV"] = str(trial / "hand-solid.csv")
    if options.contact_rhs:
        environment["NM_FEM_CONTACT_RHS_WITNESS"] = "1"
    if options.trace_fourth or options.clearance_trace_ninth:
        traced_root = "8" if options.clearance_trace_ninth else "3"
        for key in ("NM_FEM_BACKTRACKING_TRACE_ROOT", "NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT",
                    "NM_VASCULAR_TRACE_ROOT"):
            environment[key] = traced_root
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
