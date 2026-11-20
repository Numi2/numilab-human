"""Replay the pinned coarse Human with the optional complete bank audit."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import time

from build_hand_trial import HERE, digest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    assert args.steps > 0
    binary = args.build.resolve() / "numi-human-native"
    output = args.output.resolve()
    baseline = HERE / "hand-clearance-native-replay/launch.json"
    launch = json.loads(baseline.read_text())
    for asset, sha in launch["pins"].items():
        assert digest(asset) == sha, asset
    argv = launch["argv"]
    del launch["pins"][argv[0]]
    argv[0] = str(binary)
    argv[4] = str(output / "native")
    argv[argv.index("--muscle-step-count") + 1] = str(args.steps)
    launch["pins"][str(binary)] = digest(binary)
    launch["native_binary_sha256"] = launch["pins"][str(binary)]
    launch["comparison_launch_sha256"] = digest(baseline)
    env = launch["environment"]
    del env["NUMI_HUMAN_HAND_CONTACT_WITNESS_CSV"]
    paths = {
        "NUMI_HUMAN_SKIN_REFERENCE_DUMP": "registered-skin-native.csv",
        "NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP": "registered-skin-poses.json",
        "NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP": "tissue-mass.txt",
        "NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV": "initial-skin-fem.csv",
    }
    if args.audit:
        paths["NUMI_HUMAN_BATCH_CONTACT_WITNESS_CSV"] = "skin-bone-witness.csv"
    env.update({key: str(output / name) for key, name in paths.items()})
    output.mkdir()
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    run_env = {k: v for k, v in os.environ.items() if not k.startswith(("NM_", "NUMI_"))}
    run_env.update(env)
    start = time.monotonic()
    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        try:
            result = subprocess.run(argv, env=run_env, cwd=launch["cwd"], check=False,
                                    timeout=540, stdout=out, stderr=err)
            receipt = {"exit": result.returncode, "stop": "exit"}
        except subprocess.TimeoutExpired:
            receipt = {"exit": 124, "stop": "timeout"}
    receipt["wall_seconds"] = time.monotonic() - start
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
