"""Run the registered fixed-bed load seed with the matched physical gates."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from build_hand_trial import digest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output.resolve()
    ref = ROOT / "operator-resident-030/parallel-operators-128/launch.json"
    launch = json.loads(ref.read_text())
    prediction_path = HERE / "prediction.json"
    prediction = json.loads(prediction_path.read_text())
    assert digest(ref) == prediction["source_launch_sha256"]
    assert json.loads((ref.parent / "execution.json").read_text())["exit"] == 0
    for path, checksum in launch["pins"].items():
        assert digest(path) == checksum, path

    argv = launch["argv"]
    binary = HERE / "numi-human-native"
    binary_digest = digest(binary)
    replaced_binary = argv[0]
    launch["comparison_launch_sha256"] = digest(ref)
    launch["variant"] = "analytic-fixed-bed-load-seed"
    launch["native_binary_sha256"] = binary_digest
    argv[0] = str(binary)
    argv[4] = str(output / "native")
    argv[argv.index("--muscle-step-count") + 1] = "4"
    pose_start = argv.index("--root-pose") + 1
    pose = [format(value, ".17g") for value in prediction["root_pose"]]
    argv[pose_start:pose_start + len(pose)] = pose

    overrides = launch["environment"]
    del launch["pins"][replaced_binary]
    del launch["pins"][overrides.pop("NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE")]
    del overrides["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"]
    del overrides["NUMI_HUMAN_FEM_SUPPORT_BARRIER_RELEASE"]
    for path in (binary, prediction_path, HERE / "experiment.md",
                 HERE / "NumiHumanFEMOuterGroundAlignment.hpp.patch",
                 HERE / "numilab_human_myosim_visual_probe.mm.patch"):
        launch["pins"][str(path)] = digest(path)
    override_paths = {
        "NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV": "initial-skin-fem.csv",
        "NUMI_HUMAN_BATCH_CONTACT_WITNESS_CSV": "skin-bone-witness.csv",
    }
    overrides.update({key: str(output / name) for key, name in override_paths.items()})
    overrides["NUMI_EXPERIMENT_INITIAL_ROOT_POSE"] = ",".join(pose)
    overrides["NUMI_HUMAN_FEM_TRIAL_INITIAL_CLEARANCE_M"] = format(
        prediction["minimum_clearance_m"], ".17g"
    )
    assert overrides["NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS"] == "24"
    assert overrides["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] == "1"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("NM_", "NUMI_"))}
    env.update(overrides)

    output.mkdir()
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    start = monotonic()
    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        try:
            code = subprocess.run(argv, cwd=launch["cwd"], env=env, check=False,
                                  stdout=out, stderr=err, timeout=180).returncode
        except subprocess.TimeoutExpired:
            code = 124
    receipt = {"exit": code, "wall_seconds": monotonic() - start}
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
