"""Run the prepared skin/fat fit with the exact reference native controls."""
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

AUDIT_OUTPUTS = {
    "NUMI_HUMAN_BATCH_CONTACT_WITNESS_CSV": "skin-bone-witness.csv",
    "NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV": "initial-skin-fem.csv",
}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--reuse-initial-pack", type=Path)
    args = parser.parse_args()
    assert args.steps > 0
    reference = ROOT / "bed-normal-remaining-banks-native" / "launch.json"
    launch = json.loads(reference.read_text())
    for path, checksum in launch["pins"].items():
        assert digest(path) == checksum, path
    assert json.loads((reference.parent / "execution.json").read_text())["exit"] == 0
    launch["comparison_launch_sha256"] = digest(reference)
    argv = launch["argv"]
    output = args.output.resolve()
    binary = HERE / "build/numi-human-native"
    binary_digest = digest(binary)
    assert binary_digest in (HERE / "build.log").read_text().splitlines()[-1]
    del launch["pins"][argv[0]]
    argv[0] = str(binary)
    launch["native_binary_sha256"] = binary_digest
    launch["pins"][argv[0]] = binary_digest
    for asset in (HERE / "material-fit.json", HERE / "build/native.patch"):
        launch["pins"][str(asset)] = digest(asset)
    argv[4] = str(output / "native")
    argv[argv.index("--muscle-step-count") + 1] = str(args.steps)
    overrides = launch["environment"]
    for key in ("NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE",
                "NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256",
                "NUMI_HUMAN_SKIN_REFERENCE_DUMP",
                "NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP",
                "NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP"):
        overrides.pop(key, None)
    overrides.update({key: str(output / value) for key, value in AUDIT_OUTPUTS.items()})
    if args.reuse_initial_pack is not None:
        cached_run = args.reuse_initial_pack.resolve()
        cached_launch = json.loads((cached_run / "launch.json").read_text())
        assert json.loads((cached_run / "execution.json").read_text())["exit"] == 0
        assert cached_launch["pins"] == launch["pins"]
        comparable_argv = cached_launch["argv"].copy()
        comparable_argv[4] = argv[4]
        step_value = argv.index("--muscle-step-count") + 1
        comparable_argv[step_value] = argv[step_value]
        assert comparable_argv == argv
        for key in set(cached_launch["environment"]) | set(overrides):
            if key not in AUDIT_OUTPUTS:
                assert cached_launch["environment"].get(key) == overrides.get(key), key
        pack = cached_run / "native/resting-human.mrvpack"
        assert pack.is_file() and not pack.is_symlink()
        checksum = digest(pack)
        launch["pins"][str(pack)] = checksum
        overrides.update(NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE=str(pack),
                         NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256=checksum)
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
            result = subprocess.run(argv, cwd=launch["cwd"], env=env, check=False,
                                    stdout=out, stderr=err, timeout=360)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    receipt = {"exit": code, "wall_seconds": monotonic() - start}
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
