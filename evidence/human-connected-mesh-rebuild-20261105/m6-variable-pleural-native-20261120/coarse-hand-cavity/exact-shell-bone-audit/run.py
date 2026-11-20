"""Retain the first accepted root and physical bone rows on the matched model."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_hand_trial import digest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    ref = args.reference.resolve()
    launch = json.loads(ref.read_text())
    assert json.loads((ref.parent / "execution.json").read_text())["exit"] == 0
    for path, sha in launch["pins"].items():
        assert digest(path) == sha, path
    argv = launch["argv"]
    old_binary = argv[0]
    binary = HERE / "numi-human-native"
    launch["comparison_launch_sha256"] = digest(ref)
    launch["variant"] = "compiled-physical-bone-triangles-audit"
    launch["native_binary_sha256"] = digest(binary)
    argv[0] = str(binary)
    argv[4] = str(output / "native")
    argv[argv.index("--muscle-step-count") + 1] = "1"
    launch["pins"].pop(old_binary)
    for artifact in (binary, HERE / "compiled-bone-export.patch"):
        launch["pins"][str(artifact)] = digest(artifact)

    settings = launch["environment"]
    for key, name in {
        "NUMI_HUMAN_INITIAL_BONE_GEOMETRY_AUDIT": "physical-bone-initial.bin",
        "NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV": "initial-skin-fem.csv",
        "NUMI_HUMAN_BATCH_CONTACT_WITNESS_CSV": "skin-bone-witness.csv",
    }.items():
        settings[key] = str(output / name)
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("NM_", "NUMI_"))}
    env.update(settings)

    output.mkdir()
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    start = monotonic()
    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        result = subprocess.run(argv, cwd=launch["cwd"], env=env,
                                check=False, stdout=out, stderr=err, timeout=120)
    receipt = {"exit": result.returncode, "wall_seconds": monotonic() - start}
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
