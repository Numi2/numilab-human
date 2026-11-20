"""Run the full native 16-root scene with the re-emitted registered banks."""
from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "resting-clearance-and-atrium-017/floor-native-live-parent-026"


def digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("name")
    args = parser.parse_args()
    assert Path(args.name).name == args.name and args.name not in ("", ".", "..")
    source = REFERENCE / "launch.json"
    plan = json.loads(source.read_text())
    repair = json.loads((HERE / "source-replay.json").read_text())
    assert digest(source) == repair["reference_launch_sha256"]
    for path, expected in plan["pins"].items():
        assert digest(path) == expected, path
    output = HERE / args.name
    provisional_packs = 2 * (321_320_000 + 33_554_432)
    storage = os.statvfs(HERE)
    assert storage.f_bavail * storage.f_frsize >= 2_147_483_648 + provisional_packs
    argv = [value.replace(str(REFERENCE), str(output)) for value in plan["argv"]]
    environment = {key: value.replace(str(REFERENCE), str(output))
                   for key, value in plan["environment"].items()}
    pins = dict(plan["pins"])
    for flag, repair_name, previous_key, repaired_key in [
        ("--soft-tissue-payload", "registered-muscle.nhtissue", "source_sha256", "candidate_sha256"),
        ("--resting-anatomy-receipt", "resting-anatomy-receipt.json",
         "source_receipt_sha256", "candidate_receipt_sha256")]:
        key = argv.index(flag) + 1
        assert digest(argv[key]) == repair[previous_key]
        pins[str(argv[key])] = repair[previous_key]
        new = HERE / repair_name
        assert digest(new) == repair[repaired_key]
        argv[key] = str(new)
        pins[str(new)] = repair[repaired_key]
    # The native cardiac field resolves its binary sidecars beside the
    # receipt. Stage the same inode when possible and pin the bytes before launch.
    common = json.loads((HERE / "resting-anatomy-receipt.json").read_text())[
        "provenance"]["cardiac_geometry_binding"]["common_field"]
    for label in ("map", "polynomials", "domain_boxes"):
        descriptor = common[label]
        relative = Path(descriptor["path"])
        assert not relative.is_absolute() and relative.parts and ".." not in relative.parts
        initial = Path(repair["source_receipt"]).parent / relative
        staged = HERE / relative
        assert digest(initial) == descriptor["sha256"]
        staged.parent.mkdir(parents=True, exist_ok=True)
        if not staged.exists():
            os.link(initial, staged)
        assert digest(staged) == descriptor["sha256"]
        pins[str(staged)] = descriptor["sha256"]

    launch = {"argv": argv, "environment": environment, "cwd": plan["cwd"], "pins": pins,
              "reference_launch_sha256": digest(source)}
    output.mkdir()
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("NM_", "NUMI_"))}
    env.update(environment)
    start = time.monotonic()
    with (output / "stdout.log").open("wb") as out, (output / "stderr.log").open("wb") as err:
        run = subprocess.run(argv, env=env, cwd=plan["cwd"], stdout=out, stderr=err,
                             timeout=630, check=False)
    execution = {"exit": run.returncode, "wall_seconds": time.monotonic() - start}
    (output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
    print(json.dumps(execution))
    run.check_returncode()


if __name__ == "__main__":
    main()
