"""Compare native MyoSim recruitment against the repaired full-volume release."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
CONTROL = WORK / "resting-clearance-and-atrium-017/floor-native-live-parent-026"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    original = CONTROL / "launch.json"
    plan = json.loads(original.read_text())
    for path, expected in plan["pins"].items():
        assert digest(path) == expected, path
    output = HERE / "activation-010"
    storage = os.statvfs(HERE)
    assert storage.f_bavail * storage.f_frsize > 2_147_483_648 + 710_000_000
    argv = [v.replace(str(CONTROL), str(output)) for v in plan["argv"]]
    assert "--resting-release-initialization" in argv
    assert "--resting-registered-ligaments" in argv
    index = argv.index("--muscle-activation") + 1
    assert argv[index] == "0.0"
    argv[index] = "0.10"
    environment = {k: v.replace(str(CONTROL), str(output)) for k, v in plan["environment"].items()}
    output.mkdir()
    plan.update(argv=argv, environment=environment, reference_launch_sha256=digest(original))
    (output / "launch.json").write_text(json.dumps(plan, indent=2) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NM_", "NUMI_"))}
    env.update(environment)
    started = time.monotonic()
    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        try:
            child = subprocess.run(argv, cwd=plan["cwd"], env=env, stdout=out, stderr=err,
                                   timeout=630, check=False)
            result = dict(exit=child.returncode, stop="exit")
        except subprocess.TimeoutExpired:
            result = dict(exit=124, stop="timeout")
    result["wall_seconds"] = time.monotonic() - started
    (output / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
