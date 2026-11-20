"""Profile one full native root while holding the accepted model and physics fixed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "resting-clearance-and-atrium-017/floor-native-live-parent-026"

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=int, default=1)
    parser.add_argument("--roots", type=int, default=3)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert 0 <= args.profile_root < args.roots <= 16
    output = args.output.resolve()
    source = BASE / "launch.json"
    plan = json.loads(source.read_text())
    for name, expected in plan["pins"].items():
        assert sha(name) == expected, name
    st = os.statvfs(output.parent)
    assert st.f_bavail * st.f_frsize >= 2_502_338_044, "One native initial pack and safety reserve required"

    def relocate(value):
        return value.replace(str(BASE), str(output))

    argv = [relocate(value) for value in plan["argv"]]
    argv[argv.index("--muscle-step-count") + 1] = str(args.roots)
    if "--mechanics-only" not in argv:
        argv.append("--mechanics-only")

    launch_env = {key: relocate(value) for key, value in plan["environment"].items()}
    launch_env.pop("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS", None)
    for name in (
        "NUMI_MATTER_GPU_TIMING", "NUMI_MATTER_GPU_TIMING_ROOT", "NUMI_MATTER_GPU_TIMING_STAGES",
    ):
        assert name not in launch_env
    launch_env.update(NUMI_MATTER_GPU_TIMING="1", NUMI_MATTER_GPU_TIMING_STAGES="1",
                      NUMI_MATTER_GPU_TIMING_ROOT=str(args.profile_root))
    live_env = {key: value for key, value in os.environ.items()
                if not key.startswith(("NUMI_", "NM_"))}
    live_env.update(launch_env)

    output.mkdir()
    (output / "launch.json").write_text(json.dumps(dict(argv=argv, cwd=plan["cwd"],
        environment=launch_env, pins=plan["pins"], baseline_launch_sha256=sha(source)),indent=2) + "\n")
    start = time.monotonic()
    with (output / "stdout.log").open("wb") as out, (output / "stderr.log").open("wb") as err:
        run = subprocess.run(argv, cwd=plan["cwd"], env=live_env, stdout=out, stderr=err,
                             check=False, timeout=360)
    result = dict(exit=run.returncode, wall_seconds=time.monotonic() - start)
    (output / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    run.check_returncode()

if __name__ == "__main__":
    main()
