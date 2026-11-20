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
    parser.add_argument("--column-trace", action="store_true")
    parser.add_argument("--physical-trace", action="store_true")
    parser.add_argument("--weight", choices=("sqrt", "original"), required=True)
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
    build = HERE / "build"
    pins = dict(plan["pins"])
    old_app = argv[0]
    argv[0] = str(build / "numi-human-native")
    pins.pop(old_app)
    pins[argv[0]] = sha(argv[0])
    argv[argv.index("--muscle-step-count") + 1] = str(args.roots)
    if "--mechanics-only" not in argv:
        argv.append("--mechanics-only")

    launch_env = {key: relocate(value) for key, value in plan["environment"].items()}
    launch_env.pop("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS", None)
    for key, file in (("DYLD_LIBRARY_PATH", "libmetalrobo.dylib"),
                      ("NUMI_MATTER_METALLIB_OVERRIDE", "NumiMatter.metallib")):
        old = Path(launch_env[key])
        pins.pop(str(old/file) if key == "DYLD_LIBRARY_PATH" else str(old))
        pins[str(build/file)] = sha(build/file)
        launch_env[key] = str(build if key == "DYLD_LIBRARY_PATH" else build/file)
    launch_env["NM_FGMRES_PATCH_LINEARIZATION_CACHE"] = "1"
    launch_env["NM_FGMRES_ROW_TRANSFORM_WORK_DOMAIN"] = "1"
    launch_env["NM_FGMRES_ROW_SCALE_SQRT"] = str(int(args.weight == "sqrt"))
    assert launch_env["NM_FGMRES_PREFIX_PRECONDITIONER_ICB"] == "0"
    for name in (
        "NUMI_MATTER_GPU_TIMING", "NUMI_MATTER_GPU_TIMING_ROOT", "NUMI_MATTER_GPU_TIMING_STAGES",
    ):
        assert name not in launch_env
    launch_env.update(NUMI_MATTER_GPU_TIMING="1", NUMI_MATTER_GPU_TIMING_STAGES="1",
                      NUMI_MATTER_GPU_TIMING_ROOT=str(args.profile_root))
    if args.column_trace:
        launch_env.update(NM_FEM_NEWTON_TRACE_ROOT=str(args.profile_root), NM_FGMRES_COLUMN_TRACE="1")
    if args.physical_trace:
        launch_env.update(NM_FEM_TRUE_RESIDUAL_ROOT=str(args.profile_root),
                          NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT=str(args.profile_root))
        launch_env.pop("NM_FEM_TRUE_RESIDUAL_ROW_WITNESS", None)
    live_env = {key: value for key, value in os.environ.items()
                if not key.startswith(("NUMI_", "NM_"))}
    live_env.update(launch_env)

    output.mkdir()
    (output / "launch.json").write_text(json.dumps(dict(argv=argv, cwd=plan["cwd"],
        environment=launch_env, pins=pins, baseline_launch_sha256=sha(source)),indent=2) + "\n")
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
