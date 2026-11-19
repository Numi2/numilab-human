"""Re-run the matched supported Human with the authored gliding interface."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
parent = here.with_name("resting-clearance-and-atrium-007")
baseline = parent / "shared-fascia-tail-registered-pose-4/launch.json"
steps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
name = sys.argv[2] if len(sys.argv) > 2 else "deep-pectoral"
run = here / name
run.mkdir()
scratch = Path("/Volumes/NumiHumanTrial")
assert os.path.ismount(scratch)
native = scratch / name
cfg = json.loads(baseline.read_text())
argv = cfg["argv"]
env = cfg["environment"]
pins = cfg["pins"]
old_binary = argv[0]
build = "deep-pectoral-depth-build" if "-depth" in name else "deep-pectoral-build"
argv[0] = str(here / build / "numi-human-native")
argv[4] = str(native)
argv[argv.index("--muscle-step-count") + 1] = str(steps)
pins.pop(old_binary)
pins.pop(env.pop("NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE"))
env.pop("NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256")
env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"] = str(run / "failure.json")
env["NUMI_HUMAN_ACCEPTED_FEM_BODY_LOAD_CAPTURE"] = "1"
env["NM_FEM_PRIVATE_CONTACT_FAILURE_CAPTURE"] = "1"
if name.startswith("deep-pectoral-strict"):
    shader = "canonical-glide-alpha-shader" if name.endswith("-canonical-alpha") else "exact-trial-shader"
    library = here / shader / "NumiMatter.metallib"
    env["NUMI_MATTER_METALLIB_OVERRIDE"] = str(library)
    env["NM_FEM_BACKTRACKING_CONSTITUTIVE_VALIDITY"] = "1"
    env["NM_FEM_PARALLEL_BACKTRACKING"] = "1"
    env["NM_FEM_PARALLEL_CONSTITUTIVE_BACKTRACKING"] = "1"
    env["NM_FEM_COMPLETED_BACKTRACKING_COLLAPSE"] = "0"
    with library.open("rb") as f:
        pins[str(library)] = hashlib.file_digest(f, "sha256").hexdigest()
if name.endswith("-alpha-trace"):
    old_lib = Path(env["DYLD_LIBRARY_PATH"]) / "libmetalrobo.dylib"
    pins.pop(str(old_lib))
    lib = here / "pectoral-alpha-trace-runtime/libmetalrobo.dylib"
    env["DYLD_LIBRARY_PATH"] = str(lib.parent)
    env["NM_VASCULAR_TRACE_ROOT"] = "3"
    with lib.open("rb") as f:
        pins[str(lib)] = hashlib.file_digest(f, "sha256").hexdigest()
if name.endswith("-limiter-trace"):
    old_lib = Path(env["DYLD_LIBRARY_PATH"]) / "libmetalrobo.dylib"
    pins.pop(str(old_lib))
    lib = here / "pectoral-limiter-trace-runtime/libmetalrobo.dylib"
    env["DYLD_LIBRARY_PATH"] = str(lib.parent)
    env["NM_VASCULAR_TRACE_ROOT"] = "4"
    env["NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS"] = "24"
    env["NM_FEM_TERMINAL_CERTIFICATE_TRACE"] = "1"
    with lib.open("rb") as f:
        pins[str(lib)] = hashlib.file_digest(f, "sha256").hexdigest()
if name.endswith("-newton24"):
    env["NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS"] = "24"
if name.endswith("-tiled"):
    env["NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS"] = "24"
    env["NM_FEM_TILED_DEFORMABLE_LIMIT"] = "1"
    env["NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT"] = "4"
    env["NM_FEM_DEFORMABLE_LIMIT_TRACE_MICROTICK"] = "12"
    env["NM_FEM_DEFORMABLE_LIMIT_TRACE_MINIMA"] = "1"
if name.endswith("-block"):
    env["NUMI_HUMAN_RESTING_COUPLED_TANGENT_BLOCK"] = "1"
assert env["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] == "1"
assert env["NUMI_HUMAN_RESTING_PASSIVE_FASCIA"] == "1"
with open(argv[0], "rb") as f:
    pins[argv[0]] = hashlib.file_digest(f, "sha256").hexdigest()
for path, digest in pins.items():
    with open(path, "rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == digest, path
(run / "launch.json").write_text(json.dumps(cfg, indent=2) + "\n")
shell_env = {k: v for k, v in os.environ.items() if not k.startswith(("NUMI_", "NM_"))}
start = time.monotonic()
with (run / "stdout.log").open("wb") as stdout, (run / "stderr.log").open("wb") as stderr:
    try:
        code = subprocess.run(argv, cwd=cfg["cwd"], env=shell_env | env, stdout=stdout,
                              stderr=stderr, timeout=220).returncode
    except subprocess.TimeoutExpired:
        code = None
result = dict(exit_code=code, wall_seconds=time.monotonic()-start, requested_steps=steps)
(run / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
brief = run / "native"
brief.mkdir()
for artifact in native.glob("*"):
    if artifact.is_file() and artifact.stat().st_size <= 8*1024*1024:
        shutil.copy2(artifact, brief)
print(json.dumps(result), flush=True)
sys.exit(1 if code is None else code)
