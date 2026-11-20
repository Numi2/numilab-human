"""Replay the existing posterior mass and contact scene around its late solver outlier."""
from argparse import ArgumentParser
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import monotonic

EXPECTED = "214d94f8e953a270a2e41c47c10687ee64b710d6e4d9f9ce22b970ecf056b486"


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(reference, pack, out, root, steps, trace, shader, fields, scratch, staged, binary, library):
    assert 0 <= root < steps
    launch = json.loads(reference.read_text())
    assert digest(pack) == EXPECTED
    assert digest(Path(launch["argv"][0])) == launch["native_binary_sha256"]
    if binary is not None:
        launch["parent_native_binary_sha256"] = launch["native_binary_sha256"]
        launch["argv"][0] = str(binary)
        launch["native_binary_sha256"] = digest(binary)
    env = launch["environment"]
    if library is not None:
        if library.name != "libmetalrobo.dylib":
            raise ValueError("Expected libmetalrobo.dylib override")
        env["DYLD_LIBRARY_PATH"] = str(library.parent)
        launch["loaded_library_sha256"] = digest(library)
    assert env["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] == "1"
    for name in tuple(env):
        if name in {"NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV", "NUMI_HUMAN_SKIN_FAT_PARENT_MASS_LEDGER_CSV",
                    "NUMI_HUMAN_BATCH_CONTACT_WITNESS_CSV"}:
            env.pop(name)
    env.update(NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE=str(pack),
               NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256=EXPECTED,
               NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE="1" if fields else "0")
    if shader is not None:
        env["NUMI_MATTER_METALLIB_OVERRIDE"] = str(shader)
        launch["matter_library_sha256"] = digest(shader)
    if staged:
        env["NUMI_HUMAN_CERTIFIED_NEWTON_BUDGET_TRIAL"] = "1"
    if trace:
        env.update(NM_FEM_TRUE_RESIDUAL_ROOT=str(root),
                   NUMI_MATTER_GPU_TIMING="1", NUMI_MATTER_GPU_TIMING_STAGES="1",
                   NUMI_MATTER_GPU_TIMING_ROOT=str(root))
    launch["argv"][4] = str(out / "native")
    launch["argv"][launch["argv"].index("--muscle-step-count") + 1] = str(steps)
    launch["parent_launch_sha256"] = digest(reference)
    launch["profile_root"] = root if trace else None
    out.mkdir()
    if scratch is not None:
        scratch.mkdir()
        (out / "native").symlink_to(scratch, target_is_directory=True)
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("NUMI_", "NM_"))}
    start = monotonic()
    with (out / "stderr.log").open("xb") as stderr:
        if fields:
            with (out / "stdout.log.gz").open("xb") as compressed:
                with subprocess.Popen(["/usr/bin/gzip", "-1"], stdin=subprocess.PIPE,
                                      stdout=compressed) as gzip:
                    result = subprocess.run(launch["argv"], cwd=launch["cwd"], env=clean | env,
                                            stdout=gzip.stdin, stderr=stderr, check=False, timeout=360)
                    gzip.stdin.close()
                    assert gzip.wait() == 0
        else:
            with (out / "stdout.log").open("xb") as stdout:
                result = subprocess.run(launch["argv"], cwd=launch["cwd"], env=clean | env,
                                        stdout=stdout, stderr=stderr, check=False, timeout=360)
    receipt = {"returncode": result.returncode, "wall_seconds": monotonic() - start}
    print(json.dumps(receipt), flush=True)
    (out / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result.returncode


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("reference", "pack", "out"):
        parser.add_argument(name, type=lambda v: Path(v).resolve())
    parser.add_argument("--root", type=int, default=84)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--shader", type=lambda v: Path(v).resolve())
    parser.add_argument("--fields", action="store_true")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--binary", type=lambda v: Path(v).resolve())
    parser.add_argument("--library", type=lambda v: Path(v).resolve())
    parser.add_argument("--scratch", type=lambda v: Path(v).resolve())
    raise SystemExit(run(**vars(parser.parse_args())))
