"""Compare alternative Metal operator schedules on the pinned Human scene."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from build_hand_trial import digest

BASELINE = root / "bed-normal-remaining-banks-native"
ASSEMBLED = {
    "NM_FEM_ASSEMBLED_XYZ_OPERATOR": "1",
    "NM_FEM_ASSEMBLED_XYZ_HUMAN_ONLY": "1",
    "NM_FEM_ASSEMBLED_XYZ_DIRECT_EDGE": "1",
    "NM_FEM_ASSEMBLED_XYZ_SIMD_ROW": "1",
}
DUMPS = (
    "NUMI_HUMAN_SKIN_REFERENCE_DUMP", "NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP",
    "NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP", "NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV",
    "NUMI_HUMAN_BATCH_CONTACT_WITNESS_CSV",
)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("control", "assembled", "borrowed", "graph"), default="control")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--profile-root", type=int)
    parser.add_argument("--newton", type=int, choices=range(2,25))
    args = parser.parse_args()
    assert args.steps > 0
    initial_launch = BASELINE / "launch.json"
    launch = json.loads(initial_launch.read_text())
    assert json.loads((BASELINE / "execution.json").read_text())["exit"] == 0
    assert all(digest(path) == sha for path, sha in launch["pins"].items())
    launch["comparison_launch_sha256"] = digest(initial_launch)
    output = args.output.resolve()
    argv = launch["argv"]
    argv[4] = str(output / "native")
    argv[argv.index("--muscle-step-count") + 1] = str(args.steps)
    env = launch["environment"]
    for key in DUMPS:
        env.pop(key, None)
    pack = BASELINE / "native/resting-human.mrvpack"
    sha = digest(pack)
    assert sha == "129caadf7baa062451eb82f24494eafb8fec8822151ab41a36b1a634740f77b0"
    env.update(NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE=str(pack),
               NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256=sha)
    launch["pins"][str(pack)] = sha
    if args.backend == "assembled":
        env.update(ASSEMBLED)
    elif args.backend == "borrowed":
        env["NM_FGMRES_COUPLED_BORROWED_ENCODER"] = "1"
    elif args.backend == "graph":
        env["NM_FGMRES_SKIP_COMPLETED_GRAPH_WORK"] = "1"
    if args.newton is not None:
        env["NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS"] = str(args.newton)
    if args.profile_root is not None:
        assert 0 <= args.profile_root < args.steps
        env.update(NUMI_MATTER_GPU_TIMING="1", NUMI_MATTER_GPU_TIMING_STAGES="1",
                   NUMI_MATTER_GPU_TIMING_ROOT=str(args.profile_root))
    output.mkdir()
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    declared = {key: value for key, value in os.environ.items()
                if not key.startswith(("NM_", "NUMI_"))}
    declared.update(env)
    start = monotonic()
    with (output / "stdout.log").open("x") as out, (output / "stderr.log").open("x") as err:
        try:
            result = subprocess.run(argv, env=declared, cwd=launch["cwd"],
                                    stdout=out, stderr=err, timeout=360, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    receipt = {"exit": code, "wall_seconds": monotonic() - start,
               "profile_root": args.profile_root, "backend": args.backend}
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
