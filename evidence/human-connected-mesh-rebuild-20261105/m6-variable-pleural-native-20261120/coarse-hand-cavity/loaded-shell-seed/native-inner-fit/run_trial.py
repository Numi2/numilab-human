"""Launch the matched physical Human fixture on Apple GPU."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
from time import monotonic

from build_trial import HERE, REFERENCE, digest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--reuse-candidate-pack-from", type=Path)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    output = args.out.resolve()
    output.mkdir()
    original_path = REFERENCE / "launch.json"
    launch = json.loads(original_path.read_text())
    executable = HERE / "numi-human-native"
    for path, want in launch["pins"].items():
        assert digest(path) == want, path
    launch["argv"][0] = str(executable)
    launch["argv"][4] = str(output / "native")
    launch["argv"][launch["argv"].index("--muscle-step-count") + 1] = str(args.steps)
    launch["native_binary_sha256"] = digest(executable)
    launch["parent_launch_sha256"] = digest(original_path)
    launch["variant"] = "load-witness-reference-fit" if args.candidate else "same-executable-unchanged-control"
    env = launch["environment"]
    if args.candidate:
        # The V3 visual reference authenticates the changed internal layer.
        # An unchanged-scene cache cannot satisfy that reference.
        for key in ("NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE", "NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"):
            env.pop(key, None)
        env["NUMI_HUMAN_LOAD_WITNESS_INNER_FIT"] = "1"
    else:
        env.pop("NUMI_HUMAN_LOAD_WITNESS_INNER_FIT", None)
    if args.reuse_candidate_pack_from:
        prior_run = args.reuse_candidate_pack_from.resolve()
        prior = json.loads((prior_run / "launch.json").read_text())
        assert args.candidate and json.loads((prior_run / "execution.json").read_text())["returncode"] == 0
        assert prior["variant"] == launch["variant"] and prior["native_binary_sha256"] == digest(executable)
        pack = prior_run / "native/resting-human.mrvpack"
        pack_hash = digest(pack)
        launch["pins"][str(pack)] = pack_hash
        env["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE"] = str(pack)
        env["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"] = pack_hash
    env.update({"NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV": str(output / "initial-skin-fem.csv"),
                "NUMI_HUMAN_SKIN_FAT_PARENT_MASS_LEDGER_CSV": str(output / "shell-mass.csv")})
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    run_environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("NM_", "NUMI_"))}
    run_environment.update(env)
    start = monotonic()
    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        result = subprocess.run(launch["argv"], cwd=launch["cwd"], env=run_environment,
                                stdout=out, stderr=err, check=False, timeout=180)
    evidence = {file.name: digest(file) for file in sorted(output.glob("shell-mass*.csv"))}
    result = {"returncode": result.returncode, "wall_seconds": monotonic() - start, "mass_ledgers": evidence}
    (output / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
