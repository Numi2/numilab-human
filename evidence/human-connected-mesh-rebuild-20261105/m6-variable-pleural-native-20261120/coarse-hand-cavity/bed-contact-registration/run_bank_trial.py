"""Replay the pinned coarse Human with the optional complete bank audit."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import time

from build_hand_trial import HERE, digest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--mechanics-only", action="store_true",
                        help="Keep the native coupled step and omit visual output for solver diagnostics")
    parser.add_argument("--bed-barrier-release", action="store_true",
                        help="Use the existing fixed-plane release at its barrier activation distance")
    parser.add_argument("--footprint",type=Path,
                        help="Verified one-time flat-bed root orientation")
    parser.add_argument("--orientation-source",type=Path)
    parser.add_argument("--reuse-initial-pack", type=Path,
                        help="Clone the exact initial pack from a matched completed native run")
    args = parser.parse_args()
    assert args.steps > 0
    assert (args.footprint is None) == (args.orientation_source is None)
    binary = args.build.resolve() / "numi-human-native"
    output = args.output.resolve()
    baseline = HERE / "hand-clearance-native-replay/launch.json"
    launch = json.loads(baseline.read_text())
    for asset, sha in launch["pins"].items():
        assert digest(asset) == sha, asset
    argv = launch["argv"]
    del launch["pins"][argv[0]]
    argv[0] = str(binary)
    argv[4] = str(output / "native")
    argv[argv.index("--muscle-step-count") + 1] = str(args.steps)
    if args.mechanics_only:
        assert "--mechanics-only" not in argv
        argv.append("--mechanics-only")
    launch["pins"][str(binary)] = digest(binary)
    launch["native_binary_sha256"] = launch["pins"][str(binary)]
    launch["comparison_launch_sha256"] = digest(baseline)
    if args.footprint is not None:
        orientation = json.loads(args.footprint.read_text())
        source = args.orientation_source.resolve()
        assert digest(source / "launch.json") == orientation["launch_sha256"]
        assert digest(source / "initial-skin-fem.csv") == orientation["source_outer_sha256"]
        assert digest(binary) == json.loads((source / "launch.json").read_text())["native_binary_sha256"]
        i = argv.index("--root-pose")+1
        assert list(map(float,argv[i:i+3])) == orientation["root_pose_trial"][:3]
        argv[i:i+7] = [format(x,".17g") for x in orientation["root_pose_trial"]]
        launch["pins"][str(args.footprint.resolve())] = digest(args.footprint)
        launch["orientation_source_launch_sha256"] = orientation["launch_sha256"]
    env = launch["environment"]
    del env["NUMI_HUMAN_HAND_CONTACT_WITNESS_CSV"]
    if args.bed_barrier_release:
        assert env.get("NUMI_HUMAN_FEM_SKIN_SUPPORT") == "1"
        env["NUMI_HUMAN_FEM_SUPPORT_BARRIER_RELEASE"] = "1"
    paths = {
        "NUMI_HUMAN_SKIN_REFERENCE_DUMP": "registered-skin-native.csv",
        "NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP": "registered-skin-poses.json",
        "NUMI_HUMAN_TISSUE_MASS_REFERENCE_DUMP": "tissue-mass.txt",
        "NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV": "initial-skin-fem.csv",
    }
    if args.audit:
        paths["NUMI_HUMAN_BATCH_CONTACT_WITNESS_CSV"] = "skin-bone-witness.csv"
    if args.reuse_initial_pack is not None:
        source = args.reuse_initial_pack.resolve()
        receipt = json.loads((source / "execution.json").read_text())
        assert receipt["exit"] == 0, "initial pack reuse requires a completed native run"
        previous = json.loads((source / "launch.json").read_text())
        compare_argv = previous["argv"].copy()
        compare_argv[4] = argv[4]
        index = argv.index("--muscle-step-count") + 1
        compare_argv[index] = argv[index]
        assert compare_argv == argv, "initial pack reuse requires identical scene arguments"
        audit_paths = set(paths)
        previous_env = {key: value for key, value in previous["environment"].items()
                        if key not in audit_paths}
        new_env = {key: value for key, value in env.items() if key not in audit_paths}
        assert previous_env == new_env, "initial pack reuse requires identical scene environment"
        assert previous["pins"] == launch["pins"], "initial pack source pins differ"
        assert previous["cwd"] == launch["cwd"], "initial pack source working directory differs"
        pack = source / "native" / "resting-human.mrvpack"
        assert pack.is_file() and not pack.is_symlink()
        pack_hash = digest(pack)
        env.update({"NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE": str(pack),
                    "NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256": pack_hash})
        launch["initial_pack_cache_source_launch_sha256"] = digest(source / "launch.json")
        launch["pins"][str(pack)] = pack_hash
    env.update({key: str(output / name) for key, name in paths.items()})
    output.mkdir()
    (output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
    run_env = {k: v for k, v in os.environ.items() if not k.startswith(("NM_", "NUMI_"))}
    run_env.update(env)
    start = time.monotonic()
    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        try:
            result = subprocess.run(argv, env=run_env, cwd=launch["cwd"], check=False,
                                    timeout=540, stdout=out, stderr=err)
            receipt = {"exit": result.returncode, "stop": "exit"}
        except subprocess.TimeoutExpired:
            receipt = {"exit": 124, "stop": "timeout"}
    receipt["wall_seconds"] = time.monotonic() - start
    (output / "execution.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    if receipt["exit"]:
        raise SystemExit(receipt["exit"] if receipt["exit"] > 0 else 128 - receipt["exit"])


if __name__ == "__main__":
    main()
