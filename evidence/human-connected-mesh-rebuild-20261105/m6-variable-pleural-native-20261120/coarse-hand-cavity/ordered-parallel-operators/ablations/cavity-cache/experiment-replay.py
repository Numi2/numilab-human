"""Measure admitted Metal execution paths on the published cleared skin shell."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from build_hand_trial import digest

PARALLEL_BASE = {"NM_FGMRES_PARALLEL_BASE_PRECONDITIONER": "1"}
PARALLEL_OPERATORS = {
    "NM_FEM_GATHER_SIMD": "1",
    "NM_FEM_CACHED_NODE_SIMD": "1",
    "NM_VASCULAR_COOPERATIVE_CAVITY_OPERATOR": "1",
}
OWNER_INCIDENCE_CERTIFICATE = {
    "NM_MIXED_CERTIFICATE_TILED": "1",
    "NM_MIXED_CERTIFICATE_OWNER_INCIDENCE": "1",
}
MODES = {
    "simd-gather": {"NM_FEM_GATHER_SIMD": "1"},
    "simd-cached": {"NM_FEM_CACHED_NODE_SIMD": "1"},
    "cooperative-vascular": {"NM_VASCULAR_COOPERATIVE_CAVITY_OPERATOR": "1"},
    "tiled-certificate": {"NM_MIXED_CERTIFICATE_TILED": "1"},
    "incidence-certificate": OWNER_INCIDENCE_CERTIFICATE,
    "control": {},
    "parallel-base": PARALLEL_BASE,
    "parallel-operators": PARALLEL_OPERATORS,
    "combined": {**PARALLEL_BASE, **PARALLEL_OPERATORS},
    "operators-and-certificate": {**PARALLEL_OPERATORS, **OWNER_INCIDENCE_CERTIFICATE},
}
OUTPUTS = {
    "NUMI_HUMAN_BATCH_CONTACT_WITNESS_CSV": "skin-bone-witness.csv",
    "NUMI_HUMAN_INITIAL_SKIN_FEM_NODES_CSV": "initial-skin-fem.csv",
}

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cavity-cache", choices=("off", "on"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--profile-root", type=int)
    parser.add_argument("--lean-observers", action="store_true")
    args = parser.parse_args()
    assert args.steps > 0
    parent = ROOT / "last-clearance-029"
    source = parent / "native-prefix/launch.json"
    reference = json.loads(source.read_text())
    archive = json.loads((parent / "final-pack-recovery.json").read_text())
    settings = reference["environment"]
    pack = HERE / "initial.mrvpack"
    assert settings["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE_SHA256"] == archive["sha256"]
    assert digest(pack) == archive["sha256"]
    assert reference["pins"].pop(settings["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE"]) == archive["sha256"]
    reference["pins"][str(pack)] = archive["sha256"]
    for path, expected in reference["pins"].items():
        assert digest(path) == expected, path
    assert settings["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] == "1"
    assert settings["NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS"] == "24"
    argv = reference["argv"]
    if args.cavity_cache is not None:
        assert args.mode == "parallel-operators"
        trial = HERE / "cavity-cache-trial"
        settings["NM_VASCULAR_CACHE_CAVITY_OPERATOR_GEOMETRY"] = "1" if args.cavity_cache == "on" else "0"
        settings["DYLD_LIBRARY_PATH"] = str(trial)
        settings["NUMI_MATTER_METALLIB_OVERRIDE"] = str(trial / "NumiMatter.metallib")
        argv[0] = str(trial / "numi-human-native")
        for name in ("numi-human-native", "libmetalrobo.dylib", "NumiMatter.metallib"):
            path = trial / name
            reference["pins"][str(path)] = digest(path)
    output = args.output.resolve()
    argv[4] = str(output / "native")
    argv[argv.index("--muscle-step-count") + 1] = str(args.steps)
    settings["NUMI_HUMAN_RESTING_INITIAL_PACK_CACHE"] = str(pack)
    settings.update({key: str(output / name) for key, name in OUTPUTS.items()})
    settings.update(MODES[args.mode])
    if args.profile_root is not None:
        assert 0 <= args.profile_root < args.steps
        settings.update(NUMI_MATTER_GPU_TIMING="1", NUMI_MATTER_GPU_TIMING_STAGES="1",
                        NUMI_MATTER_GPU_TIMING_ROOT=str(args.profile_root))
    if args.lean_observers:
        for key in ("NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT",
                    "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT",
                    "NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT"):
            settings[key] = "0"
        for key in ("NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS",
                    "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP",
                    "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP"):
            settings.pop(key)
    reference["comparison_launch_sha256"] = digest(source)
    reference["variant"] = args.mode
    reference["cavity_cache"] = args.cavity_cache
    reference["instrumented_root"] = args.profile_root
    reference["lean_observers"] = args.lean_observers
    output.mkdir()
    (output / "launch.json").write_text(json.dumps(reference, indent=2) + "\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith(("NUMI_", "NM_"))}
    env.update(settings)
    start = monotonic()
    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        try:
            code = subprocess.run(argv, cwd=reference["cwd"], env=env, stdout=out,
                                  stderr=err, check=False, timeout=360).returncode
        except subprocess.TimeoutExpired:
            code = 124
    result = {"exit": code, "wall_seconds": monotonic() - start}
    (output / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(code)

if __name__ == "__main__":
    main()
