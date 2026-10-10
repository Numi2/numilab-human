"""Launch the existing coupled Apple Metal Human scene from owner receipts.

This module loads and checks assets, records the invocation, and starts the
native process. It has no physical or physiological stepping implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time

from .model import ImportError as HumanImportError


INVOCATION_ENVIRONMENT_KEYS = (
    "DYLD_LIBRARY_PATH", "DYLD_PRINT_LIBRARIES",
    "NUMI_HUMAN_SPLIT_STAND", "NUMI_HUMAN_EXECUTION_STAGES",
    "NUMI_HUMAN_TRAINING_PROFILE", "NUMI_HUMAN_RESTING_TRANSACTION_PROBE",
    "NUMI_HUMAN_RESTING_INSPECTION_TOUR",
    "NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS",
    "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS",
    "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT",
    "NUMI_HUMAN_RESTING_TIMESTEP_SENSITIVITY",
    "NUMI_HUMAN_RESTING_CONSOLIDATE_LINEAR_BOUNDS",
    "NUMI_HUMAN_RESTING_ELIMINATE_FIXED_BOUNDS",
    "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT",
    "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS",
    "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT",
    "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP",
    "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP",
    "NUMI_HUMAN_ACCEPTED_FORCE_AUDIT",
    "NUMI_HUMAN_GPU_TIMING", "NUMI_HUMAN_GPU_TIMING_STAGE", "NUMI_HUMAN_GPU_TIMING_ROOTS",
    "NUMI_HUMAN_BRAIN_INNER_PHASE_TIMING", "NUMI_HUMAN_BRAIN_JOINT_PATH_CALIBRATION",
    "NUMI_HUMAN_BRAIN_MOTOR_AUDIT_INTERVAL", "NUMI_HUMAN_BRAIN_MOTOR_V2",
    "NUMI_HUMAN_BRAIN_PHASE_TIMING", "NUMI_HUMAN_BRAIN_SENSOR_AUDIT",
    "NUMI_HUMAN_FRICTION_DIAG", "NUMI_HUMAN_RESIDENT_PHYSICS_PILOT",
    "NUMI_HUMAN_STATIC_EQUILIBRIUM_CACHE_PATH", "NUMI_HUMAN_STATIC_EQUILIBRIUM_CACHE_KEY",
    "NUMI_MATTER_GPU_TIMING", "NUMI_MATTER_GPU_TIMING_STAGE", "NUMI_MATTER_GPU_TIMING_DENSE45",
    "NUMI_HUMAN_SUPPORT_GPU_TIMING",
    "NUMI_HUMAN_PARALLEL_MASS_ASSEMBLY", "NUMI_HUMAN_KINEMATICS_CACHE",
    "NUMI_HUMAN_OVERLAP_GEOMETRY", "NUMI_HUMAN_GAS_TRANSPORT_SUBCYCLING",
    "NUMI_HUMAN_RESPIRATORY_SUBCYCLING",
    "NUMI_HUMAN_PARALLEL_RESPIRATORY_GAS",
    "NUMI_HUMAN_PARALLEL_RESPIRATORY_MUSCLES",
    "NUMI_HUMAN_STAND_CACHE_LIMIT_EQUALITY",
    "NUMI_HUMAN_STAND_FREE_SPLIT", "NUMI_HUMAN_STAND_FREE_HANDOFF_PROBE",
    "NUMI_HUMAN_STAND_SPARSE_OPERATOR", "NUMI_HUMAN_STAND_SPARSE_CAPTURE_PATH",
    "NUMI_HUMAN_STAND_SPARSE_CAPTURE_ROOT", "NUMI_HUMAN_STAND_FORCE_PARITY_DIAGNOSTIC",
    "NUMI_HUMAN_STAND_SOURCE_ASSEMBLY_DUMP", "NUMI_HUMAN_SUPPORT_DIAGNOSTICS",
    "NUMI_HUMAN_STAND_COMPENSATED_BODY_SUM",
    "NUMI_HUMAN_STAND_DEFER_EQUALITY_DATA",
    "NUMI_HUMAN_STAND_EQUALITY_DEFERRAL_FAULT",
    "NUMI_HUMAN_STAND_DEFER_EQUALITY_DIAGNOSTICS",
    "NUMI_HUMAN_STAND_FINISH_COUNTER_ROOTS",
    "NUMI_HUMAN_STAND_REDUCED_RESPONSE_DIAGNOSTIC_ROOTS",
    "NUMI_HUMAN_STAND_REDUCED_BASE_PROJECTION",
    "NUMI_HUMAN_STAND_REDUCED_CHOLESKY",
    "NUMI_HUMAN_STAND_REDUCED_PROJECTED_RESPONSES",
    "NUMI_HUMAN_STAND_ANALYTIC_BODY_SPATIAL_JACOBIANS",
    "NUMI_HUMAN_STAND_CACHE_CONTACT_JACOBIAN",
    "NUMI_HUMAN_STAND_CONTACT_WARMSTART",
    "NUMI_HUMAN_STAND_FIRST_SIMD_CONTACT_SWEEP",
    "NUMI_HUMAN_STAND_FUSE_CANONICAL_BODY_PROBES",
    "NUMI_HUMAN_STAND_HYBRID_EQUALITY_FACTOR_CACHE",
    "NUMI_HUMAN_STAND_ONE_PASS_ORDERED_LIMITS",
    "NUMI_HUMAN_STAND_PGS_VELOCITY_RESIDUAL_TOLERANCE",
    "NUMI_HUMAN_STAND_SPECULATIVE_CONTACT_DISTANCE_M",
    "NUMI_HUMAN_STAND_ZERO_CONTACT_RESPONSE_FASTPATH",
    "NUMI_HUMAN_STAND_CPU_ACCELERATE_FACTOR", "NUMI_HUMAN_STAND_CPU_CONDITION_NEON",
    "NUMI_HUMAN_STAND_CPU_EQUALITY", "NUMI_HUMAN_STAND_CPU_EQUALITY_FACTOR",
    "NUMI_HUMAN_STAND_CPU_EQUALITY_SHADOW", "NUMI_HUMAN_STAND_CPU_FACTOR",
    "NUMI_HUMAN_STAND_CPU_FACTOR_SHADOW", "NUMI_HUMAN_STAND_CPU_FINISH",
    "NUMI_HUMAN_STAND_CPU_FREE", "NUMI_HUMAN_STAND_CPU_ONE_HANDOFF",
    "NUMI_HUMAN_STAND_CPU_PROJECTED_RAW", "NUMI_HUMAN_STAND_CPU_SHADOW",
    "NUMI_HUMAN_SKIN_INFLUENCE_TILE32",
    "NUMI_HUMAN_SUPPORT_ANCESTRY_PRUNE",
)


def invocation_environment(environment: dict[str, str]) -> dict[str, str]:
    """Retain every supported native setting that can affect a run."""
    return {key: environment[key] for key in INVOCATION_ENVIRONMENT_KEYS if key in environment}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("resting run: " + message)


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """Resolve only the existing named owner payloads; fail on identity drift."""
    scene = json.loads(args.body_scene.read_text())
    anatomy = json.loads(args.anatomy_receipt.read_text())
    _require(scene.get("schema") == "numi.human.resting-supine-source-scene.v1",
             "unsupported body scene receipt")
    _require(scene["pose"]["kind"] == "supine", "body seed is not supine")
    _require(anatomy.get("schema") == "numi.human.resting-anatomy-receipt.v1",
             "unsupported functional anatomy receipt")
    assets: dict[str, str] = {}

    def checked(path: Path | str, expected: str | None = None) -> Path:
        p = Path(path).resolve()
        _require(p.is_file(), f"missing owner file: {p}")
        digest = _hash(p)
        _require(expected is None or digest == expected, f"owner receipt hash differs: {p}")
        assets[str(p)] = digest
        return p

    rigid = checked(scene["source"]["rigid"]["path"], scene["source"]["rigid"]["sha256"])
    skin = checked(scene["source"]["skin"]["path"], scene["source"]["skin"]["sha256"])
    contact = checked(scene["outputs"]["support_contact"]["path"],
                      scene["outputs"]["support_contact"]["sha256"])
    # Individually valid receipts can still describe different adults/skin repairs.
    # Compare the loaded bytes with the anatomical accounting owner when declared;
    # equivalent copies at another path remain valid.
    for owner, path, record, key in (
        ("rigid", rigid, anatomy["provenance"], "rigid_payload_sha256"),
        ("skin", skin, anatomy["mass_geometry_accounting"], "skin_payload_sha256"),
    ):
        if key in record:
            expected = record[key]
            _require(isinstance(expected, str) and
                     re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                     f"invalid anatomy {owner} identity")
            _require(assets[str(path)] == expected,
                     f"body scene {owner} differs from anatomy receipt")
    bones = checked(anatomy["provenance"]["bones_payload"],
                    anatomy["functional_bindings"]["bones_payload_sha256"])
    organs = checked(anatomy["payload"]["path"], anatomy["payload"]["sha256"])
    common = anatomy["provenance"].get("cardiac_geometry_binding", {}).get("common_field")
    if common is not None:
        _require(isinstance(common, dict) and
                 common.get("schema") == "numi.human.cardiac_common_field.v1",
                 "unsupported common cardiac field receipt")
        for owner in ("map", "polynomials", "domain_boxes"):
            record = common.get(owner)
            _require(isinstance(record, dict) and
                     isinstance(record.get("path"), str) and bool(record["path"]) and
                     isinstance(record.get("sha256"), str) and
                     re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
                     f"missing common cardiac {owner} identity")
            owner_path = Path(record["path"])
            if not owner_path.is_absolute():
                owner_path = args.anatomy_receipt.parent / owner_path
            checked(owner_path, record["sha256"])
    muscles = checked(rigid.with_name("myosim-fullbody-muscle-reference.nhmyo"))
    equalities = checked(rigid.with_name("myosim-fullbody-joint-equalities.nheq"))
    muscle_surfaces = anatomy["provenance"].get("native_muscle_surfaces", {})
    surface_path = Path(muscle_surfaces.get("payload_path", "bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"))
    if not surface_path.is_absolute():
        surface_path = args.anatomy_receipt.parent / surface_path
    surfaces = checked(surface_path, muscle_surfaces.get("sha256"))
    if "manifest_path" in muscle_surfaces:
        checked(muscle_surfaces["manifest_path"], muscle_surfaces["manifest_sha256"])
    tendon = checked(args.tendon)
    binary = checked(args.build / "bin" / "numi-human-native")
    for relative in ("lib/libmetalrobo.dylib", "shaders/MetalRobo.metallib",
                     "shaders/MetalRoboHyperPolicy.metallib", "shaders/NumiNeuron.metallib",
                     "matter/shaders/HumanRespiration.metallib", "matter/shaders/NumiMatter.metallib",
                     "matter/shaders/NumiMatterPhysicalStateDigest.metallib"):
        checked(args.build / relative)
    network = checked(args.circulation or args.lab / "matter/tools/fixtures/cvsim21.native.v3.json")
    respiration = checked(getattr(args, "respiration", None) or
                          args.lab / "matter/examples/resting-reference-respiration.json")
    checked(args.body_scene); checked(args.anatomy_receipt)
    _require(math.isfinite(args.seconds) and args.seconds > 0, "duration must be positive")
    dt = getattr(args, "dt", .001)
    _require(math.isfinite(dt) and 0 < dt <= .002,
             "the resting native timestep must be positive and at most 2 ms")
    steps = round(args.seconds / dt)
    _require(0 < steps <= 4_000_000 and abs(steps * dt - args.seconds) < 1e-8,
             "duration must be an integer number of native steps (at most 4000000 steps)")
    pose = scene["pose"]["root_translation_xyz_m"] + scene["pose"]["root_delta_quaternion_xyzw"]
    _require(len(pose) == 7 and all(math.isfinite(x) for x in pose), "invalid root seed")
    mass = anatomy["mass_geometry_accounting"]["reference_total_mass_kg"]
    argv = [str(binary), str(rigid), str(muscles), str(bones), str(args.output.resolve()),
            "--persistent-metal-stand", "--muscle-step-seconds", str(dt),
            "--muscle-step-count", str(steps), "--support-contact-payload", str(contact),
            "--joint-equality-payload", str(equalities), "--tendon-payload", str(tendon),
            "--root-pose", *map(str, pose), "--resting-scene", str(network), str(respiration),
            "--vascular-dense45", "--resting-reference-mass-kg", str(mass),
            "--skin-payload", str(skin), "--soft-tissue-payload", str(surfaces),
            "--torso-anatomy-payload", str(organs), "--resting-anatomy-receipt", str(args.anatomy_receipt.resolve()),
            "--dimension", str(args.dimension)]
    if args.mechanics_only:
        argv.append("--mechanics-only")
    else:
        argv.extend(["--resting-movie", str(args.output.resolve() / "native-viewer.mov")])
    activation_cap = getattr(args, "postural_activation_cap", None)
    if activation_cap is not None:
        _require(math.isfinite(activation_cap) and 0 < activation_cap <= 1,
                 "postural recruitment activation cap must be finite and within (0, 1]")
        argv.extend(["--muscle-activation", str(activation_cap)])
    if getattr(args, "release_initialization", False):
        argv.append("--resting-release-initialization")
    if getattr(args, "contoured_bed", False):
        _require(getattr(args, "release_initialization", False),
                 "contoured bed requires explicit release initialization")
        heightfield = scene.get("bed", {}).get("heightfield")
        _require(isinstance(heightfield, dict) and heightfield.get("frame") == "fixed_world",
                 "contoured bed requires a fixed-world heightfield in the body scene")
        _require(isinstance(heightfield.get("source_capture_path"), str) and
                 re.fullmatch(r"[0-9a-f]{64}", str(heightfield.get("source_capture_sha256", ""))) is not None,
                 "contoured bed requires its source capture path and hash")
        checked(heightfield["source_capture_path"], heightfield["source_capture_sha256"])
        argv.extend(["--resting-bed-surface", str(args.body_scene.resolve())])
    if getattr(args, "upper_passive_joints", False):
        argv.append("--persistent-source-passive-joint-tissue")
    hip_capsule = getattr(args, "hip_capsule_reference", False)
    hip_scale = getattr(args, "hip_capsule_scale", None)
    _require(hip_scale is None or hip_capsule,
             "hip capsule scale requires the explicit hip capsule reference")
    if hip_capsule:
        hip_scale = 1.0 if hip_scale is None else hip_scale
        _require(type(hip_scale) in (float, int) and
                 math.isfinite(hip_scale) and hip_scale > 0,
                 "hip capsule reference scale must be finite and positive")
        argv.extend(["--resting-hip-capsule-reference",
                     "--resting-hip-capsule-scale", str(float(hip_scale))])
    if getattr(args, "rigid_hands", False):
        argv.append("--resting-rigid-hands")
    contact_iterations = getattr(args, "contact_iterations", None)
    if contact_iterations is not None:
        _require(type(contact_iterations) is int and 1 <= contact_iterations <= 64,
                 "contact iterations must be an integer within [1, 64]")
        argv.extend(["--stand-contact-iterations", str(contact_iterations)])
    if args.drive_intervention:
        start, end, scale = args.drive_intervention
        _require(all(math.isfinite(x) for x in (start, end, scale)) and
                 0 <= start < end < args.seconds and 0 <= scale <= 2,
                 "intervention must be bounded within the run, with a recovery interval")
        argv.extend(["--resting-drive-intervention", *map(str, (start, end, scale))])
    return argv, assets


def loaded_metal_runtime(log: Path, expected: Path, digest: str) -> dict:
    """Bind the actual dyld image to the already hashed native build."""
    pattern = re.compile(r"^dyld\[\d+\]: <([0-9A-Fa-f-]{36})> (/.*)$")
    images = set()
    with log.open(errors="replace") as stream:
        for line in stream:
            match = pattern.fullmatch(line.rstrip("\n"))
            if match and Path(match[2]).name == "libmetalrobo.dylib":
                images.add((str(Path(match[2]).resolve()), match[1].lower()))
    expected = expected.resolve()
    paths_match = len(images) == 1 and next(iter(images))[0] == str(expected)
    verified = paths_match and expected.is_file() and _hash(expected) == digest
    return {"expected_path": str(expected), "expected_sha256": digest,
            "observed_images": [{"path": path, "dyld_uuid": uuid}
                                for path, uuid in sorted(images)],
            "verified": verified,
            "scope": "Loaded MetalRobo image path and unchanged file hash; source and shader identities remain separate bindings"}


def run(args: argparse.Namespace) -> int:
    _require(platform.system() == "Darwin" and platform.machine() == "arm64",
             "the native scene requires an Apple silicon Mac")
    argv, assets = command(args)
    _require(not args.output.exists(), "output already exists; retain earlier evidence and choose a new directory")
    args.output.mkdir(parents=True)
    env = os.environ.copy()
    _require(not any(key.startswith("NUMI_HUMAN_STAND_CPU_") and
                         (value == "1" or key == "NUMI_HUMAN_STAND_CPU_ACCELERATE_FACTOR")
                         for key, value in env.items()),
             "the resting scene requires the resident GPU solver; a CPU solver experiment is enabled")
    env.update(NUMI_HUMAN_SPLIT_STAND="1",
               NUMI_HUMAN_EXECUTION_STAGES=env.get("NUMI_HUMAN_EXECUTION_STAGES", "1"),
               NUMI_HUMAN_TRAINING_PROFILE="1",
               DYLD_LIBRARY_PATH=os.pathsep.join(str((args.build / part).resolve())
                                                for part in ("lib", "matter")),
               DYLD_PRINT_LIBRARIES="1")
    if args.inspection_tour:
        _require(not args.mechanics_only, "an inspection tour requires the native viewer")
        _require(math.isfinite(args.inspection_period_seconds) and args.inspection_period_seconds > 0,
                 "inspection period must be positive finite seconds")
        env["NUMI_HUMAN_RESTING_INSPECTION_TOUR"] = "1"
        env["NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS"] = str(args.inspection_period_seconds)
    receipt = {"argv": argv, "asset_sha256": assets,
               "environment": invocation_environment(env),
               "host": platform.node(), "machine": platform.machine(), "system": platform.platform(),
               "qualification": "native execution receipt; physiological and anatomical acceptance require separate audits"}
    (args.output / "invocation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    start = time.monotonic()
    with (args.output / "native.log").open("w") as log:
        result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    changed = [path for path, digest in assets.items() if not Path(path).is_file() or _hash(Path(path)) != digest]
    runtime_path = (args.build / "lib/libmetalrobo.dylib").resolve()
    runtime = loaded_metal_runtime(args.output / "native.log", runtime_path,
                                   assets[str(runtime_path)])
    receipt.update(exit_code=result.returncode, wall_seconds=time.monotonic() - start,
                   source_files_changed_during_run=changed, loaded_metal_runtime=runtime)
    (args.output / "run-metadata.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"exit_code": result.returncode, "output": str(args.output),
                      "changed_sources": changed, "loaded_runtime_verified": runtime["verified"]}))
    return result.returncode if result.returncode else (1 if changed or not runtime["verified"] else 0)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--body-scene", type=Path, required=True)
    parser.add_argument("--anatomy-receipt", type=Path, required=True)
    parser.add_argument("--tendon", type=Path, required=True)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--circulation", type=Path,
                        help="explicit native circulation owner payload; default is the retained upstream CVSim21 variant")
    parser.add_argument("--respiration", type=Path,
                        help="explicit native respiratory reference parameters, including the source-derived diaphragm area")
    parser.add_argument("--seconds", type=float, default=310,
                        help="physical duration; default permits 10 s initialization plus 300 s observation")
    parser.add_argument("--dt", type=float, default=.001,
                        help="physical timestep in seconds, at most .002; default .001")
    parser.add_argument("--dimension", type=int, choices=(512, 768, 1024), default=512)
    parser.add_argument("--mechanics-only", action="store_true", help="diagnostic without the native viewer or movie")
    parser.add_argument("--inspection-tour", action="store_true",
                        help="cycle the native anatomical layers every five simulated seconds without changing physics")
    parser.add_argument("--inspection-period-seconds", type=float, default=5.0,
                        help="simulated seconds per anatomical layer during the presentation-only inspection tour")
    parser.add_argument("--drive-intervention", type=float, nargs=3, metavar=("START", "END", "SCALE"))
    parser.add_argument("--postural-activation-cap", type=float,
                        help="cap initial source muscle recruitment retained as postural drive; default uses the native owner setting")
    parser.add_argument("--release-initialization", action="store_true",
                        help="explicitly initialize outside static equilibrium and let native bed contact settle the body")
    parser.add_argument("--contoured-bed", action="store_true",
                        help="use the fixed finite bed heightfield declared in the existing body scene manifest")
    parser.add_argument("--upper-passive-joints", action="store_true",
                        help="enable the existing source-bound wrist and non-thumb finger passive stiffness model")
    parser.add_argument("--hip-capsule-reference", action="store_true",
                        help="enable the reduced literature reference for passive hip periarticular restraint")
    parser.add_argument("--hip-capsule-scale", type=float,
                        help="positive potential/force/stiffness scale for declared sensitivity; requires --hip-capsule-reference")
    parser.add_argument("--rigid-hands", action="store_true",
                        help="use reference-pose rigid digits with internal GPU constraints; wrists remain free, and hand physiology is not simulated")
    parser.add_argument("--contact-iterations", type=int,
                        help="existing GPU contact/equality/limit sweeps per physical step, 1 to 64; default uses the native owner setting")
    parser.set_defaults(handler=run)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    raise SystemExit(run(p.parse_args()))
