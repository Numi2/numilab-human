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
import subprocess
import time

from .model import ImportError as HumanImportError


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
    bones = checked(anatomy["provenance"]["bones_payload"],
                    anatomy["functional_bindings"]["bones_payload_sha256"])
    organs = checked(anatomy["payload"]["path"], anatomy["payload"]["sha256"])
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
    if args.drive_intervention:
        start, end, scale = args.drive_intervention
        _require(all(math.isfinite(x) for x in (start, end, scale)) and
                 0 <= start < end < args.seconds and 0 <= scale <= 2,
                 "intervention must be bounded within the run, with a recovery interval")
        argv.extend(["--resting-drive-intervention", *map(str, (start, end, scale))])
    return argv, assets


def run(args: argparse.Namespace) -> int:
    _require(platform.system() == "Darwin" and platform.machine() == "arm64",
             "the native scene requires an Apple silicon Mac")
    argv, assets = command(args)
    _require(not args.output.exists(), "output already exists; retain earlier evidence and choose a new directory")
    args.output.mkdir(parents=True)
    env = os.environ.copy()
    _require(not any(key.startswith("NUMI_HUMAN_STAND_CPU_") and value == "1"
                     for key, value in env.items()),
             "the resting scene requires the resident GPU solver; a CPU solver experiment is enabled")
    env.update(NUMI_HUMAN_SPLIT_STAND="1", NUMI_HUMAN_EXECUTION_STAGES="1",
               NUMI_HUMAN_TRAINING_PROFILE="1")
    if args.inspection_tour:
        _require(not args.mechanics_only, "an inspection tour requires the native viewer")
        _require(math.isfinite(args.inspection_period_seconds) and args.inspection_period_seconds > 0,
                 "inspection period must be positive finite seconds")
        env["NUMI_HUMAN_RESTING_INSPECTION_TOUR"] = "1"
        env["NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS"] = str(args.inspection_period_seconds)
    receipt = {"argv": argv, "asset_sha256": assets,
               "environment": {k: env[k] for k in (
                   "NUMI_HUMAN_SPLIT_STAND", "NUMI_HUMAN_EXECUTION_STAGES",
                   "NUMI_HUMAN_TRAINING_PROFILE", "NUMI_HUMAN_RESTING_TRANSACTION_PROBE",
                   "NUMI_HUMAN_RESTING_INSPECTION_TOUR",
                   "NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS",
                   "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS",
                   "NUMI_HUMAN_GPU_TIMING", "NUMI_HUMAN_GPU_TIMING_STAGE",
                   "NUMI_MATTER_GPU_TIMING", "NUMI_MATTER_GPU_TIMING_STAGE",
                   "NUMI_HUMAN_SUPPORT_DIAGNOSTICS", "NUMI_HUMAN_SUPPORT_GPU_TIMING",
                   "NUMI_HUMAN_PARALLEL_MASS_ASSEMBLY", "NUMI_HUMAN_KINEMATICS_CACHE",
                   "NUMI_HUMAN_OVERLAP_GEOMETRY", "NUMI_HUMAN_STAND_CACHE_LIMIT_EQUALITY",
                   "NUMI_HUMAN_STAND_FREE_SPLIT", "NUMI_HUMAN_STAND_SPARSE_OPERATOR",
               ) if k in env},
               "host": platform.node(), "machine": platform.machine(), "system": platform.platform(),
               "qualification": "native execution receipt; physiological and anatomical acceptance require separate audits"}
    (args.output / "invocation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    start = time.monotonic()
    with (args.output / "native.log").open("w") as log:
        result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    changed = [path for path, digest in assets.items() if not Path(path).is_file() or _hash(Path(path)) != digest]
    receipt.update(exit_code=result.returncode, wall_seconds=time.monotonic() - start,
                   source_files_changed_during_run=changed)
    (args.output / "run-metadata.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"exit_code": result.returncode, "output": str(args.output), "changed_sources": changed}))
    return result.returncode if result.returncode else (1 if changed else 0)


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
    parser.set_defaults(handler=run)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    raise SystemExit(run(p.parse_args()))
