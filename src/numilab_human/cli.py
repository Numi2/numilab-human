from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from .model import (
    ImportError,
    bodyparts_geometry_preflight,
    bodyparts_lower_body_attachment_worklist,
    bodyparts_nerve_annotation,
    bodyparts_visual_preview,
    bodyparts_visual_layer_previews,
    build_rajagopal_distal_pin_preview,
    build_manifest,
    gate_report,
    parse_bodyparts3d,
    parse_opensim,
    rajagopal_core_reference_artifact,
    rajagopal_custom_joint_gpu_artifacts,
    rajagopal_custom_joint_ir,
    rajagopal_millard_muscle_ir,
    rajagopal_millard_reference_artifact,
    rajagopal_rigid_skeleton_ir,
    rajagopal_walking_contract,
    read_json,
    report_for,
    sha256,
    write_json,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    lock = target.with_name(target.name + ".lock")
    descriptor: int | None = None
    for attempt in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, str(os.getpid()).encode("ascii"))
            break
        except FileExistsError as error:
            try:
                owner = int(lock.read_text(encoding="ascii").strip())
                os.kill(owner, 0)
            except (FileNotFoundError, ProcessLookupError, ValueError):
                lock.unlink(missing_ok=True)
                if attempt == 0:
                    continue
            raise ImportError(f"another fetch owns {target}; wait for it to finish") from error
    if descriptor is None:
        raise ImportError(f"could not acquire fetch lock for {target}")
    try:
        os.close(descriptor)
        resume_at = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": "numilab-human/0.1"}
        if resume_at:
            headers["Range"] = f"bytes={resume_at}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            resumed = response.status == 206 and resume_at > 0
            mode = "ab" if resumed else "wb"
            with temporary.open(mode) as stream:
                shutil.copyfileobj(response, stream)
        if resume_at and not resumed:
            print(f"server did not honour resume for {target.name}; fetched a clean copy")
        temporary.replace(target)
    finally:
        lock.unlink(missing_ok=True)


def fetch(arguments: argparse.Namespace) -> int:
    source_lock = read_json(REPOSITORY_ROOT / "sources.lock.json")
    source_dir = arguments.output.resolve()
    bodyparts = source_lock["sources"]["bodyparts3d_4"]
    for filename, metadata in bodyparts["files"].items():
        target = source_dir / filename
        if not target.is_file():
            print(f"fetching {filename}")
            _download(bodyparts["base_url"] + filename, target)
        expected_bytes = metadata.get("bytes")
        if expected_bytes is not None and target.stat().st_size != expected_bytes:
            raise ImportError(
                f"byte-size mismatch for {target}; expected {expected_bytes}, "
                f"got {target.stat().st_size}"
            )
        expected = metadata["sha256"]
        actual = sha256(target)
        if expected and actual != expected:
            raise ImportError(f"SHA-256 mismatch for {target}; remove it and fetch again")
        print(f"verified {filename} {actual}")
    rajagopal = source_lock["sources"]["rajagopal_lai_uhlrich_2023"]
    target = source_dir / "RajagopalLaiUhlrich2023.osim"
    if not target.is_file():
        print("fetching RajagopalLaiUhlrich2023.osim")
        _download(rajagopal["url"], target)
    actual = sha256(target)
    if actual != rajagopal["sha256"]:
        raise ImportError("SHA-256 mismatch for RajagopalLaiUhlrich2023.osim")
    print(f"verified {target.name} {actual}")
    write_json(
        source_dir / "sources.receipt.json",
        {
            "schema": "numi.human.fetch-receipt.v1",
            "source_lock": str((REPOSITORY_ROOT / "sources.lock.json").resolve()),
            "bodyparts_attribution": bodyparts["attribution"],
            "upper_extremity_next_step": "Manually download the original authenticated MoBL-ARMS bimanual archive from SimTK.",
        },
    )
    return 0


def build(arguments: argparse.Namespace) -> int:
    if not arguments.accept_upper_noncommercial_terms:
        raise ImportError(
            "MoBL-ARMS official terms restrict this source to non-commercial use. "
            "Re-run with --accept-upper-noncommercial-terms after reviewing THIRD_PARTY_NOTICES.md"
        )
    upper_archive = arguments.upper_archive.resolve()
    if not upper_archive.is_file():
        raise ImportError(f"upper archive does not exist: {upper_archive}")
    manifest = build_manifest(
        sources=arguments.sources.resolve(),
        upper_archive=upper_archive,
        classification_path=REPOSITORY_ROOT / "config/anatomy-classification.v1.json",
        target_mapping_path=REPOSITORY_ROOT / "config/numi-targets.v1.json",
        source_lock=read_json(REPOSITORY_ROOT / "sources.lock.json"),
    )
    output = arguments.output.resolve()
    write_json(output / "human.v1.json", manifest)
    write_json(output / "report.json", report_for(manifest))
    print(f"wrote {output / 'human.v1.json'}")
    print(f"wrote {output / 'report.json'}")
    return 0


def audit(arguments: argparse.Namespace) -> int:
    report = gate_report(
        sources=arguments.sources.resolve(),
        upper_archive=(arguments.upper_archive.resolve() if arguments.upper_archive else None),
        source_lock=read_json(REPOSITORY_ROOT / "sources.lock.json"),
        runtime_contract=read_json(REPOSITORY_ROOT / "config/numi-runtime-contract.v1.json"),
        runtime_root=(arguments.runtime_root.resolve() if arguments.runtime_root else None),
    )
    if arguments.output:
        write_json(arguments.output.resolve(), report)
        print(f"wrote {arguments.output.resolve()}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def geometry_audit(arguments: argparse.Namespace) -> int:
    anatomy = parse_bodyparts3d(
        arguments.sources.resolve(),
        REPOSITORY_ROOT / "config/anatomy-classification.v1.json",
    )
    report = bodyparts_geometry_preflight(arguments.sources.resolve(), anatomy)
    if arguments.output:
        write_json(arguments.output.resolve(), report)
        print(f"wrote {arguments.output.resolve()}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def nerve_annotations(arguments: argparse.Namespace) -> int:
    anatomy = parse_bodyparts3d(
        arguments.sources.resolve(),
        REPOSITORY_ROOT / "config/anatomy-classification.v1.json",
    )
    annotation = bodyparts_nerve_annotation(anatomy)
    output = arguments.output.resolve()
    write_json(output, annotation)
    print(f"wrote {output}")
    print(
        f"annotated {annotation['component_count']} nerve components and "
        f"{annotation['hierarchy_edge_count']} source hierarchy edges"
    )
    return 0


def preview(arguments: argparse.Namespace) -> int:
    source = arguments.sources.resolve()
    lower = parse_opensim(
        source / "RajagopalLaiUhlrich2023.osim",
        "rajagopal_lai_uhlrich_2023",
    )
    urdf, report = build_rajagopal_distal_pin_preview(lower, arguments.side)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    urdf_path = output / f"rajagopal-{arguments.side}-distal-pin-preview.urdf"
    report_path = output / f"rajagopal-{arguments.side}-distal-pin-preview.json"
    urdf_path.write_text(urdf, encoding="utf-8")
    write_json(report_path, report)
    print(f"wrote {urdf_path}")
    print(f"wrote {report_path}")
    return 0


def kinematics(arguments: argparse.Namespace) -> int:
    source = arguments.sources.resolve()
    lower = parse_opensim(
        source / "RajagopalLaiUhlrich2023.osim",
        "rajagopal_lai_uhlrich_2023",
    )
    output = arguments.output.resolve()
    report_path = output / "rajagopal-custom-joint-ir.json"
    write_json(report_path, rajagopal_custom_joint_ir(lower))
    artifacts_manifest, artifacts = rajagopal_custom_joint_gpu_artifacts(lower)
    for relative_path, content in artifacts.items():
        target = output / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    program_manifest_path = output / "opensim-spatial-programs.manifest.json"
    write_json(program_manifest_path, artifacts_manifest)
    print(f"wrote {report_path}")
    print(f"wrote {program_manifest_path}")
    print(f"wrote {len(artifacts)} OpenSim spatial-transform binary sidecars")
    return 0


def muscles(arguments: argparse.Namespace) -> int:
    source = arguments.sources.resolve()
    lower = parse_opensim(
        source / "RajagopalLaiUhlrich2023.osim",
        "rajagopal_lai_uhlrich_2023",
    )
    output = arguments.output.resolve()
    write_json(output, rajagopal_millard_muscle_ir(lower))
    print(f"wrote {output}")
    return 0


def visual_preview(arguments: argparse.Namespace) -> int:
    output = arguments.output.resolve()
    manifest = bodyparts_visual_preview(
        arguments.sources.resolve(),
        output,
        archive_kind=arguments.archive,
        member_id=arguments.member,
    )
    print(f"wrote {output / manifest['preview']['glb']}")
    print(f"wrote {output / (arguments.member + '-source-static.manifest.json')}")
    return 0


def millard_reference(arguments: argparse.Namespace) -> int:
    source = arguments.sources.resolve()
    lower = parse_opensim(
        source / "RajagopalLaiUhlrich2023.osim",
        "rajagopal_lai_uhlrich_2023",
    )
    output = arguments.output.resolve()
    manifest, payload = rajagopal_millard_reference_artifact(lower)
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "rajagopal-millard-reference.nhmuscle"
    payload_path.write_bytes(payload)
    manifest_path = output / "rajagopal-millard-reference.manifest.json"
    write_json(manifest_path, manifest)
    print(f"wrote {payload_path}")
    print(f"wrote {manifest_path}")
    return 0


def skeleton(arguments: argparse.Namespace) -> int:
    source = arguments.sources.resolve()
    lower = parse_opensim(
        source / "RajagopalLaiUhlrich2023.osim",
        "rajagopal_lai_uhlrich_2023",
    )
    output = arguments.output.resolve()
    write_json(output, rajagopal_rigid_skeleton_ir(lower))
    print(f"wrote {output}")
    return 0


def core_reference(arguments: argparse.Namespace) -> int:
    source = arguments.sources.resolve()
    lower = parse_opensim(
        source / "RajagopalLaiUhlrich2023.osim",
        "rajagopal_lai_uhlrich_2023",
    )
    output = arguments.output.resolve()
    manifest, payload = rajagopal_core_reference_artifact(lower)
    output.mkdir(parents=True, exist_ok=True)
    payload_path = output / "rajagopal-core-reference.nhrigid"
    payload_path.write_bytes(payload)
    manifest_path = output / "rajagopal-core-reference.manifest.json"
    write_json(manifest_path, manifest)
    print(f"wrote {payload_path}")
    print(f"wrote {manifest_path}")
    return 0


def walking_contract(arguments: argparse.Namespace) -> int:
    source = arguments.sources.resolve()
    lower = parse_opensim(source / "RajagopalLaiUhlrich2023.osim", "rajagopal_lai_uhlrich_2023")
    output = arguments.output.resolve()
    write_json(output, rajagopal_walking_contract(lower))
    print(f"wrote {output}")
    return 0

def attachment_worklist(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    lower = parse_opensim(sources / "RajagopalLaiUhlrich2023.osim", "rajagopal_lai_uhlrich_2023")
    output = arguments.output.resolve()
    write_json(output, bodyparts_lower_body_attachment_worklist(anatomy, lower))
    print(f"wrote {output}")
    return 0

def visual_layers(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    output = arguments.output.resolve()
    write_json(output / "visual-layers.manifest.json", bodyparts_visual_layer_previews(sources, output, anatomy))
    print(f"wrote {output / 'visual-layers.manifest.json'}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Build provenance-locked NumiLab Human v1 import artifacts")
    commands = result.add_subparsers(dest="command", required=True)
    fetch_parser = commands.add_parser("fetch", help="fetch BodyParts3D 4.0 and the pinned lower-body model")
    fetch_parser.add_argument("--output", type=Path, required=True, help="local, ignored source directory")
    fetch_parser.set_defaults(handler=fetch)
    build_parser = commands.add_parser("build", help="combine exact sources into a local audit manifest")
    build_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    build_parser.add_argument("--upper-archive", type=Path, required=True, help="original SimTK MoBL-ARMS bimanual ZIP")
    build_parser.add_argument("--output", type=Path, required=True, help="local output directory")
    build_parser.add_argument("--accept-upper-noncommercial-terms", action="store_true")
    build_parser.set_defaults(handler=build)
    audit_parser = commands.add_parser("audit", help="report every source, runtime, and physics gate")
    audit_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    audit_parser.add_argument("--upper-archive", type=Path, help="original SimTK MoBL-ARMS bimanual ZIP")
    audit_parser.add_argument(
        "--runtime-root",
        type=Path,
        help="optional checked-out MetalRobo root to verify against the pinned runtime revision",
    )
    audit_parser.add_argument("--output", type=Path, help="optional JSON report path")
    audit_parser.set_defaults(handler=audit)
    geometry_parser = commands.add_parser(
        "geometry-audit",
        help="fingerprint BodyParts3D OBJ members and conservatively preflight topology",
    )
    geometry_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    geometry_parser.add_argument("--output", type=Path, help="optional JSON report path")
    geometry_parser.set_defaults(handler=geometry_audit)
    nerve_parser = commands.add_parser(
        "nerve-annotations",
        help="emit BodyParts3D nerve labels, meshes, and source hierarchy as annotations only",
    )
    nerve_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    nerve_parser.add_argument("--output", type=Path, required=True, help="annotation JSON output path")
    nerve_parser.set_defaults(handler=nerve_annotations)
    visual_preview_parser = commands.add_parser(
        "visual-preview",
        help="export one exact BodyParts3D surface as a static non-physical GLB preview",
    )
    visual_preview_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    visual_preview_parser.add_argument("--output", type=Path, required=True, help="ignored local output directory")
    visual_preview_parser.add_argument("--archive", choices=("is_a", "part_of"), default="is_a")
    visual_preview_parser.add_argument("--member", default="FJ2810", help="BodyParts3D OBJ member identity")
    visual_preview_parser.set_defaults(handler=visual_preview)
    preview_parser = commands.add_parser(
        "preview",
        help="emit an explicitly limited Rajagopal distal-leg URDF compile preview",
    )
    preview_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    preview_parser.add_argument("--side", choices=("right", "left"), default="right")
    preview_parser.add_argument("--output", type=Path, required=True, help="ignored local preview directory")
    preview_parser.set_defaults(handler=preview)
    kinematics_parser = commands.add_parser(
        "kinematics",
        help="emit source-faithful Rajagopal CustomJoint IR, Core programs, and test inputs",
    )
    kinematics_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    kinematics_parser.add_argument("--output", type=Path, required=True, help="ignored local output directory")
    kinematics_parser.set_defaults(handler=kinematics)
    muscle_parser = commands.add_parser(
        "muscles",
        help="emit the exact Rajagopal Millard muscle, curve, path, and wrap IR",
    )
    muscle_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    muscle_parser.add_argument("--output", type=Path, required=True, help="muscle IR JSON output path")
    muscle_parser.set_defaults(handler=muscles)
    millard_reference_parser = commands.add_parser(
        "millard-reference",
        help="compile all Rajagopal Millard body-frame paths and cylinders for the Core reference",
    )
    millard_reference_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    millard_reference_parser.add_argument(
        "--output", type=Path, required=True, help="ignored local output directory"
    )
    millard_reference_parser.set_defaults(handler=millard_reference)
    skeleton_parser = commands.add_parser(
        "skeleton",
        help="emit exact Rajagopal rigid-body and resolved joint-topology IR",
    )
    skeleton_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    skeleton_parser.add_argument("--output", type=Path, required=True, help="skeleton IR JSON output path")
    skeleton_parser.set_defaults(handler=skeleton)
    core_reference_parser = commands.add_parser(
        "core-reference",
        help="compile the full Rajagopal rigid tree for the Core FP64 FunctionBased reference",
    )
    core_reference_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    core_reference_parser.add_argument(
        "--output", type=Path, required=True, help="ignored local output directory"
    )
    core_reference_parser.set_defaults(handler=core_reference)
    walking_parser = commands.add_parser(
        "walking-contract",
        help="emit the source-backed mobile-root, learned-excitation, and contact/visual gate contract",
    )
    walking_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    walking_parser.add_argument("--output", type=Path, required=True, help="walking contract JSON output path")
    walking_parser.set_defaults(handler=walking_contract)
    attachment_parser = commands.add_parser("attachment-worklist", help="emit review-only lower-body BodyParts3D attachment and foot-collider work items")
    attachment_parser.add_argument("--sources", type=Path, required=True)
    attachment_parser.add_argument("--output", type=Path, required=True)
    attachment_parser.set_defaults(handler=attachment_worklist)
    layers_parser = commands.add_parser("visual-layers", help="export exact source-static previews for the five requested anatomy layers")
    layers_parser.add_argument("--sources", type=Path, required=True)
    layers_parser.add_argument("--output", type=Path, required=True)
    layers_parser.set_defaults(handler=visual_layers)
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        return arguments.handler(arguments)
    except ImportError as error:
        print(f"numilab-human: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"numilab-human: I/O failure: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
