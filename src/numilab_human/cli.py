from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from .model import (
    ImportError,
    bodyparts_foot_collider_preflight,
    bodyparts_foot_registration_receipt_template,
    validate_bodyparts_foot_registration_receipt,
    bodyparts_geometry_preflight,
    bodyparts_foot_registration_template,
    bodyparts_lower_body_attachment_worklist,
    bodyparts_right_calcaneal_tendon_continuity_preview,
    bodyparts_right_lower_leg_anatomy_preview,
    bodyparts_myosim_bone_visual_payload,
    bodyparts_myosim_fullbody_soft_tissue_visual_payload,
    bodyparts_myosim_torso_anatomy_visual_payload,
    bodyparts_myosim_skinned_shell_visual_payload,
    bodyparts_myosim_right_posterior_chain_visual_payload,
    bodyparts_pectoralis_fascia_payload,
    bodyparts_costal_cartilage_payload,
    anterior_thorax_composite_payload,
    bodyparts_myosim_attachment_surface_registration_candidate,
    bodyparts_myosim_registration_candidate,
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
    rajagopal_lower_body_pilot,
    rajagopal_rigid_skeleton_ir,
    rajagopal_walking_contract,
    myosim_fullbody_reference_artifacts,
    myosim_part_control_catalog,
    myosim_part_control_plan,
    numi_human_tendon_attachment_envelope_payload,
    numi_human_tendon_endpoint_payload,
    numi_human_achilles_surface_receipt,
    mortensen_neck_source_ir,
    read_json,
    report_for,
    sha256,
    write_json,
)
from .zanatomy import build_zanatomy_calf_visual_supplement_payload
from .lower_limb_registration import propose_lower_limb_registration
from .open_knee import SCHEMA as OPEN_KNEE_SCHEMA
from .thoracic_registration import SCHEMA as THORACIC_REGISTRATION_SCHEMA
from .pelvis_registration import SCHEMA as PELVIS_REGISTRATION_SCHEMA
from .rib_registration import SCHEMA as RIB_REGISTRATION_SCHEMA
from .abdominal_enthesis_registration import (
    SCHEMA as ABDOMINAL_ENTHESIS_REGISTRATION_SCHEMA,
)
from .sternal_girdle_registration import (
    SCHEMA as STERNAL_GIRDLE_REGISTRATION_SCHEMA,
    register_sternal_girdle,
)
from .myosim_bone_proximity import (
    AUDIT_SCHEMA as MYOSIM_SOURCE_BONE_PROXIMITY_SCHEMA,
    WORKLIST_SCHEMA as BODYPARTS_REGISTRATION_WORKLIST_SCHEMA,
    registration_worklist,
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
    public_mobl_receipt: dict[str, str] | None = None
    if arguments.include_public_mobl_41:
        if not arguments.accept_upper_noncommercial_terms:
            raise ImportError(
                "The public MoBL-ARMS 4.1 mirror remains non-commercial. "
                "Re-run with --accept-upper-noncommercial-terms after reviewing THIRD_PARTY_NOTICES.md"
            )
        public_mobl = source_lock["sources"].get("mobl_arms_ceinms_41_public_mirror")
        if not isinstance(public_mobl, dict):
            raise ImportError("sources.lock.json has no public MoBL-ARMS 4.1 mirror entry")
        model_file, url, expected = (
            public_mobl.get("model_file"), public_mobl.get("url"), public_mobl.get("sha256"),
        )
        if not all(isinstance(value, str) and value for value in (model_file, url, expected)):
            raise ImportError("public MoBL-ARMS 4.1 source-lock entry is incomplete")
        public_target = source_dir / model_file
        if not public_target.is_file():
            print(f"fetching {model_file}")
            _download(url, public_target)
        public_actual = sha256(public_target)
        if public_actual != expected:
            raise ImportError(f"SHA-256 mismatch for {model_file}; remove it and fetch again")
        public_mobl_receipt = {
            "variant": "public_unimanual_mirror",
            "file": model_file,
            "sha256": public_actual,
            "repository": str(public_mobl.get("repository")),
            "revision": str(public_mobl.get("revision")),
        }
        print(f"verified {model_file} {public_actual}")
    write_json(
        source_dir / "sources.receipt.json",
        {
            "schema": "numi.human.fetch-receipt.v1",
            "source_lock": str((REPOSITORY_ROOT / "sources.lock.json").resolve()),
            "bodyparts_attribution": bodyparts["attribution"],
            "upper_extremity_next_step": "Manually download the original authenticated MoBL-ARMS bimanual archive from SimTK.",
            "public_mobl_41": public_mobl_receipt,
        },
    )
    return 0


def _extract_pinned_tarball(archive: Path, destination: Path) -> None:
    """Extract one GitHub source archive without accepting archive traversal."""
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            roots = {member.name.split("/", 1)[0] for member in members if member.name}
            if len(roots) != 1:
                raise ImportError(f"{archive.name} must have exactly one source root")
            root = next(iter(roots))
            for member in members:
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                    raise ImportError(f"{archive.name} contains an unsafe source member")
            temporary = destination.with_name(destination.name + ".partial")
            shutil.rmtree(temporary, ignore_errors=True)
            temporary.mkdir(parents=True, exist_ok=False)
            for member in members:
                relative = Path(member.name).relative_to(root)
                if not relative.parts:
                    continue
                target = temporary / relative
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise ImportError(f"{archive.name} could not extract {member.name}")
                with target.open("wb") as stream:
                    shutil.copyfileobj(extracted, stream)
    except tarfile.TarError as error:
        raise ImportError(f"{archive.name} is not a readable gzip tar archive: {error}") from error
    shutil.rmtree(destination, ignore_errors=True)
    temporary.replace(destination)


def myosim_fetch(arguments: argparse.Namespace) -> int:
    source_lock = read_json(REPOSITORY_ROOT / "sources.lock.json")
    source_dir = arguments.output.resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    for key in ("myosim_fullbody", "mortensen_2018_neck"):
        source = source_lock["sources"][key]
        storage = source.get("storage_dir")
        if not isinstance(storage, str) or not storage:
            raise ImportError(f"source lock {key} has no storage directory")
        archive = source_dir / storage / source["archive_file"]
        if not archive.is_file():
            print(f"fetching {archive.name}")
            _download(source["archive_url"], archive)
        actual = sha256(archive)
        if actual != source["archive_sha256"]:
            raise ImportError(f"SHA-256 mismatch for {archive}; remove it and re-run myosim-fetch")
        checkout = archive.parent / "checkout"
        expected = checkout / source["expected_file"]
        if not expected.is_file():
            print(f"extracting {archive.name}")
            _extract_pinned_tarball(archive, checkout)
        if not expected.is_file():
            raise ImportError(f"{archive.name} did not contain expected source {source['expected_file']}")
        print(f"verified {key} {actual}")
    return 0


def myosim_build(arguments: argparse.Namespace) -> int:
    source_dir = arguments.sources.resolve()
    # Do not resolve the interpreter symlink: virtual-environment ``python``
    # links point at the base interpreter, and resolving them drops site-packages.
    exporter = arguments.python.expanduser().absolute()
    if not exporter.is_file() or not os.access(exporter, os.X_OK):
        raise ImportError(f"MyoSim exporter Python is unavailable: {exporter}")
    with tempfile.TemporaryDirectory(prefix="numilab-human-myosim-") as temporary:
        exported_path = Path(temporary) / "myosim-export.json"
        environment = dict(os.environ)
        source_path = str(REPOSITORY_ROOT / "src")
        environment["PYTHONPATH"] = source_path + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
        command = [
            str(exporter), "-m", "numilab_human.myosim_export",
            "--sources", str(source_dir), "--output", str(exported_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no exporter output"
            raise ImportError(f"MyoSim source export failed: {detail}")
        exported = read_json(exported_path)
    (
        manifest, rigid_payload, muscle_payload, support_payload,
        equality_payload,
    ) = myosim_fullbody_reference_artifacts(exported)
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rigid = output / manifest["payloads"]["rigid"]["file"]
    muscle = output / manifest["payloads"]["muscles"]["file"]
    support = output / manifest["payloads"]["support_contact"]["file"]
    equalities = output / manifest["payloads"]["joint_equalities"]["file"]
    rigid.write_bytes(rigid_payload)
    muscle.write_bytes(muscle_payload)
    support.write_bytes(support_payload)
    equalities.write_bytes(equality_payload)
    write_json(output / "myosim-fullbody-reference.manifest.json", manifest)
    print(f"wrote {rigid}")
    print(f"wrote {muscle}")
    print(f"wrote {support}")
    print(f"wrote {equalities}")
    print(f"wrote {output / 'myosim-fullbody-reference.manifest.json'}")
    return 0


def myosim_source_bone_proximity(arguments: argparse.Namespace) -> int:
    source_dir = arguments.sources.resolve()
    # Keep MuJoCo and MyoSim isolated in their pinned source environment.
    # Resolving a venv interpreter symlink would discard that environment.
    exporter = arguments.python.expanduser().absolute()
    if not exporter.is_file() or not os.access(exporter, os.X_OK):
        raise ImportError(f"MyoSim audit Python is unavailable: {exporter}")
    output = arguments.output.resolve()
    environment = dict(os.environ)
    source_path = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        str(exporter), "-m", "numilab_human.myosim_bone_proximity",
        "--sources", str(source_dir), "--output", str(output),
        "--maximum-distance", repr(arguments.maximum_distance),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no audit output"
        raise ImportError(f"MyoSim source-bone proximity audit failed: {detail}")
    audit = read_json(output)
    if audit.get("schema") != MYOSIM_SOURCE_BONE_PROXIMITY_SCHEMA:
        raise ImportError("MyoSim source-bone proximity audit wrote an unsupported schema")
    print(f"wrote {output}")
    return 0


def bodyparts_registration_worklist(arguments: argparse.Namespace) -> int:
    source_audit_path = arguments.source_audit.resolve()
    tendon_manifest_path = arguments.tendon_manifest.resolve()
    output = arguments.output.resolve()
    try:
        result = registration_worklist(
            read_json(source_audit_path),
            read_json(tendon_manifest_path),
            source_audit_file=source_audit_path,
            tendon_manifest_file=tendon_manifest_path,
        )
    except ValueError as error:
        raise ImportError(str(error)) from error
    if result.get("schema") != BODYPARTS_REGISTRATION_WORKLIST_SCHEMA:
        raise ImportError("BodyParts3D registration worklist has an unsupported schema")
    write_json(output, result)
    print(f"wrote {output}")
    return 0


def myosim_upper_limb_registration(arguments: argparse.Namespace) -> int:
    exporter = arguments.python.expanduser().absolute()
    if not exporter.is_file() or not os.access(exporter, os.X_OK):
        raise ImportError(f"MyoSim upper-limb registration Python is unavailable: {exporter}")
    output = arguments.output.resolve()
    environment = dict(os.environ)
    source_path = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        str(exporter), "-m", "numilab_human.upper_limb_registration",
        "--sources", str(arguments.sources.resolve()),
        "--registration", str(arguments.registration.resolve()),
        "--source-audit", str(arguments.source_audit.resolve()),
        "--worklist", str(arguments.worklist.resolve()),
        "--tendon-manifest", str(arguments.tendon_manifest.resolve()),
        "--output", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no registration output"
        raise ImportError(f"MyoSim upper-limb registration failed: {detail}")
    candidate = read_json(output)
    receipt = candidate.get("upper_limb_source_mesh_registration")
    if (
        candidate.get("schema") != "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
        or not isinstance(receipt, dict)
        or receipt.get("schema") != "numi.human.bodyparts3d-myosim-upper-limb-source-mesh-registration.v1"
    ):
        raise ImportError("MyoSim upper-limb registration wrote an unsupported schema")
    print(f"wrote {output}")
    return 0


def myosim_upper_limb_pose_audit(arguments: argparse.Namespace) -> int:
    auditor = arguments.python.expanduser().absolute()
    if not auditor.is_file() or not os.access(auditor, os.X_OK):
        raise ImportError(f"MyoSim upper-limb pose-audit Python is unavailable: {auditor}")
    output = arguments.output.resolve()
    environment = dict(os.environ)
    source_path = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        str(auditor), "-m", "numilab_human.upper_limb_pose_audit",
        "--sources", str(arguments.sources.resolve()),
        "--registration", str(arguments.registration.resolve()),
        "--output", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no pose-audit output"
        raise ImportError(f"MyoSim upper-limb pose audit failed: {detail}")
    receipt = read_json(output)
    if receipt.get("schema") != (
        "numi.human.bodyparts3d-myosim-upper-limb-multi-pose-audit.v1"
    ):
        raise ImportError("MyoSim upper-limb pose audit wrote an unsupported schema")
    print(f"wrote {output}")
    return 0


def myosim_sternal_girdle_registration(arguments: argparse.Namespace) -> int:
    output = arguments.output.resolve()
    try:
        result = register_sternal_girdle(
            sources=arguments.sources.resolve(),
            registration_path=arguments.registration.resolve(),
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise ImportError(str(error)) from error
    receipt = result.get("sternal_girdle_source_registration")
    if not isinstance(receipt, dict) or receipt.get("schema") != STERNAL_GIRDLE_REGISTRATION_SCHEMA:
        raise ImportError("MyoSim sternal-girdle registration wrote an unsupported schema")
    write_json(output, result)
    print(f"wrote {output}")
    return 0


def myosim_lower_limb_registration(arguments: argparse.Namespace) -> int:
    output = arguments.output.resolve()
    try:
        result = propose_lower_limb_registration(
            registration_path=arguments.registration.resolve(),
            rigid_foot_base_path=arguments.rigid_foot_base.resolve(),
        )
    except ValueError as error:
        raise ImportError(str(error)) from error
    write_json(output, result)
    print(f"wrote {output}")
    return 0


def myosim_lower_limb_source_registration(arguments: argparse.Namespace) -> int:
    exporter = arguments.python.expanduser().absolute()
    if not exporter.is_file() or not os.access(exporter, os.X_OK):
        raise ImportError(f"MyoSim lower-limb registration Python is unavailable: {exporter}")
    output = arguments.output.resolve()
    environment = dict(os.environ)
    source_path = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        str(exporter), "-m", "numilab_human.lower_limb_source_registration",
        "--sources", str(arguments.sources.resolve()),
        "--registration", str(arguments.registration.resolve()),
        "--tendon-manifest", str(arguments.tendon_manifest.resolve()),
        "--output", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no registration output"
        raise ImportError(f"MyoSim lower-limb source registration failed: {detail}")
    candidate = read_json(output)
    receipt = candidate.get("lower_limb_source_mesh_registration")
    if (
        candidate.get("schema") != "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
        or not isinstance(receipt, dict)
        or receipt.get("schema")
        != "numi.human.bodyparts3d-myosim-lower-limb-source-mesh-registration.v2"
    ):
        raise ImportError("MyoSim lower-limb source registration wrote an unsupported schema")
    print(f"wrote {output}")
    return 0


def open_knee_payload(arguments: argparse.Namespace) -> int:
    compiler = arguments.python.expanduser().absolute()
    if not compiler.is_file() or not os.access(compiler, os.X_OK):
        raise ImportError(f"Open Knee(s) compiler Python is unavailable: {compiler}")
    output = arguments.output.resolve()
    environment = dict(os.environ)
    source_path = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        str(compiler), "-m", "numilab_human.open_knee",
        "--sources", str(arguments.sources.resolve()),
        "--open-knee", str(arguments.open_knee.resolve()),
        "--registration", str(arguments.registration.resolve()),
        "--output", str(output),
        "--side", arguments.side,
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, env=environment
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no compiler output"
        raise ImportError(f"Open Knee(s) payload compilation failed: {detail}")
    stem = (
        "open-knee-oks003-left" if arguments.side == "left"
        else "open-knee-oks003-right-mirrored"
    )
    manifest = read_json(output / f"{stem}.manifest.json")
    if manifest.get("schema") != OPEN_KNEE_SCHEMA:
        raise ImportError("Open Knee(s) compiler wrote an unsupported manifest")
    print(f"wrote {output / f'{stem}.nhknee'}")
    print(f"wrote {output / f'{stem}.manifest.json'}")
    return 0


def myosim_thoracic_registration(arguments: argparse.Namespace) -> int:
    exporter = arguments.python.expanduser().absolute()
    if not exporter.is_file() or not os.access(exporter, os.X_OK):
        raise ImportError(f"MyoSim thoracic registration Python is unavailable: {exporter}")
    output = arguments.output.resolve()
    environment = dict(os.environ)
    source_path = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        str(exporter), "-m", "numilab_human.thoracic_registration",
        "--sources", str(arguments.sources.resolve()),
        "--registration", str(arguments.registration.resolve()),
        "--source-audit", str(arguments.source_audit.resolve()),
        "--tendon-manifest", str(arguments.tendon_manifest.resolve()),
        "--output", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no registration output"
        raise ImportError(f"MyoSim thoracic registration failed: {detail}")
    candidate = read_json(output)
    receipt = candidate.get("thoracic_source_mesh_registration")
    if (
        candidate.get("schema") != "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
        or not isinstance(receipt, dict)
        or receipt.get("schema") != THORACIC_REGISTRATION_SCHEMA
    ):
        raise ImportError("MyoSim thoracic registration wrote an unsupported schema")
    print(f"wrote {output}")
    return 0


def myosim_pelvis_registration(arguments: argparse.Namespace) -> int:
    exporter = arguments.python.expanduser().absolute()
    if not exporter.is_file() or not os.access(exporter, os.X_OK):
        raise ImportError(f"MyoSim pelvis registration Python is unavailable: {exporter}")
    output = arguments.output.resolve()
    environment = dict(os.environ)
    source_path = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        str(exporter), "-m", "numilab_human.pelvis_registration",
        "--sources", str(arguments.sources.resolve()),
        "--registration", str(arguments.registration.resolve()),
        "--source-audit", str(arguments.source_audit.resolve()),
        "--tendon-manifest", str(arguments.tendon_manifest.resolve()),
        "--output", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no registration output"
        raise ImportError(f"MyoSim pelvis registration failed: {detail}")
    candidate = read_json(output)
    receipt = candidate.get("pelvis_source_mesh_registration")
    if (
        candidate.get("schema") != "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
        or not isinstance(receipt, dict)
        or receipt.get("schema") != PELVIS_REGISTRATION_SCHEMA
    ):
        raise ImportError("MyoSim pelvis registration wrote an unsupported schema")
    print(f"wrote {output}")
    return 0


def myosim_rib_registration(arguments: argparse.Namespace) -> int:
    exporter = arguments.python.expanduser().absolute()
    if not exporter.is_file() or not os.access(exporter, os.X_OK):
        raise ImportError(f"MyoSim rib registration Python is unavailable: {exporter}")
    output = arguments.output.resolve()
    environment = dict(os.environ)
    source_path = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        str(exporter), "-m", "numilab_human.rib_registration",
        "--sources", str(arguments.sources.resolve()),
        "--registration", str(arguments.registration.resolve()),
        "--source-audit", str(arguments.source_audit.resolve()),
        "--tendon-manifest", str(arguments.tendon_manifest.resolve()),
        "--output", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no registration output"
        raise ImportError(f"MyoSim rib registration failed: {detail}")
    candidate = read_json(output)
    receipt = candidate.get("rib_source_component_registration")
    if (
        candidate.get("schema") != "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
        or not isinstance(receipt, dict)
        or receipt.get("schema") != RIB_REGISTRATION_SCHEMA
    ):
        raise ImportError("MyoSim rib registration wrote an unsupported schema")
    print(f"wrote {output}")
    return 0


def myosim_abdominal_enthesis_registration(arguments: argparse.Namespace) -> int:
    exporter = arguments.python.expanduser().absolute()
    if not exporter.is_file() or not os.access(exporter, os.X_OK):
        raise ImportError(
            f"MyoSim abdominal enthesis registration Python is unavailable: {exporter}"
        )
    output = arguments.output.resolve()
    environment = dict(os.environ)
    source_path = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH") else ""
    )
    command = [
        str(exporter), "-m", "numilab_human.abdominal_enthesis_registration",
        "--sources", str(arguments.sources.resolve()),
        "--registration", str(arguments.registration.resolve()),
        "--source-audit", str(arguments.source_audit.resolve()),
        "--worklist", str(arguments.worklist.resolve()),
        "--tendon-manifest", str(arguments.tendon_manifest.resolve()),
        "--output", str(output),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, env=environment,
    )
    if completed.returncode != 0:
        detail = (
            completed.stderr.strip() or completed.stdout.strip()
            or "no registration output"
        )
        raise ImportError(f"MyoSim abdominal enthesis registration failed: {detail}")
    candidate = read_json(output)
    receipt = candidate.get("abdominal_source_component_enthesis_registration")
    if (
        candidate.get("schema")
        != "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
        or not isinstance(receipt, dict)
        or receipt.get("schema") != ABDOMINAL_ENTHESIS_REGISTRATION_SCHEMA
    ):
        raise ImportError(
            "MyoSim abdominal enthesis registration wrote an unsupported schema"
        )
    print(f"wrote {output}")
    return 0


def myosim_probe(arguments: argparse.Namespace) -> int:
    artifact = arguments.artifact.resolve()
    manifest = read_json(artifact / "myosim-fullbody-reference.manifest.json")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, dict):
        raise ImportError("MyoSim artifact manifest has no payloads")
    rigid = artifact / str(payloads.get("rigid", {}).get("file", ""))
    muscles = artifact / str(payloads.get("muscles", {}).get("file", ""))
    if not rigid.is_file() or not muscles.is_file():
        raise ImportError("MyoSim artifact is missing its rigid or muscle payload")
    runtime_root = arguments.runtime_root.resolve()
    probe = runtime_root / "build/bin/metalrobo_numilab_human_myosim_reference_probe"
    if not probe.is_file() or not os.access(probe, os.X_OK):
        raise ImportError(
            "MyoSim Core probe is unavailable; build "
            "metalrobo_numilab_human_myosim_reference_probe under "
            f"{runtime_root / 'build'} first"
        )
    command = [str(probe), str(rigid), str(muscles)]
    if arguments.metal:
        command.append("--metal")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    transcript = artifact / "myosim-fullbody-core-probe.txt"
    transcript.write_text(
        "command: " + " ".join(command) + "\n\nstdout:\n" + completed.stdout +
        "\nstderr:\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise ImportError(f"MyoSim Core probe failed; see {transcript}")
    print(f"wrote {transcript}")
    return 0


def myosim_part_controls(arguments: argparse.Namespace) -> int:
    artifact = arguments.artifact.resolve()
    if arguments.list:
        if arguments.part or arguments.muscle or arguments.emit != "json":
            raise ImportError("MyoSim part-control --list cannot be combined with a selection or --emit")
        catalog = myosim_part_control_catalog(artifact)
        print("body_name\tcore_body_index\tsource_muscle_count")
        for part in catalog["parts"]:
            print(
                f"{part['body_name']}\t{part['core_body_index']}\t"
                f"{part['source_muscle_count']}"
            )
        return 0
    plan = myosim_part_control_plan(
        artifact, arguments.part or [], arguments.muscle or [],
    )
    if arguments.emit == "indices":
        print(" ".join(
            str(muscle["source_actuator_index"])
            for muscle in plan["selected_source_muscles"]
        ))
    elif arguments.emit == "focus":
        focus = plan["focus_core_body_index"]
        if focus is None:
            raise ImportError("MyoSim part-control focus emission requires exactly one part")
        print(focus)
    else:
        print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def myosim_visuals(arguments: argparse.Namespace) -> int:
    exporter = arguments.python.expanduser().absolute()
    if not exporter.is_file() or not os.access(exporter, os.X_OK):
        raise ImportError(f"MyoSim visual Python is unavailable: {exporter}")
    output = arguments.output.resolve()
    environment = dict(os.environ)
    source_path = str(REPOSITORY_ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        str(exporter), "-m", "numilab_human.myosim_visual",
        "--sources", str(arguments.sources.resolve()), "--output", str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment)
    output.mkdir(parents=True, exist_ok=True)
    transcript = output / "myosim-fullbody-source-visual.txt"
    transcript.write_text(
        "command: " + " ".join(command) + "\n\nstdout:\n" + completed.stdout +
        "\nstderr:\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no renderer output"
        raise ImportError(f"MyoSim multi-angle source render failed: {detail}")
    if completed.stdout:
        print(completed.stdout, end="")
    print(f"wrote {transcript}")
    return 0


def mortensen_neck(arguments: argparse.Namespace) -> int:
    """Emit the selected cervical/hyoid source IR for later MyoSim registration."""
    source = arguments.sources.resolve() / "mortensen" / "checkout" / "models/reference/Mortensen2018.osim"
    lower = parse_opensim(source, "mortensen_2018_neck")
    output = arguments.output.resolve()
    write_json(output, mortensen_neck_source_ir(lower))
    print(f"wrote {output}")
    return 0


def build(arguments: argparse.Namespace) -> int:
    if not arguments.accept_upper_noncommercial_terms:
        raise ImportError(
            "MoBL-ARMS official terms restrict this source to non-commercial use. "
            "Re-run with --accept-upper-noncommercial-terms after reviewing THIRD_PARTY_NOTICES.md"
        )
    source_lock = read_json(REPOSITORY_ROOT / "sources.lock.json")
    sources = arguments.sources.resolve()
    upper_archive = arguments.upper_archive.resolve() if arguments.upper_archive else None
    upper_public_model: Path | None = None
    if upper_archive is not None:
        if not upper_archive.is_file():
            raise ImportError(f"upper archive does not exist: {upper_archive}")
    elif arguments.upper_public_mobl_41:
        public_mobl = source_lock["sources"].get("mobl_arms_ceinms_41_public_mirror")
        if not isinstance(public_mobl, dict) or not isinstance(public_mobl.get("model_file"), str):
            raise ImportError("sources.lock.json has no complete public MoBL-ARMS 4.1 mirror entry")
        upper_public_model = sources / public_mobl["model_file"]
        if not upper_public_model.is_file():
            raise ImportError(
                f"public MoBL-ARMS 4.1 model does not exist: {upper_public_model}; "
                "run `numi human fetch --include-public-mobl-41 --accept-upper-noncommercial-terms` first"
            )
    else:
        raise ImportError("build requires an upper source")
    manifest = build_manifest(
        sources=sources,
        upper_archive=upper_archive,
        classification_path=REPOSITORY_ROOT / "config/anatomy-classification.v1.json",
        target_mapping_path=REPOSITORY_ROOT / "config/numi-targets.v1.json",
        source_lock=source_lock,
        upper_public_model=upper_public_model,
    )
    output = arguments.output.resolve()
    write_json(output / "human.v1.json", manifest)
    write_json(output / "report.json", report_for(manifest))
    print(f"wrote {output / 'human.v1.json'}")
    print(f"wrote {output / 'report.json'}")
    return 0


def audit(arguments: argparse.Namespace) -> int:
    source_lock = read_json(REPOSITORY_ROOT / "sources.lock.json")
    sources = arguments.sources.resolve()
    upper_public_model: Path | None = None
    if arguments.upper_public_mobl_41:
        public_mobl = source_lock["sources"].get("mobl_arms_ceinms_41_public_mirror")
        if not isinstance(public_mobl, dict) or not isinstance(public_mobl.get("model_file"), str):
            raise ImportError("sources.lock.json has no complete public MoBL-ARMS 4.1 mirror entry")
        upper_public_model = sources / public_mobl["model_file"]
    report = gate_report(
        sources=sources,
        upper_archive=(arguments.upper_archive.resolve() if arguments.upper_archive else None),
        upper_public_model=upper_public_model,
        source_lock=source_lock,
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


def right_lower_leg_anatomy_preview(arguments: argparse.Namespace) -> int:
    output = arguments.output.resolve()
    manifest = bodyparts_right_lower_leg_anatomy_preview(arguments.sources.resolve(), output)
    print(f"wrote {output / manifest['preview']['glb']}")
    print(f"wrote {output / 'bodyparts3d-right-lower-leg-anatomy-source-static.manifest.json'}")
    return 0


def right_calcaneal_tendon_continuity_preview(arguments: argparse.Namespace) -> int:
    output = arguments.output.resolve()
    manifest = bodyparts_right_calcaneal_tendon_continuity_preview(
        arguments.sources.resolve(), output
    )
    print(f"wrote {output / manifest['preview']['glb']}")
    print(f"wrote {output / 'bodyparts3d-right-calcaneal-tendon-continuity-source-static.manifest.json'}")
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


def lower_body_pilot(arguments: argparse.Namespace) -> int:
    source = arguments.sources.resolve()
    lower = parse_opensim(source / "RajagopalLaiUhlrich2023.osim", "rajagopal_lai_uhlrich_2023")
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    core_manifest, core_payload = rajagopal_core_reference_artifact(lower)
    millard_manifest, millard_payload = rajagopal_millard_reference_artifact(lower)
    pilot = rajagopal_lower_body_pilot(lower)
    core_path = output / core_manifest["payload"]["file"]
    millard_path = output / millard_manifest["payload"]["file"]
    core_path.write_bytes(core_payload)
    millard_path.write_bytes(millard_payload)
    write_json(output / "rajagopal-core-reference.manifest.json", core_manifest)
    write_json(output / "rajagopal-millard-reference.manifest.json", millard_manifest)
    pilot_path = output / "lower-body-pilot.json"
    write_json(pilot_path, pilot)
    print(f"wrote {pilot_path}")
    if not arguments.smoke:
        return 0
    runtime_root = arguments.runtime_root.resolve()
    probe = runtime_root / "build/bin/metalrobo_numilab_human_core_reference_probe"
    if not probe.is_file() or not os.access(probe, os.X_OK):
        raise ImportError(
            "Human Core probe is unavailable; build metalrobo_numilab_human_core_reference_probe "
            f"under {runtime_root / 'build'} first"
        )
    pad_indices = ",".join(
        str(pad["mobile_body_index"]) for pad in pilot["contact"]["pads"]
    )
    command = [
        str(probe), str(core_path), "--millard", str(millard_path), "--metal",
        "--pilot-foot-bodies", pad_indices,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    transcript = output / "lower-body-pilot-smoke.txt"
    transcript.write_text(
        "command: " + " ".join(command) + "\n\nstdout:\n" + completed.stdout +
        "\nstderr:\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise ImportError(f"lower-body pilot smoke failed; see {transcript}")
    print(f"wrote {transcript}")
    return 0

def attachment_worklist(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    lower = parse_opensim(sources / "RajagopalLaiUhlrich2023.osim", "rajagopal_lai_uhlrich_2023")
    output = arguments.output.resolve()
    write_json(output, bodyparts_lower_body_attachment_worklist(anatomy, lower))
    print(f"wrote {output}")
    return 0


def foot_registration_template(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    lower = parse_opensim(sources / "RajagopalLaiUhlrich2023.osim", "rajagopal_lai_uhlrich_2023")
    output = arguments.output.resolve()
    write_json(output, bodyparts_foot_registration_template(anatomy, lower))
    print(f"wrote {output}")
    return 0


def foot_collider_preflight(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    lower = parse_opensim(sources / "RajagopalLaiUhlrich2023.osim", "rajagopal_lai_uhlrich_2023")
    output = arguments.output.resolve()
    write_json(output, bodyparts_foot_collider_preflight(sources, anatomy, lower))
    print(f"wrote {output}")
    return 0


def foot_registration_receipt_template(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    lower = parse_opensim(sources / "RajagopalLaiUhlrich2023.osim", "rajagopal_lai_uhlrich_2023")
    output = arguments.output.resolve()
    write_json(output, bodyparts_foot_registration_receipt_template(sources, anatomy, lower))
    print(f"wrote {output}")
    return 0


def foot_registration_receipt_check(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    lower = parse_opensim(sources / "RajagopalLaiUhlrich2023.osim", "rajagopal_lai_uhlrich_2023")
    receipt = read_json(arguments.receipt.resolve())
    output = arguments.output.resolve()
    write_json(output, validate_bodyparts_foot_registration_receipt(receipt, sources, anatomy, lower))
    print(f"wrote {output}")
    return 0


def myosim_bodyparts_registration(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    output = arguments.output.resolve()
    write_json(
        output,
        bodyparts_myosim_registration_candidate(sources, anatomy, arguments.artifact.resolve()),
    )
    print(f"wrote {output}")
    return 0


def numi_human_tendon_payload(arguments: argparse.Namespace) -> int:
    manifest = numi_human_tendon_endpoint_payload(
        arguments.artifact.resolve(), arguments.output.resolve(),
        arguments.surface_receipt.resolve() if arguments.surface_receipt is not None else None,
        arguments.allow_unadmitted_surface,
    )
    print(f"wrote {arguments.output.resolve() / manifest['payload']['file']}")
    print(f"wrote {arguments.output.resolve() / 'numi-human-tendon-endpoints.manifest.json'}")
    print(f"wrote {arguments.output.resolve() / 'numi-human-pack.manifest.json'}")
    return 0


def numi_human_tendon_envelope_payload(arguments: argparse.Namespace) -> int:
    manifest = numi_human_tendon_attachment_envelope_payload(
        arguments.artifact.resolve(), arguments.bone_artifact.resolve(),
        arguments.output.resolve(), arguments.maximum_surface_distance,
        arguments.maximum_patch_radius, arguments.maximum_force_amplification,
        arguments.migrate_semantic_rigid_foot_endpoints,
        arguments.maximum_migrated_endpoint_distance,
    )
    print(f"wrote {arguments.output.resolve() / manifest['payload']['file']}")
    print(f"wrote {arguments.output.resolve() / 'numi-human-tendon-attachments.manifest.json'}")
    print(f"wrote {arguments.output.resolve() / 'numi-human-pack.manifest.json'}")
    return 0


def numi_human_pectoralis_fascia_payload(arguments: argparse.Namespace) -> int:
    manifest = bodyparts_pectoralis_fascia_payload(
        arguments.sources.resolve(), arguments.artifact.resolve(),
        arguments.output.resolve(), arguments.thickness, arguments.load_fraction,
    )
    print(f"wrote {arguments.output.resolve() / manifest['payload']['file']}")
    print(f"wrote {arguments.output.resolve() / 'bodyparts3d-pectoralis-fascia.manifest.json'}")
    return 0


def numi_human_costal_cartilage_payload(arguments: argparse.Namespace) -> int:
    manifest = bodyparts_costal_cartilage_payload(
        arguments.sources.resolve(), arguments.output.resolve(),
        arguments.maximum_volume_error, arguments.attachment_distance,
    )
    print(f"wrote {arguments.output.resolve() / manifest['payload']['file']}")
    print(f"wrote {arguments.output.resolve() / 'bodyparts3d-costal-cartilage.manifest.json'}")
    return 0


def numi_human_anterior_thorax_payload(arguments: argparse.Namespace) -> int:
    manifest = anterior_thorax_composite_payload(
        arguments.registration.resolve(), arguments.tendon_artifact.resolve(),
        arguments.output.resolve(), arguments.maximum_volume_error,
        arguments.qualification_load_fraction,
    )
    print(f"wrote {arguments.output.resolve() / manifest['payload']['file']}")
    print(f"wrote {arguments.output.resolve() / 'anterior-thorax-composite.manifest.json'}")
    return 0


def numi_human_achilles_receipt(arguments: argparse.Namespace) -> int:
    receipt = numi_human_achilles_surface_receipt(
        arguments.sources.resolve(), arguments.registration.resolve(),
        arguments.artifact.resolve(), arguments.output.resolve(),
    )
    print(f"wrote {arguments.output.resolve()}")
    print(f"registered {receipt['summary']['record_count']} bilateral Achilles insertions")
    return 0


def myosim_bodyparts_attachment_registration(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    output = arguments.output.resolve()
    write_json(
        output,
        bodyparts_myosim_attachment_surface_registration_candidate(
            sources, anatomy, arguments.artifact.resolve(),
        ),
    )
    print(f"wrote {output}")
    return 0


def myosim_bodyparts_bone_visual_payload(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    manifest = bodyparts_myosim_bone_visual_payload(
        sources, anatomy, arguments.registration.resolve(), arguments.output.resolve(),
    )
    print(f"wrote {arguments.output.resolve() / manifest['payload']['file']}")
    print(f"wrote {arguments.output.resolve() / 'bodyparts3d-myosim-major-bones.manifest.json'}")
    return 0


def myosim_bodyparts_right_posterior_chain_visual_payload(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    manifest = bodyparts_myosim_right_posterior_chain_visual_payload(
        sources, anatomy, arguments.registration.resolve(), arguments.output.resolve(),
    )
    print(f"wrote {arguments.output.resolve() / manifest['payload']['file']}")
    print(f"wrote {arguments.output.resolve() / 'bodyparts3d-myosim-right-posterior-chain.manifest.json'}")
    return 0


def myosim_bodyparts_fullbody_soft_tissue_visual_payload(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    manifest = bodyparts_myosim_fullbody_soft_tissue_visual_payload(
        sources, anatomy, arguments.registration.resolve(), arguments.artifact.resolve(),
        arguments.output.resolve(),
        set(arguments.stable_id) if arguments.stable_id else None,
    )
    print(f"wrote {arguments.output.resolve() / manifest['payload']['file']}")
    print(f"wrote {arguments.output.resolve() / 'bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json'}")
    return 0


def myosim_bodyparts_torso_anatomy_visual_payload(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    manifest = bodyparts_myosim_torso_anatomy_visual_payload(
        sources, anatomy, arguments.registration.resolve(), arguments.artifact.resolve(),
        arguments.output.resolve(),
    )
    print(f"wrote {arguments.output.resolve() / manifest['payload']['file']}")
    print(f"wrote {arguments.output.resolve() / 'bodyparts3d-myosim-torso-anatomy.manifest.json'}")
    return 0


def zanatomy_calf_visual_supplement_payload(arguments: argparse.Namespace) -> int:
    manifest = build_zanatomy_calf_visual_supplement_payload(
        arguments.sources.resolve(), arguments.registration.resolve(), arguments.base_payload.resolve(),
        arguments.zanatomy_export.resolve(), arguments.output.resolve(),
    )
    print(f"wrote {arguments.output.resolve() / manifest['payload']['file']}")
    print(f"wrote {arguments.output.resolve() / 'zanatomy-calf-myosim-tissues.manifest.json'}")
    return 0


def myosim_bodyparts_skinned_shell_visual_payload(arguments: argparse.Namespace) -> int:
    sources = arguments.sources.resolve()
    anatomy = parse_bodyparts3d(sources, REPOSITORY_ROOT / "config/anatomy-classification.v1.json")
    manifest = bodyparts_myosim_skinned_shell_visual_payload(
        sources, anatomy, arguments.registration.resolve(), arguments.output.resolve(),
    )
    print(f"wrote {arguments.output.resolve() / manifest['payload']['file']}")
    print(f"wrote {arguments.output.resolve() / 'bodyparts3d-myosim-skinned-shell.manifest.json'}")
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
    myosim_fetch_parser = commands.add_parser(
        "myosim-fetch",
        help="fetch and safely extract the selected MyoSim full-body and Mortensen neck sources",
    )
    myosim_fetch_parser.add_argument("--output", type=Path, required=True, help="ignored Sources directory")
    myosim_fetch_parser.set_defaults(handler=myosim_fetch)
    myosim_build_parser = commands.add_parser(
        "myosim-build",
        help="compile MyoSim full-body source records into native Core rigid and muscle payloads",
    )
    myosim_build_parser.add_argument("--sources", type=Path, required=True, help="directory made by myosim-fetch")
    myosim_build_parser.add_argument("--output", type=Path, required=True, help="ignored local artifact directory")
    myosim_build_parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="Python environment with the pinned myo-sim checkout and mujoco installed",
    )
    myosim_build_parser.set_defaults(handler=myosim_build)
    myosim_source_bone_parser = commands.add_parser(
        "myosim-source-bone-proximity",
        help="audit terminal MyoSim sites against exact same-body compiled source bone meshes",
    )
    myosim_source_bone_parser.add_argument("--sources", type=Path, required=True)
    myosim_source_bone_parser.add_argument("--output", type=Path, required=True)
    myosim_source_bone_parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="Python environment with the pinned myo-sim checkout and mujoco installed",
    )
    myosim_source_bone_parser.add_argument(
        "--maximum-distance", type=float, default=0.012,
        help="maximum exact source-site to same-body source-mesh distance in metres (default: 0.012)",
    )
    myosim_source_bone_parser.set_defaults(handler=myosim_source_bone_proximity)
    registration_worklist_parser = commands.add_parser(
        "bodyparts-registration-worklist",
        help="separate true BodyParts3D registration candidates from source-model non-bone endpoints",
    )
    registration_worklist_parser.add_argument("--source-audit", type=Path, required=True)
    registration_worklist_parser.add_argument("--tendon-manifest", type=Path, required=True)
    registration_worklist_parser.add_argument("--output", type=Path, required=True)
    registration_worklist_parser.set_defaults(handler=bodyparts_registration_worklist)
    upper_limb_registration_parser = commands.add_parser(
        "myosim-upper-limb-registration",
        help="propose bilateral source-mesh-constrained BodyParts3D humerus-to-finger registration",
    )
    upper_limb_registration_parser.add_argument("--sources", type=Path, required=True)
    upper_limb_registration_parser.add_argument("--registration", type=Path, required=True)
    upper_limb_registration_parser.add_argument("--source-audit", type=Path, required=True)
    upper_limb_registration_parser.add_argument("--worklist", type=Path, required=True)
    upper_limb_registration_parser.add_argument("--tendon-manifest", type=Path, required=True)
    upper_limb_registration_parser.add_argument("--output", type=Path, required=True)
    upper_limb_registration_parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="Python environment with the pinned myo-sim checkout, NumPy, and MuJoCo installed",
    )
    upper_limb_registration_parser.set_defaults(handler=myosim_upper_limb_registration)
    upper_limb_pose_audit_parser = commands.add_parser(
        "myosim-upper-limb-pose-audit",
        help="prove bilateral shoulder-to-finger registration across bounded source poses",
    )
    upper_limb_pose_audit_parser.add_argument("--sources", type=Path, required=True)
    upper_limb_pose_audit_parser.add_argument("--registration", type=Path, required=True)
    upper_limb_pose_audit_parser.add_argument("--output", type=Path, required=True)
    upper_limb_pose_audit_parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="Python environment with the pinned myo-sim checkout, NumPy, and MuJoCo installed",
    )
    upper_limb_pose_audit_parser.set_defaults(handler=myosim_upper_limb_pose_audit)
    sternal_girdle_registration_parser = commands.add_parser(
        "myosim-sternal-girdle-registration",
        help="restore exact manubrium and common-frame sternum/clavicle continuity",
    )
    sternal_girdle_registration_parser.add_argument("--sources", type=Path, required=True)
    sternal_girdle_registration_parser.add_argument("--registration", type=Path, required=True)
    sternal_girdle_registration_parser.add_argument("--output", type=Path, required=True)
    sternal_girdle_registration_parser.set_defaults(
        handler=myosim_sternal_girdle_registration
    )
    lower_limb_registration_parser = commands.add_parser(
        "myosim-lower-limb-registration",
        help="preserve qualified registration while assigning tarsals/metatarsals to Rajagopal's rigid foot",
    )
    lower_limb_registration_parser.add_argument("--registration", type=Path, required=True)
    lower_limb_registration_parser.add_argument("--rigid-foot-base", type=Path, required=True)
    lower_limb_registration_parser.add_argument("--output", type=Path, required=True)
    lower_limb_registration_parser.set_defaults(handler=myosim_lower_limb_registration)
    lower_limb_source_registration_parser = commands.add_parser(
        "myosim-lower-limb-source-mesh-registration",
        help="propose bilateral source-mesh-constrained BodyParts3D lower-limb registration",
    )
    lower_limb_source_registration_parser.add_argument("--sources", type=Path, required=True)
    lower_limb_source_registration_parser.add_argument("--registration", type=Path, required=True)
    lower_limb_source_registration_parser.add_argument("--tendon-manifest", type=Path, required=True)
    lower_limb_source_registration_parser.add_argument("--output", type=Path, required=True)
    lower_limb_source_registration_parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="Python environment with the pinned myo-sim checkout, NumPy, and MuJoCo installed",
    )
    lower_limb_source_registration_parser.set_defaults(
        handler=myosim_lower_limb_source_registration
    )
    open_knee_parser = commands.add_parser(
        "open-knee-oks003-payload",
        help="compile exact Open Knee(s) oks003 tissues for a live knee side",
    )
    open_knee_parser.add_argument("--sources", type=Path, required=True)
    open_knee_parser.add_argument("--open-knee", type=Path, required=True)
    open_knee_parser.add_argument("--registration", type=Path, required=True)
    open_knee_parser.add_argument("--output", type=Path, required=True)
    open_knee_parser.add_argument("--side", choices=("left", "right"), default="left")
    open_knee_parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="Python environment with pinned MyoSim, NumPy, and MuJoCo",
    )
    open_knee_parser.set_defaults(handler=open_knee_payload)
    thoracic_registration_parser = commands.add_parser(
        "myosim-thoracic-registration",
        help="propose exact T1-T12 source-mesh registration with enthesis and continuity gates",
    )
    thoracic_registration_parser.add_argument("--sources", type=Path, required=True)
    thoracic_registration_parser.add_argument("--registration", type=Path, required=True)
    thoracic_registration_parser.add_argument("--source-audit", type=Path, required=True)
    thoracic_registration_parser.add_argument("--tendon-manifest", type=Path, required=True)
    thoracic_registration_parser.add_argument("--output", type=Path, required=True)
    thoracic_registration_parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="Python environment with the pinned myo-sim checkout, NumPy, and MuJoCo installed",
    )
    thoracic_registration_parser.set_defaults(handler=myosim_thoracic_registration)
    pelvis_registration_parser = commands.add_parser(
        "myosim-pelvis-registration",
        help="propose paired source-mesh hip registration with enthesis and sacroiliac gates",
    )
    pelvis_registration_parser.add_argument("--sources", type=Path, required=True)
    pelvis_registration_parser.add_argument("--registration", type=Path, required=True)
    pelvis_registration_parser.add_argument("--source-audit", type=Path, required=True)
    pelvis_registration_parser.add_argument("--tendon-manifest", type=Path, required=True)
    pelvis_registration_parser.add_argument("--output", type=Path, required=True)
    pelvis_registration_parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="Python environment with the pinned myo-sim checkout, NumPy, and MuJoCo installed",
    )
    pelvis_registration_parser.set_defaults(handler=myosim_pelvis_registration)
    rib_registration_parser = commands.add_parser(
        "myosim-rib-registration",
        help="propose topology-resolved bilateral T1-T12 rib placement and enthesis recovery",
    )
    rib_registration_parser.add_argument("--sources", type=Path, required=True)
    rib_registration_parser.add_argument("--registration", type=Path, required=True)
    rib_registration_parser.add_argument("--source-audit", type=Path, required=True)
    rib_registration_parser.add_argument("--tendon-manifest", type=Path, required=True)
    rib_registration_parser.add_argument("--output", type=Path, required=True)
    rib_registration_parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="Python environment with the pinned myo-sim checkout, NumPy, and MuJoCo installed",
    )
    rib_registration_parser.set_defaults(handler=myosim_rib_registration)
    abdominal_registration_parser = commands.add_parser(
        "myosim-abdominal-enthesis-registration",
        help=(
            "resolve abdominal endpoint ownership from exact pinned thorax components"
        ),
    )
    abdominal_registration_parser.add_argument("--sources", type=Path, required=True)
    abdominal_registration_parser.add_argument(
        "--registration", type=Path, required=True,
    )
    abdominal_registration_parser.add_argument(
        "--source-audit", type=Path, required=True,
    )
    abdominal_registration_parser.add_argument("--worklist", type=Path, required=True)
    abdominal_registration_parser.add_argument(
        "--tendon-manifest", type=Path, required=True,
    )
    abdominal_registration_parser.add_argument("--output", type=Path, required=True)
    abdominal_registration_parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help=(
            "Python environment with the pinned myo-sim checkout, NumPy, and "
            "MuJoCo installed"
        ),
    )
    abdominal_registration_parser.set_defaults(
        handler=myosim_abdominal_enthesis_registration
    )
    myosim_probe_parser = commands.add_parser(
        "myosim-probe",
        help="run the native Core full-body muscle-reference probe against a compiled MyoSim artifact",
    )
    myosim_probe_parser.add_argument("--artifact", type=Path, required=True)
    myosim_probe_parser.add_argument(
        "--runtime-root", type=Path,
        default=Path(os.environ.get("NUMI_LAB_ROOT", "/Users/home/Documents/emergentnumilife/MetalRobo")),
    )
    myosim_probe_parser.add_argument(
        "--metal",
        action="store_true",
        help="also execute Apple-GPU pose/Jacobian/muscle-route/static-force parity",
    )
    myosim_probe_parser.set_defaults(handler=myosim_probe)
    myosim_part_control_parser = commands.add_parser(
        "myosim-part-controls",
        help="list or resolve exact source muscles for bounded body-part control",
    )
    myosim_part_control_parser.add_argument("--artifact", type=Path, required=True)
    myosim_part_control_parser.add_argument(
        "--list", action="store_true", help="list every source-route-controllable Core body",
    )
    myosim_part_control_parser.add_argument(
        "--part", action="append", help="select every exact source muscle routed through this body",
    )
    myosim_part_control_parser.add_argument(
        "--muscle", action="append", help="also select one exact source muscle name",
    )
    myosim_part_control_parser.add_argument(
        "--emit", choices=("json", "indices", "focus"), default="json",
        help="emit the complete plan, native source indices, or single selected focus body",
    )
    myosim_part_control_parser.set_defaults(handler=myosim_part_controls)
    myosim_visuals_parser = commands.add_parser(
        "myosim-visuals",
        help="render three default-pose MyoSim source-model views for visual validation",
    )
    myosim_visuals_parser.add_argument("--sources", type=Path, required=True)
    myosim_visuals_parser.add_argument("--output", type=Path, required=True, help="ignored local visual artifact directory")
    myosim_visuals_parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="Python environment with the pinned myo-sim checkout and mujoco installed",
    )
    myosim_visuals_parser.set_defaults(handler=myosim_visuals)
    myosim_registration_parser = commands.add_parser(
        "myosim-bodyparts-registration",
        help="infer a source-pinned visual-only BodyParts3D visual-skeleton rest-frame candidate for MyoSim",
    )
    myosim_registration_parser.add_argument("--sources", type=Path, required=True)
    myosim_registration_parser.add_argument(
        "--artifact", type=Path, required=True,
        help="compiled MyoSim full-body artifact directory from myosim-build",
    )
    myosim_registration_parser.add_argument("--output", type=Path, required=True)
    myosim_registration_parser.set_defaults(handler=myosim_bodyparts_registration)
    tendon_payload_parser = commands.add_parser(
        "numi-human-tendon-payload",
        help="compile complete Numi-owned route endpoint mechanics and optional admitted bone-triangle bindings",
    )
    tendon_payload_parser.add_argument(
        "--artifact", type=Path, required=True,
        help="compiled MyoSim full-body artifact directory from myosim-build",
    )
    tendon_payload_parser.add_argument("--output", type=Path, required=True)
    tendon_payload_parser.add_argument(
        "--surface-receipt", type=Path,
        help="optional explicit numi.human.tendon-surface-registration.v1 receipt",
    )
    tendon_payload_parser.add_argument(
        "--allow-unadmitted-surface", action="store_true",
        help="compile a rejected surface candidate for native impact measurement; never use for production",
    )
    tendon_payload_parser.set_defaults(handler=numi_human_tendon_payload)
    tendon_envelope_parser = commands.add_parser(
        "numi-human-tendon-envelope-payload",
        help="compile fail-closed source-point-preserving BodyParts3D tendon attachment envelopes",
    )
    tendon_envelope_parser.add_argument(
        "--artifact", type=Path, required=True,
        help="compiled MyoSim full-body artifact directory from myosim-build",
    )
    tendon_envelope_parser.add_argument(
        "--bone-artifact", type=Path, required=True,
        help="directory containing the exact paired NHBONES1 payload and manifest",
    )
    tendon_envelope_parser.add_argument("--output", type=Path, required=True)
    tendon_envelope_parser.add_argument(
        "--migrate-semantic-rigid-foot-endpoints", action="store_true",
        help=(
            "emit NHTENDON3 with the 18 one-to-one named rigid-foot/hallux endpoints "
            "migrated to exact registered bone surfaces; EDL/FDL remain source points"
        ),
    )
    tendon_envelope_parser.add_argument(
        "--maximum-surface-distance", type=float, default=0.012,
        help="maximum exact source-point to registered bone-surface distance in metres (default: 0.012)",
    )
    tendon_envelope_parser.add_argument(
        "--maximum-migrated-endpoint-distance", type=float, default=0.025,
        help=(
            "maximum source-site to named-bone distance for only the 18 explicit "
            "NHTENDON3 migrations (default: 0.025); ordinary admission remains at "
            "--maximum-surface-distance"
        ),
    )
    tendon_envelope_parser.add_argument(
        "--maximum-patch-radius", type=float, default=0.012,
        help="maximum connected attachment patch radius in metres (default: 0.012)",
    )
    tendon_envelope_parser.add_argument(
        "--maximum-force-amplification", type=float, default=4.0,
        help="maximum sampled sum of nodal force magnitudes for a unit terminal force (default: 4.0)",
    )
    tendon_envelope_parser.set_defaults(handler=numi_human_tendon_envelope_payload)
    pectoralis_fascia_parser = commands.add_parser(
        "numi-human-pectoralis-fascia-payload",
        help="compile an explicit source-derived pectoral-fascia thin-solid FEM fallback",
    )
    pectoralis_fascia_parser.add_argument("--sources", type=Path, required=True)
    pectoralis_fascia_parser.add_argument(
        "--artifact", type=Path, required=True,
        help="compiled MyoSim full-body artifact directory from myosim-build",
    )
    pectoralis_fascia_parser.add_argument("--output", type=Path, required=True)
    pectoralis_fascia_parser.add_argument(
        "--thickness", type=float, default=0.0006,
        help="generated sheet thickness in metres (default: 0.0006)",
    )
    pectoralis_fascia_parser.add_argument(
        "--load-fraction", type=float, default=0.10,
        help="bounded share of named pectoralis terminal force applied to fascia (default: 0.10)",
    )
    pectoralis_fascia_parser.set_defaults(handler=numi_human_pectoralis_fascia_payload)
    costal_cartilage_parser = commands.add_parser(
        "numi-human-costal-cartilage-payload",
        help="compile fourteen exact BodyParts3D costal-cartilage shells into deterministic FEM volumes",
    )
    costal_cartilage_parser.add_argument("--sources", type=Path, required=True)
    costal_cartilage_parser.add_argument("--output", type=Path, required=True)
    costal_cartilage_parser.add_argument(
        "--maximum-volume-error", type=float, default=0.03,
        help="maximum voxel-to-exact closed-surface relative volume error (default: 0.03)",
    )
    costal_cartilage_parser.add_argument(
        "--attachment-distance", type=float, default=0.004,
        help="exact source-vertex distance for named rib/sternal attachment bands in metres (default: 0.004)",
    )
    costal_cartilage_parser.set_defaults(handler=numi_human_costal_cartilage_payload)
    anterior_thorax_parser = commands.add_parser(
        "numi-human-anterior-thorax-continuum-payload",
        help="compile exact pinned anterior-thorax closed surfaces into a deterministic FEM volume",
    )
    anterior_thorax_parser.add_argument(
        "--registration", type=Path, required=True,
        help="admitted registration containing the exact source-component surfaces",
    )
    anterior_thorax_parser.add_argument(
        "--tendon-artifact", type=Path, required=True,
        help="paired NHTENDON3 artifact directory containing the embedded source receipt",
    )
    anterior_thorax_parser.add_argument("--output", type=Path, required=True)
    anterior_thorax_parser.add_argument(
        "--maximum-volume-error", type=float, default=0.03,
        help="maximum voxel-to-exact closed-surface relative volume error (default: 0.03)",
    )
    anterior_thorax_parser.add_argument(
        "--qualification-load-fraction", type=float, default=0.10,
        help="non-owning bounded terminal-load share for deformation qualification only (default: 0.10)",
    )
    anterior_thorax_parser.set_defaults(handler=numi_human_anterior_thorax_payload)
    achilles_receipt_parser = commands.add_parser(
        "numi-human-achilles-surface-receipt",
        help="register six bilateral Achilles route insertions to exact BodyParts3D calcaneus triangles",
    )
    achilles_receipt_parser.add_argument("--sources", type=Path, required=True)
    achilles_receipt_parser.add_argument("--registration", type=Path, required=True)
    achilles_receipt_parser.add_argument("--artifact", type=Path, required=True)
    achilles_receipt_parser.add_argument("--output", type=Path, required=True)
    achilles_receipt_parser.set_defaults(handler=numi_human_achilles_receipt)
    myosim_attachment_registration_parser = commands.add_parser(
        "myosim-bodyparts-attachment-registration",
        help="infer visual-only per-bone BodyParts3D surface correspondences from source MyoSim attachment sites",
    )
    myosim_attachment_registration_parser.add_argument("--sources", type=Path, required=True)
    myosim_attachment_registration_parser.add_argument(
        "--artifact", type=Path, required=True,
        help="compiled MyoSim full-body artifact directory from myosim-build",
    )
    myosim_attachment_registration_parser.add_argument("--output", type=Path, required=True)
    myosim_attachment_registration_parser.set_defaults(handler=myosim_bodyparts_attachment_registration)
    myosim_bone_payload_parser = commands.add_parser(
        "myosim-bodyparts-bone-payload",
        help="prepare source visual-skeleton triangles and articulated local transforms for the native Human visual renderer",
    )
    myosim_bone_payload_parser.add_argument("--sources", type=Path, required=True)
    myosim_bone_payload_parser.add_argument(
        "--registration", type=Path, required=True,
        help="candidate JSON from myosim-bodyparts-registration",
    )
    myosim_bone_payload_parser.add_argument("--output", type=Path, required=True)
    myosim_bone_payload_parser.set_defaults(handler=myosim_bodyparts_bone_visual_payload)
    myosim_posterior_chain_payload_parser = commands.add_parser(
        "myosim-bodyparts-right-posterior-chain-payload",
        help="prepare exact right posterior-calf muscle and calcaneal-tendon surfaces for native two-body kinematic inspection",
    )
    myosim_posterior_chain_payload_parser.add_argument("--sources", type=Path, required=True)
    myosim_posterior_chain_payload_parser.add_argument(
        "--registration", type=Path, required=True,
        help="v2 candidate JSON from myosim-bodyparts-registration",
    )
    myosim_posterior_chain_payload_parser.add_argument("--output", type=Path, required=True)
    myosim_posterior_chain_payload_parser.set_defaults(
        handler=myosim_bodyparts_right_posterior_chain_visual_payload,
    )
    myosim_fullbody_tissue_payload_parser = commands.add_parser(
        "myosim-bodyparts-fullbody-muscle-surface-payload",
        help="prepare audited BodyParts3D surfaces for native MyoSim endpoint posing with explicit shared-tendon body ownership",
    )
    myosim_fullbody_tissue_payload_parser.add_argument("--sources", type=Path, required=True)
    myosim_fullbody_tissue_payload_parser.add_argument(
        "--registration", type=Path, required=True,
        help="unmodified v2 candidate JSON from myosim-bodyparts-registration",
    )
    myosim_fullbody_tissue_payload_parser.add_argument(
        "--artifact", type=Path, required=True,
        help="compiled MyoSim full-body artifact directory from myosim-build",
    )
    myosim_fullbody_tissue_payload_parser.add_argument(
        "--stable-id", type=int, action="append",
        help="emit only this existing BodyParts3D/MyoSim surface stable ID; repeat as needed",
    )
    myosim_fullbody_tissue_payload_parser.add_argument("--output", type=Path, required=True)
    myosim_fullbody_tissue_payload_parser.set_defaults(
        handler=myosim_bodyparts_fullbody_soft_tissue_visual_payload,
    )
    myosim_torso_anatomy_payload_parser = commands.add_parser(
        "myosim-bodyparts-torso-anatomy-payload",
        help="prepare selected exact BodyParts3D organ, vessel, and spinal-cord surfaces for native torso inspection",
    )
    myosim_torso_anatomy_payload_parser.add_argument("--sources", type=Path, required=True)
    myosim_torso_anatomy_payload_parser.add_argument(
        "--registration", type=Path, required=True,
        help="unmodified visual-registration candidate from myosim-bodyparts-registration",
    )
    myosim_torso_anatomy_payload_parser.add_argument(
        "--artifact", type=Path, required=True,
        help="compiled MyoSim full-body artifact directory from myosim-build",
    )
    myosim_torso_anatomy_payload_parser.add_argument("--output", type=Path, required=True)
    myosim_torso_anatomy_payload_parser.set_defaults(
        handler=myosim_bodyparts_torso_anatomy_visual_payload,
    )
    zanatomy_calf_payload_parser = commands.add_parser(
        "zanatomy-calf-visual-supplement-payload",
        help="prepare the narrowly scoped CC-BY-SA Z-Anatomy right-calf visual supplement with existing MyoSim body bindings",
    )
    zanatomy_calf_payload_parser.add_argument("--sources", type=Path, required=True)
    zanatomy_calf_payload_parser.add_argument("--registration", type=Path, required=True)
    zanatomy_calf_payload_parser.add_argument(
        "--base-payload", type=Path, required=True,
        help="audited NHTISS3 BodyParts3D full-body muscle-surface payload",
    )
    zanatomy_calf_payload_parser.add_argument(
        "--zanatomy-export", type=Path, required=True,
        help="right-calf interchange emitted by Blender with tools/export_zanatomy_calf.py",
    )
    zanatomy_calf_payload_parser.add_argument("--output", type=Path, required=True)
    zanatomy_calf_payload_parser.set_defaults(handler=zanatomy_calf_visual_supplement_payload)
    myosim_skinned_shell_payload_parser = commands.add_parser(
        "myosim-bodyparts-skinned-shell-payload",
        help="prepare the exact BodyParts3D exterior mesh with four registered articulated visual influences per vertex",
    )
    myosim_skinned_shell_payload_parser.add_argument("--sources", type=Path, required=True)
    myosim_skinned_shell_payload_parser.add_argument(
        "--registration", type=Path, required=True,
        help="unmodified v2 candidate JSON from myosim-bodyparts-registration",
    )
    myosim_skinned_shell_payload_parser.add_argument("--output", type=Path, required=True)
    myosim_skinned_shell_payload_parser.set_defaults(
        handler=myosim_bodyparts_skinned_shell_visual_payload,
    )
    mortensen_neck_parser = commands.add_parser(
        "mortensen-neck",
        help="emit the complete selected OpenSim 3 cervical/hyoid source IR for MyoSim registration",
    )
    mortensen_neck_parser.add_argument("--sources", type=Path, required=True, help="directory made by myosim-fetch")
    mortensen_neck_parser.add_argument("--output", type=Path, required=True, help="ignored local source-IR artifact")
    mortensen_neck_parser.set_defaults(handler=mortensen_neck)
    fetch_parser = commands.add_parser("fetch", help="fetch BodyParts3D 4.0, Rajagopal, and an optional public MoBL 4.1 mirror")
    fetch_parser.add_argument("--output", type=Path, required=True, help="local, ignored source directory")
    fetch_parser.add_argument("--include-public-mobl-41", action="store_true", help="fetch the pinned public unimanual MoBL-ARMS 4.1 mirror")
    fetch_parser.add_argument("--accept-upper-noncommercial-terms", action="store_true")
    fetch_parser.set_defaults(handler=fetch)
    build_parser = commands.add_parser("build", help="combine exact sources into a local audit manifest")
    build_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    build_upper_source = build_parser.add_mutually_exclusive_group(required=True)
    build_upper_source.add_argument("--upper-archive", type=Path, help="original authenticated SimTK MoBL-ARMS bimanual ZIP")
    build_upper_source.add_argument("--upper-public-mobl-41", action="store_true", help="use the pinned public unimanual MoBL-ARMS 4.1 mirror from --sources")
    build_parser.add_argument("--output", type=Path, required=True, help="local output directory")
    build_parser.add_argument("--accept-upper-noncommercial-terms", action="store_true")
    build_parser.set_defaults(handler=build)
    audit_parser = commands.add_parser("audit", help="report every source, runtime, and physics gate")
    audit_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    audit_upper_source = audit_parser.add_mutually_exclusive_group()
    audit_upper_source.add_argument("--upper-archive", type=Path, help="original authenticated SimTK MoBL-ARMS bimanual ZIP")
    audit_upper_source.add_argument("--upper-public-mobl-41", action="store_true", help="inspect the pinned public unimanual MoBL-ARMS 4.1 mirror from --sources")
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
    lower_leg_anatomy_parser = commands.add_parser(
        "right-lower-leg-anatomy-preview",
        help="export exact source-static BodyParts3D right lower-leg bone, muscle, and tendon geometry",
    )
    lower_leg_anatomy_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    lower_leg_anatomy_parser.add_argument("--output", type=Path, required=True, help="local output directory")
    lower_leg_anatomy_parser.set_defaults(handler=right_lower_leg_anatomy_preview)
    tendon_continuity_parser = commands.add_parser(
        "right-calcaneal-tendon-continuity-preview",
        help="export the exact source-static right posterior lower-leg tendon chain for inspection",
    )
    tendon_continuity_parser.add_argument("--sources", type=Path, required=True, help="directory created by fetch")
    tendon_continuity_parser.add_argument("--output", type=Path, required=True, help="local output directory")
    tendon_continuity_parser.set_defaults(handler=right_calcaneal_tendon_continuity_preview)
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
    pilot_parser = commands.add_parser(
        "pilot",
        help="build the mobile lower-body, muscle-driven flat-ground pilot",
    )
    pilot_parser.add_argument("--sources", type=Path, required=True, help="directory containing Rajagopal source")
    pilot_parser.add_argument("--output", type=Path, required=True, help="local pilot artifact directory")
    pilot_parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path(os.environ.get("NUMI_LAB_ROOT", "/Users/home/Documents/emergentnumilife/MetalRobo")),
        help="MetalRobo checkout used only when --smoke is requested",
    )
    pilot_parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the local Apple-Metal four-pad contact and native-muscle smoke",
    )
    pilot_parser.set_defaults(handler=lower_body_pilot)
    attachment_parser = commands.add_parser("attachment-worklist", help="emit review-only lower-body BodyParts3D attachment and foot-collider work items")
    attachment_parser.add_argument("--sources", type=Path, required=True)
    attachment_parser.add_argument("--output", type=Path, required=True)
    attachment_parser.set_defaults(handler=attachment_worklist)
    foot_registration_parser = commands.add_parser(
        "foot-registration-template",
        help="emit a fail-closed four-foot registration and collider-review hand-off",
    )
    foot_registration_parser.add_argument("--sources", type=Path, required=True)
    foot_registration_parser.add_argument("--output", type=Path, required=True)
    foot_registration_parser.set_defaults(handler=foot_registration_template)
    foot_collider_parser = commands.add_parser(
        "foot-collider-preflight",
        help="derive source-local foot enclosure candidates without admitting contact",
    )
    foot_collider_parser.add_argument("--sources", type=Path, required=True)
    foot_collider_parser.add_argument("--output", type=Path, required=True)
    foot_collider_parser.set_defaults(handler=foot_collider_preflight)
    foot_receipt_parser = commands.add_parser(
        "foot-registration-receipt-template",
        help="compose a provenance-pinned reviewer receipt template for foot registration",
    )
    foot_receipt_parser.add_argument("--sources", type=Path, required=True)
    foot_receipt_parser.add_argument("--output", type=Path, required=True)
    foot_receipt_parser.set_defaults(handler=foot_registration_receipt_template)
    foot_receipt_check_parser = commands.add_parser(
        "foot-registration-receipt-check",
        help="fail closed on a reviewer-completed foot registration/contact receipt",
    )
    foot_receipt_check_parser.add_argument("--sources", type=Path, required=True)
    foot_receipt_check_parser.add_argument("--receipt", type=Path, required=True)
    foot_receipt_check_parser.add_argument("--output", type=Path, required=True)
    foot_receipt_check_parser.set_defaults(handler=foot_registration_receipt_check)
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
