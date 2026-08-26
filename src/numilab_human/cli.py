from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from .model import ImportError, build_manifest, gate_report, read_json, report_for, sha256, write_json


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
