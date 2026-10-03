"""Apply explicit, hash-bound corrections to the pinned MyoSim source tree."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


OVERLAY_RELATIVE_PATH = Path(
    "source-overlays/myosim-left-knee-translation2-range.v1.json"
)
CANONICAL_OVERLAY_PATH = (
    Path(__file__).resolve().parents[2]
    / "Sources"
    / "myosim"
    / OVERLAY_RELATIVE_PATH
)
PATELLA_REBASE_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "myosim-patella-neutral-coordinate-rebase.v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_overlay(source_package: Path) -> tuple[dict[str, Any], Path, str]:
    canonical = CANONICAL_OVERLAY_PATH
    if not canonical.is_file():
        raise RuntimeError(f"canonical MyoSim source overlay is missing: {canonical}")
    package = source_package.resolve()
    overlay_path = package / OVERLAY_RELATIVE_PATH
    if not overlay_path.is_file():
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(canonical, overlay_path)
    canonical_sha = _sha256(canonical)
    if _sha256(overlay_path) != canonical_sha:
        raise RuntimeError("MyoSim source overlay manifest differs from the repository-pinned correction")
    try:
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"MyoSim source overlay manifest is unreadable: {error}") from error
    if not isinstance(overlay, dict) or overlay.get("schema") != \
            "numi.human.myosim-source-range-overlay.v1":
        raise RuntimeError("MyoSim source overlay schema is unsupported")
    return overlay, overlay_path, canonical_sha


def _apply_patella_neutral_coordinate_rebase(
    source_package: Path,
    source_identity: dict[str, Any],
    range_overlay: dict[str, Any],
    range_overlay_sha: str,
) -> dict[str, Any]:
    """Apply the tracked, exact bilateral patella qpos0 coordinate rebase."""
    manifest_path = PATELLA_REBASE_PATH
    if not manifest_path.is_file():
        raise RuntimeError(f"tracked patella source rebase is missing: {manifest_path}")
    manifest_sha = _sha256(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"patella source rebase manifest is unreadable: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != \
            "numi.human.myosim-patella-neutral-coordinate-rebase.v1":
        raise RuntimeError("patella source rebase manifest schema is unsupported")

    rebase_source = manifest.get("source")
    required_overlay = manifest.get("requires_source_overlay")
    if not isinstance(rebase_source, dict) or not isinstance(required_overlay, dict):
        raise RuntimeError("patella source rebase identity is incomplete")
    if any(rebase_source.get(key) != source_identity.get(key) for key in (
            "repository", "revision", "archive", "archive_sha256", "license")):
        raise RuntimeError("patella source rebase does not match the pinned upstream source")
    if required_overlay.get("id") != range_overlay.get("id") or \
            required_overlay.get("manifest_sha256") != range_overlay_sha:
        raise RuntimeError("patella source rebase requires a different knee-range overlay")

    coordinate = manifest.get("coordinate_reparameterization")
    changes = coordinate.get("files") if isinstance(coordinate, dict) else None
    if not isinstance(changes, list) or not changes:
        raise RuntimeError("patella source rebase has no exact file changes")

    staged: list[tuple[Path, bytes]] = []
    file_receipts = []
    for entry in changes:
        if not isinstance(entry, dict):
            raise RuntimeError("patella source rebase file entry is malformed")
        relative_file = Path(entry.get("file", ""))
        if relative_file.is_absolute() or ".." in relative_file.parts:
            raise RuntimeError("patella source rebase target path is unsafe")
        target = source_package / relative_file
        if not target.is_file() or not target.resolve().is_relative_to(source_package):
            raise RuntimeError(f"patella source rebase target is missing or unsafe: {target}")
        before_sha = entry.get("file_sha256_before")
        after_sha = entry.get("file_sha256_after")
        actual_sha = _sha256(target)
        if actual_sha == before_sha:
            raw = target.read_bytes()
            replacements = entry.get("replacements")
            if not isinstance(replacements, list) or not replacements:
                raise RuntimeError("patella source rebase has no exact replacement witnesses")
            for replacement in replacements:
                if not isinstance(replacement, dict):
                    raise RuntimeError("patella source rebase replacement is malformed")
                old = replacement.get("source_text_before")
                new = replacement.get("source_text_after")
                if not isinstance(old, str) or not isinstance(new, str):
                    raise RuntimeError("patella source rebase replacement text is incomplete")
                old_bytes, new_bytes = old.encode("utf-8"), new.encode("utf-8")
                if raw.count(old_bytes) != 1:
                    raise RuntimeError("patella source rebase witness is absent or ambiguous")
                raw = raw.replace(old_bytes, new_bytes, 1)
            if hashlib.sha256(raw).hexdigest() != after_sha:
                raise RuntimeError("patella source rebase output differs from its pinned hash")
            staged.append((target, raw))
        elif actual_sha != after_sha:
            raise RuntimeError(
                "MyoSim source file differs from both the pinned patella rebase original and corrected overlay"
            )
        file_receipts.append({
            "file": str(relative_file),
            "file_sha256_before": before_sha,
            "file_sha256_after": after_sha,
        })

    for target, raw in staged:
        target.write_bytes(raw)
    return {
        "id": manifest["id"],
        "manifest": str(manifest_path.relative_to(Path(__file__).resolve().parents[2])),
        "manifest_sha256": manifest_sha,
        "source_archive_sha256": source_identity["archive_sha256"],
        "files": file_receipts,
        "equivalence_evidence": manifest.get("equivalence_evidence"),
        "evidence_boundary": manifest.get("evidence_boundary"),
    }


def apply_myo_sim_source_overlays(sources: Path) -> dict[str, Any]:
    """Apply or verify the repository's exact, hash-bound MyoSim overlays.

    The immutable upstream archive is hash-checked before extracted source files
    are changed. Each overlay admits only its exact before/after hashes, making
    repeated builds idempotent and rejecting local drift.
    """
    source_package = sources.resolve() / "myosim"
    overlay, overlay_path, overlay_sha = _read_overlay(source_package)
    source_identity = overlay.get("source")
    change = overlay.get("change")
    if not isinstance(source_identity, dict) or not isinstance(change, dict):
        raise RuntimeError("MyoSim source overlay identity is incomplete")

    archive = source_package / source_identity["archive"]
    if not archive.is_file() or _sha256(archive) != source_identity["archive_sha256"]:
        raise RuntimeError("MyoSim source overlay does not match the pinned upstream archive")

    relative_file = Path(change["file"])
    if relative_file.is_absolute() or ".." in relative_file.parts:
        raise RuntimeError("MyoSim source overlay target path is unsafe")
    target = source_package / relative_file
    if not target.is_file():
        raise RuntimeError(f"MyoSim source overlay target is missing: {target}")

    before_sha = change["file_sha256_before"]
    after_sha = change["file_sha256_after"]
    actual_sha = _sha256(target)
    patella_after_sha = None
    if PATELLA_REBASE_PATH.is_file():
        try:
            patella_manifest = json.loads(
                PATELLA_REBASE_PATH.read_text(encoding="utf-8")
            )
            required_overlay = patella_manifest.get("requires_source_overlay", {})
            if (patella_manifest.get("schema") ==
                    "numi.human.myosim-patella-neutral-coordinate-rebase.v1"
                    and required_overlay.get("id") == overlay.get("id")
                    and required_overlay.get("manifest_sha256") == overlay_sha):
                for entry in patella_manifest["coordinate_reparameterization"]["files"]:
                    if entry.get("file") == str(relative_file):
                        patella_after_sha = entry.get("file_sha256_after")
                        break
        except (KeyError, OSError, json.JSONDecodeError, TypeError):
            patella_after_sha = None
    if actual_sha == before_sha:
        original_text = change.get("source_text_before")
        corrected_text = change.get("source_text_after")
        if not isinstance(original_text, str) or not isinstance(corrected_text, str):
            raise RuntimeError("MyoSim source overlay has no exact replacement text")
        raw = target.read_bytes()
        original = original_text.encode("utf-8")
        corrected = corrected_text.encode("utf-8")
        if raw.count(original) != 1:
            raise RuntimeError("MyoSim source overlay witness is absent or ambiguous")
        target.write_bytes(raw.replace(original, corrected, 1))
        actual_sha = _sha256(target)
    if actual_sha not in {after_sha, patella_after_sha}:
        raise RuntimeError("MyoSim source file differs from both the pinned original and corrected overlay")

    result = {
        "id": overlay["id"],
        "manifest": str(overlay_path.relative_to(source_package)),
        "manifest_sha256": overlay_sha,
        "source_archive_sha256": source_identity["archive_sha256"],
        "modified_source_file": str(relative_file),
        "source_file_sha256_before": before_sha,
        "source_file_sha256_after": after_sha,
        "source_joint": change["source_joint"],
        "driver_joint": change["driver_joint"],
        "driver_range_rad": change["driver_range_rad"],
        "range_before_m": change["range_before_m"],
        "range_after_m": change["range_after_m"],
        "equality_polycoef": change["equality_polycoef"],
        "polynomial_range_witness_m": change["polynomial_range_witness_m"],
    }
    result["additional_source_overlays"] = [
        _apply_patella_neutral_coordinate_rebase(
            source_package, source_identity, overlay, overlay_sha,
        )
    ]
    return result
