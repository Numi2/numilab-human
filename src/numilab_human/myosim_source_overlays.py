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


def apply_myo_sim_source_overlays(sources: Path) -> dict[str, Any]:
    """Apply or verify the repository's sole, exact MyoSim source overlay.

    The immutable upstream archive is hash-checked before the extracted source
    file is changed. Only the original or the exact patched file hash is
    admitted, making repeated builds idempotent and rejecting local drift.
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
    if actual_sha != after_sha:
        raise RuntimeError("MyoSim source file differs from both the pinned original and corrected overlay")

    return {
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
