#!/usr/bin/env python3
"""Create a receipt-bound AVFoundation review of one closed registered 1170 arm."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

R = Path("/Users/n/numi-human-retained-delivery-20261009")
STUDY = R / "native-integrated-resting-study-1170"
EXECUTION = R / "native-final-pair-execution-1170"
REVIEW_ROOT = R / "native-integrated-resting-study-1170-review"
TRIAL_FOR_ARM = {"control": "resting-baseline", "treatment": "resting-drive-half"}
ADAPTER = Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-late-pose-audit-runner-1171/registered_receipt_adapter_1171.py")
ADAPTER_SHA256 = "ba0b88416dbf9cab892f9daa6f954e931003f9db5075db6b758c209370efc1e8"
SWIFT = Path("/Users/n/numi-human-resting-final-source-028/matter/tools/inspect_resting_movie.swift")
SWIFT_SHA256 = "2d6704dd5f06bffd0b8aa0072171af805fdb3dd00dcc11c02e0b692513b21649"
EXPECTED_OUTPUTS = {
    "scene/native-viewer.mov": "native-viewer.mov",
    "scene/resting-surface-audit.csv": "resting-surface-audit.csv",
}
EXPECTED_SNAPSHOTS = ("initial", "middle", "final", "skin", "muscles",
                      "skeleton", "organs", "lungs", "heart", "vessels")
FRAME_RE = re.compile(
    r"^frame=(?P<name>[a-z]+) index=(?P<index>[0-9]+) "
    r"simulated_s=(?P<sim>[0-9.eE+-]+) wall_s=(?P<wall>[0-9.eE+-]+) "
    r"width=(?P<width>[0-9]+) height=(?P<height>[0-9]+)$"
)
SUMMARY_RE = re.compile(
    r"^frames=(?P<frames>[0-9]+) timing_markers=(?P<markers>[0-9]+) "
    r"first_wall_s=(?P<first>[0-9.eE+-]+) last_wall_s=(?P<last>[0-9.eE+-]+) "
    r"max_gap_wall_s=(?P<gap>[0-9.eE+-]+) duration_wall_s=(?P<duration>[0-9.eE+-]+)$"
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path):
    if not path.is_file() or path.is_symlink() or sha(path) != ADAPTER_SHA256:
        raise ValueError("registered receipt adapter missing or changed: " + str(path))
    spec = importlib.util.spec_from_file_location("registered_receipt_adapter_1174", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import registered receipt adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "load_registered_context", None)):
        raise ValueError("registered receipt adapter has no load_registered_context API")
    return module


def require_user_immutable(path: Path):
    if path.is_symlink():
        raise ValueError("closed-run source is a symlink: " + str(path))
    st = path.stat()
    immutable_flag = getattr(stat, "UF_IMMUTABLE", 0x00000002)
    if not hasattr(st, "st_flags") or not (st.st_flags & immutable_flag):
        raise ValueError("closed-run source lacks macOS user-immutable flag: " + str(path))
    return {"path": str(path), "st_flags": int(st.st_flags),
            "required_uf_immutable": int(immutable_flag)}


def validate_retention_marker(trial_path: Path, trial_name: str):
    marker_path = EXECUTION / (trial_name + "-retention.json")
    if not marker_path.is_file() or marker_path.is_symlink():
        raise ValueError("closed-trial retention marker is missing: " + str(marker_path))
    marker = json.loads(marker_path.read_text())
    if (marker.get("trial") != str(trial_path) or
            marker.get("closed_after_science_return") is not True or
            marker.get("reversible_macOS_user_immutable") is not True):
        raise ValueError("retention marker does not prove this trial was closed and protected")
    return marker_path, marker


def surface_time_summary(path: Path):
    count = 0
    first = last = None
    previous = None
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header or not header[0]:
            raise ValueError("surface audit CSV has no time header")
        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                value = float(row[0])
            except (ValueError, IndexError):
                raise ValueError("surface audit has invalid first-column time at row " + str(row_number))
            if not math.isfinite(value):
                raise ValueError("surface audit has non-finite time at row " + str(row_number))
            if previous is not None and value <= previous:
                raise ValueError("surface audit simulation times are not strictly increasing")
            if first is None:
                first = value
            last = previous = value
            count += 1
    if count < 2:
        raise ValueError("surface audit has fewer than two data rows")
    return {"header": header, "data_rows": count, "first_simulated_time_s": first,
            "last_simulated_time_s": last}


def parse_inspector_output(stdout: str, expected_frame_count: int):
    summary = None
    snapshots = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        match = SUMMARY_RE.match(line)
        if match:
            summary = {
                "frame_count": int(match.group("frames")),
                "timing_markers": int(match.group("markers")),
                "first_presentation_time_s": float(match.group("first")),
                "last_presentation_time_s": float(match.group("last")),
                "maximum_presentation_gap_s": float(match.group("gap")),
                "asset_duration_s": float(match.group("duration")),
                "pts_validation_order": "strictly increasing presentation timestamps after sorting",
                "original_compressed_sample_decode_order_verified": False,
            }
            continue
        match = FRAME_RE.match(line)
        if match:
            name = match.group("name")
            if name in snapshots:
                raise ValueError("AVFoundation inspector emitted duplicate snapshot " + name)
            snapshots[name] = {
                "index": int(match.group("index")),
                "simulated_time_s": float(match.group("sim")),
                "presentation_time_s": float(match.group("wall")),
                "width": int(match.group("width")),
                "height": int(match.group("height")),
            }
    if summary is None:
        raise ValueError("AVFoundation inspector did not emit a decode summary")
    if summary["frame_count"] != expected_frame_count:
        raise ValueError("decoded movie frame count differs from surface-audit rows")
    if tuple(sorted(snapshots)) != tuple(sorted(EXPECTED_SNAPSHOTS)):
        raise ValueError("AVFoundation inspector did not emit initial/mid/final and all seven layer snapshots")
    if summary["frame_count"] <= 0 or summary["last_presentation_time_s"] <= summary["first_presentation_time_s"]:
        raise ValueError("AVFoundation presentation timeline is empty or non-increasing")
    return summary, snapshots


def prepare_review(arm: str, output_root: Path = REVIEW_ROOT):
    if arm not in TRIAL_FOR_ARM:
        raise ValueError("arm must be control or treatment")
    trial_name = TRIAL_FOR_ARM[arm]
    trial_path = STUDY / "trials" / trial_name
    if not trial_path.is_dir() or trial_path.is_symlink():
        raise ValueError("registered trial directory is missing or symlinked")
    retention_path, retention = validate_retention_marker(trial_path.resolve(), trial_name)
    adapter = load_module(ADAPTER)
    context = adapter.load_registered_context(trial_path, arm)
    if Path(context["trial_path"]).resolve() != trial_path.resolve():
        raise ValueError("registered adapter returned a different trial path")
    if context["exit_status"].get("returncode") != 0 or context["exit_status"].get("failure") is not None:
        raise ValueError("registered arm did not finish successfully")
    if int(context["requested_roots"]) != 155000 or len(context["steps"]) != 8 or int(context["steps"][-1]) != 155000:
        raise ValueError("registered arm lacks the exact eight-capture N=155000 terminal")
    invocation = context["invocation"]
    environment = invocation.get("environment", {})
    if environment.get("NUMI_HUMAN_RESTING_INSPECTION_TOUR") != "1":
        raise ValueError("registered invocation did not enable the seven-layer inspection tour")
    try:
        period = float(environment["NUMI_HUMAN_RESTING_INSPECTION_PERIOD_SECONDS"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("registered invocation lacks the inspection-tour period")
    if not math.isfinite(period) or period <= 0:
        raise ValueError("inspection-tour period is invalid")

    scene = Path(context["scene_path"]).resolve()
    receipt_files = context["item"]["receipt"].get("files", {})
    sources = {}
    source_flags = {}
    for relative, basename in EXPECTED_OUTPUTS.items():
        if relative not in receipt_files:
            raise ValueError("registered exit receipt omits recording review input: " + relative)
        source = scene / Path(relative).name
        expected = receipt_files[relative]
        if source.is_symlink() or not source.is_file() or sha(source) != expected:
            raise ValueError("registered output is absent or differs from sealed receipt: " + relative)
        source_flags[relative] = require_user_immutable(source)
        if context["input_hashes"].get(str(source)) != expected:
            raise ValueError("1171 registered context did not hash-bind recording review input: " + relative)
        sources[relative] = {"source": source, "sha256": expected, "basename": basename}

    surface = surface_time_summary(sources["scene/resting-surface-audit.csv"]["source"])
    if surface["last_simulated_time_s"] <= 6.5 * period:
        raise ValueError("surface capture timeline ends before the seven-layer tour can complete")
    retention_identity = {
        "path": str(retention_path), "sha256": sha(retention_path),
        "trial": retention.get("trial"),
        "closed_after_science_return": retention["closed_after_science_return"],
        "reversible_macOS_user_immutable": retention["reversible_macOS_user_immutable"],
    }
    return {
        "arm": arm, "trial_name": trial_name, "trial_path": trial_path,
        "context": context, "period_seconds": period, "surface": surface,
        "sources": sources, "source_flags": source_flags, "retention": retention_identity,
    }


def run_review(arm: str, output_root: Path = REVIEW_ROOT):
    prepared = prepare_review(arm, output_root)
    out = output_root / arm
    if out.exists():
        raise FileExistsError("refusing to overwrite existing recording review: " + str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    context = prepared["context"]
    linked = {}
    try:
        for relative, item in prepared["sources"].items():
            destination = out / item["basename"]
            os.link(item["source"], destination)
            if not os.path.samefile(item["source"], destination):
                raise ValueError("review input is not a hard link to the registered source")
            if sha(destination) != item["sha256"]:
                raise ValueError("hard-linked review input differs from registered receipt")
            linked[relative] = {
                "source_path": str(item["source"]), "review_path": str(destination),
                "sha256": item["sha256"], "bytes": destination.stat().st_size,
                "same_device": item["source"].stat().st_dev == destination.stat().st_dev,
                "same_inode": item["source"].stat().st_ino == destination.stat().st_ino,
            }
        if any(not x["same_device"] or not x["same_inode"] for x in linked.values()):
            raise ValueError("recording inputs were not hard-linked on the same device/inode")
        if not SWIFT.is_file() or SWIFT.is_symlink() or sha(SWIFT) != SWIFT_SHA256:
            raise ValueError("pinned inspect_resting_movie.swift missing or changed")
        command = ["/usr/bin/swift", str(SWIFT), str(out),
                   format(prepared["period_seconds"], ".17g"), "7"]
        completed = subprocess.run(command, cwd=out, text=True, capture_output=True)
        log = out / "inspect-resting-movie.log"
        log.write_text(completed.stdout + completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError("AVFoundation movie review failed with status " + str(completed.returncode))
        movie_summary, snapshots = parse_inspector_output(completed.stdout, prepared["surface"]["data_rows"])
        for item in prepared["sources"].values():
            if sha(item["source"]) != item["sha256"]:
                raise ValueError("registered movie or surface CSV changed during read-only review")

        receipt_chain = context["receipt_chain"]
        report = {
            "schema": "numi.human.registered-recording-review.v1",
            "status": "pass",
            "scope": "Closed registered-arm recording integrity and representative image extraction; no geometry or physiological qualification.",
            "arm": prepared["arm"], "trial_name": prepared["trial_name"],
            "trial_path": str(prepared["trial_path"]),
            "registered_receipt_chain": receipt_chain,
            "registered_output_hashes": context["input_hashes"],
            "retention_marker": prepared["retention"],
            "source_file_immutable_flags": prepared["source_flags"],
            "invocation_inspection_tour": {
                "enabled": True, "period_seconds": prepared["period_seconds"],
                "required_snapshot_layers": ["skin", "muscles", "skeleton", "organs", "lungs", "heart", "vessels"],
            },
            "surface_audit_timeline": prepared["surface"],
            "movie": {
                **movie_summary,
                "path": linked["scene/native-viewer.mov"]["review_path"],
                "sha256": linked["scene/native-viewer.mov"]["sha256"],
                "avassetreader_status": "completed (inspector exited successfully after full read)",
                "decode_order_limit": "Presentation timestamps were sorted for validation by the pinned inspector; this report makes no claim about original compressed-sample decode order.",
            },
            "hardlinked_inputs": linked,
            "inspector": {"path": str(SWIFT), "sha256": SWIFT_SHA256, "argv": command,
                          "stdout": completed.stdout, "stderr": completed.stderr,
                          "returncode": completed.returncode},
            "snapshots": snapshots,
            "snapshot_paths": {name: str(out / ("frame-" + name + ".png")) for name in snapshots},
        }
        report_path = out / "recording-review.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        if sha(linked["scene/native-viewer.mov"]["source_path"]) != linked["scene/native-viewer.mov"]["sha256"]:
            raise ValueError("registered movie source hash changed after report emission")
        return report_path, report
    except Exception as exc:
        failure = {
            "schema": "numi.human.registered-recording-review-attempt.v1",
            "status": "failed", "arm": arm, "trial_path": str(prepared["trial_path"]),
            "registered_receipt_chain": prepared["context"]["receipt_chain"],
            "hardlinked_inputs": linked, "error": f"{type(exc).__name__}: {exc}",
        }
        failure_path = out / "recording-review-failure.json"
        failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("control", "treatment"))
    parser.add_argument("--prepare-only", action="store_true",
                        help="validate a closed receipt and source hashes without hard-linking or decoding media")
    parser.add_argument("--review-root", type=Path, default=REVIEW_ROOT)
    args = parser.parse_args()
    root = args.review_root.resolve()
    if REVIEW_ROOT.resolve() not in root.parents and root != REVIEW_ROOT.resolve():
        raise ValueError("review root must remain under the retained-delivery tree")
    if args.prepare_only:
        prepared = prepare_review(args.arm, root)
        context = prepared["context"]
        print(json.dumps({
            "status": "ready_for_review", "arm": args.arm,
            "trial_path": str(prepared["trial_path"]),
            "receipt_chain": context["receipt_chain"],
            "surface_rows": prepared["surface"]["data_rows"],
            "tour_period_seconds": prepared["period_seconds"],
            "movie_sha256": prepared["sources"]["scene/native-viewer.mov"]["sha256"],
            "surface_csv_sha256": prepared["sources"]["scene/resting-surface-audit.csv"]["sha256"],
            "swift_sha256": SWIFT_SHA256,
        }, indent=2, sort_keys=True))
        return 0
    path, report = run_review(args.arm, root)
    print(json.dumps({"status": report["status"], "report": str(path),
                      "sha256": sha(path), "frames": report["movie"]["frame_count"],
                      "snapshots": sorted(report["snapshots"])}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
