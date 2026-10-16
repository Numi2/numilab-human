"""Compile the pinned 323/348 source successor through the existing ABI5 owner."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from numilab_human.resting_pons_cerebellum_source_compiler import compile_pinned_candidate


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-nha", type=Path, required=True)
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--strict-audit", type=Path, required=True)
    parser.add_argument("--output-nha", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    args = parser.parse_args()

    output_nha = args.output_nha.resolve()
    output_evidence = args.output_evidence.resolve()
    if output_nha == output_evidence:
        raise SystemExit("output payload and evidence paths must differ")
    inputs = {Path(value).resolve() for value in (
        args.base_nha, args.candidate_npz, args.preflight, args.lineage, args.strict_audit
    )}
    if output_nha in inputs or output_evidence in inputs:
        raise SystemExit("refusing to overwrite a pinned input")
    if output_nha.exists() or output_evidence.exists():
        raise SystemExit("refusing to overwrite an existing source compile output")

    payload, report = compile_pinned_candidate(
        args.base_nha, args.candidate_npz, args.preflight, args.lineage, args.strict_audit)
    report["compiled_payload_path"] = str(output_nha)
    evidence = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    _atomic_write(output_nha, payload)
    try:
        _atomic_write(output_evidence, evidence)
    except Exception:
        output_nha.unlink(missing_ok=True)
        raise
    print(json.dumps({
        "status": report["status"],
        "compiled_payload_path": str(output_nha),
        "compiled_payload_sha256": report["compiled_payload_sha256"],
        "evidence_path": str(output_evidence),
        "unmodified_parent_rows_byte_exact": report["unmodified_parent_rows_byte_exact"],
        "native_combined_scene_admission": False,
        "anatomical_admission": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
