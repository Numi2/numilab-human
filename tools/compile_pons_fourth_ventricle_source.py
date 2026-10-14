"""Compile the pinned Pons/fourth-ventricle source candidate through ABI5."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from numilab_human.resting_pons_floor_compiler import compile_pinned_candidate


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--dual-lineage", type=Path, required=True)
    parser.add_argument("--focused-gate-report", type=Path, required=True)
    parser.add_argument("--output-nha", type=Path, required=True)
    parser.add_argument("--output-evidence", type=Path, required=True)
    args = parser.parse_args()

    output_nha = args.output_nha.resolve()
    output_evidence = args.output_evidence.resolve()
    input_paths = {
        args.base_nha.resolve(),
        args.candidate_npz.resolve(),
        args.candidate_report.resolve(),
        args.dual_lineage.resolve(),
        args.focused_gate_report.resolve(),
    }
    if output_nha in input_paths or output_evidence in input_paths:
        raise SystemExit("refusing to overwrite a pinned input")
    if output_nha.exists() or output_evidence.exists():
        raise SystemExit("refusing to overwrite an existing compile output")

    payload, proof = compile_pinned_candidate(
        args.base_nha,
        args.candidate_npz,
        args.candidate_report,
        args.dual_lineage,
        args.focused_gate_report,
    )
    proof["compiled_payload_sha256"] = _sha256(payload)
    proof["compiled_payload_bytes"] = len(payload)
    proof["compiled_payload_path"] = str(output_nha)
    evidence = json.dumps(proof, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    _atomic_write(output_nha, payload)
    try:
        _atomic_write(output_evidence, evidence)
    except Exception:
        output_nha.unlink(missing_ok=True)
        raise
    print(json.dumps({
        "compiled_payload_path": str(output_nha),
        "compiled_payload_sha256": proof["compiled_payload_sha256"],
        "evidence_path": str(output_evidence),
        "evidence_sha256": _sha256(evidence),
        "replaced_stable_ids": proof["replaced_stable_ids"],
        "unmodified_rows_content_exact": proof["unmodified_rows_content_exact"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
