#!/usr/bin/env python3
"""Prepare one allocation-preserving 2-to-2 flip in a closed NHANATOMY row.

This writes an experimental geometry candidate and a preparation receipt only.
It does not admit the candidate to native runtime use.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from numilab_human.resting_intercostal_mesh_arrangement import (  # noqa: E402
    encode_payload_with_row_replacements,
)
from numilab_human.resting_pleura_proxy import _parse_payload  # noqa: E402
from numilab_human.surface_precision_retriangulation import (  # noqa: E402
    flip_closed_surface_edge_preserving_allocations,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_flip(
    base_path: Path,
    candidate_path: Path,
    stable_id: int,
    face_index: int,
    minimum_altitude_m: float = 1.1e-7,
) -> dict:
    base_path = Path(base_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    if base_path == candidate_path:
        raise ValueError("base and candidate paths must differ")
    raw = base_path.read_bytes()
    rows = _parse_payload(raw)[-1]
    if stable_id not in rows:
        raise ValueError(f"stable ID {stable_id} is absent from the source receipt")
    row = rows[stable_id]
    if face_index < 0 or face_index >= len(row["faces"]):
        raise ValueError(f"face index {face_index} is outside stable ID {stable_id}")
    face = row["faces"][face_index]
    attempts = []
    for edge in itertools.combinations(map(int, face), 2):
        try:
            faces, operation = flip_closed_surface_edge_preserving_allocations(
                row["vertices6"][:, :3].copy(),
                row["faces"],
                identity=(51010, stable_id),
                edge=edge,
                minimum_altitude_m=minimum_altitude_m,
            )
            payload, abi = encode_payload_with_row_replacements(
                raw, {stable_id: {**row, "faces": faces}}
            )
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_bytes(payload)
            return {
                "schema": "numi.human.cardiac-source-edge-flip-preparation.v1",
                "status": "experimental_geometry_candidate_prepared",
                "base": {
                    "path": str(base_path),
                    "sha256": sha256_bytes(raw),
                },
                "candidate": {
                    "path": str(candidate_path),
                    "sha256": sha256_bytes(payload),
                    "abi": abi,
                },
                "operation": {
                    "stable_id": stable_id,
                    "face_index": face_index,
                    "minimum_altitude_m": minimum_altitude_m,
                    "edge": list(edge),
                    "proof": operation,
                    "failed_edge_attempts": attempts,
                },
                "native_admission": False,
                "scope": "Position-preserving source-face flip preparation; no runtime or anatomical admission.",
            }
        except ValueError as exc:
            attempts.append({"edge": list(edge), "error": str(exc)})
    raise ValueError(f"no eligible edge for stable ID {stable_id}, face {face_index}: {attempts}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="source NHANATOMY payload")
    parser.add_argument("--candidate", required=True, type=Path, help="candidate NHANATOMY output")
    parser.add_argument(
        "--receipt", "--output", dest="receipt", required=True, type=Path,
        help="preparation receipt JSON output",
    )
    parser.add_argument("--stable-id", required=True, type=int)
    parser.add_argument("--face-index", required=True, type=int)
    parser.add_argument("--minimum-altitude-m", type=float, default=1.1e-7)
    parser.add_argument("--overwrite", action="store_true", help="allow replacing either output path")
    args = parser.parse_args()

    if args.base.resolve() in (args.candidate.resolve(), args.receipt.resolve()):
        parser.error("--base must differ from both output paths")
    if args.candidate.resolve() == args.receipt.resolve():
        parser.error("--candidate and --receipt must differ")
    if not args.overwrite and (args.candidate.exists() or args.receipt.exists()):
        parser.error("an output already exists; choose new paths or pass --overwrite")
    result = prepare_flip(
        args.base, args.candidate, args.stable_id, args.face_index, args.minimum_altitude_m
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
