#!/usr/bin/env python3
"""Reproduce a verified NHSKIN index-only repair from its quad certificate."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from numilab_human.skin_source_payload_preflight import decode_payload


def directed_boundary(faces):
    edges = Counter((int(a), int(b)) for face in faces
                    for a, b in zip(face, np.roll(face, -1)))
    return edges - Counter({(b, a): count for (a, b), count in edges.items()})


def index_pair(faces, operation, used_rows):
    row_a, row_b = operation["faces"]
    old_a, old_b = operation["old_diagonal"]
    new_a, new_b = operation["new_diagonal"]
    if (not 0 <= row_a < len(faces) or not 0 <= row_b < len(faces)
            or row_a == row_b or row_a in used_rows or row_b in used_rows):
        raise ValueError("invalid or reused source face row")
    if len({old_a, old_b, new_a, new_b}) != 4:
        raise ValueError("diagonal endpoints must describe four distinct vertices")
    original = faces[[row_a, row_b]]
    if (set(map(int, original[0])) != {old_a, old_b, new_a}
            or set(map(int, original[1])) != {old_a, old_b, new_b}):
        raise ValueError("quad certificate does not match its source face rows")
    new = np.asarray([(new_a, old_a, new_b), (new_b, old_b, new_a)],
                     dtype=faces.dtype)
    if directed_boundary(original) != directed_boundary(new):
        raise ValueError("quad repair would change the directed boundary")
    faces[[row_a, row_b]] = new
    used_rows.update((row_a, row_b))


def replay(source, certificate):
    if sha256(source).hexdigest() != certificate["source_sha256"]:
        raise ValueError("NHSKIN source SHA-256 differs from the certificate")
    if certificate["introduced_pairs"] != 0:
        raise ValueError("certificate records newly introduced crossings")
    skin = decode_payload(source)
    index_rows = skin["indices"]
    if len(index_rows) % 3 or int(index_rows.max()) >= skin["vertex_count"]:
        raise ValueError("invalid NHSKIN index buffer")
    faces = index_rows.reshape(-1, 3).copy()
    operations = certificate["edits"]
    if not operations or len(operations) != certificate["candidate_steps"]:
        raise ValueError("quad certificate count disagreement")
    used_rows = set()
    for operation in operations:
        index_pair(faces, operation, used_rows)
    start = len(source) - skin["full_weights"].nbytes - index_rows.nbytes
    end = start + index_rows.nbytes
    candidate = source[:start] + faces.tobytes() + source[end:]
    if sha256(candidate).hexdigest() != certificate["candidate_sha256"]:
        raise ValueError("reproduced NHSKIN SHA-256 differs from the certificate")
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--certificate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    certificate = json.loads(args.certificate.read_text())
    candidate = replay(args.source.read_bytes(), certificate)
    with args.output.open("xb") as out:
        out.write(candidate)
    print(certificate["candidate_sha256"], args.output)


if __name__ == "__main__":
    main()
