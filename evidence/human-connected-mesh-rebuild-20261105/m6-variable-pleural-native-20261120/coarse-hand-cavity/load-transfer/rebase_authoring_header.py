"""Reconstruct the pre-overlay source from a pinned native capture and its header."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import re

import numpy as np

from audit import digest


def header_rows(build):
    header = (build / "PosteriorBedReference.inc").read_text()
    quoted_rows = re.search(r"kPosteriorDeepMoves\{\{\n(.*?)\n\}\};", header, re.S)[1]
    entries = np.array([[float(v) for v in row.strip("{},").split(",")]
                        for row in quoted_rows.splitlines()])
    return entries[:, 0].astype(int), entries[:, 1:4], entries[:, 4]


def rebase(capture, installed, comparison_build, output):
    stamp = json.loads((capture / "launch.json").read_text())
    assert stamp["native_binary_sha256"] == digest(installed / "numi-human-native")
    source_file = capture / "shell-mass.csv.nodes.csv"
    with source_file.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    count = len(rows) // 3
    assert len(rows) == 3 * count
    rest_columns = [f"rest_{a}_m" for a in "xyz"]
    cooked = np.array([[float(row[k]) for k in rest_columns] for row in rows])
    row_ids, raw, delta = header_rows(installed)
    indices = row_ids + 2 * count
    expected = raw.copy()
    expected[:, 2] += delta
    error = np.linalg.norm(expected - cooked[indices], axis=1).max()
    assert error < 1.e-7, error
    # Keep the emitted x and y coordinates for all rows. They were not moved
    # by the previous overlays. Restore only the original z values.
    uncooked = cooked.copy()
    uncooked[indices, 2] = raw[:, 2]
    other, other_raw, _ = header_rows(comparison_build)
    check_error = np.linalg.norm(uncooked[other + 2 * count] - other_raw, axis=1).max()
    assert check_error < 1.e-7, check_error
    output.mkdir()
    for index, value in zip(indices, raw[:, 2], strict=True):
        rows[index]["rest_z_m"] = format(value, ".17g")
    ledger = output / source_file.name
    with ledger.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "source_sha256": digest(source_file),
        "restored_ledger_sha256": digest(ledger),
        "applied_header_sha256": digest(installed / "PosteriorBedReference.inc"),
        "comparison_header_sha256": digest(comparison_build / "PosteriorBedReference.inc"),
        "captured_overlay_error_m": float(error),
        "comparison_source_error_m": float(check_error),
        "restored_deep_rows": len(indices),
    }
    (output / "source-identity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for key in ("capture", "installed", "comparison_build", "output"):
        parser.add_argument(key, type=Path)
    rebase(**vars(parser.parse_args()))
