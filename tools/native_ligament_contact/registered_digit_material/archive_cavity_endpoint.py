"""Make the audited native step and its registered material coordinates recoverable."""

import json
import sys
from pathlib import Path
from archive_cavity_trial import archive_files, sha


def archive(trial, out):
    state = json.loads((trial / "admission.json").read_text())
    geometry = json.loads((trial / "terminal.receipt.json").read_text())
    if (
        state["terminal_archive_sha256"] != sha(trial / "accepted-terminal.mrvpack.zst")
        or geometry["pack_file_sha256"] != state["terminal_pack_sha256"]
    ):
        raise ValueError("Mismatched accepted pack")
    names = [
        "accepted-terminal.mrvpack.zst",
        "terminal.receipt.json",
        "physical-reference.node",
        "physical-accepted.node",
        "admission.json",
        "launch.json",
        "stdout.log",
        "stderr.log",
        "accepted-layers.json",
        "native-traces/resting-coupled.csv",
        "envelope/target-witnesses.jsonl",
        "envelope/physical-banks.json",
        "envelope/accepted-envelope.json",
        "verified-endpoint.json",
    ]
    files = {n: trial / n for n in names}
    archive_files(
        files,
        "dense-accepted-material-" + state["terminal_pack_sha256"][:16] + ".tar.zst",
        out,
        "Native accepted frame, physical node fields, configuration, material checks, crossing witnesses, and physical traces.",
    )


if __name__ == "__main__":
    archive(*(Path(s).resolve() for s in sys.argv[1:]))
